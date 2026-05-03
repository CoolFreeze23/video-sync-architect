"""
Segmented (variable-offset) synchronization.

Some pairs of files do not share a single global offset because of
structural differences in the source (different intro lengths, eyecatch
durations, ad-break placements, etc.). The visual fine-refinement step
samples the offset at many points across the Target's runtime; when
those samples disagree with each other by more than ~150 ms the file
needs piecewise alignment.

This module turns a list of dense `(target_t, offset)` samples into:
  1. A list of `Segment`s (constant-offset chunks of Target).
  2. A list of `PlaybackChunk`s (segments interleaved with Filler chunks
     that black-pad gaps when Target is shorter than Primary at a
     boundary, or that drop content when Target is longer).
  3. An FFmpeg command that renders the chunks into a single output
     whose timeline matches Primary's.

Sign convention is identical to the rest of the engine:
    offset = primary_match_time - target_ref_time
i.e. negative offset -> trim Target, positive offset -> pad Target.
"""

from __future__ import annotations

import math
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

from ..utils.ffmpeg_utils import FFMPEG
from .media_info import MediaInfo


# --- Black-section detection (for accurate segment boundaries) -----------

_BLACK_RE = re.compile(
    r"black_start:(?P<start>[\d.]+)\s+black_end:(?P<end>[\d.]+)\s+"
    r"black_duration:(?P<dur>[\d.]+)"
)


def detect_primary_black_sections(
    primary_filepath: str,
    min_duration: float = 0.30,
    pixel_threshold: float = 0.10,
    picture_threshold: float = 0.98,
) -> list[tuple[float, float]]:
    """
    Run FFmpeg's `blackdetect` filter on the Primary file to find dark
    sections (eyecatches, fade-to-black between scenes, etc.). Returns
    a sorted list of (start_seconds, end_seconds) tuples for sections
    at least `min_duration` seconds long.

    `pixel_threshold` and `picture_threshold` are blackdetect knobs:
    pix_th sets per-pixel "black" sensitivity (0.10 = up to 10% of max
    luma counts as black), pic_th sets the fraction of black pixels a
    frame must have to be flagged. Default values are FFmpeg's
    recommended starting points for "real-world" black detection.
    """
    cmd = [
        FFMPEG.path, "-hide_banner", "-nostdin",
        "-i", primary_filepath,
        "-vf", (f"blackdetect=d={min_duration:.2f}:"
                f"pix_th={pixel_threshold:.2f}:"
                f"pic_th={picture_threshold:.2f}"),
        "-an", "-sn", "-f", "null", "-",
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run(
        cmd, capture_output=True, check=False, creationflags=creation_flags,
    )

    sections: list[tuple[float, float]] = []
    stderr = proc.stderr.decode(errors="replace") if proc.stderr else ""
    for line in stderr.splitlines():
        m = _BLACK_RE.search(line)
        if m:
            try:
                start = float(m.group("start"))
                end = float(m.group("end"))
                if end - start >= min_duration:
                    sections.append((start, end))
            except ValueError:
                continue

    sections.sort(key=lambda s: s[0])
    return sections


# Tolerance for "same offset" inside a segment. Two pHash-matched
# frames at 23.976 fps can land 1 frame apart (~42 ms) due to the
# search window snapping to integer frames either side of the truth, so
# 200 ms is the smallest tolerance that doesn't fight the measurement.
JUMP_THRESHOLD_S = 0.200

# Use segmented mode only if dense samples disagree by more than this.
VARIANCE_TRIGGER_S = 0.180

# A new segment is only opened if at least this many consecutive samples
# agree on the new offset (within JUMP_THRESHOLD_S of each other) AND
# disagree with the current segment by more than NEW_SEGMENT_DELTA_S.
# This kills false-match outliers that would otherwise spawn fake
# segments.
MIN_SEGMENT_SAMPLES = 2
NEW_SEGMENT_DELTA_S = 0.300

# Drop samples whose Hamming distance is worse than this before
# segmenting. HD>8 is a noisy or false match in our pHash budget and
# its offset cannot be trusted.
MAX_USABLE_HD = 8

# A trailing run of samples may form a final segment with fewer than
# MIN_SEGMENT_SAMPLES if the last sample's HD is excellent (HD<=4).
TRAILING_SOLO_MAX_HD = 4


# --- Data classes ---------------------------------------------------------

@dataclass
class Segment:
    """A constant-offset chunk of Target."""
    target_start: float          # source-time start (seconds)
    target_end: float            # source-time end (exclusive)
    offset: float                # primary_t - target_t inside this segment
    # Diagnostic metadata used by the worker to decide whether per-
    # segment audio refinement should override the pHash offset. Set by
    # the segmenter; downstream consumers can ignore them.
    phash_inliers: int = 0       # number of pHash samples that agreed
    phash_stdev: float = 0.0     # stdev of inlier offsets (seconds)


@dataclass
class TargetChunk:
    """A piece of Target to play directly."""
    target_start: float
    target_end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.target_end - self.target_start)


@dataclass
class FillerChunk:
    """Black/silent padding inserted between segments to keep the output
    timeline aligned with Primary (used when Primary is longer at a
    boundary than Target's content covers)."""
    duration: float


PlaybackChunk = Union[TargetChunk, FillerChunk]


# --- Segment building -----------------------------------------------------

def build_segments_from_samples(
    samples: list[tuple[float, float, int]],   # (target_t, offset, hd)
    target_duration: float,
    jump_threshold: float = JUMP_THRESHOLD_S,
    new_segment_delta: float = NEW_SEGMENT_DELTA_S,
    max_usable_hd: int = MAX_USABLE_HD,
    min_segment_samples: int = MIN_SEGMENT_SAMPLES,
    trailing_solo_max_hd: int = TRAILING_SOLO_MAX_HD,
) -> list[Segment]:
    """
    Robust segmenter that treats per-sample offset noise as such instead
    of opening a fake segment per outlier.

    Pipeline:
      1. Drop samples whose Hamming distance exceeds `max_usable_hd`.
         Those are unreliable false matches.
      2. Walk the remaining samples left to right. Each sample is
         either:
           * Inside the current segment    (close to its mean), OR
           * a candidate for a new segment (far from current mean,
             held aside until a 2nd consecutive sample CONFIRMS the
             new offset by agreeing with it within `jump_threshold`),
             OR
           * an isolated outlier           (discarded).
      3. A trailing single sample may seed its own segment only if its
         HD is excellent (<= `trailing_solo_max_hd`).
      4. Each segment's offset is the MEAN of its samples - this lets
         a 50/50 mix of frame-N and frame-N+1 measurements average to
         the correct sub-frame offset (the real source of "1 frame off
         at the start").

    Boundaries in target time are placed at the midpoint between the
    last sample of segment i and the first sample of segment i+1.
    """
    if not samples:
        return []

    # 1. Sort + HD filter.
    samples = sorted(
        [s for s in samples if s[2] <= max_usable_hd],
        key=lambda s: s[0],
    )
    if not samples:
        return []

    # 2. Robust segmentation.
    groups: list[list[tuple[float, float, int]]] = [[samples[0]]]
    pending: Optional[tuple[float, float, int]] = None

    def group_mean(g: list[tuple[float, float, int]]) -> float:
        return sum(x[1] for x in g) / len(g)

    for s in samples[1:]:
        cur_mean = group_mean(groups[-1])
        if abs(s[1] - cur_mean) <= jump_threshold:
            # Sample fits the current segment; clear any pending and add.
            pending = None
            groups[-1].append(s)
        elif abs(s[1] - cur_mean) > new_segment_delta:
            # Sample is far from current segment.
            if pending is None:
                # Hold it as a candidate; need a 2nd sample to confirm.
                pending = s
            else:
                # Already had a pending candidate. Does this new sample
                # agree with it? If so, open a new segment with both.
                if abs(s[1] - pending[1]) <= jump_threshold:
                    groups.append([pending, s])
                    pending = None
                else:
                    # The two outliers disagree with each other AND the
                    # current segment. Drop the older one, keep this as
                    # the new candidate.
                    pending = s
        else:
            # In the "borderline" zone (between jump_threshold and
            # new_segment_delta) - too far to merge but not far enough
            # to call a real new segment. Add to current group; the
            # mean will pull a tiny bit but a real boundary still needs
            # the > new_segment_delta criterion below.
            pending = None
            groups[-1].append(s)

    # 3. Trailing solo: if a pending candidate was left dangling, allow
    #    it to seed a final segment only if it's a very high-confidence
    #    measurement (HD <= trailing_solo_max_hd).
    if pending is not None and pending[2] <= trailing_solo_max_hd:
        groups.append([pending])

    # 4. Drop any group that ended up with < min_segment_samples and is
    #    NOT the single trailing solo we just allowed.
    if len(groups) > 1:
        first_two_check_idx = 0
        cleaned: list[list[tuple[float, float, int]]] = []
        for gi, g in enumerate(groups):
            if (len(g) < min_segment_samples
                    and gi > first_two_check_idx
                    and gi < len(groups) - 1):
                # Mid-list undersized group: absorb into previous.
                if cleaned:
                    cleaned[-1].extend(g)
                continue
            cleaned.append(g)
        groups = cleaned

    # 5. Build Segment objects, using MEAN within group for sub-frame
    #    precision and midpoint boundaries between groups in target time.
    segments: list[Segment] = []
    n = len(groups)
    for i, group in enumerate(groups):
        avg_offset = group_mean(group)

        if i == 0:
            tA = 0.0
        else:
            prev_last = groups[i - 1][-1][0]
            this_first = group[0][0]
            tA = (prev_last + this_first) / 2.0

        if i == n - 1:
            tB = target_duration
        else:
            this_last = group[-1][0]
            next_first = groups[i + 1][0][0]
            tB = (this_last + next_first) / 2.0

        segments.append(Segment(target_start=tA, target_end=tB, offset=avg_offset))

    return segments


def offset_variance(samples: list[tuple[float, float, int]]) -> float:
    """Max-min spread of refined offsets across all samples (seconds).
    Used by the worker to decide whether to invoke segmented sync."""
    if len(samples) < 2:
        return 0.0
    offsets = [s[1] for s in samples]
    return max(offsets) - min(offsets)


def _percentile_linear(values: list[float], p: float) -> float:
    """Linear-interpolated percentile in [0, 100], inclusive."""
    if not values:
        return 0.0
    xs = sorted(values)
    n = len(xs)
    if n == 1:
        return xs[0]
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (n - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def offset_variance_robust(
    samples: list[tuple[float, float, int]],
    p_low: float = 12.0,
    p_high: float = 88.0,
) -> float:
    """
    Spread of offsets between two percentiles (default 12th–88th).
    Ignores a thin tail of extreme pHash false matches so a few bad
    samples cannot alone force segmented mode or distort clusters.
    """
    if len(samples) < 3:
        return offset_variance(samples)
    offs = [s[1] for s in samples]
    lo = _percentile_linear(offs, p_low)
    hi = _percentile_linear(offs, p_high)
    return max(0.0, hi - lo)


def despike_isolated_dense_samples(
    samples: list[tuple[float, float, int]],
    spike_threshold: float = 0.65,
    neighbor_agree: float = 0.42,
    max_passes: int = 6,
) -> tuple[list[tuple[float, float, int]], int]:
    """
    Remove *isolated* pHash glitches: a sample whose offset disagrees
    with both neighbours by > spike_threshold, while those neighbours
    agree with each other within neighbor_agree, is replaced by their
    mean offset (HD unchanged - only used for clustering).

    Runs multiple passes so back-to-back singleton spikes can be
    cleaned. Returns (new_list, total_replacements).
    """
    if len(samples) < 3:
        return samples, 0

    total_fixes = 0
    cur = list(samples)
    for _ in range(max_passes):
        s = sorted(cur, key=lambda x: x[0])
        n = len(s)
        changed = False
        out = list(s)
        for i in range(1, n - 1):
            t0, o0, h0 = s[i - 1]
            t1, o1, h1 = s[i]
            t2, o2, h2 = s[i + 1]
            if abs(o0 - o2) <= neighbor_agree:
                if abs(o1 - o0) > spike_threshold and abs(o1 - o2) > spike_threshold:
                    o_new = 0.5 * (o0 + o2)
                    out[i] = (t1, o_new, h1)
                    changed = True
                    total_fixes += 1
        if not changed:
            break
        cur = sorted(out, key=lambda x: x[0])
    return cur, total_fixes


# Cluster-based segmentation tuning knobs ----------------------------------
# How close two offsets must be to count as the *same* cluster (seconds).
# pHash measurement noise is ~1 frame at 24 fps (~42 ms); 150 ms gives some
# headroom for legitimate sub-cluster jitter without merging real cuts.
_CLUSTER_TOLERANCE_S = 0.150
# A cluster needs this many agreeing samples to be considered "real" rather
# than a coincidental noise burst. Set to 1 for short videos with sparse
# samples, 2 for typical TV-length episodes (we use 1 here and rely on the
# "two consecutive agreeing samples" rule in the walker for confirmation).
_MIN_CLUSTER_SIZE = 1


def _cluster_samples_along_time(
    pts: list[tuple[float, float, float]],
    tolerance: float = _CLUSTER_TOLERANCE_S,
) -> list[list[tuple[float, float, float]]]:
    """
    Walk samples in primary-time order and partition them into groups whose
    offsets cluster within `tolerance`. A sample whose offset is far from
    the running median of the current group becomes "pending"; the next
    sample either confirms a new cluster (by agreeing with pending) or
    rejoins the current group (and pending is discarded as a pHash
    outlier).

    Returns a list of groups, each containing at least one sample.
    """
    if not pts:
        return []
    groups: list[list[tuple[float, float, float]]] = [[pts[0]]]
    pending: Optional[tuple[float, float, float]] = None

    for i in range(1, len(pts)):
        p = pts[i]
        cur = groups[-1]
        cur_off = sorted(q[2] for q in cur)[len(cur) // 2]  # median
        if abs(p[2] - cur_off) <= tolerance:
            cur.append(p)
            pending = None
            continue
        # Far from current group - candidate for new cluster or outlier.
        if pending is None:
            pending = p
            continue
        # Two consecutive far samples - if they agree with each other,
        # we have a confirmed transition; otherwise the previous one was
        # a noise outlier and the new one becomes the new pending.
        if abs(p[2] - pending[2]) <= tolerance:
            groups.append([pending, p])
            pending = None
        else:
            pending = p
    return groups


_MERGE_BLOCKING_BLACK_MIN_DUR = 0.50


def _gap_contains_blocking_black(
    primary_lo: float,
    primary_hi: float,
    primary_blacks: list[tuple[float, float]],
    min_blocking_dur: float = _MERGE_BLOCKING_BLACK_MIN_DUR,
) -> bool:
    """
    Return True if any primary black section of duration >=
    min_blocking_dur sits inside the open primary-time interval
    (primary_lo, primary_hi). Such a black is strong evidence of a
    real edit/cut between two clusters, so a singleton on one side of
    it should not be silently absorbed into a cluster on the other.

    Trivial fades (~< 0.5 s) are NOT considered blocking - they're
    common intra-scene transitions and we don't want them to spawn
    spurious singleton segments.
    """
    if primary_hi <= primary_lo:
        return False
    for b_start, b_end in primary_blacks:
        if (b_end - b_start) < min_blocking_dur:
            continue
        mid = (b_start + b_end) / 2.0
        if primary_lo < mid < primary_hi:
            return True
    return False


def _merge_singleton_clusters(
    groups: list[list[tuple[float, float, float]]],
    max_merge_distance: float = 0.300,
    primary_blacks: Optional[list[tuple[float, float]]] = None,
) -> list[list[tuple[float, float, float]]]:
    """
    Iteratively absorb singleton clusters (groups containing exactly one
    sample) into the closer-offset adjacent multi-sample cluster, but
    only if BOTH:
      1. the offset distance is <= max_merge_distance, AND
      2. there is no primary black section of duration >= 0.5 s in the
         primary-time gap between the singleton and the target cluster
         (the black section is itself evidence of a real cut and the
         singleton may be measuring a legitimately different offset on
         the other side of it).

    Why (2) matters: a single pHash sample inside a short pre-cut
    region (e.g. before a 1.5 s eyecatch) used to be silently absorbed
    into the much-larger post-cut cluster, hiding a real micro-drift
    of 0.1-0.3 s. Honoring blocking black sections preserves these
    real boundaries, while merging is still allowed across stretches
    of continuous content (where a singleton is much more likely to
    be a pHash false-match than a real new offset).

    Singletons whose offset is too far from every neighbour to merge
    are left alone - those represent genuine outliers (or genuine but
    undersupported segments), and we'd rather keep them as visible
    fragments than silently merge them into a wrong cluster.
    """
    if len(groups) <= 1:
        return groups
    blacks = primary_blacks or []
    groups = [list(g) for g in groups]  # mutable copies
    while True:
        best: Optional[tuple[int, int, float]] = None  # (i, target_j, dist)
        for i, g in enumerate(groups):
            if len(g) > 1:
                continue
            singleton_p = g[0][0]      # primary time of the singleton
            singleton_off = g[0][2]
            for j in (i - 1, i + 1):
                if not (0 <= j < len(groups) and len(groups[j]) >= 2):
                    continue
                nbr = groups[j]
                nbr_med = sorted(p[2] for p in nbr)[len(nbr) // 2]
                dist = abs(singleton_off - nbr_med)
                if dist > max_merge_distance:
                    continue
                # Reject merge if a meaningful primary black sits between
                # the singleton and the neighbour cluster.
                if blacks:
                    nbr_p_min = min(p[0] for p in nbr)
                    nbr_p_max = max(p[0] for p in nbr)
                    if j < i:  # singleton is AFTER neighbour
                        gap_lo, gap_hi = nbr_p_max, singleton_p
                    else:      # singleton is BEFORE neighbour
                        gap_lo, gap_hi = singleton_p, nbr_p_min
                    if _gap_contains_blocking_black(gap_lo, gap_hi, blacks):
                        continue
                if best is None or dist < best[2]:
                    best = (i, j, dist)
        if best is None:
            break
        i, j, _ = best
        groups[j].extend(groups[i])
        groups.pop(i)
    return groups


def _select_boundary_in_gap(
    gap_start_p: float,
    gap_end_p: float,
    primary_black_sections: list[tuple[float, float]],
) -> float:
    """
    Pick the most plausible boundary location inside a primary-time gap.
    Prefer black sections that look like *eyecatches* over intra-scene
    fades.

    Scoring: real eyecatches are ISOLATED 2-3 second blacks surrounded
    by content. Intra-scene fades come in CLUSTERS (e.g. flashback
    sequences with multiple short fades 1-2 seconds apart). Score each
    candidate as `duration * min_distance_to_neighbour_black`. The
    isolation factor crushes scores of black sections in dense clusters
    while letting standalone eyecatches win.

    If there are no black sections in the gap, fall back to the gap's
    midpoint.
    """
    in_gap_idx = [
        i for i, (b_start, b_end) in enumerate(primary_black_sections)
        if gap_start_p < (b_start + b_end) / 2.0 < gap_end_p
    ]
    if not in_gap_idx:
        return (gap_start_p + gap_end_p) / 2.0

    def score(idx: int) -> float:
        b_start, b_end = primary_black_sections[idx]
        duration = b_end - b_start
        prev_dist = float("inf")
        next_dist = float("inf")
        if idx - 1 >= 0:
            prev_dist = b_start - primary_black_sections[idx - 1][1]
        if idx + 1 < len(primary_black_sections):
            next_dist = primary_black_sections[idx + 1][0] - b_end
        # Floor isolation distance at 1s so we don't multiply by zero
        # for adjacent fades; the duration term still differentiates
        # them. Cap at 600s so a single very-isolated black doesn't
        # dominate purely by being lonely.
        isolation = max(1.0, min(600.0, min(prev_dist, next_dist)))
        return duration * isolation

    best_idx = max(in_gap_idx, key=score)
    b_start, b_end = primary_black_sections[best_idx]
    return (b_start + b_end) / 2.0


def build_segments_from_samples_and_blacks(
    samples: list[tuple[float, float, int]],
    target_duration: float,
    primary_black_sections: list[tuple[float, float]],
    max_usable_hd: int = MAX_USABLE_HD,
    cluster_tolerance: float = _CLUSTER_TOLERANCE_S,
) -> list[Segment]:
    """
    Cluster-first segmentation: identify groups of samples that agree on
    their offset, then place a segment boundary in each *gap* between
    consecutive clusters. Boundary location is the LONGEST primary
    black section inside that gap (real eyecatch); if none exists, the
    gap midpoint.

    Why this beats greedy candidate-by-candidate validation:
      1. Outlier samples can never *create* a fake boundary. They land
         in a singleton "pending" slot that gets discarded by the
         walker rule (a transition needs two agreeing samples).
      2. Consuming a sample for the wrong boundary is impossible -
         we don't visit black sections in order, we visit *cluster
         transitions* in order, and there is exactly one black section
         placed per real transition.
      3. Empty gaps (samples that don't bracket the cut) still produce
         a boundary - it's just placed at the gap midpoint instead of
         on a confirmed black section.

    Falls back to the sample-only segmenter if there are no usable
    samples or no clusters / black sections to work with.
    """
    if not samples:
        return []

    # 1. HD-filter samples and sort by target time.
    filtered = sorted(
        [s for s in samples if s[2] <= max_usable_hd],
        key=lambda s: s[0],
    )
    if len(filtered) < 2:
        return build_segments_from_samples(samples, target_duration)

    # 2. Convert to (primary_t, target_t, offset).
    pts: list[tuple[float, float, float]] = [
        (t + off, t, off) for (t, off, _hd) in filtered
    ]

    # 3. Cluster along time, then merge any singleton clusters whose
    #    offset is within ~2x cluster tolerance of an adjacent multi-
    #    sample cluster (single-sample clusters are unreliable - one
    #    pHash false-match shouldn't anchor a whole segment).
    groups = _cluster_samples_along_time(pts, cluster_tolerance)
    groups = _merge_singleton_clusters(
        groups,
        max_merge_distance=cluster_tolerance * 2.0,
        primary_blacks=primary_black_sections,
    )
    if len(groups) <= 1:
        # Only one cluster -> single offset, no boundaries needed.
        offsets = [p[2] for p in pts]
        med = statistics.median(offsets)
        inliers = [o for o in offsets if abs(o - med) <= cluster_tolerance]
        avg = sum(inliers) / max(1, len(inliers))
        return [Segment(target_start=0.0,
                        target_end=target_duration,
                        offset=avg)]

    # 4. Place boundaries in each gap between adjacent clusters.
    boundaries_p: list[float] = []
    for i in range(len(groups) - 1):
        last_p = max(q[0] for q in groups[i])
        first_p = min(q[0] for q in groups[i + 1])
        if first_p <= last_p:
            # Pathological - just use the midpoint.
            boundaries_p.append((last_p + first_p) / 2.0)
            continue
        boundaries_p.append(_select_boundary_in_gap(
            last_p, first_p, primary_black_sections,
        ))

    # 5. Build segments. Per-cluster offset is the mean of inliers
    #    (median +/- 150 ms) so a single bad sample inside a cluster
    #    can't drag the offset off. Track inlier count + stdev so the
    #    worker can decide whether to trust pHash or fall back to audio
    #    cross-correlation per segment.
    segments: list[Segment] = []
    boundaries_p_full = [-1.0] + boundaries_p + [float("inf")]
    for i, group in enumerate(groups):
        offsets = [q[2] for q in group]
        med = statistics.median(offsets)
        inliers = [o for o in offsets if abs(o - med) <= cluster_tolerance]
        if not inliers:
            inliers = offsets
        seg_offset = sum(inliers) / len(inliers)
        seg_inliers = len(inliers)
        seg_stdev = (
            statistics.pstdev(inliers) if len(inliers) >= 2 else 0.0
        )

        lo_p = boundaries_p_full[i]
        hi_p = boundaries_p_full[i + 1]
        if i == 0:
            tA = 0.0
        else:
            tA = max(0.0, lo_p - seg_offset)
        if i == len(groups) - 1:
            tB = target_duration
        else:
            tB = max(tA + 0.001, hi_p - seg_offset)
        tB = min(tB, target_duration)
        if tB > tA:
            segments.append(Segment(
                target_start=tA, target_end=tB, offset=seg_offset,
                phash_inliers=seg_inliers, phash_stdev=seg_stdev,
            ))

    if not segments:
        return build_segments_from_samples(samples, target_duration)
    return segments


# --- Playback chunk planning ---------------------------------------------

def build_playback_chunks(
    segments: list[Segment],
    target_duration: float,
    primary_duration: float,
) -> list[PlaybackChunk]:
    """
    Convert a list of Segments (each with its own offset) into a flat
    list of TargetChunks and FillerChunks whose concatenation matches
    Primary's timeline starting at primary_t = 0 and ending at
    primary_t = primary_duration.

    Boundary handling:
      * gap > 0  (offset increased between segments, Target is shorter
                  than Primary at this point) -> insert FillerChunk.
      * gap < 0  (offset decreased, Target is longer than Primary at
                  this point) -> trim TargetChunk's start by |gap|.
      * gap = 0  -> direct concatenation.
    """
    chunks: list[PlaybackChunk] = []
    if not segments:
        return chunks

    # --- First segment: align primary t = 0 to target t = -offset ---
    s0 = segments[0]
    primary_start_of_s0 = s0.target_start + s0.offset

    if primary_start_of_s0 > 0:
        chunks.append(FillerChunk(duration=primary_start_of_s0))
        chunks.append(TargetChunk(target_start=s0.target_start,
                                  target_end=s0.target_end))
    else:
        # Trim the front of segment 0 so synced output starts at primary 0.
        new_start = s0.target_start + (-primary_start_of_s0)
        if new_start < s0.target_end:
            chunks.append(TargetChunk(target_start=new_start,
                                      target_end=s0.target_end))

    # Smallest filler we'll emit. Anything shorter than this is below 1
    # video frame at any common framerate (24/25/30/60 fps), so the
    # resulting sub-frame timing error is invisible. Tiny fillers (a
    # few ms) make FFmpeg's `lavfi color` / `anullsrc` producers emit
    # zero frames, which then hangs the `concat` filter forever.
    MIN_FILLER_S = 0.030

    # --- Subsequent segments: handle gap / overlap at each boundary ---
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]
        prev_synced_end = prev.target_end + prev.offset
        curr_synced_start = curr.target_start + curr.offset
        gap = curr_synced_start - prev_synced_end  # in primary seconds

        if gap >= MIN_FILLER_S:
            chunks.append(FillerChunk(duration=gap))
            chunks.append(TargetChunk(target_start=curr.target_start,
                                      target_end=curr.target_end))
        elif gap > 1e-6:
            # Sub-frame positive gap - swallow it (concat directly).
            # The resulting <30 ms drift is below 1 frame and below
            # human perception. Emitting a tiny filler would hang
            # FFmpeg's lavfi sources.
            chunks.append(TargetChunk(target_start=curr.target_start,
                                      target_end=curr.target_end))
        elif gap < -1e-6:
            # Target overlaps the previous segment in synced time; advance
            # the start of the new segment by |gap| of target seconds.
            new_start = curr.target_start + (-gap)
            if new_start < curr.target_end:
                chunks.append(TargetChunk(target_start=new_start,
                                          target_end=curr.target_end))
        else:
            chunks.append(TargetChunk(target_start=curr.target_start,
                                      target_end=curr.target_end))

    # --- Tail handling ---
    # If the synced output naturally exceeds Primary's runtime, trim the
    # trailing chunks so the output ends precisely at primary_duration.
    # If it falls short, leave the natural ending alone EXCEPT for a
    # small mid-gap pad (<= 5 s) which we treat as a real gap to fill;
    # don't append a huge cosmetic black tail (which is what happens when
    # Target is shorter than Primary by a full credits sequence - the
    # user has no Target content to show there anyway).
    total_duration = sum(c.duration for c in chunks)
    if total_duration > primary_duration + 1e-3:
        excess = total_duration - primary_duration
        while excess > 1e-3 and chunks:
            last = chunks[-1]
            if last.duration <= excess + 1e-3:
                excess -= last.duration
                chunks.pop()
            else:
                if isinstance(last, TargetChunk):
                    last.target_end -= excess
                elif isinstance(last, FillerChunk):
                    last.duration -= excess
                excess = 0.0

    return chunks


# --- FFmpeg rendering -----------------------------------------------------

def render_segmented_export(
    target_info: MediaInfo,
    primary_info: MediaInfo,
    chunks: list[PlaybackChunk],
    output_path: str,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> str:
    """
    Build and execute an FFmpeg command that emits `chunks` as a single
    concatenated mp4 whose duration matches Primary's. Each TargetChunk
    becomes one `-ss/-t/-i` input on the Target file; each FillerChunk
    becomes a `lavfi` color + `anullsrc` input pair. All streams are
    stitched together via the `concat` filter with consistent output
    geometry (Primary's resolution and Target's audio rate).
    """

    def log(m: str) -> None:
        if log_callback:
            log_callback(m)

    if not chunks:
        raise ValueError("Cannot render with empty chunk list.")

    # Output geometry: Primary's W/H, Primary's rational fps.
    out_w = primary_info.width or target_info.width or 1920
    out_h = primary_info.height or target_info.height or 1080
    fps_rational = primary_info.fps_rational or "24000/1001"

    # Editor-friendly fixed audio profile - match what we ask the muxer
    # to deliver below (`-c:a aac -ar 48000 -ac 2`).
    audio_rate = 48000
    audio_layout = "stereo"

    args: list[str] = [
        FFMPEG.path, "-y", "-hide_banner", "-nostdin",
        "-loglevel", "error",
        # Stream machine-readable progress on stderr; we'll parse it below
        # so the UI can show a live progress bar / log during the long
        # transcode step instead of staring at 0% until completion.
        "-progress", "pipe:2", "-stats_period", "0.5",
    ]
    input_idx = 0
    concat_pairs: list[str] = []  # ordered "[i:v][j:a]" strings

    log(f"Segmented render: {len(chunks)} chunk(s) -> {output_path}")

    for ci, chunk in enumerate(chunks):
        if isinstance(chunk, TargetChunk):
            duration = max(0.0, chunk.target_end - chunk.target_start)
            if duration < 1e-3:
                continue
            args.extend([
                "-ss", f"{chunk.target_start:.6f}",
                "-t", f"{duration:.6f}",
                "-i", target_info.filepath,
            ])
            v_label = f"[v{ci}]"
            a_label = f"[a{ci}]"
            # Each segment is normalized to the same geometry/fps/sar.
            seg_filter = (
                f"[{input_idx}:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps={fps_rational}"
                f"{v_label};"
                f"[{input_idx}:a]aresample={audio_rate},aformat=channel_layouts={audio_layout}{a_label}"
            )
            input_idx += 1
            log(
                f"  Chunk {ci + 1}/{len(chunks)}: TARGET "
                f"[{chunk.target_start:.3f}, {chunk.target_end:.3f}] "
                f"({duration:.3f}s)"
            )
            concat_pairs.append((seg_filter, v_label, a_label))
        elif isinstance(chunk, FillerChunk):
            # Defense in depth: drop fillers shorter than 1 frame at any
            # common framerate. Sub-frame fillers cause `lavfi color` /
            # `anullsrc` to emit zero frames, which deadlocks the
            # `concat` filter (encoder hangs at the boundary forever).
            if chunk.duration < 0.030:
                log(
                    f"  Chunk {ci + 1}/{len(chunks)}: FILLER "
                    f"({chunk.duration * 1000:.1f}ms) - sub-frame, skipping"
                )
                continue
            args.extend([
                "-f", "lavfi", "-t", f"{chunk.duration:.6f}",
                "-i", f"color=c=black:s={out_w}x{out_h}:r={fps_rational}",
            ])
            v_in = input_idx
            input_idx += 1
            args.extend([
                "-f", "lavfi", "-t", f"{chunk.duration:.6f}",
                "-i", f"anullsrc=channel_layout={audio_layout}:sample_rate={audio_rate}",
            ])
            a_in = input_idx
            input_idx += 1
            v_label = f"[v{ci}]"
            a_label = f"[a{ci}]"
            seg_filter = (
                f"[{v_in}:v]setsar=1,fps={fps_rational}{v_label};"
                f"[{a_in}:a]aformat=channel_layouts={audio_layout}{a_label}"
            )
            log(
                f"  Chunk {ci + 1}/{len(chunks)}: FILLER ({chunk.duration:.3f}s "
                f"black + silence)"
            )
            concat_pairs.append((seg_filter, v_label, a_label))

    if not concat_pairs:
        raise ValueError("Segmented render produced zero usable chunks.")

    # Assemble the filter graph: per-chunk normalization -> concat ->
    # explicit PTS reset on both final streams. The PTS reset is what
    # actually fixes the Premiere "MPEG Source Settings (AEVideoFilter:29)"
    # error: without it the concat filter emits streams whose first PTS
    # is a few ms above zero (~62-83 ms), which Premiere's importer treats
    # as a malformed MPEG header. setpts=PTS-STARTPTS / asetpts=PTS-STARTPTS
    # forces both output streams to begin at exactly PTS 0.
    seg_filters = ";".join(p[0] for p in concat_pairs)
    concat_inputs = "".join(p[1] + p[2] for p in concat_pairs)
    filter_complex = (
        f"{seg_filters};"
        f"{concat_inputs}concat=n={len(concat_pairs)}:v=1:a=1[v_concat][a_concat];"
        f"[v_concat]setpts=PTS-STARTPTS[v_out];"
        f"[a_concat]asetpts=PTS-STARTPTS[a_out]"
    )

    args.extend([
        "-filter_complex", filter_complex,
        "-map", "[v_out]",
        "-map", "[a_out]",
        # ----- Video: H.264 with editor-friendly profile -----
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        # 8-bit 4:2:0 is the only pixel format Premiere/Resolve/QuickTime
        # decode reliably. Without this, libx264 may emit yuv444p or
        # yuv420p10le and editors error out with "MPEG Source Settings".
        "-pix_fmt", "yuv420p",
        # High@4.0 is broadly accepted; 4.1+ trips some Premiere builds.
        "-profile:v", "high", "-level", "4.0",
        # Closed GOP at exactly 2 seconds, no scene-cut keyframes. This
        # gives Premiere a perfectly regular index it can seek into.
        "-x264-params", "keyint=48:min-keyint=48:no-scenecut=1:open-gop=0",
        # Modest B-frames / refs. Some Premiere builds have quirks with
        # x264's default B-pyramid + 16 refs; this keeps it conservative.
        "-bf", "2", "-refs", "3",
        # Explicit BT.709 metadata so Premiere doesn't guess the color
        # space and pick the wrong importer.
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", "-color_range", "tv",
        # Constant frame rate at the muxer (the fps= filter forces it
        # per segment too; this enforces it at output).
        "-fps_mode", "cfr",
        # ----- Audio: 48 kHz stereo AAC LC -----
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-profile:a", "aac_low",
        # ----- Container hygiene -----
        # +faststart moves the moov atom to the front so editors find
        # the index without scanning to EOF.
        "-movflags", "+faststart",
        # Snap any sub-frame negative PTS the concat filter may emit at
        # boundaries to zero, so the muxer doesn't write a non-monotonic
        # MP4 (the most common cause of "MPEG Source Settings" failures
        # on otherwise-valid H.264 streams).
        "-avoid_negative_ts", "make_zero",
        # Force MP4 muxer (rather than auto-detecting from the .mp4
        # extension) so we get the exact muxer flags above.
        "-f", "mp4",
        output_path,
    ])

    # Total output duration = sum of all chunk durations. We need this to
    # convert FFmpeg's `out_time_us` to a 0..1 progress fraction.
    total_duration = sum(c.duration for c in chunks if c.duration > 0)
    log(
        f"Running segmented FFmpeg export "
        f"({total_duration:.1f}s of output to encode)..."
    )

    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, bufsize=1,
        creationflags=creation_flags,
    )

    # Parse the streaming -progress output. Lines come as `key=value`,
    # and each progress block ends with `progress=continue` / `progress=end`.
    out_time_re = re.compile(r"out_time_us=(\d+)")
    fps_re = re.compile(r"^fps=([\d.]+)")
    speed_re = re.compile(r"^speed=\s*([\d.]+)x")
    error_lines: list[str] = []

    last_log_time = 0.0
    last_fps = 0.0
    last_speed = 0.0
    LOG_INTERVAL_S = 3.0

    try:
        for line in proc.stderr:
            line = line.rstrip()
            if not line:
                continue

            m = out_time_re.search(line)
            if m:
                current_us = int(m.group(1))
                current_s = current_us / 1_000_000.0
                if total_duration > 0:
                    fraction = max(0.0, min(1.0, current_s / total_duration))
                    if progress_callback:
                        progress_callback(fraction)

                    now = time.monotonic()
                    if now - last_log_time >= LOG_INTERVAL_S:
                        last_log_time = now
                        eta_s = 0.0
                        if last_speed > 0:
                            remaining = max(0.0, total_duration - current_s)
                            eta_s = remaining / last_speed
                        eta_str = (
                            f"ETA {int(eta_s) // 60:d}:{int(eta_s) % 60:02d}"
                            if eta_s > 0 else "ETA --:--"
                        )
                        log(
                            f"  Encoding... {fraction * 100:5.1f}%  "
                            f"({current_s:.1f}s / {total_duration:.1f}s)  "
                            f"@ {last_fps:.1f} fps  ({last_speed:.2f}x)  {eta_str}"
                        )
                continue

            m = fps_re.match(line)
            if m:
                try:
                    last_fps = float(m.group(1))
                except ValueError:
                    pass
                continue

            m = speed_re.match(line)
            if m:
                try:
                    last_speed = float(m.group(1))
                except ValueError:
                    pass
                continue

            # Anything that isn't a progress key is either FFmpeg's own
            # `-loglevel error` output or a `progress=...` end marker;
            # we capture errors in case the run fails so the caller gets
            # a useful error message.
            if line.startswith("progress=") or "=" in line and line.split("=", 1)[0].isalpha():
                continue
            error_lines.append(line)
    finally:
        proc.stderr.close()
        proc.wait()

    if proc.returncode != 0:
        err = "\n".join(error_lines).strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"Segmented FFmpeg export failed:\n{err}")

    if progress_callback:
        progress_callback(1.0)
    log(f"Segmented export complete: {output_path}")
    return output_path
