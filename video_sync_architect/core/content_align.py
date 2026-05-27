"""
Find where Primary (e.g. English with a long broadcaster intro) first lines up
with Target (e.g. Portuguese that starts on episode content).

Uses several early Target time probes, matches each on the full Primary
timeline, then picks the offset cluster that best explains shared content —
avoiding a false match on Primary-only intro/slate frames.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Optional

from .media_info import MediaInfo
from ..utils.ffmpeg_utils import extract_frame_at_index, extract_frames_range
from ..utils.hashing import (
    compute_phash_from_raw,
    frame_has_content,
    hamming_distance,
)


def _parabolic_subdelta(
    hd_minus: Optional[int],
    hd_zero: int,
    hd_plus: Optional[int],
) -> float:
    if hd_minus is None or hd_plus is None:
        return 0.0
    denom = float(hd_minus - 2 * hd_zero + hd_plus)
    if denom <= 1e-9:
        return 0.0
    cand = 0.5 * (hd_minus - hd_plus) / denom
    if cand <= -0.95 or cand >= 0.95:
        return 0.0
    return float(cand)


@dataclass
class ContentStartAnchor:
    """First reliable shared-content alignment (Primary intro skipped)."""
    offset_seconds: float
    primary_frame: int
    primary_time: float
    target_frame: int
    target_time: float
    hamming_distance: int
    n_probes: int
    median_hd: float


@dataclass
class _ProbeHit:
    target_time: float
    target_frame: int
    primary_frame: int
    primary_time: float
    offset_seconds: float
    hamming: int


def _match_target_on_primary(
    primary: MediaInfo,
    ref_hash,
    coarse_step: int,
    fine_window: int,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[tuple[int, int, float]]:
    """
    Coarse-to-fine pHash match of `ref_hash` on Primary.
    Returns (best_frame, best_hd, sub_delta_frames) or None.
    """
    if primary.fps <= 0 or primary.total_frames <= 0:
        return None

    total = primary.total_frames
    best_hd = 999
    best_frame = 0

    for frame_idx, raw in extract_frames_range(
        primary.filepath, 0, total, primary.fps, step=coarse_step,
    ):
        if cancel_check and cancel_check():
            return None
        d = hamming_distance(ref_hash, compute_phash_from_raw(raw))
        if d < best_hd:
            best_hd = d
            best_frame = frame_idx
        if d == 0:
            break

    window_start = max(0, best_frame - fine_window)
    window_end = min(total, best_frame + fine_window + 1)
    for frame_idx, raw in extract_frames_range(
        primary.filepath, window_start, window_end, primary.fps, step=1,
    ):
        if cancel_check and cancel_check():
            return None
        d = hamming_distance(ref_hash, compute_phash_from_raw(raw))
        if d < best_hd:
            best_hd = d
            best_frame = frame_idx

    hd_minus = (
        _hamming_at(primary, best_frame - 1, ref_hash)
        if best_frame > 0 else None
    )
    hd_plus = (
        _hamming_at(primary, best_frame + 1, ref_hash)
        if best_frame < total - 1 else None
    )
    sub = _parabolic_subdelta(hd_minus, best_hd, hd_plus)
    return best_frame, best_hd, sub


def _hamming_at(primary: MediaInfo, frame_idx: int, ref_hash) -> Optional[int]:
    raw = extract_frame_at_index(primary.filepath, frame_idx, primary.fps)
    if raw is None:
        return None
    return hamming_distance(ref_hash, compute_phash_from_raw(raw))


def _target_frame_at_time(
    target: MediaInfo,
    time_s: float,
    max_search_frames: int,
) -> Optional[tuple[int, bytes]]:
    """Frame index + raw bytes at `time_s`, searching nearby for content."""
    if target.fps <= 0:
        return None
    base = int(round(time_s * target.fps))
    base = max(0, min(base, target.total_frames - 1))
    for delta in range(0, max_search_frames, max(1, int(target.fps // 2))):
        for sign in (0, 1, -1):
            idx = base + sign * delta
            if idx < 0 or idx >= target.total_frames:
                continue
            raw = extract_frame_at_index(target.filepath, idx, target.fps)
            if raw and frame_has_content(raw):
                return idx, raw
    return None


def _cluster_offsets(hits: list[_ProbeHit], tolerance_s: float) -> list[list[_ProbeHit]]:
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: h.offset_seconds)
    clusters: list[list[_ProbeHit]] = [[ordered[0]]]
    for hit in ordered[1:]:
        if abs(hit.offset_seconds - clusters[-1][-1].offset_seconds) <= tolerance_s:
            clusters[-1].append(hit)
        else:
            clusters.append([hit])
    return clusters


def find_shared_content_anchor(
    primary: MediaInfo,
    target: MediaInfo,
    coarse_step: int = 8,
    fine_window: int = 22,
    hamming_threshold: int = 12,
    probe_times_s: tuple[float, ...] = (2.0, 8.0, 18.0, 35.0, 60.0, 90.0),
    cluster_tolerance_s: float = 0.35,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[ContentStartAnchor]:
    """
    Probe early Target times, match each on full Primary, return the best
    shared-content offset (typically after a Primary-only intro).
    """
    def log(msg: str, prog: float = -1.0) -> None:
        if progress_callback:
            progress_callback(msg, prog)

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    if target.fps <= 0 or primary.fps <= 0:
        return None

    target_duration = target.total_frames / target.fps
    probe_times = [
        t for t in probe_times_s
        if t < target_duration - 5.0
    ]
    if not probe_times:
        probe_times = [min(5.0, target_duration * 0.1)]

    hits: list[_ProbeHit] = []
    search_span = max(1, int(target.fps * 8))

    log(
        "Intro align: probing early Target times against full Primary...",
        0.02,
    )

    for pi, t_tgt in enumerate(probe_times):
        if cancelled():
            return None

        picked = _target_frame_at_time(target, t_tgt, search_span)
        if picked is None:
            continue
        tgt_frame, raw = picked
        ref_hash = compute_phash_from_raw(raw)
        tgt_time = tgt_frame / target.fps

        match = _match_target_on_primary(
            primary, ref_hash, coarse_step, fine_window, cancel_check,
        )
        if match is None:
            continue
        pri_frame, hd, sub = match
        pri_time = (pri_frame + sub) / primary.fps
        off = pri_time - tgt_time
        hits.append(_ProbeHit(
            target_time=tgt_time,
            target_frame=tgt_frame,
            primary_frame=pri_frame,
            primary_time=pri_time,
            offset_seconds=off,
            hamming=hd,
        ))
        log(
            f"  Probe target t={tgt_time:.1f}s -> primary t={pri_time:.2f}s "
            f"HD={hd} offset={off:+.3f}s",
            -1,
        )
        if progress_callback and probe_times:
            progress_callback("", 0.02 + 0.12 * (pi + 1) / len(probe_times))

    if not hits:
        log("Intro align: no early Target probes produced matches.", 0.14)
        return None

    clusters = _cluster_offsets(hits, cluster_tolerance_s)
    # Prefer: (1) enough agreeing probes, (2) lowest median HD, (3) later Primary
    # time when HD is similar (Primary-only intro false matches sit earlier).
    def cluster_score(cluster: list[_ProbeHit]) -> tuple:
        hds = [h.hamming for h in cluster]
        pri_ts = [h.primary_time for h in cluster]
        med_hd = statistics.median(hds)
        med_pri = statistics.median(pri_ts)
        med_off = statistics.median([h.offset_seconds for h in cluster])
        return (-len(cluster), med_hd, -med_pri, med_off)

    viable = [c for c in clusters if len(c) >= 2]
    if not viable:
        viable = [max(clusters, key=len)]

    best_cluster = min(viable, key=cluster_score)
    # Tie-break: if two clusters have similar HD, take later Primary content.
    best_hd = min(h.hamming for h in best_cluster)
    tied = [
        c for c in viable
        if min(h.hamming for h in c) <= best_hd + 2
    ]
    if len(tied) > 1:
        best_cluster = max(
            tied,
            key=lambda c: statistics.median([h.primary_time for h in c]),
        )

    med_off = statistics.median([h.offset_seconds for h in best_cluster])
    med_pri_t = statistics.median([h.primary_time for h in best_cluster])
    med_tgt_t = statistics.median([h.target_time for h in best_cluster])
    med_hd = statistics.median([h.hamming for h in best_cluster])
    rep = min(best_cluster, key=lambda h: h.hamming)

    if med_hd > hamming_threshold * 2:
        log(
            f"Intro align: best cluster HD={med_hd:.0f} too high — ignoring.",
            0.14,
        )
        return None

    log(
        f"Intro align: {len(best_cluster)} probe(s) agree — first shared content "
        f"~ Primary t={med_pri_t:.2f}s, Target t={med_tgt_t:.2f}s "
        f"(offset {med_off:+.3f}s, median HD={med_hd:.0f}).",
        0.14,
    )

    return ContentStartAnchor(
        offset_seconds=float(med_off),
        primary_frame=rep.primary_frame,
        primary_time=float(med_pri_t),
        target_frame=rep.target_frame,
        target_time=float(med_tgt_t),
        hamming_distance=int(round(med_hd)),
        n_probes=len(best_cluster),
        median_hd=float(med_hd),
    )
