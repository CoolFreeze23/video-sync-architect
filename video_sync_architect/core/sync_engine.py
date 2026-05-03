"""
Sync engine: abstract base class and VisualSyncEngine (pHash-based).
All offsets are computed in seconds (float64) for framerate independence.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable

import statistics

from .media_info import MediaInfo
from ..utils.hashing import (
    compute_phash_from_raw,
    hamming_distance,
    select_reference_frame,
    select_reference_frames,
    frame_has_content,
    THUMB_W,
    THUMB_H,
)
from ..utils.ffmpeg_utils import extract_frames_range, extract_frame_at_index


@dataclass
class SyncResult:
    offset_seconds: float
    matched_frame_input1: int
    matched_time_input1: float
    reference_frame_input2: int
    reference_time_input2: float
    hamming_distance: int
    confidence: str  # "high", "medium", "low"


class SyncEngine(ABC):
    """Abstract base for sync strategies (visual, audio, etc.)."""

    @abstractmethod
    def find_offset(
        self,
        primary: MediaInfo,
        target: MediaInfo,
        hamming_threshold: int = 12,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[SyncResult]:
        ...


class VisualSyncEngine(SyncEngine):
    """
    Coarse-to-Fine perceptual hash matching.
    Phase 1: Scan every Nth frame of Input 1 (coarse).
    Phase 2: Scan every frame in a window around the best coarse match (fine).
    """

    def __init__(self, coarse_step: int = 6, fine_window: int = 22,
                 skip_seconds: float = 30.0):
        self.coarse_step = coarse_step
        self.fine_window = fine_window
        self.skip_seconds = skip_seconds

    def find_offset(
        self,
        primary: MediaInfo,
        target: MediaInfo,
        hamming_threshold: int = 12,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[SyncResult]:

        def log(msg: str, progress: float = -1):
            if progress_callback:
                progress_callback(msg, progress)

        def is_cancelled() -> bool:
            return cancel_check() if cancel_check else False

        # --- Select reference frame from Input 2 (Target) ---
        log("Selecting reference frame from Target video...", 0.0)
        ref_result = select_reference_frame(
            target.filepath, target.fps, target.total_frames,
            skip_seconds=self.skip_seconds,
        )
        if ref_result is None:
            log("ERROR: Could not select a reference frame from Target.", 0.0)
            return None

        ref_frame_idx, ref_raw = ref_result
        ref_time = target.frame_to_time(ref_frame_idx)
        ref_hash = compute_phash_from_raw(ref_raw)
        log(f"Reference frame selected: frame {ref_frame_idx} (t={ref_time:.3f}s)", 0.05)

        if is_cancelled():
            return None

        # --- Phase 1: Coarse scan ---
        log(
            f"Phase 1: Coarse scan (every {self.coarse_step}th frame)...",
            0.05,
        )
        total_frames_p = primary.total_frames
        best_distance = 999
        best_frame = 0

        coarse_frames_total = total_frames_p // self.coarse_step
        coarse_count = 0

        for frame_idx, raw_data in extract_frames_range(
            primary.filepath, 0, total_frames_p, primary.fps,
            step=self.coarse_step
        ):
            if is_cancelled():
                return None

            h = compute_phash_from_raw(raw_data)
            dist = hamming_distance(ref_hash, h)

            if dist < best_distance:
                best_distance = dist
                best_frame = frame_idx
                t = primary.frame_to_time(frame_idx)
                log(f"  Coarse: frame {frame_idx} (t={t:.3f}s) HD={dist}", -1)

            if dist == 0:
                log(f"  Exact match at coarse scan! Frame {frame_idx}", -1)
                break

            coarse_count += 1
            if coarse_frames_total > 0:
                progress = 0.05 + 0.60 * (coarse_count / coarse_frames_total)
                log("", progress)

        if best_distance > hamming_threshold:
            log(
                f"WARNING: Best coarse match HD={best_distance} exceeds threshold {hamming_threshold}. "
                "No reliable match found.",
                0.65,
            )
            if best_distance > hamming_threshold * 2:
                return None

        log(f"Coarse best: frame {best_frame}, HD={best_distance}", 0.65)

        if is_cancelled():
            return None

        # --- Phase 2: Fine scan ---
        if best_distance > 0:
            window_start = max(0, best_frame - self.fine_window)
            window_end = min(total_frames_p, best_frame + self.fine_window + 1)
            log(f"Phase 2: Fine scan frames {window_start}-{window_end}...", 0.65)

            fine_total = window_end - window_start
            fine_count = 0

            for frame_idx, raw_data in extract_frames_range(
                primary.filepath, window_start, window_end, primary.fps, step=1
            ):
                if is_cancelled():
                    return None

                h = compute_phash_from_raw(raw_data)
                dist = hamming_distance(ref_hash, h)

                if dist < best_distance:
                    best_distance = dist
                    best_frame = frame_idx
                    t = primary.frame_to_time(frame_idx)
                    log(f"  Fine: frame {frame_idx} (t={t:.3f}s) HD={dist}", -1)

                if dist == 0:
                    break

                fine_count += 1
                if fine_total > 0:
                    progress = 0.65 + 0.30 * (fine_count / fine_total)
                    log("", progress)

        matched_time = primary.frame_to_time(best_frame)
        offset_seconds = matched_time - ref_time

        if best_distance <= hamming_threshold // 3:
            confidence = "high"
        elif best_distance <= hamming_threshold:
            confidence = "medium"
        else:
            confidence = "low"

        result = SyncResult(
            offset_seconds=offset_seconds,
            matched_frame_input1=best_frame,
            matched_time_input1=matched_time,
            reference_frame_input2=ref_frame_idx,
            reference_time_input2=ref_time,
            hamming_distance=best_distance,
            confidence=confidence,
        )

        direction = "after" if offset_seconds >= 0 else "before"
        log(
            f"Match found: frame {best_frame} (t={matched_time:.3f}s) HD={best_distance} "
            f"[{confidence}]\n"
            f"Offset: {offset_seconds:+.3f}s (Input 2 starts {direction} Input 1)",
            0.95,
        )

        return result

    # ---------------------------------------------------------------------
    # Multi-reference matching: try several reference frames spread across
    # the Target video and keep whichever one finds the lowest Hamming
    # distance match in the Primary. Uses a single Primary scan that
    # compares each frame to ALL reference hashes for efficiency.
    # ---------------------------------------------------------------------
    def find_offset_multi_ref(
        self,
        primary: MediaInfo,
        target: MediaInfo,
        hamming_threshold: int = 12,
        num_refs: int = 5,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[SyncResult]:

        def log(msg: str, progress: float = -1):
            if progress_callback:
                progress_callback(msg, progress)

        def is_cancelled() -> bool:
            return cancel_check() if cancel_check else False

        # --- Pick N reference frames distributed across the Target ---
        log(f"Multi-ref mode: selecting {num_refs} reference frames from Target...", 0.0)
        refs = select_reference_frames(
            target.filepath, target.fps, target.total_frames,
            num_refs=num_refs,
            skip_start_seconds=self.skip_seconds,
        )
        if not refs:
            log("ERROR: Could not select any reference frames from Target.", 0.0)
            return None

        ref_data = []  # list of (frame_idx, ref_time, ref_hash)
        for idx, raw in refs:
            ref_data.append((idx, target.frame_to_time(idx), compute_phash_from_raw(raw)))

        log(
            f"Reference frames at: "
            + ", ".join(f"t={t:.1f}s" for _, t, _ in ref_data),
            0.05,
        )

        if is_cancelled():
            return None

        # --- Coarse scan: for each Primary frame, compare to ALL refs ---
        log(
            f"Phase 1: Multi-ref coarse scan (every {self.coarse_step}th frame)...",
            0.05,
        )
        total_frames_p = primary.total_frames

        # Track best distance per reference.
        best_per_ref = [(999, 0) for _ in ref_data]  # (distance, frame)
        coarse_total = total_frames_p // self.coarse_step
        coarse_count = 0

        for frame_idx, raw_data in extract_frames_range(
            primary.filepath, 0, total_frames_p, primary.fps,
            step=self.coarse_step
        ):
            if is_cancelled():
                return None

            h = compute_phash_from_raw(raw_data)
            for ri, (_, _, rh) in enumerate(ref_data):
                d = hamming_distance(rh, h)
                if d < best_per_ref[ri][0]:
                    best_per_ref[ri] = (d, frame_idx)
                    log(
                        f"  Ref#{ri + 1} (t={ref_data[ri][1]:.1f}s): "
                        f"frame {frame_idx} HD={d}",
                        -1,
                    )

            # Stop early only if every ref has hit zero (extremely unlikely).
            if all(b[0] == 0 for b in best_per_ref):
                log("  All references found exact matches.", -1)
                break

            coarse_count += 1
            if coarse_total > 0:
                log("", 0.05 + 0.55 * (coarse_count / coarse_total))

        # --- Pick the winning reference (lowest Hamming distance) ---
        winner_ri = min(range(len(best_per_ref)), key=lambda i: best_per_ref[i][0])
        winner_dist, winner_frame = best_per_ref[winner_ri]
        winner_idx, winner_time, winner_hash = ref_data[winner_ri]

        log(
            f"Best reference: Ref#{winner_ri + 1} (Target t={winner_time:.3f}s) "
            f"matched Primary frame {winner_frame} HD={winner_dist}",
            0.60,
        )

        if is_cancelled():
            return None

        # --- Fine scan around the winning frame using the winning ref ---
        if winner_dist > 0:
            window_start = max(0, winner_frame - self.fine_window)
            window_end = min(total_frames_p, winner_frame + self.fine_window + 1)
            log(
                f"Phase 2: Fine scan frames {window_start}-{window_end} "
                f"with winning reference...",
                0.65,
            )

            fine_total = window_end - window_start
            fine_count = 0

            for frame_idx, raw_data in extract_frames_range(
                primary.filepath, window_start, window_end, primary.fps, step=1
            ):
                if is_cancelled():
                    return None

                h = compute_phash_from_raw(raw_data)
                d = hamming_distance(winner_hash, h)

                if d < winner_dist:
                    winner_dist = d
                    winner_frame = frame_idx
                    log(
                        f"  Fine: frame {frame_idx} "
                        f"(t={primary.frame_to_time(frame_idx):.3f}s) HD={d}",
                        -1,
                    )

                if d == 0:
                    break

                fine_count += 1
                if fine_total > 0:
                    log("", 0.65 + 0.30 * (fine_count / fine_total))

        # --- Build the SyncResult ---
        if winner_dist > hamming_threshold * 2:
            log(
                f"WARNING: Best multi-ref HD={winner_dist} is too high to trust. "
                "Giving up.",
                0.95,
            )
            return None

        matched_time = primary.frame_to_time(winner_frame)
        offset_seconds = matched_time - winner_time

        if winner_dist <= hamming_threshold // 3:
            confidence = "high"
        elif winner_dist <= hamming_threshold:
            confidence = "medium"
        else:
            confidence = "low"

        result = SyncResult(
            offset_seconds=offset_seconds,
            matched_frame_input1=winner_frame,
            matched_time_input1=matched_time,
            reference_frame_input2=winner_idx,
            reference_time_input2=winner_time,
            hamming_distance=winner_dist,
            confidence=confidence,
        )

        direction = "after" if offset_seconds >= 0 else "before"
        log(
            f"Multi-ref match: Primary frame {winner_frame} "
            f"(t={matched_time:.3f}s) <-> Target t={winner_time:.3f}s "
            f"HD={winner_dist} [{confidence}]\n"
            f"Offset: {offset_seconds:+.3f}s (Input 2 starts {direction} Input 1)",
            0.95,
        )

        return result

    # ---------------------------------------------------------------------
    # Dense per-sample offset measurement.
    # For each sample point in Target, predicts the matching Primary time
    # using a coarse offset and pHash-matches in a TIGHT window. Returns
    # the full list of (target_t, refined_offset, hd) tuples - the caller
    # decides how to aggregate (single offset vs. segmented warp curve).
    # ---------------------------------------------------------------------
    def sample_offsets_visually(
        self,
        primary: MediaInfo,
        target: MediaInfo,
        coarse_offset: float,
        window_seconds: float = 1.5,
        num_samples: int = 15,
        skip_start_seconds: float = 30.0,
        skip_end_seconds: float = 60.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> list[tuple[float, float, int]]:
        """
        Probe `num_samples` points in Target and frame-precision-refine
        each one's offset by pHash-scanning a small Primary window.
        Returns list of (target_time_s, refined_offset_s, hamming_distance)
        for every sample that produced a usable match (HD <= 16).
        """

        def log(msg: str, prog: float = -1.0) -> None:
            if progress_callback:
                progress_callback(msg, prog)

        def is_cancelled() -> bool:
            return bool(cancel_check and cancel_check())

        if target.fps <= 0 or target.total_frames <= 0:
            return []

        target_duration = target.total_frames / target.fps
        usable_start = min(skip_start_seconds, max(0.0, target_duration - 1.0))
        usable_end = max(usable_start + 1.0, target_duration - skip_end_seconds)
        if usable_end <= usable_start:
            usable_end = target_duration

        if num_samples <= 1:
            sample_targets = [usable_start]
        else:
            step = (usable_end - usable_start) / (num_samples + 1)
            sample_targets = [usable_start + step * (i + 1) for i in range(num_samples)]

        results: list[tuple[float, float, int]] = []

        for si, t_target_s in enumerate(sample_targets):
            if is_cancelled():
                return results

            target_frame_idx = int(round(t_target_s * target.fps))
            ref_raw = None
            for delta in range(0, int(target.fps), max(1, int(target.fps // 4))):
                for sign in (1, -1):
                    idx = target_frame_idx + sign * delta
                    if idx < 0 or idx >= target.total_frames:
                        continue
                    raw = extract_frame_at_index(target.filepath, idx, target.fps)
                    if raw and frame_has_content(raw):
                        ref_raw = raw
                        target_frame_idx = idx
                        break
                if ref_raw is not None:
                    break

            if ref_raw is None:
                log(f"  Sample {si + 1}: no usable target frame, skipping.", -1)
                continue

            ref_hash = compute_phash_from_raw(ref_raw)
            ref_t = target.frame_to_time(target_frame_idx)
            predicted_primary_t = ref_t + coarse_offset

            half = window_seconds
            window_start_t = max(0.0, predicted_primary_t - half)
            window_end_t = min(primary.total_frames / primary.fps,
                               predicted_primary_t + half)
            if window_end_t <= window_start_t:
                log(
                    f"  Sample {si + 1}: predicted primary window out of "
                    f"bounds ({window_start_t:.2f}-{window_end_t:.2f}s).",
                    -1,
                )
                continue

            window_start_f = max(0, int(window_start_t * primary.fps))
            window_end_f = min(primary.total_frames,
                               int(window_end_t * primary.fps) + 1)

            # Record full HD curve so we can parabolic-interpolate the
            # minimum to sub-frame precision (fixes the off-by-one-frame
            # issue caused by pHash matching to the wrong neighbouring
            # integer frame when adjacent frames have nearly identical
            # signatures).
            hd_by_frame: dict[int, int] = {}
            best_hd = 999
            best_frame = -1
            for frame_idx, raw in extract_frames_range(
                primary.filepath, window_start_f, window_end_f, primary.fps,
                step=1,
            ):
                if is_cancelled():
                    return results
                d = hamming_distance(ref_hash, compute_phash_from_raw(raw))
                hd_by_frame[frame_idx] = d
                if d < best_hd:
                    best_hd = d
                    best_frame = frame_idx
                    # No early break: we need the neighbours of the
                    # minimum for the parabolic fit. (HD == 0 is rare
                    # in practice and the extra ~1 sec per sample is
                    # well worth the sub-frame precision.)

            if best_frame < 0 or best_hd > 16:
                log(
                    f"  Sample {si + 1} (target t={ref_t:.2f}s): "
                    f"no good match (best HD={best_hd}), skipping.",
                    -1,
                )
                continue

            # Parabolic fit on (HD_minus, HD_zero, HD_plus). If a side
            # is missing or the curvature is non-positive we just use
            # the integer minimum.
            sub_delta = 0.0
            hd_zero = hd_by_frame.get(best_frame, best_hd)
            hd_minus = hd_by_frame.get(best_frame - 1)
            hd_plus = hd_by_frame.get(best_frame + 1)
            if hd_minus is not None and hd_plus is not None:
                denom = (hd_minus - 2.0 * hd_zero + hd_plus)
                if denom > 1e-9:
                    cand = 0.5 * (hd_minus - hd_plus) / denom
                    # Clamp to neighbouring frame to keep estimate sane.
                    if -0.95 < cand < 0.95:
                        sub_delta = cand

            primary_match_t = (best_frame + sub_delta) / primary.fps
            sample_offset = primary_match_t - ref_t
            results.append((ref_t, sample_offset, best_hd))
            sub_frac_str = (f"{sub_delta:+.2f}" if sub_delta else "+0.00")
            log(
                f"  Sample {si + 1}: target t={ref_t:.3f}s -> primary frame "
                f"{best_frame}{sub_frac_str} (t={primary_match_t:.4f}s) "
                f"HD={best_hd} offset={sample_offset:+.4f}s "
                f"(delta from coarse: {(sample_offset - coarse_offset) * 1000:+.1f} ms)",
                0.05 + 0.90 * (si + 1) / max(1, len(sample_targets)),
            )

        return results

    # ---------------------------------------------------------------------
    # Aggregate dense samples into a single offset (median + outlier
    # rejection). Used when the caller wants the simple single-offset
    # path; segmented sync uses the raw samples instead.
    # ---------------------------------------------------------------------
    def aggregate_offset_samples(
        self,
        samples: list[tuple[float, float, int]],
        max_deviation: float = 0.100,
    ) -> Optional[float]:
        if not samples:
            return None
        offsets = [s[1] for s in samples]
        median_offset = statistics.median(offsets)
        inliers = [o for o in offsets if abs(o - median_offset) <= max_deviation]
        return statistics.median(inliers) if inliers else median_offset

    # ---------------------------------------------------------------------
    # Backward-compatible shim: old callers expect a single refined offset.
    # ---------------------------------------------------------------------
    def refine_offset_visually(
        self,
        primary: MediaInfo,
        target: MediaInfo,
        coarse_offset: float,
        window_seconds: float = 1.5,
        num_samples: int = 7,
        skip_start_seconds: float = 30.0,
        skip_end_seconds: float = 60.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[float]:
        samples = self.sample_offsets_visually(
            primary, target, coarse_offset,
            window_seconds=window_seconds, num_samples=num_samples,
            skip_start_seconds=skip_start_seconds,
            skip_end_seconds=skip_end_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        return self.aggregate_offset_samples(samples)
