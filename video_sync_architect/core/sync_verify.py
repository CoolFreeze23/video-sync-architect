"""
Post-export verification (Debug Mode).

Compares a successfully-rendered synced file against the primary
reference and produces a human-readable drift report:

    <output>.sync_report.txt

Three independent checks are run:

    1. Visual frame drift
       Dense pHash sampling across Primary. For each sample, hash
       the Primary frame and search +/- a small window in the synced
       file for the closest-matching frame. Record the frame delta.

    2. Audio drift
       Split the timeline into ~30 s windows; for each window run a
       short VAD cross-correlation between Primary and the synced
       file restricted to a small lag range. Record the per-window
       residual offset.

    3. Scene-cut alignment
       Detect scene cuts in both files (whole runtime), pair every
       primary cut with the nearest synced cut, record the delta.

The report contains a summary section (max / mean drift, % within
half a frame) followed by per-moment lines for any residual above
half of one frame period (strict "frame-perfect" QA).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..utils.ffmpeg_utils import FFMPEG, extract_frame_at_index, extract_frames_range
from ..utils.hashing import compute_phash_from_raw, hamming_distance, frame_has_content
from .media_info import MediaInfo
from .audio_sync import (
    SAMPLE_RATE,
    FRAME_MS,
    _compute_vad_signal,
    _find_peak_offset,
)
from .scene_cut_sync import detect_scene_cuts


# --- Tuning ----------------------------------------------------------------

# Visual sampling (dense = slower but catches local drift)
VISUAL_NUM_SAMPLES = 150         # evenly spaced points across Primary runtime
VISUAL_SEARCH_S = 0.75           # +/- search window around expected synced time
VISUAL_SKIP_HEAD_S = 5.0
VISUAL_SKIP_TAIL_S = 5.0
VISUAL_MAX_USABLE_HD = 28

# Audio (shorter windows = finer localization, more FFmpeg work)
AUDIO_WINDOW_S = 15.0
AUDIO_MAX_RESIDUAL_S = 0.50
AUDIO_MIN_PEAK = 0.20

# Scene cuts
SCENE_PAIR_TOLERANCE_S = 1.00


def report_threshold_ms(fps: float) -> float:
    """
    Visual / scene-cut drift reporting: half of one frame at the given fps.
    """
    return 0.5 * (1000.0 / max(fps, 1e-6))


def audio_report_threshold_ms(fps: float) -> float:
    """
    Audio window residual: at least one full video frame, and never below
    one VAD bin (FRAME_MS), because VAD cross-correlation cannot resolve
    finer than ~20 ms without false positives.
    """
    one_fr = 1000.0 / max(fps, 1e-6)
    return max(one_fr, float(FRAME_MS))


# --- Data classes ----------------------------------------------------------

@dataclass
class VisualDrift:
    primary_time: float          # seconds in Primary timeline
    delta_frames: float          # synced vs primary (signed, sub-frame refined)
    delta_ms: float
    hamming: int                 # best Hamming distance at integer minimum


@dataclass
class AudioDrift:
    window_start: float          # seconds (Primary timeline)
    window_end: float
    delta_ms: float
    peak_score: float


@dataclass
class SceneDrift:
    primary_time: float
    synced_time: float
    delta_frames: float          # sub-frame from time delta * fps
    delta_ms: float


@dataclass
class VerifyReport:
    output_path: str
    primary_path: str
    fps: float
    duration: float

    visual_samples: list[VisualDrift] = field(default_factory=list)
    visual_unreliable: int = 0   # samples skipped because no usable match found

    audio_windows: list[AudioDrift] = field(default_factory=list)
    audio_skipped: int = 0       # windows skipped due to bad signal

    scene_pairs: list[SceneDrift] = field(default_factory=list)
    scene_unmatched_primary: int = 0
    scene_unmatched_synced: int = 0
    n_cuts_primary: int = 0
    n_cuts_synced: int = 0

    # Pre-computed summary stats (populated by `_populate_summary`).
    visual_max_abs_frames: float = 0.0
    visual_mean_abs_frames: float = 0.0
    visual_pct_within_half_frame: float = 0.0
    audio_max_ms: float = 0.0
    audio_mean_ms: float = 0.0
    scene_max_abs_frames: float = 0.0
    scene_mean_abs_frames: float = 0.0


# --- Time formatting --------------------------------------------------------

def _fmt_tc(seconds: float, fps: float) -> str:
    """Format a timestamp as H:MM:SS:FF using rounded integer fps."""
    if seconds < 0:
        return "-" + _fmt_tc(-seconds, fps)
    fps_int = max(1, int(round(fps)))
    total_frames = int(round(seconds * fps_int))
    f = total_frames % fps_int
    s = (total_frames // fps_int) % 60
    m = (total_frames // (fps_int * 60)) % 60
    h = total_frames // (fps_int * 3600)
    return f"{h}:{m:02d}:{s:02d}:{f:02d}"


# --- Visual drift -----------------------------------------------------------

def _hash_window(filepath: str, fps: float, total_frames: int,
                 center_time: float, window_seconds: float
                 ) -> list[tuple[int, "object"]]:
    """
    Pull a contiguous range of frames around `center_time` (+/-
    window_seconds) at native step (every frame). Returns a list of
    (relative_offset_frames_from_center, phash) pairs ordered from
    earliest to latest.
    """
    half = max(1, int(round(window_seconds * fps)))
    center_frame = int(round(center_time * fps))
    start = max(0, center_frame - half)
    end = min(total_frames, center_frame + half + 1)
    out: list[tuple[int, object]] = []
    for global_idx, raw in extract_frames_range(filepath, start, end, fps, step=1):
        try:
            h = compute_phash_from_raw(raw)
        except Exception:
            continue
        out.append((global_idx - center_frame, h))
    return out


def _refine_hd_parabolic(hd_by_rel: dict[int, int], best_rel: int) -> float:
    """
    Sub-frame correction (in synced-frame units) around the integer
    Hamming minimum, matching VisualSyncEngine.sample_offsets_visually.
    """
    hd_zero = hd_by_rel.get(best_rel)
    if hd_zero is None:
        return 0.0
    hd_minus = hd_by_rel.get(best_rel - 1)
    hd_plus = hd_by_rel.get(best_rel + 1)
    if hd_minus is None or hd_plus is None:
        return 0.0
    denom = float(hd_minus - 2 * hd_zero + hd_plus)
    if denom <= 1e-9:
        return 0.0
    cand = 0.5 * (hd_minus - hd_plus) / denom
    if cand <= -0.95 or cand >= 0.95:
        return 0.0
    return float(cand)


def _compare_visual(primary: MediaInfo, synced: MediaInfo,
                    log: Callable[[str, float], None],
                    cancel_check: Optional[Callable[[], bool]],
                    report: VerifyReport) -> None:
    """Dense pHash visual comparison populating `report.visual_samples`."""

    fps = primary.fps if primary.fps > 0 else synced.fps
    if fps <= 0:
        log("Verify(visual): unknown fps, skipping visual comparison.", -1)
        return

    duration = min(primary.duration, synced.duration)
    if duration <= 0:
        log("Verify(visual): zero duration, skipping.", -1)
        return

    head = VISUAL_SKIP_HEAD_S
    tail = VISUAL_SKIP_TAIL_S
    if duration <= head + tail + 1.0:
        head = 0.0
        tail = 0.0

    n = max(2, VISUAL_NUM_SAMPLES)
    span = max(0.0, duration - head - tail)
    if span <= 0:
        log("Verify(visual): runtime too short for sampling.", -1)
        return

    times = [head + (i + 0.5) * span / n for i in range(n)]

    log(f"Verify(visual): sampling {n} points across "
        f"{span:.1f}s of runtime...", 0.0)

    for i, t in enumerate(times):
        if cancel_check and cancel_check():
            return

        pri_frame = primary.time_to_frame(t)
        raw_p = extract_frame_at_index(primary.filepath, pri_frame, primary.fps)
        if raw_p is None:
            report.visual_unreliable += 1
            continue
        if not frame_has_content(raw_p):
            # Skip mostly-uniform frames (black, slates) - they will
            # match anything in the synced window and inflate drift.
            continue
        try:
            hash_p = compute_phash_from_raw(raw_p)
        except Exception:
            report.visual_unreliable += 1
            continue

        candidates = _hash_window(synced.filepath, synced.fps, synced.total_frames,
                                  center_time=t, window_seconds=VISUAL_SEARCH_S)
        if not candidates:
            report.visual_unreliable += 1
            continue

        hd_by_rel: dict[int, int] = {}
        for delta_frames, hash_s in candidates:
            hd_by_rel[delta_frames] = hamming_distance(hash_p, hash_s)

        if not hd_by_rel:
            report.visual_unreliable += 1
            continue

        best_delta = min(hd_by_rel, key=lambda k: hd_by_rel[k])
        best_hd = hd_by_rel[best_delta]

        if best_hd > VISUAL_MAX_USABLE_HD:
            report.visual_unreliable += 1
            continue

        sub = _refine_hd_parabolic(hd_by_rel, best_delta)
        refined_delta = float(best_delta) + sub
        frame_ms = 1000.0 / fps if fps > 0 else 0.0
        delta_ms = refined_delta * frame_ms
        report.visual_samples.append(VisualDrift(
            primary_time=t,
            delta_frames=refined_delta,
            delta_ms=float(delta_ms),
            hamming=int(best_hd),
        ))

        if (i + 1) % 15 == 0 or i == n - 1:
            log(f"Verify(visual): {i + 1}/{n} samples done", (i + 1) / n)


# --- Audio drift ------------------------------------------------------------

def _extract_window_pcm(filepath: str, start: float, duration: float
                        ) -> Optional[np.ndarray]:
    cmd = [
        FFMPEG.path, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{start:.3f}",
        "-i", filepath,
        "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-",
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run(
        cmd, capture_output=True, check=False, creationflags=creation_flags,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        return None
    return pcm.astype(np.float32) / 32768.0


def _compare_audio(primary: MediaInfo, synced: MediaInfo,
                   log: Callable[[str, float], None],
                   cancel_check: Optional[Callable[[], bool]],
                   report: VerifyReport) -> None:
    if not primary.has_audio or not synced.has_audio:
        log("Verify(audio): one or both files lack audio - skipping.", -1)
        return

    duration = min(primary.duration, synced.duration)
    if duration <= 0:
        return

    head = VISUAL_SKIP_HEAD_S
    tail = VISUAL_SKIP_TAIL_S
    span = max(0.0, duration - head - tail)
    if span < AUDIO_WINDOW_S:
        log("Verify(audio): runtime too short for windowed comparison.", -1)
        return

    n_windows = max(1, int(span // AUDIO_WINDOW_S))
    log(f"Verify(audio): {n_windows} windows of {AUDIO_WINDOW_S:.0f}s "
        "(VAD cross-correlation)...", 0.0)

    for i in range(n_windows):
        if cancel_check and cancel_check():
            return
        w_start = head + i * AUDIO_WINDOW_S
        w_end = min(duration - tail, w_start + AUDIO_WINDOW_S)
        w_dur = w_end - w_start
        if w_dur < 5.0:
            continue

        pcm_p = _extract_window_pcm(primary.filepath, w_start, w_dur)
        pcm_s = _extract_window_pcm(synced.filepath, w_start, w_dur)
        if pcm_p is None or pcm_s is None:
            report.audio_skipped += 1
            continue

        vad_p = _compute_vad_signal(pcm_p)
        vad_s = _compute_vad_signal(pcm_s)
        if len(vad_p) == 0 or len(vad_s) == 0:
            report.audio_skipped += 1
            continue
        if float(np.mean(vad_p > 0)) < 0.05 or float(np.mean(vad_s > 0)) < 0.05:
            # Mostly silent window: cross-correlation will be noise.
            report.audio_skipped += 1
            continue

        offset_s, peak = _find_peak_offset(
            vad_p, vad_s, max_lag_seconds=AUDIO_MAX_RESIDUAL_S,
        )
        if peak < AUDIO_MIN_PEAK:
            report.audio_skipped += 1
            continue

        report.audio_windows.append(AudioDrift(
            window_start=w_start,
            window_end=w_end,
            delta_ms=float(offset_s) * 1000.0,
            peak_score=float(peak),
        ))

        if (i + 1) % 4 == 0 or i == n_windows - 1:
            log(f"Verify(audio): {i + 1}/{n_windows} windows done",
                (i + 1) / n_windows)


# --- Scene-cut alignment ----------------------------------------------------

def _compare_scene_cuts(primary: MediaInfo, synced: MediaInfo,
                        log: Callable[[str, float], None],
                        cancel_check: Optional[Callable[[], bool]],
                        report: VerifyReport) -> None:
    fps = primary.fps if primary.fps > 0 else synced.fps
    duration = min(primary.duration, synced.duration)
    if duration <= 0:
        return

    log("Verify(scene-cut): detecting cuts in Primary...", 0.0)
    cuts_p = detect_scene_cuts(
        primary.filepath, max_duration=duration,
        skip_start=VISUAL_SKIP_HEAD_S,
    )
    if cancel_check and cancel_check():
        return
    log("Verify(scene-cut): detecting cuts in Synced...", 0.5)
    cuts_s = detect_scene_cuts(
        synced.filepath, max_duration=duration,
        skip_start=VISUAL_SKIP_HEAD_S,
    )
    # `detect_scene_cuts` returns times relative to skip_start.
    # Re-add skip_start so we can compare on the absolute timeline.
    cuts_p = [VISUAL_SKIP_HEAD_S + t for t in cuts_p]
    cuts_s = [VISUAL_SKIP_HEAD_S + t for t in cuts_s]
    cuts_s_sorted = sorted(cuts_s)

    report.n_cuts_primary = len(cuts_p)
    report.n_cuts_synced = len(cuts_s_sorted)

    log(f"Verify(scene-cut): primary={len(cuts_p)} cuts, "
        f"synced={len(cuts_s_sorted)} cuts; pairing...", 0.9)

    matched_synced: set[int] = set()

    for tp in cuts_p:
        # Binary search nearest in synced.
        if not cuts_s_sorted:
            report.scene_unmatched_primary += 1
            continue
        # Linear search is plenty fast for ~300 cuts.
        nearest_idx = -1
        best_abs = float("inf")
        for j, ts in enumerate(cuts_s_sorted):
            if j in matched_synced:
                continue
            d = abs(ts - tp)
            if d < best_abs:
                best_abs = d
                nearest_idx = j
            if ts - tp > SCENE_PAIR_TOLERANCE_S and best_abs < float("inf"):
                # cuts_s_sorted is sorted; once we're past tolerance
                # window the rest will only get further away.
                break

        if nearest_idx < 0 or best_abs > SCENE_PAIR_TOLERANCE_S:
            report.scene_unmatched_primary += 1
            continue

        matched_synced.add(nearest_idx)
        ts = cuts_s_sorted[nearest_idx]
        delta_s = ts - tp
        d_frames = float(delta_s * fps) if fps > 0 else 0.0
        report.scene_pairs.append(SceneDrift(
            primary_time=tp,
            synced_time=ts,
            delta_frames=d_frames,
            delta_ms=float(delta_s) * 1000.0,
        ))

    report.scene_unmatched_synced = max(
        0, len(cuts_s_sorted) - len(matched_synced)
    )


# --- Orchestrator ----------------------------------------------------------

def verify_sync(primary: MediaInfo, synced: MediaInfo, *,
                progress_callback: Optional[Callable[[str, float], None]] = None,
                cancel_check: Optional[Callable[[], bool]] = None,
                ) -> VerifyReport:
    """Run all three checks and return a populated `VerifyReport`."""

    def log(msg: str, progress: float = -1.0) -> None:
        if progress_callback:
            progress_callback(msg, progress)

    fps = primary.fps if primary.fps > 0 else synced.fps
    duration = min(primary.duration, synced.duration)
    report = VerifyReport(
        output_path=synced.filepath,
        primary_path=primary.filepath,
        fps=fps,
        duration=duration,
    )

    log("Verify: starting visual frame-drift check...", 0.0)
    _compare_visual(primary, synced, log, cancel_check, report)
    if cancel_check and cancel_check():
        return report

    log("Verify: starting audio drift check...", -1)
    _compare_audio(primary, synced, log, cancel_check, report)
    if cancel_check and cancel_check():
        return report

    log("Verify: starting scene-cut alignment check...", -1)
    _compare_scene_cuts(primary, synced, log, cancel_check, report)

    _populate_summary(report)
    return report


# --- Summary + report formatting -------------------------------------------

def _populate_summary(report: VerifyReport) -> None:
    fps = report.fps if report.fps > 0 else 24.0
    half_fr = 0.5
    if report.visual_samples:
        deltas = [abs(s.delta_frames) for s in report.visual_samples]
        report.visual_max_abs_frames = max(deltas)
        report.visual_mean_abs_frames = sum(deltas) / len(deltas)
        within = sum(1 for d in deltas if d <= half_fr)
        report.visual_pct_within_half_frame = within / len(deltas)

    if report.audio_windows:
        a = [abs(w.delta_ms) for w in report.audio_windows]
        report.audio_max_ms = max(a)
        report.audio_mean_ms = sum(a) / len(a)

    if report.scene_pairs:
        s = [abs(p.delta_frames) for p in report.scene_pairs]
        report.scene_max_abs_frames = max(s)
        report.scene_mean_abs_frames = sum(s) / len(s)


def format_report(report: VerifyReport) -> str:
    fps = report.fps if report.fps > 0 else 24.0
    thr = report_threshold_ms(fps)
    thr_audio = audio_report_threshold_ms(fps)
    L: list[str] = []
    L.append("=" * 72)
    L.append("  VIDEO SYNC ARCHITECT - DEBUG VERIFICATION REPORT")
    L.append("=" * 72)
    L.append(f"Synced file : {report.output_path}")
    L.append(f"Primary file: {report.primary_path}")
    L.append(f"Runtime     : {report.duration:.2f}s @ {fps:.3f} fps  "
             f"(1 frame ~ {1000.0 / fps:.1f} ms)")
    L.append(f"Strict QA   : visual/scene list if |drift| > {thr:.2f} ms "
             f"(half a frame); audio if > {thr_audio:.2f} ms (>= 1 frame / "
             f"VAD bin limit)")
    L.append("")

    # ---- Visual summary ----------------------------------------------------
    L.append("-" * 72)
    L.append("VISUAL FRAME DRIFT")
    L.append("-" * 72)
    if report.visual_samples:
        L.append(
            f"  Samples used     : {len(report.visual_samples)} "
            f"(skipped {report.visual_unreliable} unreliable)"
        )
        L.append(
            f"  Max |drift|      : {report.visual_max_abs_frames:.3f} frames "
            f"({report.visual_max_abs_frames * 1000.0 / fps:.2f} ms)"
        )
        L.append(
            f"  Mean |drift|     : {report.visual_mean_abs_frames:.3f} frames "
            f"({report.visual_mean_abs_frames * 1000.0 / fps:.2f} ms)"
        )
        L.append(
            f"  Within 0.5 frame : "
            f"{report.visual_pct_within_half_frame * 100.0:.1f}% of samples"
        )

        flagged = [s for s in report.visual_samples if abs(s.delta_ms) > thr]
        L.append("")
        if flagged:
            L.append(
                f"  Samples over {thr:.2f} ms ({len(flagged)} found):"
            )
            L.append(f"    {'Primary TC':<14}  {'delta (fr)':>14}  "
                     f"{'delta (ms)':>11}  {'pHash HD':>9}")
            for s in flagged:
                tc = _fmt_tc(s.primary_time, fps)
                L.append(
                    f"    {tc:<14}  {s.delta_frames:>+14.2f}  "
                    f"{s.delta_ms:>+11.2f}  {s.hamming:>9d}"
                )
        else:
            L.append(
                f"  No samples exceed the half-frame threshold ({thr:.2f} ms)."
            )

    # ---- Audio summary -----------------------------------------------------
    L.append("-" * 72)
    L.append("AUDIO DRIFT (windowed VAD cross-correlation)")
    L.append("-" * 72)
    if report.audio_windows:
        L.append(
            f"  Windows used     : {len(report.audio_windows)} "
            f"(skipped {report.audio_skipped} silent / weak)"
        )
        L.append(f"  Max  residual    : {report.audio_max_ms:+.0f} ms")
        L.append(f"  Mean |residual|  : {report.audio_mean_ms:.1f} ms")

        flagged = [w for w in report.audio_windows if abs(w.delta_ms) > thr_audio]
        L.append("")
        if flagged:
            L.append(
                f"  Windows over {thr_audio:.2f} ms ({len(flagged)} found):"
            )
            L.append(f"    {'Window (Primary TC)':<28}  {'delta':>10}  {'peak':>6}")
            for w in flagged:
                wstr = (f"{_fmt_tc(w.window_start, fps)}-"
                        f"{_fmt_tc(w.window_end, fps)}")
                L.append(
                    f"    {wstr:<28}  {w.delta_ms:>+9.2f} ms  "
                    f"{w.peak_score:>6.2f}"
                )
        else:
            L.append(
                f"  No windows exceed the audio threshold ({thr_audio:.2f} ms)."
            )
    else:
        L.append("  No usable audio windows (no audio, too short, or all skipped).")
    L.append("")

    # ---- Scene-cut summary -------------------------------------------------
    L.append("-" * 72)
    L.append("SCENE-CUT ALIGNMENT")
    L.append("-" * 72)
    L.append(
        f"  Cuts (Primary)   : {report.n_cuts_primary}    "
        f"Cuts (Synced): {report.n_cuts_synced}    "
        f"Pairs matched: {len(report.scene_pairs)}"
    )
    if report.scene_pairs:
        L.append(
            f"  Max |cut delta|  : {report.scene_max_abs_frames:.3f} frames "
            f"({report.scene_max_abs_frames * 1000.0 / fps:.2f} ms)"
        )
        L.append(
            f"  Mean |cut delta| : {report.scene_mean_abs_frames:.3f} frames"
        )
    L.append(
        f"  Unmatched cuts   : Primary={report.scene_unmatched_primary}, "
        f"Synced={report.scene_unmatched_synced}"
    )
    if report.scene_pairs:
        flagged = [p for p in report.scene_pairs if abs(p.delta_ms) > thr]
        L.append("")
        if flagged:
            L.append(
                f"  Cut pairs over {thr:.2f} ms ({len(flagged)} found):"
            )
            L.append(f"    {'Primary TC':<14}  {'Synced TC':<14}  "
                     f"{'delta (fr)':>12}  {'delta (ms)':>11}")
            for p in flagged:
                tc_p = _fmt_tc(p.primary_time, fps)
                tc_s = _fmt_tc(p.synced_time, fps)
                L.append(
                    f"    {tc_p:<14}  {tc_s:<14}  "
                    f"{p.delta_frames:>+12.2f}  {p.delta_ms:>+11.2f}"
                )
        else:
            L.append(
                f"  No cut pairs exceed the half-frame threshold ({thr:.2f} ms)."
            )
    L.append("")

    # ---- Verdict -----------------------------------------------------------
    L.append("-" * 72)
    L.append("VERDICT")
    L.append("-" * 72)
    issues = verification_issues(report)

    if not issues:
        L.append(
            "  CLEAN: visual & scene-cuts within half a frame; audio within "
            "one frame (VAD-limited) of Primary."
        )
    else:
        L.append("  ISSUES: " + "; ".join(issues))
        L.append("  See per-moment listings above.")
    L.append("=" * 72)
    L.append("")
    return "\n".join(L)


def verification_issues(report: VerifyReport) -> list[str]:
    """
    Human-readable issue strings if any modality exceeds its strict
    threshold (half-frame for visual/scene; >= 1 frame for audio/VAD).
    """
    fps = report.fps if report.fps > 0 else 24.0
    thr_v = report_threshold_ms(fps)
    thr_a = audio_report_threshold_ms(fps)
    out: list[str] = []
    if report.visual_samples:
        mx = max(abs(s.delta_ms) for s in report.visual_samples)
        if mx > thr_v:
            out.append(f"visual max {mx:.2f} ms (thr {thr_v:.2f} ms)")
    if report.audio_windows:
        mx = max(abs(w.delta_ms) for w in report.audio_windows)
        if mx > thr_a:
            out.append(f"audio max {mx:.2f} ms (thr {thr_a:.2f} ms)")
    if report.scene_pairs:
        mx = max(abs(p.delta_ms) for p in report.scene_pairs)
        if mx > thr_v:
            out.append(f"scene-cut max {mx:.2f} ms (thr {thr_v:.2f} ms)")
    return out


def write_report(report: VerifyReport,
                 path: Optional[str] = None) -> str:
    """
    Write the formatted report to disk. If `path` is None, places it
    next to the synced file as `<output>.sync_report.txt`. Returns the
    path written.
    """
    if path is None:
        base, _ = os.path.splitext(report.output_path)
        path = base + ".sync_report.txt"
    text = format_report(report)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
