"""
Scene-cut-based synchronization. Detects scene transitions (shot cuts)
in both videos via FFmpeg's `select=gt(scene,T)` filter, builds a cut
event signal at fixed time resolution, and cross-correlates the two
signals to find the offset.

Why this beats pHash matching:
  * A scene cut is an unambiguous, localized event in time.
  * A 24-minute anime episode typically has 100-300 cuts; a long
    sequence of cut times is a unique fingerprint that cannot be
    matched by accident.
  * Cuts survive color grading, resolution changes, watermarks, and
    hardsubs, which all confuse pHash similarity at low Hamming
    thresholds.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..utils.ffmpeg_utils import FFMPEG


# --- Tuning constants ------------------------------------------------------

DEFAULT_THRESHOLD = 0.30        # FFmpeg `scene` value [0,1]; 0.3 = clear cut
DEFAULT_DURATION_LIMIT = 600.0  # 10 minutes scanned per file
DEFAULT_SKIP_START = 5.0        # skip logos / black at the very front
BIN_MS = 20                     # cut signal time resolution (matches audio VAD)


_PTS_RE = re.compile(r"pts_time:([\d.]+)")


@dataclass
class SceneCutSyncResult:
    offset_seconds: float
    peak_score: float           # normalized correlation peak [0, 1]
    confidence: str             # "high" / "medium" / "low"
    n_cuts_primary: int
    n_cuts_target: int


# --- FFmpeg scene-cut detection -------------------------------------------

def detect_scene_cuts(filepath: str,
                      threshold: float = DEFAULT_THRESHOLD,
                      max_duration: float = DEFAULT_DURATION_LIMIT,
                      skip_start: float = DEFAULT_SKIP_START) -> list[float]:
    """
    Return a list of cut timestamps in seconds, RELATIVE to `skip_start`
    (i.e. shifted so that the first analyzed frame is at t=0). This makes
    the resulting cut signals from Primary and Target directly comparable.
    """
    cmd = [
        FFMPEG.path, "-hide_banner", "-nostdin",
        "-ss", f"{skip_start:.3f}",
        "-i", filepath,
        "-t", f"{max_duration:.3f}",
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-an", "-sn", "-f", "null", "-",
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run(
        cmd, capture_output=True, check=False, creationflags=creation_flags,
    )

    cuts: list[float] = []
    stderr = proc.stderr.decode(errors="replace") if proc.stderr else ""
    for line in stderr.splitlines():
        m = _PTS_RE.search(line)
        if m:
            try:
                cuts.append(float(m.group(1)))
            except ValueError:
                continue
    return cuts


# --- Cut signal construction ---------------------------------------------

def _cuts_to_signal(cuts: list[float], duration: float,
                    bin_ms: int = BIN_MS) -> np.ndarray:
    """
    Convert a list of cut timestamps into a 1-D signal sampled every
    `bin_ms` milliseconds. Each cut deposits a unit impulse, and the
    whole signal is then smoothed by a small triangular kernel so that
    cuts off by 1 frame between Primary and Target still correlate
    strongly.
    """
    if duration <= 0:
        return np.zeros(0, dtype=np.float32)

    n_bins = int(duration * 1000 / bin_ms) + 1
    sig = np.zeros(n_bins, dtype=np.float32)
    for t in cuts:
        idx = int(round(t * 1000 / bin_ms))
        if 0 <= idx < n_bins:
            sig[idx] += 1.0

    if len(sig) >= 5:
        kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25], dtype=np.float32)
        kernel = kernel / kernel.sum()
        sig = np.convolve(sig, kernel, mode="same")

    sig = sig - sig.mean()
    return sig.astype(np.float32)


# --- Cross-correlation ----------------------------------------------------

def _fft_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a) + len(b) - 1
    n_fft = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, n_fft)
    fb = np.fft.rfft(b, n_fft)
    full = np.fft.irfft(fa * np.conj(fb), n_fft)
    return np.concatenate([full[-(len(b) - 1):], full[: len(a)]])


def _refine_peak_parabolic(xcorr: np.ndarray, peak_idx: int) -> float:
    """
    Sub-bin peak refinement: fit a parabola through the integer peak
    and its neighbours, return the analytical maximum's fractional
    index. Pushes effective resolution well below the 20 ms bin size.
    """
    if peak_idx <= 0 or peak_idx >= len(xcorr) - 1:
        return float(peak_idx)
    y_minus = float(xcorr[peak_idx - 1])
    y_zero = float(xcorr[peak_idx])
    y_plus = float(xcorr[peak_idx + 1])
    denom = y_minus - 2.0 * y_zero + y_plus
    if abs(denom) < 1e-12:
        return float(peak_idx)
    delta = 0.5 * (y_minus - y_plus) / denom
    if delta < -1.0:
        delta = -1.0
    elif delta > 1.0:
        delta = 1.0
    return float(peak_idx) + delta


# --- Public API -----------------------------------------------------------

def find_scene_cut_offset(
    primary_filepath: str,
    target_filepath: str,
    threshold: float = DEFAULT_THRESHOLD,
    duration_limit: float = DEFAULT_DURATION_LIMIT,
    skip_start: float = DEFAULT_SKIP_START,
    max_search_seconds: float = 60.0,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[SceneCutSyncResult]:
    """
    Detect scene cuts in both videos and return the offset that aligns
    their cut sequences. Returns None if either file yields too few cuts
    to correlate reliably.

    Sign convention matches the visual / audio engines:
        offset > 0  -> Input 2 (Target) starts AFTER Input 1 (Primary)
                       (Target needs to be padded at the start)
        offset < 0  -> Input 2 starts BEFORE Input 1
                       (Target needs to be trimmed at the start)
    """

    def log(msg: str, prog: float = -1.0) -> None:
        if progress_callback:
            progress_callback(msg, prog)

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    log("Scene cut: detecting cuts in Primary...", 0.0)
    cuts_p = detect_scene_cuts(primary_filepath, threshold,
                               duration_limit, skip_start)
    if cancelled():
        return None
    log(f"Scene cut: Primary -> {len(cuts_p)} cut(s) detected", 0.30)

    log("Scene cut: detecting cuts in Target...", 0.35)
    cuts_t = detect_scene_cuts(target_filepath, threshold,
                               duration_limit, skip_start)
    if cancelled():
        return None
    log(f"Scene cut: Target  -> {len(cuts_t)} cut(s) detected", 0.65)

    if len(cuts_p) < 5 or len(cuts_t) < 5:
        log(
            f"Scene cut: too few cuts ({len(cuts_p)}/{len(cuts_t)}) - "
            "skipping (try a lower threshold or longer duration).",
            0.95,
        )
        return None

    sig_p = _cuts_to_signal(cuts_p, duration_limit)
    sig_t = _cuts_to_signal(cuts_t, duration_limit)
    if sig_p.size == 0 or sig_t.size == 0:
        return None

    log("Scene cut: cross-correlating cut sequences...", 0.75)
    xcorr = _fft_xcorr(sig_p, sig_t)
    zero_lag_idx = len(sig_t) - 1

    max_lag_bins = int(max_search_seconds * 1000 / BIN_MS)
    lo = max(0, zero_lag_idx - max_lag_bins)
    hi = min(len(xcorr), zero_lag_idx + max_lag_bins + 1)
    search = xcorr[lo:hi]
    peak_local = int(np.argmax(search))
    peak_idx = lo + peak_local
    # Sub-bin refinement via parabolic interpolation.
    peak_idx_refined = _refine_peak_parabolic(xcorr, peak_idx)
    lag_bins = peak_idx_refined - zero_lag_idx

    # Same sign convention as VisualSyncEngine / audio_sync.py:
    #   offset > 0  -> Target's content is EARLIER than Primary's
    #                  (Target needs padding at the start)
    #   offset < 0  -> Target's content is LATER than Primary's
    #                  (Target needs trimming at the start)
    # FFT xcorr peak sits at k = -d when Target is "Primary delayed by d",
    # i.e. lag_bins = -d in that case, which already matches the sign we
    # want (negative -> trim). Use lag_bins directly, no flip.
    offset_seconds = lag_bins * (BIN_MS / 1000.0)

    norm = float(np.sqrt(np.dot(sig_p, sig_p) * np.dot(sig_t, sig_t)) + 1e-12)
    peak_score = float(xcorr[peak_idx]) / norm
    peak_score = max(0.0, min(1.0, peak_score))

    if peak_score >= 0.50:
        confidence = "high"
    elif peak_score >= 0.30:
        confidence = "medium"
    else:
        confidence = "low"

    direction = "after" if offset_seconds >= 0 else "before"
    log(
        f"Scene cut: offset {offset_seconds:+.3f}s "
        f"(Input 2 starts {direction} Input 1) "
        f"peak={peak_score:.3f} [{confidence}] "
        f"({len(cuts_p)} vs {len(cuts_t)} cuts)",
        1.0,
    )

    return SceneCutSyncResult(
        offset_seconds=float(offset_seconds),
        peak_score=peak_score,
        confidence=confidence,
        n_cuts_primary=len(cuts_p),
        n_cuts_target=len(cuts_t),
    )
