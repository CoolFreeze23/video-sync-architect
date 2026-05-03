"""
Audio-based synchronization using Voice Activity Detection (VAD)
cross-correlation. Robust to different audio mixes / dubs because it
correlates the *timing* of speech rather than the waveforms themselves.

Pipeline:
    1. Decode each input to mono PCM at a fixed sample rate via FFmpeg.
    2. Compute a per-frame loudness signal (RMS in 20 ms windows).
    3. Threshold adaptively to produce a binary VAD signal.
    4. FFT-based cross-correlation of the two VAD signals.
    5. Peak position = offset between the videos (seconds).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..utils.ffmpeg_utils import FFMPEG


# --- Tuning constants -------------------------------------------------------

SAMPLE_RATE = 16000          # mono PCM, sufficient for speech timing
FRAME_MS = 20                # 20 ms VAD window (50 frames per second)
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# How much audio (in seconds) to extract from each file. Keeping this
# bounded keeps memory and FFT cost reasonable even for movie-length
# inputs while still containing plenty of speech to correlate.
DEFAULT_DURATION_LIMIT = 600.0  # 10 minutes

# Skip the very start of files when extracting (logos / silence) so the
# VAD threshold isn't dominated by an initial silent gap.
DEFAULT_SKIP_START = 5.0


@dataclass
class AudioSyncResult:
    offset_seconds: float          # Input2 (target) offset relative to Input1 (primary)
    peak_score: float              # Normalized correlation peak [0, 1]
    confidence: str                # "high" / "medium" / "low"
    extracted_duration: float      # Seconds of audio actually used


# --- Audio extraction ------------------------------------------------------

def _extract_pcm(filepath: str,
                 skip_start: float = DEFAULT_SKIP_START,
                 duration: float = DEFAULT_DURATION_LIMIT) -> np.ndarray:
    """
    Decode `duration` seconds of mono PCM (skipping `skip_start` from the
    front) and return a float32 numpy array normalized to [-1, 1].
    """
    cmd = [
        FFMPEG.path, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{skip_start:.3f}",
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
        raise RuntimeError(
            f"Failed to extract audio from '{filepath}': "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


# --- VAD signal -----------------------------------------------------------

def _compute_vad_signal(pcm: np.ndarray) -> np.ndarray:
    """
    Convert raw PCM to a binary VAD signal (1.0 where speech-like
    energy is present, 0.0 otherwise). Returns a float32 1-D array
    sampled at 1/FRAME_MS Hz (50 Hz by default).
    """
    n_frames = len(pcm) // FRAME_SAMPLES
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    framed = pcm[: n_frames * FRAME_SAMPLES].reshape(n_frames, FRAME_SAMPLES)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)

    # Use a log-scaled energy with adaptive threshold halfway between the
    # 30th-percentile (background) and 90th-percentile (speech peaks).
    log_rms = np.log10(rms + 1e-6)
    p30 = np.percentile(log_rms, 30)
    p90 = np.percentile(log_rms, 90)
    threshold = p30 + 0.5 * (p90 - p30)

    vad = (log_rms > threshold).astype(np.float32)

    # Smooth tiny gaps (single-frame holes inside a speech run) and remove
    # one-frame blips so the cross-correlation is dominated by stable
    # speech edges rather than noise spikes.
    if len(vad) >= 3:
        smoothed = vad.copy()
        # Fill 1-frame holes:  010 -> 011 (simple morphological close).
        for i in range(1, len(vad) - 1):
            if vad[i] == 0 and vad[i - 1] == 1 and vad[i + 1] == 1:
                smoothed[i] = 1.0
        # Remove 1-frame spikes:  101 -> 100? No -- keep zeros conservative.
        for i in range(1, len(vad) - 1):
            if smoothed[i] == 1 and smoothed[i - 1] == 0 and smoothed[i + 1] == 0:
                smoothed[i] = 0.0
        vad = smoothed

    # Mean-center so the cross-correlation peak isn't biased by signal DC.
    vad = vad - vad.mean()
    return vad.astype(np.float32)


# --- Cross-correlation ----------------------------------------------------

def _fft_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Full cross-correlation of `a` with `b` (signal a vs reference b),
    computed via FFT. Returns a length (len(a)+len(b)-1) array where
    index `len(b)-1` corresponds to lag 0.
    """
    n = len(a) + len(b) - 1
    n_fft = 1 << (n - 1).bit_length()  # next power of two
    fa = np.fft.rfft(a, n_fft)
    fb = np.fft.rfft(b, n_fft)
    full = np.fft.irfft(fa * np.conj(fb), n_fft)
    # Reorder so that lag 0 sits at len(b)-1 (matches np.correlate 'full').
    out = np.concatenate([full[-(len(b) - 1):], full[: len(a)]])
    return out


def _refine_peak_parabolic(xcorr: np.ndarray, peak_idx: int) -> float:
    """
    Sub-bin peak refinement via parabolic interpolation. Fits a parabola
    through (peak-1, peak, peak+1) and returns the analytical maximum
    location as a *fractional* index. This recovers ~1/10-bin precision
    when the underlying signal is reasonably band-limited, which lets us
    pin the FFT peak to well under one frame even though our sampling is
    20 ms / bin.
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
    # A sane parabola fit lives in (-1, 1); clamp to defend against
    # ill-conditioned (near-flat) peaks.
    if delta < -1.0:
        delta = -1.0
    elif delta > 1.0:
        delta = 1.0
    return float(peak_idx) + delta


def _find_peak_offset(vad_p: np.ndarray, vad_t: np.ndarray,
                      max_lag_seconds: Optional[float] = None
                      ) -> tuple[float, float]:
    """
    Cross-correlate the two VAD signals and return (offset_seconds, peak_score).
    `offset_seconds` is positive when target starts AFTER primary.
    `peak_score` is a normalized confidence in [0, 1].
    """
    if len(vad_p) == 0 or len(vad_t) == 0:
        return 0.0, 0.0

    xcorr = _fft_xcorr(vad_p, vad_t)
    zero_lag_idx = len(vad_t) - 1

    # Optionally restrict the searchable lag range.
    if max_lag_seconds is not None:
        max_lag_frames = int(max_lag_seconds * 1000 / FRAME_MS)
        lo = max(0, zero_lag_idx - max_lag_frames)
        hi = min(len(xcorr), zero_lag_idx + max_lag_frames + 1)
        search = xcorr[lo:hi]
        peak_local = int(np.argmax(search))
        peak_idx = lo + peak_local
    else:
        peak_idx = int(np.argmax(xcorr))

    # Sub-bin refinement: fit a parabola to the 3 points around the
    # integer peak so we can resolve the offset to a fraction of a bin.
    peak_idx_refined = _refine_peak_parabolic(xcorr, peak_idx)
    lag_frames = peak_idx_refined - zero_lag_idx

    # Sign convention must match VisualSyncEngine:
    #     offset = primary_match_time - target_ref_time
    # Concretely: offset > 0  -> Target's content arrives EARLIER than
    #             Primary's, so Target needs PADDING at the start.
    #             offset < 0  -> Target's content arrives LATER than
    #             Primary's, so Target needs TRIMMING at the start.
    #
    # FFT cross-correlation here computes xcorr[k] = sum_t p[t+k] * t[t].
    # If Target's signal is Primary delayed by d (target later by d),
    # the peak sits at k = -d in our coordinate system, i.e. lag_frames
    # is negative. Per the convention above, that case must yield a
    # NEGATIVE offset (trim Target). So we use lag_frames *directly*
    # without flipping the sign.
    frame_seconds = FRAME_MS / 1000.0
    offset_seconds = lag_frames * frame_seconds

    # Normalize peak by the autocorrelation energies.
    norm = float(np.sqrt(np.dot(vad_p, vad_p) * np.dot(vad_t, vad_t)) + 1e-12)
    peak_score = float(xcorr[peak_idx]) / norm
    peak_score = max(0.0, min(1.0, peak_score))

    return offset_seconds, peak_score


# --- Public API -----------------------------------------------------------

def find_audio_offset(
    primary_filepath: str,
    target_filepath: str,
    max_search_seconds: float = 60.0,
    duration_limit: float = DEFAULT_DURATION_LIMIT,
    skip_start: float = DEFAULT_SKIP_START,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[AudioSyncResult]:
    """
    Compute the time offset between Primary and Target via VAD
    cross-correlation. Returns None if extraction or correlation fails.

    `max_search_seconds` bounds the lag space so spurious peaks far from
    plausible offsets don't win.
    """

    def log(msg: str, progress: float = -1.0) -> None:
        if progress_callback:
            progress_callback(msg, progress)

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    log("Audio VAD: extracting Primary audio...", 0.0)
    pcm_p = _extract_pcm(primary_filepath, skip_start, duration_limit)
    if cancelled():
        return None

    log("Audio VAD: extracting Target audio...", 0.25)
    pcm_t = _extract_pcm(target_filepath, skip_start, duration_limit)
    if cancelled():
        return None

    actual_duration_p = len(pcm_p) / SAMPLE_RATE
    actual_duration_t = len(pcm_t) / SAMPLE_RATE
    log(
        f"Audio VAD: {actual_duration_p:.1f}s (Primary), "
        f"{actual_duration_t:.1f}s (Target)",
        0.50,
    )

    log("Audio VAD: computing VAD signals...", 0.55)
    vad_p = _compute_vad_signal(pcm_p)
    vad_t = _compute_vad_signal(pcm_t)
    if len(vad_p) == 0 or len(vad_t) == 0:
        log("Audio VAD: empty VAD signal - skipping.", 0.95)
        return None

    speech_p = float(np.mean(vad_p > 0))
    speech_t = float(np.mean(vad_t > 0))
    log(
        f"Audio VAD: speech ratio Primary={speech_p:.0%}, Target={speech_t:.0%}",
        0.65,
    )
    if speech_p < 0.05 or speech_t < 0.05:
        log("Audio VAD: very little speech detected - result may be unreliable.", 0.70)

    if cancelled():
        return None

    log("Audio VAD: cross-correlating...", 0.75)
    offset, peak_score = _find_peak_offset(vad_p, vad_t,
                                           max_lag_seconds=max_search_seconds)

    if peak_score >= 0.55:
        confidence = "high"
    elif peak_score >= 0.35:
        confidence = "medium"
    else:
        confidence = "low"

    direction = "after" if offset >= 0 else "before"
    log(
        f"Audio VAD: offset {offset:+.3f}s "
        f"(Input 2 starts {direction} Input 1) "
        f"peak={peak_score:.3f} [{confidence}]",
        1.0,
    )

    return AudioSyncResult(
        offset_seconds=float(offset),
        peak_score=peak_score,
        confidence=confidence,
        extracted_duration=min(actual_duration_p, actual_duration_t),
    )


# --- Per-segment offset refinement ----------------------------------------

def refine_offset_for_segment(
    primary_filepath: str,
    target_filepath: str,
    primary_start: float,
    primary_end: float,
    coarse_offset: float,
    max_residual_seconds: float = 2.0,
    min_segment_seconds: float = 30.0,
) -> Optional[tuple[float, float]]:
    """
    Audio-cross-correlate a SINGLE segment to compute its true offset.

    The trick: if `coarse_offset` is approximately right, the matching
    target window is `[primary_start - coarse_offset, primary_end -
    coarse_offset]` (recall offset = primary_t - target_t, so target_t =
    primary_t - offset; with offset = -10.5 this puts target at primary
    + 10.5). Extract that exact pair of windows from both files and run
    the same VAD cross-correlation we use for global sync. Whatever
    residual lag the cross-correlation finds is the per-segment
    correction to add to `coarse_offset`.

    This is dramatically more accurate than per-segment pHash averaging
    on visual-similarity-prone content (anime, sitcom talking heads,
    long static shots) because audio false-matches require two distinct
    speech bursts to coincidentally share envelopes - which essentially
    never happens in practice. Visual pHash false-matches at HD<=8 are
    common when frames repeat (parallax pans, static backgrounds).

    Returns (refined_offset_seconds, peak_score) or None if the segment
    is too short, has too little speech, or cross-correlation fails.
    """
    seg_dur = primary_end - primary_start
    if seg_dur < min_segment_seconds:
        return None

    p_start = max(0.0, primary_start)
    p_dur = max(0.1, primary_end - p_start)

    # Target window is the primary window shifted by -coarse_offset
    # (because target_t = primary_t - coarse_offset for our sign
    # convention where coarse_offset is negative when target trails).
    t_start = p_start - coarse_offset
    t_dur = p_dur
    if t_start < 0:
        # Move both windows forward together so they stay aligned.
        shift = -t_start
        t_start = 0.0
        p_start += shift
        p_dur -= shift
        t_dur -= shift
    if p_dur < min_segment_seconds:
        return None

    try:
        pcm_p = _extract_pcm(primary_filepath, p_start, p_dur)
        pcm_t = _extract_pcm(target_filepath, t_start, t_dur)
    except Exception:
        return None

    vad_p = _compute_vad_signal(pcm_p)
    vad_t = _compute_vad_signal(pcm_t)
    if len(vad_p) < 50 or len(vad_t) < 50:
        return None
    if float(np.mean(vad_p > 0)) < 0.05 or float(np.mean(vad_t > 0)) < 0.05:
        # Mostly silent segment - cross-correlation will be noise.
        return None

    # We extracted matched windows; any residual lag IS the correction.
    residual, peak = _find_peak_offset(
        vad_p, vad_t, max_lag_seconds=max_residual_seconds,
    )
    if peak < 0.20:
        return None

    return coarse_offset + residual, peak
