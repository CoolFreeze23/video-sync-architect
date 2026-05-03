"""
Main application window. Subsync-style dual-list UI driving the existing
sync backend. The processing layer (sync_engine, exporter, hashing,
ffmpeg_utils, media_info, batch_processor) is untouched.
"""

import os
import traceback
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QCheckBox, QMessageBox, QLabel, QComboBox, QSpinBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont

from .widgets import ConsoleLog, StyledProgressBar, SensitivityControl
from .dual_list import DualFileListWidget
from ..core.media_info import MediaInfo
from ..core.sync_engine import VisualSyncEngine, SyncResult
from ..core.exporter import export_synced_file, export_difference_preview
from ..core.audio_sync import (
    find_audio_offset,
    refine_offset_for_segment,
    AudioSyncResult,
)
from ..core.scene_cut_sync import find_scene_cut_offset, SceneCutSyncResult
from ..core.sync_verify import verify_sync, write_report, verification_issues
from ..core.segmented_sync import (
    build_segments_from_samples,
    build_segments_from_samples_and_blacks,
    detect_primary_black_sections,
    build_playback_chunks,
    render_segmented_export,
    offset_variance,
    offset_variance_robust,
    despike_isolated_dense_samples,
    VARIANCE_TRIGGER_S,
)


# ---------------------------------------------------------------------------
# Worker: process an explicit list of (primary, target) path pairs.
# Reuses the existing backend (no changes to core/ or utils/).
# ---------------------------------------------------------------------------

class PairListSyncWorker(QThread):
    log_signal = pyqtSignal(str)
    scan_progress = pyqtSignal(float)
    export_progress = pyqtSignal(float)
    finished_signal = pyqtSignal(int, int)  # (succeeded, failed)
    error_signal = pyqtSignal(str)

    # multi_ref_mode: "off", "retry", "always"
    # audio_mode / scene_cut_mode: "off", "verify", "fallback"
    #   verify   = run after visual, log agreement / disagreement
    #   fallback = same as verify, AND take over if visual fails
    # constant_offset_export_only: never use segmented/black-filler export;
    #   always one global offset (median from dense samples when available).
    def __init__(self, pairs: list[tuple[str, str]],
                 hamming_threshold: int, generate_preview: bool,
                 multi_ref_mode: str = "off", num_refs: int = 5,
                 audio_mode: str = "off", scene_cut_mode: str = "off",
                 debug_verify: bool = False,
                 constant_offset_export_only: bool = False):
        super().__init__()
        self.pairs = pairs
        self.hamming_threshold = hamming_threshold
        self.generate_preview = generate_preview
        self.multi_ref_mode = multi_ref_mode
        self.num_refs = num_refs
        self.audio_mode = audio_mode
        self.scene_cut_mode = scene_cut_mode
        self.debug_verify = debug_verify
        self.constant_offset_export_only = constant_offset_export_only
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        succeeded = 0
        failed = 0
        engine = VisualSyncEngine()

        try:
            total = len(self.pairs)
            for idx, (primary_path, target_path) in enumerate(self.pairs):
                if self._cancelled:
                    self.log_signal.emit("Cancelled by user.")
                    break

                self.log_signal.emit(
                    f"\n=== Pair {idx + 1}/{total} ===\n"
                    f"  Primary: {os.path.basename(primary_path)}\n"
                    f"  Target:  {os.path.basename(target_path)}"
                )
                self.scan_progress.emit(0.0)
                self.export_progress.emit(idx / total if total else 0.0)

                try:
                    primary_info = MediaInfo.from_file(primary_path)
                    target_info = MediaInfo.from_file(target_path)

                    self.log_signal.emit(f"Primary: {primary_info.summary()}")
                    self.log_signal.emit(f"Target:  {target_info.summary()}")

                    def scan_cb(msg, progress):
                        if msg:
                            self.log_signal.emit(msg)
                        if progress >= 0:
                            self.scan_progress.emit(progress)

                    if self.multi_ref_mode == "always":
                        self.log_signal.emit(
                            f"Multi-reference mode (always, {self.num_refs} refs)"
                        )
                        result = engine.find_offset_multi_ref(
                            primary_info, target_info,
                            hamming_threshold=self.hamming_threshold,
                            num_refs=self.num_refs,
                            progress_callback=scan_cb,
                            cancel_check=lambda: self._cancelled,
                        )
                    else:
                        result = engine.find_offset(
                            primary_info, target_info,
                            hamming_threshold=self.hamming_threshold,
                            progress_callback=scan_cb,
                            cancel_check=lambda: self._cancelled,
                        )

                        if (result is None
                                and self.multi_ref_mode == "retry"
                                and not self._cancelled):
                            self.log_signal.emit(
                                f"Single-ref failed -> retrying with multi-reference "
                                f"({self.num_refs} refs)..."
                            )
                            self.scan_progress.emit(0.0)
                            result = engine.find_offset_multi_ref(
                                primary_info, target_info,
                                hamming_threshold=self.hamming_threshold,
                                num_refs=self.num_refs,
                                progress_callback=scan_cb,
                                cancel_check=lambda: self._cancelled,
                            )

                    if self._cancelled:
                        break

                    # --- Audio VAD verification / fallback ---
                    audio_res: Optional[AudioSyncResult] = None

                    if result is not None and self.audio_mode in ("verify", "fallback"):
                        self.log_signal.emit(
                            "Audio verification: cross-correlating VAD signals..."
                        )
                        try:
                            audio_res = find_audio_offset(
                                primary_path, target_path,
                                progress_callback=scan_cb,
                                cancel_check=lambda: self._cancelled,
                            )
                        except Exception as e:
                            self.log_signal.emit(
                                f"WARNING: Audio verification failed: {e}"
                            )
                            audio_res = None

                        if audio_res is not None:
                            visual_off = result.offset_seconds
                            audio_off = audio_res.offset_seconds
                            delta = abs(visual_off - audio_off)
                            self.log_signal.emit(
                                f"Audio VAD says {audio_off:+.3f}s "
                                f"(peak={audio_res.peak_score:.3f}, "
                                f"{audio_res.confidence}); visual says "
                                f"{visual_off:+.3f}s; delta={delta * 1000:.0f} ms"
                            )
                            if delta <= 0.1:
                                self.log_signal.emit(
                                    "Audio VAD confirms visual offset (≤100 ms)."
                                )
                            elif delta <= 0.5:
                                self.log_signal.emit(
                                    f"WARNING: Audio VAD differs from visual by "
                                    f"{delta * 1000:.0f} ms - within 1 frame, acceptable."
                                )
                            else:
                                self.log_signal.emit(
                                    f"WARNING: Audio VAD disagrees with visual by "
                                    f"{delta * 1000:.0f} ms - manual review recommended."
                                )

                    # --- Scene-Cut verification (after visual succeeded) ---
                    cut_res: Optional[SceneCutSyncResult] = None

                    if result is not None and self.scene_cut_mode in ("verify", "fallback"):
                        self.log_signal.emit(
                            "Scene-cut verification: detecting cuts and "
                            "cross-correlating..."
                        )
                        try:
                            cut_res = find_scene_cut_offset(
                                primary_path, target_path,
                                progress_callback=scan_cb,
                                cancel_check=lambda: self._cancelled,
                            )
                        except Exception as e:
                            self.log_signal.emit(
                                f"WARNING: Scene-cut verification failed: {e}"
                            )
                            cut_res = None

                        if cut_res is not None:
                            visual_off = result.offset_seconds
                            cut_off = cut_res.offset_seconds
                            delta = abs(visual_off - cut_off)
                            self.log_signal.emit(
                                f"Scene-cut says {cut_off:+.3f}s "
                                f"(peak={cut_res.peak_score:.3f}, "
                                f"{cut_res.confidence}, "
                                f"{cut_res.n_cuts_primary} vs "
                                f"{cut_res.n_cuts_target} cuts); "
                                f"visual says {visual_off:+.3f}s; "
                                f"delta={delta * 1000:.0f} ms"
                            )
                            if delta <= 0.1:
                                self.log_signal.emit(
                                    "Scene-cut confirms visual offset (≤100 ms)."
                                )
                            elif delta <= 0.5:
                                self.log_signal.emit(
                                    f"WARNING: Scene-cut differs from visual by "
                                    f"{delta * 1000:.0f} ms - within ~1 frame, "
                                    f"borderline."
                                )
                            else:
                                self.log_signal.emit(
                                    f"WARNING: Scene-cut DISAGREES with visual by "
                                    f"{delta * 1000:.0f} ms - manual review "
                                    f"recommended."
                                )

                    # --- Scene-Cut fallback when visual fails entirely ---
                    if (result is None
                            and self.scene_cut_mode == "fallback"
                            and not self._cancelled):
                        self.log_signal.emit(
                            "Visual sync failed -> falling back to Scene-Cut sync..."
                        )
                        self.scan_progress.emit(0.0)
                        try:
                            cut_res = find_scene_cut_offset(
                                primary_path, target_path,
                                progress_callback=scan_cb,
                                cancel_check=lambda: self._cancelled,
                            )
                        except Exception as e:
                            self.log_signal.emit(
                                f"ERROR: Scene-cut fallback failed: {e}"
                            )
                            cut_res = None

                        if cut_res is not None and cut_res.peak_score >= 0.30:
                            self.log_signal.emit(
                                f"Scene-cut match: offset={cut_res.offset_seconds:+.3f}s "
                                f"peak={cut_res.peak_score:.3f} "
                                f"[{cut_res.confidence}]"
                            )
                            result = SyncResult(
                                offset_seconds=cut_res.offset_seconds,
                                matched_frame_input1=0,
                                matched_time_input1=0.0,
                                reference_frame_input2=0,
                                reference_time_input2=0.0,
                                hamming_distance=-2,  # sentinel: came from scene-cut
                                confidence=cut_res.confidence,
                            )
                        elif cut_res is not None:
                            self.log_signal.emit(
                                f"WARNING: Scene-cut fallback peak too low "
                                f"({cut_res.peak_score:.3f}) - skipping pair."
                            )

                    # --- Consensus override: audio + scene-cut both agree
                    # with each other but disagree with visual -> trust quorum.
                    if (result is not None
                            and audio_res is not None
                            and cut_res is not None):
                        a_off = audio_res.offset_seconds
                        c_off = cut_res.offset_seconds
                        v_off = result.offset_seconds
                        ac_delta = abs(a_off - c_off)
                        av_delta = abs(a_off - v_off)
                        cv_delta = abs(c_off - v_off)
                        if (ac_delta <= 0.15
                                and av_delta > 0.5
                                and cv_delta > 0.5):
                            # Weight by each engine's normalized peak
                            # score: a stronger correlation should pull
                            # the consensus toward its own estimate.
                            # Falls back to a plain mean if both peaks
                            # are unusable.
                            a_w = max(0.0, float(audio_res.peak_score))
                            c_w = max(0.0, float(cut_res.peak_score))
                            total_w = a_w + c_w
                            if total_w > 1e-6:
                                consensus = (a_w * a_off + c_w * c_off) / total_w
                                weight_str = (
                                    f" (weights audio={a_w:.2f}, cut={c_w:.2f})"
                                )
                            else:
                                consensus = (a_off + c_off) / 2.0
                                weight_str = ""
                            self.log_signal.emit(
                                f"CONSENSUS OVERRIDE: Audio ({a_off:+.3f}s) and "
                                f"Scene-Cut ({c_off:+.3f}s) agree to within "
                                f"{ac_delta * 1000:.0f} ms but disagree with visual "
                                f"({v_off:+.3f}s) by {av_delta * 1000:.0f} ms / "
                                f"{cv_delta * 1000:.0f} ms - using audio/scene-cut "
                                f"consensus {consensus:+.3f}s{weight_str}."
                            )
                            result = SyncResult(
                                offset_seconds=consensus,
                                matched_frame_input1=0,
                                matched_time_input1=0.0,
                                reference_frame_input2=0,
                                reference_time_input2=0.0,
                                hamming_distance=-3,  # sentinel: consensus override
                                confidence="high",
                            )

                    # --- Audio fallback when visual fails entirely ---
                    if (result is None
                            and self.audio_mode == "fallback"
                            and not self._cancelled):
                        self.log_signal.emit(
                            "Visual sync failed -> falling back to Audio VAD..."
                        )
                        self.scan_progress.emit(0.0)
                        try:
                            audio_res = find_audio_offset(
                                primary_path, target_path,
                                progress_callback=scan_cb,
                                cancel_check=lambda: self._cancelled,
                            )
                        except Exception as e:
                            self.log_signal.emit(
                                f"ERROR: Audio fallback failed: {e}"
                            )
                            audio_res = None

                        if audio_res is not None and audio_res.peak_score >= 0.35:
                            self.log_signal.emit(
                                f"Audio VAD match: offset={audio_res.offset_seconds:+.3f}s "
                                f"peak={audio_res.peak_score:.3f} "
                                f"[{audio_res.confidence}]"
                            )
                            # Synthesize a SyncResult so downstream export stays happy.
                            result = SyncResult(
                                offset_seconds=audio_res.offset_seconds,
                                matched_frame_input1=0,
                                matched_time_input1=0.0,
                                reference_frame_input2=0,
                                reference_time_input2=0.0,
                                hamming_distance=-1,  # sentinel: came from audio
                                confidence=audio_res.confidence,
                            )
                        elif audio_res is not None:
                            self.log_signal.emit(
                                f"WARNING: Audio fallback peak too low "
                                f"({audio_res.peak_score:.3f}) - skipping pair."
                            )

                    if self._cancelled:
                        break

                    if result is None:
                        self.log_signal.emit(
                            "WARNING: No reliable match found for this pair. Skipping."
                        )
                        failed += 1
                        continue

                    # --- Dense visual sampling: feeds either single-offset
                    #     refinement OR full segmented sync, depending on
                    #     how much the offset varies across the runtime.
                    samples: list[tuple[float, float, int]] = []
                    use_segmented = False

                    self.log_signal.emit(
                        "Dense offset sampling (48 points across Target)..."
                    )
                    self.scan_progress.emit(0.0)
                    try:
                        samples = engine.sample_offsets_visually(
                            primary_info, target_info,
                            coarse_offset=result.offset_seconds,
                            window_seconds=2.0,
                            num_samples=48,
                            progress_callback=scan_cb,
                            cancel_check=lambda: self._cancelled,
                        )
                    except Exception as e:
                        self.log_signal.emit(
                            f"WARNING: Dense sampling failed: {e} - "
                            "falling back to coarse offset."
                        )
                        samples = []

                    if samples:
                        var_raw_before = offset_variance(samples)
                        if len(samples) >= 8 and var_raw_before > 0.45:
                            samples, n_despike = despike_isolated_dense_samples(
                                samples,
                            )
                            if n_despike:
                                self.log_signal.emit(
                                    f"Dense samples: despiked {n_despike} isolated "
                                    f"pHash glitch(es) (neighbours agree; one stray jump)."
                                )
                        var_raw = offset_variance(samples)
                        var_rob = (
                            offset_variance_robust(samples)
                            if len(samples) >= 4 else var_raw
                        )
                        self.log_signal.emit(
                            f"Dense sampling: {len(samples)} usable samples, "
                            f"variance raw {var_raw * 1000:.0f} ms, "
                            f"robust {var_rob * 1000:.0f} ms "
                            f"(segmented threshold {VARIANCE_TRIGGER_S * 1000:.0f} ms)."
                        )
                        use_segmented = var_rob > VARIANCE_TRIGGER_S
                        if use_segmented:
                            self.log_signal.emit(
                                "Variable offset detected -> would use SEGMENTED "
                                "sync (black boundaries + piecewise encode)."
                            )
                        if use_segmented and self.constant_offset_export_only:
                            self.log_signal.emit(
                                "Constant-offset export only: skipping segmented "
                                "mode — one global trim/pad from median dense offset "
                                "(no black filler segments, no piecewise re-encode)."
                            )
                            use_segmented = False

                        if not use_segmented:
                            refined = engine.aggregate_offset_samples(samples)
                            if refined is not None:
                                delta_ms = (refined - result.offset_seconds) * 1000
                                self.log_signal.emit(
                                    f"Refined single offset: {refined:+.4f}s "
                                    f"(adjusted coarse by {delta_ms:+.1f} ms)"
                                )
                                result = SyncResult(
                                    offset_seconds=refined,
                                    matched_frame_input1=result.matched_frame_input1,
                                    matched_time_input1=result.matched_time_input1,
                                    reference_frame_input2=result.reference_frame_input2,
                                    reference_time_input2=result.reference_time_input2,
                                    hamming_distance=result.hamming_distance,
                                    confidence=result.confidence,
                                )

                    self.scan_progress.emit(1.0)
                    self.log_signal.emit("Starting export...")

                    overall_base = idx / total if total else 0.0
                    overall_step = 1.0 / total if total else 1.0

                    def export_cb(frac):
                        self.export_progress.emit(overall_base + overall_step * frac)

                    if use_segmented:
                        # --- Segmented render: build piecewise warp from samples ---
                        target_duration = (target_info.total_frames / target_info.fps
                                           if target_info.fps > 0 else target_info.duration)
                        primary_duration = primary_info.duration

                        # First locate eyecatches/scene transitions in
                        # PRIMARY via blackdetect - those are the only
                        # places the offset actually jumps. Then build
                        # segments using those real boundaries; falls
                        # back to sample-only segmentation if blackdetect
                        # finds nothing usable.
                        self.log_signal.emit(
                            "Scanning Primary for black/eyecatch sections..."
                        )
                        try:
                            black_sections = detect_primary_black_sections(
                                primary_info.filepath,
                                min_duration=0.30,
                            )
                        except Exception as e:
                            self.log_signal.emit(
                                f"WARNING: blackdetect failed ({e}); "
                                "using sample-only segmentation."
                            )
                            black_sections = []

                        if black_sections:
                            self.log_signal.emit(
                                f"Found {len(black_sections)} candidate "
                                "boundary section(s) in Primary:"
                            )
                            for (b_start, b_end) in black_sections:
                                self.log_signal.emit(
                                    f"  Primary [{b_start:.2f}, "
                                    f"{b_end:.2f}]s "
                                    f"(dur {b_end - b_start:.2f}s)"
                                )
                            segments = build_segments_from_samples_and_blacks(
                                samples,
                                target_duration=target_duration,
                                primary_black_sections=black_sections,
                            )
                        else:
                            self.log_signal.emit(
                                "No black sections detected; falling "
                                "back to sample-only segmentation."
                            )
                            segments = build_segments_from_samples(
                                samples, target_duration=target_duration,
                            )

                        self.log_signal.emit(
                            f"Built {len(segments)} segment(s):"
                        )
                        for si, seg in enumerate(segments):
                            self.log_signal.emit(
                                f"  Seg {si + 1}: target [{seg.target_start:.2f}, "
                                f"{seg.target_end:.2f}]s (dur "
                                f"{seg.target_end - seg.target_start:.2f}s) "
                                f"offset {seg.offset:+.4f}s"
                            )

                        # --- Per-segment offset adjudication ---------------
                        # Run audio cross-correlation for EVERY segment
                        # (even ones where pHash looks tight) so we can
                        # detect "joint pHash false-match" - when multiple
                        # pHash samples in a flashback / repetitive scene
                        # all match the same wrong frame and look like
                        # tight agreement but are actually 0.5-1s wrong.
                        #
                        # Decision matrix:
                        #
                        #   pHash trusted (>=3 inliers, stdev<=80ms)
                        #   AND audio agrees within 300 ms        -> pHash
                        #
                        #   pHash trusted but audio strongly DISAGREES
                        #   (|delta|>300ms) AND audio peak >=0.30 -> AUDIO
                        #     (3 jointly-wrong samples scenario)
                        #
                        #   pHash undertrusted (<3 inliers OR loose)
                        #   AND audio peak >= 0.30                 -> AUDIO
                        #
                        #   pHash undertrusted AND audio weak       -> pHash
                        #     (last-resort fallback)
                        #
                        #   Audio refinement disabled / failed      -> pHash
                        from dataclasses import replace as _replace
                        PHASH_TRUST_MIN_INLIERS = 3
                        PHASH_TRUST_MAX_STDEV = 0.080
                        MIN_AUDIO_PEAK = 0.30
                        MAX_AUDIO_DELTA = 1.500
                        DISAGREEMENT_THRESHOLD = 0.300
                        self.log_signal.emit(
                            "Adjudicating per-segment offset (pHash vs audio)..."
                        )
                        refined_segments = []
                        for si, seg in enumerate(segments):
                            phash_trusted = (
                                seg.phash_inliers >= PHASH_TRUST_MIN_INLIERS
                                and seg.phash_stdev <= PHASH_TRUST_MAX_STDEV
                            )
                            seg_p_start = max(0.0, seg.target_start + seg.offset)
                            seg_p_end = min(primary_duration,
                                            seg.target_end + seg.offset)
                            try:
                                res = refine_offset_for_segment(
                                    primary_info.filepath,
                                    target_info.filepath,
                                    primary_start=seg_p_start,
                                    primary_end=seg_p_end,
                                    coarse_offset=seg.offset,
                                    max_residual_seconds=1.5,
                                    min_segment_seconds=30.0,
                                )
                            except Exception as e:
                                self.log_signal.emit(
                                    f"  Seg {si + 1}: audio refine failed "
                                    f"({e}); keeping pHash {seg.offset:+.4f}s."
                                )
                                refined_segments.append(seg)
                                continue

                            phash_summary = (
                                f"{seg.phash_inliers} inliers, "
                                f"stdev {seg.phash_stdev * 1000:.0f} ms"
                            )

                            if res is None:
                                # Audio unusable (silent/short) - pHash is
                                # the only signal we have.
                                self.log_signal.emit(
                                    f"  Seg {si + 1}: audio unusable "
                                    f"(short/quiet); pHash ({phash_summary}) "
                                    f"keep {seg.offset:+.4f}s."
                                )
                                refined_segments.append(seg)
                                continue

                            new_off, peak = res
                            delta = new_off - seg.offset
                            delta_ms = delta * 1000
                            audio_passes_quality = (
                                peak >= MIN_AUDIO_PEAK
                                and abs(delta) <= MAX_AUDIO_DELTA
                            )

                            if phash_trusted and abs(delta) <= DISAGREEMENT_THRESHOLD:
                                # pHash trusted AND audio agrees: pHash
                                # is sub-frame precise on tight clusters,
                                # so prefer it over audio's ~20ms-binned
                                # cross-correlation.
                                self.log_signal.emit(
                                    f"  Seg {si + 1}: pHash trusted "
                                    f"({phash_summary}); audio AGREES "
                                    f"(delta {delta_ms:+.0f} ms, peak "
                                    f"{peak:.2f}); keep pHash {seg.offset:+.4f}s."
                                )
                                refined_segments.append(seg)
                                continue

                            if phash_trusted and audio_passes_quality:
                                # pHash looks tight but audio strongly
                                # disagrees - this is the joint-false-
                                # match flag. Trust audio.
                                self.log_signal.emit(
                                    f"  Seg {si + 1}: pHash 'trusted' "
                                    f"({phash_summary}) but audio DISAGREES "
                                    f"by {delta_ms:+.0f} ms (peak {peak:.2f}); "
                                    f"likely joint false-match -> use audio "
                                    f"{seg.offset:+.4f}s -> {new_off:+.4f}s."
                                )
                                refined_segments.append(_replace(seg, offset=new_off))
                                continue

                            if phash_trusted:
                                # Audio disagrees but its peak is too low
                                # to overrule pHash. Keep pHash.
                                self.log_signal.emit(
                                    f"  Seg {si + 1}: pHash trusted "
                                    f"({phash_summary}); audio disagrees "
                                    f"(delta {delta_ms:+.0f} ms) but peak "
                                    f"{peak:.2f} too low to overrule; "
                                    f"keep pHash {seg.offset:+.4f}s."
                                )
                                refined_segments.append(seg)
                                continue

                            # pHash undertrusted from here on.
                            if audio_passes_quality:
                                self.log_signal.emit(
                                    f"  Seg {si + 1}: pHash undertrusted "
                                    f"({phash_summary}); audio "
                                    f"{seg.offset:+.4f}s -> {new_off:+.4f}s "
                                    f"(delta {delta_ms:+.0f} ms, peak {peak:.2f})."
                                )
                                refined_segments.append(_replace(seg, offset=new_off))
                            else:
                                self.log_signal.emit(
                                    f"  Seg {si + 1}: pHash undertrusted "
                                    f"({phash_summary}); audio peak {peak:.2f} "
                                    f"delta {delta_ms:+.0f} ms REJECTED; "
                                    f"keeping pHash {seg.offset:+.4f}s."
                                )
                                refined_segments.append(seg)
                        segments = refined_segments

                        chunks = build_playback_chunks(
                            segments,
                            target_duration=target_duration,
                            primary_duration=primary_duration,
                        )

                        base, ext = os.path.splitext(target_info.filepath)
                        output_path = base + "_synced.mp4"

                        try:
                            render_segmented_export(
                                target_info, primary_info,
                                chunks=chunks,
                                output_path=output_path,
                                log_callback=lambda m: self.log_signal.emit(m),
                                progress_callback=export_cb,
                            )
                        except Exception as e:
                            self.log_signal.emit(
                                f"ERROR: Segmented export failed: {e}"
                            )
                            failed += 1
                            continue
                    else:
                        output_path = export_synced_file(
                            target_info, primary_info, result,
                            progress_callback=export_cb,
                            log_callback=lambda m: self.log_signal.emit(m),
                        )

                    if self.generate_preview and not self._cancelled:
                        self.log_signal.emit("Generating difference preview...")
                        export_difference_preview(
                            primary_info, output_path,
                            log_callback=lambda m: self.log_signal.emit(m),
                        )

                    # --- Debug verification (post-export) -----------------
                    if self.debug_verify and not self._cancelled:
                        self.log_signal.emit(
                            "DEBUG VERIFY: comparing synced output to "
                            "Primary (visual + audio + scene-cuts)..."
                        )
                        try:
                            synced_info = MediaInfo.from_file(output_path)

                            def verify_cb(msg: str, _progress: float = -1.0):
                                if msg:
                                    self.log_signal.emit(msg)

                            report = verify_sync(
                                primary_info, synced_info,
                                progress_callback=verify_cb,
                                cancel_check=lambda: self._cancelled,
                            )
                            report_path = write_report(report)
                            self.log_signal.emit(
                                f"DEBUG VERIFY: report saved to {report_path}"
                            )
                            issues = verification_issues(report)
                            if issues:
                                self.log_signal.emit(
                                    "DEBUG VERIFY: ISSUES -> "
                                    + "; ".join(issues)
                                )
                            else:
                                self.log_signal.emit(
                                    "DEBUG VERIFY: CLEAN - visual/scene "
                                    "within half a frame; audio within one frame "
                                    "(VAD limit)."
                                )
                        except Exception as e:
                            self.log_signal.emit(
                                f"WARNING: Debug verification failed: {e}"
                            )

                    self.export_progress.emit(overall_base + overall_step)
                    if use_segmented:
                        self.log_signal.emit(
                            f"=== Pair complete (segmented, "
                            f"{len(segments)} segments): coarse offset "
                            f"{result.offset_seconds:+.3f}s ==="
                        )
                    elif result.hamming_distance == -1:
                        self.log_signal.emit(
                            f"=== Pair complete (audio): offset {result.offset_seconds:+.3f}s, "
                            f"confidence={result.confidence} ==="
                        )
                    elif result.hamming_distance == -2:
                        self.log_signal.emit(
                            f"=== Pair complete (scene-cut): offset {result.offset_seconds:+.3f}s, "
                            f"confidence={result.confidence} ==="
                        )
                    elif result.hamming_distance == -3:
                        self.log_signal.emit(
                            f"=== Pair complete (consensus): offset {result.offset_seconds:+.3f}s, "
                            f"confidence={result.confidence} ==="
                        )
                    else:
                        self.log_signal.emit(
                            f"=== Pair complete: offset {result.offset_seconds:+.3f}s, "
                            f"HD={result.hamming_distance}, confidence={result.confidence} ==="
                        )
                    succeeded += 1

                except Exception as e:
                    self.log_signal.emit(f"ERROR on pair {idx + 1}: {e}")
                    failed += 1

            self.export_progress.emit(1.0)
            self.log_signal.emit(
                f"\n=== Batch complete: {succeeded} succeeded, {failed} failed ==="
            )
            self.finished_signal.emit(succeeded, failed)

        except Exception as e:
            self.error_signal.emit(f"{e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Window styling
# ---------------------------------------------------------------------------

WINDOW_STYLE = """
QMainWindow, QWidget {
    background: #1e1e2e;
    color: #e0e0e0;
}
QGroupBox {
    border: 1px solid #444;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #c0c0c0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 6px;
}
QCheckBox {
    color: #e0e0e0;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
}
QScrollBar:vertical {
    background: #2b2b3b;
    width: 12px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #555;
    border-radius: 6px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #777;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Sync Architect")
        self.setMinimumSize(960, 760)
        self.resize(1100, 860)
        self.setStyleSheet(WINDOW_STYLE)

        self._worker: Optional[QThread] = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 12, 16, 12)

        # --- Header ---
        header = QLabel("Video Sync Architect")
        header.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        header.setStyleSheet("color: #3a9bd5; padding: 2px 0;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        subtitle = QLabel(
            "Drag files into a list, reorder by dragging within a list, "
            "or use Auto-Sort to align by name."
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888; font-size: 11px; margin-bottom: 4px;")
        main_layout.addWidget(subtitle)

        # --- Dual list panel ---
        list_group = QGroupBox("File Pairs (paired by row)")
        list_layout = QVBoxLayout(list_group)
        self._dual_list = DualFileListWidget()
        list_layout.addWidget(self._dual_list)
        main_layout.addWidget(list_group, stretch=1)

        # --- Settings ---
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)

        self._sensitivity = SensitivityControl()
        settings_layout.addWidget(self._sensitivity)

        # --- Multi-reference matching ---
        multi_row = QHBoxLayout()
        multi_lbl = QLabel("Multi-Reference:")
        multi_lbl.setFixedWidth(130)
        multi_lbl.setStyleSheet("font-weight: bold; color: #e0e0e0;")
        multi_row.addWidget(multi_lbl)

        self._multi_ref_combo = QComboBox()
        self._multi_ref_combo.addItem("Off (single reference frame)", "off")
        self._multi_ref_combo.addItem("Auto-retry on failure (recommended)", "retry")
        self._multi_ref_combo.addItem("Always use (slower, more reliable)", "always")
        self._multi_ref_combo.setCurrentIndex(1)
        self._multi_ref_combo.setStyleSheet(
            "QComboBox { background: #2b2b2b; color: #e0e0e0; border: 1px solid #555; "
            "border-radius: 4px; padding: 4px 8px; }"
            "QComboBox QAbstractItemView { background: #2b2b2b; color: #e0e0e0; "
            "selection-background-color: #3a7bd5; }"
        )
        multi_row.addWidget(self._multi_ref_combo, stretch=1)

        ref_count_lbl = QLabel("# refs:")
        ref_count_lbl.setStyleSheet("color: #c0c0c0;")
        multi_row.addWidget(ref_count_lbl)

        self._num_refs_spin = QSpinBox()
        self._num_refs_spin.setRange(2, 12)
        self._num_refs_spin.setValue(5)
        self._num_refs_spin.setFixedWidth(60)
        self._num_refs_spin.setStyleSheet(
            "QSpinBox { background: #2b2b2b; color: #e0e0e0; border: 1px solid #555; "
            "border-radius: 4px; padding: 4px; }"
        )
        multi_row.addWidget(self._num_refs_spin)

        settings_layout.addLayout(multi_row)

        # --- Audio VAD verification ---
        audio_row = QHBoxLayout()
        audio_lbl = QLabel("Audio Verify:")
        audio_lbl.setFixedWidth(130)
        audio_lbl.setStyleSheet("font-weight: bold; color: #e0e0e0;")
        audio_row.addWidget(audio_lbl)

        self._audio_combo = QComboBox()
        self._audio_combo.addItem("Off", "off")
        self._audio_combo.addItem("Verify (cross-check visual offset)", "verify")
        self._audio_combo.addItem(
            "Verify + Fallback (use audio if visual fails)", "fallback"
        )
        self._audio_combo.setCurrentIndex(0)
        self._audio_combo.setStyleSheet(
            "QComboBox { background: #2b2b2b; color: #e0e0e0; border: 1px solid #555; "
            "border-radius: 4px; padding: 4px 8px; }"
            "QComboBox QAbstractItemView { background: #2b2b2b; color: #e0e0e0; "
            "selection-background-color: #3a7bd5; }"
        )
        audio_row.addWidget(self._audio_combo, stretch=1)
        settings_layout.addLayout(audio_row)

        # --- Scene-Cut sync verification ---
        cut_row = QHBoxLayout()
        cut_lbl = QLabel("Scene-Cut Verify:")
        cut_lbl.setFixedWidth(130)
        cut_lbl.setStyleSheet("font-weight: bold; color: #e0e0e0;")
        cut_row.addWidget(cut_lbl)

        self._cut_combo = QComboBox()
        self._cut_combo.addItem("Off", "off")
        self._cut_combo.addItem("Verify (cross-check visual offset)", "verify")
        self._cut_combo.addItem(
            "Verify + Fallback (use scene cuts if visual fails)", "fallback"
        )
        self._cut_combo.setCurrentIndex(0)
        self._cut_combo.setStyleSheet(
            "QComboBox { background: #2b2b2b; color: #e0e0e0; border: 1px solid #555; "
            "border-radius: 4px; padding: 4px 8px; }"
            "QComboBox QAbstractItemView { background: #2b2b2b; color: #e0e0e0; "
            "selection-background-color: #3a7bd5; }"
        )
        cut_row.addWidget(self._cut_combo, stretch=1)
        settings_layout.addLayout(cut_row)

        self._preview_check = QCheckBox("Generate difference verification preview")
        self._preview_check.setChecked(True)
        settings_layout.addWidget(self._preview_check)

        self._constant_offset_check = QCheckBox(
            "Constant offset export only (no segmented / black filler / piecewise encode)"
        )
        self._constant_offset_check.setChecked(False)
        self._constant_offset_check.setToolTip(
            "When dense sampling sees a drifting offset, the app normally builds "
            "segments (blackdetect boundaries, filler gaps, per-chunk encode). "
            "Check this for sources that only need one global shift: always "
            "export a single trim-or-pad pass (faster; typical for uniform A/V lag)."
        )
        settings_layout.addWidget(self._constant_offset_check)

        self._debug_verify_check = QCheckBox(
            "Debug Mode: verify synced output (sub-frame strict) and write "
            "drift report (slower; ~3-8 min/pair on long files)"
        )
        self._debug_verify_check.setChecked(False)
        self._debug_verify_check.setStyleSheet("color: #f1c40f;")
        settings_layout.addWidget(self._debug_verify_check)

        main_layout.addWidget(settings_group)

        # --- Progress ---
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self._scan_bar = StyledProgressBar("Scan Progress:")
        progress_layout.addWidget(self._scan_bar)

        self._export_bar = StyledProgressBar("Overall Progress:")
        progress_layout.addWidget(self._export_bar)

        main_layout.addWidget(progress_group)

        # --- Buttons ---
        btn_layout = QHBoxLayout()

        self._pair_count_lbl = QLabel("0 pair(s) ready")
        self._pair_count_lbl.setStyleSheet("color: #888; font-size: 12px;")
        btn_layout.addWidget(self._pair_count_lbl)
        btn_layout.addStretch(1)

        self._start_btn = QPushButton("Start Sync")
        self._start_btn.setFixedHeight(42)
        self._start_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self._start_btn.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; border: none; "
            "border-radius: 8px; padding: 0 32px; }"
            "QPushButton:hover { background: #2ecc71; }"
            "QPushButton:pressed { background: #1e8449; }"
            "QPushButton:disabled { background: #555; color: #888; }"
        )
        self._start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self._start_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(42)
        self._cancel_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self._cancel_btn.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; border: none; "
            "border-radius: 8px; padding: 0 32px; }"
            "QPushButton:hover { background: #e74c3c; }"
            "QPushButton:pressed { background: #a93226; }"
            "QPushButton:disabled { background: #555; color: #888; }"
        )
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)

        main_layout.addLayout(btn_layout)

        # --- Console ---
        console_group = QGroupBox("Console Log")
        console_layout = QVBoxLayout(console_group)

        self._console = ConsoleLog()
        console_layout.addWidget(self._console)

        main_layout.addWidget(console_group, stretch=1)

        self._dual_list.pairs_changed.connect(self._update_pair_count)
        self._update_pair_count()

        self._console.append_info("Ready. Drag files into the lists or use the toolbar.")

    # --- Helpers ------------------------------------------------------------

    def _update_pair_count(self):
        pairs = self._dual_list.pairs()
        valid = sum(1 for p, t in pairs if p and t)
        self._pair_count_lbl.setText(f"{valid} pair(s) ready")
        self._pair_count_lbl.setStyleSheet(
            "color: #2ecc71; font-size: 12px; font-weight: bold;" if valid > 0
            else "color: #888; font-size: 12px;"
        )

    def _set_running(self, running: bool):
        self._start_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._dual_list.setEnabled(not running)

    # --- Actions ------------------------------------------------------------

    def _on_start(self):
        raw_pairs = self._dual_list.pairs()
        # Filter out placeholder rows (paths == "") and missing files.
        pairs = [
            (p, t) for p, t in raw_pairs
            if p and t and os.path.isfile(p) and os.path.isfile(t)
        ]
        if not pairs:
            QMessageBox.warning(
                self, "No Pairs",
                "No valid file pairs to process. Add files to both lists "
                "and make sure rows are aligned."
            )
            return

        self._scan_bar.reset()
        self._export_bar.reset()
        self._console.clear()

        threshold = self._sensitivity.value()
        preview = self._preview_check.isChecked()
        multi_mode = self._multi_ref_combo.currentData()
        num_refs = self._num_refs_spin.value()
        audio_mode = self._audio_combo.currentData()
        cut_mode = self._cut_combo.currentData()
        debug_verify = self._debug_verify_check.isChecked()
        constant_offset_only = self._constant_offset_check.isChecked()

        mode_label = {
            "off": "single ref",
            "retry": f"single ref, retry with {num_refs} refs on failure",
            "always": f"always {num_refs} refs",
        }.get(multi_mode, multi_mode)

        verify_label = {
            "off": "off",
            "verify": "verify only",
            "fallback": "verify + fallback",
        }

        self._console.append_info(
            f"Starting sync of {len(pairs)} pair(s) | "
            f"Hamming threshold: {threshold} | "
            f"Multi-ref: {mode_label} | "
            f"Audio VAD: {verify_label.get(audio_mode, audio_mode)} | "
            f"Scene-Cut: {verify_label.get(cut_mode, cut_mode)} | "
            f"Difference preview: {'on' if preview else 'off'} | "
            f"Constant-offset export: {'on' if constant_offset_only else 'off'} | "
            f"Debug verify: {'on' if debug_verify else 'off'}"
        )

        worker = PairListSyncWorker(pairs, threshold, preview,
                                    multi_ref_mode=multi_mode, num_refs=num_refs,
                                    audio_mode=audio_mode, scene_cut_mode=cut_mode,
                                    debug_verify=debug_verify,
                                    constant_offset_export_only=constant_offset_only)
        worker.log_signal.connect(self._on_log)
        worker.scan_progress.connect(self._scan_bar.set_progress)
        worker.export_progress.connect(self._export_bar.set_progress)
        worker.finished_signal.connect(self._on_finished)
        worker.error_signal.connect(self._on_error)

        self._worker = worker
        self._set_running(True)
        worker.start()

    def _on_cancel(self):
        if self._worker:
            self._console.append_warning("Cancelling...")
            self._worker.cancel()

    # --- Slots --------------------------------------------------------------

    @pyqtSlot(str)
    def _on_log(self, message: str):
        if not message:
            return
        lower = message.lower()
        if lower.startswith("error") or lower.startswith("✖"):
            self._console.append_error(message)
        elif lower.startswith("warning") or "⚠" in message:
            self._console.append_warning(message)
        elif "===" in message or "complete" in lower:
            self._console.append_success(message)
        else:
            self._console.append_info(message)

    @pyqtSlot(int, int)
    def _on_finished(self, succeeded: int, failed: int):
        self._set_running(False)
        if failed == 0:
            self._console.append_success(
                f"All done. {succeeded} pair(s) synced successfully."
            )
        else:
            self._console.append_warning(
                f"Done. {succeeded} succeeded, {failed} failed. See log above."
            )

    @pyqtSlot(str)
    def _on_error(self, message: str):
        self._set_running(False)
        self._console.append_error(message)
