"""
Batch processing: pair files from two directories by name similarity
and process them sequentially.
"""

import os
import difflib
from dataclasses import dataclass
from typing import Optional, Callable

from .media_info import MediaInfo
from .sync_engine import VisualSyncEngine, SyncResult
from .exporter import export_synced_file, export_difference_preview

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".mxf", ".ts", ".webm", ".flv", ".wmv"}


def list_video_files(directory: str) -> list[str]:
    files = []
    for f in os.listdir(directory):
        _, ext = os.path.splitext(f)
        if ext.lower() in VIDEO_EXTENSIONS:
            files.append(os.path.join(directory, f))
    files.sort()
    return files


def _normalize_name(filepath: str) -> str:
    name = os.path.splitext(os.path.basename(filepath))[0]
    for suffix in ("_primary", "_anchor", "_cam1", "_cam2", "_target", "_synced"):
        name = name.replace(suffix, "")
    return name.lower().strip()


@dataclass
class FilePair:
    primary: str
    target: str
    similarity: float


def pair_files(primary_dir: str, target_dir: str,
               min_similarity: float = 0.4) -> tuple[list[FilePair], list[str], list[str]]:
    """
    Pair files by normalized name similarity.
    Returns (pairs, unmatched_primary, unmatched_target).
    """
    primary_files = list_video_files(primary_dir)
    target_files = list_video_files(target_dir)

    primary_names = {f: _normalize_name(f) for f in primary_files}
    target_names = {f: _normalize_name(f) for f in target_files}

    pairs = []
    used_targets = set()

    for pf, pname in sorted(primary_names.items(), key=lambda x: x[1]):
        best_score = 0.0
        best_target = None

        for tf, tname in target_names.items():
            if tf in used_targets:
                continue
            score = difflib.SequenceMatcher(None, pname, tname).ratio()
            if score > best_score:
                best_score = score
                best_target = tf

        if best_target and best_score >= min_similarity:
            pairs.append(FilePair(primary=pf, target=best_target, similarity=best_score))
            used_targets.add(best_target)

    unmatched_primary = [f for f in primary_files if not any(p.primary == f for p in pairs)]
    unmatched_target = [f for f in target_files if f not in used_targets]

    return pairs, unmatched_primary, unmatched_target


@dataclass
class BatchResult:
    pair: FilePair
    sync_result: Optional[SyncResult]
    output_path: Optional[str]
    preview_path: Optional[str]
    error: Optional[str]


def process_batch(
    primary_dir: str,
    target_dir: str,
    hamming_threshold: int = 12,
    generate_preview: bool = True,
    progress_callback: Optional[Callable[[str, float, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> list[BatchResult]:
    """
    Process all paired files sequentially.
    progress_callback(message, pair_progress 0-1, overall_progress 0-1)
    """
    pairs, unmatched_p, unmatched_t = pair_files(primary_dir, target_dir)

    def log(msg, pair_prog=0.0, overall_prog=0.0):
        if progress_callback:
            progress_callback(msg, pair_prog, overall_prog)

    if not pairs:
        log("No file pairs found between directories.", 0, 0)
        return []

    log(f"Found {len(pairs)} file pair(s).", 0, 0)
    for up in unmatched_p:
        log(f"  WARNING: Unmatched primary file: {os.path.basename(up)}", 0, 0)
    for ut in unmatched_t:
        log(f"  WARNING: Unmatched target file: {os.path.basename(ut)}", 0, 0)

    engine = VisualSyncEngine()
    results = []

    for i, pair in enumerate(pairs):
        if cancel_check and cancel_check():
            break

        overall_base = i / len(pairs)
        overall_step = 1.0 / len(pairs)

        log(
            f"\n--- Pair {i + 1}/{len(pairs)} ---\n"
            f"  Primary: {os.path.basename(pair.primary)}\n"
            f"  Target:  {os.path.basename(pair.target)}\n"
            f"  Name similarity: {pair.similarity:.0%}",
            0.0,
            overall_base,
        )

        try:
            primary_info = MediaInfo.from_file(pair.primary)
            target_info = MediaInfo.from_file(pair.target)

            def scan_progress(msg, prog):
                if msg:
                    log(msg, prog if prog >= 0 else 0, overall_base + overall_step * 0.5 * max(prog, 0))

            sync_result = engine.find_offset(
                primary_info, target_info,
                hamming_threshold=hamming_threshold,
                progress_callback=scan_progress,
                cancel_check=cancel_check,
            )

            if sync_result is None:
                results.append(BatchResult(pair, None, None, None, "No match found"))
                log("No match found for this pair.", 1.0, overall_base + overall_step)
                continue

            def export_progress(frac):
                log("", frac, overall_base + overall_step * (0.5 + 0.4 * frac))

            output_path = export_synced_file(
                target_info, primary_info, sync_result,
                progress_callback=export_progress,
                log_callback=lambda m: log(m, -1, -1),
            )

            preview_path = None
            if generate_preview:
                def preview_progress(frac):
                    log("", frac, overall_base + overall_step * (0.9 + 0.1 * frac))

                preview_path = export_difference_preview(
                    primary_info, output_path,
                    progress_callback=preview_progress,
                    log_callback=lambda m: log(m, -1, -1),
                )

            results.append(BatchResult(pair, sync_result, output_path, preview_path, None))

        except Exception as e:
            results.append(BatchResult(pair, None, None, None, str(e)))
            log(f"ERROR processing pair: {e}", 1.0, overall_base + overall_step)

    log(f"\nBatch complete: {len(results)} pair(s) processed.", 1.0, 1.0)
    return results
