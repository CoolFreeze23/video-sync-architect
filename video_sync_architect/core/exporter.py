"""
FFmpeg export: synced output + difference verification preview.
"""

import os
from typing import Optional, Callable

from .media_info import MediaInfo
from .sync_engine import SyncResult
from ..utils.ffmpeg_utils import (
    build_sync_command,
    build_difference_preview_command,
    run_ffmpeg_with_progress,
)


def generate_output_path(target_path: str, suffix: str = "_synced") -> str:
    base, ext = os.path.splitext(target_path)
    return f"{base}{suffix}{ext}"


def export_synced_file(
    target_info: MediaInfo,
    primary_info: MediaInfo,
    sync_result: SyncResult,
    output_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Export Input 2 shifted by the calculated offset.
    Returns the output file path.
    """
    if output_path is None:
        output_path = generate_output_path(target_info.filepath)

    offset = sync_result.offset_seconds
    duration = primary_info.duration

    if log_callback:
        if offset >= 0:
            log_callback(f"Padding Input 2 by {offset:.3f}s at start...")
        else:
            log_callback(f"Trimming Input 2 by {abs(offset):.3f}s from start...")
        log_callback(f"Output duration: {duration:.3f}s")
        log_callback(f"Output: {output_path}")

    cmd = build_sync_command(
        target_info.filepath, output_path, offset, duration
    )

    if log_callback:
        log_callback(f"FFmpeg command: {' '.join(cmd)}")

    proc = run_ffmpeg_with_progress(cmd, duration, progress_callback)

    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg export failed with return code {proc.returncode}"
        )

    if log_callback:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log_callback(f"Export complete: {output_path} ({size_mb:.1f} MB)")

    return output_path


def export_difference_preview(
    primary_info: MediaInfo,
    synced_path: str,
    output_path: Optional[str] = None,
    duration: float = 5.0,
    seek: float = 30.0,
    progress_callback: Optional[Callable[[float], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Generate a 5-second difference-blend preview.
    Black pixels = perfect visual sync.
    """
    if output_path is None:
        base, ext = os.path.splitext(synced_path)
        output_path = f"{base}_diff_preview{ext}"

    actual_seek = min(seek, max(0, primary_info.duration - duration - 1))

    if log_callback:
        log_callback(
            f"Generating difference preview ({duration}s from t={actual_seek:.1f}s)..."
        )

    cmd = build_difference_preview_command(
        primary_info.filepath, synced_path, output_path,
        duration=duration, seek=actual_seek,
    )

    if log_callback:
        log_callback(f"FFmpeg command: {' '.join(cmd)}")

    proc = run_ffmpeg_with_progress(cmd, duration, progress_callback)

    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg difference preview failed with code {proc.returncode}"
        )

    if log_callback:
        log_callback(f"Difference preview saved: {output_path}")

    return output_path
