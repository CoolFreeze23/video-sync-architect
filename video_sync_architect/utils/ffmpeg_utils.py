"""
Low-level ffmpeg / ffprobe subprocess helpers.
All duration and offset values use seconds (float64) for framerate independence.
"""

import json
import os
import re
import subprocess
import shutil
from typing import Optional


def _find_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"'{name}' not found on PATH. Install FFmpeg and ensure it is accessible."
        )
    return path


class _LazyBinary:
    """Resolve binary path on first use so imports don't fail without ffmpeg."""

    def __init__(self, name: str):
        self._name = name
        self._path: Optional[str] = None

    def __str__(self) -> str:
        return self.path

    def __fspath__(self) -> str:
        return self.path

    @property
    def path(self) -> str:
        if self._path is None:
            self._path = _find_binary(self._name)
        return self._path


FFMPEG = _LazyBinary("ffmpeg")
FFPROBE = _LazyBinary("ffprobe")


def run_ffprobe_json(filepath: str) -> dict:
    cmd = [
        FFPROBE.path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        filepath,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on '{filepath}': {result.stderr.strip()}")
    return json.loads(result.stdout)


def extract_frame_at_index(filepath: str, frame_index: int, fps: float,
                           width: int = 160, height: int = 90) -> Optional[bytes]:
    """Extract a single frame as raw RGB24 bytes at the given frame index."""
    timestamp = frame_index / fps
    cmd = [
        FFMPEG.path,
        "-ss", f"{timestamp:.6f}",
        "-i", filepath,
        "-vf", f"scale={width}:{height}",
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "quiet",
        "-y",
        "pipe:1",
    ]
    result = subprocess.run(
        cmd, capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0 or len(result.stdout) == 0:
        return None
    return result.stdout


def extract_frames_range(filepath: str, start_frame: int, end_frame: int,
                         fps: float, step: int = 1,
                         width: int = 160, height: int = 90):
    """
    Generator that yields (frame_index, raw_rgb_bytes) for frames in [start_frame, end_frame).
    Uses a single ffmpeg process with piped output for efficiency.
    """
    start_time = start_frame / fps
    num_frames = end_frame - start_frame
    cmd = [
        FFMPEG.path,
        "-ss", f"{start_time:.6f}",
        "-i", filepath,
        "-vf", f"scale={width}:{height}",
        "-frames:v", str(num_frames),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "quiet",
        "-y",
        "pipe:1",
    ]
    frame_size = width * height * 3
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        local_idx = 0
        while True:
            data = proc.stdout.read(frame_size)
            if len(data) < frame_size:
                break
            global_idx = start_frame + local_idx
            if local_idx % step == 0:
                yield global_idx, data
            local_idx += 1
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def build_sync_command(input2_path: str, output_path: str,
                       offset_seconds: float, input1_duration: float,
                       copy_codec: bool = False) -> list[str]:
    """
    Build the ffmpeg command to shift Input 2's audio+video by offset_seconds.
    Positive offset = pad start (Input 2 starts after Input 1).
    Negative offset = trim start (Input 2 starts before Input 1).
    """
    cmd = [FFMPEG.path, "-y"]

    if offset_seconds < 0:
        trim = abs(offset_seconds)
        cmd += ["-ss", f"{trim:.6f}", "-i", input2_path]
        if copy_codec:
            cmd += ["-c", "copy"]
        cmd += ["-t", f"{input1_duration:.6f}", output_path]
    else:
        cmd += ["-i", input2_path]
        offset_ms = int(offset_seconds * 1000)
        vf = f"tpad=start_duration={offset_seconds:.6f}:start_mode=clone"
        af = f"adelay={offset_ms}|{offset_ms}:all=1"
        cmd += [
            "-vf", vf,
            "-af", af,
            "-t", f"{input1_duration:.6f}",
            output_path,
        ]

    return cmd


def build_difference_preview_command(input1_path: str, synced_path: str,
                                     output_path: str, duration: float = 5.0,
                                     seek: float = 30.0) -> list[str]:
    """Build ffmpeg command for a difference-blend verification preview.
    Scales both inputs to the same resolution (the smaller of the two)
    so the blend filter doesn't choke on mismatched dimensions."""
    filter_str = (
        "[0:v]scale=854:480:force_original_aspect_ratio=decrease,"
        "pad=854:480:(ow-iw)/2:(oh-ih)/2,setsar=1[a];"
        "[1:v]scale=854:480:force_original_aspect_ratio=decrease,"
        "pad=854:480:(ow-iw)/2:(oh-ih)/2,setsar=1[b];"
        "[a][b]blend=all_mode=difference[v]"
    )
    cmd = [
        FFMPEG.path, "-y",
        "-ss", f"{seek:.6f}", "-i", input1_path,
        "-ss", f"{seek:.6f}", "-i", synced_path,
        "-filter_complex", filter_str,
        "-map", "[v]",
        "-t", f"{duration:.6f}",
        "-an",
        output_path,
    ]
    return cmd


def run_ffmpeg_with_progress(cmd: list[str], total_duration: float,
                             progress_callback=None) -> subprocess.Popen:
    """
    Run an ffmpeg command and parse stderr for progress.
    Calls progress_callback(fraction: float) where fraction is 0.0..1.0.
    Returns the Popen object.
    """
    full_cmd = list(cmd)
    use_progress = progress_callback is not None and total_duration > 0

    if use_progress:
        if "-progress" not in full_cmd:
            full_cmd.insert(1, "-progress")
            full_cmd.insert(2, "pipe:2")
            idx = full_cmd.index("-y") if "-y" in full_cmd else 0
            if "-stats_period" not in full_cmd:
                full_cmd.insert(idx + 1, "-stats_period")
                full_cmd.insert(idx + 2, "0.5")

        proc = subprocess.Popen(
            full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        time_re = re.compile(r"out_time_ms=(\d+)")
        for line in proc.stderr:
            m = time_re.search(line)
            if m:
                current_us = int(m.group(1))
                fraction = min(current_us / (total_duration * 1_000_000), 1.0)
                progress_callback(fraction)
        proc.wait()
        progress_callback(1.0)
    else:
        proc = subprocess.Popen(
            full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        proc.wait()

    return proc
