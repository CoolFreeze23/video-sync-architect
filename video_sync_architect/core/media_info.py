"""
MediaInfo dataclass and ffprobe wrapper.
Extracts video metadata needed for sync calculations.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from ..utils.ffmpeg_utils import run_ffprobe_json


@dataclass
class MediaInfo:
    filepath: str
    duration: float          # seconds
    fps: float               # frames per second (float, e.g. 23.976)
    fps_rational: str        # rational string e.g. "24000/1001"
    total_frames: int
    width: int
    height: int
    has_audio: bool
    codec_name: str
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None

    @classmethod
    def from_file(cls, filepath: str) -> "MediaInfo":
        data = run_ffprobe_json(filepath)

        video_stream = None
        audio_stream = None
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and video_stream is None:
                video_stream = s
            elif s.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = s

        if video_stream is None:
            raise ValueError(f"No video stream found in '{filepath}'")

        fps_str = video_stream.get("r_frame_rate", "24/1")
        frac = Fraction(fps_str)
        fps = float(frac)

        duration = float(data.get("format", {}).get("duration", 0))
        if duration == 0:
            duration = float(video_stream.get("duration", 0))

        nb_frames_str = video_stream.get("nb_frames", "0")
        try:
            total_frames = int(nb_frames_str)
        except (ValueError, TypeError):
            total_frames = 0
        if total_frames == 0 and duration > 0 and fps > 0:
            total_frames = int(duration * fps)

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        codec_name = video_stream.get("codec_name", "unknown")

        has_audio = audio_stream is not None
        audio_codec = audio_stream.get("codec_name") if audio_stream else None
        audio_sr = None
        if audio_stream and audio_stream.get("sample_rate"):
            try:
                audio_sr = int(audio_stream["sample_rate"])
            except (ValueError, TypeError):
                pass

        return cls(
            filepath=filepath,
            duration=duration,
            fps=fps,
            fps_rational=fps_str,
            total_frames=total_frames,
            width=width,
            height=height,
            has_audio=has_audio,
            codec_name=codec_name,
            audio_codec=audio_codec,
            audio_sample_rate=audio_sr,
        )

    def frame_to_time(self, frame_index: int) -> float:
        return frame_index / self.fps if self.fps > 0 else 0.0

    def time_to_frame(self, time_seconds: float) -> int:
        return int(time_seconds * self.fps) if self.fps > 0 else 0

    def summary(self) -> str:
        return (
            f"{self.filepath}\n"
            f"  Duration: {self.duration:.3f}s | FPS: {self.fps:.3f} ({self.fps_rational})\n"
            f"  Resolution: {self.width}x{self.height} | Codec: {self.codec_name}\n"
            f"  Frames: {self.total_frames} | Audio: {self.audio_codec or 'none'}"
        )
