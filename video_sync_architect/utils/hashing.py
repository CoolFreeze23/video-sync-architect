"""
Frame extraction and perceptual hashing (pHash) computation.
All frame data is handled at a downscaled resolution for performance.
"""

import numpy as np
import cv2
from PIL import Image
import imagehash
from typing import Optional


HASH_SIZE = 16
THUMB_W, THUMB_H = 160, 90


def raw_rgb_to_pil(raw_bytes: bytes, width: int = THUMB_W, height: int = THUMB_H) -> Image.Image:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 3))
    return Image.fromarray(arr, "RGB")


def compute_phash(image: Image.Image, hash_size: int = HASH_SIZE) -> imagehash.ImageHash:
    return imagehash.phash(image, hash_size=hash_size)


def compute_phash_from_raw(raw_bytes: bytes,
                           width: int = THUMB_W,
                           height: int = THUMB_H,
                           hash_size: int = HASH_SIZE) -> imagehash.ImageHash:
    img = raw_rgb_to_pil(raw_bytes, width, height)
    return compute_phash(img, hash_size)


def hamming_distance(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    return h1 - h2


def frame_has_content(raw_bytes: bytes, width: int = THUMB_W,
                      height: int = THUMB_H, variance_threshold: float = 50.0) -> bool:
    """
    Check if a frame has meaningful visual content by computing
    the variance of a Laplacian edge-detection filter.
    Low variance = solid color / black frame / slate.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 3))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > variance_threshold


def select_reference_frame(filepath: str, fps: float, total_frames: int,
                           skip_seconds: float = 30.0,
                           variance_threshold: float = 50.0) -> Optional[tuple[int, bytes]]:
    """
    Select a good reference frame from a video by skipping the initial
    portion and finding a frame with high visual content.
    Returns (frame_index, raw_rgb_bytes) or None.
    """
    from .ffmpeg_utils import extract_frame_at_index

    start_frame = int(skip_seconds * fps)
    start_frame = min(start_frame, max(0, total_frames - int(fps * 10)))

    for offset in range(0, min(int(fps * 30), total_frames - start_frame), int(fps)):
        idx = start_frame + offset
        raw = extract_frame_at_index(filepath, idx, fps)
        if raw and frame_has_content(raw, variance_threshold=variance_threshold):
            return idx, raw

    for offset in range(0, min(int(fps * 60), total_frames - start_frame)):
        idx = start_frame + offset
        raw = extract_frame_at_index(filepath, idx, fps)
        if raw and frame_has_content(raw, variance_threshold=variance_threshold / 2):
            return idx, raw

    idx = start_frame
    raw = extract_frame_at_index(filepath, idx, fps)
    if raw:
        return idx, raw
    return None


def select_reference_frames(filepath: str, fps: float, total_frames: int,
                            num_refs: int = 5,
                            skip_start_seconds: float = 30.0,
                            skip_end_seconds: float = 60.0,
                            variance_threshold: float = 50.0
                            ) -> list[tuple[int, bytes]]:
    """
    Select multiple reference frames distributed across the video's
    middle section (skipping intros/credits). For each target timestamp,
    find a nearby frame with sufficient visual content.
    Returns a list of (frame_index, raw_rgb_bytes); may be shorter than
    num_refs if some timestamps yielded no usable frame.
    """
    from .ffmpeg_utils import extract_frame_at_index

    if total_frames <= 0 or fps <= 0:
        return []

    skip_start = int(skip_start_seconds * fps)
    skip_end = int(skip_end_seconds * fps)
    usable_start = min(skip_start, max(0, total_frames - 1))
    usable_end = max(usable_start + 1, total_frames - skip_end)

    if usable_end <= usable_start:
        usable_end = total_frames

    # Distribute target frames across the usable range.
    if num_refs <= 1:
        targets = [usable_start]
    else:
        span = usable_end - usable_start
        step = span / (num_refs + 1)
        targets = [int(usable_start + step * (i + 1)) for i in range(num_refs)]

    results: list[tuple[int, bytes]] = []
    seen: set[int] = set()
    search_window = int(fps * 5)  # +/- 5 seconds around each target

    for target_frame in targets:
        chosen: Optional[tuple[int, bytes]] = None
        for delta in range(0, search_window, max(1, int(fps // 2))):
            for sign in (1, -1):
                idx = target_frame + sign * delta
                if idx < 0 or idx >= total_frames or idx in seen:
                    continue
                raw = extract_frame_at_index(filepath, idx, fps)
                if raw and frame_has_content(raw, variance_threshold=variance_threshold):
                    chosen = (idx, raw)
                    break
            if chosen:
                break

        if chosen is None:
            # Fall back: accept whatever frame we can grab at the target.
            raw = extract_frame_at_index(filepath, target_frame, fps)
            if raw:
                chosen = (target_frame, raw)

        if chosen and chosen[0] not in seen:
            seen.add(chosen[0])
            results.append(chosen)

    return results
