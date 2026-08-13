"""Extract training frames from the source videos at a fixed time interval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from telemetry_ocr.errors import VideoError

from .paths import SUPPORTED_VIDEO_EXTENSIONS, ProjectPaths


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    video: Path
    output_dir: Path
    saved_images: int
    source_fps: float


def find_videos(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise VideoError(f"Video directory not found: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def extract_frames(
    video: Path,
    output_dir: Path,
    *,
    interval_seconds: float,
    jpeg_quality: int = 95,
) -> ExtractionResult:
    """Write one JPEG every `interval_seconds` of playback into `output_dir`."""
    import cv2

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise VideoError(f"Cannot decode video: {video}")

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        # Some containers report 0 or NaN. Assuming 30 keeps the interval
        # roughly right instead of writing every single frame.
        source_fps = fps if fps and fps > 0 else 30.0
        frame_step = max(1, round(source_fps * interval_seconds))

        frame_index = 0
        saved = 0
        while True:
            read_ok, frame = capture.read()
            if not read_ok:
                break
            if frame_index % frame_step == 0:
                destination = output_dir / f"{video.stem}_{saved:06d}.jpg"
                if not cv2.imwrite(
                    str(destination), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                ):
                    raise VideoError(f"Cannot write frame to {destination}")
                saved += 1
            frame_index += 1
    finally:
        capture.release()

    return ExtractionResult(
        video=video, output_dir=output_dir, saved_images=saved, source_fps=source_fps
    )


def extract_from_directory(
    paths: ProjectPaths,
    source: Path | None = None,
    destination: Path | None = None,
    *,
    interval_seconds: float = 1.0,
    jpeg_quality: int = 95,
) -> list[ExtractionResult]:
    """Extract frames from every video under `source` into per-video folders."""
    source_dir = source or paths.asset_dir("videos")
    destination_dir = destination or paths.asset_dir("frames")

    videos = find_videos(source_dir)
    if not videos:
        raise VideoError(
            f"No {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))} files under {source_dir}"
        )

    results = []
    for position, video in enumerate(videos, start=1):
        print(f"[{position}/{len(videos)}] {video.name}")
        result = extract_frames(
            video,
            destination_dir / video.stem,
            interval_seconds=interval_seconds,
            jpeg_quality=jpeg_quality,
        )
        print(f"    {result.saved_images} images -> {result.output_dir}")
        results.append(result)
    return results
