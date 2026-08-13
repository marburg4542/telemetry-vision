from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .errors import DependencyError, VideoError


def save_roi_preview(
    video_path: str | Path,
    config: AppConfig,
    output_path: str | Path,
    frame_index: int = 0,
) -> Path:
    """Save a video frame annotated with configured normalized ROI boxes."""
    try:
        import cv2
    except ImportError as exc:
        raise DependencyError("OpenCV is required to render an ROI preview") from exc
    if frame_index < 0:
        raise ValueError("frame_index cannot be negative")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"Cannot open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise VideoError(f"Cannot decode preview frame {frame_index}")

    height, width = frame.shape[:2]
    for position, item in enumerate(config.fields):
        x, y, roi_width, roi_height = item.box
        left = round(x * width)
        top = round(y * height)
        right = round((x + roi_width) * width)
        bottom = round((y + roi_height) * height)
        color = _color(position)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = item.field_id
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        label_top = max(0, top - text_height - baseline - 6)
        cv2.rectangle(
            frame,
            (left, label_top),
            (min(width - 1, left + text_width + 8), top),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (left + 4, max(text_height + 1, top - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise OSError(f"Cannot write ROI preview: {destination}")
    return destination


def _color(position: int) -> tuple[int, int, int]:
    palette = [
        (80, 220, 80),
        (80, 180, 255),
        (255, 160, 80),
        (220, 100, 220),
        (220, 220, 80),
    ]
    return palette[position % len(palette)]
