"""Draw detector boxes onto a copy of a frame.

Its own module, importing nothing but OpenCV, because two processes that must
not share imports both draw boxes: the server writing a capture's marked-up copy,
which holds PaddleOCR, and the render child writing every frame of a video,
which holds torch. capture_service imports the OCR pipeline, so the render child
cannot import the drawing code from there.
"""

from __future__ import annotations

from typing import Any


def annotate_frame(
    frame: Any, boxes: list[dict[str, Any]], source_width: int, source_height: int
) -> Any:
    """Draw the detector's boxes and labels onto a copy of the frame.

    A copy, always. The clean frame is the record and the thing OCR is re-read
    from, and a box drawn across an OSD field would corrupt the reading taken
    from it -- which is why the annotated picture is written beside the original
    rather than instead of it.

    Boxes arrive in the coordinate space of whatever the detector was given, so
    they are scaled to this frame. Using them unscaled would put every box in the
    top-left corner at a fraction of its size, which looks like a tracking
    failure rather than a units mistake.
    """
    import cv2

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    if not source_width or not source_height:
        return annotated
    scale_x = width / float(source_width)
    scale_y = height / float(source_height)

    # Line and text scaled to the frame: what reads well on 640 px is invisible
    # at 1920.
    thickness = max(2, round(width / 640))
    font_scale = max(0.5, width / 1600.0)

    for box in boxes:
        try:
            x1 = int(float(box["x1"]) * scale_x)
            y1 = int(float(box["y1"]) * scale_y)
            x2 = int(float(box["x2"]) * scale_x)
            y2 = int(float(box["y2"]) * scale_y)
        except (KeyError, TypeError, ValueError):
            continue
        colour = class_colour(int(box.get("class_id", 0) or 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

        label = str(box.get("label", "?"))
        confidence = box.get("confidence")
        if isinstance(confidence, (int, float)):
            label = f"{label} {float(confidence):.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # Above the box, unless that would be off the top of the frame; pulled
        # back in at the right edge rather than cut off.
        top = y1 - text_h - baseline
        if top < 0:
            top = y2
        left = max(0, min(x1, width - text_w - 6))
        cv2.rectangle(
            annotated,
            (left, top),
            (left + text_w + 6, top + text_h + baseline + 4),
            colour,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (left + 3, top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (16, 16, 16),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def class_colour(class_id: int) -> tuple[int, int, int]:
    """One stable hue per class, as BGR.

    The same class gets the same colour in a capture's marked-up copy and in a
    rendered video, so an operator comparing the two is not left wondering
    whether a green box and an orange one are the same detection.
    """
    import colorsys

    hue = ((class_id * 67 + 88) % 360) / 360.0
    red, green, blue = colorsys.hls_to_rgb(hue, 0.62, 0.88)
    return (int(blue * 255), int(green * 255), int(red * 255))
