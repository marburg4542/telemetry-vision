"""Render the PWA icons: a white viewfinder on a blue tile.

Drawn rather than shipped as a binary, so the mark lives in one place, its
geometry can be read, and a change is a rerun rather than an image edit.

Rendered at four times the output size and reduced with INTER_AREA. OpenCV's
antialiasing is a one-pixel ramp, which leaves strokes this thick visibly
stepped at icon sizes; supersampling gives them a real edge.

Usage:
    python scripts/generate_pwa_icons.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "telemetry_ocr" / "web_static" / "icons"

SUPERSAMPLE = 4
# The page's own blues (the --blue family in app.css), as BGR.
BLUE_LIGHT = (240, 134, 47)  # #2f86f0, top left
BLUE_DARK = (168, 79, 11)  # #0b4fa8, bottom right
WHITE = (255, 255, 255)
# Corner radius as a fraction of the side: close to the rounding Windows and
# Android give their own tiles, so the icon does not look pasted onto the shelf.
CORNER = 0.22
# How much a maskable icon's mark shrinks. The platform may crop it to any shape
# inside the central 80% circle; at this scale the bracket corners, the part
# furthest from the middle, stay about a tenth of the side inside that circle.
MASKABLE_SCALE = 0.78


def _gradient(side: int) -> np.ndarray:
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    t = ((xx + yy) / (2 * (side - 1)))[..., None]
    start = np.array(BLUE_LIGHT, np.float32)
    end = np.array(BLUE_DARK, np.float32)
    return (start * (1 - t) + end * t).astype(np.uint8)


def _stroke(
    image: np.ndarray, points: list[tuple[float, float]], thickness: float
) -> None:
    corners = [(round(x), round(y)) for x, y in points]
    width = max(1, round(thickness))
    for start, end in itertools.pairwise(corners):
        cv2.line(image, start, end, WHITE, width, cv2.LINE_AA)
    # Round caps and joins. A square end on a stroke this heavy reads as a
    # rendering fault at 48 pixels rather than as a choice.
    for corner in corners:
        cv2.circle(image, corner, width // 2, WHITE, -1, cv2.LINE_AA)


def _viewfinder(image: np.ndarray, side: int, scale: float) -> None:
    """Four corner brackets and a centre dot, scaled about the middle."""
    centre = side / 2
    reach = side * 0.26 * scale  # from the centre to each bracket's corner
    arm = side * 0.17 * scale
    thickness = side * 0.06 * scale
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = centre + sx * reach, centre + sy * reach
            # Both arms point back toward the middle.
            _stroke(image, [(cx, cy - sy * arm), (cx, cy), (cx - sx * arm, cy)], thickness)
    radius = round(side * 0.075 * scale)
    cv2.circle(image, (round(centre), round(centre)), radius, WHITE, -1, cv2.LINE_AA)


def _rounded_alpha(side: int) -> np.ndarray:
    alpha = np.zeros((side, side), np.uint8)
    radius = round(side * CORNER)
    cv2.rectangle(alpha, (radius, 0), (side - radius, side), 255, -1)
    cv2.rectangle(alpha, (0, radius), (side, side - radius), 255, -1)
    for cx, cy in (
        (radius, radius),
        (side - radius, radius),
        (radius, side - radius),
        (side - radius, side - radius),
    ):
        cv2.circle(alpha, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    return alpha


def render(size: int, *, maskable: bool = False) -> np.ndarray:
    side = size * SUPERSAMPLE
    image = _gradient(side)
    _viewfinder(image, side, MASKABLE_SCALE if maskable else 1.0)
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    # A maskable icon stays full-bleed: the platform cuts the shape, and
    # rounding it here as well would leave a transparent notch at each corner.
    if not maskable:
        bgra[..., 3] = _rounded_alpha(side)
    return cv2.resize(bgra, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT / "icon-192.png"), render(192))
    cv2.imwrite(str(OUTPUT / "icon-512.png"), render(512))
    cv2.imwrite(str(OUTPUT / "icon-maskable-512.png"), render(512, maskable=True))


if __name__ == "__main__":
    main()
