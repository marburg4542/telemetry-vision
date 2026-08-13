from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "telemetry_ocr" / "web_static" / "icons"


def render(size: int, *, maskable: bool = False) -> np.ndarray:
    image = np.full((size, size, 3), (14, 11, 7), dtype=np.uint8)
    center = size // 2
    accent = (75, 240, 168)
    cyan = (221, 215, 101)
    radius = int(size * (0.28 if maskable else 0.34))
    thickness = max(2, size // 64)

    cv2.circle(image, (center, center), radius, accent, thickness, cv2.LINE_AA)
    cv2.circle(image, (center, center), int(radius * 0.56), cyan, thickness, cv2.LINE_AA)
    cv2.line(image, (center, center - radius - size // 18), (center, center + radius + size // 18), accent, thickness, cv2.LINE_AA)
    cv2.line(image, (center - radius - size // 18, center), (center + radius + size // 18, center), accent, thickness, cv2.LINE_AA)
    cv2.circle(image, (center, center), max(4, size // 40), accent, -1, cv2.LINE_AA)
    cv2.line(image, (center, center), (center + int(radius * 0.68), center - int(radius * 0.68)), cyan, thickness, cv2.LINE_AA)
    return image


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT / "icon-192.png"), render(192))
    cv2.imwrite(str(OUTPUT / "icon-512.png"), render(512))
    cv2.imwrite(str(OUTPUT / "icon-maskable-512.png"), render(512, maskable=True))


if __name__ == "__main__":
    main()
