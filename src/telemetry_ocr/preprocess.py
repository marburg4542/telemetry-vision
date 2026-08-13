from __future__ import annotations

from typing import Any

from .config import RecipeConfig
from .errors import DependencyError


def _deps() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise DependencyError(
            "OpenCV and NumPy are required for video preprocessing; install project dependencies"
        ) from exc
    return cv2, np


def extract_roi(frame: Any, box: tuple[float, float, float, float], padding: float) -> Any:
    """Crop a normalized ROI with padding relative to the ROI size."""
    _, np = _deps()
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("frame must be a valid image array")
    frame_h, frame_w = frame.shape[:2]
    x, y, width, height = box
    pad_x = width * padding
    pad_y = height * padding
    left = max(0, int(round((x - pad_x) * frame_w)))
    top = max(0, int(round((y - pad_y) * frame_h)))
    right = min(frame_w, int(round((x + width + pad_x) * frame_w)))
    bottom = min(frame_h, int(round((y + height + pad_y) * frame_h)))
    if right <= left or bottom <= top:
        raise ValueError("ROI collapsed to an empty crop")
    crop = frame[top:bottom, left:right]
    return np.ascontiguousarray(crop)


def apply_recipe(crop: Any, recipe: RecipeConfig) -> Any:
    """Apply a deterministic color/contrast recipe and return a 3-channel BGR image."""
    cv2, np = _deps()
    if crop is None or crop.size == 0:
        raise ValueError("Cannot preprocess an empty crop")

    image = cv2.resize(
        crop,
        None,
        fx=recipe.scale,
        fy=recipe.scale,
        interpolation=cv2.INTER_CUBIC,
    )
    image = _select_channel(image, recipe.channel, cv2)

    if recipe.clahe:
        image = _apply_clahe(image, recipe, cv2)
    if recipe.unsharp:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0, sigmaY=1.0)
        image = cv2.addWeighted(
            image,
            1.0 + recipe.unsharp_amount,
            blurred,
            -recipe.unsharp_amount,
            0,
        )
    if recipe.morphology != "none":
        gray = _gray(image, cv2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (recipe.morphology_kernel, recipe.morphology_kernel)
        )
        operation = cv2.MORPH_TOPHAT if recipe.morphology == "tophat" else cv2.MORPH_BLACKHAT
        image = cv2.morphologyEx(gray, operation, kernel)
    if recipe.threshold != "none":
        gray = _gray(image, cv2)
        if recipe.threshold == "otsu":
            _, image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            image = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                recipe.adaptive_block_size,
                recipe.adaptive_c,
            )
    if recipe.invert:
        image = cv2.bitwise_not(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _select_channel(image: Any, channel: str, cv2: Any) -> Any:
    if channel == "color":
        return image
    if channel == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channel in {"b", "g", "r"}:
        return image[:, :, {"b": 0, "g": 1, "r": 2}[channel]]
    if channel.startswith("lab_"):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return lab[:, :, {"lab_l": 0, "lab_a": 1, "lab_b": 2}[channel]]
    if channel.startswith("hsv_"):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv[:, :, {"hsv_s": 1, "hsv_v": 2}[channel]]
    raise ValueError(f"Unsupported channel: {channel}")


def _apply_clahe(image: Any, recipe: RecipeConfig, cv2: Any) -> Any:
    clahe = cv2.createCLAHE(
        clipLimit=recipe.clahe_clip,
        tileGridSize=(recipe.clahe_grid, recipe.clahe_grid),
    )
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _gray(image: Any, cv2: Any) -> Any:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def default_calibration_recipes() -> list[RecipeConfig]:
    """A bounded search space that covers luminance, chroma, morphology and binarization."""
    recipes = [
        RecipeConfig(
            name="color_baseline", channel="color", clahe=False, unsharp=False
        ),
        RecipeConfig(name="color_clahe", channel="color", clahe=True, unsharp=True),
        RecipeConfig(name="gray_clahe", channel="gray", clahe=True, unsharp=True),
        RecipeConfig(name="lab_l_clahe", channel="lab_l", clahe=True, unsharp=True),
        RecipeConfig(name="hsv_v_clahe", channel="hsv_v", clahe=True, unsharp=True),
        RecipeConfig(name="hsv_s_clahe", channel="hsv_s", clahe=True, unsharp=True),
        RecipeConfig(
            name="gray_otsu", channel="gray", clahe=True, unsharp=True, threshold="otsu"
        ),
        RecipeConfig(
            name="gray_adaptive",
            channel="gray",
            clahe=True,
            unsharp=True,
            threshold="adaptive",
        ),
        RecipeConfig(
            name="gray_otsu_inverted",
            channel="gray",
            clahe=True,
            unsharp=True,
            threshold="otsu",
            invert=True,
        ),
        RecipeConfig(
            name="gray_adaptive_inverted",
            channel="gray",
            clahe=True,
            unsharp=True,
            threshold="adaptive",
            invert=True,
        ),
        RecipeConfig(
            name="lab_l_tophat",
            channel="lab_l",
            clahe=True,
            unsharp=True,
            morphology="tophat",
        ),
        RecipeConfig(
            name="lab_l_blackhat",
            channel="lab_l",
            clahe=True,
            unsharp=True,
            morphology="blackhat",
        ),
    ]
    for channel in ("b", "g", "r", "lab_a", "lab_b"):
        recipes.append(
            RecipeConfig(name=f"{channel}_clahe", channel=channel, clahe=True, unsharp=True)
        )
    return recipes
