"""Generate a short 1080p30 video and fixed-ROI config for a real OCR smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/smoke"))
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--model-name", default="PP-OCRv6_tiny_rec")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "telemetry_smoke.avi"
    config_path = args.output_dir / "roi_config.yaml"
    width, height, fps = 1920, 1080, 30.0
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("MJPG encoder is unavailable")
    try:
        x_gradient = np.linspace(35, 100, width, dtype=np.uint8)
        base = np.repeat(x_gradient[np.newaxis, :], height, axis=0)
        for frame_index in range(int(args.seconds * fps)):
            frame = np.dstack(
                (
                    np.roll(base, frame_index * 3, axis=1),
                    np.roll(base, frame_index * 2, axis=1),
                    base,
                )
            )
            cv2.rectangle(frame, (650, 730), (1590, 930), (55, 65, 75), -1)
            cv2.putText(frame, "SPEED", (710, 775), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (130, 130, 130), 2)
            cv2.putText(frame, "123.4", (700, 875), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (205, 205, 205), 5, cv2.LINE_AA)
            cv2.putText(frame, "RPM", (1065, 775), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (130, 130, 130), 2)
            cv2.putText(frame, "2450", (1040, 875), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (190, 205, 215), 5, cv2.LINE_AA)
            cv2.putText(frame, "GEAR", (1390, 775), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (130, 130, 130), 2)
            cv2.putText(frame, "D", (1425, 875), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (190, 215, 190), 5, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()

    common_primary = {
        "name": "lab_l_clahe",
        "channel": "lab_l",
        "scale": 2.0,
        "clahe": True,
        "unsharp": True,
    }
    common_fallback = {
        "name": "gray_adaptive",
        "channel": "gray",
        "scale": 2.0,
        "clahe": True,
        "unsharp": True,
        "threshold": "adaptive",
    }
    config = {
        "version": 1,
        "runtime": {
            "model_name": args.model_name,
            "model_source": "BOS",
            "model_cache_dir": "models/paddlex",
            "device": "cpu",
            "enable_hpi": False,
            "enable_mkldnn": True,
            "cpu_threads": 4,
            "batch_size": 3,
            "warmup_iterations": 2,
            "ocr_hz": 12.0,
            "queue_size": 2,
            "realtime": True,
            "confidence_threshold": 0.80,
            "padding": 0.04,
            "aspect_bucket_base": 4.0,
        },
        "temporal": {
            "fast": {"window": 3, "required": 2},
            "stable": {"window": 5, "required": 3},
        },
        "fields": [
            {
                "field_id": "speed",
                "box": [0.35, 0.735, 0.17, 0.09],
                "allowed_chars": "0123456789.",
                "regex": r"^\d{1,3}(?:\.\d)?$",
                "data_type": "float",
                "decimals": 1,
                "min_value": 0,
                "max_value": 400,
                "profile": "fast",
                "primary": common_primary,
                "fallback": common_fallback,
            },
            {
                "field_id": "rpm",
                "box": [0.525, 0.735, 0.15, 0.09],
                "allowed_chars": "0123456789",
                "regex": r"^\d{1,5}$",
                "data_type": "integer",
                "min_value": 0,
                "max_value": 20000,
                "profile": "fast",
                "primary": common_primary,
                "fallback": common_fallback,
            },
            {
                "field_id": "gear",
                "box": [0.72, 0.735, 0.07, 0.09],
                "allowed_chars": "PRND123456",
                "regex": r"^(?:P|R|N|D|[1-6])$",
                "data_type": "string",
                "profile": "stable",
                "primary": common_primary,
                "fallback": common_fallback,
            },
        ],
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(video_path)
    print(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
