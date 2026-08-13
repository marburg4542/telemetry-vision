"""Probe configured ROIs at selected video timestamps without temporal voting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from telemetry_ocr.config import load_config
from telemetry_ocr.ocr import PaddleRecognizer
from telemetry_ocr.pipeline import recognize_grouped
from telemetry_ocr.preprocess import apply_recipe, extract_roi
from telemetry_ocr.validation import prefer_candidate, validate_text


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--times", default="0,30,60,120,240,360,480,510")
    parser.add_argument("--output-dir", type=Path, default=Path("output/roi-probe"))
    args = parser.parse_args()
    times = [float(value.strip()) for value in args.times.split(",") if value.strip()]
    config = load_config(args.config)
    recognizer = PaddleRecognizer(config.runtime)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    rows = []
    try:
        for time_seconds in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            crops = []
            for field in config.fields:
                padding = config.runtime.padding if field.padding is None else field.padding
                crop = extract_roi(frame, field.box, padding)
                processed = apply_recipe(crop, field.primary)
                crops.append(processed)
                safe_time = str(time_seconds).replace(".", "_")
                cv2.imwrite(
                    str(
                        args.output_dir
                        / f"{safe_time}s_{field.field_id}_{field.primary.name}.png"
                    ),
                    processed,
                )
            results = recognize_grouped(recognizer, crops, config.runtime.aspect_bucket_base)
            candidates = []
            fallback_indexes = []
            fallback_images = []
            for index, (field, result) in enumerate(zip(config.fields, results, strict=True)):
                checked = validate_text(result.text, field)
                candidates.append((result, checked, field.primary.name))
                if config.runtime.enable_fallback and (
                    not checked.valid
                    or result.confidence < config.runtime.confidence_threshold
                ):
                    padding = config.runtime.padding if field.padding is None else field.padding
                    crop = extract_roi(frame, field.box, padding)
                    fallback_images.append(apply_recipe(crop, field.fallback))
                    fallback_indexes.append(index)
            if fallback_images:
                fallback_results = recognize_grouped(
                    recognizer, fallback_images, config.runtime.aspect_bucket_base
                )
                for index, fallback_result in zip(fallback_indexes, fallback_results, strict=True):
                    field = config.fields[index]
                    fallback_checked = validate_text(fallback_result.text, field)
                    primary_result, primary_checked, primary_variant = candidates[index]
                    preferred = prefer_candidate(
                        (primary_checked, primary_result.confidence),
                        (fallback_checked, fallback_result.confidence),
                    )
                    if preferred[0] is fallback_checked:
                        candidates[index] = (
                            fallback_result,
                            fallback_checked,
                            field.fallback.name,
                        )
                    else:
                        candidates[index] = (
                            primary_result,
                            primary_checked,
                            primary_variant,
                        )
            for field, (result, checked, variant) in zip(config.fields, candidates, strict=True):
                rows.append(
                    {
                        "time_seconds": time_seconds,
                        "field_id": field.field_id,
                        "raw_text": result.text,
                        "confidence": result.confidence,
                        "valid": checked.valid,
                        "normalized": checked.normalized_text,
                        "value": checked.value,
                        "reason": checked.reason,
                        "preprocess_variant": variant,
                    }
                )
    finally:
        capture.release()
    report_path = args.output_dir / "probe.json"
    report_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
