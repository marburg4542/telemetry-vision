from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import AppConfig, RecipeConfig, clone_config, dump_config
from .errors import DependencyError, VideoError
from .ocr import Recognizer, close_recognizer
from .pipeline import recognize_grouped
from .preprocess import apply_recipe, default_calibration_recipes, extract_roi
from .validation import validate_text


@dataclass(slots=True)
class RecipeStats:
    attempts: int = 0
    valid: int = 0
    labeled: int = 0
    correct: int = 0
    confidence_sum: float = 0.0
    latency_ms: list[float] = field(default_factory=list)

    def report(self) -> dict[str, float | int]:
        return {
            "attempts": self.attempts,
            "valid_rate": self.valid / self.attempts if self.attempts else 0.0,
            "labeled": self.labeled,
            "exact_accuracy": self.correct / self.labeled if self.labeled else 0.0,
            "mean_confidence": self.confidence_sum / self.attempts if self.attempts else 0.0,
            "mean_latency_ms": statistics.fmean(self.latency_ms) if self.latency_ms else 0.0,
        }


def load_labels(path: str | Path | None) -> dict[tuple[int, str], str]:
    if path is None:
        return {}
    labels: dict[tuple[int, str], str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_index", "field_id", "expected"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Labels CSV must contain: {', '.join(sorted(required))}")
        for row in reader:
            expected = "".join(row["expected"].split())
            if expected:
                labels[(int(row["frame_index"]), row["field_id"])] = expected
    return labels


def calibration_score(stats: RecipeStats) -> tuple[float, float, float, float]:
    report = stats.report()
    if stats.labeled:
        first = float(report["exact_accuracy"])
    else:
        first = float(report["valid_rate"])
    return (
        first,
        float(report["valid_rate"]),
        float(report["mean_confidence"]),
        -float(report["mean_latency_ms"]),
    )


def calibrate(
    video_path: str | Path,
    config: AppConfig,
    recognizer: Recognizer,
    output_config: str | Path,
    report_path: str | Path,
    labels_path: str | Path | None = None,
    sample_count: int = 30,
    recipes: list[RecipeConfig] | None = None,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise DependencyError("OpenCV is required for calibration") from exc
    labels = load_labels(labels_path)
    fields_by_id = {item.field_id: item for item in config.fields}
    unknown_fields = sorted({field_id for _, field_id in labels} - set(fields_by_id))
    if unknown_fields:
        raise ValueError(f"Labels reference unknown field IDs: {', '.join(unknown_fields)}")
    for (frame_index, field_id), expected in labels.items():
        if frame_index < 0:
            raise ValueError("Label frame indexes cannot be negative")
        checked = validate_text(expected, fields_by_id[field_id])
        if not checked.valid:
            raise ValueError(
                f"Invalid label for frame {frame_index}/{field_id}: {expected!r} ({checked.reason})"
            )
    candidate_recipes = recipes or default_calibration_recipes()
    recipe_names = [recipe.name for recipe in candidate_recipes]
    if len(set(recipe_names)) != len(recipe_names):
        raise ValueError("Calibration recipe names must be unique")
    if len(candidate_recipes) < 2:
        raise ValueError("Calibration requires at least two recipes")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"Cannot open video: {video_path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if labels:
            frame_indexes = sorted({key[0] for key in labels})
        elif total_frames > 0:
            count = min(sample_count, total_frames)
            frame_indexes = sorted(
                {round(index * (total_frames - 1) / max(count - 1, 1)) for index in range(count)}
            )
        else:
            frame_indexes = list(range(sample_count))
        stats: dict[str, dict[str, RecipeStats]] = {
            item.field_id: {recipe.name: RecipeStats() for recipe in candidate_recipes}
            for item in config.fields
        }
        processed_frames = 0
        for frame_index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            processed_frames += 1
            for field_config in config.fields:
                padding = config.runtime.padding if field_config.padding is None else field_config.padding
                crop = extract_roi(frame, field_config.box, padding)
                images = [apply_recipe(crop, recipe) for recipe in candidate_recipes]
                started = time.perf_counter()
                results = recognize_grouped(
                    recognizer, images, config.runtime.aspect_bucket_base
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                per_result_latency = elapsed_ms / max(len(results), 1)
                expected = labels.get((frame_index, field_config.field_id))
                for recipe, result in zip(candidate_recipes, results):
                    item_stats = stats[field_config.field_id][recipe.name]
                    validated = validate_text(result.text, field_config)
                    item_stats.attempts += 1
                    item_stats.valid += int(validated.valid)
                    item_stats.confidence_sum += result.confidence
                    item_stats.latency_ms.append(per_result_latency)
                    if expected is not None:
                        item_stats.labeled += 1
                        item_stats.correct += int(
                            validated.valid and validated.normalized_text == expected
                        )
    finally:
        capture.release()
    if processed_frames == 0:
        raise VideoError("No calibration frames could be decoded")

    updated = clone_config(config)
    report: dict[str, Any] = {
        "video": str(video_path),
        "processed_frames": processed_frames,
        "used_labels": bool(labels),
        "fields": {},
    }
    recipes_by_name = {recipe.name: recipe for recipe in candidate_recipes}
    for field_config in updated.fields:
        field_stats = stats[field_config.field_id]
        ranked = sorted(field_stats, key=lambda name: calibration_score(field_stats[name]), reverse=True)
        field_config.primary = recipes_by_name[ranked[0]]
        field_config.fallback = recipes_by_name[ranked[1] if len(ranked) > 1 else ranked[0]]
        report["fields"][field_config.field_id] = {
            "selected_primary": field_config.primary.name,
            "selected_fallback": field_config.fallback.name,
            "recipes": {name: field_stats[name].report() for name in ranked},
        }
    dump_config(updated, output_config)
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def benchmark_threads(
    video_path: str | Path,
    config: AppConfig,
    recognizer_factory: Any,
    thread_counts: list[int],
    iterations: int,
    warmup: int,
    frame_index: int = 0,
) -> tuple[dict[str, Any], int]:
    try:
        import cv2
    except ImportError as exc:
        raise DependencyError("OpenCV is required for benchmarking") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"Cannot open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise VideoError(f"Cannot decode benchmark frame {frame_index}")
    images = []
    for item in config.fields:
        if item.derive_from_source_time:
            continue
        if item.ocr_interval > 1 and frame_index % item.ocr_interval != item.ocr_phase:
            continue
        padding = config.runtime.padding if item.padding is None else item.padding
        crop = extract_roi(frame, item.box, padding)
        images.append(apply_recipe(crop, item.primary))

    report: dict[str, Any] = {"video": str(video_path), "frame_index": frame_index, "threads": {}}
    for threads in thread_counts:
        runtime = clone_config(config).runtime
        runtime.cpu_threads = threads
        recognizer = recognizer_factory(runtime)
        try:
            for _ in range(warmup):
                recognize_grouped(recognizer, images, runtime.aspect_bucket_base)
            timings = []
            for _ in range(iterations):
                started = time.perf_counter()
                recognize_grouped(recognizer, images, runtime.aspect_bucket_base)
                timings.append((time.perf_counter() - started) * 1000.0)
        finally:
            # Each pass builds its own models and thread pool; holding every
            # one of them open skews the later thread counts being measured.
            close_recognizer(recognizer)
        ordered = sorted(timings)
        p95_index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
        report["threads"][str(threads)] = {
            "mean_ms": statistics.fmean(timings),
            "p95_ms": ordered[p95_index],
            "estimated_update_hz": 1000.0 / statistics.fmean(timings),
        }
    best = min(thread_counts, key=lambda count: report["threads"][str(count)]["p95_ms"])
    report["recommended_cpu_threads"] = best
    return report, best


def export_validation_crops(
    video_path: str | Path,
    config: AppConfig,
    output_dir: str | Path,
    crop_count: int = 1000,
) -> Path:
    """Export uniformly sampled raw ROI crops and an empty labeling manifest."""
    try:
        import cv2
    except ImportError as exc:
        raise DependencyError("OpenCV is required to export validation crops") from exc
    if crop_count <= 0:
        raise ValueError("crop_count must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoError(f"Cannot open video: {video_path}")
    root = Path(output_dir)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "labels.csv"
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise VideoError("Video does not expose a finite frame count")
        frame_count = min(total_frames, max(1, math.ceil(crop_count / len(config.fields))))
        indexes = sorted(
            {round(index * (total_frames - 1) / max(frame_count - 1, 1)) for index in range(frame_count)}
        )
        rows: list[dict[str, Any]] = []
        exported = 0
        for frame_index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            for field_position, item in enumerate(config.fields):
                if exported >= crop_count:
                    break
                padding = config.runtime.padding if item.padding is None else item.padding
                crop = extract_roi(frame, item.box, padding)
                safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in item.field_id)
                relative_path = Path("images") / (
                    f"frame_{frame_index:08d}_{field_position:02d}_{safe_id}.png"
                )
                if not cv2.imwrite(str(root / relative_path), crop):
                    raise OSError(f"Cannot write crop: {root / relative_path}")
                rows.append(
                    {
                        "frame_index": frame_index,
                        "field_id": item.field_id,
                        "expected": "",
                        "crop_path": relative_path.as_posix(),
                    }
                )
                exported += 1
    finally:
        capture.release()
    if not rows:
        raise VideoError("No validation crops could be exported")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["frame_index", "field_id", "expected", "crop_path"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path
