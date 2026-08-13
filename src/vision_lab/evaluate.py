"""Run YOLO detection or tracking over a whole video and summarise the result.

This is the offline counterpart to the realtime overlay in the OCR web UI: same
checkpoints, same tracker, but it walks the entire file and writes per-frame
counts plus a summary instead of drawing boxes on a live player.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ProjectPaths
from .torch_env import import_ultralytics, relaxed_checkpoint_loading


@dataclass(slots=True)
class EvaluationOptions:
    video: str | Path
    model: str | Path | None = None
    output: Path | None = None
    name: str | None = None
    confidence: float = 0.45
    iou: float = 0.45
    image_size: int = 640
    max_detections: int = 300
    video_stride: int = 1
    device: str | None = None
    classes: list[int] | None = None
    track: bool = False
    tracker: str = "botsort.yaml"
    show: bool = False
    half: bool = False
    agnostic_nms: bool = False
    save_txt: bool = False
    save_conf: bool = False
    save_video: bool = True
    quiet: bool = False
    log_every: int = 100
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        for name, value in (
            ("image_size", self.image_size),
            ("max_detections", self.max_detections),
            ("video_stride", self.video_stride),
            ("log_every", self.log_every),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0")


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def make_run_directory(output_root: Path, video: Path, requested_name: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = requested_name or f"{video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    run_name = base_name
    suffix = 2
    while (output_root / run_name).exists():
        run_name = f"{base_name}_{suffix}"
        suffix += 1
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True)
    return run_dir


def evaluate(paths: ProjectPaths, options: EvaluationOptions) -> dict[str, Any]:
    options.validate()
    video_path = paths.resolve_video(options.video)
    model_path = paths.resolve_model(options.model)
    output_root = options.output or paths.asset_dir("detection_runs")

    torch, yolo_class = import_ultralytics()

    print(f"model: {model_path}")
    print(f"video: {video_path}")
    with relaxed_checkpoint_loading(torch):
        model = yolo_class(str(model_path))
    names = model.names

    run_dir = make_run_directory(output_root, video_path, options.name)
    csv_path = run_dir / "frame_summary.csv"
    summary_path = run_dir / "summary.json"

    known_class_ids = (
        sorted(int(class_id) for class_id in names)
        if isinstance(names, dict)
        else list(range(len(names)))
    )
    known_class_names = [class_name(names, class_id) for class_id in known_class_ids]
    csv_fields = ["processed_frame", "detections", "inference_ms"] + [
        f"count_{name}" for name in known_class_names
    ]

    call_options: dict[str, Any] = {
        "source": str(video_path),
        "stream": True,
        "conf": options.confidence,
        "iou": options.iou,
        "imgsz": options.image_size,
        "max_det": options.max_detections,
        "vid_stride": options.video_stride,
        "show": options.show,
        "save": options.save_video,
        "save_txt": options.save_txt,
        "save_conf": options.save_conf,
        "agnostic_nms": options.agnostic_nms,
        "line_width": 2,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        "verbose": not options.quiet,
        **options.extra,
    }
    # Ultralytics 8.4 deprecated `half` in favour of `quantize` and warns on
    # every run even when it is False, so only send it when it was asked for.
    if options.half:
        call_options["half"] = True
    if options.device is not None:
        call_options["device"] = options.device
    if options.classes is not None:
        call_options["classes"] = options.classes

    mode = "track" if options.track else "predict"
    if options.track:
        call_options["tracker"] = options.tracker
        call_options["persist"] = True
        result_stream = model.track(**call_options)
    else:
        result_stream = model.predict(**call_options)

    processed_frames = 0
    frames_with_detections = 0
    total_detections = 0
    per_class_detections: Counter[str] = Counter()
    track_class_votes: dict[int, Counter[int]] = defaultdict(Counter)
    inference_ms_total = 0.0
    interrupted = False
    started_at = time.perf_counter()

    print(f"{mode} -> {run_dir}")
    if options.show:
        print("press q in the video window, or Ctrl+C here, to stop")

    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()

            for processed_frames, result in enumerate(result_stream, start=1):
                boxes = result.boxes
                class_counts: Counter[str] = Counter()
                detection_count = 0

                if boxes is not None and len(boxes) > 0:
                    class_ids = boxes.cls.int().cpu().tolist()
                    detection_count = len(class_ids)
                    class_counts.update(class_name(names, class_id) for class_id in class_ids)
                    per_class_detections.update(class_counts)

                    if boxes.id is not None:
                        track_ids = boxes.id.int().cpu().tolist()
                        for track_id, class_id in zip(track_ids, class_ids, strict=True):
                            track_class_votes[int(track_id)][int(class_id)] += 1

                inference_ms = float(result.speed.get("inference", 0.0))
                inference_ms_total += inference_ms
                total_detections += detection_count
                if detection_count:
                    frames_with_detections += 1

                row: dict[str, Any] = {
                    "processed_frame": processed_frames,
                    "detections": detection_count,
                    "inference_ms": round(inference_ms, 3),
                }
                for name in known_class_names:
                    row[f"count_{name}"] = class_counts.get(name, 0)
                writer.writerow(row)

                if processed_frames % options.log_every == 0:
                    print(
                        f"frames {processed_frames:,} | "
                        f"detections {total_detections:,} | "
                        f"inference {inference_ms:.1f} ms"
                    )
    except KeyboardInterrupt:
        interrupted = True
        print("\nstopped by user")
    finally:
        close_stream = getattr(result_stream, "close", None)
        if callable(close_stream):
            close_stream()

    elapsed_seconds = time.perf_counter() - started_at
    processed_fps = round(processed_frames / elapsed_seconds, 3) if elapsed_seconds > 0 else 0.0
    unique_tracks_per_class: Counter[str] = Counter()
    for votes in track_class_votes.values():
        if votes:
            best_class_id = votes.most_common(1)[0][0]
            unique_tracks_per_class[class_name(names, best_class_id)] += 1

    summary = {
        "status": "interrupted" if interrupted else "completed",
        "mode": mode,
        "created_at": datetime.now().astimezone().isoformat(),
        "model": str(model_path),
        "video": str(video_path),
        "output_directory": str(run_dir),
        "saved_videos": sorted(
            str(path.resolve()) for ext in ("*.mp4", "*.avi") for path in run_dir.rglob(ext)
        ),
        "settings": {
            "conf": options.confidence,
            "iou": options.iou,
            "imgsz": options.image_size,
            "max_det": options.max_detections,
            "vid_stride": options.video_stride,
            "device": options.device or "auto",
            "half": options.half,
            "classes": options.classes,
            "tracker": options.tracker if options.track else None,
        },
        "metrics": {
            "processed_frames": processed_frames,
            "frames_with_detections": frames_with_detections,
            "total_frame_detections": total_detections,
            "per_class_frame_detections": dict(sorted(per_class_detections.items())),
            "unique_tracks": len(track_class_votes) if options.track else None,
            "unique_tracks_per_class": (
                dict(sorted(unique_tracks_per_class.items())) if options.track else None
            ),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "processed_fps": processed_fps,
            "average_inference_ms": (
                round(inference_ms_total / processed_frames, 3) if processed_frames else 0.0
            ),
        },
        "notes": {
            "total_frame_detections": "boxes summed over frames, not a count of distinct objects",
            "unique_tracks": "approximates distinct objects, and only when --track is used",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nprocessed frames: {processed_frames:,}")
    print(f"frames with detections: {frames_with_detections:,}")
    print(f"total boxes: {total_detections:,}")
    print(f"per class: {dict(sorted(per_class_detections.items()))}")
    if options.track:
        print(f"unique track ids: {len(track_class_votes):,}")
        print(f"tracks per class: {dict(sorted(unique_tracks_per_class.items()))}")
    print(f"pipeline throughput: {processed_fps:.2f} FPS")
    print(f"summary: {summary_path}")
    print(f"per-frame csv: {csv_path}")
    return summary
