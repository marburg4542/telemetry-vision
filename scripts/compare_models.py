"""Compare detector checkpoints the two ways that matter to this project.

1. On a labelled test set (--data, a YOLOv8 data.yaml such as the one a Roboflow
   export or `telemetry-ocr training-frames` writes): mAP and, per class,
   precision and recall -- motorcycle recall is the number this project is
   trying to raise.
2. On a stretch of real footage (--video), through the app's own detector and
   tracker settings: how many numbers the tracker gives out, how often a
   vehicle is given a second one, how many boxes a frame, and how fast.

Run with the CUDA environment, which has torch:

    .venv-cuda\\Scripts\\python.exe scripts\\compare_models.py ^
        --models models\\weights\\14th_opt.pt path\\to\\15th.pt ^
        --data path\\to\\dataset\\data.yaml ^
        --video videos\\2026-06-02T15-27-28-591.mp4 --start 67 --length 40

Nothing is written; the results are printed as Markdown tables.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RELABEL_WINDOW_S = 3.0
RELABEL_SIZES = 1.5


def test_set(model_path: Path, data: Path, image_size: int, split: str) -> dict[str, Any]:
    from ultralytics import YOLO

    # Ultralytics writes a runs/ folder wherever it is started; kept out of the repo.
    with tempfile.TemporaryDirectory() as temp:
        metrics = YOLO(str(model_path)).val(
            data=str(data), split=split, imgsz=image_size, verbose=False, plots=False,
            project=temp, name="val",
        )
    names = metrics.names
    per_class = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, _ = metrics.box.class_result(index)
        per_class[names[int(class_id)]] = (precision, recall, ap50)
    return {"map50": metrics.box.map50, "map": metrics.box.map, "classes": per_class}


def footage(
    model_path: Path, video: Path, start_s: float, length_s: float, stride: int,
    confidence: float, image_size: int,
) -> dict[str, Any]:
    """The app's own detector and tracker over a stretch of a video."""
    import cv2

    from telemetry_ocr.detection import UltralyticsDetector
    from telemetry_ocr.render_job import tracker_config

    detector = UltralyticsDetector(model_path)
    with tempfile.TemporaryDirectory() as temp:
        tracker = tracker_config(Path(temp), confidence)
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        first = int(start_s * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        rows, times = [], []
        for index in range(int(length_s * fps)):
            if index % stride:
                capture.grab()
                continue
            ok, frame = capture.read()
            if not ok:
                break
            began = time.perf_counter()
            found = detector.track(
                frame, confidence=confidence, iou=0.45, image_size=image_size, tracker=tracker
            )
            times.append(time.perf_counter() - began)
            rows.append((first + index, [item for item in found if item.track_id is not None]))
        capture.release()

    tracks: dict[int, dict[str, Any]] = {}
    boxes = 0
    for frame, found in rows:
        boxes += len(found)
        for item in found:
            track = tracks.setdefault(
                item.track_id, {"label": item.label, "first": frame, "first_box": item}
            )
            track["last"], track["last_box"] = frame, item
    relabel = 0
    window = RELABEL_WINDOW_S * fps
    for track_id, track in tracks.items():
        for other_id, other in tracks.items():
            if other_id == track_id or other["label"] != track["label"]:
                continue
            if not 0 < track["first"] - other["last"] <= window:
                continue
            a, b = track["first_box"], other["last_box"]
            size = max(b.x2 - b.x1, b.y2 - b.y1, 1.0)
            dx = (a.x1 + a.x2 - b.x1 - b.x2) / 2
            dy = (a.y1 + a.y2 - b.y1 - b.y2) / 2
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= RELABEL_SIZES * size:
                relabel += 1
                break
    by_class: dict[str, int] = {}
    for track in tracks.values():
        by_class[track["label"]] = by_class.get(track["label"], 0) + 1
    per_frame = statistics.mean(times[5:] or times) if times else 0.0
    return {
        "ids": len(tracks),
        "by_class": by_class,
        "relabel": relabel,
        "boxes_per_frame": boxes / max(1, len(rows)),
        "x_realtime": (stride / fps) / per_frame if per_frame else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--data", type=Path, help="YOLOv8 data.yaml with a labelled split")
    parser.add_argument("--split", default="test")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="Seconds into the video")
    parser.add_argument("--length", type=float, default=40.0, help="Seconds to track")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    if not args.data and not args.video:
        parser.error("give --data, --video, or both")

    if args.data:
        results = {
            path: test_set(path, args.data, args.image_size, args.split) for path in args.models
        }
        classes = sorted({name for result in results.values() for name in result["classes"]})
        print(f"\n## Test set ({args.data}, split {args.split}, {args.image_size} px)\n")
        header = ["model", "mAP50", "mAP50-95"] + [f"{name} P / R" for name in classes]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for path, result in results.items():
            cells = [path.stem, f"{result['map50']:.3f}", f"{result['map']:.3f}"]
            for name in classes:
                values = result["classes"].get(name)
                cells.append(f"{values[0]:.2f} / {values[1]:.2f}" if values else "-")
            print("| " + " | ".join(cells) + " |")

    if args.video:
        print(
            f"\n## {args.video.name} from {args.start:g} s for {args.length:g} s "
            f"(every {args.stride} frames, confidence {args.confidence}, {args.image_size} px)\n"
        )
        print(
            "| model | numbers | by class | same vehicle, new number | boxes / frame "
            "| x real time |"
        )
        print("|---|---|---|---|---|---|")
        for path in args.models:
            result = footage(
                path, args.video, args.start, args.length, args.stride,
                args.confidence, args.image_size,
            )
            by_class = ", ".join(
                f"{name} {count}" for name, count in sorted(result["by_class"].items())
            )
            print(
                f"| {path.stem} | {result['ids']} | {by_class} | {result['relabel']} | "
                f"{result['boxes_per_frame']:.2f} | {result['x_realtime']:.2f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
