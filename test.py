r"""Run YOLO detection or tracking on a video and save an evaluation summary.

Examples (PowerShell):
    python test.py "data\my_video.mp4"
    python test.py "data\my_video.mp4" --track --show
    python test.py "data\my_video.mp4" --model "visdrone_custom_4th\weights\best.pt" --imgsz 960
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "visdrone_custom_6th" / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "video_results"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ทดสอบโมเดล YOLO กับวิดีโอและบันทึกผลพร้อม summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "ตัวอย่าง:\n"
            "  python test.py data\\my_video.mp4\n"
            "  python test.py data\\my_video.mp4 --track --show\n"
            "  python test.py data\\my_video.mp4 --conf 0.50 --no-save-video"
        ),
    )
    parser.add_argument("video", type=Path, help="พาธวิดีโอที่ต้องการทดสอบ")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"พาธ checkpoint (ค่าเริ่มต้น: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"โฟลเดอร์ผลลัพธ์ (ค่าเริ่มต้น: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--name", help="ชื่อรอบทดสอบ; ถ้าไม่ระบุจะสร้างจากชื่อวิดีโอและเวลา")
    parser.add_argument("--conf", type=float, default=0.45, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold สำหรับ NMS")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="ขนาดภาพสำหรับ inference (ค่าเริ่มต้น: 640 ให้ตรงกับ imgsz ที่ใช้ตอนเทรนโมเดล; "
        "ถ้าใส่ไม่ตรงกับตอนเทรน ความแม่นยำจะลดลง)",
    )
    parser.add_argument("--max-det", type=int, default=300, help="จำนวน detection สูงสุดต่อเฟรม")
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="ประมวลผลทุก N เฟรม; ใช้ 2 หรือ 3 เพื่อทดสอบวิดีโอยาวให้เร็วขึ้น",
    )
    parser.add_argument(
        "--device",
        help="อุปกรณ์ เช่น 0, 0,1 หรือ cpu; ไม่ระบุเพื่อให้ Ultralytics เลือกอัตโนมัติ",
    )
    parser.add_argument("--classes", type=int, nargs="+", help="เลือกเฉพาะ class id ที่ต้องการ")
    parser.add_argument("--track", action="store_true", help="เปิด object tracking เพื่อลดกรอบกระพริบ")
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="tracker config เช่น botsort.yaml หรือ bytetrack.yaml",
    )
    parser.add_argument("--show", action="store_true", help="แสดงหน้าต่างวิดีโอขณะประมวลผล")
    parser.add_argument("--half", action="store_true", help="ใช้ FP16; เหมาะกับ CUDA GPU")
    parser.add_argument("--agnostic-nms", action="store_true", help="ทำ NMS ข้ามคลาส")
    parser.add_argument("--save-txt", action="store_true", help="บันทึก detection เป็นไฟล์ข้อความ")
    parser.add_argument("--save-conf", action="store_true", help="บันทึก confidence ในไฟล์ข้อความ")
    parser.add_argument("--no-save-video", action="store_true", help="ไม่บันทึกวิดีโอที่วาดกรอบแล้ว")
    parser.add_argument("--quiet", action="store_true", help="ลด log รายเฟรมจาก Ultralytics")
    parser.add_argument("--log-every", type=int, default=100, help="แสดงความคืบหน้าทุก N เฟรม")
    return parser


def resolve_existing_file(path: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"ไม่พบ{label}: {candidate}")
    return candidate


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf ต้องอยู่ระหว่าง 0 และ 1")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou ต้องอยู่ระหว่าง 0 และ 1")
    if args.imgsz <= 0:
        raise ValueError("--imgsz ต้องมากกว่า 0")
    if args.max_det <= 0:
        raise ValueError("--max-det ต้องมากกว่า 0")
    if args.vid_stride <= 0:
        raise ValueError("--vid-stride ต้องมากกว่า 0")
    if args.log_every <= 0:
        raise ValueError("--log-every ต้องมากกว่า 0")


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def make_run_directory(output_root: Path, video: Path, requested_name: str | None) -> tuple[Path, str]:
    output_root = output_root if output_root.is_absolute() else PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if requested_name:
        base_name = requested_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{video.stem}_{timestamp}"

    run_name = base_name
    suffix = 2
    while (output_root / run_name).exists():
        run_name = f"{base_name}_{suffix}"
        suffix += 1

    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_name


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    video_path = resolve_existing_file(args.video, "วิดีโอ")
    model_path = resolve_existing_file(args.model, "โมเดล")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"ไฟล์วิดีโอต้องเป็นหนึ่งใน: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ยังไม่ได้ติดตั้ง ultralytics ใน Python environment นี้ "
            "กรุณาเปิด environment ที่ใช้เทรนโมเดล หรือติดตั้ง ultralytics ก่อน"
        ) from exc

    # PyTorch 2.6+ เปลี่ยนค่า default ของ weights_only เป็น True ทำให้โหลด
    # checkpoint ที่เทรนไว้ก่อนหน้าไม่ได้ จึงต้อง patch กลับเป็น False (เหมือนใน train.py)
    _original_torch_load = torch.load

    def _patched_torch_load(*load_args: Any, **load_kwargs: Any) -> Any:
        if "weights_only" not in load_kwargs:
            load_kwargs["weights_only"] = False
        return _original_torch_load(*load_args, **load_kwargs)

    torch.load = _patched_torch_load

    print(f"โมเดล: {model_path}")
    print(f"วิดีโอ: {video_path}")
    print("กำลังโหลดโมเดล...")
    model = YOLO(str(model_path))
    names = model.names

    run_dir, run_name = make_run_directory(args.output, video_path, args.name)
    output_root = run_dir.parent
    csv_path = run_dir / "frame_summary.csv"
    summary_path = run_dir / "summary.json"

    known_class_ids = sorted(int(class_id) for class_id in names.keys()) if isinstance(names, dict) else list(range(len(names)))
    known_class_names = [class_name(names, class_id) for class_id in known_class_ids]
    csv_fields = ["processed_frame", "detections", "inference_ms"] + [
        f"count_{name}" for name in known_class_names
    ]

    common_options: dict[str, Any] = {
        "source": str(video_path),
        "stream": True,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "vid_stride": args.vid_stride,
        "show": args.show,
        "save": not args.no_save_video,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "half": args.half,
        "agnostic_nms": args.agnostic_nms,
        "line_width": 2,
        "project": str(output_root),
        "name": run_name,
        "exist_ok": True,
        "verbose": not args.quiet,
    }
    if args.device is not None:
        common_options["device"] = args.device
    if args.classes is not None:
        common_options["classes"] = args.classes

    mode = "track" if args.track else "predict"
    if args.track:
        common_options["tracker"] = args.tracker
        common_options["persist"] = True
        result_stream = model.track(**common_options)
    else:
        result_stream = model.predict(**common_options)

    processed_frames = 0
    frames_with_detections = 0
    total_detections = 0
    per_class_detections: Counter[str] = Counter()
    track_class_votes: dict[int, Counter[int]] = defaultdict(Counter)
    inference_ms_total = 0.0
    interrupted = False
    started_at = time.perf_counter()

    print(f"เริ่ม {mode}... ผลลัพธ์จะอยู่ที่: {run_dir}")
    if args.show:
        print("กด q ในหน้าต่างวิดีโอ หรือ Ctrl+C ใน terminal เพื่อหยุด")

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
                        for track_id, class_id in zip(track_ids, class_ids):
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

                if processed_frames % args.log_every == 0:
                    print(
                        f"เฟรมที่ประมวลผล: {processed_frames:,} | "
                        f"detections สะสม: {total_detections:,} | "
                        f"inference: {inference_ms:.1f} ms"
                    )
    except KeyboardInterrupt:
        interrupted = True
        print("\nหยุดการประมวลผลตามคำสั่งผู้ใช้")
    finally:
        close_stream = getattr(result_stream, "close", None)
        if callable(close_stream):
            close_stream()

    elapsed_seconds = time.perf_counter() - started_at
    unique_tracks_per_class: Counter[str] = Counter()
    for votes in track_class_votes.values():
        if votes:
            best_class_id = votes.most_common(1)[0][0]
            unique_tracks_per_class[class_name(names, best_class_id)] += 1

    saved_videos = sorted(
        str(path.resolve())
        for ext in ("*.mp4", "*.avi")
        for path in run_dir.rglob(ext)
    )
    summary = {
        "status": "interrupted" if interrupted else "completed",
        "mode": mode,
        "created_at": datetime.now().astimezone().isoformat(),
        "model": str(model_path),
        "video": str(video_path),
        "output_directory": str(run_dir),
        "saved_videos": saved_videos,
        "settings": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "max_det": args.max_det,
            "vid_stride": args.vid_stride,
            "device": args.device or "auto",
            "half": args.half,
            "classes": args.classes,
            "tracker": args.tracker if args.track else None,
        },
        "metrics": {
            "processed_frames": processed_frames,
            "frames_with_detections": frames_with_detections,
            "total_frame_detections": total_detections,
            "per_class_frame_detections": dict(sorted(per_class_detections.items())),
            "unique_tracks": len(track_class_votes) if args.track else None,
            "unique_tracks_per_class": (
                dict(sorted(unique_tracks_per_class.items())) if args.track else None
            ),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "processed_fps": round(processed_frames / elapsed_seconds, 3)
            if elapsed_seconds > 0
            else 0.0,
            "average_inference_ms": round(inference_ms_total / processed_frames, 3)
            if processed_frames
            else 0.0,
        },
        "notes": {
            "total_frame_detections": "เป็นจำนวนกรอบรวมทุกเฟรม ไม่ใช่จำนวนรถที่ไม่ซ้ำ",
            "unique_tracks": "ใช้เป็นค่าประมาณจำนวนวัตถุไม่ซ้ำเฉพาะเมื่อเปิด --track",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nประมวลผลเสร็จแล้ว")
    print(f"เฟรมที่ประมวลผล: {processed_frames:,}")
    print(f"เฟรมที่พบวัตถุ: {frames_with_detections:,}")
    print(f"กรอบตรวจจับรวม: {total_detections:,}")
    print(f"Detection ต่อคลาส: {dict(sorted(per_class_detections.items()))}")
    if args.track:
        print(f"Track ID ไม่ซ้ำ: {len(track_class_votes):,}")
        print(f"Track ต่อคลาส: {dict(sorted(unique_tracks_per_class.items()))}")
    print(f"ความเร็วเฉลี่ยทั้ง pipeline: {summary['metrics']['processed_fps']:.2f} FPS")
    print(f"ผลลัพธ์: {run_dir}")
    print(f"สรุป: {summary_path}")
    print(f"ข้อมูลรายเฟรม: {csv_path}")
    return run_dir


def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: การทดสอบล้มเหลว: {exc}", file=sys.stderr)
        print(
            "ตรวจสอบว่า ultralytics/PyTorch เป็น environment เดียวกับที่ใช้สร้าง checkpoint",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
