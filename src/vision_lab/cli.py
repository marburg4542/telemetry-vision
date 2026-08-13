"""`vision-lab`: the detection-side commands.

Kept separate from `telemetry-ocr` on purpose. That entry point must never
import torch, because PaddlePaddle and PyTorch each ship their own cuDNN and
the second one loaded fails on Windows. Two console scripts in one distribution
is what lets both stacks live in one virtual environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from telemetry_ocr.errors import TelemetryOCRError

from .paths import load_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-lab",
        description="Dataset preparation, YOLO evaluation and environment checks.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser(
        "check", help="Report the torch/paddle stacks and which assets are reachable"
    )
    check.add_argument("--json", action="store_true", help="Emit the raw report as JSON")

    models = commands.add_parser("models", help="List the registered checkpoints")
    models.add_argument("--json", action="store_true")

    extract = commands.add_parser(
        "extract-frames", help="Sample training frames out of the source videos"
    )
    extract.add_argument("--source", type=Path, help="Video directory (default: asset videos dir)")
    extract.add_argument("--output", type=Path, help="Frame directory (default: asset frames dir)")
    extract.add_argument("--interval", type=float, default=1.0, help="Seconds between frames")
    extract.add_argument("--jpeg-quality", type=int, default=95)

    evaluate = commands.add_parser(
        "evaluate", help="Run detection or tracking over a video and write a summary"
    )
    evaluate.add_argument("video", help="Video path, or a bare filename in the asset videos dir")
    evaluate.add_argument("--model", help="Registered name (see `vision-lab models`) or a path")
    evaluate.add_argument("--output", type=Path, help="Run directory root")
    evaluate.add_argument("--name", help="Run name; defaults to the video stem plus a timestamp")
    evaluate.add_argument("--conf", type=float, default=0.45)
    evaluate.add_argument("--iou", type=float, default=0.45)
    evaluate.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference size; match the size the checkpoint was trained at",
    )
    evaluate.add_argument("--max-det", type=int, default=300)
    evaluate.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame")
    evaluate.add_argument("--device", help="e.g. 0, 0,1 or cpu; omit to select automatically")
    evaluate.add_argument("--classes", type=int, nargs="+", help="Keep only these class ids")
    evaluate.add_argument("--track", action="store_true", help="Enable tracking across frames")
    evaluate.add_argument("--tracker", default="botsort.yaml")
    evaluate.add_argument("--show", action="store_true")
    evaluate.add_argument("--half", action="store_true", help="FP16; CUDA only")
    evaluate.add_argument("--agnostic-nms", action="store_true")
    evaluate.add_argument("--save-txt", action="store_true")
    evaluate.add_argument("--save-conf", action="store_true")
    evaluate.add_argument("--no-save-video", action="store_true")
    evaluate.add_argument("--quiet", action="store_true")
    evaluate.add_argument("--log-every", type=int, default=100)

    report = commands.add_parser("report", help="Build the Thai theory and dataset PDF")
    report.add_argument("--output", type=Path, help="Destination PDF path")
    report.add_argument("--run", help="Training run to illustrate (default: visdrone_custom_6th)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = load_paths()

        if args.command == "check":
            from .envcheck import collect, render

            report = collect(paths)
            print(json.dumps(report, indent=2) if args.json else render(report))
            return 0

        if args.command == "models":
            entries = {
                name: {
                    "path": str(path),
                    "present": path.is_file(),
                    "default": name == paths.default_model,
                }
                for name, path in sorted(paths.models.items())
            }
            if args.json:
                print(json.dumps(entries, indent=2))
            else:
                for name, entry in entries.items():
                    marker = "  " if entry["present"] else "! "
                    default = " (default)" if entry["default"] else ""
                    print(f"{marker}{name:<12}{entry['path']}{default}")
                if any(not entry["present"] for entry in entries.values()):
                    print("\n'!' marks a checkpoint listed in configs/paths.yaml but not on disk")
            return 0

        if args.command == "extract-frames":
            from .frames import extract_from_directory

            results = extract_from_directory(
                paths,
                args.source,
                args.output,
                interval_seconds=args.interval,
                jpeg_quality=args.jpeg_quality,
            )
            total = sum(result.saved_images for result in results)
            print(f"\n{total:,} images from {len(results)} videos")
            return 0

        if args.command == "evaluate":
            from .evaluate import EvaluationOptions, evaluate

            options = EvaluationOptions(
                video=args.video,
                model=args.model,
                output=args.output,
                name=args.name,
                confidence=args.conf,
                iou=args.iou,
                image_size=args.imgsz,
                max_detections=args.max_det,
                video_stride=args.vid_stride,
                device=args.device,
                classes=args.classes,
                track=args.track,
                tracker=args.tracker,
                show=args.show,
                half=args.half,
                agnostic_nms=args.agnostic_nms,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                save_video=not args.no_save_video,
                quiet=args.quiet,
                log_every=args.log_every,
            )
            evaluate(paths, options)
            return 0

        if args.command == "report":
            from .report import build_report

            destination = build_report(paths, output=args.output, run_name=args.run)
            print(f"{destination} ({destination.stat().st_size / 1024:.0f} KB)")
            return 0
    except (TelemetryOCRError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
