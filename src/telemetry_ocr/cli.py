from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__
from .config import clone_config, dump_config, load_config
from .errors import TelemetryOCRError
from .pipeline import is_stream_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="telemetry-ocr",
        description="Extract fixed-position telemetry from video with PaddleOCR.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-config", help="Validate ROI configuration only")
    validate.add_argument("--config", required=True, type=Path)

    run = commands.add_parser(
        "run", help="Run real-time telemetry extraction on a file or a live stream"
    )
    # A string, not a Path: an rtsp:// URL survives Path() on POSIX but has its
    # double slash collapsed on Windows, which produces a URL the backend cannot
    # open and an error that blames the camera.
    run.add_argument(
        "--video",
        required=True,
        help="Video file, or a stream URL such as rtsp://user:pass@host:554/stream",
    )
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--jsonl", type=Path, default=Path("output/telemetry.jsonl"))
    run.add_argument("--csv", type=Path, default=Path("output/telemetry.csv"))
    run.add_argument(
        "--no-realtime",
        action="store_true",
        help=(
            "Decode a file as fast as possible; stale scheduled frames are still "
            "dropped. Ignored for streams, which already arrive at their own rate"
        ),
    )
    run.add_argument(
        "--max-seconds",
        type=float,
        help="Stop after this long. A stream otherwise runs until Ctrl+C",
    )
    run.add_argument(
        "--detect-model",
        help=(
            "Also run YOLO on the same frames: a name from configs/paths.yaml or a "
            "path to a .pt checkpoint. Runs in a separate process, as the web UI does"
        ),
    )
    run.add_argument(
        "--detections",
        type=Path,
        default=Path("output/detections.jsonl"),
        help="Where detections go when --detect-model is given",
    )
    run.add_argument("--detect-confidence", type=float, default=0.45)
    run.add_argument("--detect-iou", type=float, default=0.45)
    run.add_argument(
        "--detect-image-size",
        type=int,
        default=640,
        help="Match the size the checkpoint was trained at",
    )

    web = commands.add_parser(
        "web", help="Open the local browser UI for interactive video OCR"
    )
    web.add_argument("--config", required=True, type=Path)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument(
        "--stream",
        help=(
            "Open this stream on startup and keep it open, so an auto-started "
            "server needs no one present. Retried every 30s while it is down"
        ),
    )
    web.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the web UI in the default browser after the OCR server starts",
    )
    web.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help=(
            "Allow a trusted web app origin to call the OCR API; may be repeated "
            "(loopback origins are always allowed)"
        ),
    )

    calibrate_cmd = commands.add_parser(
        "calibrate", help="Select the best preprocessing recipes per ROI"
    )
    calibrate_cmd.add_argument("--video", required=True, type=Path)
    calibrate_cmd.add_argument("--config", required=True, type=Path)
    calibrate_cmd.add_argument("--output-config", required=True, type=Path)
    calibrate_cmd.add_argument("--report", type=Path, default=Path("output/calibration.json"))
    calibrate_cmd.add_argument("--labels", type=Path)
    calibrate_cmd.add_argument("--samples", type=int, default=30)

    export = commands.add_parser(
        "export-crops", help="Export uniformly sampled ROI crops for manual labeling"
    )
    export.add_argument("--video", required=True, type=Path)
    export.add_argument("--config", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument("--crop-count", type=int, default=1000)

    preview = commands.add_parser(
        "preview-rois", help="Save one video frame annotated with configured ROI boxes"
    )
    preview.add_argument("--video", required=True, type=Path)
    preview.add_argument("--config", required=True, type=Path)
    preview.add_argument("--output", required=True, type=Path)
    preview.add_argument("--frame-index", type=int, default=0)

    benchmark = commands.add_parser("benchmark", help="Benchmark PaddleOCR CPU thread counts")
    benchmark.add_argument("--video", required=True, type=Path)
    benchmark.add_argument("--config", required=True, type=Path)
    benchmark.add_argument("--threads", default="2,4,6")
    benchmark.add_argument("--iterations", type=int, default=30)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--frame-index", type=int, default=0)
    benchmark.add_argument("--report", type=Path, default=Path("output/benchmark.json"))
    benchmark.add_argument("--update-config", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(f"Configuration OK: {len(config.fields)} fields at {config.runtime.ocr_hz:g} Hz")
            return 0

        from .ocr import create_recognizer

        config = load_config(args.config)
        # A stream URL has nothing to check on disk; whether it is reachable is
        # the decoder's answer to give, with the backend's own message.
        if (
            hasattr(args, "video")
            and not is_stream_source(args.video)
            and not Path(args.video).is_file()
        ):
            raise ValueError(f"Video file not found: {args.video}")
        if args.command == "preview-rois":
            from .preview import save_roi_preview

            destination = save_roi_preview(
                args.video, config, args.output, frame_index=args.frame_index
            )
            print(f"ROI preview saved: {destination}")
            return 0
        if args.command == "export-crops":
            from .calibration import export_validation_crops

            manifest = export_validation_crops(
                args.video, config, args.output_dir, crop_count=args.crop_count
            )
            print(f"Validation crops exported; fill expected values in: {manifest}")
            return 0
        if args.command == "run":
            from .pipeline import TelemetryPipeline

            if args.no_realtime:
                config = clone_config(config)
                config.runtime.realtime = False
            if args.max_seconds is not None and args.max_seconds <= 0:
                parser.error("--max-seconds must be positive")

            sidecar = None
            if args.detect_model:
                from .detect_sidecar import DetectionSidecar

                sidecar = DetectionSidecar(
                    _resolve_checkpoint(args.detect_model),
                    args.detections,
                    confidence=args.detect_confidence,
                    iou=args.detect_iou,
                    image_size=args.detect_image_size,
                )
                # Started before the recognizer so a bad checkpoint fails before
                # PaddleOCR spends time loading its own model.
                sidecar.start()

            recognizer = create_recognizer(config.runtime)
            if is_stream_source(args.video):
                print(f"Reading stream: {args.video}")
                if args.max_seconds is None:
                    print("Press Ctrl+C to stop.")
            try:
                summary = TelemetryPipeline(config, recognizer).run(
                    args.video,
                    args.jsonl,
                    args.csv,
                    max_seconds=args.max_seconds,
                    on_frame=None if sidecar is None else _detect_step(sidecar),
                )
            finally:
                if sidecar is not None:
                    print(json.dumps(sidecar.close(), indent=2, ensure_ascii=False))
            print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
            return 0
        if args.command == "web":
            from .webapp import run_web_app

            if not 1 <= args.port <= 65535:
                parser.error("--port must be between 1 and 65535")
            recognizer = create_recognizer(config.runtime)
            if args.stream and not is_stream_source(args.stream):
                parser.error("--stream must be a URL such as rtsp://host:554/path")
            run_web_app(
                config,
                recognizer,
                host=args.host,
                port=args.port,
                open_browser=args.open_browser,
                stream_url=args.stream,
                allowed_origins=args.allow_origin,
            )
            return 0
        if args.command == "calibrate":
            from .calibration import calibrate

            if args.samples <= 0:
                parser.error("--samples must be positive")
            recognizer = create_recognizer(config.runtime)
            report = calibrate(
                args.video,
                config,
                recognizer,
                args.output_config,
                args.report,
                labels_path=args.labels,
                sample_count=args.samples,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        if args.command == "benchmark":
            from .calibration import benchmark_threads

            thread_counts = _parse_threads(args.threads)
            if args.iterations <= 0 or args.warmup < 0 or args.frame_index < 0:
                parser.error(
                    "iterations must be positive; warmup and frame-index cannot be negative"
                )
            report, best = benchmark_threads(
                args.video,
                config,
                create_recognizer,
                thread_counts,
                args.iterations,
                args.warmup,
                args.frame_index,
            )
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if args.update_config:
                config.runtime.cpu_threads = best
                dump_config(config, args.config)
            print(json.dumps(report, indent=2))
            return 0
    except (TelemetryOCRError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


def _detect_step(sidecar: Any) -> Callable[[Any], None]:
    """Submit, then collect, once per scheduled frame.

    Deliberately not a wait: the worker's input queue is newest-wins, so a frame
    submitted now may be superseded before it is ever inferred. Draining what has
    already finished keeps the OCR loop free-running while still emptying the
    result queue often enough that the worker never discards a finished result.
    """

    def step(packet: Any) -> None:
        sidecar.submit(packet)
        sidecar.drain()

    return step


def _resolve_checkpoint(reference: str) -> Path:
    """Accept a path, or a short name from the vision_lab registry.

    Importing vision_lab.paths is safe here -- it reads YAML and touches no ML
    framework -- and it means `--detect-model best50` works the same way it does
    for `vision-lab evaluate`.
    """
    direct = Path(reference)
    if direct.is_file():
        return direct
    try:
        from vision_lab.paths import load_paths

        return load_paths().resolve_model(reference)
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve checkpoint {reference!r}: not a file, and the "
            f"configs/paths.yaml registry could not resolve it ({exc})"
        ) from exc


def _parse_threads(value: str) -> list[int]:
    try:
        result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError("--threads must be a comma-separated list of integers") from exc
    if not result or any(item <= 0 for item in result):
        raise ValueError("--threads values must be positive")
    return result
