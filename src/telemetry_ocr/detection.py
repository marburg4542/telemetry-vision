from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import DependencyError


@dataclass(frozen=True, slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str
    track_id: int | None = None


class Detector(Protocol):
    @property
    def names(self) -> Sequence[str] | dict[int, str]: ...

    def detect(
        self,
        frame: Any,
        *,
        confidence: float,
        iou: float,
        image_size: int,
    ) -> list[Detection]: ...

    def track(
        self,
        frame: Any,
        *,
        confidence: float,
        iou: float,
        image_size: int,
        tracker: str,
    ) -> list[Detection]: ...

    def reset_tracking(self) -> None: ...

    def metadata(self) -> dict[str, Any]: ...


def configure_ultralytics_home() -> Path:
    """Point Ultralytics at a writable settings directory inside the project.

    Ultralytics otherwise drops a settings folder wherever the process happens
    to start, which is how a stray empty `Ultralytics/` directory kept appearing
    at the repository root. Both the OCR detection worker and the offline
    evaluation CLI call this before importing the package.
    """
    configured_dir = os.environ.get("TELEMETRY_OCR_YOLO_CONFIG_DIR")
    config_dir = (
        Path(configured_dir) if configured_dir else Path.cwd() / "output" / "ultralytics-config"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    return config_dir


def resolve_torch_device(torch: Any, requested: str | None) -> str:
    """Pick the device YOLO runs on, failing loudly when CUDA was asked for and is absent.

    Silently falling back to the CPU would turn a misconfiguration into a
    tenfold latency regression that only shows up as dropped frames.
    """
    choice = (requested or os.environ.get("TELEMETRY_OCR_YOLO_DEVICE", "auto")).strip().lower()
    if choice in {"", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if choice.startswith("cuda") and not torch.cuda.is_available():
        raise DependencyError(
            f"YOLO device {choice!r} was requested but this PyTorch build reports no CUDA "
            "device. Install the CUDA build of torch, or set TELEMETRY_OCR_YOLO_DEVICE=cpu."
        )
    return choice


class UltralyticsDetector:
    def __init__(self, model_path: str | Path, device: str | None = None):
        configure_ultralytics_home()
        try:
            import torch
            from ultralytics import YOLO
            from ultralytics.utils import torch_utils
        except ImportError as exc:
            raise DependencyError(
                "Ultralytics YOLO is not installed; run `pip install -e .[ai]`."
            ) from exc

        self.device = resolve_torch_device(torch, device)

        # PaddleOCR remains the primary realtime workload. Limiting PyTorch to
        # two CPU threads prevents a large YOLO model from occupying every core
        # and starving telemetry recognition. This still applies on CUDA, where
        # letterboxing and NMS run on the CPU. The value remains configurable
        # for machines with more cores or a dedicated GPU.
        try:
            cpu_threads = max(
                1, int(os.environ.get("TELEMETRY_OCR_YOLO_THREADS", "2"))
            )
            # select_device() reapplies the module-level Ultralytics default on
            # the first prediction, so update both values.
            torch_utils.NUM_THREADS = cpu_threads
            torch.set_num_threads(cpu_threads)
        except ValueError as exc:
            raise ValueError("TELEMETRY_OCR_YOLO_THREADS must be an integer") from exc

        # Checkpoints produced by older Ultralytics releases contain model
        # objects, while PyTorch 2.6+ defaults torch.load to weights_only=True.
        # Scope the compatibility override to model construction only.
        original_torch_load = torch.load

        def compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = compatible_torch_load
        try:
            self.model = YOLO(str(model_path))
        finally:
            torch.load = original_torch_load

    @property
    def names(self) -> Sequence[str] | dict[int, str]:
        return self.model.names

    def detect(
        self,
        frame: Any,
        *,
        confidence: float,
        iou: float,
        image_size: int,
    ) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            max_det=300,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def track(
        self,
        frame: Any,
        *,
        confidence: float,
        iou: float,
        image_size: int,
        tracker: str = "botsort.yaml",
    ) -> list[Detection]:
        results = self.model.track(
            source=frame,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            max_det=300,
            persist=True,
            tracker=tracker,
            device=self.device,
            # `half` is intentionally not passed. Ultralytics 8.4 deprecated it
            # in favour of `quantize` and warns once per call whatever the
            # value, which was 17k lines of log in one session; the default is
            # already FP32. Benchmarked on the GTX 1650 at 640 px: FP32 averages
            # 30.7 ms against FP16's 164.1 ms, same 4 tracks and one box of
            # difference over 256 frames. The card has no tensor cores and
            # emulates half precision, so FP32 is the fast path, not a stopgap.
            verbose=False,
        )
        return self._parse_results(results)

    def reset_tracking(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()) or ():
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def metadata(self) -> dict[str, Any]:
        checkpoint = getattr(self.model, "ckpt", None) or {}
        train_args = checkpoint.get("train_args", {}) or {}
        trained_size = train_args.get("imgsz")
        if isinstance(trained_size, (list, tuple)):
            trained_size = max(int(value) for value in trained_size)
        return {
            "classes": detector_names(self),
            "architecture": Path(str(train_args.get("model", "YOLO"))).stem,
            "epochs": _optional_int(train_args.get("epochs")),
            "trained_image_size": _optional_int(trained_size),
            "task": str(getattr(self.model, "task", "detect")),
            "device": self.device,
        }

    def _parse_results(self, results: Any) -> list[Detection]:
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.int().detach().cpu().tolist()
        track_ids = (
            boxes.id.int().detach().cpu().tolist()
            if boxes.id is not None
            else [None] * len(class_ids)
        )
        return [
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(score),
                class_id=int(class_id),
                label=_class_name(self.names, int(class_id)),
                track_id=None if track_id is None else int(track_id),
            )
            for (x1, y1, x2, y2), score, class_id, track_id in zip(
                coordinates, confidences, class_ids, track_ids, strict=True
            )
        ]


def load_ultralytics_detector(model_path: str | Path) -> UltralyticsDetector:
    return UltralyticsDetector(model_path)


def detector_names(detector: Detector) -> list[str]:
    names = detector.names
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    return [str(name) for name in names]


def _class_name(names: Sequence[str] | dict[int, str], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
