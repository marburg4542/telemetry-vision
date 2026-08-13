from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .detection import load_ultralytics_detector


@dataclass(frozen=True, slots=True)
class DetectionFrame:
    payload: bytes
    source_id: str
    generation: int
    sequence: int
    source_time_ms: float
    confidence: float
    iou: float
    image_size: int
    reset_tracker: bool = False


class DetectionWorker(Protocol):
    def start(self, model_path: Path, filename: str) -> None: ...

    def restart(self) -> None: ...

    def stop(self, timeout: float = 3.0) -> None: ...

    def status(self) -> dict[str, Any]: ...

    def submit(self, frame: DetectionFrame) -> bool: ...

    def latest(self) -> dict[str, Any] | None: ...

    def reset(self, source_id: str, generation: int) -> None: ...


class ProcessDetectionWorker:
    """Owns an Ultralytics model in an isolated spawn process.

    The input queue is deliberately bounded to one item. Submitting a newer
    frame replaces an older queued frame instead of allowing playback latency
    to grow without bound.
    """

    def __init__(
        self,
        *,
        worker_target: Callable[..., None] | None = None,
        tracker: str = "botsort.yaml",
    ) -> None:
        self._context = mp.get_context("spawn")
        self._worker_target = worker_target or _worker_main
        self._tracker = tracker
        self._process: Any = None
        self._frames: Any = None
        self._results: Any = None
        self._model_path: Path | None = None
        self._filename: str | None = None
        self._state = "unloaded"
        self._error: str | None = None
        self._metadata: dict[str, Any] = {}
        self._latest_result: dict[str, Any] | None = None
        self._dropped_frames = 0
        self._active_source_id = ""
        self._active_generation = 0
        self._pending_reset = False

    def start(self, model_path: Path, filename: str) -> None:
        self.stop()
        self._frames = self._context.Queue(maxsize=1)
        self._results = self._context.Queue(maxsize=4)
        self._model_path = Path(model_path)
        self._filename = filename
        self._state = "loading"
        self._error = None
        self._metadata = {}
        self._latest_result = None
        self._dropped_frames = 0
        self._active_source_id = ""
        self._active_generation = 0
        self._pending_reset = True
        self._process = self._context.Process(
            target=self._worker_target,
            args=(str(self._model_path), self._filename, self._tracker, self._frames, self._results),
            name="telemetry-ocr-yolo",
            daemon=True,
        )
        self._process.start()

    def restart(self) -> None:
        if self._model_path is None or self._filename is None:
            raise ValueError("No YOLO model is available to restart")
        self.start(self._model_path, self._filename)

    def stop(self, timeout: float = 3.0) -> None:
        process = self._process
        frames = self._frames
        if process is not None and process.is_alive():
            if frames is not None:
                put_latest(frames, {"kind": "shutdown"})
            process.join(timeout=max(0.0, timeout))
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        for channel in (self._frames, self._results):
            close = getattr(channel, "close", None)
            if callable(close):
                close()
            join_thread = getattr(channel, "join_thread", None)
            if callable(join_thread):
                join_thread()
        self._process = None
        self._frames = None
        self._results = None
        if self._state != "unloaded":
            self._state = "stopped"

    def status(self) -> dict[str, Any]:
        self._drain_results()
        process = self._process
        if (
            process is not None
            and not process.is_alive()
            and self._state in {"loading", "ready"}
        ):
            self._state = "error"
            self._error = f"Detection worker exited with code {process.exitcode}"
        return {
            "status": self._state,
            "loaded": self._state == "ready",
            "filename": self._filename,
            "worker_pid": process.pid if process is not None and process.is_alive() else None,
            "tracker": "BoT-SORT" if self._tracker == "botsort.yaml" else self._tracker,
            "device": "cpu",
            "error": self._error,
            "dropped_frames": self._dropped_frames,
            **self._metadata,
        }

    def submit(self, frame: DetectionFrame) -> bool:
        self._drain_results()
        if self._state != "ready" or self._frames is None:
            return False
        if (
            frame.source_id != self._active_source_id
            or frame.generation != self._active_generation
        ):
            self.reset(frame.source_id, frame.generation)
        packet = asdict(frame)
        packet["kind"] = "frame"
        packet["reset_tracker"] = bool(frame.reset_tracker or self._pending_reset)
        dropped = put_latest(self._frames, packet)
        self._dropped_frames += dropped
        self._pending_reset = False
        return True

    def latest(self) -> dict[str, Any] | None:
        self._drain_results()
        return self._latest_result

    def reset(self, source_id: str, generation: int) -> None:
        self._active_source_id = source_id
        self._active_generation = generation
        self._latest_result = None
        self._pending_reset = True
        if self._frames is not None:
            self._dropped_frames += drain_queue(self._frames)

    def _drain_results(self) -> None:
        if self._results is None:
            return
        while True:
            try:
                message = self._results.get_nowait()
            except queue.Empty:
                break
            kind = message.get("kind")
            if kind == "ready":
                self._metadata = dict(message.get("metadata") or {})
                self._state = "ready"
                self._error = None
            elif kind == "fatal_error":
                self._state = "error"
                self._error = str(message.get("error") or "Detection worker failed")
            elif kind == "frame_error":
                self._error = str(message.get("error") or "Detection frame failed")
            elif kind == "result":
                result = dict(message["result"])
                if (
                    result.get("source_id") == self._active_source_id
                    and int(result.get("generation", -1)) == self._active_generation
                ):
                    result["dropped_frames"] = self._dropped_frames
                    self._latest_result = result
                    self._error = None


def put_latest(channel: Any, item: Any) -> int:
    """Put without blocking, replacing queued items and returning drop count."""

    dropped = 0
    while True:
        try:
            channel.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                channel.get_nowait()
                dropped += 1
            except queue.Empty:
                # multiprocessing.Queue can briefly report a full semaphore
                # before its feeder makes the item visible to get_nowait().
                time.sleep(0.001)


def drain_queue(channel: Any) -> int:
    removed = 0
    while True:
        try:
            channel.get_nowait()
            removed += 1
        except queue.Empty:
            return removed


def _publish(channel: Any, message: dict[str, Any]) -> None:
    put_latest(channel, message)


def _worker_main(
    model_path: str,
    filename: str,
    tracker: str,
    frames: Any,
    results: Any,
) -> None:
    del filename
    cpu_threads = max(1, int(os.environ.get("TELEMETRY_OCR_YOLO_THREADS", "2")))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    import cv2

    cv2.setNumThreads(cpu_threads)
    try:
        detector = load_ultralytics_detector(model_path)
        _publish(results, {"kind": "ready", "metadata": detector.metadata()})
    except Exception as exc:
        _publish(
            results,
            {"kind": "fatal_error", "error": f"{type(exc).__name__}: {exc}"},
        )
        return

    previous_source = ""
    previous_generation = -1
    previous_source_time_ms: float | None = None
    previous_options: tuple[float, float, int] | None = None
    while True:
        try:
            packet = frames.get(timeout=0.25)
        except queue.Empty:
            continue
        if packet.get("kind") == "shutdown":
            return
        if packet.get("kind") != "frame":
            continue

        source_id = str(packet["source_id"])
        generation = int(packet["generation"])
        source_time_ms = float(packet["source_time_ms"])
        options = (
            float(packet["confidence"]),
            float(packet["iou"]),
            int(packet["image_size"]),
        )
        time_gap = (
            None
            if previous_source_time_ms is None
            else source_time_ms - previous_source_time_ms
        )
        reset = bool(packet.get("reset_tracker")) or any(
            (
                source_id != previous_source,
                generation != previous_generation,
                time_gap is not None and (time_gap <= 0 or time_gap > 1000),
                previous_options is not None and options != previous_options,
            )
        )
        if reset:
            detector.reset_tracking()

        started_at = time.perf_counter()
        try:
            frame = _decode_image(packet["payload"])
            detections = detector.track(
                frame,
                confidence=options[0],
                iou=options[1],
                image_size=options[2],
                tracker=tracker,
            )
            inference_ms = (time.perf_counter() - started_at) * 1000.0
            _publish(
                results,
                {
                    "kind": "result",
                    "result": {
                        "source_id": source_id,
                        "generation": generation,
                        "sequence": int(packet["sequence"]),
                        "source_time_ms": source_time_ms,
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                        "inference_ms": inference_ms,
                        "completed_at": time.time(),
                        "detections": [asdict(detection) for detection in detections],
                    },
                },
            )
            previous_source = source_id
            previous_generation = generation
            previous_source_time_ms = source_time_ms
            previous_options = options
        except Exception as exc:
            _publish(
                results,
                {
                    "kind": "frame_error",
                    "sequence": packet.get("sequence"),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )


def _decode_image(payload: bytes) -> Any:
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Frame is not a supported JPEG/PNG/WebP image")
    return frame
