from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .detection import load_ultralytics_detector

_LOG = logging.getLogger(__name__)


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

    def drain(self) -> None: ...


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
        self._frame_error: str | None = None
        self._frame_errors = 0
        # Monotonic pair rather than a flag; see submit().
        self._reset_requests = 0
        self._resets_sent = 0

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
        self._frame_error = None
        self._frame_errors = 0
        self._reset_requests += 1
        self._process = self._context.Process(
            target=self._worker_target,
            args=(
                str(self._model_path), self._filename, self._tracker,
                self._frames, self._results,
            ),
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
        results = self._results
        # Detached before anything is closed. A reader that is already inside
        # _drain_results would otherwise call get_nowait() on a queue this
        # thread has just closed, which raises from under it.
        self._frames = None
        self._results = None

        if process is not None and process.is_alive():
            if frames is not None:
                put_latest(frames, {"kind": "shutdown"}, deadline_s=1.0)
            process.join(timeout=max(0.0, timeout))
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                # terminate() is a signal a native library can sit through --
                # a long CUDA call, an NMS kernel. Without this last step the
                # only handle to the process is dropped below and it keeps the
                # GPU allocation and an open handle on the checkpoint for the
                # rest of the server's life, which on Windows is also what makes
                # the temp directory refuse to delete.
                _LOG.warning("Detection worker ignored terminate(); killing it")
                process.kill()
                process.join(timeout=1.0)

        for channel in (frames, results):
            close = getattr(channel, "close", None)
            if callable(close):
                close()
            join_thread = getattr(channel, "join_thread", None)
            if callable(join_thread):
                join_thread()

        self._process = None
        # Everything below described the model that has just gone away. Leaving
        # it in place made status() keep advertising the removed model's classes
        # and device, and latest() keep serving its final detection.
        self._latest_result = None
        self._metadata = {}
        self._dropped_frames = 0
        self._resets_sent = self._reset_requests
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
            # Replaced by the worker's real device once it reports ready; until
            # then this is only what was requested.
            "device": os.environ.get("TELEMETRY_OCR_YOLO_DEVICE", "auto"),
            "error": self._error,
            "frame_error": self._frame_error,
            "frame_errors": self._frame_errors,
            "dropped_frames": self._dropped_frames,
            **self._metadata,
        }

    def submit(self, frame: DetectionFrame) -> bool:
        # Deliberately does not drain. submit() runs in a worker thread via
        # asyncio.to_thread, so draining here mutated _latest_result, _state and
        # _metadata while handlers read them on the event loop. The background
        # pump owns collection now, and it runs on the loop.
        if self._state != "ready" or self._frames is None:
            return False
        if (
            frame.source_id != self._active_source_id
            or frame.generation != self._active_generation
        ):
            self.reset(frame.source_id, frame.generation)
        packet = asdict(frame)
        packet["kind"] = "frame"
        # A counter, not a flag. submit() runs in a thread while reset() is
        # called from the event loop, and with a boolean a reset that arrived
        # mid-submit was overwritten by the `= False` that followed it — the
        # frame after a seek then went out without reset_tracker and BoT-SORT
        # carried its identities across the cut. Consuming the value that was
        # actually observed cannot lose a reset that arrives later.
        observed_resets = self._reset_requests
        packet["reset_tracker"] = bool(frame.reset_tracker or observed_resets > self._resets_sent)
        # Stamped in this process's clock and echoed back untouched by the
        # worker, so the round trip can be measured without the two processes
        # ever having to agree on an epoch.
        packet["enqueued_at"] = time.perf_counter()
        dropped = put_latest(self._frames, packet)
        self._dropped_frames += dropped
        self._resets_sent = observed_resets
        return True

    def latest(self) -> dict[str, Any] | None:
        self._drain_results()
        return self._latest_result

    def drain(self) -> None:
        """Collect whatever the worker has published.

        Public so that a background task can keep the IPC queue empty. Without
        one, a finished result waits in the queue until the next HTTP request
        happens to call status(), latest() or submit() — and the queue holds
        only four, after which the worker silently discards the oldest.
        """
        self._drain_results()

    def reset(self, source_id: str, generation: int) -> None:
        self._active_source_id = source_id
        self._active_generation = generation
        self._latest_result = None
        self._reset_requests += 1
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
                _LOG.info("Detection worker ready: %s", self._metadata.get("classes"))
            elif kind == "fatal_error":
                self._state = "error"
                self._error = str(message.get("error") or "Detection worker failed")
                _LOG.error("Detection worker failed to start: %s", self._error)
            elif kind == "frame_error":
                # Kept apart from _error, which describes the worker itself. A
                # frame failure left there was erased by the very next successful
                # result — often in this same drain, since messages arrive in
                # order — so a source failing every frame reported nothing at all
                # and the overlay just froze.
                self._frame_error = str(message.get("error") or "Detection frame failed")
                self._frame_errors += 1
                _LOG.warning("Detection frame failed: %s", self._frame_error)
            elif kind == "result":
                result = dict(message["result"])
                if (
                    result.get("source_id") == self._active_source_id
                    and int(result.get("generation", -1)) == self._active_generation
                ):
                    result["dropped_frames"] = self._dropped_frames
                    # Both ends of this span were taken in this process, so the
                    # echoed enqueued_at is comparable here and nowhere else.
                    available_at = time.perf_counter()
                    result["available_at"] = available_at
                    result["pipeline_ms"] = (
                        available_at - float(result.get("enqueued_at", available_at))
                    ) * 1000.0
                    self._latest_result = result
                    self._error = None
                    self._frame_error = None


def put_latest(channel: Any, item: Any, deadline_s: float = 2.0) -> int:
    """Put without blocking, replacing queued items and returning drop count.

    The deadline is not a nicety. The retry below exists for a queue that
    briefly reports full before its feeder thread makes the item visible, but
    the same state is indistinguishable from a queue closed by another thread,
    or a child that died mid-transfer leaving a partial message in the pipe.
    Without a bound this spins at a thousand iterations a second forever — and
    stop() calls it, so an unbounded spin there hangs application shutdown and
    burns an executor thread permanently.
    """
    dropped = 0
    expires_at = time.monotonic() + deadline_s
    while True:
        try:
            channel.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                channel.get_nowait()
                dropped += 1
                continue
            except queue.Empty:
                pass
            if time.monotonic() >= expires_at:
                _LOG.warning("Gave up enqueuing after %.1fs; the queue never drained", deadline_s)
                return dropped
            time.sleep(0.001)
        except (OSError, ValueError):
            # The queue was closed underneath us, which stop() can do.
            return dropped


def drain_queue(channel: Any) -> int:
    removed = 0
    while True:
        try:
            channel.get_nowait()
            removed += 1
        except queue.Empty:
            return removed


def _publish(channel: Any, message: dict[str, Any]) -> None:
    """Publish a message, never letting a result evict a state change.

    Results are newest-wins by design: an old detection is worthless. `ready`,
    `fatal_error` and `frame_error` are not — dropping one leaves the parent
    permanently unaware of what the worker is doing. So control messages wait
    for room instead of shoving something out of the way.
    """
    if message.get("kind") == "result":
        put_latest(channel, message)
        return
    try:
        channel.put(message, timeout=2.0)
    except queue.Full:
        _LOG.error("Could not publish %s; the result queue never drained", message.get("kind"))
    except (OSError, ValueError):
        pass


def _worker_main(
    model_path: str,
    filename: str,
    tracker: str,
    frames: Any,
    results: Any,
) -> None:
    del filename
    # A spawned process starts with no logging configuration, so anything this
    # side reports would otherwise be discarded.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s yolo-worker %(message)s",
    )
    cpu_threads = max(1, int(os.environ.get("TELEMETRY_OCR_YOLO_THREADS", "2")))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    import cv2

    cv2.setNumThreads(cpu_threads)
    try:
        detector = load_ultralytics_detector(model_path)
        _publish(results, {"kind": "ready", "metadata": detector.metadata()})
    except Exception as exc:
        _LOG.exception("Cannot load the YOLO checkpoint")
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
        dequeued_at = time.perf_counter()
        if packet.get("kind") == "shutdown":
            return
        if packet.get("kind") != "frame":
            continue

        # Unpacking and the tracker reset used to sit outside the try below, so
        # a malformed packet or a tracker that objected killed this loop without
        # publishing anything — the parent only found out via the exit code.
        try:
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
        except Exception as exc:
            _LOG.exception("Rejected a malformed detection packet")
            _publish(results, {"kind": "frame_error", "error": f"{type(exc).__name__}: {exc}"})
            continue

        started_at = time.perf_counter()
        try:
            frame = _decode_image(packet["payload"])
            decoded_at = time.perf_counter()
            detections = detector.track(
                frame,
                confidence=options[0],
                iou=options[1],
                image_size=options[2],
                tracker=tracker,
            )
            finished_at = time.perf_counter()
            decode_ms = (decoded_at - started_at) * 1000.0
            track_ms = (finished_at - decoded_at) * 1000.0
            # Unchanged meaning so the status panel and the README timings still
            # describe the same span; decode_ms and track_ms break it down.
            inference_ms = (finished_at - started_at) * 1000.0
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
                        "decode_ms": decode_ms,
                        "track_ms": track_ms,
                        "worker_total_ms": (finished_at - dequeued_at) * 1000.0,
                        # Opaque to this process: the parent's clock, echoed so
                        # it can measure the full round trip on its own side.
                        "enqueued_at": float(packet.get("enqueued_at", 0.0)),
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
            _LOG.exception("Detection failed on sequence %s", packet.get("sequence"))
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
