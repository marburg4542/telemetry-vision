"""Run YOLO on the same frames the OCR pipeline is reading, from the CLI.

The web UI already does this, but it does it by having the browser post frames.
A stream has no browser in the loop, so the frames have to be handed over
in-process -- and the one thing that cannot happen is torch being imported here.
PaddlePaddle is already loaded by the time this runs, and on Windows the second
cuDNN to load fails.

So this reuses ProcessDetectionWorker, exactly as the web server does: the model
lives in a spawned process, only JPEG bytes and plain dicts cross the boundary,
and this side never touches torch.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .detection_worker import DetectionFrame, ProcessDetectionWorker
from .errors import DependencyError, VideoError
from .pipeline import FramePacket

_LOG = logging.getLogger(__name__)

# How long to wait for the spawned process to load the checkpoint before giving
# up. Cold-loading a YOLOv8m onto a GPU is slow enough to need this much room.
_READY_TIMEOUT_S = 120.0


class DetectionSidecar:
    """Feeds frames to a detection worker and writes what comes back."""

    def __init__(
        self,
        model_path: Path,
        output_path: Path,
        *,
        confidence: float = 0.45,
        iou: float = 0.45,
        image_size: int = 640,
        source_id: str = "stream",
    ) -> None:
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.source_id = source_id
        self.submitted = 0
        self.written = 0
        self._sequence = 0
        self._last_written_sequence = -1
        self._worker = ProcessDetectionWorker()
        self._handle: Any = None

    def start(self) -> None:
        if not self.model_path.is_file():
            raise VideoError(f"Checkpoint not found: {self.model_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("w", encoding="utf-8")
        self._worker.start(self.model_path, self.model_path.name)

        deadline = time.monotonic() + _READY_TIMEOUT_S
        while time.monotonic() < deadline:
            status = self._worker.status()
            state = status.get("status")
            if state == "ready":
                _LOG.info(
                    "Detection worker ready on %s: %s",
                    status.get("device"),
                    status.get("classes"),
                )
                return
            if state == "error":
                raise DependencyError(
                    f"Detection worker failed to start: {status.get('error')}"
                )
            time.sleep(0.1)
        raise DependencyError(
            f"Detection worker did not become ready within {_READY_TIMEOUT_S:.0f}s"
        )

    def submit(self, packet: FramePacket) -> None:
        """Hand a frame over. Dropping is normal and handled by the worker."""
        import cv2

        height, width = packet.frame.shape[:2]
        scale = min(1.0, self.image_size / max(width, height))
        frame = (
            packet.frame
            if scale >= 1.0
            else cv2.resize(packet.frame, (round(width * scale), round(height * scale)))
        )
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if not ok:
            _LOG.warning("Could not encode frame %d for detection", packet.frame_index)
            return
        accepted = self._worker.submit(
            DetectionFrame(
                payload=buffer.tobytes(),
                source_id=self.source_id,
                generation=0,
                sequence=self._sequence,
                source_time_ms=packet.source_time_ms,
                confidence=self.confidence,
                iou=self.iou,
                image_size=self.image_size,
            )
        )
        if accepted:
            self._sequence += 1
            self.submitted += 1

    def drain(self) -> None:
        """Collect whatever has finished and append it.

        Nothing here waits for a particular frame. The worker's queue is
        newest-wins, so asking for a specific one would mean blocking the OCR
        loop on a result that may have been superseded and discarded.
        """
        self._worker.drain()
        latest = self._worker.latest()
        if latest is None:
            return
        sequence = int(latest.get("sequence", -1))
        if sequence <= self._last_written_sequence:
            return
        self._last_written_sequence = sequence
        record = {
            "source_time_ms": latest.get("source_time_ms"),
            "sequence": sequence,
            "width": latest.get("width"),
            "height": latest.get("height"),
            "inference_ms": latest.get("inference_ms"),
            "detections": latest.get("detections") or [],
        }
        assert self._handle is not None
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.written += 1

    def close(self) -> dict[str, Any]:
        status = {}
        try:
            self.drain()
            status = dict(self._worker.status())
        except Exception:
            _LOG.exception("Detection sidecar failed while shutting down")
        finally:
            self._worker.stop()
            if self._handle is not None:
                self._handle.close()
                self._handle = None
        return {
            "submitted": self.submitted,
            "written": self.written,
            "dropped_frames": status.get("dropped_frames", 0),
            "frame_errors": status.get("frame_errors", 0),
            "output": str(self.output_path),
        }
