"""Run a live stream on the server and publish what it sees, for the web UI.

The browser cannot help here. Everywhere else in this application the page owns
the video: it plays the file, grabs frames and posts them, and the server only
ever answers. RTSP is the one source a browser cannot open at all, so for a
stream the roles swap -- the server decodes, runs OCR, and the page becomes a
viewer of results it did not produce.

Everything published here is newest-wins. A viewer that has fallen behind wants
the current state of the aircraft, never a backlog of where it used to be, and
the alternative -- letting a slow or paused browser tab apply back pressure to
the decoder -- would put a delay on OCR that nothing downstream could remove.

Nothing in this module imports torch. Detection frames leave through the
on_frame callback and are dealt with by the caller, which is what keeps the
process that has already loaded PaddlePaddle from ever loading a second cuDNN.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AppConfig
from .ocr import Recognizer
from .output import TelemetryRecord
from .pipeline import FramePacket, PipelineSummary, TelemetryPipeline, is_stream_source

_LOG = logging.getLogger(__name__)

# The viewer is a browser on the same machine, not a recording. Sending full-HD
# JPEGs at the OCR rate would spend bandwidth and encode time on detail no one
# can see in a letterboxed panel, and the overlay is drawn from normalised box
# coordinates, so nothing downstream depends on this size.
_PREVIEW_MAX_WIDTH = 1280
_PREVIEW_JPEG_QUALITY = 80


@dataclass(slots=True)
class StreamSnapshot:
    """What the UI needs to draw one moment of the stream."""

    revision: int = 0
    source_time_ms: float = 0.0
    frame_index: int = 0
    fields: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class StreamService:
    """Owns one live source: decode, OCR, and publish. Not restartable."""

    def __init__(
        self,
        config: AppConfig,
        recognizer: Recognizer,
        url: str,
        *,
        jsonl_path: Path,
        csv_path: Path,
        on_frame: Callable[[FramePacket], None] | None = None,
        on_update: Callable[[], None] | None = None,
    ) -> None:
        if not is_stream_source(url):
            raise ValueError(
                f"Not a stream URL: {url!r}. Expected a scheme such as rtsp://, "
                "rtmp:// or http://"
            )
        self.config = config
        self.recognizer = recognizer
        self.url = url
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path
        self._on_frame = on_frame
        self._on_update = on_update

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot = StreamSnapshot()
        self._preview: bytes | None = None
        self._preview_revision = 0
        self._started_at: float | None = None
        self._error: str | None = None
        self._summary: PipelineSummary | None = None
        self._pipeline = TelemetryPipeline(config, recognizer)

        # The raw frame waiting to be encoded, and the thread that encodes it.
        # Encoding on the OCR thread would add its cost to every measured
        # latency, to serve a preview that is not what this application is for.
        self._pending_frame: Any = None
        # Kept separately from _pending_frame, which the encoder clears as soon
        # as it has a preview. A shot needs the full-resolution frame, and needs
        # it whenever the operator presses the button rather than only in the
        # instant between decode and encode.
        self._last_frame: Any = None
        # The frame the current snapshot was read from. Distinct from
        # _last_frame, which is simply the newest decoded one.
        self._read_frame: Any = None
        self._frame_ready = threading.Event()
        self._encoder = threading.Thread(
            target=self._encode_loop, name="stream-preview", daemon=True
        )
        self._runner = threading.Thread(target=self._run, name="stream-service", daemon=True)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._started_at = time.time()
        self._encoder.start()
        self._runner.start()

    def stop(self, timeout: float = 15.0) -> None:
        """Ask the run to end and wait for it.

        The timeout is not a formality. A decoder blocked on a camera that has
        gone quiet only returns once its read timeout expires, and that is
        deliberately generous, so stopping is allowed to take a moment.
        """
        self._stop.set()
        self._frame_ready.set()
        self._runner.join(timeout=timeout)
        self._encoder.join(timeout=2.0)

    @property
    def running(self) -> bool:
        return self._runner.is_alive()

    # -- what the web layer reads -----------------------------------------

    def snapshot(self) -> StreamSnapshot:
        with self._lock:
            return StreamSnapshot(
                revision=self._snapshot.revision,
                source_time_ms=self._snapshot.source_time_ms,
                frame_index=self._snapshot.frame_index,
                fields=dict(self._snapshot.fields),
                latency_ms=self._snapshot.latency_ms,
            )

    def preview(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._preview, self._preview_revision

    def last_frame(self) -> Any:
        """The frame the published telemetry was read from, full resolution.

        Deliberately not the newest decoded frame. A capture pairs a picture with
        its numbers, and OCR finishes behind the decoder, so the newest frame and
        the newest reading are different moments -- which produced saved captures
        whose telemetry described a frame they did not contain. Falls back to the
        newest decoded frame only before the first reading exists, when there is
        nothing better and no telemetry to contradict.
        """
        with self._lock:
            return self._read_frame if self._read_frame is not None else self._last_frame

    def current_reading(self) -> tuple[Any, StreamSnapshot]:
        """The picture and its telemetry, taken together.

        Two separate calls were not enough. Even with the frame and the snapshot
        published as a pair, a caller reading them one after the other can have a
        new reading land in between and walk away with frame N and snapshot N+1 --
        which is the same mismatch, just narrower.
        """
        with self._lock:
            frame = self._read_frame if self._read_frame is not None else self._last_frame
            return frame, StreamSnapshot(
                revision=self._snapshot.revision,
                source_time_ms=self._snapshot.source_time_ms,
                frame_index=self._snapshot.frame_index,
                fields=dict(self._snapshot.fields),
                latency_ms=self._snapshot.latency_ms,
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            summary = self._summary
            # The summary only exists once the run has returned, which for a
            # live stream is never. While one is up, the decoder's own counters
            # are the live version of the same two numbers -- and reporting a
            # flat zero for the whole life of a stream made a camera that
            # reconnects every twenty seconds look perfectly healthy.
            decoder = self._pipeline.decoder
            if summary is not None:
                reconnects: int = summary.reconnects
                decode_failures: int = summary.decode_failures
            elif decoder is not None:
                reconnects = decoder.reconnects
                decode_failures = decoder.decode_failures
            else:
                reconnects = 0
                decode_failures = 0
            return {
                "url": self.url,
                "running": self.running,
                "error": self._error,
                "started_at": self._started_at,
                "uptime_seconds": (
                    None if self._started_at is None else time.time() - self._started_at
                ),
                "revision": self._snapshot.revision,
                "fields": len(self._snapshot.fields),
                "reconnects": reconnects,
                "decode_failures": decode_failures,
                "update_hz": None if summary is None else summary.update_hz,
                "latency_mean_ms": None if summary is None else summary.latency_mean_ms,
            }

    # -- the run itself ----------------------------------------------------

    def _run(self) -> None:
        try:
            summary = self._pipeline.run(
                self.url,
                self.jsonl_path,
                self.csv_path,
                on_frame=self._handle_frame,
                on_records=self._handle_records,
                stop_event=self._stop,
            )
            with self._lock:
                self._summary = summary
            _LOG.info("Stream ended: %s", self.url)
        except Exception as exc:
            # Reaching the operator matters more than the traceback: this thread
            # has no console, and without this the PWA would show a stream that
            # simply stopped updating for no stated reason.
            _LOG.exception("Stream failed: %s", self.url)
            with self._lock:
                self._error = str(exc)
        finally:
            self._stop.set()
            self._frame_ready.set()
            self._notify()

    def _handle_frame(self, packet: FramePacket) -> None:
        # Newest-wins, and deliberately a reference rather than a copy: the
        # decoder does not reuse frame buffers, and copying full-HD here would
        # charge the OCR thread for the preview a second time.
        with self._lock:
            self._pending_frame = packet.frame
            self._last_frame = packet.frame
        self._frame_ready.set()
        if self._on_frame is not None:
            self._on_frame(packet)

    def _handle_records(
        self, records: Sequence[TelemetryRecord], packet: FramePacket
    ) -> None:
        fields = {
            record.field_id: {
                "value": record.value,
                "raw_text": record.raw_text,
                "confidence": record.confidence,
                "confirmed": record.confirmed,
                # Carried so the stream renders through the same row code as a
                # file, rather than a near-copy that drifts from it.
                "preprocess_variant": record.preprocess_variant,
                "field_id": record.field_id,
            }
            for record in records
        }
        latency = max((record.latency_ms for record in records), default=0.0)
        with self._lock:
            self._snapshot = StreamSnapshot(
                revision=self._snapshot.revision + 1,
                source_time_ms=packet.source_time_ms,
                frame_index=packet.frame_index,
                fields=fields,
                latency_ms=latency,
            )
            # Published together, under one lock, because a capture needs the
            # picture these numbers were read from -- not merely a recent one.
            # Pairing the newest decoded frame with the newest reading looked
            # equivalent and was not: OCR finishes behind the decoder, so the two
            # came from different moments and a saved capture carried telemetry
            # describing a frame it did not contain.
            self._read_frame = packet.frame
        self._notify()

    def _encode_loop(self) -> None:
        import cv2

        while not self._stop.is_set():
            if not self._frame_ready.wait(timeout=0.5):
                continue
            self._frame_ready.clear()
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None:
                continue
            try:
                height, width = frame.shape[:2]
                scale = min(1.0, _PREVIEW_MAX_WIDTH / max(width, 1))
                resized = (
                    frame
                    if scale >= 1.0
                    else cv2.resize(frame, (round(width * scale), round(height * scale)))
                )
                ok, buffer = cv2.imencode(
                    ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), _PREVIEW_JPEG_QUALITY]
                )
                if not ok:
                    continue
                with self._lock:
                    self._preview = buffer.tobytes()
                    self._preview_revision += 1
            except Exception:
                # A preview that cannot be encoded is a picture the operator does
                # not get. It is not a reason to stop extracting telemetry, which
                # is the part of this that matters.
                _LOG.exception("Could not encode a preview frame")
            self._notify()

    def _notify(self) -> None:
        if self._on_update is None:
            return
        try:
            self._on_update()
        except Exception:
            _LOG.exception("Stream update notification failed")
