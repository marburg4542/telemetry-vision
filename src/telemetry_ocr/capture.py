"""Write frames to disk first, so OCR can be done later without a deadline.

The realtime pipeline this project grew out of had to answer while the video was
still playing, and it paid for that in two ways that cost accuracy: a frame OCR
could not keep up with was dropped, and every reading had to come from a single
pass whose cost was capped by the frame interval.

Capturing first removes both. Nothing is dropped -- the decoder is made to wait
when the writer falls behind, because a hole in the record is permanent and no
later processing can fill it -- and the frames sit on disk afterwards, where a
slower and more thorough read of them costs only time.

What is written is a session: the frames themselves, and a manifest naming each
one with the moment it came from. The manifest is what makes the telemetry
produced later checkable against the picture it was read from, which a realtime
log has no way to offer.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .errors import TelemetryOCRError, VideoError
from .pipeline import _END, FramePacket, VideoDecoder, is_stream_source

_LOG = logging.getLogger(__name__)

# The manifest and the frame directory. Fixed names: a session is meant to be
# handed to `extract` by directory, not by remembering what its parts were
# called.
MANIFEST_NAME = "frames.jsonl"
SESSION_NAME = "session.json"
FRAME_DIR = "frames"

# How many decoded frames may wait to be written. Small on purpose: this is the
# back pressure that makes a file source lossless, and a deep buffer would only
# delay the moment the decoder has to slow down while costing memory in full
# frames.
_WRITE_QUEUE_DEPTH = 8

# JPEG at quality 95, on measurement rather than instinct.
#
# The instinct was PNG: JPEG ringing lands on thin high-contrast glyphs, which
# is exactly what an OSD is made of, so lossless looked like the safe default
# for a project whose whole purpose is reading them. Measured against 195 frames
# of real footage, quality 95 produced *identical* readings to PNG on all 2,340
# values -- not similar, identical -- while capturing at 23.8 Hz instead of
# 7.3 Hz and using 74 MB instead of 238 MB.
#
# That speed is not a nicety. PNG encoding is slower than realtime here, which a
# file source absorbs by making the decoder wait, but a live camera does not
# wait: capturing a stream to PNG would lose frames in the backend's buffer,
# which is the exact failure this rearrangement exists to avoid.
#
# PNG remains available for footage where the overlay is smaller, dimmer or on a
# busier background, where the trade may come out differently.
_DEFAULT_FORMAT = "jpeg"
_FORMATS = ("png", "jpeg")


@dataclass(slots=True)
class CaptureSummary:
    session_dir: str
    source: str
    frames: int
    dropped_frames: int
    bytes_written: int
    wall_time_seconds: float
    capture_hz: float
    reconnects: int = 0
    decode_failures: int = 0
    write_errors: int = 0


@dataclass(slots=True)
class FrameRecord:
    """One captured frame, and enough about it to line telemetry up later."""

    frame_index: int
    file: str
    source_time_ms: float
    captured_at: str
    width: int
    height: int
    bytes: int
    # Present when the shot was taken through the web UI with a detector loaded:
    # what YOLO counted in this frame, recorded at the time so the numbers belong
    # to the picture. A whole-span capture from the CLI leaves it empty.
    detections: dict[str, Any] | None = None


@dataclass(slots=True)
class SessionInfo:
    source: str
    is_stream: bool
    started_at: str
    capture_hz: float
    image_format: str
    frame_count: int = 0
    config_fields: list[str] = field(default_factory=list)


class _FrameWriter:
    """Encodes and writes frames on its own thread.

    Encoding a full-HD PNG costs more than the interval between frames, so doing
    it inline would make the decoder the slower party for the wrong reason and
    would show up as capture rate rather than as disk speed.
    """

    def __init__(self, frame_dir: Path, image_format: str, jpeg_quality: int) -> None:
        self.frame_dir = frame_dir
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.queue: queue.Queue[tuple[int, Any, float] | None] = queue.Queue(
            maxsize=_WRITE_QUEUE_DEPTH
        )
        self.records: list[FrameRecord] = []
        self.bytes_written = 0
        self.write_errors = 0
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="frame-writer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, index: int, frame: Any, source_time_ms: float) -> None:
        self._enqueue((index, frame, source_time_ms))

    def close(self) -> None:
        if self._thread.is_alive():
            self._enqueue(None)
        self._thread.join()
        if self.error:
            raise self.error

    def _enqueue(self, item: tuple[int, Any, float] | None) -> None:
        """Hand one item to the writer, or say why it cannot be handed over.

        A bare `queue.put` on a bounded queue blocks forever once the consumer
        is gone, and this thread's consumer can die: the per-frame handler below
        covers imwrite, but not the cv2 import, not frame.shape, and not the
        append. When that happened the ninth submit blocked, the deadline and
        max-frames checks in capture_session never ran again, stop_event was
        never set, and the capture looked alive from the outside forever.

        This is RecordWriter._enqueue in output.py, which has had the same
        structure and this same guard from the start. The `error` attribute here
        was declared and then never written or read.
        """
        while True:
            if self.error:
                raise self.error
            if not self._thread.is_alive():
                raise RuntimeError("Frame writer stopped unexpectedly")
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        try:
            self._write_frames()
        except Exception as exc:
            # Reported rather than lost. Everything outside the per-frame
            # handler below -- the import, the shape read, the append -- used to
            # kill this thread silently.
            _LOG.exception("Frame writer stopped")
            self.error = exc

    def _write_frames(self) -> None:
        import cv2

        params = (
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            if self.image_format == "jpeg"
            else [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        )
        suffix = "jpg" if self.image_format == "jpeg" else "png"
        while True:
            item = self.queue.get()
            if item is None:
                return
            index, frame, source_time_ms = item
            name = f"{index:08d}.{suffix}"
            path = self.frame_dir / name
            try:
                if not cv2.imwrite(str(path), frame, params):
                    raise VideoError(f"OpenCV refused to write {path}")
                size = path.stat().st_size
            except Exception:
                # One unwritable frame is a gap, not a reason to abandon a
                # capture that may already be minutes long. It is counted, and
                # the summary reports it.
                _LOG.exception("Could not write frame %d", index)
                self.write_errors += 1
                continue
            height, width = frame.shape[:2]
            self.bytes_written += size
            self.records.append(
                FrameRecord(
                    frame_index=index,
                    file=f"{FRAME_DIR}/{name}",
                    source_time_ms=source_time_ms,
                    captured_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    width=int(width),
                    height=int(height),
                    bytes=int(size),
                )
            )


def capture_session(
    video_path: str | Path,
    session_dir: str | Path,
    config: AppConfig,
    *,
    capture_hz: float | None = None,
    max_seconds: float | None = None,
    max_frames: int | None = None,
    image_format: str = _DEFAULT_FORMAT,
    jpeg_quality: int = 95,
    on_progress: Any = None,
) -> CaptureSummary:
    """Record frames and a manifest into session_dir.

    capture_hz defaults to the config's OCR rate, so a session captured with the
    shipped settings holds exactly the frames the realtime pipeline would have
    tried to read -- which makes the two comparable.
    """
    if image_format not in _FORMATS:
        raise TelemetryOCRError(f"image_format must be one of {_FORMATS}, not {image_format!r}")
    if not 1 <= jpeg_quality <= 100:
        raise TelemetryOCRError("jpeg_quality must be between 1 and 100")

    rate = config.runtime.ocr_hz if capture_hz is None else capture_hz
    if rate <= 0:
        raise TelemetryOCRError("capture_hz must be positive")

    session_dir = Path(session_dir)
    frame_dir = session_dir / FRAME_DIR
    frame_dir.mkdir(parents=True, exist_ok=True)
    if any(frame_dir.iterdir()):
        raise TelemetryOCRError(
            f"{frame_dir} already holds frames. Capturing into it would interleave "
            "two recordings under one manifest; choose an empty directory."
        )

    stream = is_stream_source(video_path)
    info = SessionInfo(
        source=str(video_path),
        is_stream=stream,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        capture_hz=rate,
        image_format=image_format,
        config_fields=[item.field_id for item in config.fields],
    )

    frame_queue: queue.Queue[FramePacket | object] = queue.Queue(maxsize=_WRITE_QUEUE_DEPTH)
    stop_event = threading.Event()
    decoder = VideoDecoder(
        video_path,
        frame_queue,
        rate,
        config.runtime.realtime,
        stop_event,
        # The whole point. See VideoDecoder.drop_when_full.
        drop_when_full=False,
    )
    writer = _FrameWriter(frame_dir, image_format, jpeg_quality)

    started = time.perf_counter()
    deadline = None if max_seconds is None else started + max_seconds
    captured = 0
    writer.start()
    decoder.start()
    caught: BaseException | None = None
    try:
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if max_frames is not None and captured >= max_frames:
                break
            try:
                item = frame_queue.get(timeout=0.25)
            except queue.Empty:
                if decoder.error is not None or not decoder.thread.is_alive():
                    break
                continue
            try:
                if item is _END:
                    break
                assert isinstance(item, FramePacket)
                writer.submit(captured, item.frame, item.source_time_ms)
                captured += 1
                if on_progress is not None and captured % 25 == 0:
                    on_progress(captured, writer.bytes_written)
            finally:
                frame_queue.task_done()
    except KeyboardInterrupt:
        # A stream has no other ending, and a session interrupted halfway is
        # still a session: the manifest below is written on every path.
        _LOG.info("Capture stopped on interrupt")
    except BaseException as exc:
        caught = exc
    finally:
        stop_event.set()
        decoder.join()
        try:
            writer.close()
        except Exception as exc:
            # close() re-raises whatever killed the writer thread, and letting
            # that escape from a finally would skip the manifest below and undo
            # the reason it was moved up here. It is kept and re-raised after,
            # and only when nothing worse is already on its way out.
            if caught is None:
                caught = exc
            else:
                _LOG.exception("Frame writer also failed while closing")

    # Written before anything is re-raised, and that ordering is the point.
    #
    # The frames are already on disk by now. A manifest written only on the
    # clean path meant that a stream which dropped and did not come back left
    # every JPEG in place with no session.json and no frames.jsonl beside them
    # -- and load_session rejects a directory without those two, so a long
    # capture became permanently unreadable by the tooling that wrote it. The
    # comment above about an interrupted session still being a session was true
    # for Ctrl+C and false for every other ending.
    #
    # A partial manifest describing what was actually recorded is worth more
    # than none, and the error still reaches the caller immediately below.
    info.frame_count = len(writer.records)
    _write_manifest(session_dir, info, writer.records)

    if caught is not None:
        raise caught
    if decoder.error is not None:
        raise decoder.error

    wall = max(time.perf_counter() - started, 1e-9)
    return CaptureSummary(
        session_dir=str(session_dir),
        source=str(video_path),
        frames=len(writer.records),
        dropped_frames=decoder.dropped_frames,
        bytes_written=writer.bytes_written,
        wall_time_seconds=wall,
        capture_hz=len(writer.records) / wall,
        reconnects=decoder.reconnects,
        decode_failures=decoder.decode_failures,
        write_errors=writer.write_errors,
    )


def _write_manifest(
    session_dir: Path, info: SessionInfo, records: list[FrameRecord]
) -> None:
    (session_dir / SESSION_NAME).write_text(
        json.dumps(asdict(info), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (session_dir / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _known_fields(kind: type, data: dict[str, Any]) -> dict[str, Any]:
    """Drop keys this version does not know about.

    A session written by a later build carries fields this one has never heard
    of, and refusing to open it over an unknown key would make recordings
    unreadable by exactly the tooling meant to outlive them.
    """
    from dataclasses import fields

    allowed = {item.name for item in fields(kind)}
    return {key: value for key, value in data.items() if key in allowed}


def load_session(session_dir: str | Path) -> tuple[SessionInfo, list[FrameRecord]]:
    """Read back what capture_session wrote."""
    session_dir = Path(session_dir)
    session_file = session_dir / SESSION_NAME
    manifest = session_dir / MANIFEST_NAME
    if not session_file.is_file() or not manifest.is_file():
        raise TelemetryOCRError(
            f"{session_dir} is not a capture session: expected {SESSION_NAME} and "
            f"{MANIFEST_NAME}"
        )
    info = SessionInfo(**_known_fields(SessionInfo, json.loads(session_file.read_text("utf-8"))))
    records = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(FrameRecord(**_known_fields(FrameRecord, json.loads(line))))
    # Frames are read back in capture order because temporal voting depends on
    # it; a manifest reordered by hand would silently change the readings.
    records.sort(key=lambda item: item.frame_index)
    return info, records
