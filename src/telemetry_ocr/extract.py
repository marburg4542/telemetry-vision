"""Read telemetry from a captured session, at whatever pace it takes.

This is the half that the realtime pipeline could not do well. Reading from disk
there is no frame interval to finish inside, so nothing is skipped and nothing is
rushed: every captured frame is recognised, in the order it was captured, which
is what temporal voting needs to mean anything.

Every record names the image it was read from. That is the practical difference
between this and a realtime log -- a value that looks wrong can be checked
against the exact picture that produced it, instead of against a moment in a
video that has already gone past.
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capture import SessionInfo, load_session
from .config import AppConfig
from .errors import TelemetryOCRError, VideoError
from .ocr import Recognizer
from .output import RecordWriter
from .pipeline import FramePacket, TelemetryPipeline, percentile

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractSummary:
    session_dir: str
    frames: int
    processed_frames: int
    missing_frames: int
    unreadable_frames: int
    emitted_records: int
    confirmed_records: int
    confirmed_ratio: float
    wall_time_seconds: float
    frames_per_second: float
    latency_mean_ms: float
    latency_p95_ms: float


def extract_session(
    session_dir: str | Path,
    config: AppConfig,
    recognizer: Recognizer,
    jsonl_path: str | Path,
    csv_path: str | Path,
    *,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> ExtractSummary:
    """OCR every frame of a session and write telemetry that points back at it."""
    session_dir = Path(session_dir)
    info, records = load_session(session_dir)
    if not records:
        raise TelemetryOCRError(f"{session_dir} holds no frames")
    _warn_on_config_drift(info, config)

    if limit is not None:
        records = records[:limit]

    pipeline = TelemetryPipeline(config, recognizer)
    writer = RecordWriter(jsonl_path, csv_path)
    latencies: list[float] = []
    processed = 0
    emitted = 0
    confirmed = 0
    missing = 0
    unreadable = 0
    warmed = False

    started = time.perf_counter()
    writer.start()
    try:
        for position, record in enumerate(records, start=1):
            frame = _read_frame(session_dir / record.file)
            if frame is None:
                # Counted separately: a frame the manifest promised and disk does
                # not have is a different problem from one that will not decode,
                # and the two point at different things going wrong.
                if not (session_dir / record.file).is_file():
                    missing += 1
                else:
                    unreadable += 1
                continue
            if not warmed:
                # Same reason the realtime pipeline warms up, and the same
                # benefit: the first frame's timing is kernel compilation, not
                # recognition, and it would otherwise dominate the mean.
                pipeline._warmup_from_frame(frame)
                warmed = True
            packet = FramePacket(
                frame, record.frame_index, record.source_time_ms, time.perf_counter()
            )
            produced, latency = pipeline.process_frame(packet)
            for item in produced:
                writer.write(dataclasses.replace(item, image=record.file))
                if item.confirmed:
                    confirmed += 1
            processed += 1
            emitted += len(produced)
            latencies.append(latency)
            if on_progress is not None and position % 25 == 0:
                on_progress(position, len(records))
    finally:
        writer.close()

    wall = max(time.perf_counter() - started, 1e-9)
    return ExtractSummary(
        session_dir=str(session_dir),
        frames=len(records),
        processed_frames=processed,
        missing_frames=missing,
        unreadable_frames=unreadable,
        emitted_records=emitted,
        confirmed_records=confirmed,
        confirmed_ratio=confirmed / emitted if emitted else 0.0,
        wall_time_seconds=wall,
        frames_per_second=processed / wall,
        latency_mean_ms=statistics.fmean(latencies) if latencies else 0.0,
        latency_p95_ms=percentile(latencies, 95),
    )


def _read_frame(path: Path) -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by the CLI import
        raise VideoError("OpenCV is required to read captured frames") from exc
    # str() rather than the Path: OpenCV's imread takes a byte path and silently
    # returns None for anything it cannot convert, which on Windows includes
    # perfectly valid non-ASCII directories.
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    return frame


def _warn_on_config_drift(info: SessionInfo, config: AppConfig) -> None:
    """Say so when the session was captured against different ROIs.

    The boxes are fractions of the frame, so a config with different fields will
    happily read the wrong parts of an old session and produce confident
    nonsense. Not fatal -- re-reading a session with adjusted ROIs is a
    legitimate thing to want, and is half the reason to keep the frames.
    """
    captured = set(info.config_fields)
    current = {item.field_id for item in config.fields}
    if not captured or captured == current:
        return
    added = sorted(current - captured)
    removed = sorted(captured - current)
    _LOG.warning(
        "This session was captured with different fields (added: %s, missing: %s). "
        "ROI boxes are fractions of the frame, so check they still land on the "
        "same overlay.",
        added or "none",
        removed or "none",
    )
