from __future__ import annotations

import math
import queue
import statistics
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .errors import DependencyError, VideoError
from .ocr import OCRResult, Recognizer
from .output import RecordWriter, TelemetryRecord
from .preprocess import apply_recipe, extract_roi
from .temporal import TemporalStabilizer
from .validation import ValidatedValue, prefer_candidate, validate_text


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame: Any
    frame_index: int
    source_time_ms: float
    enqueued_at: float


@dataclass(slots=True)
class PipelineSummary:
    processed_frames: int
    emitted_records: int
    scheduled_frames: int
    dropped_frames: int
    wall_time_seconds: float
    update_hz: float
    latency_mean_ms: float
    latency_p95_ms: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    validated: ValidatedValue
    confidence: float
    raw_text: str
    variant: str


_END = object()


class VideoDecoder:
    def __init__(
        self,
        video_path: str | Path,
        output_queue: queue.Queue[FramePacket | object],
        ocr_hz: float,
        realtime: bool,
        stop_event: threading.Event,
    ):
        self.video_path = str(video_path)
        self.output_queue = output_queue
        self.ocr_hz = ocr_hz
        self.realtime = realtime
        self.stop_event = stop_event
        self.scheduled_frames = 0
        self.dropped_frames = 0
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="video-decoder", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        self.thread.join()

    def _run(self) -> None:
        try:
            try:
                import cv2
            except ImportError as exc:
                raise DependencyError("OpenCV is required to decode video") from exc
            capture = cv2.VideoCapture(self.video_path)
            if not capture.isOpened():
                raise VideoError(f"Cannot open video: {self.video_path}")
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or fps <= 0:
                    fps = 30.0
                interval_ms = 1000.0 / self.ocr_hz
                next_ocr_ms = 0.0
                first_source_ms: float | None = None
                start_wall = time.perf_counter()
                frame_index = 0
                while not self.stop_event.is_set():
                    # Inter-frame codecs still need every packet to be grabbed,
                    # but BGR conversion/copy is only needed for scheduled OCR
                    # frames. This removes needless full-HD materialization at
                    # the source frame rate.
                    ok = capture.grab()
                    if not ok:
                        break
                    source_time_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    if not math.isfinite(source_time_ms) or source_time_ms < 0:
                        source_time_ms = frame_index * 1000.0 / fps
                    if frame_index > 0 and source_time_ms == 0:
                        source_time_ms = frame_index * 1000.0 / fps
                    if first_source_ms is None:
                        first_source_ms = source_time_ms
                    if self.realtime:
                        due = start_wall + max(0.0, source_time_ms - first_source_ms) / 1000.0
                        remaining = due - time.perf_counter()
                        if remaining > 0 and self.stop_event.wait(remaining):
                            break
                    epsilon_ms = 500.0 / fps
                    if source_time_ms + epsilon_ms >= next_ocr_ms:
                        ok, frame = capture.retrieve()
                        if not ok:
                            raise VideoError(
                                f"Cannot retrieve scheduled frame {frame_index}: {self.video_path}"
                            )
                        self._put_latest(
                            FramePacket(frame, frame_index, source_time_ms, time.perf_counter())
                        )
                        self.scheduled_frames += 1
                        while next_ocr_ms <= source_time_ms + epsilon_ms:
                            next_ocr_ms += interval_ms
                    frame_index += 1
                if frame_index == 0:
                    raise VideoError(f"Video contains no decodable frames: {self.video_path}")
            finally:
                capture.release()
        except Exception as exc:
            self.error = exc
        finally:
            if not self.stop_event.is_set():
                while not self.stop_event.is_set():
                    try:
                        self.output_queue.put(_END, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def _put_latest(self, packet: FramePacket) -> None:
        while not self.stop_event.is_set():
            try:
                self.output_queue.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self.output_queue.get_nowait()
                    self.output_queue.task_done()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass


def recognize_grouped(
    recognizer: Recognizer, images: Sequence[Any], bucket_base: float
) -> list[OCRResult]:
    """Batch crops with similar aspect ratios while preserving caller order."""
    if not images:
        return []
    buckets: dict[int, list[tuple[int, Any]]] = {}
    for index, image in enumerate(images):
        height, width = image.shape[:2]
        ratio = max(width / max(height, 1), 1e-6)
        key = math.floor(math.log(ratio, bucket_base))
        buckets.setdefault(key, []).append((index, image))
    ordered: list[OCRResult | None] = [None] * len(images)
    for key in sorted(buckets):
        members = buckets[key]
        results = recognizer.recognize([image for _, image in members])
        if len(results) != len(members):
            raise RuntimeError("Recognizer returned a result count that does not match its batch")
        for (index, _), result in zip(members, results, strict=True):
            ordered[index] = result
    if any(item is None for item in ordered):
        raise RuntimeError("Internal error while restoring OCR batch order")
    return [item for item in ordered if item is not None]


class TelemetryPipeline:
    def __init__(self, config: AppConfig, recognizer: Recognizer):
        self.config = config
        self.recognizer = recognizer
        self.stabilizer = TemporalStabilizer(config)
        self._time_anchors: dict[str, tuple[int, float]] = {}
        self._last_candidates: dict[str, _Candidate] = {}

    def run(
        self,
        video_path: str | Path,
        jsonl_path: str | Path,
        csv_path: str | Path,
    ) -> PipelineSummary:
        runtime = self.config.runtime
        self._warmup(video_path)
        frame_queue: queue.Queue[FramePacket | object] = queue.Queue(maxsize=runtime.queue_size)
        stop_event = threading.Event()
        decoder = VideoDecoder(
            video_path, frame_queue, runtime.ocr_hz, runtime.realtime, stop_event
        )
        writer = RecordWriter(jsonl_path, csv_path)
        latencies: list[float] = []
        processed = 0
        emitted = 0
        start_wall = time.perf_counter()
        writer.start()
        decoder.start()
        caught: BaseException | None = None
        try:
            while True:
                item = frame_queue.get()
                try:
                    if item is _END:
                        break
                    assert isinstance(item, FramePacket)
                    records, latency = self.process_frame(item)
                    for record in records:
                        writer.write(record)
                    processed += 1
                    emitted += len(records)
                    latencies.append(latency)
                finally:
                    frame_queue.task_done()
        except BaseException as exc:
            caught = exc
            stop_event.set()
        finally:
            stop_event.set()
            decoder.join()
            try:
                writer.close()
            except Exception as exc:
                # Reporting a flush failure instead of the error that stopped the
                # run would hide the reason the run stopped.
                if caught is None:
                    caught = exc
        if caught is not None:
            raise caught
        if decoder.error is not None:
            raise decoder.error
        wall_time = max(time.perf_counter() - start_wall, 1e-9)
        return PipelineSummary(
            processed_frames=processed,
            emitted_records=emitted,
            scheduled_frames=decoder.scheduled_frames,
            dropped_frames=decoder.dropped_frames,
            wall_time_seconds=wall_time,
            update_hz=processed / wall_time,
            latency_mean_ms=statistics.fmean(latencies) if latencies else 0.0,
            latency_p95_ms=percentile(latencies, 95),
        )

    def _warmup(self, video_path: str | Path) -> None:
        iterations = self.config.runtime.warmup_iterations
        if iterations == 0:
            return
        try:
            import cv2
        except ImportError as exc:
            raise DependencyError("OpenCV is required to warm up OCR") from exc
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoError(f"Cannot open video for OCR warm-up: {video_path}")
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise VideoError(f"Cannot decode the first frame for OCR warm-up: {video_path}")
        images: list[Any] = []
        for item in self.config.fields:
            if item.derive_from_source_time:
                continue
            padding = self.config.runtime.padding if item.padding is None else item.padding
            crop = extract_roi(frame, item.box, padding)
            images.append(apply_recipe(crop, item.primary))
        for _ in range(iterations):
            recognize_grouped(
                self.recognizer, images, self.config.runtime.aspect_bucket_base
            )

    def process_frame(self, packet: FramePacket) -> tuple[list[TelemetryRecord], float]:
        primary_images: list[Any] = []
        primary_indexes: list[int] = []
        candidates: list[_Candidate | None] = [None] * len(self.config.fields)
        for index, item in enumerate(self.config.fields):
            if item.derive_from_source_time and item.field_id in self._time_anchors:
                text = self._derived_timestamp(item.field_id, packet.source_time_ms)
                candidates[index] = _Candidate(
                    validate_text(text, item),
                    1.0,
                    text,
                    "derived_source_time",
                )
                continue
            if (
                item.ocr_interval > 1
                and packet.frame_index % item.ocr_interval != item.ocr_phase
                and item.field_id in self._last_candidates
            ):
                previous = self._last_candidates[item.field_id]
                candidates[index] = _Candidate(
                    previous.validated,
                    previous.confidence,
                    previous.raw_text,
                    "held_last_ocr",
                )
                continue
            padding = self.config.runtime.padding if item.padding is None else item.padding
            crop = extract_roi(packet.frame, item.box, padding)
            primary_images.append(apply_recipe(crop, item.primary))
            primary_indexes.append(index)
        primary_results = recognize_grouped(
            self.recognizer, primary_images, self.config.runtime.aspect_bucket_base
        )
        fallback_indexes: list[int] = []
        fallback_images: list[Any] = []
        for index, result in zip(primary_indexes, primary_results, strict=True):
            field = self.config.fields[index]
            validated = validate_text(result.text, field)
            candidates[index] = _Candidate(
                validated, result.confidence, result.text, field.primary.name
            )
            if self.config.runtime.enable_fallback and (
                not validated.valid
                or result.confidence < self.config.runtime.confidence_threshold
            ):
                padding = self.config.runtime.padding if field.padding is None else field.padding
                crop = extract_roi(packet.frame, field.box, padding)
                fallback_images.append(apply_recipe(crop, field.fallback))
                fallback_indexes.append(index)
        if fallback_images:
            fallback_results = recognize_grouped(
                self.recognizer, fallback_images, self.config.runtime.aspect_bucket_base
            )
            for index, result in zip(fallback_indexes, fallback_results, strict=True):
                field = self.config.fields[index]
                fallback = _Candidate(
                    validate_text(result.text, field),
                    result.confidence,
                    result.text,
                    field.fallback.name,
                )
                primary = candidates[index]
                assert primary is not None
                preferred = prefer_candidate(
                    (primary.validated, primary.confidence),
                    (fallback.validated, fallback.confidence),
                )
                if preferred[0] is fallback.validated:
                    candidates[index] = fallback

        for field, candidate in zip(self.config.fields, candidates, strict=True):
            assert candidate is not None
            if candidate.variant not in {"derived_source_time", "held_last_ocr"}:
                self._last_candidates[field.field_id] = candidate
            if (
                field.derive_from_source_time
                and field.field_id not in self._time_anchors
                and candidate.validated.valid
            ):
                self._set_time_anchor(
                    field.field_id,
                    candidate.validated.normalized_text,
                    packet.source_time_ms,
                )

        # This includes time spent waiting in the bounded frame queue, not just OCR.
        latency_ms = (time.perf_counter() - packet.enqueued_at) * 1000.0
        records: list[TelemetryRecord] = []
        for field, candidate in zip(self.config.fields, candidates, strict=True):
            assert candidate is not None
            stable = self.stabilizer.add(
                field.field_id,
                candidate.validated,
                candidate.confidence,
                packet.source_time_ms,
            )
            records.append(
                TelemetryRecord(
                    source_time_ms=packet.source_time_ms,
                    frame_index=packet.frame_index,
                    field_id=field.field_id,
                    value=stable.value,
                    raw_text=candidate.raw_text,
                    confidence=candidate.confidence,
                    confirmed=stable.confirmed,
                    preprocess_variant=candidate.variant,
                    latency_ms=latency_ms,
                )
            )
        return records, latency_ms

    def _set_time_anchor(self, field_id: str, text: str, source_time_ms: float) -> None:
        # validate_config only proves the field's regex accepts "00:00:00"; a
        # wider regex can still deliver something else here.
        parts = text.split(":")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            return
        hours, minutes, seconds = (int(part) for part in parts)
        self._time_anchors[field_id] = (hours * 3600 + minutes * 60 + seconds, source_time_ms)

    def _derived_timestamp(self, field_id: str, source_time_ms: float) -> str:
        anchor_seconds, anchor_source_ms = self._time_anchors[field_id]
        elapsed_seconds = math.floor(max(0.0, source_time_ms - anchor_source_ms) / 1000.0)
        clock_seconds = (anchor_seconds + elapsed_seconds) % (24 * 3600)
        hours, remainder = divmod(clock_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
