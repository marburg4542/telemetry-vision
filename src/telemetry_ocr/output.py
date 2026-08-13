from __future__ import annotations

import csv
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    source_time_ms: float
    frame_index: int
    field_id: str
    value: Any
    raw_text: str
    confidence: float
    confirmed: bool
    preprocess_variant: str
    latency_ms: float


_STOP = object()


class RecordWriter:
    fieldnames: ClassVar[list[str]] = [
        "source_time_ms",
        "frame_index",
        "field_id",
        "value",
        "raw_text",
        "confidence",
        "confirmed",
        "preprocess_variant",
        "latency_ms",
    ]

    def __init__(self, jsonl_path: str | Path, csv_path: str | Path, queue_size: int = 100):
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        self.queue: queue.Queue[TelemetryRecord | object] = queue.Queue(maxsize=queue_size)
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="telemetry-writer", daemon=True)

    def start(self) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def write(self, record: TelemetryRecord) -> None:
        self._enqueue(record)

    def close(self) -> None:
        if self.thread.is_alive():
            self._enqueue(_STOP)
        self.thread.join()
        if self.error:
            raise self.error

    def _enqueue(self, item: TelemetryRecord | object) -> None:
        while True:
            if self.error:
                raise self.error
            if not self.thread.is_alive():
                raise RuntimeError("Telemetry writer stopped unexpectedly")
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        try:
            with self.jsonl_path.open("w", encoding="utf-8") as json_file, self.csv_path.open(
                "w", encoding="utf-8", newline=""
            ) as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                csv_writer.writeheader()
                next_flush = time.monotonic() + 0.1
                while True:
                    item = self.queue.get()
                    try:
                        if item is _STOP:
                            break
                        assert isinstance(item, TelemetryRecord)
                        data = asdict(item)
                        json_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                        csv_writer.writerow(data)
                        if time.monotonic() >= next_flush:
                            json_file.flush()
                            csv_file.flush()
                            next_flush = time.monotonic() + 0.1
                    finally:
                        self.queue.task_done()
                json_file.flush()
                csv_file.flush()
        except Exception as exc:
            self.error = exc
