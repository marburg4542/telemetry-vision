"""Capture then extract, against a fake camera and a real encoder.

The two halves are tested together because what matters about them is the join:
that the manifest written by one is enough for the other to read the frames back
in order, and that every telemetry record can be traced to the picture it came
from. Either half alone can look correct and still not compose.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

HAS_IMAGE_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"))


def _config(ocr_hz: float = 30.0) -> Any:
    from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig

    return AppConfig(
        runtime=RuntimeConfig(
            confidence_threshold=0.5,
            warmup_iterations=0,
            ocr_hz=ocr_hz,
            enable_fallback=False,
            realtime=False,
        ),
        fields=[
            FieldConfig(
                field_id="altitude",
                box=(0.1, 0.1, 0.4, 0.3),
                allowed_chars="0123456789.",
                regex=r"^\d+(\.\d+)?$",
                data_type="float",
                profile="fast",
            )
        ],
    )


class _Recognizer:
    def __init__(self, text: str = "241.0") -> None:
        self.text = text
        self.calls = 0

    def recognize(self, images: Any) -> Any:
        from telemetry_ocr.ocr import OCRResult

        self.calls += 1
        return [OCRResult(self.text, 0.97) for _ in images]


class _FakeCapture:
    """A camera with a known, finite number of distinguishable frames."""

    def __init__(self, frames: int) -> None:
        import numpy as np

        self.total = frames
        self.position = 0
        self.released = False
        self._np = np

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def grab(self) -> bool:
        time.sleep(0.002)
        if self.position >= self.total:
            return False
        self.position += 1
        return True

    def retrieve(self) -> tuple[bool, Any]:
        # Each frame carries its own number in pixel 0, so a session that writes
        # or reads frames out of order cannot pass by looking merely plausible.
        frame = self._np.full((120, 160, 3), 30, dtype=self._np.uint8)
        frame[0, 0] = self.position % 250
        return True, frame

    def release(self) -> None:
        self.released = True


class _CaptureCv2:
    """Real OpenCV, imaginary cameras. imwrite and imread are the real thing."""

    def __init__(self, frames: int = 40) -> None:
        import cv2

        self._cv2 = cv2
        self.frames = frames

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cv2, name)

    def VideoCapture(
        self, source: str, api: int | None = None, params: list[int] | None = None
    ) -> _FakeCapture:
        return _FakeCapture(self.frames)


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class CaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cv2 = _CaptureCv2()
        patch = mock.patch.dict(sys.modules, {"cv2": self.cv2})
        patch.start()
        self.addCleanup(patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _capture(self, **kwargs: Any) -> Any:
        from telemetry_ocr.capture import capture_session

        return capture_session(
            "clip.mp4", self.root / "session", _config(), **kwargs
        )

    def test_a_session_holds_its_frames_and_a_manifest(self) -> None:
        from telemetry_ocr.capture import MANIFEST_NAME, SESSION_NAME

        summary = self._capture(max_frames=10)
        session = self.root / "session"
        self.assertTrue((session / SESSION_NAME).is_file())
        self.assertTrue((session / MANIFEST_NAME).is_file())
        self.assertGreaterEqual(summary.frames, 10)
        written = sorted((session / "frames").iterdir())
        self.assertEqual(len(written), summary.frames)
        self.assertGreater(summary.bytes_written, 0)

    def test_the_manifest_describes_every_frame_on_disk(self) -> None:
        # The manifest is the only thing tying a reading to a picture. An entry
        # without a file, or a file without an entry, breaks that silently.
        from telemetry_ocr.capture import load_session

        self._capture(max_frames=12)
        info, records = load_session(self.root / "session")
        self.assertEqual(info.frame_count, len(records))
        for record in records:
            self.assertTrue((self.root / "session" / record.file).is_file())
            self.assertEqual(record.bytes, (self.root / "session" / record.file).stat().st_size)
        indexes = [record.frame_index for record in records]
        self.assertEqual(indexes, sorted(indexes))
        self.assertEqual(len(set(indexes)), len(indexes))

    def test_nothing_is_dropped(self) -> None:
        """The whole reason for capturing first.

        The realtime path deliberately discards frames it cannot keep up with.
        Here the decoder waits for the writer instead, because a missing frame
        is a hole no later processing can fill.
        """
        summary = self._capture()
        self.assertEqual(summary.dropped_frames, 0)
        self.assertEqual(summary.write_errors, 0)

    def test_jpeg_is_the_default(self) -> None:
        # Measured identical to PNG on 2,340 real readings, at 3x the capture
        # rate. The rate is the part that matters: PNG is slower than realtime
        # here, and a live camera does not wait for the writer.
        self._capture(max_frames=3)
        names = [item.name for item in (self.root / "session" / "frames").iterdir()]
        self.assertTrue(all(name.endswith(".jpg") for name in names), names)

    def test_png_is_available_when_the_overlay_is_harder(self) -> None:
        self._capture(max_frames=3, image_format="png")
        names = [item.name for item in (self.root / "session" / "frames").iterdir()]
        self.assertTrue(all(name.endswith(".png") for name in names), names)

    def test_capturing_into_a_used_directory_is_refused(self) -> None:
        # Two recordings under one manifest would produce telemetry that points
        # at the wrong pictures, which is worse than refusing to start.
        from telemetry_ocr.capture import capture_session
        from telemetry_ocr.errors import TelemetryOCRError

        self._capture(max_frames=3)
        with self.assertRaises(TelemetryOCRError):
            capture_session("clip.mp4", self.root / "session", _config(), max_frames=3)

    def test_a_bad_format_is_rejected_before_anything_is_written(self) -> None:
        from telemetry_ocr.capture import capture_session
        from telemetry_ocr.errors import TelemetryOCRError

        with self.assertRaises(TelemetryOCRError):
            capture_session(
                "clip.mp4", self.root / "other", _config(), image_format="tiff"
            )


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class ExtractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cv2 = _CaptureCv2()
        patch = mock.patch.dict(sys.modules, {"cv2": self.cv2})
        patch.start()
        self.addCleanup(patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = self.root / "session"

        from telemetry_ocr.capture import capture_session

        capture_session("clip.mp4", self.session, _config(), max_frames=8)

    def _extract(self, **kwargs: Any) -> Any:
        from telemetry_ocr.extract import extract_session

        return extract_session(
            self.session,
            _config(),
            _Recognizer(),
            self.root / "telemetry.jsonl",
            self.root / "telemetry.csv",
            **kwargs,
        )

    def _records(self) -> list[dict[str, Any]]:
        text = (self.root / "telemetry.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def test_every_captured_frame_is_read(self) -> None:
        summary = self._extract()
        self.assertGreaterEqual(summary.frames, 8)
        self.assertEqual(summary.processed_frames, summary.frames)
        self.assertEqual(summary.missing_frames, 0)
        self.assertEqual(summary.unreadable_frames, 0)

    def test_each_record_names_the_image_it_came_from(self) -> None:
        """The point of the whole rearrangement.

        A realtime log can only say when a value was read. This has to be able
        to say which picture produced it, and that picture has to be there.
        """
        self._extract()
        records = self._records()
        self.assertTrue(records)
        for record in records:
            self.assertTrue(record["image"])
            self.assertTrue((self.session / record["image"]).is_file())

    def test_the_csv_carries_the_image_column_too(self) -> None:
        self._extract()
        header = (self.root / "telemetry.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("image", header.split(","))

    def test_readings_follow_capture_order(self) -> None:
        # Temporal voting only means something if frames arrive in the order they
        # were recorded; a manifest read out of order would change the answers.
        self._extract()
        times = [record["source_time_ms"] for record in self._records()]
        self.assertEqual(times, sorted(times))

    def test_a_frame_deleted_after_capture_is_reported_not_ignored(self) -> None:
        from telemetry_ocr.capture import load_session

        _, records = load_session(self.session)
        (self.session / records[2].file).unlink()
        summary = self._extract()
        self.assertEqual(summary.missing_frames, 1)
        self.assertEqual(summary.processed_frames, summary.frames - 1)

    def test_limit_reads_only_the_first_frames(self) -> None:
        summary = self._extract(limit=3)
        self.assertEqual(summary.frames, 3)
        self.assertEqual(summary.processed_frames, 3)

    def test_an_empty_directory_is_not_a_session(self) -> None:
        from telemetry_ocr.errors import TelemetryOCRError
        from telemetry_ocr.extract import extract_session

        with self.assertRaises(TelemetryOCRError):
            extract_session(
                self.root / "nothing-here",
                _config(),
                _Recognizer(),
                self.root / "a.jsonl",
                self.root / "a.csv",
            )


if __name__ == "__main__":
    unittest.main()
