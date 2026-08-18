"""Re-reading a saved capture, and reporting detection confidence.

The second reading exists because the live telemetry and the saved picture are
not guaranteed to describe the same frame. What matters is that disagreement is
surfaced rather than resolved: the live reading has temporal voting behind it and
the re-read is provably of this exact image, so neither is automatically right
and picking one silently would invent certainty.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HAS_IMAGE_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"))


@dataclass
class _Summary:
    status: str
    counts: dict[str, int] = field(default_factory=dict)
    confidence: dict[str, dict[str, float]] = field(default_factory=dict)
    total: int = 0
    age_ms: float | None = None


class DetectionConfidenceTests(unittest.TestCase):
    def test_confidence_is_reported_per_class(self) -> None:
        from telemetry_ocr.line_client import describe_detections

        text = describe_detections(
            _Summary("ok", {"car": 2}, {"car": {"mean": 0.88, "min": 0.86, "max": 0.9}}, 2)
        )
        self.assertIn("car 2", text)
        self.assertIn("0.88", text)

    def test_a_weak_box_among_strong_ones_is_called_out(self) -> None:
        """The mean alone hides the box that makes a count doubtful.

        Three cars at 0.9 and three where one sits at 0.31 are different claims,
        and the reader is deciding whether to trust the number.
        """
        from telemetry_ocr.line_client import describe_detections

        text = describe_detections(
            _Summary("ok", {"car": 3}, {"car": {"mean": 0.70, "min": 0.31, "max": 0.93}}, 3)
        )
        self.assertIn("ต่ำสุด", text)
        self.assertIn("0.31", text)

    def test_a_single_box_does_not_repeat_itself(self) -> None:
        # With one box the mean is the minimum, and printing both is noise.
        from telemetry_ocr.line_client import describe_detections

        text = describe_detections(
            _Summary("ok", {"person": 1}, {"person": {"mean": 0.8, "min": 0.8, "max": 0.8}}, 1)
        )
        self.assertNotIn("ต่ำสุด", text)

    def test_counts_without_confidence_still_render(self) -> None:
        # Sessions captured before confidence was recorded must still be sendable.
        from telemetry_ocr.line_client import describe_detections

        self.assertIn("car 2", describe_detections(_Summary("ok", {"car": 2}, {}, 2)))


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class VerifyShotTests(unittest.TestCase):
    def setUp(self) -> None:
        import cv2
        import numpy as np

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "frame.jpg"
        frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.image), frame))

    def _pipeline(self, text: str) -> Any:
        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.pipeline import TelemetryPipeline

        class Recognizer:
            def recognize(self, images: Any) -> Any:
                return [OCRResult(text, 0.95) for _ in images]

        config = AppConfig(
            runtime=RuntimeConfig(
                enable_fallback=False, confidence_threshold=0.5, warmup_iterations=0
            ),
            fields=[
                FieldConfig(
                    field_id="lrf_m",
                    box=(0.1, 0.1, 0.4, 0.4),
                    allowed_chars="0123456789",
                    regex=r"^\d+$",
                    data_type="integer",
                    profile="fast",
                )
            ],
        )
        return TelemetryPipeline(config, Recognizer())

    def test_matching_ocr_text_means_the_same_picture(self) -> None:
        """Agreement is about the text, not the published value.

        Measured on a real stream, 10 of 11 fields produced identical OCR text
        while 8 of 11 published values differed -- because voting had not yet
        confirmed what the frame said. Comparing values would have reported that
        as the reading being wrong, when the two were reading the same picture.
        """
        from telemetry_ocr.capture_service import verify_shot

        live = [{"field_id": "lrf_m", "value": 721, "raw_text": "721"}]
        result = verify_shot(self.image, live, self._pipeline("721"))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.agreed, 1)
        self.assertEqual(result.disagreed, 0)
        self.assertEqual(result.lagging, 0)
        self.assertTrue(result.fields[0].agrees)

    def test_a_value_the_live_feed_had_not_caught_up_to_is_called_lagging(self) -> None:
        # Same text, different published value: temporal voting was still
        # holding an earlier confirmation. Normal, and not a disagreement.
        from telemetry_ocr.capture_service import verify_shot

        live = [{"field_id": "lrf_m", "value": 999, "raw_text": "721"}]
        result = verify_shot(self.image, live, self._pipeline("721"))
        self.assertEqual(result.agreed, 1)
        self.assertEqual(result.disagreed, 0)
        self.assertEqual(result.lagging, 1)
        self.assertEqual(result.fields[0].value, 721)
        self.assertEqual(result.fields[0].live_value, 999)

    def test_different_ocr_text_is_a_real_disagreement(self) -> None:
        # Different text means the two readings did not see the same picture,
        # which is the failure this check exists to catch.
        from telemetry_ocr.capture_service import verify_shot

        live = [{"field_id": "lrf_m", "value": 999, "raw_text": "999"}]
        result = verify_shot(self.image, live, self._pipeline("721"))
        self.assertEqual(result.disagreed, 1)
        self.assertEqual(result.agreed, 0)
        self.assertFalse(result.fields[0].agrees)

    def test_a_missing_image_is_reported_not_raised(self) -> None:
        # A capture is already safely on disk with its live telemetry; a failed
        # second reading must not cost the operator that.
        from telemetry_ocr.capture_service import verify_shot

        result = verify_shot(self.root / "absent.jpg", [], self._pipeline("721"))
        self.assertEqual(result.status, "unreadable")

    def test_the_message_reports_the_image_value_not_the_live_one(self) -> None:
        """The message names a file someone may open.

        So the numbers beside it have to be the ones that file contains, with
        the trailing live value mentioned rather than published as the answer.
        """
        from telemetry_ocr.capture_service import ShotResult, Verification, VerifiedField
        from telemetry_ocr.line_client import format_capture_message

        shot = ShotResult(
            session="s",
            frame_index=0,
            image="frames/00000000.jpg",
            source_time_ms=0.0,
            captured_at="2026-08-18T05:00:00+00:00",
            bytes=1,
            records=[{"field_id": "lrf_m", "value": 999, "raw_text": "721", "confirmed": True}],
            verification=Verification(
                status="ok",
                fields=[
                    VerifiedField("lrf_m", 721, "721", 0.95, True, agrees=True, live_value=999)
                ],
                agreed=1,
                disagreed=0,
                lagging=1,
            ),
        )
        text = format_capture_message(shot)
        self.assertIn("LRF   721 m", text)
        self.assertIn("ตามไม่ทัน 1", text)
        # The trailing live value is counted, not listed: the reader wants the
        # numbers, and a per-field diff of normal voting behaviour is noise.
        self.assertNotIn("999", text)

    def test_full_agreement_says_where_the_numbers_came_from(self) -> None:
        from telemetry_ocr.capture_service import Verification
        from telemetry_ocr.line_client import describe_verification

        text = describe_verification(Verification("ok", [], agreed=11, disagreed=0), {})
        self.assertIn("11/11", text)
        self.assertIn("ไฟล์ภาพ", text)


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class RecognizerLockTests(unittest.TestCase):
    def test_the_recognizer_is_entered_by_one_thread_at_a_time(self) -> None:
        """A stream reads frames on its own thread while a capture re-reads a file.

        PaddleOCR makes no thread-safety promise, so overlapping entries are the
        collision to prevent; the recognizer here records whether it ever saw
        two callers at once.
        """
        import threading
        import time

        import numpy as np

        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.pipeline import FramePacket, TelemetryPipeline

        overlaps = []
        inside = threading.Semaphore(1)

        class JealousRecognizer:
            def recognize(self, images: Any) -> Any:
                if not inside.acquire(blocking=False):
                    overlaps.append(True)
                else:
                    time.sleep(0.02)
                    inside.release()
                return [OCRResult("12", 0.9) for _ in images]

        config = AppConfig(
            runtime=RuntimeConfig(
                enable_fallback=False, confidence_threshold=0.5, warmup_iterations=0
            ),
            fields=[
                FieldConfig(
                    field_id="lrf_m",
                    box=(0.1, 0.1, 0.4, 0.4),
                    allowed_chars="0123456789",
                    regex=r"^\d+$",
                    data_type="integer",
                    profile="fast",
                )
            ],
        )
        recognizer = JealousRecognizer()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        def work() -> None:
            # Separate pipelines, as the real stream and the verifier are.
            pipeline = TelemetryPipeline(config, recognizer)
            for index in range(8):
                pipeline.process_frame(FramePacket(frame, index, index * 100.0, 0.0))

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(overlaps, [], "two threads were inside the recognizer at once")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class AnnotationTests(unittest.TestCase):
    """Boxes drawn onto a copy, never onto the record."""

    def _frame(self) -> Any:
        import numpy as np

        return np.full((1080, 1920, 3), 40, dtype=np.uint8)

    def _boxes(self) -> list[dict[str, Any]]:
        # In the detector's own space, which is a downscaled copy of the frame.
        return [
            {"x1": 100.0, "y1": 50.0, "x2": 200.0, "y2": 130.0,
             "confidence": 0.91, "class_id": 0, "label": "car"}
        ]

    def test_boxes_are_scaled_from_the_detector_space_to_the_frame(self) -> None:
        """Unscaled boxes land in the corner at a fraction of their size.

        Which looks like a tracking failure rather than a units mistake, so it
        is worth pinning that the scaling happens at all.
        """
        import numpy as np

        from telemetry_ocr.capture_service import annotate_frame

        frame = self._frame()
        marked = annotate_frame(frame, self._boxes(), 640, 360)
        changed = np.nonzero(np.abs(marked.astype(int) - frame.astype(int)).sum(axis=2) > 30)
        xs, ys = changed[1], changed[0]
        # 100/640 of 1920 is 300; 200/640 is 600. Allow for line width and label.
        self.assertGreater(xs.max(), 500)
        self.assertLess(xs.min(), 340)
        self.assertGreater(ys.max(), 300)

    def test_the_original_frame_is_never_written_on(self) -> None:
        # The clean frame is the record and the thing OCR is re-read from; a box
        # across an OSD field would corrupt the reading taken from it.
        import numpy as np

        from telemetry_ocr.capture_service import annotate_frame

        frame = self._frame()
        before = frame.copy()
        annotate_frame(frame, self._boxes(), 640, 360)
        self.assertTrue(np.array_equal(frame, before))

    def test_the_label_carries_the_class_and_its_confidence(self) -> None:
        import numpy as np

        from telemetry_ocr.capture_service import annotate_frame

        frame = self._frame()
        plain = annotate_frame(frame, [{**self._boxes()[0], "confidence": None}], 640, 360)
        with_conf = annotate_frame(frame, self._boxes(), 640, 360)
        # The confidence text makes the label wider, so more pixels differ.
        plain_changed = np.count_nonzero(
            np.abs(plain.astype(int) - frame.astype(int)).sum(axis=2) > 30
        )
        conf_changed = np.count_nonzero(
            np.abs(with_conf.astype(int) - frame.astype(int)).sum(axis=2) > 30
        )
        self.assertGreater(conf_changed, plain_changed)

    def test_a_malformed_box_is_skipped_not_fatal(self) -> None:
        # One unusable box must not cost the whole annotation.
        from telemetry_ocr.capture_service import annotate_frame

        boxes = [{"x1": "nonsense"}, *self._boxes()]
        marked = annotate_frame(self._frame(), boxes, 640, 360)
        self.assertEqual(marked.shape, (1080, 1920, 3))

    def test_the_message_points_at_the_annotated_copy(self) -> None:
        from telemetry_ocr.capture_service import ShotResult
        from telemetry_ocr.line_client import format_capture_message

        shot = ShotResult(
            session="s",
            frame_index=0,
            image="frames/00000000.jpg",
            source_time_ms=0.0,
            captured_at="2026-08-18T05:00:00+00:00",
            bytes=1,
            records=[],
            annotated="frames/00000000.boxes.jpg",
        )
        text = format_capture_message(shot, directory="out")
        self.assertIn("00000000.boxes.jpg", text)
        # And still names the original, which is the record.
        self.assertIn("00000000.jpg", text)
