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
from unittest import mock

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

        So the numbers beside it are the ones that file contains, wherever the
        picture could be read at all. Temporal voting is deliberately behind the
        current frame, so a differing live value is expected rather than
        alarming, and is not repeated in the message.
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
        # The live value it disagreed with is not repeated anywhere: the reader
        # wants the reading, not a diff of normal voting behaviour.
        self.assertNotIn("999", text)

    def test_full_agreement_says_where_the_numbers_came_from(self) -> None:
        from telemetry_ocr.capture_service import Verification
        from telemetry_ocr.line_client import describe_verification

        text = describe_verification(Verification("ok", [], agreed=11, disagreed=0), {})
        self.assertIn("11/11", text)
        self.assertIn("ไฟล์ภาพ", text)


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class EnsembleTests(unittest.TestCase):
    """A capture is read by several models through several channels, and voted.

    Measured against an answer key -- the camera-yaw ROI of 42 saved captures
    read by eye, so it counts correct answers rather than answers that pass the
    regex. One model on one channel gets 35 right, 4 wrong and 3 blank; three
    models on three channels get 40 right, 2 wrong and none blank. Neither half
    does it alone: the two extra models score 26 and 21 on their own, which is
    why they are voters rather than replacements.
    """

    override = True

    def _config(self):
        from telemetry_ocr.config import AppConfig, FieldConfig, RecipeConfig, RuntimeConfig

        return AppConfig(
            runtime=RuntimeConfig(model_name="model-a"),
            fields=[
                FieldConfig(
                    field_id="speed",
                    box=(0.1, 0.1, 0.4, 0.4),
                    allowed_chars="0123456789",
                    regex=r"^\d+$",
                    data_type="integer",
                    profile="fast",
                    ensemble_override=self.override,
                    primary=RecipeConfig(name="color_baseline", channel="color"),
                )
            ],
        )

    def _pipeline(self, answers_by_model):
        """answers_by_model: {model: [(text, confidence) per channel]}."""
        import telemetry_ocr.capture_service as service
        from telemetry_ocr.ocr import OCRResult

        config = self._config()

        class Reader:
            def __init__(self, answers):
                self.answers = answers
                self.calls = 0

            def recognize(self, images):
                self.calls += 1
                return [OCRResult(*self.answers[i % len(self.answers)]) for i in range(len(images))]

        readers = {name: Reader(rows) for name, rows in answers_by_model.items()}
        service._ensemble_readers.clear()
        service._ensemble_failed.clear()
        service._ensemble_readers.update(readers)

        class Pipeline:
            def __init__(self):
                self.config = config
                self.recognizer = readers.get("model-a")

        self.addCleanup(service._ensemble_readers.clear)
        return Pipeline(), readers

    def _record(self, value):
        from telemetry_ocr.output import TelemetryRecord

        return TelemetryRecord(
            source_time_ms=0.0, frame_index=0, field_id="speed", value=value,
            raw_text="" if value is None else str(value), confidence=0.5,
            confirmed=False, preprocess_variant="color_baseline", latency_ms=1.0,
        )

    def _frame(self):
        import numpy as np

        return np.zeros((60, 80, 3), dtype="uint8")

    def test_the_majority_wins_over_one_confident_reader(self) -> None:
        """The failure this exists for.

        The colour recipe read "3.40" off a ROI plainly showing "33.40" -- at
        0.91 confidence, so nothing downstream had any reason to doubt it. Only
        readers that are wrong in different places can outvote that.
        """
        from telemetry_ocr.capture_service import _ensemble_read

        pipeline, _readers = self._pipeline({
            "model-a": [("340", 0.99), ("340", 0.99), ("340", 0.99)],
            "model-b": [("3340", 0.60), ("3340", 0.62), ("3340", 0.61)],
            "model-c": [("3340", 0.55), ("3340", 0.58), ("3340", 0.57)],
        })
        out = _ensemble_read(self._frame(), pipeline, [self._record(340)])
        self.assertEqual(out[0].value, 3340)

    def test_a_lone_dissenter_does_not_win(self) -> None:
        from telemetry_ocr.capture_service import _ensemble_read

        pipeline, _readers = self._pipeline({
            "model-a": [("128", 0.7), ("128", 0.7), ("128", 0.7)],
            "model-b": [("128", 0.7), ("128", 0.7), ("128", 0.7)],
            "model-c": [("999", 0.99), ("999", 0.99), ("999", 0.99)],
        })
        out = _ensemble_read(self._frame(), pipeline, [self._record(128)])
        self.assertEqual(out[0].value, 128)

    def test_a_tie_goes_to_the_more_confident_side(self) -> None:
        from telemetry_ocr.capture_service import _ensemble_read

        pipeline, _readers = self._pipeline({
            "model-a": [("111", 0.20), ("111", 0.20), ("111", 0.20)],
            "model-b": [("222", 0.90), ("222", 0.90), ("222", 0.90)],
        })
        out = _ensemble_read(self._frame(), pipeline, [self._record(111)])
        self.assertEqual(out[0].value, 222)

    def test_every_reader_sees_every_channel(self) -> None:
        # Three channels in one batch per model: the crops are built once and
        # recognised together, not once per channel per model.
        from telemetry_ocr.capture_service import _ENSEMBLE_CHANNELS, _ensemble_read

        pipeline, readers = self._pipeline({
            "model-a": [("1", 0.5)], "model-b": [("1", 0.5)],
        })
        _ensemble_read(self._frame(), pipeline, [self._record(1)])
        for name, reader in readers.items():
            self.assertEqual(reader.calls, 1, name)
        self.assertEqual(len(_ENSEMBLE_CHANNELS), 3)

    def test_a_field_nothing_can_read_keeps_what_it_had(self) -> None:
        from telemetry_ocr.capture_service import _ensemble_read

        pipeline, _readers = self._pipeline({
            "model-a": [("not a number", 0.9)], "model-b": [("also not", 0.9)],
        })
        out = _ensemble_read(self._frame(), pipeline, [self._record(128)])
        self.assertEqual(out[0].value, 128)

    def test_a_blank_field_the_ensemble_can_read_gets_filled(self) -> None:
        from telemetry_ocr.capture_service import _ensemble_read

        pipeline, _readers = self._pipeline({
            "model-a": [("128", 0.8)], "model-b": [("128", 0.9)],
        })
        out = _ensemble_read(self._frame(), pipeline, [self._record(None)])
        self.assertEqual(out[0].value, 128)
        # Agreement between readers is not the temporal quorum `confirmed`
        # means, and borrowing the word would misreport it.
        self.assertFalse(out[0].confirmed)

    def test_the_winning_read_says_where_it_came_from(self) -> None:
        from telemetry_ocr.capture_service import _ensemble_read

        pipeline, _readers = self._pipeline({
            "model-a": [("128", 0.4)], "model-b": [("128", 0.95)],
        })
        out = _ensemble_read(self._frame(), pipeline, [self._record(None)])
        self.assertIn("model-b", out[0].preprocess_variant)
        self.assertIn("6/6", out[0].preprocess_variant)

    def test_a_field_that_has_not_been_measured_keeps_what_it_read(self) -> None:
        """Blanks only, unless someone has built an answer key for the field.

        Letting the vote override everywhere was tried: it took the camera yaw
        from 35 of 42 correct to 40 and in the same run cost the camera pitch
        several readings, mostly by losing the minus sign -- "-8.75" became
        "8.75". Trading one field's accuracy for another's is not an
        improvement, it is a different set of wrong numbers.
        """
        from telemetry_ocr.capture_service import _ensemble_read

        self.override = False
        pipeline, _readers = self._pipeline({
            "model-a": [("999", 0.99)], "model-b": [("999", 0.99)],
        })
        out = _ensemble_read(self._frame(), pipeline, [self._record(128)])
        self.assertEqual(out[0].value, 128)

    def test_a_blank_is_filled_even_where_overriding_is_off(self) -> None:
        # Nothing to make worse: there was no reading.
        from telemetry_ocr.capture_service import _ensemble_read

        self.override = False
        pipeline, _readers = self._pipeline({
            "model-a": [("128", 0.8)], "model-b": [("128", 0.9)],
        })
        out = _ensemble_read(self._frame(), pipeline, [self._record(None)])
        self.assertEqual(out[0].value, 128)

    def test_a_model_that_will_not_load_is_not_retried_every_capture(self) -> None:
        """The usual reason is that it was never downloaded, and the next
        capture will not change that. The ensemble votes with what did load."""
        import telemetry_ocr.capture_service as service

        service._ensemble_readers.clear()
        service._ensemble_failed.clear()
        self.addCleanup(service._ensemble_readers.clear)
        self.addCleanup(service._ensemble_failed.clear)
        attempts = []

        def refuse(runtime):
            attempts.append(runtime.model_name)
            raise RuntimeError("not downloaded")

        with mock.patch("telemetry_ocr.ocr.create_recognizer", refuse):
            config = self._config()
            first = service._ensemble_recognizers(config.runtime, "primary")
            second = service._ensemble_recognizers(config.runtime, "primary")
        self.assertEqual(first, {"model-a": "primary"})
        self.assertEqual(second, {"model-a": "primary"})
        # Each missing model tried once, not once per capture.
        self.assertEqual(len(attempts), len(set(attempts)))


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

    def test_the_message_is_the_readings_and_nothing_about_the_file(self) -> None:
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
        text = format_capture_message(shot)
        # The annotated picture goes with the message; where it sits on this
        # machine is of no use to whoever reads that message on a phone.
        self.assertNotIn("00000000.boxes.jpg", text)
        self.assertNotIn("00000000.jpg", text)
        self.assertNotIn("รูป:", text)
