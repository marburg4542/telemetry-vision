"""Whose turn it is, counted in the right thing.

`ocr_interval` and `ocr_phase` stagger slowly-changing fields so they do not all
cost a recognition on every OCR frame. That is a round robin over successive OCR
frames -- the README describes it as staggering the two altitude fields, and the
walkthrough as "fields whose turn has not yet come".

It was tested against the decoder's frame index instead, which counts every
frame *grabbed* from the source whether it was scheduled for recognition or not.
Only a subset of those indices reaches the recogniser, and which residues that
subset covers falls out of the fps/ocr_hz ratio. When it covers fewer than
`ocr_interval` of them, a field can have a phase whose turn never arrives.
"""

from __future__ import annotations

import importlib.util
import time
import unittest
from typing import Any

HAS_IMAGE_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"))


def _pipeline(interval: int, phase: int) -> tuple[Any, Any]:
    from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
    from telemetry_ocr.ocr import OCRResult
    from telemetry_ocr.pipeline import TelemetryPipeline

    class CountingRecognizer:
        def __init__(self) -> None:
            self.calls = 0

        def recognize(self, images: Any) -> Any:
            self.calls += 1
            return [OCRResult("12", 0.99) for _ in images]

    recognizer = CountingRecognizer()
    config = AppConfig(
        runtime=RuntimeConfig(enable_fallback=False),
        fields=[
            FieldConfig(
                field_id="altitude",
                box=(0.1, 0.1, 0.3, 0.2),
                allowed_chars="0123456789",
                regex=r"^\d+$",
                data_type="integer",
                ocr_interval=interval,
                ocr_phase=phase,
            )
        ],
    )
    return TelemetryPipeline(config, recognizer), recognizer


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class TurnCountingTests(unittest.TestCase):
    def _feed(self, pipeline: Any, indexes: list[int]) -> list[Any]:
        import numpy as np

        from telemetry_ocr.pipeline import FramePacket

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        out = []
        for i, index in enumerate(indexes):
            records, _ = pipeline.process_frame(
                FramePacket(frame, index, i * 33.3, time.perf_counter())
            )
            out.append(records)
        return out

    def test_a_phase_the_source_indexes_never_hit_still_gets_its_turn(self) -> None:
        """The bug, in the shape it actually occurs.

        30 fps at ocr_hz 15 schedules every second source frame, so the indexes
        reaching the recogniser are 0, 2, 4, 6 ... and `frame_index % 4` is only
        ever 0 or 2. A field with ocr_phase 1 was recognised once -- before
        _last_candidates had anything to hold -- and then republished from that
        one held candidate for the rest of the run.
        """
        pipeline, recognizer = _pipeline(interval=4, phase=1)
        even_only = [0, 2, 4, 6, 8, 10, 12, 14]
        self._feed(pipeline, even_only)
        # Eight OCR frames as turns 0..7. Turn 0 recognises because nothing is
        # cached to hold yet; then turns 1 and 5 match phase 1. Three in all.
        # Keyed off the source index it was exactly one -- the first frame --
        # and every frame after it republished that same reading.
        self.assertEqual(
            recognizer.calls,
            3,
            "a phase unreachable in the source indexes got no turn at all",
        )

    def test_the_turn_comes_round_at_the_configured_interval(self) -> None:
        pipeline, recognizer = _pipeline(interval=3, phase=2)
        self._feed(pipeline, list(range(12)))
        # Turn 0, then turns 2, 5, 8 and 11: five, whatever the indexes were.
        self.assertEqual(recognizer.calls, 5)

    def test_the_count_does_not_follow_the_source_index_at_all(self) -> None:
        # Same number of OCR frames, wildly different source indexes -- a stream
        # that dropped packets, say. The stagger must not notice.
        first, first_rec = _pipeline(interval=2, phase=0)
        second, second_rec = _pipeline(interval=2, phase=0)
        self._feed(first, [0, 1, 2, 3, 4, 5])
        self._feed(second, [0, 7, 19, 44, 45, 900])
        self.assertEqual(first_rec.calls, second_rec.calls)

    def test_the_first_frame_recognises_every_field_whatever_its_phase(self) -> None:
        # There is nothing cached to hold on the first frame, so the gate's
        # third condition lets it through. The benchmark in calibration.py
        # depends on this: it measures the first frame's batch.
        pipeline, recognizer = _pipeline(interval=4, phase=3)
        self._feed(pipeline, [0])
        self.assertEqual(recognizer.calls, 1)

    def test_a_held_value_is_still_marked_as_held(self) -> None:
        # The existing behaviour this must not have changed: a field whose turn
        # has not come is republished from its last candidate and says so.
        pipeline, _ = _pipeline(interval=2, phase=0)
        results = self._feed(pipeline, [0, 1])
        self.assertEqual(results[1][0].preprocess_variant, "held_last_ocr")
        self.assertEqual(results[1][0].raw_text, "12")


class BenchmarkBatchTests(unittest.TestCase):
    def test_the_benchmark_no_longer_drops_phased_fields(self) -> None:
        """It measures the first OCR frame, which recognises everything.

        Skipping by phase there measured a smaller batch than the run it exists
        to predict -- with the shipped config, one ROI smaller, because
        acft_alt_m carries ocr_phase 2.
        """
        import inspect

        from telemetry_ocr import calibration

        source = inspect.getsource(calibration.benchmark_threads)
        self.assertNotIn("ocr_phase", source)


if __name__ == "__main__":
    unittest.main()
