import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HAS_IMAGE_DEPS = bool(
    importlib.util.find_spec("numpy")
    and importlib.util.find_spec("cv2")
    and importlib.util.find_spec("yaml")
)


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class PipelineIntegrationTests(unittest.TestCase):
    def test_timestamp_is_ocr_anchored_then_derived_from_video_time(self):
        import time

        import numpy as np

        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.pipeline import FramePacket, TelemetryPipeline

        class TimestampRecognizer:
            def __init__(self):
                self.calls = 0

            def recognize(self, images):
                self.calls += 1
                return [OCRResult("16:31:23", 0.99) for _ in images]

        recognizer = TimestampRecognizer()
        config = AppConfig(
            runtime=RuntimeConfig(confidence_threshold=0.8),
            fields=[
                FieldConfig(
                    field_id="timestamp",
                    box=(0.1, 0.1, 0.3, 0.2),
                    allowed_chars="0123456789:",
                    regex=r"^\d{2}:\d{2}:\d{2}$",
                    data_type="string",
                    derive_from_source_time=True,
                    profile="fast",
                )
            ],
        )
        pipeline = TelemetryPipeline(config, recognizer)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        first, _ = pipeline.process_frame(FramePacket(frame, 0, 0.0, time.perf_counter()))
        second, _ = pipeline.process_frame(FramePacket(frame, 45, 1500.0, time.perf_counter()))
        self.assertEqual(first[0].raw_text, "16:31:23")
        self.assertEqual(second[0].raw_text, "16:31:24")
        self.assertEqual(second[0].preprocess_variant, "derived_source_time")
        self.assertEqual(recognizer.calls, 1)

    def test_slow_field_reuses_last_ocr_between_scheduled_reads(self):
        import time

        import numpy as np

        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.pipeline import FramePacket, TelemetryPipeline

        class CountingRecognizer:
            def __init__(self):
                self.calls = 0

            def recognize(self, images):
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
                    ocr_interval=2,
                    ocr_phase=0,
                )
            ],
        )
        pipeline = TelemetryPipeline(config, recognizer)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        pipeline.process_frame(FramePacket(frame, 0, 0.0, time.perf_counter()))
        second, _ = pipeline.process_frame(FramePacket(frame, 1, 33.3, time.perf_counter()))
        self.assertEqual(recognizer.calls, 1)
        self.assertEqual(second[0].raw_text, "12")
        self.assertEqual(second[0].preprocess_variant, "held_last_ocr")

    def test_synthetic_video_reaches_confirmed_output(self):
        import cv2
        import numpy as np

        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.pipeline import TelemetryPipeline
        from telemetry_ocr.preview import save_roi_preview

        class FakeRecognizer:
            def recognize(self, images):
                return [OCRResult("12", 0.95) for _ in images]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 180)
            )
            if not writer.isOpened():
                self.skipTest("MJPG encoder is unavailable")
            for index in range(15):
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "12",
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (180, 180, 180),
                    2,
                )
                writer.write(frame)
            writer.release()

            config = AppConfig(
                runtime=RuntimeConfig(
                    ocr_hz=10,
                    realtime=True,
                    confidence_threshold=0.8,
                    queue_size=2,
                ),
                fields=[
                    FieldConfig(
                        field_id="speed",
                        box=(0.05, 0.15, 0.35, 0.45),
                        allowed_chars="0123456789",
                        regex=r"^\d+$",
                        data_type="integer",
                        profile="fast",
                    )
                ],
            )
            jsonl = root / "output.jsonl"
            csv_path = root / "output.csv"
            summary = TelemetryPipeline(config, FakeRecognizer()).run(video, jsonl, csv_path)
            rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(summary.processed_frames, 4)
            self.assertEqual(summary.emitted_records, len(rows))
            self.assertTrue(any(row["confirmed"] for row in rows))
            self.assertEqual(rows[-1]["value"], 12)
            self.assertGreaterEqual(summary.latency_p95_ms, 0)
            preview = save_roi_preview(video, config, root / "preview.jpg", frame_index=2)
            self.assertTrue(preview.is_file())

    def test_calibration_and_crop_export_on_synthetic_video(self):
        import csv

        import cv2
        import numpy as np

        from telemetry_ocr.calibration import calibrate, export_validation_crops
        from telemetry_ocr.config import AppConfig, FieldConfig, RecipeConfig, RuntimeConfig, load_config
        from telemetry_ocr.ocr import OCRResult

        class FakeRecognizer:
            def recognize(self, images):
                return [OCRResult("12", 0.9) for _ in images]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 90)
            )
            if not writer.isOpened():
                self.skipTest("MJPG encoder is unavailable")
            for _ in range(6):
                writer.write(np.full((90, 160, 3), 80, dtype=np.uint8))
            writer.release()
            config = AppConfig(
                runtime=RuntimeConfig(realtime=False),
                fields=[
                    FieldConfig(
                        field_id="speed",
                        box=(0.1, 0.1, 0.4, 0.4),
                        allowed_chars="0123456789",
                        regex=r"^\d+$",
                        data_type="integer",
                    )
                ],
            )
            labels = root / "labels.csv"
            labels.write_text(
                "frame_index,field_id,expected\n0,speed,12\n5,speed,12\n", encoding="utf-8"
            )
            output_config = root / "calibrated.yaml"
            report_path = root / "report.json"
            report = calibrate(
                video,
                config,
                FakeRecognizer(),
                output_config,
                report_path,
                labels_path=labels,
                recipes=[
                    RecipeConfig(name="gray", channel="gray", clahe=False, unsharp=False),
                    RecipeConfig(name="lab", channel="lab_l", clahe=True, unsharp=False),
                ],
            )
            self.assertEqual(report["processed_frames"], 2)
            self.assertEqual(report["fields"]["speed"]["recipes"]["gray"]["exact_accuracy"], 1)
            self.assertEqual(load_config(output_config).fields[0].primary.name, "gray")

            manifest = export_validation_crops(video, config, root / "validation", crop_count=4)
            with manifest.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all((manifest.parent / row["crop_path"]).is_file() for row in rows))


if __name__ == "__main__":
    unittest.main()
