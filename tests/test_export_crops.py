"""Exporting crops for a second labelling pass.

The first pass labelled all twelve fields and reached 95.9%. The gap to the 97%
target is concentrated in three of them, so a second pass wants those three and
wants frames the first pass did not already answer -- otherwise it is four times
the reading for answers that are known, on pictures that have been read before.
"""

from __future__ import annotations

import csv
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

HAS_IMAGE_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"))

TARGET_FIELDS = ["sensor_angle_deg", "eo_detail", "ir_detail"]


def _config() -> Any:
    from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig

    def field(field_id: str, x: float) -> Any:
        return FieldConfig(
            field_id=field_id,
            box=(x, 0.1, 0.15, 0.1),
            allowed_chars="0123456789.x",
            regex=r"^.+$",
        )

    return AppConfig(
        runtime=RuntimeConfig(),
        fields=[
            field("timestamp", 0.05),
            field("eo_detail", 0.25),
            field("ir_detail", 0.45),
            field("sensor_angle_deg", 0.65),
        ],
    )


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class ExportCropsTests(unittest.TestCase):
    def setUp(self) -> None:
        import cv2
        import numpy as np

        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.video = self.root / "clip.mp4"
        writer = cv2.VideoWriter(
            str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360)
        )
        for _ in range(60):
            writer.write(np.full((360, 640, 3), 40, np.uint8))
        writer.release()
        self.assertTrue(self.video.is_file(), "the fixture video was not written")

    def _export(self, name: str, **kwargs: Any) -> list[dict[str, str]]:
        from telemetry_ocr.calibration import export_validation_crops

        manifest = export_validation_crops(
            self.video, _config(), self.root / name, crop_count=24, **kwargs
        )
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_without_a_filter_every_field_is_exported(self) -> None:
        rows = self._export("all")
        self.assertEqual({row["field_id"] for row in rows}, {f.field_id for f in _config().fields})

    def test_fields_narrows_the_export_to_the_ones_named(self) -> None:
        rows = self._export("three", fields=TARGET_FIELDS)
        self.assertEqual({row["field_id"] for row in rows}, set(TARGET_FIELDS))
        # And the crop budget goes to those three rather than being divided by
        # twelve, so the same --crop-count reaches four times as many frames.
        self.assertGreater(len({row["frame_index"] for row in rows}), 1)

    def test_a_field_that_does_not_exist_says_which_ones_do(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._export("typo", fields=["sensor_angle", "ir_detail"])
        message = str(caught.exception)
        self.assertIn("sensor_angle", message)
        self.assertIn("sensor_angle_deg", message, "the message should name the real field")

    def test_the_manifest_matches_the_existing_label_file_layout(self) -> None:
        """So a second pass concatenates onto the first instead of being merged.

        load_labels only reads three of these, but the shipped
        examples/validation_labels_*.csv carry all six, and two files with
        different headers cannot simply be appended.
        """
        from telemetry_ocr.calibration import LABEL_COLUMNS

        rows = self._export("layout", fields=TARGET_FIELDS)
        self.assertEqual(list(rows[0].keys()), LABEL_COLUMNS)

    def test_ocr_raw_is_left_empty_on_export(self) -> None:
        # Deliberate. Showing the labeller what OCR thought before they have
        # decided anchors the answer, and the measured accuracy in the README
        # was produced by reading the picture first.
        rows = self._export("blank", fields=TARGET_FIELDS)
        self.assertTrue(all(row["ocr_raw"] == "" for row in rows))
        self.assertTrue(all(row["expected"] == "" for row in rows))

    def test_already_answered_frames_are_skipped(self) -> None:
        first = self._export("pass1", fields=TARGET_FIELDS)
        answered = {int(first[0]["frame_index"])}
        done = {(frame, field) for frame in answered for field in TARGET_FIELDS}

        second = self._export("pass2", fields=TARGET_FIELDS, already_labelled=done)
        self.assertTrue(second, "the second pass exported nothing at all")
        self.assertNotIn(
            answered.pop(),
            {int(row["frame_index"]) for row in second},
            "a frame the first pass answered was exported again",
        )

    def test_an_export_with_nothing_left_says_what_to_do(self) -> None:
        first = self._export("only", fields=TARGET_FIELDS)
        done = {(int(row["frame_index"]), row["field_id"]) for row in first}

        from telemetry_ocr.errors import VideoError

        with self.assertRaises(VideoError) as caught:
            self._export("empty", fields=TARGET_FIELDS, already_labelled=done)
        self.assertIn("--crop-count", str(caught.exception))


class LabelledPairsTests(unittest.TestCase):
    def _write(self, rows: list[dict[str, str]]) -> Path:
        from telemetry_ocr.calibration import LABEL_COLUMNS

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        path = root / "labels.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in LABEL_COLUMNS})
        return path

    def test_only_answered_rows_count_as_done(self) -> None:
        """A blank `expected` is a question, not an answer.

        An earlier export that was never filled in must be exported again, or
        the second pass would skip exactly the crops still needing a reading.
        """
        from telemetry_ocr.calibration import labelled_pairs

        path = self._write(
            [
                {"frame_index": "0", "field_id": "eo_detail", "expected": "1.0x"},
                {"frame_index": "1", "field_id": "eo_detail", "expected": ""},
                {"frame_index": "2", "field_id": "eo_detail", "expected": "   "},
            ]
        )
        self.assertEqual(labelled_pairs(path), {(0, "eo_detail")})

    def test_no_file_means_nothing_is_done(self) -> None:
        from telemetry_ocr.calibration import labelled_pairs

        self.assertEqual(labelled_pairs(None), set())

    def test_it_reads_the_file_the_project_actually_ships(self) -> None:
        # The real one, so a change to its layout is caught here rather than in
        # a labelling session.
        from telemetry_ocr.calibration import labelled_pairs

        shipped = Path("examples/validation_labels_15-31-25-0.csv")
        if not shipped.is_file():  # pragma: no cover - checkout without examples
            self.skipTest("the shipped labels file is not present")
        pairs = labelled_pairs(shipped)
        self.assertGreater(len(pairs), 800)
        for target in TARGET_FIELDS:
            self.assertTrue(
                any(field == target for _, field in pairs),
                f"{target} has no labels in the shipped file",
            )


if __name__ == "__main__":
    unittest.main()
