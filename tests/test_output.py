import csv
import json
import tempfile
import unittest
from pathlib import Path

from telemetry_ocr.output import RecordWriter, TelemetryRecord


class OutputTests(unittest.TestCase):
    def test_writes_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = RecordWriter(root / "data.jsonl", root / "data.csv")
            writer.start()
            writer.write(
                TelemetryRecord(100.0, 3, "speed", 12.5, "12.5", 0.9, True, "lab", 8.0)
            )
            writer.close()
            payload = json.loads((root / "data.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(payload["value"], 12.5)
            with (root / "data.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["field_id"], "speed")


if __name__ == "__main__":
    unittest.main()

