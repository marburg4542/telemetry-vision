import unittest
from typing import ClassVar

from telemetry_ocr.errors import RecognitionError
from telemetry_ocr.ocr import OCRResult, ParallelRecognizer, extract_ocr_result


class FakeResult:
    json: ClassVar[dict] = {"res": {"rec_text": "123", "rec_score": 0.91}}


class OCRResultTests(unittest.TestCase):
    def test_parallel_recognizer_preserves_batch_order(self):
        class PrefixRecognizer:
            def __init__(self, prefix):
                self.prefix = prefix

            def recognize(self, images):
                return [OCRResult(f"{self.prefix}{value}", 1.0) for value in images]

        recognizer = ParallelRecognizer([PrefixRecognizer("a"), PrefixRecognizer("b")])
        results = recognizer.recognize([0, 1, 2, 3, 4])
        self.assertEqual([item.text for item in results], ["a0", "a1", "a2", "b3", "b4"])

    def test_extracts_nested_result(self):
        result = extract_ocr_result(FakeResult())
        self.assertEqual(result.text, "123")
        self.assertAlmostEqual(result.confidence, 0.91)

    def test_extracts_json_string(self):
        result = extract_ocr_result('{"rec_text":"RPM","rec_score":0.8}')
        self.assertEqual(result.text, "RPM")

    def test_rejects_missing_fields(self):
        with self.assertRaises(RecognitionError):
            extract_ocr_result({"res": {"other": 1}})


if __name__ == "__main__":
    unittest.main()
