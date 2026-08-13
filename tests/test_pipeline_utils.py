import unittest

from telemetry_ocr.pipeline import percentile


class PipelineUtilityTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([], 95), 0)
        self.assertEqual(percentile([5], 95), 5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 50), 3)


if __name__ == "__main__":
    unittest.main()

