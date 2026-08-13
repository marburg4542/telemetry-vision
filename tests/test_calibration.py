import unittest

from telemetry_ocr.calibration import RecipeStats, calibration_score


class CalibrationTests(unittest.TestCase):
    def test_labeled_accuracy_is_primary_score(self):
        accurate = RecipeStats(attempts=10, valid=8, labeled=10, correct=8, confidence_sum=7)
        confident_wrong = RecipeStats(
            attempts=10, valid=10, labeled=10, correct=7, confidence_sum=9.9
        )
        self.assertGreater(calibration_score(accurate), calibration_score(confident_wrong))

    def test_unlabeled_uses_valid_rate(self):
        valid = RecipeStats(attempts=10, valid=9, confidence_sum=6)
        confident_invalid = RecipeStats(attempts=10, valid=5, confidence_sum=9)
        self.assertGreater(calibration_score(valid), calibration_score(confident_invalid))


if __name__ == "__main__":
    unittest.main()

