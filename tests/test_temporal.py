import unittest

from telemetry_ocr.config import FieldConfig, TemporalProfileConfig
from telemetry_ocr.temporal import FieldStabilizer
from telemetry_ocr.validation import ValidatedValue


def valid(text, value):
    return ValidatedValue(text, text, value, True)


class TemporalTests(unittest.TestCase):
    def make_field(self, **kwargs):
        params = dict(
            field_id="speed",
            box=(0.0, 0.0, 0.1, 0.1),
            allowed_chars="0123456789",
            regex=r"^\d+$",
            data_type="integer",
            profile="fast",
        )
        params.update(kwargs)
        return FieldConfig(**params)

    def test_fast_requires_two_of_three(self):
        stabilizer = FieldStabilizer(self.make_field(), TemporalProfileConfig(3, 2))
        first = stabilizer.add(valid("10", 10), 0.9, 0)
        second = stabilizer.add(valid("10", 10), 0.8, 80)
        self.assertFalse(first.confirmed)
        self.assertTrue(second.confirmed)
        self.assertEqual(second.value, 10)

    def test_invalid_observation_retains_last_value(self):
        stabilizer = FieldStabilizer(self.make_field(), TemporalProfileConfig(3, 2))
        stabilizer.add(valid("10", 10), 0.9, 0)
        stabilizer.add(valid("10", 10), 0.9, 80)
        invalid = ValidatedValue("bad", "bad", None, False)
        result = stabilizer.add(invalid, 0.99, 160)
        self.assertEqual(result.value, 10)

    def test_invalid_observations_age_out_confirmation(self):
        stabilizer = FieldStabilizer(self.make_field(), TemporalProfileConfig(3, 2))
        stabilizer.add(valid("10", 10), 0.9, 0)
        stabilizer.add(valid("10", 10), 0.9, 80)
        invalid = ValidatedValue("bad", "bad", None, False)
        stabilizer.add(invalid, 0.9, 160)
        result = stabilizer.add(invalid, 0.9, 240)
        self.assertFalse(result.confirmed)
        self.assertEqual(result.value, 10)

    def test_changing_numeric_values_use_multi_frame_median(self):
        stabilizer = FieldStabilizer(self.make_field(), TemporalProfileConfig(3, 2))
        first = stabilizer.add(valid("10", 10), 0.9, 0)
        second = stabilizer.add(valid("11", 11), 0.9, 100)
        third = stabilizer.add(valid("100", 100), 0.9, 200)
        self.assertFalse(first.confirmed)
        self.assertTrue(second.confirmed)
        self.assertEqual(second.value, 11)
        self.assertTrue(third.confirmed)
        self.assertEqual(third.value, 11)

    def test_delta_rule_rejects_implausible_jump(self):
        stabilizer = FieldStabilizer(
            self.make_field(max_delta_per_second=10), TemporalProfileConfig(2, 2)
        )
        stabilizer.add(valid("10", 10), 0.9, 0)
        stabilizer.add(valid("10", 10), 0.9, 100)
        stabilizer.add(valid("100", 100), 0.9, 200)
        result = stabilizer.add(valid("100", 100), 0.9, 300)
        self.assertFalse(result.confirmed)
        self.assertEqual(result.value, 10)


if __name__ == "__main__":
    unittest.main()
