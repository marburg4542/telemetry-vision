import unittest

from telemetry_ocr.config import FieldConfig
from telemetry_ocr.validation import prefer_candidate, validate_text


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.field = FieldConfig(
            field_id="speed",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.",
            regex=r"^\d{1,3}(?:\.\d)?$",
            data_type="float",
            min_value=0,
            max_value=400,
            profile="fast",
        )

    def test_valid_numeric_result(self):
        result = validate_text(" 12.3 ", self.field)
        self.assertTrue(result.valid)
        self.assertEqual(result.value, 12.3)
        self.assertFalse(result.corrected)

    def test_confusion_correction_only_after_failure(self):
        result = validate_text("1O.5", self.field)
        self.assertTrue(result.valid)
        self.assertEqual(result.normalized_text, "10.5")
        self.assertTrue(result.corrected)

    def test_range_failure(self):
        result = validate_text("999", self.field)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "above_maximum")

    def test_decimal_limit(self):
        field = FieldConfig(
            field_id="temperature",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.",
            regex=r"^\d+(?:\.\d+)?$",
            data_type="float",
            decimals=1,
        )
        result = validate_text("12.34", field)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "too_many_decimals")

    def test_validity_beats_confidence(self):
        valid = validate_text("12", self.field)
        invalid = validate_text("ABC", self.field)
        selected = prefer_candidate((invalid, 0.99), (valid, 0.60))
        self.assertIs(selected[0], valid)

    def test_numeric_normalizer_strips_unit_noise(self):
        field = FieldConfig(
            field_id="range",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.",
            regex=r"^\d{1,4}\.\d$",
            data_type="float",
            normalizer="numeric",
            decimals=1,
        )
        result = validate_text("307.4元", field)
        self.assertTrue(result.valid)
        self.assertEqual(result.value, 307.4)

    def test_numeric_normalizer_repairs_fixed_precision_osd_confusions(self):
        field = FieldConfig(
            field_id="angle",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="-0123456789.",
            regex=r"^-?\d{1,3}\.\d{2}$",
            data_type="float",
            normalizer="numeric",
            decimals=2,
            min_value=-180,
            max_value=180,
        )
        self.assertEqual(validate_text("=0726", field).value, -0.26)
        self.assertEqual(validate_text("1.851", field).value, 1.85)
        self.assertEqual(validate_text("-24.9%", field).value, -24.96)
        self.assertEqual(validate_text("-2496", field).value, -24.96)
        self.assertEqual(validate_text("-3.6", field).value, -3.6)

    def test_missing_decimal_point_is_resolved_only_on_evidence(self):
        """A wrong guess here is a 10x error, so ambiguity must fail the read.

        Every case below was taken from the 25,095 fixed-precision reads in
        output/real/telemetry-9.78hz.jsonl.
        """
        angle = FieldConfig(
            field_id="sensor_angle_deg",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="+-0123456789.",
            regex=r"^[+-]?\d{1,3}\.\d{2}$",
            data_type="float",
            normalizer="numeric",
            decimals=2,
            min_value=-180,
            max_value=180,
        )

        # The overlay prints 0.26, never 07.26, so the leading zero decides it.
        self.assertEqual(validate_text("0726", angle).value, 0.26)
        self.assertEqual(validate_text("=0726", angle).value, -0.26)
        # 527.27 is outside the field's range, so only 52.27 survives.
        self.assertEqual(validate_text("52727", angle).value, 52.27)
        # 17.50 and 1.50 are both well-formed and in range: refuse to choose.
        for ambiguous in ("1750", "-2784", "8789"):
            result = validate_text(ambiguous, angle)
            self.assertFalse(result.valid, ambiguous)
            self.assertIsNone(result.value, ambiguous)
        # A digit other than 7 at the boundary was never ambiguous.
        self.assertEqual(validate_text("-2496", angle).value, -24.96)
        self.assertEqual(validate_text("1234", angle).value, 12.34)

    def test_numeric_normalizer_recovers_missing_altitude_decimal(self):
        field = FieldConfig(
            field_id="altitude",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.",
            regex=r"^\d{1,4}\.\d{3}$",
            data_type="float",
            normalizer="numeric",
            decimals=3,
            min_value=0,
            max_value=10000,
        )
        self.assertEqual(validate_text("359 000m", field).value, 359.0)
        self.assertEqual(validate_text("310-300m", field).value, 310.3)
        self.assertEqual(validate_text("-0.000m", field).value, 0.0)

    def test_optics_normalizer_repairs_zoom_and_fov_punctuation(self):
        eo = FieldConfig(
            field_id="eo_detail",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.xFOV°",
            regex=r"^\d{1,2}\.\dxFOV\d{1,3}\.\d°$",
            normalizer="optics",
            profile="fast",
        )
        ir = FieldConfig(
            field_id="ir_detail",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.x",
            regex=r"^\d{1,2}\.\dx$",
            normalizer="optics",
            profile="fast",
        )
        self.assertEqual(validate_text("1.0x F0V 664°", eo).value, "1.0xFOV66.4°")
        self.assertEqual(validate_text("40-0x FOV 1.9°", eo).value, "40.0xFOV1.9°")
        self.assertEqual(validate_text("1:0x", ir).value, "1.0x")
        self.assertEqual(validate_text("1-0x", ir).value, "1.0x")

    def test_dms_normalizer_uses_fixed_digit_layout(self):
        field = FieldConfig(
            field_id="latitude",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.°'\"",
            regex=r"^\d{1,2}°\d{2}'\d{2}\.\d{2}\"$",
            data_type="dms",
            normalizer="dms",
            dms_degrees_digits=2,
            dms_decimal_places=2,
            hemisphere="N",
        )
        result = validate_text("14°52<17 80″", field)
        self.assertTrue(result.valid)
        self.assertEqual(result.normalized_text, "14°52'17.80\"")
        self.assertAlmostEqual(result.value, 14 + 52 / 60 + 17.8 / 3600)

    def test_dms_normalizer_repairs_quote_read_as_trailing_two(self):
        field = FieldConfig(
            field_id="latitude",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.°'\"",
            regex=r"^\d{1,2}°\d{2}'\d{2}\.\d{2}\"$",
            data_type="dms",
            normalizer="dms",
            dms_degrees_digits=2,
            dms_decimal_places=2,
            hemisphere="N",
        )
        result = validate_text("14°52'17.802", field)
        self.assertTrue(result.valid)
        self.assertEqual(result.normalized_text, "14°52'17.80\"")

    def test_dms_normalizer_canonicalizes_misread_symbols(self):
        longitude = FieldConfig(
            field_id="longitude",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.°'\"",
            regex=r"^\d{1,3}°\d{2}'\d{2}\.\d{2}\"$",
            data_type="dms",
            normalizer="dms",
            dms_degrees_digits=3,
            dms_decimal_places=2,
            hemisphere="E",
        )
        latitude = FieldConfig(
            field_id="latitude",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.°'\"",
            regex=r"^\d{1,2}°\d{2}'\d{2}\.\d{2}\"$",
            data_type="dms",
            normalizer="dms",
            dms_degrees_digits=2,
            dms_decimal_places=2,
            hemisphere="N",
        )
        self.assertEqual(
            validate_text('102°00/27.10"', longitude).normalized_text,
            '102°00\'27.10"',
        )
        self.assertEqual(
            validate_text("14*50′27.01", latitude).normalized_text,
            '14°50\'27.01"',
        )


if __name__ == "__main__":
    unittest.main()
