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

    def test_optics_recovers_the_mark_the_font_never_renders_cleanly(self):
        """Every number in this overlay ends in a mark -- an "x" or a degree
        sign -- and it is the one glyph in the ROI that is not a digit, so it is
        the one the recognizer gets wrong. All of these are real readings taken
        from captures on disk; before this they were all thrown away."""
        eo, ir = self._optics_fields()
        for raw, expected in [
            ("12.8xFOV58a", "12.8xFOV5.8°"),      # degree sign read as a letter
            ("1.0x FOV 66.4ª", "1.0xFOV66.4°"),
            ("1.0x FOv 66.4º", "1.0xFOV66.4°"),
            ("10.3xFov72”", "10.3xFOV7.2°"),  # and as a quote
            ("10.3x FOV 7.2", "10.3xFOV7.2°"),     # and as nothing at all
            ("400xFOV19\"", "40.0xFOV1.9°"),
            ("L0xF0V 664°", "1.0xFOV66.4°"),   # L for 1, zero for O
            ("2Bx FOV 26.7°", "2.8xFOV26.7°"), # B for 8
            ("3BxFDV 19.5", "3.8xFOV19.5°"),        # FDV for FOV
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(validate_text(raw, eo).value, expected)
        for raw in ("10x", "102", "1.02", "1Ox", "1Dx", "I0x", "10A", "10."):
            with self.subTest(raw=raw):
                self.assertEqual(validate_text(raw, ir).value, "1.0x")

    def test_a_surviving_point_keeps_every_digit_after_it(self):
        """The trailing digit is only the mark when there is no point.

        With one, the digits after it are real: dropping the last of "19.5"
        gave 1.9 and the last of "10.3" gave 1.0.
        """
        eo, ir = self._optics_fields()
        self.assertEqual(validate_text("3 BxFDv 19.5", eo).value, "3.8xFOV19.5°")
        self.assertEqual(validate_text("10.3F0V 72", eo).value, "10.3xFOV7.2°")
        self.assertEqual(validate_text("10.2x", ir).value, "10.2x")

    def test_optics_refuses_a_reading_that_lost_a_digit(self):
        # Nothing here invents a digit. Only the position of the point is
        # reconstructed, and that is fixed by the overlay's one decimal place.
        eo, ir = self._optics_fields()
        for raw in ("19310v72", "20 6s Fov3ré\"", "10.3F"):
            with self.subTest(raw=raw):
                self.assertFalse(validate_text(raw, eo).valid)
        for raw in ("1", "x", ""):
            with self.subTest(raw=raw):
                self.assertFalse(validate_text(raw, ir).valid)

    def _optics_fields(self):
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
        return eo, ir

    def test_a_point_that_was_seen_is_never_shifted(self):
        """The worst failure this file guards against: a number that is wrong
        and passes.

        "646.Om" lost only the zero after the point, but the reader extracted
        "646", found no point in it, and shifted one in -- giving 64.6, which
        sits inside the field's range, reads as confirmed, and is ten times too
        small. It also pre-empted the retry that maps O to 0 and reads it right.
        """
        field = FieldConfig(
            field_id="lrf_m",
            box=(0.0, 0.0, 0.2, 0.1),
            allowed_chars="0123456789.",
            regex=r"^\d{1,4}\.\d$",
            data_type="float",
            normalizer="numeric",
            decimals=1,
            min_value=15,
            max_value=9999,
            profile="fast",
        )
        self.assertEqual(validate_text("646.Om", field).value, 646.0)
        self.assertEqual(validate_text("533：6", field).value, 533.6)
        # And where the digit really is gone, it fails rather than guessing.
        for raw in ("47&.2m", "513.án"):
            with self.subTest(raw=raw):
                self.assertFalse(validate_text(raw, field).valid)

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
