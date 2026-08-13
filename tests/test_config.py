import unittest

from telemetry_ocr.config import ConfigError, parse_config


def minimal_data():
    return {
        "version": 1,
        "runtime": {"ocr_hz": 12, "queue_size": 2},
        "fields": [
            {
                "field_id": "speed",
                "box": [0.1, 0.2, 0.2, 0.1],
                "allowed_chars": "0123456789.",
                "regex": r"^\d{1,3}(?:\.\d)?$",
                "data_type": "float",
                "decimals": 1,
                "min_value": 0,
                "max_value": 400,
                "profile": "fast",
            }
        ],
    }


class ConfigTests(unittest.TestCase):
    def test_parses_defaults_and_box(self):
        config = parse_config(minimal_data())
        self.assertEqual(config.fields[0].box, (0.1, 0.2, 0.2, 0.1))
        self.assertEqual(config.runtime.model_name, "PP-OCRv6_tiny_rec")
        self.assertEqual(config.runtime.model_source, "BOS")
        self.assertEqual(config.runtime.model_cache_dir, "models/paddlex")
        self.assertEqual(config.temporal.fast.required, 2)

    def test_rejects_box_outside_frame(self):
        data = minimal_data()
        data["fields"][0]["box"] = [0.9, 0.2, 0.2, 0.1]
        with self.assertRaises(ConfigError):
            parse_config(data)

    def test_rejects_duplicate_ids(self):
        data = minimal_data()
        data["fields"].append(dict(data["fields"][0]))
        with self.assertRaises(ConfigError):
            parse_config(data)

    def test_rejects_bad_recipe_kernel(self):
        data = minimal_data()
        data["fields"][0]["primary"] = {"morphology_kernel": 4}
        with self.assertRaises(ConfigError):
            parse_config(data)

    def test_source_time_derivation_requires_timestamp_string(self):
        data = minimal_data()
        data["fields"][0]["derive_from_source_time"] = True
        with self.assertRaises(ConfigError):
            parse_config(data)

    def test_rejects_ocr_phase_outside_interval(self):
        data = minimal_data()
        data["fields"][0]["ocr_interval"] = 2
        data["fields"][0]["ocr_phase"] = 2
        with self.assertRaises(ConfigError):
            parse_config(data)


if __name__ == "__main__":
    unittest.main()
