import importlib.util
import unittest

HAS_IMAGE_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"))


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class PreprocessTests(unittest.TestCase):
    def test_all_default_recipes_return_three_channels(self):
        import numpy as np

        from telemetry_ocr.preprocess import apply_recipe, default_calibration_recipes

        crop = np.zeros((20, 80, 3), dtype=np.uint8)
        crop[:, 20:60] = (40, 80, 120)
        for recipe in default_calibration_recipes():
            with self.subTest(recipe=recipe.name):
                output = apply_recipe(crop, recipe)
                self.assertEqual(output.ndim, 3)
                self.assertEqual(output.shape[2], 3)


if __name__ == "__main__":
    unittest.main()

