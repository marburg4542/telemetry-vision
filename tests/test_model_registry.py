from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_ocr.errors import ConfigError
from vision_lab import paths
from vision_lab.model_registry import ModelRegistry

MANIFEST = """# Comments here are why the registry is never rewritten.
version: 1
asset_root: .
models:
  primary: models/yolo/best.pt
  spare: models/yolo/other.pt
default_model: primary
"""


class ModelRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name).resolve() / "repo"
        (self.root / "configs").mkdir(parents=True)
        self.manifest = self.root / "configs" / "paths.yaml"
        self.manifest.write_text(MANIFEST, encoding="utf-8")
        for relative in (
            "models/yolo/best.pt",
            "models/yolo/extra.pt",
            "runs/exp_2/weights/best.pt",
            "runs/exp_2/weights/last.pt",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"checkpoint")
        os.environ["TELEMETRY_VISION_ROOT"] = str(self.root)
        self.addCleanup(os.environ.pop, "TELEMETRY_VISION_ROOT", None)
        self.registry = ModelRegistry(self.root)

    def candidate(self, label: str):
        return next(item for item in self.registry.candidates() if item.label == label)

    def test_candidates_are_unregistered_checkpoints_with_usable_names(self) -> None:
        offered = {item.label: item.suggested_name for item in self.registry.candidates()}
        self.assertEqual(
            offered,
            {
                "models/yolo/extra.pt": "extra",
                "runs/exp_2/weights/best.pt": "exp_2",
                "runs/exp_2/weights/last.pt": "exp_2-last",
            },
        )

    def test_adding_registers_the_file_without_touching_paths_yaml(self) -> None:
        entry = self.registry.add(self.candidate("runs/exp_2/weights/best.pt").id, "run8")
        self.assertEqual(entry.path, self.root / "runs/exp_2/weights/best.pt")
        self.assertIn("run8", paths.load_paths().models)
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), MANIFEST)
        self.assertNotIn(
            "runs/exp_2/weights/best.pt", [item.label for item in self.registry.candidates()]
        )

    def test_only_a_discovered_candidate_can_be_added(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.add("not-an-id", "sneaky")
        with self.assertRaises(ValueError):
            self.registry.add("../../etc/passwd", "sneaky")

    def test_a_name_must_be_valid_and_unused(self) -> None:
        candidate = self.candidate("models/yolo/extra.pt")
        for bad in ("", "has space", "../up", "-leading", "x" * 41):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                self.registry.add(candidate.id, bad)
        with self.assertRaises(ValueError):
            self.registry.add(candidate.id, "primary")

    def test_removing_forgets_the_model_and_keeps_the_file(self) -> None:
        self.registry.remove("primary")
        self.assertNotIn("primary", paths.load_paths().models)
        self.assertTrue((self.root / "models/yolo/best.pt").is_file())
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), MANIFEST)
        # Forgotten, so it is on offer again.
        self.assertIn("models/yolo/best.pt", [item.label for item in self.registry.candidates()])

    def test_removing_an_added_model_drops_it_from_the_overlay(self) -> None:
        self.registry.add(self.candidate("models/yolo/extra.pt").id, "extra")
        self.registry.remove("extra")
        overlay = paths.read_overlay(self.root)
        self.assertEqual(overlay["added"], {})
        self.assertEqual(overlay["removed"], [])

    def test_removing_the_default_moves_it_to_a_model_that_remains(self) -> None:
        self.registry.remove("primary")
        self.assertEqual(self.registry.default(), "spare")

    def test_a_removed_base_name_can_be_added_again_to_another_file(self) -> None:
        self.registry.remove("primary")
        self.registry.add(self.candidate("models/yolo/extra.pt").id, "primary")
        self.assertEqual(paths.load_paths().models["primary"], self.root / "models/yolo/extra.pt")

    def test_the_default_can_be_changed_and_persists(self) -> None:
        self.registry.set_default("spare")
        self.assertEqual(ModelRegistry(self.root).default(), "spare")
        self.assertEqual(paths.load_paths().default_model, "spare")
        with self.assertRaises(ValueError):
            self.registry.set_default("ghost")

    def test_an_overlay_default_that_no_longer_exists_falls_back_to_the_base(self) -> None:
        (self.root / paths.OVERLAY_RELATIVE_PATH).write_text("default: ghost\n", encoding="utf-8")
        self.assertEqual(paths.load_paths().default_model, "primary")

    def test_entries_report_presence_and_the_default(self) -> None:
        listed = {entry.name: entry for entry in self.registry.entries()}
        self.assertTrue(listed["primary"].available)
        self.assertTrue(listed["primary"].default)
        self.assertFalse(listed["spare"].available)
        self.assertEqual(listed["primary"].label, "models/yolo/best.pt")

    def test_the_base_registry_alone_ignores_the_overlay(self) -> None:
        self.registry.remove("spare")
        self.assertIn("spare", paths.load_paths(overlay=False).models)

    def test_a_corrupt_overlay_is_reported(self) -> None:
        (self.root / paths.OVERLAY_RELATIVE_PATH).write_text("added: [oops", encoding="utf-8")
        with self.assertRaises(ConfigError):
            paths.load_paths()

    def test_importing_the_registry_loads_no_ml_framework(self) -> None:
        # The web server holding PaddleOCR imports this module.
        source_root = Path(__file__).resolve().parents[1] / "src"
        probe = (
            "import sys; import vision_lab.model_registry;"
            "print('torch' in sys.modules or 'paddle' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(source_root)},
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
