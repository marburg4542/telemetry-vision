from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_ocr.errors import ConfigError
from vision_lab import cli, paths

MANIFEST = """
version: 1
asset_root: ../media
asset_dirs:
  videos: clips
  frames: frames
models:
  primary: models/yolo/best.pt
  spare: models/yolo/other.pt
default_model: primary
"""


class ProjectPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name).resolve() / "repo"
        (self.root / "configs").mkdir(parents=True)
        (self.root / "configs" / "paths.yaml").write_text(MANIFEST, encoding="utf-8")
        (self.root / "models" / "yolo").mkdir(parents=True)
        (self.root / "models" / "yolo" / "best.pt").write_bytes(b"checkpoint")
        self.clips = Path(self._temp.name).resolve() / "media" / "clips"
        self.clips.mkdir(parents=True)
        (self.clips / "flight.mp4").write_bytes(b"video")

        import os

        os.environ["TELEMETRY_VISION_ROOT"] = str(self.root)
        os.environ.pop("TELEMETRY_VISION_ASSETS", None)
        self.addCleanup(os.environ.pop, "TELEMETRY_VISION_ROOT", None)
        self.paths = paths.load_paths()

    def test_root_comes_from_the_environment_override(self) -> None:
        self.assertEqual(self.paths.root, self.root)

    def test_asset_root_is_resolved_relative_to_the_repository(self) -> None:
        self.assertEqual(self.paths.asset_root, self.clips.parent)
        self.assertEqual(self.paths.asset_dir("videos"), self.clips)

    def test_environment_variable_overrides_the_configured_asset_root(self) -> None:
        import os

        elsewhere = Path(self._temp.name).resolve() / "elsewhere"
        elsewhere.mkdir()
        os.environ["TELEMETRY_VISION_ASSETS"] = str(elsewhere)
        self.addCleanup(os.environ.pop, "TELEMETRY_VISION_ASSETS", None)
        reloaded = paths.load_paths()
        self.assertEqual(reloaded.asset_root, elsewhere)
        self.assertEqual(reloaded.asset_dir("videos"), elsewhere / "clips")

    def test_registered_name_resolves_to_the_checkpoint(self) -> None:
        self.assertEqual(
            self.paths.resolve_model("primary"), self.root / "models" / "yolo" / "best.pt"
        )

    def test_no_model_argument_falls_back_to_the_default(self) -> None:
        self.assertEqual(
            self.paths.resolve_model(None), self.root / "models" / "yolo" / "best.pt"
        )

    def test_a_registered_name_pointing_at_nothing_is_reported(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            self.paths.resolve_model("spare")
        self.assertIn("primary", str(caught.exception))

    def test_an_unknown_name_lists_the_registered_ones(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            self.paths.resolve_model("typo")
        self.assertIn("Registered names", str(caught.exception))

    def test_a_bare_filename_is_found_in_the_external_video_directory(self) -> None:
        self.assertEqual(self.paths.resolve_video("flight.mp4"), self.clips / "flight.mp4")

    def test_a_file_that_is_not_a_video_is_rejected(self) -> None:
        note = self.clips / "notes.txt"
        note.write_text("not a video", encoding="utf-8")
        with self.assertRaises(ConfigError):
            self.paths.resolve_video(note)

    def test_a_missing_video_reports_every_location_searched(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            self.paths.resolve_video("absent.mp4")
        self.assertIn(str(self.clips), str(caught.exception))

    def test_a_default_model_outside_the_table_is_rejected(self) -> None:
        (self.root / "configs" / "paths.yaml").write_text(
            MANIFEST.replace("default_model: primary", "default_model: ghost"), encoding="utf-8"
        )
        with self.assertRaises(ConfigError):
            paths.load_paths()

    def test_a_root_without_a_manifest_is_rejected(self) -> None:
        import os

        os.environ["TELEMETRY_VISION_ROOT"] = str(self.root / "models")
        with self.assertRaises(ConfigError):
            paths.load_paths()


class CommandLineTests(unittest.TestCase):
    def test_every_subcommand_parses(self) -> None:
        parser = cli.build_parser()
        for argv in (
            ["check"],
            ["models", "--json"],
            ["extract-frames", "--interval", "0.5"],
            ["evaluate", "clip.mp4", "--model", "best50", "--track"],
            ["report", "--run", "visdrone_custom_7th"],
            ["handbook"],
            ["handbook", "--run", "visdrone_custom_6th"],
        ):
            with self.subTest(argv=argv):
                self.assertTrue(parser.parse_args(argv).command)

    def test_importing_the_detection_cli_does_not_pull_in_torch(self) -> None:
        """The heavy stacks stay lazy so `vision-lab --help` is instant.

        It also keeps torch out of any process that has already loaded Paddle,
        which on Windows is what makes the two coexist at all.
        """
        import subprocess

        source_root = Path(__file__).resolve().parents[1] / "src"
        probe = (
            "import sys;"
            "import vision_lab.cli;"
            "print('torch' in sys.modules or 'paddle' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(source_root)},
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
