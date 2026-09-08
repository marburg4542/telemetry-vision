"""The line torch must not cross, tested where it is actually drawn.

PaddlePaddle and PyTorch each bundle cuDNN 9, and on Windows the second one
loaded fails with WinError 127. So the OCR process hides torch from itself and
YOLO runs in a spawn child. The consequence that is easy to forget: everything
sent back across that boundary has to be a plain builtin, because unpickling
imports the module that defines the class. `torch.__version__` is a str
subclass; returning it would drag torch into the Paddle process and break OCR
after the model had already loaded and looked fine.

DetectionBoundaryTests already guards the queue, but it drives a fake worker, so
it never reaches `UltralyticsDetector.metadata()` -- the one function that
actually touches torch and ultralytics objects and coerces them by hand. Adding
a field there without a str()/int() around it would break OCR at runtime with a
green suite. These tests reach that function with torch-shaped stand-ins.
"""

from __future__ import annotations

import pickle
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock


class _TorchString(str):
    """A str subclass, the way torch.__version__ and a device repr are.

    Pickling one of these carries a reference to this module, which is the whole
    hazard: the receiving process imports it to rebuild the value.
    """


class _TorchInt(int):
    """An int subclass, the way a tensor's .item() result can be."""


class _FakeModel:
    task = _TorchString("detect")
    names: ClassVar[dict] = {1: _TorchString("car"), 0: _TorchString("bus")}
    ckpt: ClassVar[dict] = {
        "train_args": {
            "model": _TorchString("yolov8n.pt"),
            "epochs": _TorchInt(50),
            # Ultralytics stores this as a list on some checkpoints.
            "imgsz": [_TorchInt(640), _TorchInt(640)],
        }
    }


def _detector() -> Any:
    from telemetry_ocr.detection import UltralyticsDetector

    detector = UltralyticsDetector.__new__(UltralyticsDetector)
    detector.model = _FakeModel()
    detector.device = _TorchString("cuda:0")
    return detector


def _plain(value: Any) -> bool:
    """Whether this value is of an exact builtin type, not a subclass of one."""
    if value is None:
        return True
    return type(value) in (bool, int, float, str)


class MetadataCoercionTests(unittest.TestCase):
    def test_every_metadata_value_is_a_plain_builtin(self) -> None:
        metadata = _detector().metadata()
        for key, value in metadata.items():
            if isinstance(value, list):
                for item in value:
                    self.assertTrue(
                        _plain(item),
                        f"{key} carries a {type(item).__name__}, which would import its module",
                    )
                continue
            self.assertTrue(
                _plain(value),
                f"{key} is a {type(value).__name__}, not a plain builtin",
            )

    def test_it_survives_a_round_trip_without_importing_this_module(self) -> None:
        """The real test of the invariant, rather than a proxy for it.

        A pickle of the metadata must not name the module that defines the
        stand-in classes above. If it does, the receiving process -- the one
        holding Paddle -- imports it, and in production that module is torch.
        """
        payload = pickle.dumps(_detector().metadata())
        self.assertNotIn(
            b"test_torch_boundary",
            payload,
            "the metadata pickles a reference to a non-builtin class",
        )
        self.assertEqual(pickle.loads(payload), _detector().metadata())

    def test_the_values_are_still_right_after_coercion(self) -> None:
        # Coercion that loses the answer is not a fix.
        metadata = _detector().metadata()
        self.assertEqual(metadata["classes"], ["bus", "car"])
        self.assertEqual(metadata["architecture"], "yolov8n")
        self.assertEqual(metadata["epochs"], 50)
        self.assertEqual(metadata["trained_image_size"], 640)
        self.assertEqual(metadata["task"], "detect")
        self.assertEqual(metadata["device"], "cuda:0")

    def test_a_checkpoint_with_no_train_args_still_answers(self) -> None:
        # A cosmetic field must not be able to stop a model loading.
        from telemetry_ocr.detection import UltralyticsDetector

        class _Bare:
            task = "detect"
            names: ClassVar[list] = ["car"]
            ckpt = None

        detector = UltralyticsDetector.__new__(UltralyticsDetector)
        detector.model = _Bare()
        detector.device = "cpu"
        metadata = detector.metadata()
        self.assertEqual(metadata["architecture"], "YOLO")
        self.assertIsNone(metadata["epochs"])
        self.assertIsNone(metadata["trained_image_size"])


class TorchEnvTests(unittest.TestCase):
    """vision_lab's side of the same rules. It may import torch; OCR may not."""

    def test_relaxed_loading_puts_torch_load_back_even_on_an_error(self) -> None:
        """The override disables a protection, so the window has to close.

        These checkpoints execute code on load, which is why weights_only=True
        refuses them and why leaving the override installed would apply to every
        later load in the process.
        """
        from vision_lab.torch_env import relaxed_checkpoint_loading

        torch = mock.Mock()
        original = torch.load

        with relaxed_checkpoint_loading(torch):
            self.assertIsNot(torch.load, original)
        self.assertIs(torch.load, original)

        with self.assertRaises(RuntimeError), relaxed_checkpoint_loading(torch):
            raise RuntimeError("checkpoint is corrupt")
        self.assertIs(torch.load, original, "torch.load was left overridden")

    def test_it_relaxes_the_default_without_overriding_a_caller(self) -> None:
        from vision_lab.torch_env import relaxed_checkpoint_loading

        seen: list[Any] = []

        class _Torch:
            @staticmethod
            def load(*args: Any, **kwargs: Any) -> Any:
                seen.append(kwargs.get("weights_only"))
                return "loaded"

        torch = _Torch()
        with relaxed_checkpoint_loading(torch):
            torch.load("best.pt")
            torch.load("best.pt", weights_only=True)
        self.assertEqual(seen, [False, True])

    def test_a_missing_torch_is_reported_not_raised(self) -> None:
        # `vision-lab check` exists to say what is missing. Raising here would
        # take down the diagnostic in the one case it is most needed.
        import builtins

        from vision_lab.torch_env import describe_environment

        real_import = builtins.__import__

        def _no_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", _no_torch):
            report = describe_environment()
        self.assertIsNone(report["torch"])
        self.assertFalse(report["cuda_available"])
        self.assertIn("PyTorch is not installed", report["error"])

    def test_import_ultralytics_names_the_extra_to_install(self) -> None:
        import builtins

        from telemetry_ocr.errors import DependencyError
        from vision_lab.torch_env import import_ultralytics

        real_import = builtins.__import__

        def _no_ultralytics(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("ultralytics"):
                raise ImportError("No module named 'ultralytics'")
            return real_import(name, *args, **kwargs)

        with (
            mock.patch.object(builtins, "__import__", _no_ultralytics),
            self.assertRaises(DependencyError) as caught,
        ):
            import_ultralytics()
        self.assertIn('[ai]', str(caught.exception))


class SidecarRecordTests(unittest.TestCase):
    """detect_sidecar had no tests at all, and it writes a file."""

    def _sidecar(self, tmp: Path) -> Any:
        from telemetry_ocr.detect_sidecar import DetectionSidecar

        sidecar = DetectionSidecar.__new__(DetectionSidecar)
        sidecar._worker = mock.Mock()
        # Closed inside each test, before the temporary directory goes: on
        # Windows an open file cannot be deleted, and the removal fails rather
        # than the test.
        sidecar._handle = (tmp / "detections.jsonl").open("w", encoding="utf-8")
        sidecar.output_path = tmp / "detections.jsonl"
        sidecar._last_written_sequence = -1
        sidecar.submitted = 0
        sidecar.written = 0
        return sidecar

    def test_a_superseded_result_is_not_written_twice(self) -> None:
        """The worker's queue is newest-wins, so drain can see the same one.

        Writing it again would put duplicate lines in detections.jsonl that
        differ in nothing, and inflate `written` past what actually happened.
        """
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            sidecar = self._sidecar(tmp)
            result = {
                "sequence": 4,
                "source_time_ms": 400.0,
                "width": 640,
                "height": 360,
                "inference_ms": 5.0,
                "detections": [],
            }
            sidecar._worker.latest.return_value = result
            sidecar.drain()
            sidecar.drain()
            sidecar.drain()
            sidecar._handle.close()

            lines = [
                line
                for line in (tmp / "detections.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 1)
            self.assertEqual(sidecar.written, 1)
            self.assertEqual(json.loads(lines[0])["sequence"], 4)

    def test_nothing_finished_yet_writes_nothing(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            sidecar = self._sidecar(tmp)
            sidecar._worker.latest.return_value = None
            sidecar.drain()
            sidecar._handle.close()
            self.assertEqual((tmp / "detections.jsonl").read_text(encoding="utf-8"), "")
            self.assertEqual(sidecar.written, 0)

    def test_a_newer_result_is_written_after_an_older_one(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            sidecar = self._sidecar(tmp)
            for sequence in (1, 5, 9):
                sidecar._worker.latest.return_value = {
                    "sequence": sequence,
                    "source_time_ms": sequence * 100.0,
                    "width": 640,
                    "height": 360,
                    "inference_ms": 5.0,
                    "detections": [],
                }
                sidecar.drain()
            sidecar._handle.close()
            lines = (tmp / "detections.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(sidecar.written, 3)


if __name__ == "__main__":
    unittest.main()
