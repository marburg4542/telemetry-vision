"""StreamService, driven by a fake camera.

The real thing needs an RTSP server, a GPU and PaddleOCR. What is worth testing
here needs none of those: that telemetry reaches a viewer as it is produced,
that a preview is a real JPEG, that a camera which never opens is reported
rather than silently absent, and that stopping actually stops.

Only VideoCapture is faked. Everything else -- imencode, resize, the constants --
is the real OpenCV, because a preview that claims to be a JPEG and is not would
pass a test written against a fake encoder.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

HAS_IMAGE_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"))

URL = "rtsp://camera.invalid/stream"


def _config() -> Any:
    from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig

    return AppConfig(
        runtime=RuntimeConfig(confidence_threshold=0.5, warmup_iterations=0, ocr_hz=30.0),
        fields=[
            FieldConfig(
                field_id="altitude",
                box=(0.1, 0.1, 0.4, 0.3),
                allowed_chars="0123456789.",
                regex=r"^\d+(\.\d+)?$",
                data_type="float",
                profile="fast",
            )
        ],
    )


class _Recognizer:
    def __init__(self, text: str = "120.5") -> None:
        self.text = text
        self.calls = 0

    def recognize(self, images: Any) -> Any:
        from telemetry_ocr.ocr import OCRResult

        self.calls += 1
        return [OCRResult(self.text, 0.97) for _ in images]


class _FakeCapture:
    """A camera that hands out real frames until told it has none left."""

    def __init__(self, frames: int, opens: bool = True) -> None:
        import numpy as np

        self.remaining = frames
        self.opens = opens
        self.released = False
        # Distinguishable content, so a preview that encodes the wrong buffer
        # cannot pass by accident.
        self._frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        self._frame[60:120, 80:200] = 210

    def isOpened(self) -> bool:
        return self.opens

    def get(self, prop: int) -> float:
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def grab(self) -> bool:
        # A little real time per frame, so the decoder's scheduling window
        # advances and frames after the first are actually retrieved.
        time.sleep(0.005)
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True

    def retrieve(self) -> tuple[bool, Any]:
        return True, self._frame.copy()

    def release(self) -> None:
        self.released = True


class _StreamCv2:
    """Real OpenCV, except that cameras are imaginary."""

    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2
        self.frames = 400
        self.opens = True
        self.captures: list[_FakeCapture] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cv2, name)

    def VideoCapture(
        self, source: str, api: int | None = None, params: list[int] | None = None
    ) -> _FakeCapture:
        capture = _FakeCapture(self.frames, self.opens)
        self.captures.append(capture)
        return capture


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class StreamServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cv2 = _StreamCv2()
        patch = mock.patch.dict(sys.modules, {"cv2": self.cv2})
        patch.start()
        self.addCleanup(patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _service(self, **kwargs: Any) -> Any:
        from telemetry_ocr.stream_service import StreamService

        service = StreamService(
            _config(),
            _Recognizer(),
            URL,
            jsonl_path=self.root / "telemetry.jsonl",
            csv_path=self.root / "telemetry.csv",
            **kwargs,
        )
        self.addCleanup(service.stop, 5.0)
        return service

    def _wait_for(self, predicate: Any, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def test_a_url_that_is_not_a_stream_is_refused_up_front(self) -> None:
        # Handing a file path to the stream service would silently get file
        # behaviour -- pacing, container timestamps, ending at EOF -- under a
        # name that promises none of it.
        from telemetry_ocr.stream_service import StreamService

        with self.assertRaises(ValueError):
            StreamService(
                _config(),
                _Recognizer(),
                "clip.mp4",
                jsonl_path=self.root / "a.jsonl",
                csv_path=self.root / "a.csv",
            )

    def test_telemetry_is_published_as_it_is_produced(self) -> None:
        service = self._service()
        service.start()
        self.assertTrue(self._wait_for(lambda: service.snapshot().revision >= 3))
        snapshot = service.snapshot()
        self.assertIn("altitude", snapshot.fields)
        self.assertEqual(snapshot.fields["altitude"]["value"], 120.5)
        self.assertGreater(snapshot.fields["altitude"]["confidence"], 0.9)

    def test_the_revision_only_moves_forward(self) -> None:
        # The viewer long-polls on this number. If it ever repeated or went
        # backwards a browser would either miss an update or spin.
        service = self._service()
        service.start()
        self.assertTrue(self._wait_for(lambda: service.snapshot().revision >= 5))
        seen = []
        for _ in range(5):
            seen.append(service.snapshot().revision)
            time.sleep(0.05)
        self.assertEqual(seen, sorted(seen))

    def test_the_preview_is_a_real_decodable_jpeg(self) -> None:
        import cv2
        import numpy as np

        service = self._service()
        service.start()
        self.assertTrue(self._wait_for(lambda: service.preview()[0] is not None))
        payload, revision = service.preview()
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload[:2], b"\xff\xd8")
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[:2], (240, 320))
        self.assertGreater(revision, 0)

    def test_detection_gets_the_same_frames(self) -> None:
        # This callback is how YOLO is fed without this module knowing torch
        # exists. If it stopped firing, detection would go quiet on streams only.
        seen: list[Any] = []
        service = self._service(on_frame=seen.append)
        service.start()
        self.assertTrue(self._wait_for(lambda: len(seen) >= 3))
        self.assertTrue(all(packet.frame is not None for packet in seen))

    def test_a_camera_that_never_opens_is_reported(self) -> None:
        # Not raising into a thread nobody is watching: the PWA has to be able to
        # say why there is no picture.
        self.cv2.opens = False
        service = self._service()
        service.start()
        self.assertTrue(self._wait_for(lambda: service.status()["error"] is not None))
        status = service.status()
        self.assertIn(URL, status["error"])
        self.assertFalse(status["running"])

    def test_stopping_ends_the_run_and_releases_the_camera(self) -> None:
        service = self._service()
        service.start()
        self.assertTrue(self._wait_for(lambda: service.snapshot().revision >= 2))
        service.stop(timeout=10.0)
        self.assertFalse(service.running)
        self.assertTrue(all(capture.released for capture in self.cv2.captures))

    def test_updates_wake_the_web_layer(self) -> None:
        # The server long-polls on this notification rather than sleeping, so a
        # missing call would turn every viewer into a timeout-length delay.
        woken = threading.Event()
        service = self._service(on_update=woken.set)
        service.start()
        self.assertTrue(woken.wait(timeout=10.0))

    def test_telemetry_is_still_written_to_disk(self) -> None:
        # The web UI is a viewer, not the record. A stream that only ever
        # existed on screen would leave nothing to check afterwards.
        service = self._service()
        service.start()
        self.assertTrue(self._wait_for(lambda: service.snapshot().revision >= 3))
        service.stop(timeout=10.0)
        written = (self.root / "telemetry.jsonl").read_text(encoding="utf-8").strip()
        self.assertTrue(written)
        self.assertIn("altitude", written.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
