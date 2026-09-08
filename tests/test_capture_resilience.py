"""What survives a capture that ends badly.

A capture is the one thing in this project that writes evidence. The frames are
on disk the moment they are taken; the manifest beside them is what makes them
readable, and `load_session` refuses a directory without it. So an ending that
writes the frames and not the manifest does not lose a little -- it loses all of
it, silently, in exactly the case the capture existed for: a long flight whose
feed dropped at the end.

Two endings are covered here. A decoder that gave up, and a writer thread that
died. Both used to leave a directory full of unreadable JPEGs.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

HAS_IMAGE_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"))


def _config() -> Any:
    from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig

    return AppConfig(
        runtime=RuntimeConfig(
            confidence_threshold=0.5,
            warmup_iterations=0,
            ocr_hz=30.0,
            enable_fallback=False,
            realtime=False,
        ),
        fields=[
            FieldConfig(
                field_id="altitude",
                box=(0.1, 0.1, 0.4, 0.3),
                allowed_chars="0123456789.",
                regex=r"^\d+(\.\d+)?$",
            )
        ],
    )


class _FakeCapture:
    """A camera that yields `frames` frames and then stops grabbing."""

    def __init__(self, frames: int) -> None:
        import numpy as np

        self.frames = frames
        self.grabbed = 0
        self.released = False
        self._image = np.zeros((120, 160, 3), dtype=np.uint8)

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        return 30.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def grab(self) -> bool:
        if self.grabbed >= self.frames:
            return False
        self.grabbed += 1
        return True

    def retrieve(self) -> tuple[bool, Any]:
        return True, self._image.copy()

    def release(self) -> None:
        self.released = True


class _Cv2:
    def __init__(self, frames: int) -> None:
        import cv2

        self._cv2 = cv2
        self.frames = frames

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cv2, name)

    def VideoCapture(self, source: str, api: int | None = None, params: Any = None) -> Any:
        return _FakeCapture(self.frames)


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class _CaptureTestCase(unittest.TestCase):
    frames = 12

    def setUp(self) -> None:
        patch = mock.patch.dict(sys.modules, {"cv2": _Cv2(self.frames)})
        patch.start()
        self.addCleanup(patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Path(self.temp.name) / "session"


class DroppedStreamTests(_CaptureTestCase):
    def test_a_stream_that_never_came_back_still_leaves_a_readable_session(self) -> None:
        """The reason the manifest write moved above the re-raises.

        A 40-minute RTSP capture whose camera dies at minute 40 has every frame
        on disk. Writing the manifest only on the clean path meant none of them
        could be read back.
        """
        from telemetry_ocr.capture import (
            MANIFEST_NAME,
            SESSION_NAME,
            capture_session,
            load_session,
        )
        from telemetry_ocr.errors import VideoError
        from telemetry_ocr.pipeline import VideoDecoder

        real_run = VideoDecoder._run

        def _explode(self_: Any) -> None:
            # Whatever the decoder managed first, it ends by giving up -- which
            # is what _reconnect does when a stream does not come back.
            real_run(self_)
            self_.error = VideoError("Stream did not come back after 4 attempts")

        with (
            mock.patch.object(VideoDecoder, "_run", _explode),
            self.assertRaises(VideoError),
        ):
            capture_session("rtsp://cam/live", self.session, _config(), max_frames=6)

        self.assertTrue((self.session / SESSION_NAME).is_file(), "session.json was not written")
        self.assertTrue((self.session / MANIFEST_NAME).is_file(), "frames.jsonl was not written")

        # The real test: the tooling that reads a capture accepts it. Before
        # this fix load_session raised "is not a capture session" here.
        info, records = load_session(self.session)
        written = sorted((self.session / "frames").iterdir())
        self.assertEqual(len(records), len(written))
        self.assertGreater(len(records), 0)
        self.assertEqual(info.frame_count, len(records))

    def test_the_error_still_reaches_the_caller(self) -> None:
        # Writing the manifest must not turn a failed capture into a silent one.
        from telemetry_ocr.capture import capture_session
        from telemetry_ocr.errors import VideoError
        from telemetry_ocr.pipeline import VideoDecoder

        real_run = VideoDecoder._run

        def _explode(self_: Any) -> None:
            real_run(self_)
            self_.error = VideoError("Stream did not come back after 4 attempts")

        with (
            mock.patch.object(VideoDecoder, "_run", _explode),
            self.assertRaises(VideoError) as caught,
        ):
            capture_session("rtsp://cam/live", self.session, _config(), max_frames=6)
        self.assertIn("did not come back", str(caught.exception))


class StoppedEarlyTests(_CaptureTestCase):
    frames = 0

    def test_a_stop_before_the_first_frame_is_not_a_corrupt_source(self) -> None:
        """Stop and "this file is unreadable" are different answers.

        A camera that accepts the connection but has not published yet, stopped
        by the operator, used to be reported as "Video contains no decodable
        frames" -- and being a decoder error, it also destroyed the manifest.
        """
        import queue as queue_mod
        import threading

        from telemetry_ocr.errors import VideoError
        from telemetry_ocr.pipeline import VideoDecoder

        stopped = threading.Event()
        stopped.set()
        decoder = VideoDecoder(
            video_path="rtsp://cam/live",
            output_queue=queue_mod.Queue(maxsize=1),
            ocr_hz=10.0,
            realtime=False,
            stop_event=stopped,
        )
        decoder._run()
        self.assertIsNone(decoder.error, "a requested stop was reported as a decode fault")

        # And the genuine case -- a source that really yielded nothing -- still
        # raises, so the guard has not disabled the check it guards.
        decoder = VideoDecoder(
            video_path="clip.mp4",
            output_queue=queue_mod.Queue(maxsize=1),
            ocr_hz=10.0,
            realtime=False,
            stop_event=threading.Event(),
        )
        decoder._run()
        self.assertIsInstance(decoder.error, VideoError)
        self.assertIn("no decodable frames", str(decoder.error))


@unittest.skipUnless(HAS_IMAGE_DEPS, "NumPy/OpenCV not installed")
class FrameWriterTests(unittest.TestCase):
    """The writer thread's own failures.

    `submit` is a put on a bounded queue. Once the consumer is gone it blocks
    forever, and capture_session's deadline and max-frames checks are past it in
    the loop -- so the capture never stops and looks alive from the outside.
    """

    def _writer(self) -> Any:
        from telemetry_ocr.capture import _FrameWriter

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        frames = Path(temp.name) / "frames"
        frames.mkdir()
        return _FrameWriter(frames, "jpeg", 95)

    def test_a_dead_writer_raises_on_submit_instead_of_blocking(self) -> None:
        import numpy as np

        writer = self._writer()
        writer.start()
        writer._thread.join(timeout=0)
        # Kill the consumer the way an unhandled error would: stop it, then fill
        # the queue past its depth.
        writer.error = RuntimeError("writer died")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            writer.submit(0, frame, 0.0)

    def test_an_error_outside_the_per_frame_handler_is_kept_not_lost(self) -> None:
        """The import, the shape read and the append are all outside it.

        Before, any of the three killed the thread with no trace: `error` was
        declared on this class and never assigned or read.
        """
        import numpy as np

        writer = self._writer()

        def _boom() -> None:
            raise RuntimeError("cv2 is broken in this process")

        with mock.patch.object(writer, "_write_frames", _boom):
            writer.start()
            writer._thread.join(timeout=5)

        self.assertIsInstance(writer.error, RuntimeError)
        with self.assertRaises(RuntimeError):
            writer.close()
        del np

    def test_a_healthy_writer_still_closes_quietly(self) -> None:
        import numpy as np

        writer = self._writer()
        writer.start()
        writer.submit(0, np.zeros((8, 8, 3), dtype=np.uint8), 0.0)
        writer.close()
        self.assertIsNone(writer.error)
        self.assertEqual(len(writer.records), 1)
        self.assertEqual(writer.write_errors, 0)


if __name__ == "__main__":
    unittest.main()
