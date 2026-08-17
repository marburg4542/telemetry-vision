"""The stream paths of VideoDecoder, exercised without a network source.

A live feed differs from a file in ways that only show up when something goes
wrong -- it drops and comes back, it delivers corrupt packets, its container
timestamps restart at zero. None of that can be reproduced by pointing the
decoder at an mp4, so the capture is faked and made to misbehave on purpose.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import unittest
from typing import ClassVar
from unittest import mock

from telemetry_ocr.errors import VideoError
from telemetry_ocr.pipeline import _END, FramePacket, VideoDecoder, is_stream_source


class SourceClassificationTests(unittest.TestCase):
    def test_network_schemes_are_streams(self) -> None:
        for url in (
            "rtsp://cam/stream",
            "RTSP://CAM/STREAM",
            "rtsps://cam/stream",
            "http://host/live.m3u8",
            "udp://239.0.0.1:1234",
            "srt://host:9000",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_stream_source(url))

    def test_files_are_not_streams(self) -> None:
        for path in ("clip.mp4", r"C:\videos\clip.mp4", "/mnt/c/videos/clip.mp4", "./a.mkv"):
            with self.subTest(path=path):
                self.assertFalse(is_stream_source(path))


class _FakeCapture:
    """Scripted VideoCapture. Each script entry is one grab() outcome."""

    instances: ClassVar[list[_FakeCapture]] = []

    # The decoder only retrieves a frame once the clock has moved past
    # 500/fps ms since the last one. A fake that returns instantly would have
    # every frame after the first fall inside that window and never be
    # retrieved, so grab() lets a little real time pass and the reported frame
    # rate keeps the window narrow.
    grab_delay_s = 0.004

    def __init__(self, script: list[str], fps: float = 1000.0) -> None:
        self.script = list(script)
        self.fps = fps
        self.position = 0
        self.released = False
        self.buffersize: int | None = None
        _FakeCapture.instances.append(self)

    def isOpened(self) -> bool:  # mirrors the OpenCV API
        return True

    def get(self, prop: int) -> float:
        if prop == _FAKE_CV2.CAP_PROP_FPS:
            return self.fps
        if prop == _FAKE_CV2.CAP_PROP_POS_MSEC:
            return float(self.position) * 1000.0 / self.fps
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == _FAKE_CV2.CAP_PROP_BUFFERSIZE:
            self.buffersize = int(value)
        return True

    def grab(self) -> bool:
        time.sleep(self.grab_delay_s)
        if self.position >= len(self.script):
            return False
        outcome = self.script[self.position]
        self.position += 1
        return outcome != "grab-fail"

    def retrieve(self) -> tuple[bool, object]:
        outcome = self.script[self.position - 1]
        if outcome == "retrieve-fail":
            return False, None
        return True, object()

    def release(self) -> None:
        self.released = True


class _FakeCv2:
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 0
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self) -> None:
        self.scripts: list[list[str]] = []

    def VideoCapture(self, source: str) -> _FakeCapture:
        script = self.scripts.pop(0) if self.scripts else []
        return _FakeCapture(script)


_FAKE_CV2 = _FakeCv2()


class StreamDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeCapture.instances.clear()
        _FAKE_CV2.scripts = []
        self._patch = mock.patch.dict(sys.modules, {"cv2": _FAKE_CV2})
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _drain(self, decoder: VideoDecoder, output: queue.Queue) -> list[FramePacket]:
        decoder.start()
        decoder.join()
        packets = []
        while True:
            try:
                item = output.get_nowait()
            except queue.Empty:
                break
            if item is not _END:
                packets.append(item)
        return packets

    def test_a_stream_is_never_paced_even_when_realtime_is_set(self) -> None:
        # Pacing a live feed against its own timestamps adds delay that
        # accumulates for as long as the stream is up.
        decoder = VideoDecoder(
            "rtsp://cam/stream", queue.Queue(maxsize=1), 10.0, True, threading.Event()
        )
        self.assertTrue(decoder.stream)
        self.assertFalse(decoder.realtime)

    def test_a_file_still_paces(self) -> None:
        decoder = VideoDecoder(
            "clip.mp4", queue.Queue(maxsize=1), 10.0, True, threading.Event()
        )
        self.assertFalse(decoder.stream)
        self.assertTrue(decoder.realtime)

    def test_a_stream_reconnects_instead_of_ending(self) -> None:
        # First connection delivers two frames then drops; the second delivers
        # one more and ends. A file would have stopped at the drop.
        _FAKE_CV2.scripts = [["ok", "ok", "grab-fail"], ["ok"] * 500]
        output: queue.Queue = queue.Queue(maxsize=4)
        stop = threading.Event()
        decoder = VideoDecoder("rtsp://cam/stream", output, 1000.0, False, stop)
        # The second connection keeps delivering until stopped, so the backoff
        # is not run to exhaustion.
        watchdog = threading.Timer(1.0, stop.set)
        watchdog.start()
        self.addCleanup(watchdog.cancel)
        self._drain(decoder, output)
        self.assertGreaterEqual(decoder.reconnects, 1)
        self.assertGreaterEqual(len(_FakeCapture.instances), 2)
        self.assertTrue(_FakeCapture.instances[0].released)

    def test_a_stream_tolerates_a_corrupt_frame(self) -> None:
        # One bad packet on a network source is routine and must not be fatal.
        _FAKE_CV2.scripts = [["ok", "retrieve-fail", "ok"]]
        output: queue.Queue = queue.Queue(maxsize=4)
        stop = threading.Event()
        decoder = VideoDecoder("rtsp://cam/stream", output, 1000.0, False, stop)
        watchdog = threading.Timer(3.0, stop.set)
        watchdog.start()
        self.addCleanup(watchdog.cancel)
        packets = self._drain(decoder, output)
        self.assertEqual(decoder.decode_failures, 1)
        self.assertGreaterEqual(len(packets), 1)
        self.assertIsNone(decoder.error)

    def test_a_file_still_treats_a_corrupt_frame_as_an_error(self) -> None:
        # A file that cannot produce a frame it just grabbed is broken, not flaky.
        _FAKE_CV2.scripts = [["ok", "retrieve-fail"]]
        output: queue.Queue = queue.Queue(maxsize=4)
        decoder = VideoDecoder("clip.mp4", output, 1000.0, False, threading.Event())
        self._drain(decoder, output)
        self.assertIsInstance(decoder.error, VideoError)

    def test_a_stream_asks_the_backend_for_the_smallest_buffer(self) -> None:
        # Whatever the backend buffers is latency this pipeline cannot undo.
        _FAKE_CV2.scripts = [["ok"]]
        output: queue.Queue = queue.Queue(maxsize=2)
        stop = threading.Event()
        decoder = VideoDecoder("rtsp://cam/stream", output, 1000.0, False, stop)
        watchdog = threading.Timer(3.0, stop.set)
        watchdog.start()
        self.addCleanup(watchdog.cancel)
        self._drain(decoder, output)
        self.assertEqual(_FakeCapture.instances[0].buffersize, 1)

    def test_stream_timestamps_come_from_the_wall_clock(self) -> None:
        """A live feed's own timestamps restart at zero when it reconnects.

        Temporal voting, max_delta_per_second and the latency figures all need a
        clock that only moves forward, so elapsed time is used instead.
        """
        _FAKE_CV2.scripts = [["ok", "ok"]]
        output: queue.Queue = queue.Queue(maxsize=4)
        stop = threading.Event()
        decoder = VideoDecoder("rtsp://cam/stream", output, 1000.0, False, stop)
        watchdog = threading.Timer(3.0, stop.set)
        watchdog.start()
        self.addCleanup(watchdog.cancel)
        started = time.perf_counter()
        packets = self._drain(decoder, output)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.assertTrue(packets)
        for packet in packets:
            # The fake reports POS_MSEC as position/fps*1000 -- 0 and 33.3 ms --
            # so a wall-clock reading is the only way these stay under the real
            # elapsed time while not being those values.
            self.assertGreaterEqual(packet.source_time_ms, 0.0)
            self.assertLessEqual(packet.source_time_ms, elapsed_ms + 1.0)


if __name__ == "__main__":
    unittest.main()
