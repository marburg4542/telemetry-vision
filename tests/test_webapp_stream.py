"""The server-side stream endpoints.

These exist because a browser cannot open RTSP, so the usual direction of this
application is reversed and the server becomes the one holding the video. What
is worth pinning is the part that reversal breaks: there is now one recognizer
with two possible callers, and a viewer that has to be told why there is no
picture.

The camera is fake; OpenCV's encoder is not.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
import unittest
from typing import Any
from unittest import mock

HAS_WEB_DEPS = bool(
    importlib.util.find_spec("aiohttp")
    and importlib.util.find_spec("cv2")
    and importlib.util.find_spec("numpy")
)

URL = "rtsp://camera.invalid/live"


class _FakeCapture:
    def __init__(self, opens: bool) -> None:
        import numpy as np

        self.opens = opens
        self.released = False
        self._frame = np.full((180, 320, 3), 60, dtype=np.uint8)

    def isOpened(self) -> bool:
        return self.opens

    def get(self, prop: int) -> float:
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def grab(self) -> bool:
        time.sleep(0.005)
        return True

    def retrieve(self) -> tuple[bool, Any]:
        return True, self._frame.copy()

    def release(self) -> None:
        self.released = True


class _StreamCv2:
    """Real OpenCV with imaginary cameras."""

    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2
        self.opens = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cv2, name)

    def VideoCapture(
        self, source: str, api: int | None = None, params: list[int] | None = None
    ) -> _FakeCapture:
        return _FakeCapture(self.opens)


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp/NumPy/OpenCV not installed")
class StreamEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import tempfile

        import cv2
        import numpy as np
        from aiohttp.test_utils import TestClient, TestServer

        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.webapp import GUARD_HEADER, create_app

        self.cv2 = _StreamCv2()
        patch = mock.patch.dict(sys.modules, {"cv2": self.cv2})
        patch.start()
        self.addCleanup(patch.stop)

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        class FakeRecognizer:
            def recognize(self, images: Any) -> Any:
                return [OCRResult("42", 0.99) for _ in images]

        config = AppConfig(
            runtime=RuntimeConfig(
                enable_fallback=False, confidence_threshold=0.5, warmup_iterations=0
            ),
            fields=[
                FieldConfig(
                    field_id="speed",
                    box=(0.1, 0.1, 0.4, 0.4),
                    allowed_chars="0123456789",
                    regex=r"^\d+$",
                    data_type="integer",
                    profile="fast",
                )
            ],
        )
        server = TestServer(
            create_app(config, FakeRecognizer(), stream_output_dir=self.temp.name)
        )
        self.client = TestClient(server, headers={GUARD_HEADER: "1"})
        await self.client.start_server()
        self.raw_client = TestClient(server)
        await self.raw_client.start_server()

        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        self.jpeg = encoded.tobytes()

    async def asyncTearDown(self) -> None:
        await self.raw_client.close()
        await self.client.close()

    async def _start(self, url: str = URL) -> Any:
        return await self.client.post("/api/stream/start", json={"url": url})

    async def _wait_for_revision(self, minimum: int = 1, timeout: float = 10.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = await self.client.get("/api/stream/telemetry?wait_ms=1000")
            body = await response.json()
            if body.get("revision", 0) >= minimum:
                return int(body["revision"])
            await asyncio.sleep(0.05)
        self.fail(f"Stream never reached revision {minimum}")

    async def test_asking_before_starting_says_so(self) -> None:
        status = await self.client.get("/api/stream/status")
        self.assertEqual(status.status, 200)
        self.assertFalse((await status.json())["running"])
        # Distinct from a stream that is running but has no picture yet.
        self.assertEqual((await self.client.get("/api/stream/telemetry")).status, 404)

    async def test_a_file_path_is_refused(self) -> None:
        response = await self._start("C:/videos/clip.mp4")
        self.assertEqual(response.status, 400)
        self.assertIn("stream", (await response.json())["error"].lower())

    async def test_starting_publishes_telemetry_and_a_picture(self) -> None:
        self.assertEqual((await self._start()).status, 200)
        await self._wait_for_revision(2)

        telemetry = await self.client.get("/api/stream/telemetry")
        body = await telemetry.json()
        self.assertTrue(body["running"])
        self.assertEqual(body["fields"]["speed"]["value"], 42)

        frame = await self.client.get("/api/stream/frame.jpg")
        self.assertEqual(frame.status, 200)
        self.assertEqual(frame.content_type, "image/jpeg")
        self.assertEqual((await frame.read())[:2], b"\xff\xd8")
        self.assertGreater(int(frame.headers["X-Stream-Revision"]), 0)

    async def test_the_long_poll_waits_for_something_new(self) -> None:
        # A poll that returned immediately would put the viewer straight back
        # into the request flood this codebase has already had once.
        self.assertEqual((await self._start()).status, 200)
        current = await self._wait_for_revision(2)
        started = time.monotonic()
        response = await self.client.get(
            f"/api/stream/telemetry?after_revision={current + 20}&wait_ms=300"
        )
        elapsed = time.monotonic() - started
        self.assertEqual(response.status, 200)
        self.assertGreaterEqual(elapsed, 0.25)

    async def test_a_running_stream_owns_the_recognizer(self) -> None:
        """A second caller must be refused, not quietly interleaved.

        One process, one PaddleOCR. The stream calls it from its own thread,
        outside the loop's lock, so a tab posting frames at the same time would
        put two threads inside the recognizer at once.
        """
        session = await self.client.post("/api/session")
        session_id = (await session.json())["session_id"]
        self.assertEqual((await self._start()).status, 200)
        await self._wait_for_revision(1)

        response = await self.client.post(
            f"/api/session/{session_id}/frame?source_time_ms=0&frame_index=0",
            data=self.jpeg,
        )
        self.assertEqual(response.status, 409)
        self.assertIn("stream", (await response.json())["error"].lower())

    async def test_stopping_frees_the_recognizer_again(self) -> None:
        session = await self.client.post("/api/session")
        session_id = (await session.json())["session_id"]
        self.assertEqual((await self._start()).status, 200)
        await self._wait_for_revision(1)
        stopped = await self.client.post("/api/stream/stop")
        self.assertEqual(stopped.status, 200)
        self.assertFalse((await stopped.json())["status"]["running"])

        response = await self.client.post(
            f"/api/session/{session_id}/frame?source_time_ms=0&frame_index=0",
            data=self.jpeg,
        )
        self.assertEqual(response.status, 200)

    async def test_two_streams_at_once_are_refused(self) -> None:
        self.assertEqual((await self._start()).status, 200)
        await self._wait_for_revision(1)
        second = await self._start("rtsp://other.invalid/live")
        self.assertEqual(second.status, 400)
        self.assertIn("already running", (await second.json())["error"])

    async def test_a_camera_that_never_opens_reaches_the_viewer(self) -> None:
        # Starting succeeds -- the failure happens on the service's thread -- so
        # the only way the operator learns of it is through status.
        self.cv2.opens = False
        self.assertEqual((await self._start()).status, 200)
        deadline = time.monotonic() + 10.0
        error = None
        while time.monotonic() < deadline and not error:
            error = (await (await self.client.get("/api/stream/status")).json())["error"]
            await asyncio.sleep(0.05)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn(URL, error)

    async def test_mjpeg_sends_parts_a_browser_can_render(self) -> None:
        self.assertEqual((await self._start()).status, 200)
        await self._wait_for_revision(1)
        response = await self.client.get("/api/stream/mjpeg")
        self.assertEqual(response.status, 200)
        self.assertIn("multipart/x-mixed-replace", response.headers["Content-Type"])
        chunk = await asyncio.wait_for(response.content.read(2048), timeout=10.0)
        self.assertIn(b"--frame", chunk)
        self.assertIn(b"image/jpeg", chunk)
        response.close()

    async def test_starting_a_stream_needs_the_guard_header(self) -> None:
        # Same reasoning as every other mutating endpoint: without it any site
        # the user has open could point this server at a camera of its choosing.
        response = await self.raw_client.post("/api/stream/start", json={"url": URL})
        self.assertEqual(response.status, 403)


if __name__ == "__main__":
    unittest.main()
