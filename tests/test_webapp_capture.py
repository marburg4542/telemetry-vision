"""The capture button, from the server's side.

This is the only endpoint in the application that writes something permanent, so
what is worth pinning is that it writes the right thing: the whole frame, the
telemetry that belongs to it, and a manifest that ties the two together well
enough for `extract` to read the session back.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

HAS_WEB_DEPS = bool(
    importlib.util.find_spec("aiohttp")
    and importlib.util.find_spec("cv2")
    and importlib.util.find_spec("numpy")
)

STREAM_URL = "rtsp://camera.invalid/live"


class _FakeCapture:
    def __init__(self) -> None:
        import numpy as np

        self._frame = np.full((180, 320, 3), 70, dtype=np.uint8)

    def isOpened(self) -> bool:
        return True

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
        return None


class _StreamCv2:
    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cv2, name)

    def VideoCapture(
        self, source: str, api: int | None = None, params: list[int] | None = None
    ) -> _FakeCapture:
        return _FakeCapture()


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp/NumPy/OpenCV not installed")
class CaptureShotTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
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
        self.root = Path(self.temp.name)

        class FakeRecognizer:
            def recognize(self, images: Any) -> Any:
                return [OCRResult("128", 0.99) for _ in images]

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
            create_app(config, FakeRecognizer(), stream_output_dir=self.root)
        )
        self.client = TestClient(server, headers={GUARD_HEADER: "1"})
        await self.client.start_server()
        self.raw_client = TestClient(server)
        await self.raw_client.start_server()

        frame = np.full((120, 160, 3), 90, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        self.jpeg = encoded.tobytes()

        session = await self.client.post("/api/session")
        self.session_id = (await session.json())["session_id"]

    async def asyncTearDown(self) -> None:
        await self.raw_client.close()
        await self.client.close()

    def _dir(self, name: str) -> Path:
        return self.root / "sessions" / name

    async def _shot(self, name: str = "flight01", **query: Any) -> Any:
        params = {
            "session": name,
            "source_time_ms": "1500",
            "session_id": self.session_id,
            **{key: str(value) for key, value in query.items()},
        }
        from urllib.parse import urlencode

        return await self.client.post(
            f"/api/capture/shot?{urlencode(params)}",
            data=self.jpeg,
            headers={"Content-Type": "image/jpeg"},
        )

    async def test_a_shot_writes_the_frame_and_its_telemetry(self) -> None:
        response = await self._shot()
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(body["frame_index"], 0)
        self.assertTrue((self._dir("flight01") / body["image"]).is_file())
        self.assertTrue(body["records"])
        # Not the value yet: a `fast` field needs 2 of 3 frames to agree, so the
        # first shot of a session carries the reading without a confirmed value.
        self.assertEqual(body["records"][0]["raw_text"], "128")
        self.assertFalse(body["records"][0]["confirmed"])
        # Every record names the picture, which is the entire point.
        for record in body["records"]:
            self.assertEqual(record["image"], body["image"])

        # And by the third, voting has something to agree with.
        await self._shot()
        third = await (await self._shot()).json()
        self.assertEqual(third["records"][0]["value"], 128)
        self.assertTrue(third["records"][0]["confirmed"])

    async def test_shots_append_rather_than_overwrite(self) -> None:
        first = await (await self._shot()).json()
        second = await (await self._shot()).json()
        self.assertEqual(first["frame_index"], 0)
        self.assertEqual(second["frame_index"], 1)
        self.assertNotEqual(first["image"], second["image"])
        self.assertTrue((self._dir("flight01") / first["image"]).is_file())
        self.assertTrue((self._dir("flight01") / second["image"]).is_file())

    async def test_the_session_it_writes_is_one_extract_can_read(self) -> None:
        """The format is shared with the CLI on purpose.

        A capture the offline reader cannot open would be a private file format
        wearing the name of the real one.
        """
        from telemetry_ocr.capture import load_session

        await self._shot()
        await self._shot()
        info, records = load_session(self._dir("flight01"))
        self.assertEqual(len(records), 2)
        self.assertEqual(info.frame_count, 2)
        for record in records:
            self.assertTrue((self._dir("flight01") / record.file).is_file())

    async def test_telemetry_lands_in_the_session_too(self) -> None:
        await self._shot()
        lines = (self._dir("flight01") / "telemetry.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertTrue(lines)
        record = json.loads(lines[0])
        self.assertEqual(record["field_id"], "speed")
        self.assertTrue(record["image"])

    async def test_a_traversing_session_name_is_refused(self) -> None:
        # The name comes from a text field in the browser and becomes a
        # directory. "../.." is an ordinary thing for a text field to contain.
        response = await self._shot("../../escape")
        self.assertEqual(response.status, 400)
        self.assertFalse((self.root.parent / "escape").exists())

    async def test_status_reports_what_has_been_captured(self) -> None:
        await self._shot()
        await self._shot()
        body = await (await self.client.get("/api/capture/status?session=flight01")).json()
        self.assertTrue(body["open"])
        self.assertEqual(body["frames"], 2)
        self.assertGreater(body["bytes"], 0)

    async def test_status_for_an_unknown_session_is_not_an_error(self) -> None:
        body = await (await self.client.get("/api/capture/status?session=nothing")).json()
        self.assertFalse(body["open"])
        self.assertEqual(body["frames"], 0)

    async def test_capturing_needs_the_guard_header(self) -> None:
        response = await self.raw_client.post(
            "/api/capture/shot?session=x&source_time_ms=0", data=self.jpeg
        )
        self.assertEqual(response.status, 403)

    async def test_a_stream_shot_reuses_the_reading_it_already_made(self) -> None:
        """Not a second OCR pass.

        The stream is already reading these frames from its own thread. Running
        the recognizer again here would put two threads inside PaddleOCR, which
        is what the stream guard exists to prevent, and would answer slightly
        differently for the same picture.
        """
        started = await self.client.post("/api/stream/start", json={"url": STREAM_URL})
        self.assertEqual(started.status, 200)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            body = await (await self.client.get("/api/stream/telemetry?wait_ms=1000")).json()
            if body.get("revision", 0) >= 2:
                break
            await asyncio.sleep(0.05)

        # No body: the server holds the frame, so the page has none to send.
        response = await self.client.post(
            "/api/capture/shot?session=streamshot&source_time_ms=0"
        )
        self.assertEqual(response.status, 200)
        shot = await response.json()
        self.assertTrue((self._dir("streamshot") / shot["image"]).is_file())
        self.assertTrue(shot["records"])
        self.assertEqual(shot["records"][0]["field_id"], "speed")
        await self.client.post("/api/stream/stop")


if __name__ == "__main__":
    unittest.main()
