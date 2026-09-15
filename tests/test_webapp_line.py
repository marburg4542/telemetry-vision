"""The send-to-LINE endpoint.

This is the only endpoint that sends anything outside the machine, and every
call spends a message from the account's monthly quota. What is worth pinning is
therefore less about formatting than about restraint: that it refuses when it is
not configured, that it cannot be made to fire in a loop, that the guard applies,
and that the credential never comes back out.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

HAS_WEB_DEPS = bool(
    importlib.util.find_spec("aiohttp")
    and importlib.util.find_spec("cv2")
    and importlib.util.find_spec("numpy")
)

TOKEN = "secret-line-token-must-not-leak"


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp/NumPy/OpenCV not installed")
class LineEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import cv2
        import numpy as np
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.webapp import GUARD_HEADER, create_app

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        # A local stand-in for LINE, so nothing leaves the machine.
        self.pushes: list[dict[str, Any]] = []
        self.reply_status = 200
        self.reply_body: dict[str, Any] = {}

        async def line_handler(request: Any) -> Any:
            self.pushes.append(
                {"auth": request.headers.get("Authorization", ""), "body": await request.json()}
            )
            return web.json_response(self.reply_body, status=self.reply_status)

        line_app = web.Application()
        line_app.router.add_post("/push", line_handler)
        self.line_server = TestServer(line_app)
        await self.line_server.start_server()
        url_patch = mock.patch(
            "telemetry_ocr.line_client.PUSH_URL", str(self.line_server.make_url("/push"))
        )
        url_patch.start()
        self.addCleanup(url_patch.stop)

        env_patch = mock.patch.dict(
            "os.environ",
            {"TELEMETRY_OCR_LINE_TOKEN": TOKEN, "TELEMETRY_OCR_LINE_TO": "U1234567890"},
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)

        # Point the loader's file fallback somewhere that does not exist. Without
        # this the suite reads whatever configs/line.json the developer happens to
        # have, so the "not configured" cases passed on a machine with no LINE
        # set up and failed on one with it -- and worse, a real token would have
        # been loaded into a test run.
        path_patch = mock.patch(
            "telemetry_ocr.line_client.DEFAULT_CONFIG_PATH", self.root / "absent-line.json"
        )
        path_patch.start()
        self.addCleanup(path_patch.stop)

        class FakeRecognizer:
            def recognize(self, images: Any) -> Any:
                return [OCRResult("128", 0.99) for _ in images]

        config = AppConfig(
            runtime=RuntimeConfig(
                enable_fallback=False, confidence_threshold=0.5, warmup_iterations=0
            ),
            fields=[
                FieldConfig(
                    field_id="lrf_m",
                    box=(0.1, 0.1, 0.4, 0.4),
                    allowed_chars="0123456789",
                    regex=r"^\d+$",
                    data_type="integer",
                    profile="fast",
                )
            ],
        )
        from _video_fixture import write_test_video
        from telemetry_ocr.video_library import VideoLibrary

        write_test_video(self.root / "videos" / "flight.avi")
        server = TestServer(
            # Port 0: the share server binds an ephemeral port so two tests in
            # this class cannot collide on the real one, and so the suite never
            # fights whatever is already listening on this machine.
            create_app(
                config,
                FakeRecognizer(),
                output_dir=self.root,
                share_port=0,
                video_library=VideoLibrary(self.root / "videos"),
            )
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
        await self.line_server.close()

    async def _shot(self) -> Any:
        from urllib.parse import urlencode

        params = urlencode(
            {
                "session": "flight01",
                "source_time_ms": "1500",
                "session_id": self.session_id,
                "video": "flight.avi",
            }
        )
        return await self.client.post(f"/api/capture/shot?{params}")

    async def _send(self, **query: Any) -> Any:
        from urllib.parse import urlencode

        params = urlencode({"session": "flight01", **query})
        return await self.client.post(f"/api/line/send?{params}")

    async def _send_frames(self, *indexes: int) -> Any:
        from urllib.parse import urlencode

        params = urlencode(
            [("session", "flight01")] + [("frame_index", str(i)) for i in indexes]
        )
        return await self.client.post(f"/api/line/send?{params}")

    async def test_status_says_configured_without_revealing_the_token(self) -> None:
        response = await self.client.get("/api/line/status")
        body = await response.json()
        self.assertTrue(body["configured"])
        self.assertNotIn(TOKEN, json.dumps(body))
        # Enough of the destination to recognise, not enough to reuse.
        self.assertEqual(body["to"], "U123...90")

    async def test_status_says_what_to_set_when_it_is_not_configured(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            body = await (await self.client.get("/api/line/status")).json()
        self.assertFalse(body["configured"])
        self.assertIn("TELEMETRY_OCR_LINE_TOKEN", body["how"])

    async def test_sending_pushes_the_latest_shot(self) -> None:
        await self._shot()
        response = await self._send()
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["frame_index"], 0)
        self.assertEqual(len(self.pushes), 1)
        self.assertEqual(self.pushes[0]["auth"], f"Bearer {TOKEN}")
        sent = self.pushes[0]["body"]["messages"][0]["text"]
        self.assertIn("flight01", sent)
        self.assertIn("frames/00000000.jpg", sent)

    async def test_the_token_never_comes_back_to_the_browser(self) -> None:
        await self._shot()
        body = await (await self._send()).text()
        self.assertNotIn(TOKEN, body)

    async def test_it_refuses_when_nothing_has_been_captured(self) -> None:
        # Nothing captured means no session, which is 404 -- a different thing
        # from a session whose frame you asked for is not in it. Either way no
        # push goes out, because spending quota to say nothing is the failure
        # worth preventing.
        response = await self._send()
        self.assertEqual(response.status, 404)
        self.assertEqual(self.pushes, [])

    async def test_asking_for_a_frame_the_session_does_not_have(self) -> None:
        await self._shot()
        response = await self._send(frame_index=99)
        self.assertEqual(response.status, 409)
        self.assertEqual(self.pushes, [])

    async def test_an_unconfigured_server_refuses_before_reaching_line(self) -> None:
        await self._shot()
        with mock.patch.dict("os.environ", {}, clear=True):
            response = await self._send()
        self.assertEqual(response.status, 400)
        self.assertIn("TELEMETRY_OCR_LINE_TOKEN", (await response.json())["error"])
        self.assertEqual(self.pushes, [])

    async def test_a_second_send_too_soon_is_refused(self) -> None:
        """Every push costs a message from a monthly allowance.

        Disabling the button is not enough, because the endpoint can be called
        directly and a wedged client would drain the quota.
        """
        await self._shot()
        first = await self._send()
        self.assertEqual(first.status, 200)
        second = await self._send()
        self.assertEqual(second.status, 429)
        self.assertIn("retry_after_s", await second.json())
        self.assertEqual(len(self.pushes), 1)

    async def test_a_failed_send_does_not_lock_out_the_retry(self) -> None:
        # The operator fixes the cause and tries again immediately; making them
        # wait out the limiter for a send that never happened would be perverse.
        await self._shot()
        self.reply_status = 401
        self.reply_body = {"message": "Invalid channel access token"}
        failed = await self._send()
        self.assertEqual(failed.status, 502)
        self.assertIn("Invalid channel access token", (await failed.json())["line_message"])

        self.reply_status = 200
        self.reply_body = {}
        retried = await self._send()
        self.assertEqual(retried.status, 200)

    async def test_an_older_frame_can_be_sent_by_index(self) -> None:
        await self._shot()
        await self._shot()
        response = await self._send(frame_index=0)
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["frame_index"], 0)
        self.assertIn("frames/00000000.jpg", self.pushes[0]["body"]["messages"][0]["text"])

    async def test_an_unknown_session_is_not_found(self) -> None:
        response = await self.client.post("/api/line/send?session=never-existed")
        self.assertEqual(response.status, 404)
        self.assertEqual(self.pushes, [])

    async def test_sending_needs_the_guard_header(self) -> None:
        # Without it, any site the user has open could spend their quota.
        await self._shot()
        response = await self.raw_client.post("/api/line/send?session=flight01")
        self.assertEqual(response.status, 403)
        self.assertEqual(self.pushes, [])

    async def test_the_shot_records_what_the_detector_saw(self) -> None:
        """Counted when the frame was taken, not when it is sent.

        The video in this test was never rendered, so the honest answer is
        `no_render` -- which has to be distinguishable from having looked and
        found nothing.
        """
        body = await (await self._shot()).json()
        self.assertEqual(body["detections"]["status"], "no_render")
        self.assertEqual(body["detections"]["counts"], {})
        await self._send()
        self.assertIn("ยังไม่ได้ประมวลผลวิดีโอด้วย AI", self.pushes[0]["body"]["messages"][0]["text"])

    async def test_the_counts_reach_the_manifest(self) -> None:
        from telemetry_ocr.capture import load_session

        await self._shot()
        _, records = await asyncio.to_thread(load_session, self.root / "sessions" / "flight01")
        self.assertEqual(len(records), 1)
        assert records[0].detections is not None
        self.assertEqual(records[0].detections["status"], "no_render")



class MultiFrameSendTests(LineEndpointTests):
    """Several chosen frames, sent together.

    Together rather than one after another because LINE counts a push by the
    request: four separate sends spend four messages from the monthly allowance
    to say what one request can carry.
    """

    async def test_every_chosen_frame_goes_in_one_request(self) -> None:
        for _ in range(3):
            await self._shot()
        response = await self._send_frames(0, 2)
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(body["frame_indexes"], [0, 2])
        self.assertEqual(len(self.pushes), 1)
        sent = self.pushes[0]["body"]["messages"][0]["text"]
        self.assertIn("[Capture x2]", sent)
        self.assertIn("frames/00000000.jpg", sent)
        self.assertIn("frames/00000002.jpg", sent)
        # The one in the middle was not chosen and must not have come along.
        self.assertNotIn("frames/00000001.jpg", sent)

    async def test_the_same_frame_twice_is_sent_once(self) -> None:
        await self._shot()
        body = await (await self._send_frames(0, 0)).json()
        self.assertEqual(body["frame_indexes"], [0])

    async def test_frames_arrive_in_order_however_they_were_ticked(self) -> None:
        # The pictures follow the text in the same order, so a reader can match
        # the blocks to the images without opening them.
        for _ in range(3):
            await self._shot()
        body = await (await self._send_frames(2, 0, 1)).json()
        self.assertEqual(body["frame_indexes"], [0, 1, 2])

    async def test_more_than_four_is_refused_with_the_reason(self) -> None:
        """Four is LINE's limit showing through, not a preference.

        Five messages per request, and the first is the telemetry -- so a fifth
        picture is a second request and a second message off the allowance. It
        is refused rather than trimmed because the operator picked those frames.
        """
        for _ in range(5):
            await self._shot()
        response = await self._send_frames(0, 1, 2, 3, 4)
        self.assertEqual(response.status, 400)
        self.assertIn("4", (await response.text()))
        self.assertEqual(self.pushes, [])

    async def test_a_missing_frame_does_not_lose_the_rest(self) -> None:
        await self._shot()
        body = await (await self._send_frames(0, 99)).json()
        self.assertEqual(body["frame_indexes"], [0])

    async def test_choosing_nothing_still_sends_the_last_one(self) -> None:
        # The old behaviour, kept: a caller that names no frame means "the one
        # I just took".
        await self._shot()
        await self._shot()
        body = await (await self._send()).json()
        self.assertEqual(body["frame_indexes"], [1])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp/NumPy/OpenCV not installed")
class ImageSendTests(LineEndpointTests):
    """Sending the picture, not only the numbers.

    Inherits the fake LINE server and the capture helpers; what is added here is
    a public base URL, which is the setting that turns images on at all.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from aiohttp.test_utils import TestClient

        env = mock.patch.dict(
            "os.environ", {"TELEMETRY_OCR_PUBLIC_URL": "https://drone.example.com"}
        )
        env.start()
        self.addCleanup(env.stop)
        # A second client standing in for LINE coming through the tunnel: a
        # public Host and no guard header.
        self.outside = TestClient(
            self.client._server, headers={"Host": "drone.example.com"}
        )
        await self.outside.start_server()
        self.addAsyncCleanup(self.outside.close)

        # drone.example.com does not resolve, and the send path now refuses to
        # hand LINE a URL it could not fetch itself. These tests are about what
        # the push carries; UnreachableUrlTests covers the check.
        async def _always_reachable(url: str, **kwargs: Any) -> bool:
            self.checked.append(url)
            return True

        self.checked: list[str] = []
        reach = mock.patch("telemetry_ocr.webapp._reachable", _always_reachable)
        reach.start()
        self.addCleanup(reach.stop)

    async def test_status_says_images_are_on(self) -> None:
        body = await (await self.client.get("/api/line/status")).json()
        self.assertTrue(body["images"])

    async def test_the_push_carries_an_image_message(self) -> None:
        await self._shot()
        response = await self._send()
        self.assertEqual(response.status, 200)
        messages = self.pushes[0]["body"]["messages"]
        kinds = [item["type"] for item in messages]
        self.assertEqual(kinds[0], "text")
        self.assertIn("image", kinds)
        image = next(item for item in messages if item["type"] == "image")
        self.assertTrue(image["originalContentUrl"].startswith("https://drone.example.com/"))
        self.assertEqual(image["originalContentUrl"], image["previewImageUrl"])

    async def test_one_request_carries_everything(self) -> None:
        # LINE counts a push per request, so batching keeps a capture at one
        # message from the monthly allowance rather than several.
        await self._shot()
        await self._send()
        self.assertEqual(len(self.pushes), 1)

    async def test_the_url_is_on_the_public_base_and_carries_a_real_token(self) -> None:
        """What the push carries. Serving it is the share server's job.

        Those two live in different processes on purpose, so they are checked
        separately -- see ShareServerTests for the fetch.
        """
        await self._shot()
        await self._send()
        image = next(
            item for item in self.pushes[0]["body"]["messages"] if item["type"] == "image"
        )
        url = image["originalContentUrl"]
        self.assertTrue(url.startswith("https://drone.example.com/public/img/"))
        token = url.rsplit("/", 1)[1]
        # The token is the entire authorisation on a public URL.
        self.assertGreaterEqual(len(token), 40)

    async def test_the_application_itself_still_serves_nothing(self) -> None:
        # Even the path in the URL it just handed to LINE is refused here.
        await self._shot()
        await self._send()
        image = next(
            item for item in self.pushes[0]["body"]["messages"] if item["type"] == "image"
        )
        path = image["originalContentUrl"].split("drone.example.com", 1)[1]
        self.assertEqual((await self.outside.get(path)).status, 403)

    async def test_the_url_is_checked_before_line_is_given_it(self) -> None:
        # The whole point of the guard: nothing is sent that was not fetched.
        await self._shot()
        await self._send()
        self.assertEqual(len(self.checked), 1)
        self.assertTrue(self.checked[0].startswith("https://drone.example.com/"))

    async def test_text_still_goes_when_no_public_url_is_set(self) -> None:
        # Images are a separate decision; turning them off must not stop the
        # telemetry, which is the part someone is waiting for.
        with mock.patch.dict("os.environ", {"TELEMETRY_OCR_PUBLIC_URL": ""}):
            await self._shot()
            response = await self._send()
        self.assertEqual(response.status, 200)
        kinds = [item["type"] for item in self.pushes[0]["body"]["messages"]]
        self.assertEqual(kinds, ["text"])

    async def test_only_the_annotated_picture_is_sent(self) -> None:
        """One image, not two nearly identical ones.

        Sending the original alongside the annotated copy put two pictures in
        the chat that differ only by the boxes, and the reader could not tell
        which was which without opening both. The original stays on disk.
        """
        await self._shot()
        await self._send()
        images = [
            item for item in self.pushes[0]["body"]["messages"] if item["type"] == "image"
        ]
        self.assertEqual(len(images), 1)

    async def test_the_message_names_one_path(self) -> None:
        await self._shot()
        await self._send()
        text = self.pushes[0]["body"]["messages"][0]["text"]
        self.assertEqual(text.count("รูป"), 1)


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp/NumPy/OpenCV not installed")
class UnreachableUrlTests(LineEndpointTests):
    """What happens when the public address does not answer.

    A quick tunnel can be torn down while cloudflared keeps running, so the
    server can be advertising a hostname that no longer resolves. LINE fetches
    an image URL once; if it fails, the chat shows a broken picture forever and
    nothing retries. So an unfetchable URL must not be sent at all.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        env = mock.patch.dict(
            "os.environ", {"TELEMETRY_OCR_PUBLIC_URL": "https://gone.example.com"}
        )
        env.start()
        self.addCleanup(env.stop)

        async def _never_reachable(url: str, **kwargs: Any) -> bool:
            return False

        reach = mock.patch("telemetry_ocr.webapp._reachable", _never_reachable)
        reach.start()
        self.addCleanup(reach.stop)

    async def test_no_image_is_sent_when_the_url_does_not_answer(self) -> None:
        await self._shot()
        response = await self._send()
        self.assertEqual(response.status, 200)
        kinds = [item["type"] for item in self.pushes[0]["body"]["messages"]]
        self.assertEqual(kinds, ["text"])

    async def test_the_telemetry_still_goes(self) -> None:
        """A picture that cannot be published is not a reason to withhold the numbers.

        Those are the part someone is waiting for.
        """
        await self._shot()
        await self._send()
        sent = self.pushes[0]["body"]["messages"][0]["text"]
        self.assertIn("flight01", sent)

    async def test_the_caller_is_told_why_there_is_no_picture(self) -> None:
        await self._shot()
        body = await (await self._send()).json()
        self.assertEqual(body["images"], 0)
        self.assertIn("URL", body["share_error"])
