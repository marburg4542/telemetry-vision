"""The capture button, from the server's side.

This is the only endpoint in the application that writes something permanent, so
what is worth pinning is that it writes the right thing: the whole frame, the
telemetry that belongs to it, and a manifest that ties the two together well
enough for `extract` to read the session back.
"""

from __future__ import annotations

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


class _FakeDetector:
    """Answers only for the frame it was actually given.

    Written this way on purpose: returning a canned result regardless of input
    would pass whether or not the capture submits its own frame, which is the
    behaviour under test.
    """

    def __init__(self) -> None:
        self.state = "ready"
        self.submissions: list[Any] = []
        self._result: dict[str, Any] | None = None
        self._active_source = ""
        # Hooks for the tests that need the capture to still be inside the
        # detector when something else happens.
        self.on_submit: Any = None
        self.withhold = False

    def start(self, path: Any, filename: str) -> None:
        self.state = "ready"

    def restart(self) -> None:
        self.state = "ready"

    def stop(self, timeout: float = 3.0) -> None:
        self.state = "stopped"

    def status(self) -> dict[str, Any]:
        return {
            "status": self.state,
            "loaded": self.state == "ready",
            "filename": "fake.pt",
            "worker_pid": 4242 if self.state == "ready" else None,
            "tracker": "BoT-SORT",
            "device": "cpu",
            "error": None,
            "dropped_frames": 0,
            "classes": ["car"],
            "architecture": "fake-yolo",
            "epochs": 1,
            "trained_image_size": 640,
        }

    def submit(self, frame: Any) -> bool:
        if self.state != "ready":
            return False
        self.submissions.append(frame)
        # Like the real worker: one active source at a time, and a result that
        # does not carry it is discarded rather than merely superseded.
        if frame.source_id != self._active_source:
            self._active_source = frame.source_id
            self._result = None
        if self.on_submit is not None:
            self.on_submit(frame)
        if frame.source_id != self._active_source:
            return True
        self._result = {
            "source_id": frame.source_id,
            "generation": frame.generation,
            "sequence": frame.sequence,
            "source_time_ms": frame.source_time_ms,
            "width": 640,
            "height": 360,
            "inference_ms": 11.0,
            "detections": [
                {
                    "x1": 10.0,
                    "y1": 20.0,
                    "x2": 60.0,
                    "y2": 70.0,
                    "confidence": 0.83,
                    "class_id": 0,
                    "label": "car",
                    "track_id": 1,
                }
            ],
        }
        return True

    def latest(self) -> dict[str, Any] | None:
        return None if self.withhold else self._result

    def take_over(self, source_id: str) -> None:
        """What an overlay frame does to a capture that is still waiting."""
        self._active_source = source_id
        self._result = None

    def reset(self, source_id: str, generation: int) -> None:
        self._active_source = source_id
        self._result = None

    def drain(self) -> None:
        return None


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp/NumPy/OpenCV not installed")
class CaptureShotTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import cv2
        import numpy as np
        from aiohttp.test_utils import TestClient, TestServer

        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.webapp import GUARD_HEADER, create_app

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
        self.fake_detection_worker = _FakeDetector()
        server = TestServer(
            create_app(
                config,
                FakeRecognizer(),
                output_dir=self.root,
                detection_worker_factory=lambda: self.fake_detection_worker,
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

    async def test_detection_runs_on_the_captured_frame(self) -> None:
        """Not on whatever result happened to be newest.

        Reusing the latest result put boxes on a capture that visibly did not
        sit on the vehicles: the newest result belongs to whichever frame the
        page last posted, and on a moving camera that is a different scene.
        """
        body = await (await self._shot()).json()
        self.assertEqual(body["detections"]["status"], "ok")
        # Zero, and meant literally: this is the frame that was inferred.
        self.assertEqual(body["detections"]["age_ms"], 0.0)

    async def test_the_capture_submits_its_own_frame(self) -> None:
        """The frame goes to the detector rather than a result coming back from it.

        And under its own source id: a one-off frame pushed under the page's id
        would perturb the tracker the overlay depends on.
        """
        await self._shot()
        self.assertEqual(len(self.fake_detection_worker.submissions), 1)
        submitted = self.fake_detection_worker.submissions[0]
        self.assertTrue(submitted.source_id.startswith("capture-"))
        self.assertTrue(submitted.payload)

    async def test_no_model_is_distinct_from_finding_nothing(self) -> None:
        # A bare empty count cannot tell the two apart, and they are fixed in
        # completely different ways.
        self.fake_detection_worker.state = "unloaded"
        body = await (await self._shot()).json()
        self.assertEqual(body["detections"]["status"], "no_model")
        self.assertEqual(body["detections"]["counts"], {})

    async def test_the_boxes_are_drawn_from_that_result(self) -> None:
        body = await (await self._shot()).json()
        self.assertEqual(body["detections"]["counts"], {"car": 1})
        self.assertTrue(body["annotated"])
        self.assertTrue((self._dir("flight01") / body["annotated"]).is_file())

    async def test_the_shot_list_is_newest_first(self) -> None:
        """The order the page shows them in.

        Newest first because the shot you have just taken is the one you are
        about to look at; scrolling to the end of a long strip to find it is the
        panel getting in the way of the only common case.
        """
        await self._shot()
        await self._shot()
        await self._shot()
        body = await (await self.client.get("/api/capture/shots?session=flight01")).json()
        self.assertEqual([shot["frame_index"] for shot in body["shots"]], [2, 1, 0])

    async def test_each_listed_shot_carries_what_the_panel_shows(self) -> None:
        await self._shot()
        body = await (await self.client.get("/api/capture/shots?session=flight01")).json()
        shot = body["shots"][0]
        self.assertTrue(shot["image"])
        self.assertTrue(shot["annotated"])
        self.assertEqual(shot["detections"]["counts"], {"car": 1})
        self.assertEqual(shot["records"][0]["field_id"], "speed")

    async def test_the_listed_reading_is_the_one_taken_from_the_image(self) -> None:
        """Not the live reading held beside it.

        The picture and the numbers under it have to describe the same moment,
        and only the reading taken from the saved file is guaranteed to.
        """
        from telemetry_ocr.capture_service import VerifiedField

        await self._shot()
        session = self.root / "sessions" / "flight01"
        self.assertTrue(session.is_dir())
        body = await (await self.client.get("/api/capture/shots?session=flight01")).json()
        listed = {record["field_id"]: record for record in body["shots"][0]["records"]}
        self.assertIn("speed", listed)
        # The verification reading is the one with these keys and no others.
        self.assertEqual(
            set(listed["speed"]),
            {"field_id", "value", "raw_text", "confidence", "confirmed"},
        )
        self.assertTrue(hasattr(VerifiedField, "__dataclass_fields__"))

    async def test_an_unknown_session_lists_nothing_rather_than_failing(self) -> None:
        # The page asks before the first capture of a run, which is not an error.
        response = await self.client.get("/api/capture/shots?session=nothing")
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["shots"], [])

    async def test_the_image_endpoint_serves_the_frame_and_the_marked_up_copy(self) -> None:
        await self._shot()
        plain = await self.client.get("/api/capture/image?session=flight01&frame_index=0")
        boxed = await self.client.get(
            "/api/capture/image?session=flight01&frame_index=0&annotated=1"
        )
        self.assertEqual(plain.status, 200)
        self.assertEqual(boxed.status, 200)
        self.assertEqual(plain.headers["Content-Type"], "image/jpeg")
        # Different files: the boxes are drawn on a copy, never over the capture.
        self.assertNotEqual(await plain.read(), await boxed.read())

    async def test_asking_for_boxes_that_were_never_drawn_gives_the_plain_frame(self) -> None:
        """Rather than a 404.

        The toggle is disabled for a shot with nothing on it, but a stale page or
        a direct call should get the picture, not an error about a copy that was
        never made because there was nothing to draw.
        """
        self.fake_detection_worker.state = "unloaded"
        body = await (await self._shot()).json()
        self.assertFalse(body["annotated"])
        response = await self.client.get(
            "/api/capture/image?session=flight01&frame_index=0&annotated=1"
        )
        self.assertEqual(response.status, 200)
        plain = await self.client.get("/api/capture/image?session=flight01&frame_index=0")
        self.assertEqual(await response.read(), await plain.read())

    async def test_a_frame_index_cannot_name_a_file_elsewhere(self) -> None:
        """The index selects a manifest entry; it never becomes part of a path."""
        await self._shot()
        for index in ("../../../../windows/win.ini", "-1", "99"):
            response = await self.client.get(
                f"/api/capture/image?session=flight01&frame_index={index}"
            )
            self.assertIn(response.status, (400, 404), index)

    async def test_the_frame_is_readable_without_the_guard_header(self) -> None:
        """Deliberate, and worth writing down so it is not "fixed" later.

        The panel shows the frame with an ordinary <img>, and an <img> cannot
        send a custom header. Requiring one here would leave the preview blank.
        What stands in front of these reads is the Host allowlist and the Origin
        check; the header exists for the mutating calls, where it forces a
        preflight in front of the model upload's pickle-based load.
        """
        await self._shot()
        for path in (
            "/api/capture/shots?session=flight01",
            "/api/capture/image?session=flight01&frame_index=0",
        ):
            self.assertEqual((await self.raw_client.get(path)).status, 200, path)

    async def test_another_site_cannot_fetch_a_capture(self) -> None:
        """A cross-origin read carries an Origin, and that is refused.

        An <img> on someone else's page sends no Origin and so still loads the
        picture -- but it cannot read the pixels back, which is the difference
        between a frame being displayable and a frame being exfiltrated.
        """
        await self._shot()
        response = await self.raw_client.get(
            "/api/capture/image?session=flight01&frame_index=0",
            headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(response.status, 403)


    async def test_a_listed_shot_says_whether_it_went_through_the_model(self) -> None:
        """The page will not call a frame ready until this says so.

        The whole point of the stage is that "the model found nothing" and "the
        model never ran" are the same empty count, and only one of them means
        the picture has been checked.
        """
        await self._shot()
        body = await (await self.client.get("/api/capture/shots?session=flight01")).json()
        self.assertEqual(body["shots"][0]["stage"], "ready")

        self.fake_detection_worker.state = "unloaded"
        await self._shot()
        body = await (await self.client.get("/api/capture/shots?session=flight01")).json()
        stages = {shot["frame_index"]: shot["stage"] for shot in body["shots"]}
        self.assertEqual(stages[0], "ready")
        self.assertEqual(stages[1], "no_model")

    async def test_finding_nothing_still_counts_as_having_run(self) -> None:
        # A clear frame is an answer. Filing it with the failures would have the
        # page warn about every capture of an empty road.
        self.fake_detection_worker.state = "ready"
        self.fake_detection_worker._result = None
        await self._shot()
        body = await (await self.client.get("/api/capture/shots?session=flight01")).json()
        self.assertEqual(body["shots"][0]["detections"]["counts"], {"car": 1})
        self.assertEqual(body["shots"][0]["stage"], "ready")

    async def test_listing_reads_each_file_once(self) -> None:
        """Rather than once per shot.

        The per-index form re-parsed the manifest and the whole telemetry log for
        every frame, so listing was O(n^2): 0.4 s at 50 shots, 5.8 s at 200,
        33 s at 500 -- and the page lists after every capture, which made holding
        the button slower with each frame it took.
        """
        for _ in range(6):
            await self._shot()
        session = self._dir("flight01")
        reads: list[str] = []
        real = Path.read_text

        def counting(self: Path, *args: Any, **kwargs: Any) -> str:
            if self.parent == session:
                reads.append(self.name)
            return real(self, *args, **kwargs)

        with mock.patch.object(Path, "read_text", counting):
            body = await (await self.client.get("/api/capture/shots?session=flight01")).json()
        self.assertEqual(len(body["shots"]), 6)
        self.assertEqual(reads.count("frames.jsonl"), 1)
        self.assertEqual(reads.count("telemetry.jsonl"), 1)

    async def test_a_thumbnail_is_smaller_and_stays_out_of_the_recording(self) -> None:
        """`frames/` is the recording `extract` reads, and the next frame index
        is worked out by scanning it. A cached thumbnail dropped in there would
        be picked up as a frame."""
        await self._shot()
        full = await self.client.get("/api/capture/image?session=flight01&frame_index=0")
        thumb = await self.client.get(
            "/api/capture/image?session=flight01&frame_index=0&thumb=1"
        )
        self.assertEqual(thumb.status, 200)
        # The test frame is 160 px wide, under the thumbnail width, so the full
        # picture is served rather than an upscale of it.
        self.assertEqual(len(await thumb.read()), len(await full.read()))
        self.assertEqual(
            sorted(item.name for item in (self._dir("flight01") / "frames").glob("*")),
            ["00000000.boxes.jpg", "00000000.jpg"],
        )

    async def test_a_big_frame_gets_a_real_thumbnail(self) -> None:
        import cv2
        import numpy as np

        await self._shot()
        # Replace the saved frame with one worth shrinking.
        big = np.random.default_rng(0).integers(
            0, 255, (1080, 1920, 3), dtype=np.uint8
        )
        cv2.imwrite(str(self._dir("flight01") / "frames" / "00000000.jpg"), big)
        response = await self.client.get(
            "/api/capture/image?session=flight01&frame_index=0&thumb=1"
        )
        self.assertEqual(response.status, 200)
        decoded = cv2.imdecode(
            np.frombuffer(await response.read(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(decoded.shape[1], 320)
        cached = self._dir("flight01") / "thumbs" / "00000000.jpg"
        self.assertTrue(cached.is_file())
        self.assertLess(cached.stat().st_size, 60_000)

    async def test_reopening_a_session_keeps_counting_past_the_extra_files(self) -> None:
        """The marked-up copies and the thumbnails share a name stem with the
        frames. Counting files instead of reading the highest index restarts the
        numbering and overwrites frames the manifest still points at."""
        from telemetry_ocr.capture_service import CaptureSession

        for _ in range(3):
            await self._shot()
        await self.client.get("/api/capture/image?session=flight01&frame_index=0&thumb=1")
        reopened = CaptureSession(self.root / "sessions", "flight01", "reopened", [])
        self.assertEqual(reopened.frame_count, 3)


    async def test_the_live_overlay_cannot_take_the_detector_mid_capture(self) -> None:
        """The bug this guards: every capture came back with no boxes.

        The worker keeps one active source id and discards any result that does
        not carry it. A capture submits under its own `capture-` id, and the
        page's overlay -- which posts a frame at least every 400 ms while a
        video plays -- takes the active id straight back. Inference takes longer
        than that, so the capture's answer was thrown away before it arrived and
        every shot was filed as "no_result" with no annotated copy written.
        """
        overlay_frames = []

        def overlay_cuts_in(frame: Any) -> None:
            # Exactly once, and only for the capture: the first overlay frame
            # that was already in flight when the capture claimed the detector.
            if frame.source_id.startswith("capture-") and not overlay_frames:
                overlay_frames.append(frame)
                self.fake_detection_worker.take_over("page-source")

        self.fake_detection_worker.on_submit = overlay_cuts_in
        body = await (await self._shot()).json()
        self.assertEqual(len(overlay_frames), 1)
        # Resubmitted rather than given up on.
        self.assertEqual(body["detections"]["status"], "ok")
        self.assertEqual(body["detections"]["counts"], {"car": 1})
        self.assertTrue(body["annotated"])

    async def test_the_overlay_stands_down_while_a_capture_holds_it(self) -> None:
        """The other half: the overlay is told to skip, not queued behind.

        A frame held for half a second and then inferred would draw boxes from
        the wrong moment, which is the fault this whole path exists to avoid.
        """
        import asyncio as _asyncio

        released = _asyncio.Event()
        overlay_result: dict[str, Any] = {}

        async def overlay_posts() -> None:
            response = await self.client.post(
                "/api/detection/frame?source_id=page&generation=0&sequence=1"
                "&source_time_ms=0&image_size=640",
                data=self.jpeg,
                headers={"Content-Type": "image/jpeg"},
            )
            overlay_result.update(await response.json())
            # Only now may the capture see its answer, so the overlay is
            # guaranteed to have been refused while the claim was held rather
            # than served after it was released.
            self.fake_detection_worker.withhold = False
            released.set()

        # submit() runs in a worker thread, so the loop is captured here and the
        # overlay's post is handed back to it rather than started from there.
        loop = _asyncio.get_running_loop()

        def hold(frame: Any) -> None:
            if frame.source_id.startswith("capture-") and not overlay_result:
                # The overlay tries to post while the capture is inside the
                # detector. It must be refused rather than served.
                self.fake_detection_worker.withhold = True
                _asyncio.run_coroutine_threadsafe(overlay_posts(), loop)

        self.fake_detection_worker.on_submit = hold
        shot = await (await self._shot()).json()
        self.assertEqual(shot["detections"]["status"], "ok")
        await _asyncio.wait_for(released.wait(), timeout=5.0)
        self.assertTrue(overlay_result.get("deferred"))
        self.assertFalse(overlay_result.get("accepted"))
        # And nothing from the page reached the detector during the capture.
        sources = [item.source_id for item in self.fake_detection_worker.submissions]
        self.assertNotIn("page", sources)


    async def _discard(self, *indexes: int, session: str = "flight01") -> Any:
        from urllib.parse import urlencode

        params = urlencode(
            [("session", session)] + [("frame_index", str(i)) for i in indexes]
        )
        return await self.client.delete(f"/api/capture/shot?{params}")

    async def test_discarding_removes_the_pictures_and_the_records(self) -> None:
        """Both, together.

        A telemetry line naming an image that is no longer on disk is worse than
        no line: `extract` reads the session back and reports a frame nobody can
        look at.
        """
        first = await (await self._shot()).json()
        await self._shot()
        session = self._dir("flight01")
        self.assertTrue((session / first["image"]).is_file())

        response = await self._discard(0)
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(body["removed"], [0])
        # What is left, not the next number to hand out.
        self.assertEqual(body["remaining"], 1)

        self.assertFalse((session / first["image"]).is_file())
        self.assertFalse((session / first["annotated"]).is_file())
        for name in ("telemetry.jsonl", "telemetry-verified.jsonl"):
            path = session / name
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        self.assertNotEqual(json.loads(line)["image"], first["image"])

    async def test_what_was_not_discarded_is_untouched(self) -> None:
        await self._shot()
        second = await (await self._shot()).json()
        await self._shot()
        await self._discard(0, 2)
        body = await (await self.client.get("/api/capture/shots?session=flight01")).json()
        self.assertEqual([shot["frame_index"] for shot in body["shots"]], [1])
        self.assertTrue(body["shots"][0]["records"])
        self.assertTrue((self._dir("flight01") / second["image"]).is_file())

    async def test_the_session_still_reads_back_after_a_deletion(self) -> None:
        # The format is shared with the CLI, and a half-deleted session that
        # `extract` chokes on would be worse than one that kept the frame.
        from telemetry_ocr.capture import load_session

        for _ in range(3):
            await self._shot()
        await self._discard(1)
        info, records = load_session(self._dir("flight01"))
        self.assertEqual(info.frame_count, 2)
        for record in records:
            self.assertTrue((self._dir("flight01") / record.file).is_file())

    async def test_a_deleted_number_is_not_handed_out_again(self) -> None:
        """A gap stays a gap.

        Reusing the number would give a new capture an old capture's filename,
        and the manifest would then describe two different pictures under one
        index.
        """
        from telemetry_ocr.capture_service import CaptureSession

        for _ in range(3):
            await self._shot()
        await self._discard(2)
        reopened = CaptureSession(self.root / "sessions", "flight01", "reopened", [])
        self.assertEqual(reopened.frame_count, 2)
        fourth = await (await self._shot()).json()
        self.assertEqual(fourth["frame_index"], 3)

    async def test_discarding_a_frame_that_is_not_there_is_not_an_error(self) -> None:
        # The page can ask twice -- two clicks, or a stale list -- and the
        # second ask should report that nothing was removed, not fail.
        await self._shot()
        body = await (await self._discard(99)).json()
        self.assertEqual(body["removed"], [])
        self.assertEqual(body["remaining"], 1)

    async def test_discarding_needs_the_guard_header(self) -> None:
        # It deletes. Of everything in this file it is the one call that must
        # not be reachable from another page.
        await self._shot()
        response = await self.raw_client.delete(
            "/api/capture/shot?session=flight01&frame_index=0"
        )
        self.assertEqual(response.status, 403)
        self.assertTrue((self._dir("flight01") / "frames" / "00000000.jpg").is_file())

    async def test_the_thumbnail_goes_too(self) -> None:
        await self._shot()
        await self.client.get("/api/capture/image?session=flight01&frame_index=0&thumb=1")
        await self._discard(0)
        thumbs = self._dir("flight01") / "thumbs"
        self.assertEqual(list(thumbs.glob("*")) if thumbs.is_dir() else [], [])


if __name__ == "__main__":
    unittest.main()
