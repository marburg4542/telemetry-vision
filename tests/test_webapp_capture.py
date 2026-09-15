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


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp/NumPy/OpenCV not installed")
class CaptureShotTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from aiohttp.test_utils import TestClient, TestServer

        from _video_fixture import car, write_fake_render, write_test_video
        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.render_job import RenderManager
        from telemetry_ocr.video_library import VideoLibrary
        from telemetry_ocr.webapp import GUARD_HEADER, create_app

        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
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
        # A 20-frame video at 10 fps whose frame N is filled with brightness
        # N * 10, and a finished render of it with a car on frame 15 -- the
        # moment _shot takes by default -- and nothing on frame 5.
        self.videos = self.root / "videos"
        self.video = write_test_video(self.videos / "flight.avi")
        self.model = self.root / "best.pt"
        self.model.write_bytes(b"checkpoint")
        self.renders_root = self.root / "renders"
        self.render_id = "flight__best50__c0.45-s640-st1"
        self.write_fake_render = write_fake_render
        self.car = car
        write_fake_render(
            self.renders_root,
            self.render_id,
            video_id="flight.avi",
            video_path=self.video,
            model_path=self.model,
            detections={15: [car(track_id=4)]},
        )
        server = TestServer(
            create_app(
                config,
                FakeRecognizer(),
                output_dir=self.root,
                video_library=VideoLibrary(self.videos),
                render_manager=RenderManager(self.renders_root, ffmpeg="ffmpeg-for-tests"),
            )
        )
        self.client = TestClient(server, headers={GUARD_HEADER: "1"})
        await self.client.start_server()
        self.raw_client = TestClient(server)
        await self.raw_client.start_server()

        session = await self.client.post("/api/session")
        self.session_id = (await session.json())["session_id"]

    async def asyncTearDown(self) -> None:
        await self.raw_client.close()
        await self.client.close()

    def _dir(self, name: str) -> Path:
        return self.root / "sessions" / name

    async def _shot(self, name: str = "flight01", **query: Any) -> Any:
        from urllib.parse import urlencode

        params = {
            "session": name,
            "source_time_ms": "1500",
            "session_id": self.session_id,
            "video": "flight.avi",
            "render": self.render_id,
            **{key: str(value) for key, value in query.items()},
        }
        # An empty value leaves the parameter out: render="" is a shot taken
        # from a video that has not been run through a model.
        params = {key: value for key, value in params.items() if value != ""}
        return await self.client.post(f"/api/capture/shot?{urlencode(params)}")

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
            "/api/capture/shot?session=x&source_time_ms=0&video=flight.avi"
        )
        self.assertEqual(response.status, 403)

    async def test_the_frame_is_read_from_the_video_at_that_time(self) -> None:
        """From the original file on the server, at the moment on screen.

        The page used to post the frame it was showing. What it shows can now be
        the rendered video, whose boxes sit over the OSD digits OCR reads, so the
        clean frame has to come from the file instead.
        """
        import cv2

        body = await (await self._shot(source_time_ms=1500)).json()
        saved = cv2.imread(str(self._dir("flight01") / body["image"]))
        # Frame 15 of the test video is filled with brightness 150.
        self.assertAlmostEqual(float(saved.mean()), 150.0, delta=6)
        self.assertEqual(body["source_time_ms"], 1500.0)

    async def test_the_boxes_come_from_the_render_of_that_frame(self) -> None:
        """Not from asking a model again: the render already drew this frame."""
        body = await (await self._shot()).json()
        self.assertEqual(body["detections"]["status"], "ok")
        self.assertEqual(body["detections"]["counts"], {"car": 1})
        # Zero, and meant literally: these boxes were drawn on this frame.
        self.assertEqual(body["detections"]["age_ms"], 0.0)

    async def test_a_shot_without_a_render_says_so(self) -> None:
        # "Never ran" and "found nothing" are fixed in different ways, so they
        # must not look alike.
        body = await (await self._shot(render="")).json()
        self.assertEqual(body["detections"]["status"], "no_render")
        self.assertEqual(body["detections"]["counts"], {})
        self.assertFalse(body["annotated"])

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
        body = await (await self._shot(render="")).json()
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

        await self._shot(render="")
        body = await (await self.client.get("/api/capture/shots?session=flight01")).json()
        stages = {shot["frame_index"]: shot["stage"] for shot in body["shots"]}
        self.assertEqual(stages[0], "ready")
        self.assertEqual(stages[1], "no_render")

    async def test_finding_nothing_still_counts_as_having_run(self) -> None:
        # A clear frame is an answer. Filing it with the failures would have the
        # page warn about every capture of an empty road.
        # Frame 5 of the render has no boxes on it.
        await self._shot(source_time_ms=500)
        body = await (await self.client.get("/api/capture/shots?session=flight01")).json()
        self.assertEqual(body["shots"][0]["detections"]["status"], "ok")
        self.assertEqual(body["shots"][0]["detections"]["counts"], {})
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


    async def test_a_render_of_another_video_is_refused(self) -> None:
        from _video_fixture import write_test_video

        write_test_video(self.videos / "other.avi")
        response = await self._shot(video="other.avi")
        self.assertEqual(response.status, 400)
        self.assertFalse(self._dir("flight01").exists())

    async def test_a_video_outside_the_folder_is_not_found(self) -> None:
        for video in ("../flight.avi", "missing.avi", str(self.video)):
            with self.subTest(video=video):
                response = await self._shot(video=video, render="")
                self.assertEqual(response.status, 404)

    async def test_a_time_past_the_end_takes_the_last_frame(self) -> None:
        body = await (await self._shot(source_time_ms=60_000, render="")).json()
        self.assertEqual(body["source_time_ms"], 1900.0)

    async def test_a_striding_render_snaps_to_a_frame_it_drew(self) -> None:
        """A render made every second frame shows only even frames.

        1.45 s is frame 14.5 of the source. Rounding to a source frame could land
        on 15, which the render never drew, and file boxes from nowhere.
        """
        stride_id = "flight__best50__c0.45-s640-st2"
        self.write_fake_render(
            self.renders_root,
            stride_id,
            video_id="flight.avi",
            video_path=self.video,
            model_path=self.model,
            stride=2,
            detections={14: [self.car(track_id=9)]},
        )
        body = await (await self._shot(render=stride_id, source_time_ms=1450)).json()
        self.assertEqual(body["source_time_ms"], 1400.0)
        self.assertEqual(body["detections"]["counts"], {"car": 1})

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
