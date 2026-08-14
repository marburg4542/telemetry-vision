import asyncio
import contextlib
import importlib.util
import time
import unittest
from dataclasses import asdict

HAS_WEB_DEPS = bool(
    importlib.util.find_spec("aiohttp")
    and importlib.util.find_spec("cv2")
    and importlib.util.find_spec("numpy")
)


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp/NumPy/OpenCV not installed")
class WebAppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import cv2
        import numpy as np
        from aiohttp import FormData
        from aiohttp.test_utils import TestClient, TestServer

        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.detection import Detection
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.webapp import GUARD_HEADER, create_app

        class FakeRecognizer:
            def recognize(self, images):
                return [OCRResult("12", 0.99) for _ in images]

        self.FormData = FormData
        self.loaded_model_paths = []

        class FakeDetectionWorker:
            def __init__(worker_self):
                worker_self.state = "unloaded"
                worker_self.path = None
                worker_self.filename = None
                worker_self.latest_result = None
                worker_self.dropped_frames = 0
                worker_self.pid = 2468

            def start(worker_self, path, filename):
                worker_self.path = path
                worker_self.filename = filename
                worker_self.state = "ready"
                worker_self.latest_result = None
                self.loaded_model_paths.append(path)

            def restart(worker_self):
                if worker_self.path is None:
                    raise ValueError("No model")
                worker_self.state = "ready"

            def stop(worker_self, timeout=3.0):
                del timeout
                worker_self.state = "stopped"

            def status(worker_self):
                return {
                    "status": worker_self.state,
                    "loaded": worker_self.state == "ready",
                    "filename": worker_self.filename,
                    "worker_pid": worker_self.pid if worker_self.state == "ready" else None,
                    "tracker": "BoT-SORT",
                    "device": "cpu",
                    "error": None,
                    "dropped_frames": worker_self.dropped_frames,
                    "classes": ["aircraft"] if worker_self.state == "ready" else [],
                    "architecture": "fake-yolo",
                    "epochs": 1,
                    "trained_image_size": 640,
                }

            def submit(worker_self, frame):
                if worker_self.state != "ready":
                    return False
                worker_self.latest_result = {
                    "source_id": frame.source_id,
                    "generation": frame.generation,
                    "sequence": frame.sequence,
                    "source_time_ms": frame.source_time_ms,
                    "width": 160,
                    "height": 90,
                    "inference_ms": 12.5,
                    "decode_ms": 2.5,
                    "track_ms": 10.0,
                    "worker_total_ms": 14.0,
                    # Monotonic readings of this machine's clock; the webapp has
                    # to turn them into spans and drop them from the response.
                    "enqueued_at": time.perf_counter() - 0.05,
                    "available_at": time.perf_counter(),
                    "pipeline_ms": 50.0,
                    "completed_at": 1.0,
                    "dropped_frames": worker_self.dropped_frames,
                    "detections": [
                        asdict(
                            Detection(
                                10.0,
                                12.0,
                                80.0,
                                70.0,
                                0.91,
                                0,
                                "aircraft",
                                7,
                            )
                        )
                    ],
                }
                return True

            def latest(worker_self):
                return worker_self.latest_result

            def reset(worker_self, source_id, generation):
                worker_self.latest_result = None
                worker_self.source_id = source_id
                worker_self.generation = generation

            def drain(worker_self):
                worker_self.drain_calls = getattr(worker_self, "drain_calls", 0) + 1

        self.fake_detection_worker = FakeDetectionWorker()

        config = AppConfig(
            runtime=RuntimeConfig(enable_fallback=False, confidence_threshold=0.8),
            fields=[
                FieldConfig(
                    field_id="timestamp",
                    box=(0.0, 0.0, 0.1, 0.1),
                    allowed_chars="0123456789:",
                    regex=r"^\d{2}:\d{2}:\d{2}$",
                    data_type="string",
                    derive_from_source_time=True,
                    profile="fast",
                ),
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
            create_app(
                config,
                FakeRecognizer(),
                allowed_origins={"https://marburg4542.github.io"},
                detection_worker_factory=lambda: self.fake_detection_worker,
            )
        )
        # The real page sends the guard header on every API call, so the default
        # client does too. `raw_client` omits it to stand in for a foreign site.
        self.client = TestClient(server, headers={GUARD_HEADER: "1"})
        await self.client.start_server()
        self.raw_client = TestClient(server)
        await self.raw_client.start_server()
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        self.jpeg = encoded.tobytes()

    async def asyncTearDown(self):
        await self.raw_client.close()
        await self.client.close()

    async def test_page_health_config_and_frame_session(self):
        page = await self.client.get("/")
        self.assertEqual(page.status, 200)
        self.assertIn("Telemetry OCR", await page.text())

        health = await (await self.client.get("/api/health")).json()
        self.assertEqual(health["status"], "ready")
        config = await (await self.client.get("/api/config")).json()
        self.assertEqual(len(config["fields"]), 1)
        self.assertEqual(config["fields"][0]["field_id"], "speed")

        created = await self.client.post("/api/session")
        self.assertEqual(created.status, 201)
        session_id = (await created.json())["session_id"]
        latest = None
        for index in range(3):
            response = await self.client.post(
                f"/api/session/{session_id}/frame?source_time_ms={index * 100}&frame_index={index}",
                data=self.jpeg,
                headers={"Content-Type": "image/jpeg"},
            )
            self.assertEqual(response.status, 200)
            latest = await response.json()
        self.assertEqual(latest["width"], 160)
        self.assertEqual(latest["records"][0]["raw_text"], "12")
        self.assertTrue(latest["records"][0]["confirmed"])
        self.assertEqual(latest["records"][0]["value"], 12)

        reset = await self.client.post(f"/api/session/{session_id}/reset")
        self.assertEqual(reset.status, 200)

    async def test_idle_sessions_are_evicted_before_the_cap_applies(self):
        """A closed tab never says so, and each session pins its own pipeline."""
        from telemetry_ocr import webapp

        first = (await (await self.client.post("/api/session")).json())["session_id"]
        original_ttl = webapp.SESSION_TTL_SECONDS
        webapp.SESSION_TTL_SECONDS = -1.0
        try:
            second = (await (await self.client.post("/api/session")).json())["session_id"]
        finally:
            webapp.SESSION_TTL_SECONDS = original_ttl

        self.assertEqual(
            (await self.client.post(f"/api/session/{first}/reset")).status, 404
        )
        self.assertEqual(
            (await self.client.post(f"/api/session/{second}/reset")).status, 200
        )

    async def test_rejects_invalid_frame_and_unknown_session(self):
        missing = await self.client.post(
            "/api/session/not-found/frame?source_time_ms=0&frame_index=0",
            data=self.jpeg,
        )
        self.assertEqual(missing.status, 404)

        session_id = (await (await self.client.post("/api/session")).json())["session_id"]
        invalid = await self.client.post(
            f"/api/session/{session_id}/frame?source_time_ms=0&frame_index=0",
            data=b"not-an-image",
        )
        self.assertEqual(invalid.status, 400)
        self.assertIn("error", await invalid.json())

    async def test_static_assets_are_served(self):
        for path in (
            "/assets/app.css",
            "/assets/app.js",
            "/assets/icons/icon-192.png",
            "/assets/icons/icon-512.png",
            "/assets/icons/icon-maskable-512.png",
        ):
            response = await self.client.get(path)
            self.assertEqual(response.status, 200)
            self.assertGreater(len(await response.read()), 100)

    async def test_pwa_manifest_and_root_service_worker_are_served(self):
        page = await self.client.get("/")
        page_text = await page.text()
        self.assertIn('rel="manifest"', page_text)
        self.assertIn('id="modelInput"', page_text)
        self.assertIn('id="detectionOverlay"', page_text)
        self.assertIn("/service-worker.js", await (await self.client.get("/assets/app.js")).text())

        manifest_response = await self.client.get("/manifest.webmanifest")
        self.assertEqual(manifest_response.status, 200)
        self.assertEqual(
            manifest_response.headers.get("Content-Type"),
            "application/manifest+json",
        )
        manifest = await manifest_response.json()
        self.assertEqual(manifest["display"], "standalone")
        self.assertEqual(manifest["start_url"], "/")
        self.assertEqual(len(manifest["icons"]), 3)

        worker = await self.client.get("/service-worker.js")
        self.assertEqual(worker.status, 200)
        self.assertEqual(worker.headers.get("Service-Worker-Allowed"), "/")
        self.assertIn("telemetry-ocr-pwa", await worker.text())

    async def test_yolo_model_upload_detection_and_clear(self):
        unloaded = await self.client.get("/api/detection/model")
        unloaded_payload = await unloaded.json()
        self.assertFalse(unloaded_payload["loaded"])
        self.assertEqual(unloaded_payload["status"], "unloaded")
        no_model = await self.client.post(
            "/api/detection/frame?source_id=video-a&generation=0&sequence=0&source_time_ms=0",
            data=self.jpeg,
            headers={"Content-Type": "image/jpeg"},
        )
        self.assertEqual(no_model.status, 409)

        form = self.FormData()
        form.add_field(
            "model",
            b"fake-yolo-checkpoint",
            filename="best.pt",
            content_type="application/octet-stream",
        )
        uploaded = await self.client.post("/api/detection/model", data=form)
        self.assertEqual(uploaded.status, 202)
        upload_payload = await uploaded.json()
        self.assertEqual(upload_payload["filename"], "best.pt")
        self.assertEqual(upload_payload["classes"], ["aircraft"])
        self.assertEqual(upload_payload["tracker"], "BoT-SORT")
        self.assertEqual(upload_payload["device"], "cpu")
        self.assertTrue(self.loaded_model_paths[0].is_file())

        detected = await self.client.post(
            "/api/detection/frame?source_id=video-a&generation=1&sequence=5&source_time_ms=500&confidence=0.4&iou=0.5&image_size=640",
            data=self.jpeg,
            headers={"Content-Type": "image/jpeg"},
        )
        self.assertEqual(detected.status, 200)
        detection_payload = await detected.json()
        self.assertTrue(detection_payload["accepted"])
        latest = detection_payload["latest_result"]
        self.assertEqual(latest["width"], 160)
        self.assertEqual(latest["sequence"], 5)
        self.assertEqual(latest["detections"][0]["label"], "aircraft")
        self.assertEqual(latest["detections"][0]["track_id"], 7)
        self.assertAlmostEqual(latest["detections"][0]["confidence"], 0.91)

        # Raw monotonic clock readings are meaningless off this machine, so the
        # response carries only the differences.
        self.assertNotIn("available_at", latest)
        self.assertNotIn("enqueued_at", latest)
        self.assertNotIn("pipeline_ms", latest)
        timing = detection_payload["timing"]
        self.assertAlmostEqual(timing["worker_total_ms"], 14.0)
        self.assertAlmostEqual(timing["transport_ms"], 36.0)
        self.assertGreaterEqual(timing["harvest_ms"], 0.0)

        latest_response = await self.client.get(
            "/api/detection/result?source_time_ms=650"
        )
        self.assertEqual(latest_response.status, 200)
        self.assertEqual((await latest_response.json())["result_age_ms"], 150.0)

        reset = await self.client.post(
            "/api/detection/reset?source_id=video-a&generation=2"
        )
        self.assertEqual(reset.status, 200)
        after_reset = await self.client.get("/api/detection/result?source_time_ms=700")
        self.assertEqual(after_reset.status, 202)
        self.assertIsNone((await after_reset.json())["latest_result"])

        restarted = await self.client.post("/api/detection/restart")
        self.assertEqual(restarted.status, 202)
        self.assertTrue((await restarted.json())["loaded"])

        cleared = await self.client.delete("/api/detection/model")
        self.assertEqual(cleared.status, 200)
        self.assertFalse((await cleared.json())["loaded"])
        self.assertFalse(self.loaded_model_paths[0].exists())

    async def _load_model_and_submit(self, source_id, generation, sequence):
        if not self.loaded_model_paths:
            form = self.FormData()
            form.add_field(
                "model", b"fake", filename="best.pt", content_type="application/octet-stream"
            )
            await self.client.post("/api/detection/model", data=form)
        response = await self.client.post(
            f"/api/detection/frame?source_id={source_id}&generation={generation}"
            f"&sequence={sequence}&source_time_ms=500&confidence=0.4&iou=0.5&image_size=640",
            data=self.jpeg,
            headers={"Content-Type": "image/jpeg"},
        )
        return await response.json()

    async def test_result_long_poll_returns_early_and_times_out_cleanly(self):
        await self._load_model_and_submit("video-a", 1, 5)

        # A revision newer than the one held already exists, so this must not
        # wait out the full window.
        started = time.monotonic()
        early = await self.client.get(
            "/api/detection/result?source_time_ms=650&after_revision=0&wait_ms=1000"
        )
        self.assertEqual(early.status, 200)
        self.assertLess(time.monotonic() - started, 0.9)
        payload = await early.json()
        self.assertEqual(payload["latest_result"]["sequence"], 5)
        self.assertGreater(payload["revision"], 0)

        # Nothing newer will arrive, so this one waits out its window and still
        # answers with the current payload rather than an error.
        started = time.monotonic()
        timed_out = await self.client.get(
            f"/api/detection/result?source_time_ms=700"
            f"&after_revision={payload['revision']}&wait_ms=200"
        )
        self.assertEqual(timed_out.status, 200)
        # It has to have actually blocked. Asserting only the status let a
        # regression that ignores wait_ms entirely pass unnoticed.
        self.assertGreaterEqual(time.monotonic() - started, 0.15)

    async def test_a_client_that_renumbers_its_frames_still_blocks(self):
        """The regression guard for an unthrottled poll loop.

        The page restarts its frame sequence at zero on every seek, model change
        and settings change. When the server tracked the high-water mark of that
        sequence, a renumbered client asked for "newer than 0", the server
        compared it against the pre-reset peak, concluded there was nothing to
        wait for and answered instantly — every time, as fast as the client
        could ask. One session logged 1.8 million of those against 19,728 real
        frames, on the event loop that also serves OCR.

        The revision counter is the server's own and never goes backwards, so
        renumbering cannot make it look stale.
        """
        await self._load_model_and_submit("video-a", 1, 300)
        after = (await (await self.client.get(
            "/api/detection/result?source_time_ms=100&after_revision=0&wait_ms=0"
        )).json())["revision"]

        # Exactly what the client does on a seek: reset, then start again at 0.
        await self.client.post("/api/detection/reset?source_id=video-a&generation=2")
        await self._load_model_and_submit("video-a", 2, 0)

        # The renumbered result is newer, so this returns without waiting.
        fresh = await self.client.get(
            f"/api/detection/result?source_time_ms=100&after_revision={after}&wait_ms=500"
        )
        self.assertEqual(fresh.status, 200)
        renumbered = await fresh.json()
        self.assertEqual(renumbered["latest_result"]["sequence"], 0)
        self.assertGreater(renumbered["revision"], after)

        # And once that one is acknowledged there is nothing newer, so the next
        # poll blocks instead of returning instantly. This is the assertion the
        # flood would have failed.
        started = time.monotonic()
        blocked = await self.client.get(
            f"/api/detection/result?source_time_ms=100"
            f"&after_revision={renumbered['revision']}&wait_ms=200"
        )
        self.assertEqual(blocked.status, 200)
        self.assertGreaterEqual(time.monotonic() - started, 0.15)

    async def test_result_long_poll_does_not_starve_the_ocr_path(self):
        """OCR shares this event loop and must stay responsive during a poll.

        A long poll that blocked the loop rather than awaiting would stall the
        frames the page keeps sending, which is the one regression this change
        must not cause.
        """
        session = await (await self.client.post("/api/session")).json()
        poll = asyncio.ensure_future(
            self.client.get(
                "/api/detection/result?source_time_ms=100&after_sequence=99&wait_ms=1000"
            )
        )
        await asyncio.sleep(0)
        try:
            recognized = await asyncio.wait_for(
                self.client.post(
                    f"/api/session/{session['session_id']}/frame"
                    "?source_time_ms=120&frame_index=1",
                    data=self.jpeg,
                    headers={"Content-Type": "image/jpeg"},
                ),
                timeout=5,
            )
            self.assertEqual(recognized.status, 200)
        finally:
            poll.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await poll

    async def test_health_reports_the_build_the_page_checks_against(self):
        from pathlib import Path

        from telemetry_ocr import __version__, webapp

        payload = await (await self.client.get("/api/health")).json()
        self.assertEqual(payload["version"], __version__)
        # The page hard-codes its own build and warns when the two differ, so a
        # stale service-worker copy of app.js is reported out loud instead of
        # being mistaken for a bug in the current code.
        source = (Path(webapp.__file__).with_name("web_static") / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn(f'const APP_BUILD = "{__version__}"', source)

    async def test_result_long_poll_rejects_a_bad_wait(self):
        response = await self.client.get(
            "/api/detection/result?source_time_ms=100&after_sequence=0&wait_ms=abc"
        )
        self.assertEqual(response.status, 400)

    async def test_yolo_upload_rejects_non_pt_files(self):
        form = self.FormData()
        form.add_field(
            "model",
            b"not-a-model",
            filename="best.onnx",
            content_type="application/octet-stream",
        )
        response = await self.client.post("/api/detection/model", data=form)
        self.assertEqual(response.status, 400)
        self.assertIn(".pt", (await response.json())["error"])

    async def test_api_allows_loopback_and_configured_cors_origins(self):
        response = await self.client.options(
            "/api/session",
            headers={
                "Origin": "http://localhost:8123",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status, 204)
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Origin"),
            "http://localhost:8123",
        )

        response = await self.client.options(
            "/api/session",
            headers={
                "Origin": "https://marburg4542.github.io",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        self.assertEqual(response.status, 204)
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Origin"),
            "https://marburg4542.github.io",
        )
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Private-Network"), "true"
        )

        response = await self.client.get(
            "/api/health", headers={"Origin": "https://example.com"}
        )
        self.assertEqual(response.status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)

        allowed = await self.client.options(
            "/api/detection/model",
            headers={
                "Origin": "https://marburg4542.github.io",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-telemetry-ocr",
            },
        )
        self.assertIn(
            "X-Telemetry-OCR", allowed.headers.get("Access-Control-Allow-Headers", "")
        )

    async def test_mutating_api_requires_the_guard_header(self):
        """A cross-origin multipart upload must never reach the .pt loader.

        multipart/form-data is CORS-safelisted, so without a header that forces a
        preflight any page the user has open could POST a pickle-backed model.
        """
        form = self.FormData()
        form.add_field(
            "model", b"payload", filename="evil.pt", content_type="application/octet-stream"
        )
        blocked = await self.raw_client.post("/api/detection/model", data=form)
        self.assertEqual(blocked.status, 403)
        self.assertIn("X-Telemetry-OCR", (await blocked.json())["error"])
        self.assertEqual(self.loaded_model_paths, [])

        for path in ("/api/session", "/api/detection/reset?source_id=a&generation=0"):
            self.assertEqual((await self.raw_client.post(path)).status, 403)
        self.assertEqual(
            (await self.raw_client.delete("/api/detection/model")).status, 403
        )

        # Read-only endpoints stay reachable without it.
        self.assertEqual((await self.raw_client.get("/api/health")).status, 200)

    async def test_rejects_a_disallowed_origin_before_the_handler_runs(self):
        evil = {"Origin": "https://evil.example.com"}
        created = await self.client.post("/api/session", headers=evil)
        self.assertEqual(created.status, 403)
        self.assertEqual((await created.json())["error"], "Origin is not allowed")

        form = self.FormData()
        form.add_field(
            "model", b"payload", filename="evil.pt", content_type="application/octet-stream"
        )
        upload = await self.client.post("/api/detection/model", data=form, headers=evil)
        self.assertEqual(upload.status, 403)
        self.assertEqual(self.loaded_model_paths, [])

    async def test_rejects_a_rebound_host_header(self):
        for path in ("/", "/api/health"):
            response = await self.client.get(path, headers={"Host": "attacker.example.com"})
            self.assertEqual(response.status, 403)
            self.assertEqual((await response.json())["error"], "Host is not allowed")

        self.assertEqual((await self.client.get("/api/health")).status, 200)


if __name__ == "__main__":
    unittest.main()
