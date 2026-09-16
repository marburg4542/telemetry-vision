import asyncio
import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.parse import urlencode

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
        from aiohttp.test_utils import TestClient, TestServer

        from _video_fixture import thread_launcher, write_test_video
        from telemetry_ocr.config import AppConfig, FieldConfig, RuntimeConfig
        from telemetry_ocr.ocr import OCRResult
        from telemetry_ocr.render_job import RenderManager
        from telemetry_ocr.video_library import VideoLibrary
        from telemetry_ocr.webapp import GUARD_HEADER, create_app
        from vision_lab.model_registry import ModelRegistry

        class FakeRecognizer:
            def recognize(self, images):
                return [OCRResult("12", 0.99) for _ in images]

        self.FakeRecognizer = FakeRecognizer
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)

        # A repository with a registry: one checkpoint on disk, one whose file
        # has gone, and one unregistered .pt the app can offer to add.
        self.repo = self.root / "repo"
        (self.repo / "configs").mkdir(parents=True)
        (self.repo / "configs" / "paths.yaml").write_text(
            "version: 1\n"
            "asset_root: .\n"
            "models:\n"
            "  best50: models/yolo/best.pt\n"
            "  run7: runs/visdrone_custom_7th/weights/best.pt\n"
            "default_model: best50\n",
            encoding="utf-8",
        )
        self.best_path = self.repo / "models" / "yolo" / "best.pt"
        self.best_path.parent.mkdir(parents=True)
        self.best_path.write_bytes(b"fake-yolo-checkpoint")
        self.extra_path = self.repo / "models" / "yolo" / "extra.pt"
        self.extra_path.write_bytes(b"another-checkpoint")
        self.registry = ModelRegistry(self.repo)

        self.videos = self.root / "videos"
        self.video = write_test_video(self.videos / "flight.avi")
        # Set, so renders run straight through unless a test holds them.
        self.render_gate = threading.Event()
        self.render_gate.set()
        self.render_entered = threading.Event()
        self.renders = RenderManager(
            self.root / "renders",
            launcher=thread_launcher(self.render_gate, self.render_entered),
            ffmpeg="ffmpeg-for-tests",
        )

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
        self.config = config
        server = TestServer(
            create_app(
                config,
                FakeRecognizer(),
                output_dir=self.root / "output",
                allowed_origins={"https://marburg4542.github.io"},
                model_registry=self.registry,
                video_library=VideoLibrary(self.videos),
                render_manager=self.renders,
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
        self.render_gate.set()
        await self.raw_client.close()
        await self.client.close()
        await asyncio.to_thread(self.renders.shutdown)
        self.temp.cleanup()

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
        self.assertIn('id="modelSelect"', page_text)
        self.assertIn('id="videoSelect"', page_text)
        self.assertIn('id="renderButton"', page_text)
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

    async def test_health_reports_the_build_the_page_checks_against(self):
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

    # -- models ------------------------------------------------------------------

    async def test_the_model_list_is_the_registry(self):
        listed = await (await self.client.get("/api/models")).json()
        by_name = {model["name"]: model for model in listed["models"]}
        self.assertEqual(set(by_name), {"best50", "run7"})
        self.assertTrue(by_name["best50"]["available"])
        self.assertFalse(by_name["run7"]["available"])
        self.assertTrue(by_name["best50"]["default"])
        self.assertEqual(listed["default"], "best50")
        self.assertEqual(by_name["best50"]["label"], "models/yolo/best.pt")
        # Labels only: the absolute path stays on the server.
        self.assertNotIn(str(self.best_path), json.dumps(listed))

    async def test_a_model_is_added_from_the_files_the_server_found(self):
        found = await (await self.client.get("/api/models/candidates")).json()
        self.assertEqual([item["label"] for item in found["candidates"]], ["models/yolo/extra.pt"])
        self.assertNotIn(str(self.extra_path), json.dumps(found))
        added = await self.client.post(
            "/api/models", json={"candidate_id": found["candidates"][0]["id"], "name": "extra"}
        )
        self.assertEqual(added.status, 201)
        listed = await (await self.client.get("/api/models")).json()
        self.assertIn("extra", [model["name"] for model in listed["models"]])

    async def test_a_path_cannot_be_added_as_a_model(self):
        """A checkpoint is a pickle, so what is loaded is decided on this machine."""
        for candidate in (str(self.extra_path), "models/yolo/extra.pt", "../../x.pt", ""):
            with self.subTest(candidate=candidate):
                response = await self.client.post(
                    "/api/models", json={"candidate_id": candidate, "name": "sneaky"}
                )
                self.assertEqual(response.status, 400)
        not_json = await self.client.post(
            "/api/models", data=b"not json", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(not_json.status, 400)

    async def test_removing_a_model_keeps_its_file_and_moves_the_default(self):
        response = await self.client.delete("/api/models/best50")
        self.assertEqual(response.status, 200)
        listed = await response.json()
        self.assertNotIn("best50", [model["name"] for model in listed["models"]])
        self.assertTrue(self.best_path.is_file(), "removing a model deleted its checkpoint")
        self.assertEqual(listed["default"], "run7")
        self.assertEqual((await self.client.delete("/api/models/best50")).status, 400)

    async def test_the_default_model_can_be_chosen(self):
        chosen = await self.client.post("/api/models/default", json={"name": "run7"})
        self.assertEqual((await chosen.json())["default"], "run7")
        unknown = await self.client.post("/api/models/default", json={"name": "ghost"})
        self.assertEqual(unknown.status, 400)

    # -- videos ------------------------------------------------------------------

    async def test_videos_are_listed_from_the_folder(self):
        listed = await (await self.client.get("/api/videos")).json()
        self.assertEqual([video["id"] for video in listed["videos"]], ["flight.avi"])
        video = listed["videos"][0]
        self.assertEqual(video["frames"], 20)
        self.assertAlmostEqual(video["fps"], 10.0)
        self.assertEqual(video["width"], 160)

    async def test_a_video_is_served_whole_and_by_range(self):
        whole = await self.client.get("/api/videos/source?video=flight.avi")
        self.assertEqual(whole.status, 200)
        self.assertEqual(len(await whole.read()), self.video.stat().st_size)
        # A range is what lets the page seek a long file without downloading it.
        part = await self.client.get(
            "/api/videos/source?video=flight.avi", headers={"Range": "bytes=0-99"}
        )
        self.assertEqual(part.status, 206)
        self.assertEqual(len(await part.read()), 100)
        for video in ("missing.avi", "../videos/flight.avi", str(self.video)):
            with self.subTest(video=video):
                response = await self.client.get(
                    f"/api/videos/source?{urlencode({'video': video})}"
                )
                self.assertEqual(response.status, 404)

    # -- renders -----------------------------------------------------------------

    def _render_request(self, **overrides):
        return {
            "video": "flight.avi",
            "model": "best50",
            "confidence": 0.45,
            "image_size": 640,
            "stride": 1,
            **overrides,
        }

    async def _wait_for_render(self, *statuses):
        deadline = time.monotonic() + 10
        job = {}
        while time.monotonic() < deadline:
            job = await (await self.client.get("/api/renders/current")).json()
            if job["status"] in statuses:
                return job
            await asyncio.sleep(0.05)
        self.fail(f"render never reached {statuses}: {job}")

    async def test_a_render_runs_is_listed_played_reused_and_deleted(self):
        started = await self.client.post("/api/renders", json=self._render_request())
        self.assertEqual(started.status, 202)
        job = await self._wait_for_render("complete")
        self.assertEqual(job["done"], 20)

        listed = await (await self.client.get("/api/renders?video=flight.avi")).json()
        self.assertEqual(len(listed["renders"]), 1)
        render = listed["renders"][0]
        self.assertTrue(render["current"])
        self.assertEqual(render["model"], "best50")

        self.assertEqual(render["status"], "complete")
        query = urlencode({"render": render["render_id"]})
        stream = await (await self.client.get(f"/api/renders/stream?{query}")).json()
        self.assertTrue(stream["complete"])
        run = stream["runs"][0]
        for file in ("init.mp4", run["segments"][0]["file"]):
            params = urlencode({"render": render["render_id"], "run": run["run"], "file": file})
            segment = await self.client.get(f"/api/renders/segment?{params}")
            self.assertEqual(segment.status, 200)
        for run_name, file in (("..", "init.mp4"), (run["run"], "../meta.json")):
            params = urlencode({"render": render["render_id"], "run": run_name, "file": file})
            self.assertEqual((await self.client.get(f"/api/renders/segment?{params}")).status, 404)
        # Only renders from before segments are played as one file.
        self.assertEqual((await self.client.get(f"/api/renders/video?{query}")).status, 404)

        # The same files and settings again: played, not rendered twice.
        again = await self.client.post("/api/renders", json=self._render_request())
        self.assertEqual(again.status, 200)
        self.assertEqual((await again.json())["existing"]["render_id"], render["render_id"])

        deleted = await self.client.delete(
            f"/api/renders?{urlencode({'render': render['render_id']})}"
        )
        self.assertEqual(deleted.status, 200)
        self.assertEqual((await (await self.client.get("/api/renders")).json())["renders"], [])
        self.assertTrue(self.video.is_file(), "deleting a render deleted the video")

    async def test_a_second_render_waits_and_a_cancelled_one_continues(self):
        self.render_gate.clear()
        first = await self.client.post("/api/renders", json=self._render_request())
        self.assertEqual(first.status, 202)
        busy = await self.client.post("/api/renders", json=self._render_request(stride=2))
        self.assertEqual(busy.status, 409)
        self.assertTrue(await asyncio.to_thread(self.render_entered.wait, 5))
        cancelled = await self.client.delete("/api/renders/current")
        self.assertTrue((await cancelled.json())["cancelled"])
        self.render_gate.set()
        job = await self._wait_for_render("cancelled")
        self.assertTrue(job["kept"])
        listed = (await (await self.client.get("/api/renders")).json())["renders"]
        self.assertEqual(
            [(item["status"], item["current"]) for item in listed], [("partial", True)]
        )

        # Kept to watch, and started again it carries on rather than starting over.
        again = await self.client.post("/api/renders", json=self._render_request())
        self.assertEqual(again.status, 202)
        job = await self._wait_for_render("complete")
        self.assertGreater(job["start_frame"], 0)

    async def test_a_render_needs_a_registered_model_file_and_offered_settings(self):
        for body in (
            self._render_request(model="best99"),
            self._render_request(model="run7"),
            self._render_request(image_size=500),
            self._render_request(stride=9),
            self._render_request(confidence=0),
        ):
            with self.subTest(body=body):
                self.assertEqual((await self.client.post("/api/renders", json=body)).status, 400)
        missing = await self.client.post("/api/renders", json=self._render_request(video="x.avi"))
        self.assertEqual(missing.status, 404)
        job = await (await self.client.get("/api/renders/current")).json()
        self.assertEqual(job["status"], "idle")

    async def test_a_render_id_cannot_reach_outside_the_renders(self):
        for render in ("../repo", "..", "a/b", ""):
            with self.subTest(render=render):
                query = urlencode({"render": render})
                played = await self.client.get(f"/api/renders/video?{query}")
                self.assertIn(played.status, (400, 404))
                streamed = await self.client.get(f"/api/renders/stream?{query}")
                self.assertIn(streamed.status, (400, 404))
                segment = await self.client.get(
                    f"/api/renders/segment?{query}&run=run-000&file=init.mp4"
                )
                self.assertIn(segment.status, (400, 404))
                deleted = await self.client.delete(f"/api/renders?{query}")
                self.assertIn(deleted.status, (200, 400, 404))
        self.assertTrue(self.best_path.is_file())
        self.assertTrue((self.repo / "configs" / "paths.yaml").is_file())

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
            "/api/renders",
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
        """A foreign page must not be able to start a render or change the registry.

        A render occupies the GPU for as long as the video runs, and the registry
        decides which pickles this machine will load, so both stay behind the
        header that forces a preflight a foreign origin cannot pass.
        """
        for method, path in (
            ("post", "/api/renders"),
            ("post", "/api/models"),
            ("post", "/api/models/default"),
            ("delete", "/api/models/best50"),
            ("delete", "/api/renders/current"),
            ("delete", "/api/renders?render=x"),
            ("post", "/api/session"),
        ):
            with self.subTest(method=method, path=path):
                response = await getattr(self.raw_client, method)(path, json={})
                self.assertEqual(response.status, 403)
        listed = await (await self.raw_client.get("/api/models")).json()
        self.assertIn("best50", [model["name"] for model in listed["models"]])
        # Read-only endpoints stay reachable without it.
        self.assertEqual((await self.raw_client.get("/api/health")).status, 200)

    async def test_rejects_a_disallowed_origin_before_the_handler_runs(self):
        evil = {"Origin": "https://evil.example.com"}
        created = await self.client.post("/api/session", headers=evil)
        self.assertEqual(created.status, 403)
        self.assertEqual((await created.json())["error"], "Origin is not allowed")

        started = await self.client.post(
            "/api/renders", json=self._render_request(), headers=evil
        )
        self.assertEqual(started.status, 403)
        job = await (await self.client.get("/api/renders/current")).json()
        self.assertEqual(job["status"], "idle")

    async def test_rejects_a_rebound_host_header(self):
        for path in ("/", "/api/health"):
            response = await self.client.get(path, headers={"Host": "attacker.example.com"})
            self.assertEqual(response.status, 403)
            self.assertEqual((await response.json())["error"], "Host is not allowed")

        self.assertEqual((await self.client.get("/api/health")).status, 200)


if __name__ == "__main__":
    unittest.main()
