"""The one path reachable from outside this machine.

A tunnel makes the local server addressable from the internet, so what matters
is not that the image endpoint works but that nothing else does. These tests are
written from the position of someone who has the public hostname and is trying
everything else with it.
"""

from __future__ import annotations

import importlib.util
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

HAS_WEB_DEPS = bool(
    importlib.util.find_spec("aiohttp")
    and importlib.util.find_spec("cv2")
    and importlib.util.find_spec("numpy")
)

PUBLIC_HOST = "drone.example.com"


class ShareStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "sessions"
        (self.root / "s1" / "frames").mkdir(parents=True)
        self.image = self.root / "s1" / "frames" / "00000000.jpg"
        self.image.write_bytes(b"\xff\xd8pretend-jpeg")

    def _store(self, ttl: float = 600.0) -> Any:
        from telemetry_ocr.public_share import ShareStore

        return ShareStore(self.root, ttl_s=ttl)

    def test_a_token_resolves_to_its_file(self) -> None:
        store = self._store()
        token = store.mint(self.image)
        self.assertEqual(store.resolve(token), self.image.resolve())

    def test_tokens_are_long_and_unguessable(self) -> None:
        # The token is the entire authorisation on a publicly reachable URL.
        store = self._store()
        tokens = {store.mint(self.image) for _ in range(20)}
        self.assertEqual(len(tokens), 20)
        self.assertTrue(all(len(token) >= 40 for token in tokens))

    def test_a_file_outside_the_recordings_is_refused(self) -> None:
        """The path is assembled from a session name and a manifest entry.

        Neither is beyond the reach of something malformed, and a token that
        could name any file on disk would turn a picture endpoint into a file
        server.
        """
        outside = Path(self.temp.name) / "secrets.txt"
        outside.write_text("private", encoding="utf-8")
        store = self._store()
        with self.assertRaises(ValueError):
            store.mint(outside)

    def test_a_traversal_out_of_the_root_is_refused(self) -> None:
        outside = Path(self.temp.name) / "secrets.txt"
        outside.write_text("private", encoding="utf-8")
        store = self._store()
        with self.assertRaises(ValueError):
            store.mint(self.root / "s1" / ".." / ".." / "secrets.txt")

    def test_a_missing_file_is_refused_before_a_token_exists(self) -> None:
        store = self._store()
        with self.assertRaises(FileNotFoundError):
            store.mint(self.root / "s1" / "frames" / "absent.jpg")

    def test_a_token_stops_working_when_it_expires(self) -> None:
        # A URL that leaks or is logged somewhere should die on its own rather
        # than staying live because nobody remembered to revoke it.
        store = self._store(ttl=0.05)
        token = store.mint(self.image)
        self.assertIsNotNone(store.resolve(token))
        time.sleep(0.1)
        self.assertIsNone(store.resolve(token))

    def test_revoking_takes_effect_at_once(self) -> None:
        store = self._store()
        token = store.mint(self.image)
        store.revoke(token)
        self.assertIsNone(store.resolve(token))

    def test_an_unknown_token_is_simply_unknown(self) -> None:
        self.assertIsNone(self._store().resolve("not-a-real-token"))

    def test_a_url_must_be_https(self) -> None:
        # LINE rejects a non-HTTPS image URL, and finding that out from LINE's
        # error is a worse way to learn it than from the setting.
        from telemetry_ocr.public_share import public_url

        with self.assertRaises(ValueError):
            public_url("http://drone.example.com", "token")
        self.assertEqual(
            public_url("https://drone.example.com/", "abc"),
            "https://drone.example.com/public/img/abc",
        )


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp/NumPy/OpenCV not installed")
class PublicRouteTests(unittest.IsolatedAsyncioTestCase):
    """What someone holding the tunnel hostname can and cannot reach."""

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
        self.sessions = self.root / "sessions" / "s1" / "frames"
        self.sessions.mkdir(parents=True)
        frame = np.full((60, 80, 3), 120, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        self.image = self.sessions / "00000000.jpg"
        self.image.write_bytes(encoded.tobytes())

        class FakeRecognizer:
            def recognize(self, images: Any) -> Any:
                return [OCRResult("7", 0.9) for _ in images]

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
        self.app = create_app(config, FakeRecognizer(), output_dir=self.root)
        server = TestServer(self.app)
        self.client = TestClient(server, headers={GUARD_HEADER: "1"})
        await self.client.start_server()
        # Stands in for a request arriving through the tunnel: a public Host and
        # no guard header, neither of which LINE can be made to send.
        self.outside = TestClient(server, headers={"Host": PUBLIC_HOST})
        await self.outside.start_server()

    async def asyncTearDown(self) -> None:
        await self.outside.close()
        await self.client.close()

    async def test_even_the_image_path_is_refused_by_the_application(self) -> None:
        """Images moved to their own process, and this is the point of that.

        The application used to carry an exception for this path so LINE could
        reach it. It does not any more: a tunnel is aimed at the share server
        instead, so the surface reachable here is nothing at all, with no
        middleware exception standing between the API and the internet.
        """
        from telemetry_ocr.public_share import PUBLIC_PREFIX

        response = await self.outside.get(f"{PUBLIC_PREFIX}/anything")
        self.assertEqual(response.status, 403)

    async def test_everything_else_is_refused_from_outside(self) -> None:
        # The whole security case for the tunnel rests on this.
        for path in (
            "/",
            "/api/health",
            "/api/config",
            "/api/detection/models",
            "/api/capture/status?session=s1",
            "/api/line/status",
            "/assets/app.js",
        ):
            with self.subTest(path=path):
                response = await self.outside.get(path)
                self.assertEqual(response.status, 403, f"{path} was reachable")

    async def test_mutating_endpoints_are_refused_from_outside(self) -> None:
        for path in ("/api/session", "/api/detection/model", "/api/line/send?session=s1"):
            with self.subTest(path=path):
                response = await self.outside.post(path)
                self.assertEqual(response.status, 403)

    async def test_the_public_prefix_does_not_accept_writes(self) -> None:
        # GET only: the exemption is for a fetch, and nothing else.
        from telemetry_ocr.public_share import PUBLIC_PREFIX

        response = await self.outside.post(f"{PUBLIC_PREFIX}/anything")
        self.assertEqual(response.status, 403)

    async def test_a_prefix_lookalike_is_still_guarded(self) -> None:
        # /publicity or /public/img-of-something must not slip through a
        # startswith check that forgot the separator.
        for path in ("/publicimg/x", "/public/imgx", "/public", "/public/img"):
            with self.subTest(path=path):
                response = await self.outside.get(path)
                self.assertIn(response.status, (403, 404))
                if response.status == 404:
                    # A 404 from the router is fine; a 200 would not be.
                    self.assertNotEqual(response.status, 200)

    async def test_loopback_still_works_normally(self) -> None:
        # The exemption must not have loosened anything for the local page.
        self.assertEqual((await self.client.get("/api/health")).status, 200)


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp/NumPy/OpenCV not installed")
class ShareServerTests(unittest.IsolatedAsyncioTestCase):
    """The process a tunnel is actually pointed at.

    Its whole vocabulary is one route, which is the argument for it existing:
    a flaw in the application's guard cannot be reached from here because the
    endpoints it would expose are not in this app.
    """

    async def asyncSetUp(self) -> None:
        from aiohttp.test_utils import TestClient, TestServer

        from telemetry_ocr.public_share import ShareServer, ShareStore

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "sessions"
        (self.root / "s1" / "frames").mkdir(parents=True)
        self.image = self.root / "s1" / "frames" / "00000000.jpg"
        self.image.write_bytes(b"\xff\xd8pretend-jpeg-body")

        self.store = ShareStore(self.root)
        server = ShareServer(self.store)
        self.client = TestClient(TestServer(server.build_app()))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_a_minted_token_serves_the_file(self) -> None:
        from telemetry_ocr.public_share import PUBLIC_PREFIX

        token = self.store.mint(self.image)
        response = await self.client.get(f"{PUBLIC_PREFIX}/{token}")
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), self.image.read_bytes())
        # Nothing about a capture should sit in a cache anywhere.
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    async def test_an_unknown_token_is_a_plain_404(self) -> None:
        from telemetry_ocr.public_share import PUBLIC_PREFIX

        self.assertEqual((await self.client.get(f"{PUBLIC_PREFIX}/nope")).status, 404)

    async def test_an_expired_token_stops_serving(self) -> None:
        from telemetry_ocr.public_share import PUBLIC_PREFIX, ShareStore

        self.store.ttl_s = 0.05
        token = self.store.mint(self.image)
        time.sleep(0.1)
        self.assertEqual((await self.client.get(f"{PUBLIC_PREFIX}/{token}")).status, 404)
        del ShareStore

    async def test_there_is_nothing_else_to_reach(self) -> None:
        # The reason this process exists rather than an exception in the main
        # one: there are no other endpoints here to find.
        for path in ("/", "/api/health", "/api/session", "/assets/app.js", "/public/img"):
            with self.subTest(path=path):
                self.assertEqual((await self.client.get(path)).status, 404)

    async def test_it_does_not_accept_writes(self) -> None:
        from telemetry_ocr.public_share import PUBLIC_PREFIX

        token = self.store.mint(self.image)
        response = await self.client.post(f"{PUBLIC_PREFIX}/{token}")
        self.assertEqual(response.status, 405)


if __name__ == "__main__":
    unittest.main()
