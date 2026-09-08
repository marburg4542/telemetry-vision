"""LINE's two size ceilings, and the picture that has to fit under both.

An image message carries two URLs and LINE measures them against limits an order
of magnitude apart: 10 MB for the picture, 1 MB for its preview. The captures
this app produces are full-resolution frames at q95, which sit comfortably under
the first and close enough to the second that a busy scene crosses it. LINE
refuses such a push whole -- the text goes with it -- so the check has to happen
here, before the request.

The constants describing those limits were declared and never read.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _Shot:
    image: str
    annotated: str | None = None


class _Session:
    def __init__(self, directory: Path) -> None:
        self.directory = directory


class _Store:
    """Mints a predictable token per path so URLs can be read in assertions."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.minted: list[Path] = []

    def mint(self, path: Path) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        self.minted.append(resolved)
        return f"token-{len(self.minted)}"


class _ShareTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.frames = self.root / "frames"
        self.frames.mkdir()
        self.session = _Session(self.root)
        self.store = _Store(self.root)

    def _write_frame(self, name: str, width: int, height: int) -> Path:
        """A real JPEG, because the size in bytes is the thing under test."""
        import cv2
        import numpy as np

        rng = np.random.default_rng(0)
        # Noise, not a flat fill: a solid colour compresses to a few KB and
        # would never reach either ceiling, which is the case that needs
        # exercising.
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        path = self.frames / name
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return path


class LimitTests(unittest.TestCase):
    def test_a_picture_over_ten_megabytes_is_named_and_refused(self) -> None:
        from telemetry_ocr.line_client import MAX_IMAGE_BYTES, oversized_for_line

        self.assertEqual(oversized_for_line(MAX_IMAGE_BYTES), "")
        message = oversized_for_line(MAX_IMAGE_BYTES + 1)
        self.assertTrue(message)
        # The size is in the message: "too big" without a number leaves the
        # operator no way to tell how far over it was.
        self.assertIn("10", message)

    def test_the_preview_ceiling_is_a_tenth_of_the_pictures(self) -> None:
        from telemetry_ocr.line_client import (
            MAX_IMAGE_BYTES,
            MAX_PREVIEW_BYTES,
            needs_smaller_preview,
        )

        self.assertLess(MAX_PREVIEW_BYTES, MAX_IMAGE_BYTES)
        self.assertFalse(needs_smaller_preview(MAX_PREVIEW_BYTES))
        self.assertTrue(needs_smaller_preview(MAX_PREVIEW_BYTES + 1))

    def test_a_capture_sized_frame_is_its_own_preview(self) -> None:
        # 650 KB is the largest capture on disk here. It must not be re-encoded:
        # that would cost an encode and a second token on every ordinary send.
        from telemetry_ocr.line_client import needs_smaller_preview

        self.assertFalse(needs_smaller_preview(650_000))


class ShareImageTests(_ShareTestCase):
    def test_a_frame_that_fits_is_sent_as_one_url_twice(self) -> None:
        from telemetry_ocr.webapp import _share_images

        self._write_frame("00000000.jpg", 320, 240)
        images, notes = _share_images(
            self.store, self.session, [_Shot("frames/00000000.jpg")], "https://x.example"
        )
        self.assertEqual(notes, [])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].original, images[0].preview)
        self.assertEqual(len(self.store.minted), 1, "a second token was minted for nothing")

    def test_a_heavy_frame_gets_a_separate_smaller_preview(self) -> None:
        from telemetry_ocr.line_client import MAX_PREVIEW_BYTES
        from telemetry_ocr.webapp import _share_images

        path = self._write_frame("00000000.jpg", 1920, 1080)
        self.assertGreater(
            path.stat().st_size, MAX_PREVIEW_BYTES, "the fixture is not over the ceiling"
        )

        images, notes = _share_images(
            self.store, self.session, [_Shot("frames/00000000.jpg")], "https://x.example"
        )
        self.assertEqual(notes, [])
        self.assertEqual(len(images), 1)
        self.assertNotEqual(images[0].original, images[0].preview)
        # The full frame is still what LINE serves when the picture is opened.
        self.assertEqual(self.store.minted[0], path.resolve())

        preview = self.store.minted[1]
        self.assertLessEqual(preview.stat().st_size, MAX_PREVIEW_BYTES)
        # Beside the thumbnails, never in frames/: the next capture index is
        # worked out by scanning that directory.
        self.assertEqual(preview.parent.name, "thumbs")
        self.assertEqual(
            sorted(p.name for p in self.frames.iterdir()), ["00000000.jpg"]
        )

    def test_the_annotated_copy_is_the_one_that_goes(self) -> None:
        from telemetry_ocr.webapp import _share_images

        self._write_frame("00000000.jpg", 320, 240)
        annotated = self._write_frame("00000000.boxes.jpg", 320, 240)
        images, _ = _share_images(
            self.store,
            self.session,
            [_Shot("frames/00000000.jpg", "frames/00000000.boxes.jpg")],
            "https://x.example",
        )
        self.assertEqual(len(images), 1)
        self.assertEqual(self.store.minted[0], annotated.resolve())

    def test_a_missing_file_is_reported_rather_than_raised(self) -> None:
        """The telemetry is the part someone is waiting for.

        One frame that cannot be published must not take the other frames, or
        the text, down with it.
        """
        from telemetry_ocr.webapp import _share_images

        self._write_frame("00000001.jpg", 320, 240)
        images, notes = _share_images(
            self.store,
            self.session,
            [_Shot("frames/gone.jpg"), _Shot("frames/00000001.jpg")],
            "https://x.example",
        )
        self.assertEqual(len(images), 1)
        self.assertEqual(len(notes), 1)
        self.assertIn("gone.jpg", notes[0])


class PushShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_urls_reach_the_payload(self) -> None:
        from telemetry_ocr import line_client
        from telemetry_ocr.line_client import ImageMessage, LineConfig, push_capture

        captured: dict[str, object] = {}

        async def _fake_push(config, messages, *, timeout_s):
            captured["messages"] = messages
            return {"ok": True, "status": 200, "request_id": "", "line_message": ""}

        original = line_client._push
        line_client._push = _fake_push
        self.addCleanup(setattr, line_client, "_push", original)

        await push_capture(
            LineConfig("token", "U1"),
            "telemetry",
            [ImageMessage("https://x.example/full", "https://x.example/small")],
        )
        messages = captured["messages"]
        assert isinstance(messages, list)
        self.assertEqual(messages[0]["type"], "text")
        self.assertEqual(messages[1]["originalContentUrl"], "https://x.example/full")
        self.assertEqual(messages[1]["previewImageUrl"], "https://x.example/small")

    async def test_no_more_than_four_pictures_go_in_one_push(self) -> None:
        # Five messages is LINE's cap on a push and one of them is the text, so
        # a fifth picture would spend a second message from the monthly quota.
        from telemetry_ocr import line_client
        from telemetry_ocr.line_client import ImageMessage, LineConfig, push_capture

        captured: dict[str, object] = {}

        async def _fake_push(config, messages, *, timeout_s):
            captured["messages"] = messages
            return {"ok": True, "status": 200, "request_id": "", "line_message": ""}

        original = line_client._push
        line_client._push = _fake_push
        self.addCleanup(setattr, line_client, "_push", original)

        await push_capture(
            LineConfig("token", "U1"),
            "telemetry",
            [ImageMessage.same(f"https://x.example/{i}") for i in range(6)],
        )
        messages = captured["messages"]
        assert isinstance(messages, list)
        self.assertEqual(len(messages), 5)


if __name__ == "__main__":
    unittest.main()
