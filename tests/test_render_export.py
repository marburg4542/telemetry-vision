from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2
except ImportError:  # pragma: no cover - the CPU test venv has it
    cv2 = None

from _video_fixture import FakeDetector, FakeWriter, car, write_fake_render, write_test_video
from telemetry_ocr.render_export import (
    ExportBusy,
    ExportManager,
    ExportUnavailable,
    boxes_at,
    encoder_args,
    export_filename,
)
from telemetry_ocr.render_job import (
    SOFTWARE_ENCODER,
    RenderManager,
    RenderSettings,
    find_ffmpeg,
    run_render,
)

FLIGHT_META = {"video": "day 2/flight #1.avi", "model": "best50"}


def box(track_id, x, label="car"):
    return {
        "x1": float(x), "y1": 10.0, "x2": float(x) + 20.0, "y2": 30.0,
        "confidence": 0.8, "class_id": 0, "label": label, "track_id": track_id,
    }


class FilenameTests(unittest.TestCase):
    def test_the_name_says_which_flight_which_model_and_how_much_of_it(self) -> None:
        self.assertEqual(export_filename(FLIGHT_META, "full", None), "flight-1__best50__AI.mp4")
        self.assertEqual(
            export_filename(FLIGHT_META, "small", None), "flight-1__best50__AI-720p.mp4"
        )
        # Stopped partway: said in the name, so nobody forwards ten minutes of a
        # flight believing it is all of it.
        self.assertEqual(
            export_filename(FLIGHT_META, "full", 40), "flight-1__best50__AI_partial-40%.mp4"
        )


class BoxPlacementTests(unittest.TestCase):
    """The same cases tests/track-overlay.test.js holds the page's drawing to."""

    def setUp(self) -> None:
        # Every 3rd frame: car 1 moving right, car 2 arriving at 3, car 3 leaving
        # after 0, and a box the tracker had not numbered yet at 3.
        self.frames = [0, 3, 6]
        self.boxes = {
            0: [box(1, 0), box(3, 100)],
            3: [box(1, 30), box(2, 200), box(None, 300)],
            6: [box(1, 60), box(2, 200)],
        }

    def at(self, frame, frames=None, stride=3):
        placed = boxes_at(frame, frames or self.frames, self.boxes, stride)
        return {(item["track_id"], round(item["x1"], 3)) for item in placed}

    def test_a_processed_frame_shows_its_own_boxes(self) -> None:
        self.assertEqual(self.at(3), {(1, 30.0), (2, 200.0), (None, 300.0)})

    def test_a_vehicle_on_both_sides_is_placed_between(self) -> None:
        self.assertIn((1, 10.0), self.at(1))
        self.assertIn((1, 20.0), self.at(2))

    def test_one_seen_on_one_side_only_shows_for_the_nearer_half(self) -> None:
        self.assertIn((3, 100.0), self.at(1), "leaving: still there just after")
        self.assertNotIn(3, {track for track, _ in self.at(2)}, "gone before the next frame")
        self.assertNotIn(2, {track for track, _ in self.at(1)}, "arriving: not yet")
        self.assertIn((2, 200.0), self.at(2))
        self.assertIn((None, 300.0), self.at(2), "an unnumbered box, from its side")
        self.assertNotIn(None, {track for track, _ in self.at(1)})

    def test_past_the_last_frame_the_boxes_are_held_one_stride(self) -> None:
        self.assertEqual(self.at(8), {(1, 60.0), (2, 200.0)})
        self.assertEqual(self.at(9), set())
        self.assertEqual(boxes_at(0, [], {}, 3), [])

    def test_frames_not_processed_are_not_bridged(self) -> None:
        # A gap wider than the stride: frames 0 and 9 only, as after a stop.
        frames = [0, 9]
        self.boxes[9] = [box(1, 90)]
        self.assertEqual(self.at(2, frames), {(1, 0.0), (3, 100.0)})
        self.assertEqual(self.at(4, frames), set())


class EncoderArgsTests(unittest.TestCase):
    def test_full_keeps_the_encoder_and_small_is_scaled_down_never_up(self) -> None:
        nvenc = ("h264_nvenc", ("-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "27", "-maxrate", "8M"))
        self.assertEqual(encoder_args("full", nvenc)[:8], list(nvenc[1]))
        small = encoder_args("small", nvenc)
        self.assertIn("scale=-2:'min(720,ih)'", small)
        # The render's own quality settings go; the chat preset's replace them.
        self.assertNotIn("27", small)
        self.assertNotIn("8M", small)
        self.assertEqual(small[small.index("-cq") + 1], "30")
        self.assertIn("vbr", small, "NVENC's rate mode is kept for -cq to mean anything")


@unittest.skipIf(cv2 is None, "OpenCV is required")
class ExportManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = Path(self._temp.name)
        self.video = write_test_video(self.dir / "videos" / "flight.avi")
        self.model = self.dir / "best.pt"
        self.model.write_bytes(b"checkpoint")
        self.renders = RenderManager(self.dir / "renders", ffmpeg="ffmpeg-for-tests")
        self.exports_root = self.dir / "exports"
        self.exports = self.manager()
        FakeWriter.instances.clear()

    def manager(self, video_path=None, writer=FakeWriter) -> ExportManager:
        return ExportManager(
            self.renders,
            self.exports_root,
            video_path=video_path or self.video_of,
            writer_factory=writer,
        )

    def video_of(self, video_id: str) -> Path | None:
        return self.video if video_id == "flight.avi" else None

    def fake_render(self, render_id: str = "flight__best50", **options) -> str:
        write_fake_render(
            self.dir / "renders",
            render_id,
            video_id="flight.avi",
            video_path=self.video,
            model_path=self.model,
            **options,
        )
        return render_id

    def wait(self, export_id: str, exports=None) -> dict:
        exports = exports or self.exports
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = exports.status(export_id)
            if status and status["status"] != "running":
                return status
            time.sleep(0.02)
        self.fail(f"export never finished: {exports.status(export_id)}")

    def test_every_frame_of_the_original_is_written_with_its_boxes(self) -> None:
        detections = {4: [car(1, x=40.0)], 6: [car(1, x=60.0)]}
        render_id = self.fake_render(stride=2, detections=detections)
        job = self.exports.start(render_id, "full")
        status = self.wait(job["id"])
        self.assertEqual(status["status"], "ready", status["error"])
        self.assertEqual(status["filename"], "flight__best50__AI.mp4")
        writer = FakeWriter.instances[0]
        # All 20 frames, at the source's rate: not one per processed frame.
        self.assertEqual(len(writer.frames), 20)
        self.assertEqual(writer.fps, 10.0)
        self.assertEqual((writer.width, writer.height), (160, 120))
        # Frame 1 has no box (nothing at 0 or 2); frame 5 has the car half-way
        # between where it was at 4 and at 6.
        clean, boxed = writer.frames[1], writer.frames[5]
        self.assertTrue((clean == clean[0, 0]).all(), "a frame with no box was drawn on")
        self.assertFalse((boxed == boxed[0, 0]).all(), "the box between two frames is missing")
        path, _ = self.exports.ready_file(job["id"])
        self.assertEqual(status["size_bytes"], path.stat().st_size)
        self.exports.finished(job["id"])
        self.assertFalse(path.exists())
        self.assertIsNone(self.exports.status(job["id"]))
        self.assertIsNone(self.exports.ready_file(job["id"]))

    def test_a_partial_render_exports_what_it_has_and_says_so(self) -> None:
        render_id = self.fake_render(status="partial", processed=8)
        status = self.wait(self.exports.start(render_id, "small")["id"])
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["filename"], "flight__best50__AI-720p_partial-40%.mp4")
        self.assertEqual(len(FakeWriter.instances[0].frames), 8)
        self.assertEqual(FakeWriter.instances[0].quality, "small")

    def test_one_export_at_a_time(self) -> None:
        gate = threading.Event()

        class SlowWriter(FakeWriter):
            def write(self, frame):
                gate.wait(5)
                super().write(frame)

        exports = self.manager(writer=SlowWriter)
        render_id = self.fake_render()
        first = exports.start(render_id, "full")
        try:
            with self.assertRaises(ExportBusy):
                exports.start(render_id, "small")
        finally:
            # Let the first finish before the temporary folder is taken away.
            gate.set()
            exports.shutdown()
        self.assertEqual(exports.status(first["id"])["status"], "ready")

    def test_what_cannot_be_exported_is_refused(self) -> None:
        with self.assertRaises(ExportUnavailable):
            self.exports.start("never-rendered", "full")
        with self.assertRaises(ValueError):
            self.exports.start(self.fake_render(), "huge")
        # The boxes are drawn onto the original, so without it there is nothing.
        without_video = self.manager(video_path=lambda video_id: None)
        with self.assertRaises(ExportUnavailable):
            without_video.start(self.fake_render(), "full")
        without_ffmpeg = ExportManager(
            RenderManager(self.dir / "renders", ffmpeg=""),
            self.exports_root,
            video_path=lambda video_id: self.video,
        )
        with self.assertRaises(ExportUnavailable):
            without_ffmpeg.start(self.fake_render(), "full")

    def test_an_id_that_is_not_one_of_ours_finds_nothing(self) -> None:
        for bad in ("", "../renders", "0" * 31, "g" * 32):
            with self.subTest(export_id=bad):
                self.assertIsNone(self.exports.status(bad))
                self.assertIsNone(self.exports.ready_file(bad))

    def test_a_failed_export_says_why_and_leaves_nothing(self) -> None:
        class BrokenWriter(FakeWriter):
            def write(self, frame):
                raise RuntimeError("ffmpeg exited with code 1: no space left")

        exports = self.manager(writer=BrokenWriter)
        job = exports.start(self.fake_render(), "full")
        status = self.wait(job["id"], exports)
        self.assertEqual(status["status"], "error")
        self.assertIn("no space left", status["error"])
        self.assertTrue(BrokenWriter.instances[-1].aborted)
        self.assertFalse((self.exports_root / job["id"]).exists())


@unittest.skipIf(cv2 is None or find_ffmpeg() is None, "needs ffmpeg and OpenCV")
class RealExportTests(unittest.TestCase):
    """The whole path with ffmpeg: a render stopped and continued, saved both ways."""

    def test_a_continued_render_becomes_one_playable_video_both_ways(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = write_test_video(root / "flight.avi", frames=40)
            out = root / "renders" / "flight__best50"
            seen = {"n": 0}

            def cancel_after_twenty() -> bool:
                seen["n"] += 1
                return seen["n"] > 20

            for cancelled in (cancel_after_twenty, lambda: False):
                run_render(
                    video=str(video),
                    output_dir=str(out),
                    settings={"confidence": 0.45, "iou": 0.45, "image_size": 640, "stride": 3},
                    meta={"render_id": "flight__best50", "video": "flight.avi", "model": "best50"},
                    detector=FakeDetector(),
                    publish=lambda message: None,
                    cancelled=cancelled,
                )

            renders = RenderManager(root / "renders", ffmpeg=find_ffmpeg())
            exports = ExportManager(
                renders, root / "exports", video_path=lambda video_id: video,
                encoder=SOFTWARE_ENCODER,
            )
            for quality in ("full", "small"):
                with self.subTest(quality=quality):
                    job = exports.start("flight__best50", quality)
                    deadline = time.monotonic() + 60
                    while exports.status(job["id"])["status"] == "running":
                        self.assertLess(time.monotonic(), deadline, "export took too long")
                        time.sleep(0.1)
                    status = exports.status(job["id"])
                    self.assertEqual(status["status"], "ready", status["error"])
                    path, _ = exports.ready_file(job["id"])
                    capture = cv2.VideoCapture(str(path))
                    frames = 0
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    while capture.read()[0]:
                        frames += 1
                    capture.release()
                    # Every frame of the original, at every 3rd frame tracked.
                    self.assertEqual(frames, 40)
                    # 120 px high already: the small version must not scale it up.
                    self.assertEqual(height, 120)
                    exports.finished(job["id"])

    def test_render_settings_stay_the_default_the_page_sends(self) -> None:
        # Guards the fixture above against a default that would change the frame count.
        self.assertEqual(RenderSettings().stride, 1)


if __name__ == "__main__":
    unittest.main()
