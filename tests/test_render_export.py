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

from _video_fixture import (
    FakeDetector,
    copying_export_runner,
    write_fake_render,
    write_test_video,
)
from telemetry_ocr.render_export import (
    ExportBusy,
    ExportManager,
    ExportUnavailable,
    export_filename,
    ffmpeg_command,
    join_run,
)
from telemetry_ocr.render_job import (
    SOFTWARE_ENCODER,
    FfmpegEncoder,
    RenderManager,
    RenderSettings,
    find_ffmpeg,
    run_render,
)

copying_runner = copying_export_runner


FLIGHT_META = {"video": "day 2/flight #1.avi", "model": "best50"}


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


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = Path(self._temp.name)

    def test_segments_are_joined_in_order_after_their_init(self) -> None:
        run = self.dir / "run-000"
        run.mkdir()
        for name, body in (("init.mp4", b"I"), ("seg_00000.m4s", b"A"), ("seg_00001.m4s", b"B")):
            (run / name).write_bytes(body)
        target = self.dir / "joined.mp4"
        index_run = {
            "run": "run-000",
            "segments": [{"file": "seg_00000.m4s"}, {"file": "seg_00001.m4s"}],
        }
        self.assertEqual(join_run(self.dir, index_run, target), 3)
        self.assertEqual(target.read_bytes(), b"IAB")

    def test_a_copy_keeps_the_stream_and_puts_the_index_first(self) -> None:
        command = ffmpeg_command("ffmpeg", [self.dir / "run-000.mp4"], self.dir / "out.mp4", "full")
        self.assertIn("copy", command)
        self.assertNotIn("-vf", command)
        self.assertIn("+faststart", command)
        self.assertEqual(command[command.index("-i") + 1], str(self.dir / "run-000.mp4"))

    def test_several_runs_are_read_through_the_concat_demuxer(self) -> None:
        inputs = [self.dir / "run-000.mp4", self.dir / "run-001.mp4"]
        command = ffmpeg_command("ffmpeg", inputs, self.dir / "out.mp4", "full")
        self.assertIn("concat", command)
        listing = (self.dir / "runs.txt").read_text(encoding="utf-8")
        self.assertEqual(listing.count("file '"), 2)
        self.assertLess(listing.index("run-000"), listing.index("run-001"))

    def test_the_small_version_is_scaled_down_never_up_and_sized_for_a_chat(self) -> None:
        nvenc = ("h264_nvenc", ("-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "27", "-maxrate", "8M"))
        command = ffmpeg_command(
            "ffmpeg", [self.dir / "a.mp4"], self.dir / "out.mp4", "small", nvenc
        )
        self.assertIn("scale=-2:'min(720,ih)'", command)
        # The render's own quality settings go; the chat preset's replace them.
        self.assertNotIn("27", command)
        self.assertNotIn("8M", command)
        self.assertEqual(command[command.index("-cq") + 1], "30")
        self.assertIn("vbr", command, "NVENC's rate mode is kept for -cq to mean anything")


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
        self.exports = ExportManager(self.renders, self.exports_root, runner=copying_runner)

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

    def wait(self, export_id: str) -> dict:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = self.exports.status(export_id)
            if status and status["status"] != "running":
                return status
            time.sleep(0.02)
        self.fail(f"export never finished: {self.exports.status(export_id)}")

    def test_an_export_is_made_handed_over_and_then_removed(self) -> None:
        job = self.exports.start(self.fake_render(), "full")
        status = self.wait(job["id"])
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["filename"], "flight__best50__AI.mp4")
        path, name = self.exports.ready_file(job["id"])
        self.assertTrue(path.is_file())
        # The init segment followed by the one media segment, joined.
        self.assertEqual(status["size_bytes"], path.stat().st_size)
        self.assertEqual([p.name for p in path.parent.iterdir()], [name], "joined runs left behind")

        self.exports.finished(job["id"])
        self.assertFalse(path.exists())
        self.assertIsNone(self.exports.status(job["id"]))
        self.assertIsNone(self.exports.ready_file(job["id"]))

    def test_a_partial_render_exports_what_it_has_and_says_so(self) -> None:
        render_id = self.fake_render(status="partial", processed=8)
        status = self.wait(self.exports.start(render_id, "small")["id"])
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["filename"], "flight__best50__AI-720p_partial-40%.mp4")

    def test_one_export_at_a_time(self) -> None:
        gate = threading.Event()

        def slow(command, duration_s, progress):
            gate.wait(5)
            copying_runner(command, duration_s, progress)

        exports = ExportManager(self.renders, self.exports_root, runner=slow)
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
        without = ExportManager(
            RenderManager(self.dir / "renders", ffmpeg=""), self.exports_root, runner=copying_runner
        )
        with self.assertRaises(ExportUnavailable):
            without.start(self.fake_render(), "full")

    def test_an_id_that_is_not_one_of_ours_finds_nothing(self) -> None:
        for bad in ("", "../renders", "0" * 31, "g" * 32):
            with self.subTest(export_id=bad):
                self.assertIsNone(self.exports.status(bad))
                self.assertIsNone(self.exports.ready_file(bad))

    def test_a_failed_export_says_why_and_leaves_nothing(self) -> None:
        def broken(command, duration_s, progress):
            raise RuntimeError("ffmpeg exited with code 1: no space left")

        exports = ExportManager(self.renders, self.exports_root, runner=broken)
        job = exports.start(self.fake_render(), "full")
        deadline = time.monotonic() + 10
        while exports.status(job["id"])["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        status = exports.status(job["id"])
        self.assertEqual(status["status"], "error")
        self.assertIn("no space left", status["error"])
        self.assertFalse((self.exports_root / job["id"]).exists())


@unittest.skipIf(cv2 is None or find_ffmpeg() is None, "needs ffmpeg and OpenCV")
class RealExportTests(unittest.TestCase):
    """The whole path with ffmpeg: a render stopped and continued, exported as one file."""

    def test_two_runs_become_one_playable_video_both_ways(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = write_test_video(root / "flight.avi", frames=40)
            out = root / "renders" / "flight__best50"
            ffmpeg = find_ffmpeg()
            seen = {"n": 0}

            def cancel_after_twenty() -> bool:
                seen["n"] += 1
                return seen["n"] > 20

            for cancelled in (cancel_after_twenty, lambda: False):
                run_render(
                    video=str(video),
                    output_dir=str(out),
                    settings={"confidence": 0.45, "iou": 0.45, "image_size": 640, "stride": 1},
                    meta={"render_id": "flight__best50", "video": "flight.avi", "model": "best50"},
                    detector=FakeDetector(),
                    encoder_factory=lambda run_dir, w, h, fps: FfmpegEncoder(
                        ffmpeg, run_dir, w, h, fps
                    ),
                    publish=lambda message: None,
                    cancelled=cancelled,
                )
            self.assertTrue((out / "run-001").is_dir(), "the render was not continued")

            renders = RenderManager(root / "renders", ffmpeg=ffmpeg)
            exports = ExportManager(renders, root / "exports", encoder=SOFTWARE_ENCODER)
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
                    self.assertEqual(frames, 40, "both runs are in the file, once each")
                    # 120 px high already: the small version must not scale it up.
                    self.assertEqual(height, 120)
                    exports.finished(job["id"])

    def test_render_settings_stay_the_default_the_page_sends(self) -> None:
        # Guards the fixture above against a default that would change the frame count.
        self.assertEqual(RenderSettings().stride, 1)


if __name__ == "__main__":
    unittest.main()
