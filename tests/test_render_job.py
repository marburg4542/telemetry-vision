from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - the CPU test venv has both
    cv2 = None

from telemetry_ocr import render_job
from telemetry_ocr.detection import Detection
from telemetry_ocr.render_job import (
    TRACKS_FILE,
    VIDEO_FILE,
    RenderBusy,
    RenderManager,
    RenderSettings,
    RenderUnavailable,
    is_current,
    read_meta,
    render_id_for,
    run_render,
)
from telemetry_ocr.video_library import SourceFrames, VideoLibrary

FPS = 10.0
FRAMES = 12


def write_video(path: Path, frames: int = FRAMES, width: int = 64, height: int = 48) -> None:
    """Each frame's index is readable back from its brightness."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height))
    for index in range(frames):
        writer.write(np.full((height, width, 3), index * 20, dtype=np.uint8))
    writer.release()


class FakeDetector:
    """One vehicle drifting right, tracked as id 7."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.calls = 0
        self.gate = gate

    def track(self, frame, *, confidence, iou, image_size, tracker):
        if self.gate is not None:
            self.gate.wait(5)
        self.calls += 1
        x = float(self.calls)
        return [Detection(x, 2.0, x + 10.0, 12.0, 0.9, 0, "car", 7)]


class FakeEncoder:
    instances: ClassVar[list[FakeEncoder]] = []

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path, self.width, self.height, self.fps = path, width, height, fps
        self.shapes: list[tuple[int, ...]] = []
        self.closed_with_abort: bool | None = None
        FakeEncoder.instances.append(self)

    def write(self, frame) -> None:
        self.shapes.append(frame.shape)

    def close(self, abort: bool = False) -> None:
        self.closed_with_abort = abort
        if not abort:
            self.path.write_bytes(b"mp4")


@unittest.skipIf(cv2 is None, "OpenCV is required")
class RunRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = Path(self._temp.name)
        self.video = self.dir / "flight.avi"
        write_video(self.video)
        self.out = self.dir / "render"
        self.messages: list[dict] = []
        FakeEncoder.instances.clear()

    def render(self, settings: RenderSettings | None = None, cancelled=lambda: False, video=None):
        run_render(
            video=str(video or self.video),
            output_dir=str(self.out),
            settings=_asdict(settings),
            meta={"render_id": "x", "video": "flight.avi"},
            detector=FakeDetector(),
            encoder_factory=FakeEncoder,
            publish=self.messages.append,
            cancelled=cancelled,
        )

    def test_every_frame_is_tracked_encoded_and_recorded(self) -> None:
        self.render()
        lines = (self.out / TRACKS_FILE).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), FRAMES)
        first = json.loads(lines[0])
        self.assertEqual(first["frame_index"], 0)
        self.assertEqual(first["detections"][0]["track_id"], 7)
        self.assertEqual(json.loads(lines[5])["time_ms"], 500.0)
        self.assertEqual(len(FakeEncoder.instances[0].shapes), FRAMES)
        self.assertFalse(FakeEncoder.instances[0].closed_with_abort)
        meta = read_meta(self.out)
        self.assertEqual(meta["status"], "complete")
        self.assertEqual(meta["frames_written"], FRAMES)
        self.assertEqual(meta["video"], "flight.avi")
        self.assertTrue((self.out / VIDEO_FILE).is_file())
        self.assertFalse(list(self.out.glob("*.part")))
        self.assertEqual(self.messages[-1]["kind"], "complete")

    def test_a_stride_keeps_every_nth_frame_and_halves_the_rate(self) -> None:
        self.render(RenderSettings(stride=2))
        indexes = [
            json.loads(line)["frame_index"]
            for line in (self.out / TRACKS_FILE).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(indexes, list(range(0, FRAMES, 2)))
        self.assertAlmostEqual(FakeEncoder.instances[0].fps, FPS / 2)
        self.assertEqual(read_meta(self.out)["source_frames"], FRAMES)

    def test_cancelling_leaves_no_result_behind(self) -> None:
        seen = {"n": 0}

        def cancelled() -> bool:
            seen["n"] += 1
            return seen["n"] > 3

        self.render(cancelled=cancelled)
        self.assertEqual(self.messages[-1]["kind"], "cancelled")
        self.assertTrue(FakeEncoder.instances[0].closed_with_abort)
        self.assertFalse((self.out / VIDEO_FILE).exists())
        self.assertFalse((self.out / TRACKS_FILE).exists())
        self.assertIsNone(read_meta(self.out))

    def test_odd_dimensions_are_cropped_for_the_encoder(self) -> None:
        odd = self.dir / "odd.avi"
        write_video(odd, width=65, height=49)
        self.render(video=odd)
        encoder = FakeEncoder.instances[0]
        self.assertEqual((encoder.width, encoder.height), (64, 48))
        self.assertEqual(encoder.shapes[0][:2], (48, 64))

    def test_an_unreadable_video_is_an_error(self) -> None:
        broken = self.dir / "broken.avi"
        broken.write_bytes(b"not a video")
        with self.assertRaises(RuntimeError):
            self.render(video=broken)


def _asdict(settings):
    from dataclasses import asdict

    return asdict(settings or RenderSettings())


class ThreadHandle:
    def __init__(self, thread: threading.Thread) -> None:
        self.thread = thread
        self.exitcode = 0

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def join(self, timeout=None) -> None:
        self.thread.join(timeout)

    def terminate(self) -> None:
        pass


def thread_launcher(detector_factory):
    def launch(job):
        progress: queue.Queue = queue.Queue()
        cancel = threading.Event()

        def body() -> None:
            try:
                run_render(
                    video=job["video"],
                    output_dir=job["output_dir"],
                    settings=job["settings"],
                    meta=job["meta"],
                    detector=detector_factory(),
                    encoder_factory=FakeEncoder,
                    publish=progress.put,
                    cancelled=cancel.is_set,
                )
            except Exception as exc:
                progress.put({"kind": "error", "error": str(exc)})

        thread = threading.Thread(target=body, daemon=True)
        thread.start()
        return ThreadHandle(thread), progress, cancel

    return launch


@unittest.skipIf(cv2 is None, "OpenCV is required")
class RenderManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = Path(self._temp.name)
        self.video = self.dir / "videos" / "flight.avi"
        self.video.parent.mkdir()
        write_video(self.video)
        self.model = self.dir / "best.pt"
        self.model.write_bytes(b"checkpoint")
        self.gate = threading.Event()
        self.manager = RenderManager(
            self.dir / "renders",
            launcher=thread_launcher(lambda: FakeDetector(self.gate)),
            ffmpeg="ffmpeg-for-tests",
        )
        self.addCleanup(self.gate.set)
        FakeEncoder.instances.clear()

    def start(self, settings: RenderSettings | None = None) -> dict:
        settings = settings or RenderSettings()
        return self.manager.start(
            render_id=render_id_for("flight.avi", "best50", settings),
            video_id="flight.avi",
            video_path=self.video,
            model_name="best50",
            model_label="models/yolo/best.pt",
            model_path=self.model,
            settings=settings,
        )

    def wait_for(self, *statuses: str) -> dict:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = self.manager.status()
            if status["status"] in statuses:
                return status
            time.sleep(0.02)
        self.fail(f"render never reached {statuses}: {self.manager.status()}")

    def test_a_render_runs_to_completion_and_is_listed(self) -> None:
        started = self.start()
        self.assertIn(started["status"], {"loading", "running"})
        self.gate.set()
        done = self.wait_for("complete")
        self.assertEqual(done["done"], FRAMES)
        listed = self.manager.renders("flight.avi")
        self.assertEqual(len(listed), 1)
        self.assertTrue(is_current(listed[0], self.video, self.model))

    def test_only_one_render_runs_at_a_time(self) -> None:
        self.start()
        with self.assertRaises(RenderBusy):
            self.start(RenderSettings(stride=2))
        self.gate.set()
        self.wait_for("complete")

    def test_a_render_can_be_cancelled(self) -> None:
        self.start()
        self.assertTrue(self.manager.cancel())
        self.gate.set()
        self.wait_for("cancelled")
        self.assertEqual(self.manager.renders(), [])
        self.assertFalse(self.manager.cancel())

    def test_a_replaced_video_or_checkpoint_makes_a_render_stale(self) -> None:
        self.gate.set()
        self.start()
        self.wait_for("complete")
        meta = self.manager.renders()[0]
        self.model.write_bytes(b"retrained checkpoint")
        self.assertFalse(is_current(meta, self.video, self.model))

    def test_detections_are_looked_up_by_source_frame(self) -> None:
        self.gate.set()
        settings = RenderSettings(stride=2)
        render_id = self.start(settings)["render_id"]
        self.wait_for("complete")
        self.assertEqual(self.manager.detections_at(render_id, 4)[0]["track_id"], 7)
        self.assertIsNone(self.manager.detections_at(render_id, 5))
        self.assertIsNone(self.manager.detections_at(render_id, 999))

    def test_deleting_removes_the_output_but_not_a_running_one(self) -> None:
        render_id = self.start()["render_id"]
        with self.assertRaises(RenderBusy):
            self.manager.delete(render_id)
        self.gate.set()
        self.wait_for("complete")
        self.manager.delete(render_id)
        self.assertFalse(self.manager.directory(render_id).exists())
        self.assertTrue(self.video.is_file())

    def test_render_ids_cannot_name_a_path(self) -> None:
        for bad in ("../escape", "a/b", "..", ""):
            with self.subTest(render_id=bad), self.assertRaises(ValueError):
                self.manager.directory(bad)

    def test_no_ffmpeg_is_reported_before_anything_starts(self) -> None:
        manager = RenderManager(
            self.dir / "renders", launcher=thread_launcher(FakeDetector), ffmpeg=""
        )
        with self.assertRaises(RenderUnavailable):
            manager.start(
                render_id="x",
                video_id="flight.avi",
                video_path=self.video,
                model_name="best50",
                model_label="best.pt",
                model_path=self.model,
                settings=RenderSettings(),
            )

    def test_settings_outside_the_offered_values_are_rejected(self) -> None:
        for bad in (
            RenderSettings(confidence=0.0),
            RenderSettings(image_size=500),
            RenderSettings(stride=5),
        ):
            with self.subTest(settings=bad), self.assertRaises(ValueError):
                self.start(bad)


@unittest.skipIf(cv2 is None, "OpenCV is required")
class VideoLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = Path(self._temp.name) / "videos"
        (self.dir / "day2").mkdir(parents=True)
        write_video(self.dir / "flight.avi")
        write_video(self.dir / "day2" / "second.avi", frames=5)
        (self.dir / "notes.txt").write_text("not a video", encoding="utf-8")

    def test_videos_are_listed_by_relative_id_with_their_timing(self) -> None:
        listed = {video.id: video for video in VideoLibrary(self.dir).videos()}
        self.assertEqual(set(listed), {"flight.avi", "day2/second.avi"})
        self.assertEqual(listed["flight.avi"].frames, FRAMES)
        self.assertAlmostEqual(listed["flight.avi"].fps, FPS)
        self.assertEqual(listed["day2/second.avi"].duration_ms, 500.0)

    def test_an_id_that_is_not_in_the_listing_is_not_found(self) -> None:
        library = VideoLibrary(self.dir)
        self.assertIsNone(library.get("../outside.avi"))
        self.assertIsNone(library.get("notes.txt"))

    def test_a_missing_folder_lists_nothing(self) -> None:
        self.assertEqual(VideoLibrary(self.dir / "absent").videos(), [])
        self.assertEqual(VideoLibrary(None).videos(), [])

    def test_frames_are_read_by_index_forwards_and_after_a_seek(self) -> None:
        frames = SourceFrames()
        self.addCleanup(frames.close)
        path = self.dir / "flight.avi"
        for index in (0, 1, 3, 10, 2, 11):
            with self.subTest(index=index):
                # MJPG is lossy, so compare brightness within a margin.
                self.assertAlmostEqual(float(frames.read(path, index).mean()), index * 20, delta=6)
        with self.assertRaises(ValueError):
            frames.read(path, FRAMES + 5)


class FfmpegTests(unittest.TestCase):
    def test_the_environment_variable_names_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / "ffmpeg.exe"
            fake.write_bytes(b"")
            os.environ["TELEMETRY_OCR_FFMPEG"] = str(fake)
            try:
                self.assertEqual(render_job.find_ffmpeg(), str(fake))
            finally:
                os.environ.pop("TELEMETRY_OCR_FFMPEG", None)

    @unittest.skipIf(cv2 is None or shutil.which("ffmpeg") is None, "needs ffmpeg and OpenCV")
    def test_the_encoder_writes_a_video_opencv_can_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "out.mp4"
            encoder = render_job.FfmpegEncoder(shutil.which("ffmpeg"), path, 64, 48, 10.0)
            for index in range(8):
                encoder.write(np.full((48, 64, 3), index * 25, dtype=np.uint8))
            encoder.close()
            capture = cv2.VideoCapture(str(path))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, 8)


class ImportBoundaryTests(unittest.TestCase):
    def test_the_server_side_modules_load_no_ml_framework(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        probe = (
            "import sys; import telemetry_ocr.render_job, telemetry_ocr.video_library,"
            " telemetry_ocr.annotate;"
            "print('torch' in sys.modules or 'paddle' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(source_root)},
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
