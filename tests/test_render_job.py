from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - the CPU test venv has both
    cv2 = None

from _video_fixture import FakeDetector, ThreadHandle, thread_launcher, write_fake_render
from telemetry_ocr import render_job
from telemetry_ocr.render_job import (
    META_FILE,
    TRACKER_FILE,
    TRACKS_FILE,
    RenderBusy,
    RenderManager,
    RenderSettings,
    TrackIndex,
    is_current,
    last_frame,
    read_meta,
    render_id_for,
    render_state,
    run_render,
    settle_render,
    tracker_config,
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


def _asdict(settings):
    from dataclasses import asdict

    return asdict(settings or RenderSettings())


def _cancel_after(frames: int):
    """Asked once per frame read; says stop once that many have been read."""
    seen = {"n": 0}

    def cancelled() -> bool:
        seen["n"] += 1
        return seen["n"] > frames

    return cancelled


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

    def render(
        self,
        settings: RenderSettings | None = None,
        cancelled=lambda: False,
        video=None,
        detector=None,
        meta=None,
        fresh=False,
    ):
        run_render(
            video=str(video or self.video),
            output_dir=str(self.out),
            settings=_asdict(settings),
            meta=meta or {"render_id": "x", "video": "flight.avi"},
            detector=detector or FakeDetector(),
            publish=self.messages.append,
            cancelled=cancelled,
            fresh=fresh,
        )

    def lines(self) -> list[dict]:
        text = (self.out / TRACKS_FILE).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]

    def test_every_frame_is_tracked_and_recorded_and_nothing_else_written(self) -> None:
        self.render()
        lines = self.lines()
        self.assertEqual(len(lines), FRAMES)
        self.assertEqual(lines[0]["frame_index"], 0)
        self.assertEqual(lines[0]["detections"][0]["track_id"], 7)
        self.assertEqual(lines[5]["time_ms"], 500.0)
        meta = read_meta(self.out)
        self.assertEqual(meta["status"], "complete")
        self.assertEqual(meta["format"], "tracks")
        self.assertEqual(meta["frames_written"], FRAMES)
        self.assertEqual(meta["video"], "flight.avi")
        self.assertEqual([run["start_frame"] for run in meta["runs"]], [0])
        # The boxes are the render: no video is drawn or encoded beside them.
        self.assertEqual(
            sorted(path.name for path in self.out.iterdir()),
            sorted([META_FILE, TRACKS_FILE, TRACKER_FILE]),
        )
        state = render_state(self.out, meta)
        self.assertTrue(state["complete"])
        self.assertEqual(state["processed_ms"], 1200.0)
        self.assertEqual(self.messages[0]["kind"], "started")
        self.assertEqual(self.messages[-1]["kind"], "complete")
        # Where the time went, so a slow render says which stage to fix.
        self.assertEqual(set(meta["stages_ms"]), {"decode", "track"})

    def test_the_tracker_is_this_project_s_settings_with_reid_on(self) -> None:
        # Ultralytics' defaults track by motion alone, so a vehicle that goes
        # behind a tree comes back as a new number. The file shipped here turns
        # appearance matching on, which is the whole reason it exists.
        detector = FakeDetector()
        self.render(detector=detector)
        settings = Path(detector.tracker)
        self.assertTrue(settings.is_file(), "the render asked for a file that is not there")
        # Written into the render's own folder: what it was made with stays with it.
        self.assertEqual(settings.name, TRACKER_FILE)
        self.assertEqual(settings.parent, self.out)
        values = yaml.safe_load(settings.read_text(encoding="utf-8"))
        self.assertIs(values["with_reid"], True)
        self.assertEqual(values["model"], "auto")
        # The trusted thresholds follow the confidence asked for: the first stage
        # takes what is drawn, a new number needs a margin more.
        self.assertAlmostEqual(values["track_high_thresh"], 0.45)
        self.assertAlmostEqual(values["new_track_thresh"], 0.6)
        self.assertAlmostEqual(values["track_low_thresh"], 0.1)

    def test_the_tracker_thresholds_follow_the_confidence(self) -> None:
        for confidence, new_track in ((0.3, 0.45), (0.9, 0.95)):
            with self.subTest(confidence=confidence):
                path = Path(tracker_config(self.dir, confidence))
                values = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertAlmostEqual(values["track_high_thresh"], confidence)
                self.assertAlmostEqual(values["new_track_thresh"], new_track)
        self.assertTrue(tracker_config().endswith("botsort.yaml"), "the shared file itself")

    def test_a_stride_keeps_every_nth_frame(self) -> None:
        self.render(RenderSettings(stride=2))
        self.assertEqual([line["frame_index"] for line in self.lines()], list(range(0, FRAMES, 2)))
        meta = read_meta(self.out)
        self.assertEqual(meta["source_frames"], FRAMES)
        self.assertEqual(render_state(self.out, meta)["processed_ms"], 1200.0)

    def test_cancelling_keeps_what_was_tracked(self) -> None:
        self.render(cancelled=_cancel_after(5))
        self.assertEqual(self.messages[-1], {"kind": "cancelled", "kept": True})
        meta = read_meta(self.out)
        self.assertEqual(meta["status"], "partial")
        self.assertEqual(meta["next_frame"], 5)
        self.assertEqual(render_state(self.out, meta)["processed_ms"], 500.0)
        self.assertEqual(len(self.lines()), 5)

    def test_at_a_stride_the_boxes_cover_the_frames_until_the_next(self) -> None:
        self.render(RenderSettings(stride=3), cancelled=_cancel_after(7))
        # Frames 0, 3 and 6 tracked; 6 covers 6, 7 and 8.
        self.assertEqual(render_state(self.out, read_meta(self.out))["processed_ms"], 900.0)

    def test_a_partial_render_continues_where_it_stopped(self) -> None:
        self.render(cancelled=_cancel_after(5))
        second = FakeDetector()
        self.render(detector=second)
        meta = read_meta(self.out)
        self.assertEqual(meta["status"], "complete")
        self.assertEqual([run["start_frame"] for run in meta["runs"]], [0, 5])
        lines = self.lines()
        self.assertEqual([line["frame_index"] for line in lines], list(range(FRAMES)))
        # The tracker restarts at 7 in the new run; its ids move past those used.
        self.assertEqual({line["detections"][0]["track_id"] for line in lines[:5]}, {7})
        self.assertEqual({line["detections"][0]["track_id"] for line in lines[5:]}, {14})
        # It really read on from frame 5 (brightness 100), not from the start.
        self.assertAlmostEqual(second.brightness[0], 5 * 20, delta=6)
        self.assertEqual(second.calls, FRAMES - 5)
        self.assertEqual(meta["frames_written"], FRAMES)

    def test_a_killed_render_continues_after_its_last_whole_line(self) -> None:
        # What a killed child leaves: a meta still saying "running" from when it
        # started, and tracks up to the line it was writing.
        self.render(cancelled=_cancel_after(6))
        meta = read_meta(self.out)
        meta.update(status="running", next_frame=0)
        (self.out / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
        with (self.out / TRACKS_FILE).open("a", encoding="utf-8") as tracks:
            tracks.write('{"frame_index": 6, "time_ms": 600.0, "detec')
        self.render()
        self.assertEqual([line["frame_index"] for line in self.lines()], list(range(FRAMES)))
        self.assertEqual(read_meta(self.out)["runs"][1]["start_frame"], 6)

    def test_a_render_made_with_video_segments_continues_as_tracks(self) -> None:
        # Renders from before the page drew the boxes: format "hls", run folders.
        self.render(cancelled=_cancel_after(4))
        meta = read_meta(self.out)
        meta.update(format="hls", runs=[{"dir": "run-000", "start_frame": 0}])
        (self.out / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
        (self.out / "run-000").mkdir()
        (self.out / "run-000" / "seg_00000.m4s").write_bytes(b"old")
        self.render()
        meta = read_meta(self.out)
        self.assertEqual(meta["format"], "tracks")
        self.assertEqual([line["frame_index"] for line in self.lines()], list(range(FRAMES)))
        # Its old video is left where it is, not cleared by continuing it.
        self.assertTrue((self.out / "run-000" / "seg_00000.m4s").is_file())

    def test_other_files_or_settings_start_over(self) -> None:
        self.render(cancelled=_cancel_after(5), meta={"render_id": "x", "video_fingerprint": 1})
        self.render(meta={"render_id": "x", "video_fingerprint": 2})
        self.assertEqual([run["start_frame"] for run in read_meta(self.out)["runs"]], [0])
        self.assertEqual(len(self.lines()), FRAMES)

        self.render(cancelled=_cancel_after(5), fresh=True)
        self.render(RenderSettings(confidence=0.3))
        self.assertEqual(read_meta(self.out)["runs"][0]["start_frame"], 0)
        self.assertEqual(len(self.lines()), FRAMES)

    def test_fresh_starts_over_even_when_it_could_continue(self) -> None:
        self.render(cancelled=_cancel_after(5))
        self.render(fresh=True)
        meta = read_meta(self.out)
        self.assertEqual([run["start_frame"] for run in meta["runs"]], [0])
        self.assertEqual(len(self.lines()), FRAMES)

    def test_a_render_that_finished_nothing_leaves_no_folder(self) -> None:
        self.render(cancelled=lambda: True)
        self.assertEqual(self.messages[-1], {"kind": "cancelled", "kept": False})
        self.assertFalse(self.out.exists())

    def test_an_unreadable_video_is_an_error(self) -> None:
        broken = self.dir / "broken.avi"
        broken.write_bytes(b"not a video")
        with self.assertRaises(RuntimeError):
            self.render(video=broken)
        self.assertFalse(self.out.exists())

    def test_a_render_that_fails_midway_keeps_what_it_finished(self) -> None:
        class FailingDetector(FakeDetector):
            def track(self, frame, **kwargs):
                if self.calls == 3:
                    raise RuntimeError("CUDA out of memory")
                return super().track(frame, **kwargs)

        with self.assertRaises(RuntimeError):
            self.render(detector=FailingDetector())
        meta = read_meta(self.out)
        self.assertEqual(meta["status"], "partial")
        self.assertEqual(meta["next_frame"], 3)


class TrackIndexTests(unittest.TestCase):
    def test_a_growing_file_is_indexed_as_it_grows(self) -> None:
        def line(index: int, track_id: int) -> str:
            record = {
                "frame_index": index,
                "time_ms": index * 100.0,
                "detections": [{"track_id": track_id}],
            }
            return json.dumps(record) + "\n"

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / TRACKS_FILE
            path.write_text(line(0, 1) + line(1, 2), encoding="utf-8")
            index = render_job.TrackIndex(path)
            index.refresh()
            self.assertEqual(index.at(1)[0]["track_id"], 2)
            self.assertIsNone(index.at(2))

            with path.open("a", encoding="utf-8") as stream:
                stream.write(line(2, 3) + '{"frame_index": 3, "ti')
            index.refresh()
            self.assertEqual(index.at(2)[0]["track_id"], 3)
            self.assertIsNone(index.at(3), "a line still being written was read")

            replacement = Path(temp) / "replacement.jsonl"
            replacement.write_text(line(0, 9), encoding="utf-8")
            os.replace(replacement, path)
            index.refresh()
            self.assertEqual(index.at(0)[0]["track_id"], 9)
            self.assertIsNone(index.at(1))

    def test_a_range_of_frames_comes_back_compact_and_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / TRACKS_FILE
            boxes = {
                "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0,
                "confidence": 0.9, "class_id": 1, "label": "truck", "track_id": 5,
            }
            path.write_text(
                "".join(
                    json.dumps({"frame_index": frame, "time_ms": frame * 100.0,
                                "detections": [boxes] if frame == 4 else []}) + "\n"
                    for frame in range(0, 10, 2)
                ),
                encoding="utf-8",
            )
            index = TrackIndex(path)
            index.refresh()
            self.assertEqual(
                index.between(3, 6), [[4, [[1.0, 2.0, 3.0, 4.0, 0.9, 1, 5, "truck"]]], [6, []]]
            )
            self.assertEqual([row[0] for row in index.between(7)], [8])
            self.assertEqual(index.between(20, 30), [])
            self.assertEqual(last_frame(path), 8)

    def test_the_last_frame_is_the_last_whole_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / TRACKS_FILE
            self.assertIsNone(last_frame(path), "no file")
            path.write_text('{"frame_index": 0, "de', encoding="utf-8")
            self.assertIsNone(last_frame(path), "only a line being written")
            # Long lines, so the answer is not in the last block read.
            long_line = json.dumps({"frame_index": 3, "detections": ["x" * 90_000]}) + "\n"
            path.write_text(
                json.dumps({"frame_index": 2}) + "\n" + long_line + '{"frame_index": 4, "d',
                encoding="utf-8",
            )
            self.assertEqual(last_frame(path), 3)


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
        self.entered = threading.Event()
        self.manager = RenderManager(
            self.dir / "renders",
            launcher=thread_launcher(self.gate, self.entered),
            ffmpeg="ffmpeg-for-tests",
        )
        self.addCleanup(self.gate.set)

    def start(self, settings: RenderSettings | None = None, manager=None) -> dict:
        settings = settings or RenderSettings()
        return (manager or self.manager).start(
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
        self.assertEqual(listed[0]["status"], "complete")
        self.assertEqual(listed[0]["processed_ms"], 1200.0)
        self.assertTrue(is_current(listed[0], self.video, self.model))

    def test_only_one_render_runs_at_a_time(self) -> None:
        self.start()
        with self.assertRaises(RenderBusy):
            self.start(RenderSettings(stride=2))
        self.gate.set()
        self.wait_for("complete")

    def test_a_cancelled_render_is_kept_and_continues(self) -> None:
        render_id = self.start()["render_id"]
        self.assertTrue(self.entered.wait(5))
        self.assertTrue(self.manager.cancel())
        self.gate.set()
        job = self.wait_for("cancelled")
        self.assertTrue(job["kept"])
        listed = self.manager.renders()
        self.assertEqual(
            [(item["render_id"], item["status"]) for item in listed], [(render_id, "partial")]
        )
        self.assertFalse(self.manager.cancel())

        self.start()
        done = self.wait_for("complete")
        self.assertGreater(done["start_frame"], 0)
        text = (self.manager.directory(render_id) / TRACKS_FILE).read_text(encoding="utf-8")
        frames = [json.loads(line)["frame_index"] for line in text.splitlines()]
        self.assertEqual(frames, list(range(FRAMES)))
        self.assertEqual(self.manager.renders()[0]["status"], "complete")

    def test_a_render_process_that_died_keeps_what_it_tracked(self) -> None:
        settings = RenderSettings()
        render_id = render_id_for("flight.avi", "best50", settings)

        def launch_that_dies(tracked: int):
            def launch(job):
                # What a killed process leaves: a meta still saying "running" and
                # the frames it had written.
                output = Path(job["output_dir"])
                output.mkdir(parents=True)
                with (output / TRACKS_FILE).open("w", encoding="utf-8") as tracks:
                    for frame in range(tracked):
                        tracks.write(json.dumps({"frame_index": frame, "detections": []}) + "\n")
                meta = {
                    **job["meta"],
                    "format": "tracks",
                    "status": "running",
                    "settings": job["settings"],
                    "source_fps": FPS,
                    "stride": 1,
                    "runs": [{"start_frame": 0, "track_id_offset": 0}],
                }
                (output / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
                thread = threading.Thread(target=lambda: None)
                thread.start()
                thread.join()
                handle = ThreadHandle(thread)
                handle.exitcode = 1
                return handle, queue.Queue(), threading.Event()

            return RenderManager(self.dir / "renders", launcher=launch, ffmpeg="ffmpeg-for-tests")

        manager = launch_that_dies(tracked=6)
        self.start(manager=manager)
        job = manager.status()
        self.assertEqual(job["status"], "error")
        self.assertTrue(job["kept"])
        self.assertEqual(read_meta(manager.directory(render_id))["next_frame"], 6)
        self.assertEqual(manager.renders()[0]["status"], "partial")
        manager.delete(render_id)

        manager = launch_that_dies(tracked=0)
        self.start(manager=manager)
        self.assertEqual(manager.status()["status"], "error")
        self.assertFalse(manager.directory(render_id).exists())

    def test_settling_never_touches_a_finished_render(self) -> None:
        self.gate.set()
        render_id = self.start()["render_id"]
        self.wait_for("complete")
        directory = self.manager.directory(render_id)
        self.assertEqual(settle_render(directory), "complete")
        self.assertEqual(len(self.manager.renders()), 1)

    def test_the_tracks_of_a_window_come_with_how_far_the_render_got(self) -> None:
        self.gate.set()
        render_id = self.start(RenderSettings(stride=2))["render_id"]
        self.wait_for("complete")
        answer = self.manager.tracks(render_id, 300.0, 700.0)
        self.assertTrue(answer["complete"])
        self.assertEqual(answer["processed_ms"], 1200.0)
        self.assertEqual(answer["stride"], 2)
        self.assertEqual(answer["source_fps"], FPS)
        # Frames 3 to 8 asked for; frames 4, 6 and 8 were tracked.
        self.assertEqual([row[0] for row in answer["frames"]], [4, 6, 8])
        *_, class_id, track_id, label = answer["frames"][0][1][0]
        self.assertEqual((class_id, track_id, label), (0, 7, "car"))
        self.assertIsNone(self.manager.tracks("never-rendered"))

    def test_the_tracks_say_how_fast_a_running_render_goes(self) -> None:
        # Held at its first frame: running, with nothing tracked yet.
        render_id = self.start(RenderSettings(stride=2))["render_id"]
        self.wait_for("running")
        deadline = time.monotonic() + 10
        while self.manager.tracks(render_id) is None and time.monotonic() < deadline:
            time.sleep(0.02)
        with self.manager._lock:
            self.manager._job["fps"] = 4.0  # frames tracked a second, as reported
        # Four frames a second at every second frame of a 10 fps source: 0.8 s of
        # video for each second of waiting.
        answer = self.manager.tracks(render_id)
        self.assertAlmostEqual(answer["rate"], 0.8)
        self.assertEqual(answer["status"], "running")
        self.gate.set()
        self.wait_for("complete")
        self.assertIsNone(self.manager.tracks(render_id)["rate"], "nothing left to wait for")

    def test_a_render_made_with_video_segments_still_lists_and_plays(self) -> None:
        directory = write_fake_render(
            self.dir / "renders",
            "old",
            video_id="flight.avi",
            video_path=self.video,
            model_path=self.model,
            segmented=True,
        )
        listed = self.manager.renders("flight.avi")
        self.assertEqual([(item["render_id"], item["format"]) for item in listed], [("old", "hls")])
        # Its folder, video included: what deleting it gives back.
        self.assertGreater(listed[0]["size_bytes"], (directory / TRACKS_FILE).stat().st_size)
        self.assertEqual(len(self.manager.tracks("old")["frames"]), 20)

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

    def test_a_render_needs_no_ffmpeg(self) -> None:
        # Only saving an MP4 encodes; tracking writes boxes.
        manager = RenderManager(self.dir / "renders", launcher=thread_launcher(), ffmpeg="")
        manager.start(
            render_id="x",
            video_id="flight.avi",
            video_path=self.video,
            model_name="best50",
            model_label="best.pt",
            model_path=self.model,
            settings=RenderSettings(),
        )
        deadline = time.monotonic() + 10
        while manager.status()["status"] != "complete" and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(manager.status()["status"], "complete")

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


class EncoderChoiceTests(unittest.TestCase):
    def test_the_first_encoder_that_works_is_chosen(self) -> None:
        from unittest import mock

        tried: list[str] = []

        def works(ffmpeg, args):
            tried.append(args[1])
            return args[1] == "h264_qsv"

        with mock.patch.object(render_job, "_encoder_works", works):
            name, args = render_job.choose_encoder("ffmpeg")
        self.assertEqual(name, "h264_qsv")
        self.assertEqual(tried, ["h264_nvenc", "h264_qsv"])
        self.assertIn("h264_qsv", args)

    def test_software_encoding_is_the_last_resort(self) -> None:
        from unittest import mock

        with mock.patch.object(render_job, "_encoder_works", lambda ffmpeg, args: False):
            self.assertEqual(render_job.choose_encoder("ffmpeg")[0], "libx264")

    def test_the_environment_can_name_the_encoder(self) -> None:
        from unittest import mock

        with (
            mock.patch.dict(os.environ, {"TELEMETRY_OCR_RENDER_ENCODER": "libx264"}),
            mock.patch.object(render_job, "_encoder_works", lambda ffmpeg, args: True),
        ):
            self.assertEqual(render_job.choose_encoder("ffmpeg")[0], "libx264")


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
