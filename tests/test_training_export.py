from __future__ import annotations

import csv
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2
except ImportError:  # pragma: no cover - the CPU test venv has it
    cv2 = None

from _video_fixture import write_fake_render, write_test_video
from telemetry_ocr.training_export import (
    Flight,
    class_names,
    export,
    load_feedback,
    load_flights,
    select,
)


def box(track_id, label="car", x=10.0, class_id=0):
    return {
        "x1": x, "y1": 10.0, "x2": x + 20.0, "y2": 30.0,
        "confidence": 0.8, "class_id": class_id, "label": label, "track_id": track_id,
    }


def flight(video="a.mp4", frames=300, fps=10.0, stride=1, boxes=None):
    boxes = boxes or {}
    return Flight(
        video=video, path=Path(video), render_dir=Path("r"), fps=fps, stride=stride,
        frames=list(range(0, frames, stride)),
        boxes={frame: boxes.get(frame, []) for frame in range(0, frames, stride)},
    )


class SelectionTests(unittest.TestCase):
    def test_flagged_frames_all_go_in_and_the_rest_is_shared_out(self) -> None:
        motorcycles = {frame: [box(5, "motorcycle", class_id=1)] for frame in range(100, 200)}
        flights = [flight(boxes=motorcycles)]
        feedback = [{"video": "a.mp4", "source_time_ms": 25_000, "kind": "wrong"}]
        picks = select(flights, feedback, count=10, holdout=set(), test_count=0)
        reasons = [pick.reason for pick in picks]
        self.assertEqual(reasons[0], "flagged")
        self.assertEqual(picks[0].frame_index, 250)
        self.assertEqual(picks[0].note, "wrong")
        self.assertEqual(len(picks), 10)
        self.assertEqual(reasons.count("motorcycle"), 3, "35% of the nine left")
        self.assertTrue(all(100 <= p.frame_index < 200 for p in picks if p.reason == "motorcycle"))

    def test_frames_of_a_video_are_kept_apart(self) -> None:
        picks = select([flight(frames=1000)], [], count=50, holdout=set(), test_count=0)
        frames = sorted(pick.frame_index for pick in picks)
        self.assertTrue(all(b - a >= 20 for a, b in itertools.pairwise(frames)))

    def test_a_held_out_flight_goes_to_test_whole(self) -> None:
        flights = [flight("a.mp4"), flight("b.mp4")]
        picks = select(flights, [], count=8, holdout={"b.mp4"}, test_count=4)
        self.assertEqual({pick.video for pick in picks if pick.split == "test"}, {"b.mp4"})
        self.assertEqual({pick.video for pick in picks if pick.split == "train"}, {"a.mp4"})
        self.assertEqual(sum(1 for pick in picks if pick.split == "test"), 4)

    def test_a_vehicle_given_a_second_number_is_a_frame_to_label(self) -> None:
        # Track 1 until frame 40; track 2 from frame 50 where 1 vanished.
        boxes = {frame: [box(1)] for frame in range(0, 41)}
        boxes.update({frame: [box(2, x=12.0)] for frame in range(50, 120)})
        picks = select([flight(boxes=boxes)], [], count=5, holdout=set(), test_count=0)
        self.assertIn(50, [pick.frame_index for pick in picks if pick.reason == "relabel"])

    def test_the_draft_between_processed_frames_is_placed_as_the_page_places_it(self) -> None:
        boxes = {0: [box(1, x=0.0)], 3: [box(1, x=30.0)]}
        f = flight(frames=6, stride=3, boxes=boxes)
        feedback = [{"video": "a.mp4", "source_time_ms": 100.0, "kind": "missed"}]
        [pick] = [p for p in select([f], feedback, count=1, holdout=set(), test_count=0)
                  if p.reason == "flagged"]
        self.assertEqual(pick.frame_index, 1)
        self.assertAlmostEqual(pick.boxes[0]["x1"], 10.0)


@unittest.skipIf(cv2 is None, "OpenCV is required")
class ExportTests(unittest.TestCase):
    def test_a_yolov8_dataset_roboflow_can_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = write_test_video(root / "videos" / "flight.avi", frames=40)
            model = root / "best.pt"
            model.write_bytes(b"x")
            write_fake_render(
                root / "renders", "flight__best50", video_id="flight.avi", video_path=video,
                model_path=model, frames=40,
                detections={
                    frame: [box(3, "motorcycle", x=40.0, class_id=1), box(4, x=100.0)]
                    for frame in range(40)
                },
            )
            sessions = root / "sessions" / "s1"
            sessions.mkdir(parents=True)
            # The second mark was taken back ("none").
            (sessions / "feedback.jsonl").write_text(
                "".join(
                    json.dumps(
                        {"shot": shot, "video": "flight.avi", "source_time_ms": ms, "kind": kind}
                    )
                    + "\n"
                    for shot, ms, kind in ((0, 500, "wrong"), (1, 900, "none"))
                ),
                encoding="utf-8",
            )
            flights = load_flights(root / "renders", {"flight.avi": video})
            feedback = load_feedback(root / "sessions")
            self.assertEqual(len(feedback), 1, "a mark taken back still counted")
            names = class_names(flights)
            self.assertEqual(names, ["car", "motorcycle"])
            picks = select(flights, feedback, count=3, holdout=set(), test_count=0, spacing_s=1.0)
            out = root / "dataset"
            summary = export(picks, flights, out, names)
            self.assertEqual(summary["frames"], 3)
            self.assertIn('names: ["car", "motorcycle"]', (out / "data.yaml").read_text("utf-8"))
            rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
            first = next(row for row in rows if row["reason"] == "flagged")
            image = out / first["file"]
            self.assertTrue(image.is_file())
            # The frame really is frame 5 (brightness 50), not a neighbour.
            self.assertAlmostEqual(float(cv2.imread(str(image)).mean()), 50, delta=3)
            label = (out / first["file"].replace("images", "labels").replace(".jpg", ".txt"))
            lines = [line.split() for line in label.read_text("utf-8").splitlines()]
            self.assertEqual(len(lines), 2, "one line per box")
            _, cx, _, w, _ = next(line for line in lines if line[0] == "1")
            self.assertAlmostEqual(float(cx), 50 / 160, places=4)
            self.assertAlmostEqual(float(w), 20 / 160, places=4)


if __name__ == "__main__":
    unittest.main()
