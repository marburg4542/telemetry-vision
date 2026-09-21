from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_ocr.auto_capture import AutoLineRun, AutoRules, Selector

FPS = 10.0


def box(track_id, label="motorcycle", confidence=0.8, x=100.0, size=20.0):
    return {
        "x1": x, "y1": 50.0, "x2": x + size, "y2": 50.0 + size,
        "confidence": confidence, "class_id": 1, "label": label, "track_id": track_id,
    }


def feed(selector, frames):
    """frames: {frame_index: [boxes]}; returns every candidate, in order."""
    found = []
    for frame in sorted(frames):
        found.extend(selector.feed(frame, frame * 1000.0 / FPS, frames[frame]))
    return found


RULES = AutoRules(classes={"motorcycle": 0.5, "car": 0.7})


class SelectorTests(unittest.TestCase):
    def test_a_vehicle_is_reported_once_after_a_second_in_view(self) -> None:
        # Seen from 0 to 1.5 s: reported at 1 s, and only then.
        frames = {n: [box(3, confidence=0.6 + n / 100)] for n in range(0, 16)}
        selector = Selector(RULES)
        early = feed(selector, {n: frames[n] for n in range(0, 10)})
        self.assertEqual(early, [], "reported before a second had passed")
        found = feed(selector, {n: frames[n] for n in range(10, 16)})
        self.assertEqual(len(found), 1)
        # The frame of that second the model was surest about.
        self.assertEqual(found[0].frame_index, 10)
        self.assertEqual(found[0].note, "motorcycle #3 0.70")

    def test_each_class_has_its_own_confidence(self) -> None:
        frames = {n: [box(1, "car", 0.65), box(2, "motorcycle", 0.55, x=400)] for n in range(12)}
        found = feed(Selector(RULES), frames)
        self.assertEqual([item.label for item in found], ["motorcycle"], "a 0.65 car was reported")

    def test_classes_not_asked_for_are_ignored(self) -> None:
        frames = {n: [box(1, "truck", 0.99)] for n in range(20)}
        self.assertEqual(feed(Selector(RULES), frames), [])

    def test_a_flicker_is_not_a_vehicle(self) -> None:
        # In view for a second, but its confident sightings are 0.8 s apart:
        # each run starts over and none lasts a second.
        frames = {n: [box(4, confidence=0.9 if n % 8 == 0 else 0.3)] for n in range(0, 25)}
        self.assertEqual(feed(Selector(RULES), frames), [])

    def test_a_frame_or_two_missed_does_not_restart_the_second(self) -> None:
        frames = {n: [box(5)] for n in range(0, 12) if n not in (4, 5, 6)}
        self.assertEqual(len(feed(Selector(RULES), frames)), 1)

    def test_a_new_number_where_a_reported_one_vanished_is_the_same_vehicle(self) -> None:
        selector = Selector(RULES)
        first = {n: [box(7, x=100 + n)] for n in range(0, 12)}
        # Gone for 1 s, then back a few pixels on under a new number.
        again = {n: [box(9, x=100 + n)] for n in range(22, 36)}
        found = feed(selector, {**first, **again})
        self.assertEqual([item.track_id for item in found], [7])
        self.assertEqual(selector.same_vehicle, 1)

    def test_far_away_or_long_after_or_at_the_same_time_is_another_vehicle(self) -> None:
        cases = {
            "far away": {n: [box(9, x=900)] for n in range(22, 36)},
            "long after": {n: [box(9, x=112)] for n in range(60, 74)},
        }
        first = {n: [box(7, x=100 + n)] for n in range(0, 12)}
        for name, second in cases.items():
            with self.subTest(case=name):
                found = feed(Selector(RULES), {**first, **second})
                self.assertEqual([item.track_id for item in found], [7, 9])
        # Two numbers in view together are two vehicles, however close.
        together = {n: [box(7, x=100), box(8, x=110)] for n in range(0, 12)}
        self.assertEqual(len(feed(Selector(RULES), together)), 2)

    def test_the_same_vehicle_under_a_third_number_is_still_the_same(self) -> None:
        selector = Selector(RULES)
        frames = {n: [box(1, x=100)] for n in range(0, 12)}
        frames.update({n: [box(2, x=105)] for n in range(20, 32)})
        frames.update({n: [box(3, x=110)] for n in range(40, 52)})
        self.assertEqual([item.track_id for item in feed(selector, frames)], [1])
        self.assertEqual(selector.same_vehicle, 2)

    def test_rules_from_the_page_are_checked(self) -> None:
        rules = AutoRules.from_request({"classes": {"motorcycle": "0.5"}, "send": True})
        self.assertEqual(rules.classes, {"motorcycle": 0.5})
        self.assertTrue(rules.send)
        self.assertEqual(rules.reserve, 10)
        self.assertFalse(AutoRules.from_request({"classes": {"car": 0.7}}).send, "sends by default")
        for bad in (
            None,
            {},
            {"classes": {}},
            {"classes": {"car": 2}},
            {"classes": {"car": "high"}},
            {"classes": {"": 0.5}},
            {"classes": {"car": 0.5}, "reserve": -1},
        ):
            with self.subTest(body=bad), self.assertRaises(ValueError):
                AutoRules.from_request(bad)


class AutoLineRunTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.path = Path(self._temp.name) / "tracks.jsonl"
        self.path.write_text("", encoding="utf-8")
        self.active = True
        self.captured: list = []
        self.sent: list = []
        self.send_outcomes: list[tuple[str, str]] = []

    def write(self, frames) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            for frame in sorted(frames):
                record = {
                    "frame_index": frame, "time_ms": frame * 100.0, "detections": frames[frame],
                }
                stream.write(json.dumps(record) + "\n")

    def run_for(self, rules: AutoRules) -> AutoLineRun:
        class Shot:
            def __init__(self, frame_index):
                self.frame_index = frame_index

        async def capture(candidate):
            self.captured.append(candidate)
            return Shot(len(self.captured) - 1)

        async def send(shot, candidate):
            self.sent.append(shot.frame_index)
            return self.send_outcomes.pop(0) if self.send_outcomes else ("sent", "")

        return AutoLineRun(
            "flight", self.path, rules, session="s", capture=capture, send=send,
            is_active=lambda: self.active, poll_s=0.01,
        )

    async def test_it_follows_the_file_as_it_grows_and_stops_with_the_render(self) -> None:
        run = self.run_for(AutoRules(classes={"motorcycle": 0.5}, send=True))
        task = asyncio.create_task(run.run())
        self.write({n: [box(1)] for n in range(0, 8)})
        await asyncio.sleep(0.05)
        self.assertEqual(self.captured, [], "reported before a second in view")
        # A line still being written is not read until it is whole.
        self.write({n: [box(1)] for n in range(8, 12)})
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write('{"frame_index": 12, "time_ms": 1200.0, "detec')
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.captured), 1)
        self.assertEqual(self.sent, [0])
        self.active = False
        await asyncio.wait_for(task, 2)
        status = run.status()
        self.assertFalse(status["active"])
        self.assertEqual((status["captured"], status["sent"]), (1, 1))
        self.assertEqual(status["events"][0]["outcome"], "sent")

    async def test_capture_only_sends_nothing(self) -> None:
        run = self.run_for(AutoRules(classes={"motorcycle": 0.5}, send=False))
        self.write({n: [box(1)] for n in range(0, 12)})
        self.active = False
        await run.run()
        self.assertEqual(len(self.captured), 1)
        self.assertEqual(self.sent, [])
        self.assertEqual(run.status()["events"][0]["outcome"], "captured")

    async def test_the_quota_reserve_stops_sending_but_not_capturing(self) -> None:
        run = self.run_for(AutoRules(classes={"motorcycle": 0.5}, send=True))
        self.send_outcomes = [("skipped", "quota: 10 left, kept for sending by hand")]
        frames = {n: [box(1, x=100)] for n in range(0, 12)}
        frames.update({n: [box(2, x=900)] for n in range(20, 32)})
        self.write(frames)
        self.active = False
        await run.run()
        self.assertEqual(len(self.captured), 2)
        self.assertEqual(self.sent, [0], "sent again after the reserve was reached")
        status = run.status()
        self.assertFalse(status["sending"])
        self.assertIn("quota", status["stopped_sending"])

    async def test_a_capture_that_fails_is_reported_and_the_run_goes_on(self) -> None:
        run = self.run_for(AutoRules(classes={"motorcycle": 0.5}))

        async def broken(candidate):
            raise RuntimeError("the video is gone")

        run._capture = broken
        frames = {n: [box(1, x=100)] for n in range(0, 12)}
        self.write(frames)
        self.active = False
        await run.run()
        status = run.status()
        self.assertEqual((status["captured"], status["failed"]), (0, 1))
        self.assertIn("the video is gone", status["events"][0]["detail"])


if __name__ == "__main__":
    unittest.main()
