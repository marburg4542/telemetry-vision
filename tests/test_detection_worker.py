import os
import queue
import tempfile
import time
import unittest
from pathlib import Path

from telemetry_ocr.detection_worker import (
    DetectionFrame,
    ProcessDetectionWorker,
    drain_queue,
    put_latest,
)


def _fake_worker(model_path, filename, tracker, frames, results):
    del model_path, filename, tracker
    results.put(
        {
            "kind": "ready",
            "metadata": {
                "classes": ["vehicle"],
                "architecture": "fake",
                "epochs": 1,
                "trained_image_size": 640,
            },
        }
    )
    while True:
        packet = frames.get()
        if packet.get("kind") == "shutdown":
            return
        results.put(
            {
                "kind": "result",
                "result": {
                    "source_id": packet["source_id"],
                    "generation": packet["generation"],
                    "sequence": packet["sequence"],
                    "source_time_ms": packet["source_time_ms"],
                    "width": 640,
                    "height": 360,
                    "inference_ms": 5.0,
                    "decode_ms": 1.0,
                    "track_ms": 4.0,
                    "worker_total_ms": 6.0,
                    # Echoed back exactly as received; the parent's clock has no
                    # meaning in this process.
                    "enqueued_at": float(packet.get("enqueued_at", 0.0)),
                    "completed_at": time.time(),
                    "detections": [],
                },
            }
        )


def _crashing_worker(model_path, filename, tracker, frames, results):
    del model_path, filename, tracker, frames
    results.put({"kind": "ready", "metadata": {"classes": []}})
    time.sleep(0.15)
    os._exit(7)


class DetectionWorkerQueueTests(unittest.TestCase):
    def test_put_latest_replaces_an_older_queued_frame(self):
        channel = queue.Queue(maxsize=1)
        self.assertEqual(put_latest(channel, {"sequence": 1}), 0)
        self.assertEqual(put_latest(channel, {"sequence": 2}), 1)
        self.assertEqual(channel.qsize(), 1)
        self.assertEqual(channel.get_nowait()["sequence"], 2)

    def test_drain_queue_removes_every_pending_item(self):
        channel = queue.Queue()
        for sequence in range(3):
            channel.put_nowait(sequence)
        self.assertEqual(drain_queue(channel), 3)
        self.assertTrue(channel.empty())

    def test_process_worker_lifecycle_submit_reset_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "best.pt"
            model_path.write_bytes(b"fake")
            worker = ProcessDetectionWorker(worker_target=_fake_worker)
            try:
                worker.start(model_path, "best.pt")
                self.assertEqual(_wait_for_status(worker, "ready")["worker_pid"] > 0, True)
                worker.reset("video-a", 3)
                accepted = worker.submit(
                    DetectionFrame(
                        payload=b"jpeg",
                        source_id="video-a",
                        generation=3,
                        sequence=9,
                        source_time_ms=900.0,
                        confidence=0.45,
                        iou=0.45,
                        image_size=640,
                    )
                )
                self.assertTrue(accepted)
                latest = _wait_for_result(worker)
                self.assertEqual(latest["sequence"], 9)
                self.assertEqual(latest["generation"], 3)

                worker.stop()
                self.assertEqual(worker.status()["status"], "stopped")
                worker.restart()
                self.assertEqual(_wait_for_status(worker, "ready")["status"], "ready")
            finally:
                worker.stop()

    def test_drain_publishes_a_result_without_any_other_call(self):
        """The background pump relies on drain() alone doing the collection.

        Before it existed, a finished result waited in the IPC queue until some
        HTTP request happened to call status(), latest() or submit() — which is
        the delivery delay the pump removes. This guards that premise: nothing
        here calls status() or latest() until after drain() has run.
        """
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "best.pt"
            model_path.write_bytes(b"fake")
            worker = ProcessDetectionWorker(worker_target=_fake_worker)
            try:
                worker.start(model_path, "best.pt")
                _wait_for_status(worker, "ready")
                worker.reset("video-a", 1)
                worker.submit(
                    DetectionFrame(
                        payload=b"jpeg",
                        source_id="video-a",
                        generation=1,
                        sequence=4,
                        source_time_ms=400.0,
                        confidence=0.45,
                        iou=0.45,
                        image_size=640,
                    )
                )
                deadline = time.monotonic() + 10.0
                collected = None
                while time.monotonic() < deadline:
                    worker.drain()
                    collected = worker._latest_result
                    if collected is not None:
                        break
                    time.sleep(0.02)
                self.assertIsNotNone(collected, "drain() never collected the result")
                self.assertEqual(collected["sequence"], 4)
                # The parent measures the round trip against its own clock.
                self.assertGreaterEqual(collected["pipeline_ms"], 0.0)
                self.assertIn("available_at", collected)
            finally:
                worker.stop()

    def test_stopping_clears_everything_that_described_the_old_model(self):
        """A cleared model must not keep answering for itself.

        stop() used to leave _latest_result and _metadata in place, so after a
        DELETE the status payload still advertised the removed model's classes
        and device and /api/detection/result still served its last detection.
        """
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "best.pt"
            model_path.write_bytes(b"fake")
            worker = ProcessDetectionWorker(worker_target=_fake_worker)
            try:
                worker.start(model_path, "best.pt")
                _wait_for_status(worker, "ready")
                worker.reset("video-a", 1)
                worker.submit(
                    DetectionFrame(
                        payload=b"jpeg",
                        source_id="video-a",
                        generation=1,
                        sequence=1,
                        source_time_ms=100.0,
                        confidence=0.45,
                        iou=0.45,
                        image_size=640,
                    )
                )
                _wait_for_result(worker)
                self.assertIsNotNone(worker.latest())

                worker.stop()
                self.assertIsNone(worker.latest())
                status = worker.status()
                self.assertEqual(status["status"], "stopped")
                self.assertEqual(status["dropped_frames"], 0)
                self.assertNotIn("classes", status)
            finally:
                worker.stop()

    def test_put_latest_gives_up_instead_of_spinning_forever(self):
        """stop() enqueues through this, so an unbounded spin hangs shutdown."""

        class NeverAccepts:
            def put_nowait(self, item):
                raise queue.Full

            def get_nowait(self):
                raise queue.Empty

        started = time.monotonic()
        dropped = put_latest(NeverAccepts(), {"kind": "shutdown"}, deadline_s=0.05)
        self.assertEqual(dropped, 0)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_process_worker_reports_an_unexpected_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "best.pt"
            model_path.write_bytes(b"fake")
            worker = ProcessDetectionWorker(worker_target=_crashing_worker)
            try:
                worker.start(model_path, "best.pt")
                _wait_for_status(worker, "ready")
                status = _wait_for_status(worker, "error")
                self.assertIn("code 7", status["error"])
            finally:
                worker.stop()


class DetectionBoundaryTests(unittest.TestCase):
    """The result queue must carry builtins only.

    On CUDA, Paddle and Torch each ship their own cuDNN and the second one
    loaded into a process fails. The worker keeps them apart by living in a
    spawned process, but unpickling a value in the parent imports whatever
    module defines its class. torch.__version__, for instance, is a str
    subclass, so returning it would drag Torch into the Paddle process and
    break OCR. Everything crossing the boundary is therefore coerced to a
    builtin, and this test is what keeps it that way.
    """

    ALLOWED = (bool, int, float, str, type(None))

    def assert_only_builtins(self, value, path="result"):
        if isinstance(value, dict):
            for key, item in value.items():
                self.assertIs(type(key), str, f"{path}: key {key!r} is {type(key).__name__}")
                self.assert_only_builtins(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self.assert_only_builtins(item, f"{path}[{index}]")
            return
        self.assertIn(type(value), self.ALLOWED, f"{path} is {type(value).__name__}")

    def test_detection_converts_to_builtins(self):
        from dataclasses import asdict

        from telemetry_ocr.detection import Detection

        self.assert_only_builtins(asdict(Detection(1.0, 2.0, 3.0, 4.0, 0.9, 0, "car", 7)))
        self.assert_only_builtins(asdict(Detection(1.0, 2.0, 3.0, 4.0, 0.9, 0, "car", None)))

    def test_worker_results_reach_the_parent_as_builtins(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "best.pt"
            model_path.write_bytes(b"fake")
            worker = ProcessDetectionWorker(worker_target=_fake_worker)
            try:
                worker.start(model_path, "best.pt")
                status = _wait_for_status(worker, "ready")
                self.assert_only_builtins(
                    {key: value for key, value in status.items() if key != "worker_pid"},
                    "status",
                )
                worker.reset("video-a", 1)
                worker.submit(
                    DetectionFrame(
                        payload=b"jpeg",
                        source_id="video-a",
                        generation=1,
                        sequence=0,
                        source_time_ms=0.0,
                        confidence=0.45,
                        iou=0.45,
                        image_size=640,
                    )
                )
                self.assert_only_builtins(_wait_for_result(worker))
            finally:
                worker.stop()


def _wait_for_status(worker, expected, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = worker.status()
        if status["status"] == expected:
            return status
        time.sleep(0.02)
    raise AssertionError(f"worker did not reach {expected}: {worker.status()}")


def _wait_for_result(worker, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.latest()
        if result is not None:
            return result
        time.sleep(0.02)
    raise AssertionError("worker did not return a result")


if __name__ == "__main__":
    unittest.main()
