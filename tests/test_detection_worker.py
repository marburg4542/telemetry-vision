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
