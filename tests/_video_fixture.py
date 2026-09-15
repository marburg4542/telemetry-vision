"""Small real videos and finished renders for the web app tests.

Real files rather than mocks: the capture path now reads its frame out of the
video on the server, and a test that faked the reader would pass whether or not
the right frame came back.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

FPS = 10.0


def write_test_video(
    path: Path, frames: int = 20, width: int = 160, height: int = 120, fps: float = FPS
) -> Path:
    """Each frame's index can be read back from its brightness (index * 10)."""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    for index in range(frames):
        writer.write(np.full((height, width, 3), min(255, index * 10), dtype=np.uint8))
    writer.release()
    return path


def car(track_id: int = 1, x: float = 10.0) -> dict[str, Any]:
    return {
        "x1": x,
        "y1": 20.0,
        "x2": x + 50.0,
        "y2": 70.0,
        "confidence": 0.83,
        "class_id": 0,
        "label": "car",
        "track_id": track_id,
    }


def write_fake_render(
    renders_root: Path,
    render_id: str,
    *,
    video_id: str,
    video_path: Path,
    model_path: Path,
    frames: int = 20,
    detections: dict[int, list[dict[str, Any]]] | None = None,
    stride: int = 1,
    fps: float = FPS,
) -> Path:
    """A finished render on disk, as the render child would have left it."""
    from telemetry_ocr.render_job import META_FILE, TRACKS_FILE, VIDEO_FILE, fingerprint

    directory = renders_root / render_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / VIDEO_FILE).write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    chosen = detections if detections is not None else {}
    with (directory / TRACKS_FILE).open("w", encoding="utf-8") as tracks:
        for index in range(0, frames, stride):
            record = {
                "frame_index": index,
                "time_ms": round(index * 1000.0 / fps, 3),
                "detections": chosen.get(index, []),
            }
            tracks.write(json.dumps(record) + "\n")
    meta = {
        "render_id": render_id,
        "video": video_id,
        "video_fingerprint": fingerprint(video_path),
        "model": "best50",
        "model_label": "models/yolo/best.pt",
        "model_fingerprint": fingerprint(model_path),
        "status": "complete",
        "settings": {"confidence": 0.45, "iou": 0.45, "image_size": 640, "stride": stride},
        "source_fps": fps,
        "fps": fps / stride,
        "stride": stride,
        "source_frames": frames,
        "frames_written": len(range(0, frames, stride)),
        "width": 160,
        "height": 120,
        "duration_ms": frames * 1000.0 / fps,
        "elapsed_s": 1.0,
        "created_at": "2026-09-15T12:00:00+07:00",
        "video_file": VIDEO_FILE,
        "tracks_file": TRACKS_FILE,
    }
    (directory / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    return directory


class FakeDetector:
    """One car, tracked as id 7, drifting right a pixel a frame."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.calls = 0
        self.gate = gate

    def track(self, frame: Any, *, confidence: float, iou: float, image_size: int, tracker: str):
        from telemetry_ocr.detection import Detection

        if self.gate is not None:
            self.gate.wait(5)
        self.calls += 1
        x = float(self.calls)
        return [Detection(x, 2.0, x + 10.0, 12.0, 0.9, 0, "car", 7)]


class FakeEncoder:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        self.frames = 0

    def write(self, frame: Any) -> None:
        self.frames += 1

    def close(self, abort: bool = False) -> None:
        if not abort:
            self.path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)


class _ThreadHandle:
    def __init__(self, thread: threading.Thread) -> None:
        self.thread = thread
        self.exitcode = 0

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout)

    def terminate(self) -> None:
        pass


def thread_launcher(gate: threading.Event | None = None):
    """Runs a render in a thread with a fake detector and encoder, for tests."""
    from telemetry_ocr.render_job import run_render

    def launch(job: dict[str, Any]):
        progress: queue.Queue = queue.Queue()
        cancel = threading.Event()

        def body() -> None:
            try:
                run_render(
                    video=job["video"],
                    output_dir=job["output_dir"],
                    settings=job["settings"],
                    meta=job["meta"],
                    detector=FakeDetector(gate),
                    encoder_factory=FakeEncoder,
                    publish=progress.put,
                    cancelled=cancel.is_set,
                )
            except Exception as exc:
                progress.put({"kind": "error", "error": str(exc)})

        thread = threading.Thread(target=body, daemon=True)
        thread.start()
        return _ThreadHandle(thread), progress, cancel

    return launch
