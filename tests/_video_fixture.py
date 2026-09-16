"""Small real videos and renders for the web app and render tests.

Real files rather than mocks: the capture path now reads its frame out of the
video on the server, and a test that faked the reader would pass whether or not
the right frame came back.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, ClassVar

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
    status: str = "complete",
    processed: int | None = None,
    legacy: bool = False,
) -> Path:
    """A render on disk, as the render child would have left it.

    ``processed`` stops the recorded frames short of ``frames``, for a render
    that is partial or still being made. ``legacy`` writes the single-MP4 layout
    of renders made before segments.
    """
    from telemetry_ocr.render_job import (
        INIT_FILE,
        LEGACY_VIDEO_FILE,
        META_FILE,
        PLAYLIST_FILE,
        TRACKS_FILE,
        fingerprint,
    )

    directory = renders_root / render_id
    directory.mkdir(parents=True, exist_ok=True)
    done = frames if processed is None else processed
    recorded = range(0, done, stride)
    chosen = detections if detections is not None else {}
    with (directory / TRACKS_FILE).open("w", encoding="utf-8") as tracks:
        for index in recorded:
            record = {
                "frame_index": index,
                "time_ms": round(index * 1000.0 / fps, 3),
                "detections": chosen.get(index, []),
            }
            tracks.write(json.dumps(record) + "\n")
    meta: dict[str, Any] = {
        "render_id": render_id,
        "video": video_id,
        "video_fingerprint": fingerprint(video_path),
        "model": "best50",
        "model_label": "models/yolo/best.pt",
        "model_fingerprint": fingerprint(model_path),
        "status": status,
        "settings": {"confidence": 0.45, "iou": 0.45, "image_size": 640, "stride": stride},
        "source_fps": fps,
        "fps": fps / stride,
        "stride": stride,
        "source_frames": frames,
        "frames_written": len(recorded),
        "width": 160,
        "height": 120,
        "duration_ms": frames * 1000.0 / fps,
        "elapsed_s": 1.0,
        "created_at": "2026-09-15T12:00:00+07:00",
        "tracks_file": TRACKS_FILE,
    }
    if legacy:
        (directory / LEGACY_VIDEO_FILE).write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
        meta["video_file"] = LEGACY_VIDEO_FILE
    else:
        run = directory / "run-000"
        run.mkdir(exist_ok=True)
        (run / INIT_FILE).write_bytes(b"\x00\x00\x00\x18ftypiso5" + b"\x00" * 16)
        (run / "seg_00000.m4s").write_bytes(b"\x00\x00\x00\x10moof" + b"\x00" * 8)
        seconds = len(recorded) * stride / fps
        (run / PLAYLIST_FILE).write_text(
            f'#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-MAP:URI="{INIT_FILE}"\n'
            f"#EXTINF:{seconds:.6f},\nseg_00000.m4s\n",
            encoding="utf-8",
        )
        meta.update(
            format="hls",
            runs=[{"dir": "run-000", "start_frame": 0, "track_id_offset": 0}],
            next_frame=done,
        )
    (directory / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    return directory


class FakeDetector:
    """One car, tracked as id 7, drifting right a pixel a frame.

    ``entered`` is set as the first frame arrives, so a test can cancel a render
    that has certainly started; ``brightness`` records each frame it was given.
    """

    def __init__(
        self, gate: threading.Event | None = None, entered: threading.Event | None = None
    ) -> None:
        self.calls = 0
        self.gate = gate
        self.entered = entered
        self.brightness: list[float] = []

    def track(self, frame: Any, *, confidence: float, iou: float, image_size: int, tracker: str):
        from telemetry_ocr.detection import Detection

        if self.entered is not None:
            self.entered.set()
        if self.gate is not None:
            self.gate.wait(5)
        self.calls += 1
        self.brightness.append(float(frame.mean()))
        x = float(self.calls)
        return [Detection(x, 2.0, x + 10.0, 12.0, 0.9, 0, "car", 7)]


class FakeEncoder:
    """Leaves what ffmpeg's HLS muxer leaves, with placeholder bytes.

    An init segment, a numbered segment every ``segment_frames`` frames, and a
    playlist that names each one once it is finished. Closing finishes the last
    short segment; aborting does not, like an ffmpeg that was killed.
    """

    instances: ClassVar[list[FakeEncoder]] = []
    segment_frames = 4

    def __init__(self, run_dir: Path, width: int, height: int, fps: float) -> None:
        self.run_dir = Path(run_dir)
        self.width, self.height, self.fps = width, height, fps
        self.shapes: list[tuple[int, ...]] = []
        self.closed_with_abort: bool | None = None
        self._pending = 0
        self._segments: list[float] = []
        (self.run_dir / "init.mp4").write_bytes(b"\x00\x00\x00\x18ftypiso5" + b"\x00" * 16)
        FakeEncoder.instances.append(self)

    @property
    def frames(self) -> int:
        return len(self.shapes)

    def write(self, frame: Any) -> None:
        self.shapes.append(tuple(frame.shape))
        self._pending += 1
        if self._pending == self.segment_frames:
            self._finish_segment()

    def close(self, abort: bool = False) -> None:
        self.closed_with_abort = abort
        if not abort and self._pending:
            self._finish_segment()
        self._write_playlist(ended=not abort)

    def _finish_segment(self) -> None:
        name = f"seg_{len(self._segments):05d}.m4s"
        (self.run_dir / name).write_bytes(b"\x00\x00\x00\x10moof" + b"\x00" * 8)
        self._segments.append(self._pending / self.fps)
        self._pending = 0
        self._write_playlist(ended=False)

    def _write_playlist(self, ended: bool) -> None:
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:7",
            "#EXT-X-TARGETDURATION:2",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:EVENT",
            "#EXT-X-INDEPENDENT-SEGMENTS",
            '#EXT-X-MAP:URI="init.mp4"',
        ]
        for number, duration in enumerate(self._segments):
            lines += [f"#EXTINF:{duration:.6f},", f"seg_{number:05d}.m4s"]
        if ended:
            lines.append("#EXT-X-ENDLIST")
        (self.run_dir / "index.m3u8").write_text("\n".join(lines) + "\n", encoding="utf-8")


class ThreadHandle:
    def __init__(self, thread: threading.Thread) -> None:
        self.thread = thread
        self.exitcode = 0

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout)

    def terminate(self) -> None:
        pass


def thread_launcher(
    gate: threading.Event | None = None,
    entered: threading.Event | None = None,
    detector_factory: Any = None,
):
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
                    detector=(detector_factory or FakeDetector)(gate, entered),
                    encoder_factory=FakeEncoder,
                    publish=progress.put,
                    cancelled=cancel.is_set,
                    fresh=bool(job.get("fresh")),
                )
            except Exception as exc:
                progress.put({"kind": "error", "error": str(exc)})

        thread = threading.Thread(target=body, daemon=True)
        thread.start()
        return ThreadHandle(thread), progress, cancel

    return launch
