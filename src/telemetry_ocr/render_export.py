"""Turn a render into one MP4 file that can be sent to someone.

A render is the boxes the tracker found, one line per processed frame; the page
draws them over the original video as it plays. Nobody can be sent that, so an
export draws them into the video itself: every frame of the original is read,
the boxes of that moment are placed on it -- between two processed frames, where
the vehicle was on either side, exactly as the page places them -- and ffmpeg
encodes the result at the source's own frame rate. Either at the source's size,
or scaled to 720p and squeezed for a chat that will not take a gigabyte.

Drawing costs the export what it saves the render: a 25-minute flight takes
minutes to save rather than seconds. What it buys is renders that are nothing
but tracks, a tracker that has the machine to itself, and a saved file as smooth
as the original at every stride.

The server holds PaddleOCR, so like render_job this module must not import torch.
Exports run in a thread; one at a time, which is all a laptop GPU's encoder and
a single disk want anyway.
"""

from __future__ import annotations

import bisect
import logging
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .render_job import (
    TRACKS_FILE,
    RenderManager,
    TrackIndex,
    choose_encoder,
    read_meta,
    render_state,
    with_track_ids,
)

_LOG = logging.getLogger(__name__)

QUALITIES = ("full", "small")
SMALL_HEIGHT = 720
# An export nobody came back for is removed after this long.
STALE_AFTER_S = 3600.0
_EXPORT_ID = re.compile(r"[0-9a-f]{32}")
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")

# Size-for-a-chat settings per encoder: about 11 MB a minute at 720p with NVENC.
_SMALL_QUALITY: dict[str, tuple[str, ...]] = {
    "h264_nvenc": ("-cq", "30", "-maxrate", "3M", "-bufsize", "6M"),
    "h264_qsv": ("-global_quality", "30"),
    "libx264": ("-crf", "28"),
}
# The render encoder's own quality options, each followed by its value.
_RATE_CONTROL = frozenset({"-cq", "-crf", "-global_quality", "-maxrate", "-bufsize"})

VideoPath = Callable[[str], Path | None]
WriterFactory = Callable[[Path, int, int, float, str], Any]


class ExportBusy(RuntimeError):
    """Another export is still being made."""


class ExportUnavailable(RuntimeError):
    """Nothing to export, or no ffmpeg or source video to export it with."""


@dataclass
class ExportJob:
    id: str
    render_id: str
    quality: str
    filename: str
    directory: Path
    status: str = "running"  # running | ready | error
    percent: float = 0.0
    size_bytes: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def path(self) -> Path:
        return self.directory / self.filename

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "render_id": self.render_id,
            "quality": self.quality,
            "filename": self.filename,
            "status": self.status,
            "percent": round(self.percent, 1),
            "size_bytes": self.size_bytes,
            "error": self.error,
        }


def export_filename(meta: dict[str, Any], quality: str, percent: int | None) -> str:
    """The name the browser saves the file under.

    The video and the model, so two exports of the same flight say which checkpoint
    found the boxes, and "partial-40%" on one that was stopped partway, so nobody
    forwards the first ten minutes of a flight believing it is the whole of it.
    """
    stem = Path(str(meta.get("video") or "video")).stem
    parts = [_safe(stem), _safe(str(meta.get("model") or "model")), "AI"]
    name = "__".join(parts)
    if quality == "small":
        name += f"-{SMALL_HEIGHT}p"
    if percent is not None:
        name += f"_partial-{percent}%"
    return f"{name}.mp4"


# ---------------------------------------------------------------------------
# Where the boxes are on a frame the tracker did not look at. The page does the
# same in track-overlay.js; tests on both sides hold them to the same answers.
# ---------------------------------------------------------------------------


def boxes_at(
    frame: int, frames: list[int], boxes: dict[int, list[dict[str, Any]]], stride: int
) -> list[dict[str, Any]]:
    """The boxes to draw on one source frame.

    On a processed frame, its own boxes. Between two processed frames a stride
    apart, a vehicle numbered on both sides is placed along the line between
    them; one seen on only one side is shown for the half nearer to it, so a
    box neither lingers after its vehicle has gone nor appears before it has
    come. Past the last processed frame, or across a gap where frames were not
    processed, the last boxes are held for at most one stride.
    """
    at = bisect.bisect_right(frames, frame) - 1
    if at < 0:
        return []
    before = frames[at]
    if before == frame:
        return boxes[before]
    after = frames[at + 1] if at + 1 < len(frames) else None
    if after is None or after - before > stride:
        return boxes[before] if frame - before < stride else []
    t = (frame - before) / (after - before)
    later = {box["track_id"]: box for box in boxes[after] if box.get("track_id") is not None}
    placed: list[dict[str, Any]] = []
    matched: set[Any] = set()
    for box in boxes[before]:
        partner = later.get(box.get("track_id")) if box.get("track_id") is not None else None
        if partner is not None:
            matched.add(box["track_id"])
            placed.append(
                {
                    **(box if t < 0.5 else partner),
                    **{
                        key: box[key] + (partner[key] - box[key]) * t
                        for key in ("x1", "y1", "x2", "y2")
                    },
                }
            )
        elif t < 0.5:
            placed.append(box)
    if t >= 0.5:
        placed.extend(
            box
            for box in boxes[after]
            if box.get("track_id") is None or box["track_id"] not in matched
        )
    return placed


def encoder_args(quality: str, encoder: tuple[str, tuple[str, ...]]) -> list[str]:
    """ffmpeg's output options: the render encoder's for full, a chat's for small."""
    name, codec = encoder
    if quality != "small":
        return [*codec, "-pix_fmt", "yuv420p"]
    # Rate control set to keep detail is dropped for the size-first settings below.
    cleaned: list[str] = []
    skip = False
    for arg in codec:
        if skip:
            skip = False
            continue
        if arg in _RATE_CONTROL:
            skip = True
            continue
        cleaned.append(arg)
    return [
        # Never scale up: a clip already smaller than 720p is left its size.
        "-vf", f"scale=-2:'min({SMALL_HEIGHT},ih)'",
        *cleaned,
        *_SMALL_QUALITY.get(name, _SMALL_QUALITY["libx264"]),
        "-pix_fmt", "yuv420p",
    ]


class Mp4Writer:
    """Raw BGR frames in on stdin; one MP4 with its index at the front out."""

    def __init__(
        self,
        ffmpeg: str,
        target: Path,
        width: int,
        height: int,
        fps: float,
        output_args: list[str],
    ) -> None:
        # A file rather than a pipe: nothing reads stderr while frames go in.
        self._log = tempfile.TemporaryFile()  # noqa: SIM115 - closed in close()
        self._process = subprocess.Popen(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
                "-an", *output_args, "-movflags", "+faststart", str(target),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def write(self, frame: Any) -> None:
        import numpy as np

        stdin = self._process.stdin
        assert stdin is not None
        try:
            stdin.write(np.ascontiguousarray(frame).tobytes())
        except OSError as exc:
            raise RuntimeError(self._failure("ffmpeg stopped accepting frames")) from exc

    def close(self, abort: bool = False) -> None:
        stdin = self._process.stdin
        if stdin is not None:
            with suppress(OSError):
                stdin.close()
        if abort:
            with suppress(OSError):
                self._process.kill()
        try:
            code = self._process.wait(timeout=600)
            if not abort and code != 0:
                raise RuntimeError(self._failure(f"ffmpeg exited with code {code}"))
        finally:
            self._log.close()

    def _failure(self, message: str) -> str:
        with suppress(OSError, ValueError):
            self._log.seek(0)
            tail = self._log.read()[-800:].decode("utf-8", "replace").strip()
            if tail:
                return f"{message}: {tail}"
        return message


class ExportManager:
    """At most one export at a time, kept until it is downloaded or goes stale."""

    def __init__(
        self,
        renders: RenderManager,
        exports_root: Path,
        *,
        video_path: VideoPath,
        ffmpeg: str | None = None,
        writer_factory: WriterFactory | None = None,
        encoder: tuple[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.renders = renders
        self.root = Path(exports_root)
        self._video_path = video_path
        self._ffmpeg = ffmpeg if ffmpeg is not None else renders.ffmpeg
        self._writer_factory = writer_factory
        self._encoder = encoder
        self._lock = threading.Lock()
        self._jobs: dict[str, ExportJob] = {}
        self._thread: threading.Thread | None = None

    def start(self, render_id: str, quality: str) -> dict[str, Any]:
        if quality not in QUALITIES:
            raise ValueError(f"quality must be one of {QUALITIES}")
        if not self._ffmpeg and self._writer_factory is None:
            raise ExportUnavailable("ไม่พบ ffmpeg จึงบันทึกเป็น MP4 ไม่ได้")
        directory = self.renders.directory(render_id)
        meta = read_meta(directory)
        if meta is None or not (directory / TRACKS_FILE).is_file():
            raise ExportUnavailable("ไม่พบผลการประมวลผลนี้")
        state = render_state(directory, meta)
        if state["last_frame"] is None:
            raise ExportUnavailable("ผลนี้ยังไม่มีภาพที่ประมวลผลเสร็จให้บันทึก")
        video = self._video_path(str(meta.get("video") or ""))
        if video is None:
            raise ExportUnavailable("ไม่พบไฟล์วิดีโอต้นฉบับของผลนี้ในโฟลเดอร์วิดีโอ")
        percent = None
        if not state["complete"]:
            duration = float(state.get("duration_ms") or 0.0)
            processed = float(state.get("processed_ms") or 0.0)
            percent = min(99, round(processed / duration * 100)) if duration else 0
        with self._lock:
            running = next((job for job in self._jobs.values() if job.status == "running"), None)
            if running is not None:
                raise ExportBusy("กำลังเตรียมไฟล์อีกไฟล์อยู่ รอให้เสร็จก่อน")
            self._remove_stale_locked()
            export_id = secrets.token_hex(16)
            job = ExportJob(
                id=export_id,
                render_id=render_id,
                quality=quality,
                filename=export_filename(meta, quality, percent),
                directory=self.root / export_id,
            )
            self._jobs[export_id] = job
        job.directory.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, args=(job, directory, state, video), name="render-export", daemon=True
        )
        self._thread.start()
        _LOG.info("Exporting %s as %s (%s)", render_id, job.filename, quality)
        return job.snapshot()

    def status(self, export_id: str) -> dict[str, Any] | None:
        job = self._get(export_id)
        return job.snapshot() if job else None

    def ready_file(self, export_id: str) -> tuple[Path, str] | None:
        job = self._get(export_id)
        if job is None or job.status != "ready" or not job.path.is_file():
            return None
        return job.path, job.filename

    def finished(self, export_id: str) -> None:
        """The file has been handed over: nothing of it stays on this machine."""
        with self._lock:
            job = self._jobs.pop(export_id, None)
        if job is not None:
            shutil.rmtree(job.directory, ignore_errors=True)

    def shutdown(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)

    # -- helpers ----------------------------------------------------------------

    def _get(self, export_id: str) -> ExportJob | None:
        if not _EXPORT_ID.fullmatch(export_id or ""):
            return None
        with self._lock:
            return self._jobs.get(export_id)

    def _writer(self, job: ExportJob, width: int, height: int, fps: float) -> Any:
        if self._writer_factory is not None:
            return self._writer_factory(job.path, width, height, fps, job.quality)
        encoder = self._encoder or choose_encoder(str(self._ffmpeg))
        return Mp4Writer(
            str(self._ffmpeg), job.path, width, height, fps, encoder_args(job.quality, encoder)
        )

    def _run(self, job: ExportJob, render_dir: Path, state: dict[str, Any], video: Path) -> None:
        import cv2

        from .annotate import annotate_frame

        writer = None
        capture = None
        try:
            index = TrackIndex(render_dir / TRACKS_FILE)
            index.refresh()
            rows = index.between(0, state["last_frame"])
            frames = [row[0] for row in rows]
            boxes = {row[0]: [_box(item) for item in row[1]] for row in rows}
            stride = int(state["stride"])
            # The last frame there are boxes for, plus the stride they cover.
            end = frames[-1] + stride if frames else 0

            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {video}")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if not 0.0 < fps < 1000.0:
                fps = float(state["source_fps"])
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if state["complete"] and total:
                end = max(end, total)
            # yuv420p has half-resolution chroma, so x264 refuses odd dimensions.
            out_width, out_height = width - width % 2, height - height % 2
            writer = self._writer(job, out_width, out_height, fps)
            frame_index = -1
            while frame_index + 1 < end:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_index += 1
                placed = boxes_at(frame_index, frames, boxes, stride)
                if placed:
                    frame = annotate_frame(frame, with_track_ids(placed), width, height)
                if (out_width, out_height) != (width, height):
                    frame = frame[:out_height, :out_width]
                writer.write(frame)
                if frame_index % 30 == 0:
                    job.percent = min(99.0, (frame_index + 1) / end * 100.0)
            writer.close()
            writer = None
            job.size_bytes = job.path.stat().st_size
            job.percent = 100.0
            job.status = "ready"
            _LOG.info("Exported %s (%.1f MB)", job.filename, job.size_bytes / 1e6)
        except Exception as exc:
            _LOG.exception("Export of %s failed", job.render_id)
            if writer is not None:
                with suppress(Exception):
                    writer.close(abort=True)
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            shutil.rmtree(job.directory, ignore_errors=True)
        finally:
            if capture is not None:
                capture.release()
            job.finished_at = time.time()

    def _remove_stale_locked(self) -> None:
        now = time.time()
        for export_id, job in list(self._jobs.items()):
            if job.finished_at is not None and now - job.finished_at > STALE_AFTER_S:
                self._jobs.pop(export_id, None)
                shutil.rmtree(job.directory, ignore_errors=True)
        # Left over from a server that stopped before anyone downloaded them.
        if self.root.is_dir():
            for child in self.root.iterdir():
                ours = _EXPORT_ID.fullmatch(child.name) and child.name not in self._jobs
                if child.is_dir() and ours:
                    try:
                        age = now - child.stat().st_mtime
                    except OSError:
                        continue
                    if age > STALE_AFTER_S:
                        shutil.rmtree(child, ignore_errors=True)


def _box(item: list[Any]) -> dict[str, Any]:
    """TrackIndex.between's compact box back to the dict annotate_frame draws."""
    x1, y1, x2, y2, confidence, class_id, track_id, label = item
    return {
        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        "confidence": confidence, "class_id": class_id, "track_id": track_id, "label": label,
    }


def _safe(value: str) -> str:
    return _UNSAFE_NAME.sub("-", value).strip("-.") or "x"
