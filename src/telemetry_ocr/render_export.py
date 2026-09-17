"""Turn a render into one MP4 file that can be sent to someone.

A render is kept as two-second fragmented-MP4 segments, one folder per run,
because that is what lets it be watched while it is being made. Nobody can be
sent that. An export joins the segments back into a single MP4 with its index at
the front: either as they are -- a copy of the stream, which took 2.7 seconds for
a 25-minute render -- or scaled to 720p and re-encoded, about a fifth of the size,
for a chat that will not take a gigabyte.

Segments are joined by concatenating their bytes, not by handing ffmpeg the
playlist. An fMP4 init segment followed by its media segments is itself a valid
fragmented MP4; a playlist from a render that was stopped has no end marker, and
ffmpeg's HLS reader takes that for a live stream and waits for more.

The server holds PaddleOCR, so like render_job this module must not import torch.
Exports run ffmpeg in a thread; one at a time, which is all a laptop GPU's
encoder and a single disk want anyway.
"""

from __future__ import annotations

import logging
import re
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .render_job import (
    FORMAT,
    INIT_FILE,
    LEGACY_FORMAT,
    LEGACY_VIDEO_FILE,
    SOFTWARE_ENCODER,
    RenderManager,
    choose_encoder,
    read_meta,
)

_LOG = logging.getLogger(__name__)

QUALITIES = ("full", "small")
SMALL_HEIGHT = 720
# An export nobody came back for is removed after this long.
STALE_AFTER_S = 3600.0
_EXPORT_ID = re.compile(r"[0-9a-f]{32}")
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")

# Size-for-a-chat settings per encoder. Measured with NVENC over a whole
# 25-minute flight: 286 MB, about 11 MB a minute at 720p, against 945 MB for the
# same render copied as it is.
_SMALL_QUALITY: dict[str, tuple[str, ...]] = {
    "h264_nvenc": ("-cq", "30", "-maxrate", "3M", "-bufsize", "6M"),
    "h264_qsv": ("-global_quality", "30"),
    "libx264": ("-crf", "28"),
}
# The render encoder's own quality options, each followed by its value.
_RATE_CONTROL = frozenset({"-cq", "-crf", "-global_quality", "-maxrate", "-bufsize"})

Runner = Callable[[list[str], float, Callable[[float], None]], None]


class ExportBusy(RuntimeError):
    """Another export is still being made."""


class ExportUnavailable(RuntimeError):
    """Nothing to export, or no ffmpeg to export it with."""


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
    drew the boxes, and "partial-40%" on one that was stopped partway, so nobody
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


def join_run(render_dir: Path, run: dict[str, Any], target: Path) -> int:
    """Write one run's init segment and media segments, in order, as one fMP4 file."""
    run_dir = render_dir / str(run["run"])
    written = 0
    with target.open("wb") as out:
        for name in [INIT_FILE, *(segment["file"] for segment in run["segments"])]:
            with (run_dir / name).open("rb") as part:
                shutil.copyfileobj(part, out, length=1024 * 1024)
                written += part.tell()
    return written


def ffmpeg_command(
    ffmpeg: str,
    inputs: list[Path],
    output: Path,
    quality: str,
    encoder: tuple[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """The ffmpeg call that turns the joined runs into one MP4 with its index first."""
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-nostats"]
    if len(inputs) == 1:
        command += ["-i", str(inputs[0])]
    else:
        # Several runs: the concat demuxer, reading a list written beside them.
        listing = output.with_name("runs.txt")
        listing.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in inputs), encoding="utf-8"
        )
        command += ["-f", "concat", "-safe", "0", "-i", str(listing)]
    command += ["-an"]
    if quality == "small":
        name, codec = encoder or SOFTWARE_ENCODER
        # The render's rate control is set to keep detail; a file for a chat wants
        # the opposite, so those settings are dropped for the ones below.
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
        command += [
            # Never scale up: a clip already smaller than 720p is left its size.
            "-vf", f"scale=-2:'min({SMALL_HEIGHT},ih)'",
            *cleaned,
            *_SMALL_QUALITY.get(name, _SMALL_QUALITY["libx264"]),
            "-pix_fmt", "yuv420p",
        ]
    else:
        command += ["-c", "copy"]
    command += ["-movflags", "+faststart", "-progress", "pipe:1", str(output)]
    return command


def run_ffmpeg(command: list[str], duration_s: float, progress: Callable[[float], None]) -> None:
    """Run ffmpeg, reporting how far through the video it has written."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None
    for line in process.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and duration_s > 0:
            try:
                progress(min(99.0, int(value) / 1_000_000 / duration_s * 100.0))
            except ValueError:
                continue
    code = process.wait()
    if code != 0:
        tail = (process.stderr.read() if process.stderr else "").strip()[-600:]
        raise RuntimeError(f"ffmpeg exited with code {code}: {tail}")


class ExportManager:
    """At most one export at a time, kept until it is downloaded or goes stale."""

    def __init__(
        self,
        renders: RenderManager,
        exports_root: Path,
        *,
        ffmpeg: str | None = None,
        runner: Runner | None = None,
        encoder: tuple[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.renders = renders
        self.root = Path(exports_root)
        self._ffmpeg = ffmpeg if ffmpeg is not None else renders.ffmpeg
        self._runner = runner or run_ffmpeg
        self._encoder = encoder
        self._lock = threading.Lock()
        self._jobs: dict[str, ExportJob] = {}
        self._thread: threading.Thread | None = None

    def start(self, render_id: str, quality: str) -> dict[str, Any]:
        if quality not in QUALITIES:
            raise ValueError(f"quality must be one of {QUALITIES}")
        if not self._ffmpeg:
            raise ExportUnavailable("ไม่พบ ffmpeg จึงบันทึกเป็น MP4 ไม่ได้")
        directory = self.renders.directory(render_id)
        index = self.renders.stream(render_id)
        if index is None:
            raise ExportUnavailable("ไม่พบผลการประมวลผลนี้")
        if index["format"] == FORMAT and not index["runs"]:
            raise ExportUnavailable("ผลนี้ยังไม่มีภาพที่ประมวลผลเสร็จให้บันทึก")
        meta = read_meta(directory) or {}
        percent = None
        if not index.get("complete"):
            duration = float(index.get("duration_ms") or 0.0)
            processed = float(index.get("processed_ms") or 0.0)
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
            target=self._run, args=(job, directory, index), name="render-export", daemon=True
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

    def _run(self, job: ExportJob, render_dir: Path, index: dict[str, Any]) -> None:
        try:
            inputs: list[Path] = []
            if index["format"] == LEGACY_FORMAT:
                inputs.append(render_dir / LEGACY_VIDEO_FILE)
            else:
                for number, run in enumerate(index["runs"]):
                    joined = job.directory / f"run-{number:03d}.mp4"
                    join_run(render_dir, run, joined)
                    inputs.append(joined)
            duration_s = float(index.get("processed_ms") or 0.0) / 1000.0
            encoder = self._encoder
            if job.quality == "small" and encoder is None:
                encoder = choose_encoder(str(self._ffmpeg))
            command = ffmpeg_command(str(self._ffmpeg), inputs, job.path, job.quality, encoder)

            def progress(percent: float) -> None:
                job.percent = percent

            self._runner(command, duration_s, progress)
            for joined in inputs:
                if joined.parent == job.directory:
                    joined.unlink(missing_ok=True)
            (job.directory / "runs.txt").unlink(missing_ok=True)
            job.size_bytes = job.path.stat().st_size
            job.percent = 100.0
            job.status = "ready"
            _LOG.info("Exported %s (%.1f MB)", job.filename, job.size_bytes / 1e6)
        except Exception as exc:
            _LOG.exception("Export of %s failed", job.render_id)
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            shutil.rmtree(job.directory, ignore_errors=True)
        finally:
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


def _safe(value: str) -> str:
    return _UNSAFE_NAME.sub("-", value).strip("-.") or "x"
