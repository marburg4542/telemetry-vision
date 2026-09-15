"""Run a whole video through a detector once, and keep the result to play.

The page used to post a frame to the detector every hundred milliseconds or so
and draw the answer over the playing video. Two things were wrong with that and
neither was the drawing. BoT-SORT was fed whatever frames the GPU had time for,
ten to sixteen a second out of thirty and never evenly spaced, so it lost
identities whenever the camera moved; and every box was already a frame or
three old by the time it was painted over a picture that had moved on.

Here the tracker sees every frame of the file, in order, once. The boxes are
drawn into the frames themselves and encoded as an H.264 MP4 the browser plays
like any other video, so what is on screen is exactly what the model saw at
that instant. The per-frame detections are kept beside it in tracks.jsonl, which
is what a capture reads its boxes from rather than asking the model again.

Process layout is the same as the live worker's was: the server holds
PaddleOCR, so torch loads only in a spawned child, and nothing but builtins
crosses back. This module is imported by the server and must stay free of both.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

VIDEO_FILE = "annotated.mp4"
TRACKS_FILE = "tracks.jsonl"
META_FILE = "meta.json"

_PROGRESS_INTERVAL_S = 0.5
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_RENDER_ID = re.compile(r"[A-Za-z0-9_.-]{1,200}")
_ACTIVE = frozenset({"loading", "running", "cancelling"})

# Offered in the page. The trained sizes of the checkpoints in use, plus 1280 for
# anyone who trains larger; anything else is a typo rather than a choice.
IMAGE_SIZES = (416, 640, 960, 1280)
STRIDES = (1, 2, 3)


class RenderBusy(RuntimeError):
    """A render is already running; the GPU takes one at a time."""


class RenderUnavailable(RuntimeError):
    """Rendering cannot run on this machine, e.g. there is no ffmpeg."""


@dataclass(frozen=True, slots=True)
class RenderSettings:
    confidence: float = 0.45
    iou: float = 0.45
    image_size: int = 640
    # 1 is every frame. 2 halves the work and the output frame rate; tracking
    # still sees an even cadence, which is what the live overlay never had.
    stride: int = 1

    def validate(self) -> None:
        if not 0.05 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.05 and 1")
        if not 0.05 <= self.iou <= 1.0:
            raise ValueError("iou must be between 0.05 and 1")
        if self.image_size not in IMAGE_SIZES:
            raise ValueError(f"image_size must be one of {IMAGE_SIZES}")
        if self.stride not in STRIDES:
            raise ValueError(f"stride must be one of {STRIDES}")

    def key(self) -> str:
        return f"c{self.confidence:.2f}-s{self.image_size}-st{self.stride}"


def render_id_for(video_id: str, model_name: str, settings: RenderSettings) -> str:
    """One directory name per video, model and settings, so a result is reused."""
    stem = Path(video_id).with_suffix("").as_posix()
    return "__".join((_safe(stem), _safe(model_name), settings.key()))


def fingerprint(path: Path) -> dict[str, int]:
    """Enough to notice that a file was replaced after it was rendered."""
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def read_meta(directory: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def is_current(meta: dict[str, Any], video_path: Path | None, model_path: Path | None) -> bool:
    """A finished render whose video and checkpoint are still the files it was made from."""
    if meta.get("status") != "complete":
        return False
    try:
        if video_path is None or fingerprint(video_path) != meta.get("video_fingerprint"):
            return False
        if model_path is None or fingerprint(model_path) != meta.get("model_fingerprint"):
            return False
    except OSError:
        return False
    return True


# Tried in order, each confirmed with a real two-frame encode: an encoder that
# `ffmpeg -encoders` lists can still fail at run time for want of a driver or of
# the silicon block. The hardware ones matter because x264 competes with the
# tracker for the same few CPU cores, while NVENC runs on a fixed-function block
# beside the GPU cores YOLO uses. Measured on the GTX 1650 machine over 600
# frames of 1080p30: NVENC ~275 fps, QuickSync ~135, x264 veryfast ~105.
_ENCODERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "h264_nvenc",
        ("-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "27",
         "-b:v", "0", "-maxrate", "8M", "-bufsize", "16M"),
    ),
    ("h264_qsv", ("-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "27")),
    ("libx264", ("-c:v", "libx264", "-preset", "veryfast", "-crf", "23")),
)
SOFTWARE_ENCODER = _ENCODERS[-1]


def choose_encoder(ffmpeg: str) -> tuple[str, tuple[str, ...]]:
    """The first encoder that works here, or the one TELEMETRY_OCR_RENDER_ENCODER names."""
    forced = os.environ.get("TELEMETRY_OCR_RENDER_ENCODER", "").strip()
    candidates = [item for item in _ENCODERS if item[0] == forced] if forced else []
    for name, args in candidates or list(_ENCODERS):
        if _encoder_works(ffmpeg, args):
            return name, args
    return SOFTWARE_ENCODER


def _encoder_works(ffmpeg: str, args: tuple[str, ...]) -> bool:
    try:
        completed = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10",
                "-frames:v", "2", *args, "-pix_fmt", "yuv420p", "-f", "null", "-",
            ],
            capture_output=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def find_ffmpeg() -> str | None:
    """ffmpeg from TELEMETRY_OCR_FFMPEG, the PATH, or where winget puts it.

    The last one matters because the server runs as a scheduled task, whose
    PATH is captured at logon and can predate an ffmpeg installed since.
    """
    configured = os.environ.get("TELEMETRY_OCR_FFMPEG", "").strip()
    if configured:
        return configured if Path(configured).is_file() else None
    found = shutil.which("ffmpeg")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        winget = Path(local) / "Microsoft" / "WinGet"
        link = winget / "Links" / "ffmpeg.exe"
        if link.is_file():
            return str(link)
        for candidate in sorted((winget / "Packages").glob("*FFmpeg*/**/bin/ffmpeg.exe")):
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# The render itself. Runs in the child, or in a thread under test.
# ---------------------------------------------------------------------------


def discard_partial(directory: Path) -> None:
    """Remove what an unfinished render left behind.

    Its .part files, then the folder itself if that emptied it -- otherwise every
    cancelled or failed render leaves an empty folder in output/renders. A folder
    that still holds a finished result is left alone: rmdir refuses it.
    """
    for name in (VIDEO_FILE, TRACKS_FILE):
        # Windows refuses while ffmpeg still has the file open; nothing to add.
        with suppress(OSError):
            (directory / f"{name}.part").unlink(missing_ok=True)
    with suppress(OSError):
        directory.rmdir()


def run_render(
    *,
    video: str,
    output_dir: str,
    settings: dict[str, Any],
    meta: dict[str, Any],
    detector: Any,
    encoder_factory: Callable[[Path, int, int, float], Any],
    publish: Callable[[dict[str, Any]], None],
    cancelled: Callable[[], bool],
) -> None:
    import cv2

    from .annotate import annotate_frame

    options = RenderSettings(**settings)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # A previous result under this name is replaced, never mixed with.
    for stale in (VIDEO_FILE, TRACKS_FILE, META_FILE):
        (out / stale).unlink(missing_ok=True)
    video_part = out / f"{VIDEO_FILE}.part"
    tracks_part = out / f"{TRACKS_FILE}.part"

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        discard_partial(out)
        raise RuntimeError(f"Cannot open video: {video}")
    started = time.perf_counter()
    index = -1
    written = 0
    # Where a frame's time goes, averaged over the frames written. Reported with
    # the progress, because "slow" has a different fix for each of these.
    spans = {"decode": 0.0, "track": 0.0, "draw": 0.0, "encode": 0.0}
    finished = False
    was_cancelled = False
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not 0.0 < fps < 1000.0:
            fps = 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        # yuv420p has half-resolution chroma, so x264 refuses odd dimensions.
        out_width, out_height = width - width % 2, height - height % 2
        out_fps = fps / options.stride
        encoder = encoder_factory(video_part, out_width, out_height, out_fps)
        publish({"kind": "progress", "done": 0, "total": total, "fps": 0.0})
        last_published = time.perf_counter()
        try:
            with tracks_part.open("w", encoding="utf-8") as tracks:
                while True:
                    if cancelled():
                        was_cancelled = True
                        break
                    step = time.perf_counter()
                    if not capture.grab():
                        break
                    index += 1
                    # grab() without retrieve() skips the decode to pixels, so a
                    # stride costs little more than the frames it keeps.
                    if index % options.stride:
                        continue
                    ok, frame = capture.retrieve()
                    if not ok or frame is None:
                        break
                    decoded = time.perf_counter()
                    detections = detector.track(
                        frame,
                        confidence=options.confidence,
                        iou=options.iou,
                        image_size=options.image_size,
                        tracker="botsort.yaml",
                    )
                    tracked = time.perf_counter()
                    boxes = [_plain(item) for item in detections]
                    marked = annotate_frame(frame, with_track_ids(boxes), width, height)
                    if (out_width, out_height) != (width, height):
                        marked = marked[:out_height, :out_width]
                    drawn = time.perf_counter()
                    # Blocks when ffmpeg falls behind, so this also measures
                    # whether the encoder is keeping up.
                    encoder.write(marked)
                    spans["decode"] += decoded - step
                    spans["track"] += tracked - decoded
                    spans["draw"] += drawn - tracked
                    spans["encode"] += time.perf_counter() - drawn
                    tracks.write(
                        json.dumps(
                            {
                                "frame_index": index,
                                "time_ms": round(index * 1000.0 / fps, 3),
                                "detections": boxes,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
                    now = time.perf_counter()
                    if now - last_published >= _PROGRESS_INTERVAL_S:
                        last_published = now
                        elapsed = now - started
                        publish(
                            {
                                "kind": "progress",
                                "done": index + 1,
                                "total": total,
                                "fps": round(written / elapsed, 2) if elapsed > 0 else 0.0,
                                "stages_ms": _averages(spans, written),
                            }
                        )
            finished = not was_cancelled
        finally:
            encoder.close(abort=not finished)
    except BaseException:
        discard_partial(out)
        raise
    finally:
        capture.release()

    if was_cancelled:
        discard_partial(out)
        publish({"kind": "cancelled"})
        return
    if written == 0:
        discard_partial(out)
        raise RuntimeError("The video yielded no frames to render")

    os.replace(video_part, out / VIDEO_FILE)
    os.replace(tracks_part, out / TRACKS_FILE)
    source_frames = index + 1
    record = {
        **meta,
        "status": "complete",
        "settings": asdict(options),
        "source_fps": fps,
        "fps": out_fps,
        "stride": options.stride,
        "source_frames": source_frames,
        "frames_written": written,
        "width": out_width,
        "height": out_height,
        "duration_ms": round(source_frames * 1000.0 / fps, 3),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "stages_ms": _averages(spans, written),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "video_file": VIDEO_FILE,
        "tracks_file": TRACKS_FILE,
    }
    _write_json(out / META_FILE, record)
    publish({"kind": "complete", "meta": record})


class FfmpegEncoder:
    """Raw BGR frames in on stdin, a browser-playable H.264 MP4 out."""

    def __init__(
        self,
        ffmpeg: str,
        path: Path,
        width: int,
        height: int,
        fps: float,
        codec: tuple[str, ...] = SOFTWARE_ENCODER[1],
    ) -> None:
        # A file rather than a pipe: nothing reads ffmpeg's stderr while frames
        # are going in, and a full pipe would stall the encoder mid-render.
        self._log = tempfile.TemporaryFile()  # noqa: SIM115 - closed in close()
        self._path = path
        self._process = subprocess.Popen(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
                "-an", *codec,
                "-pix_fmt", "yuv420p",
                # The index at the front, so the page can start playing and
                # seeking before it has fetched the whole file.
                "-movflags", "+faststart",
                "-f", "mp4", str(path),
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
        finally:
            failure = "" if abort or self._process.returncode == 0 else self._failure(
                f"ffmpeg exited with code {self._process.returncode}"
            )
            self._log.close()
        if abort:
            self._path.unlink(missing_ok=True)
        elif code != 0:
            raise RuntimeError(failure)

    def _failure(self, message: str) -> str:
        with suppress(OSError, ValueError):
            self._log.seek(0)
            tail = self._log.read()[-800:].decode("utf-8", "replace").strip()
            if tail:
                return f"{message}: {tail}"
        return message


def _render_process_main(job: dict[str, Any], progress: Any, cancel: Any) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s render %(message)s"
    )
    cpu_threads = max(1, int(os.environ.get("TELEMETRY_OCR_YOLO_THREADS", "2")))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))

    def publish(message: dict[str, Any]) -> None:
        with suppress(OSError, ValueError):
            progress.put(message)

    try:
        from .detection import load_ultralytics_detector

        detector = load_ultralytics_detector(job["model_path"])
        publish({"kind": "loaded", "metadata": detector.metadata()})
        ffmpeg = str(job["ffmpeg"])
        encoder_name, codec = choose_encoder(ffmpeg)
        _LOG.info("Encoding with %s", encoder_name)
        publish({"kind": "encoder", "name": encoder_name})
        run_render(
            video=job["video"],
            output_dir=job["output_dir"],
            settings=job["settings"],
            meta={**job["meta"], "encoder": encoder_name},
            detector=detector,
            encoder_factory=lambda path, width, height, fps: FfmpegEncoder(
                ffmpeg, path, width, height, fps, codec
            ),
            publish=publish,
            cancelled=cancel.is_set,
        )
    except Exception as exc:
        _LOG.exception("Render failed")
        publish({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# The server side.
# ---------------------------------------------------------------------------

Launcher = Callable[[dict[str, Any]], tuple[Any, Any, Any]]


def spawn_launcher(job: dict[str, Any]) -> tuple[Any, Any, Any]:
    context = multiprocessing.get_context("spawn")
    progress = context.Queue()
    cancel = context.Event()
    process = context.Process(
        target=_render_process_main,
        args=(job, progress, cancel),
        name="telemetry-ocr-render",
        daemon=True,
    )
    process.start()
    return process, progress, cancel


class RenderManager:
    """At most one render at a time, and the results already on disk."""

    def __init__(
        self,
        renders_root: Path,
        *,
        launcher: Launcher | None = None,
        ffmpeg: str | None = None,
    ) -> None:
        self.root = Path(renders_root)
        self._launch = launcher or spawn_launcher
        self._ffmpeg = ffmpeg if ffmpeg is not None else find_ffmpeg()
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None
        self._handle: Any = None
        self._progress: Any = None
        self._cancel: Any = None
        self._tracks: dict[str, tuple[int, TrackIndex]] = {}

    @property
    def ffmpeg(self) -> str | None:
        return self._ffmpeg

    def directory(self, render_id: str) -> Path:
        if not _RENDER_ID.fullmatch(render_id) or render_id in {".", ".."}:
            raise ValueError("Invalid render id")
        return self.root / render_id

    def start(
        self,
        *,
        render_id: str,
        video_id: str,
        video_path: Path,
        model_name: str,
        model_label: str,
        model_path: Path,
        settings: RenderSettings,
    ) -> dict[str, Any]:
        settings.validate()
        if not self._ffmpeg:
            raise RenderUnavailable(
                "ไม่พบ ffmpeg — ติดตั้ง ffmpeg หรือตั้ง TELEMETRY_OCR_FFMPEG ให้ชี้ไปที่ ffmpeg.exe"
            )
        directory = self.directory(render_id)
        with self._lock:
            self._poll_locked()
            if self._job is not None and self._job["status"] in _ACTIVE:
                raise RenderBusy(
                    f"กำลังประมวลผล {self._job['video']} ด้วย {self._job['model']} อยู่ "
                    "รอให้เสร็จหรือยกเลิกก่อน"
                )
            job = {
                "video": str(video_path),
                "model_path": str(model_path),
                "output_dir": str(directory),
                "settings": asdict(settings),
                "ffmpeg": self._ffmpeg,
                "meta": {
                    "render_id": render_id,
                    "video": video_id,
                    "video_fingerprint": fingerprint(video_path),
                    "model": model_name,
                    "model_label": model_label,
                    "model_fingerprint": fingerprint(model_path),
                },
            }
            self._tracks.pop(render_id, None)
            self._handle, self._progress, self._cancel = self._launch(job)
            self._job = {
                "render_id": render_id,
                "video": video_id,
                "model": model_name,
                "settings": asdict(settings),
                "status": "loading",
                "done": 0,
                "total": 0,
                "fps": 0.0,
                "started_at": time.time(),
                "error": None,
            }
            _LOG.info("Rendering %s with %s (%s)", video_id, model_name, settings.key())
            return self._snapshot_locked()

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._poll_locked()
            if self._job is None:
                return {"status": "idle"}
            return self._snapshot_locked()

    def cancel(self) -> bool:
        with self._lock:
            self._poll_locked()
            if self._job is None or self._job["status"] not in _ACTIVE:
                return False
            if self._cancel is not None:
                self._cancel.set()
            self._job["status"] = "cancelling"
            return True

    def shutdown(self, timeout: float = 10.0) -> None:
        with self._lock:
            handle = self._handle
            job = self._job
            if self._cancel is not None:
                self._cancel.set()
        if handle is None:
            return
        handle.join(timeout)
        if handle.is_alive():
            _LOG.warning("Render process ignored cancellation; terminating it")
            handle.terminate()
            handle.join(2.0)
            if job is not None:
                discard_partial(self.directory(job["render_id"]))

    def renders(self, video_id: str | None = None) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        found = []
        for directory in sorted(self.root.iterdir()):
            meta = read_meta(directory) if directory.is_dir() else None
            if meta is None or meta.get("status") != "complete":
                continue
            if video_id is not None and meta.get("video") != video_id:
                continue
            if not (directory / VIDEO_FILE).is_file():
                continue
            found.append(meta)
        return found

    def delete(self, render_id: str) -> None:
        directory = self.directory(render_id)
        with self._lock:
            self._poll_locked()
            if (
                self._job is not None
                and self._job["status"] in _ACTIVE
                and self._job["render_id"] == render_id
            ):
                raise RenderBusy("ยกเลิกการประมวลผลนี้ก่อนลบ")
            self._tracks.pop(render_id, None)
        if directory.is_dir():
            shutil.rmtree(directory)

    def detections_at(self, render_id: str, frame_index: int) -> list[dict[str, Any]] | None:
        """The boxes a render drew on one source frame, or None if it has no such frame."""
        path = self.directory(render_id) / TRACKS_FILE
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return None
        with self._lock:
            cached = self._tracks.get(render_id)
            if cached is None or cached[0] != stamp:
                # Few renders are looked at in one sitting; keep the index of the
                # last couple rather than one per render ever opened.
                if len(self._tracks) >= 2:
                    self._tracks.pop(next(iter(self._tracks)))
                cached = (stamp, TrackIndex(path))
                self._tracks[render_id] = cached
        return cached[1].at(frame_index)

    # -- helpers ----------------------------------------------------------------

    def _snapshot_locked(self) -> dict[str, Any]:
        assert self._job is not None
        job = dict(self._job)
        rate = float(job.get("fps") or 0.0) * int(job["settings"].get("stride", 1))
        remaining = max(0, int(job["total"]) - int(job["done"]))
        running = rate > 0 and job["status"] == "running"
        job["eta_s"] = round(remaining / rate, 1) if running else None
        return job

    def _poll_locked(self) -> None:
        job = self._job
        if job is None or self._progress is None:
            return
        while True:
            try:
                message = self._progress.get_nowait()
            except (queue.Empty, OSError, ValueError, EOFError):
                break
            kind = message.get("kind")
            if kind == "loaded":
                if job["status"] == "loading":
                    job["status"] = "running"
                job["model_metadata"] = message.get("metadata") or {}
            elif kind == "progress":
                if job["status"] == "loading":
                    job["status"] = "running"
                job["done"] = int(message.get("done", 0))
                job["total"] = int(message.get("total", 0))
                job["fps"] = float(message.get("fps", 0.0))
                if isinstance(message.get("stages_ms"), dict):
                    job["stages_ms"] = message["stages_ms"]
            elif kind == "encoder":
                job["encoder"] = str(message.get("name") or "")
            elif kind == "complete":
                job["status"] = "complete"
                job["done"] = job["total"] = int(message["meta"].get("source_frames", job["total"]))
                job["meta"] = message.get("meta")
            elif kind == "cancelled":
                job["status"] = "cancelled"
            elif kind == "error":
                job["status"] = "error"
                job["error"] = str(message.get("error") or "Render failed")
        if job["status"] in _ACTIVE and self._handle is not None and not self._handle.is_alive():
            # One last look: the final message can land between the drain above
            # and the process exiting.
            with suppress(queue.Empty, OSError, ValueError, EOFError):
                message = self._progress.get(timeout=0.2)
                self._progress_put_back(message)
                self._poll_locked()
                return
            job["status"] = "error"
            job["error"] = f"Render process exited with code {self._handle.exitcode}"
        if job["status"] not in _ACTIVE and self._handle is not None:
            self._handle.join(1.0)
            self._handle = None
            if job["status"] != "complete":
                # The render tidies up after itself, unless it died before it could.
                discard_partial(self.directory(job["render_id"]))

    def _progress_put_back(self, message: dict[str, Any]) -> None:
        # Re-queued locally so the drain loop handles it like any other message.
        pending: queue.Queue[dict[str, Any]] = queue.Queue()
        pending.put(message)
        remaining = self._progress
        self._progress = _ChainedQueue(pending, remaining)


class _ChainedQueue:
    def __init__(self, first: Any, second: Any) -> None:
        self._first = first
        self._second = second

    def get_nowait(self) -> Any:
        try:
            return self._first.get_nowait()
        except queue.Empty:
            return self._second.get_nowait()

    def get(self, timeout: float | None = None) -> Any:
        try:
            return self._first.get_nowait()
        except queue.Empty:
            return self._second.get(timeout=timeout)


class TrackIndex:
    """Byte offsets of each frame's line in tracks.jsonl, so a lookup reads one line."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offsets: dict[int, int] = {}
        with path.open("rb") as stream:
            offset = 0
            for line in stream:
                # The frame index is the first key written; reading just that
                # much avoids parsing forty thousand detection lists up front.
                match = re.match(rb'\{"frame_index": (\d+)', line)
                if match:
                    self._offsets[int(match.group(1))] = offset
                offset += len(line)

    def at(self, frame_index: int) -> list[dict[str, Any]] | None:
        offset = self._offsets.get(frame_index)
        if offset is None:
            return None
        with self._path.open("rb") as stream:
            stream.seek(offset)
            record = json.loads(stream.readline())
        detections = record.get("detections")
        return detections if isinstance(detections, list) else []


def _averages(spans: dict[str, float], frames: int) -> dict[str, float]:
    if not frames:
        return {}
    return {name: round(total * 1000.0 / frames, 1) for name, total in spans.items()}


def _plain(detection: Any) -> dict[str, Any]:
    """Builtins only: this dict crosses into the process that holds Paddle."""
    data = detection if isinstance(detection, dict) else asdict(detection)
    track_id = data.get("track_id")
    return {
        "x1": round(float(data["x1"]), 1),
        "y1": round(float(data["y1"]), 1),
        "x2": round(float(data["x2"]), 1),
        "y2": round(float(data["y2"]), 1),
        "confidence": round(float(data["confidence"]), 3),
        "class_id": int(data["class_id"]),
        "label": str(data["label"]),
        "track_id": None if track_id is None else int(track_id),
    }


def with_track_ids(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labelled = []
    for box in boxes:
        if box.get("track_id") is None:
            labelled.append(box)
        else:
            labelled.append({**box, "label": f"{box['label']} #{box['track_id']}"})
    return labelled


def _safe(value: str) -> str:
    return _UNSAFE.sub("-", value).strip("-.") or "x"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
