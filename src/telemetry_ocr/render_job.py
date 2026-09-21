"""Run a whole video through a detector once, and let the page draw it as it goes.

The page used to post a frame to the detector every hundred milliseconds or so
and draw the answer over the playing video. BoT-SORT was fed whatever frames the
GPU had time for, ten to sixteen a second out of thirty and never evenly spaced,
so it lost identities whenever the camera moved; and every box was already a
frame or three old by the time it was painted over a picture that had moved on.

Here the tracker sees every frame of the file (or every second or third, evenly
spaced), in order, once, and writes each frame's boxes to tracks.jsonl as it
goes. Nothing is drawn and nothing is encoded: the page plays the original video
and draws the boxes of the frame on screen over it, placing a box between two
processed frames by where it was on either side. That is what keeps the picture
at the source's full frame rate at every stride, and the tracker the only thing
the GPU and the CPU are spent on. A capture reads its boxes from the same file.

Renders made before this also drew their boxes into H.264 segments beside the
tracks; those segments are no longer read, and the tracks play the same way.
A render that stops early keeps what it tracked and continues from the frame
after the last one written.

Process layout is the same as the live worker's was: the server holds
PaddleOCR, so torch loads only in a spawned child, and nothing but builtins
crosses back. This module is imported by the server and must stay free of both.
"""

from __future__ import annotations

import bisect
import json
import logging
import multiprocessing
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_LOG = logging.getLogger(__name__)

TRACKS_FILE = "tracks.jsonl"
META_FILE = "meta.json"
# The tracker settings one render ran with, written beside its output.
TRACKER_FILE = "tracker.yaml"
# A detection this much above the confidence asked for is needed to start a new
# number. Measured on two flights: the numbers a vehicle was given twice fell
# from 7 to 3, and from 2 to 0, and nothing that stayed on screen longer than a
# fifth of a second went unnumbered.
NEW_TRACK_MARGIN = 0.15
# Renders made before the boxes were drawn by the page also wrote video: first
# one MP4 of it, then two-second segments in run folders. Their tracks play like
# any other; the video is only cleared away when the render is made again.
LEGACY_VIDEO_FILE = "annotated.mp4"
FORMAT = "tracks"
SEGMENTED_FORMAT = "hls"
LEGACY_FORMAT = "mp4"
# Keyframes this far apart in what an encoder writes, which also bounds how far
# a player seeks to find one.
SEGMENT_SECONDS = 2

_PROGRESS_INTERVAL_S = 0.5
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_RENDER_ID = re.compile(r"[A-Za-z0-9_.-]{1,200}")
_RUN_DIR = re.compile(r"run-(\d{3})")
_FRAME_INDEX = re.compile(rb'\{"frame_index": (\d+)')
_TRACK_ID = re.compile(rb'"track_id": (\d+)')
_ACTIVE = frozenset({"loading", "running", "cancelling"})

# Offered in the page. The trained sizes of the checkpoints in use, plus 1280 for
# anyone who trains larger; anything else is a typo rather than a choice. 1024 is
# the size the current checkpoints were trained at: it finds more cars than 640
# on this footage, and fewer motorcycles, so it is a choice rather than a default.
IMAGE_SIZES = (416, 640, 960, 1024, 1280)
STRIDES = (1, 2, 3)


class RenderBusy(RuntimeError):
    """A render is already running; the GPU takes one at a time."""


class RenderUnavailable(RuntimeError):
    """Rendering cannot run on this machine, e.g. there is no ffmpeg."""


class EncoderFailed(RuntimeError):
    """ffmpeg stopped taking frames; the segment it was writing is lost."""


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


def render_format(meta: dict[str, Any]) -> str:
    """"hls" for segmented renders; renders from before segments carry no format."""
    return str(meta.get("format") or LEGACY_FORMAT)


def made_from(meta: dict[str, Any], video_path: Path | None, model_path: Path | None) -> bool:
    """Whether the video and checkpoint are still the files this render was made from."""
    try:
        if video_path is None or fingerprint(video_path) != meta.get("video_fingerprint"):
            return False
        if model_path is None or fingerprint(model_path) != meta.get("model_fingerprint"):
            return False
    except OSError:
        return False
    return True


def is_current(meta: dict[str, Any], video_path: Path | None, model_path: Path | None) -> bool:
    """A finished render whose video and checkpoint are still the files it was made from."""
    return meta.get("status") == "complete" and made_from(meta, video_path, model_path)


def tracker_config(directory: Path | None = None, confidence: float | None = None) -> str:
    """The tracker settings a render uses: this project's, or Ultralytics' own.

    configs/botsort.yaml turns ReID on, which is what keeps a vehicle's number
    through the seconds it spends behind a tree: without it the tracker has only
    where the motion model expected the vehicle to be, and a long occlusion ends
    as a new number. Falling back to the bare name keeps a checkout without that
    file rendering rather than failing.

    Given a render's folder and confidence, the two thresholds that decide what
    the tracker trusts follow that confidence -- the first matching stage takes
    what will be drawn, a new number needs NEW_TRACK_MARGIN more -- and the
    result is written into the folder, so a render says what it was made with.
    """
    candidate = Path(__file__).resolve().parents[2] / "configs" / "botsort.yaml"
    if not candidate.is_file():
        return "botsort.yaml"
    if directory is None or confidence is None:
        return str(candidate)
    settings = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    settings["track_high_thresh"] = round(confidence, 3)
    settings["new_track_thresh"] = round(min(0.95, confidence + NEW_TRACK_MARGIN), 3)
    path = directory / TRACKER_FILE
    path.write_text(
        f"# configs/botsort.yaml as this render ran it, at confidence {confidence:.2f}.\n"
        + yaml.safe_dump(settings, sort_keys=False),
        encoding="utf-8",
    )
    return str(path)


def last_frame(path: Path) -> int | None:
    """The frame index of the last whole line of a tracks file, or None if it has none.

    Read from the end: a render is tens of thousands of lines, and this is asked
    every time the page polls. A line still being written has no newline yet and
    is not counted.
    """
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            chunk = 1 << 16
            while True:
                start = max(0, end - chunk)
                stream.seek(start)
                data = stream.read(end - start)
                cut_at = data.rfind(b"\n")
                if cut_at >= 0:
                    whole = data[: cut_at + 1]
                    lines = whole.splitlines()
                    # The first line of a chunk that did not start the file may
                    # be the tail of a longer one; only trust it at the start.
                    candidates = lines if start == 0 else lines[1:]
                    for line in reversed(candidates):
                        match = _FRAME_INDEX.match(line)
                        if match:
                            return int(match.group(1))
                if start == 0:
                    return None
                chunk *= 4
    except OSError:
        return None


def render_state(directory: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """How far a render has got, on the source video's timeline.

    processed_ms is where the boxes stop: the last frame tracked, plus the stride
    it would have covered before the next one. A finished render covers the
    whole video.
    """
    source_fps = float(meta.get("source_fps") or 0.0) or 30.0
    stride = int(meta.get("stride") or 1)
    status = str(meta.get("status") or "")
    duration_ms = meta.get("duration_ms")
    last = last_frame(directory / TRACKS_FILE)
    if status == "complete":
        processed_ms = float(duration_ms or 0.0)
    elif last is None:
        processed_ms = 0.0
    else:
        processed_ms = (last + stride) * 1000.0 / source_fps
        if duration_ms:
            processed_ms = min(processed_ms, float(duration_ms))
    return {
        "render_id": meta.get("render_id"),
        "status": status,
        "complete": status == "complete",
        "source_fps": source_fps,
        "stride": stride,
        "duration_ms": duration_ms,
        "processed_ms": round(processed_ms, 3),
        "last_frame": last,
    }


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
         "-b:v", "0", "-maxrate", "8M", "-bufsize", "16M", "-forced-idr", "1"),
    ),
    ("h264_qsv", ("-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "27")),
    ("libx264", ("-c:v", "libx264", "-preset", "veryfast", "-crf", "23")),
)
SOFTWARE_ENCODER = _ENCODERS[-1]
# Every segment starts on a keyframe so it decodes on its own, which is what lets
# the page start or seek anywhere. No B-frames: they give fragments a composition
# offset that Media Source and the segment timing would each have to allow for.
KEYFRAME_ARGS = ("-bf", "0", "-force_key_frames", f"expr:gte(t,n_forced*{SEGMENT_SECONDS})")


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
                "-frames:v", "2", *args, *KEYFRAME_ARGS,
                "-pix_fmt", "yuv420p", "-f", "null", "-",
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


def settle_render(directory: Path) -> str | None:
    """Leave a render that stopped before the end in a state it can continue from.

    What it finished is kept and marked partial, with the frame to continue at.
    If it finished nothing, the files it wrote go and so does its folder, so a
    cancel in the first seconds leaves no empty folder in output/renders. Returns
    the status left on disk, or None when nothing was left.
    """
    meta = read_meta(directory)
    if meta is not None and meta.get("status") == "complete":
        return "complete"
    last = last_frame(directory / TRACKS_FILE)
    if meta is None or last is None:
        _clear_render(directory)
        with suppress(OSError):
            directory.rmdir()
        return None
    next_frame = last + 1
    if meta.get("status") != "partial" or meta.get("next_frame") != next_frame:
        _write_json(
            directory / META_FILE,
            {**meta, "status": "partial", "next_frame": next_frame, "updated_at": _now()},
        )
    return "partial"


def run_render(
    *,
    video: str,
    output_dir: str,
    settings: dict[str, Any],
    meta: dict[str, Any],
    detector: Any,
    publish: Callable[[dict[str, Any]], None],
    cancelled: Callable[[], bool],
    fresh: bool = False,
) -> None:
    """Render a video into ``output_dir``, continuing a partial render there if it can.

    It continues only a render made from the same files with the same settings;
    anything else in the folder is replaced, and ``fresh`` replaces it regardless.
    """
    import cv2

    options = RenderSettings(**settings)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous = read_meta(out)
    if fresh or previous is None or not _can_continue(previous, meta, options):
        _clear_render(out)
        previous = None

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        settle_render(out)
        raise RuntimeError(f"Cannot open video: {video}")

    started = time.perf_counter()
    # Where a frame's time goes, averaged over the frames tracked. Reported with
    # the progress, because "slow" has a different fix for each of these.
    spans = {"decode": 0.0, "track": 0.0}
    record: dict[str, Any] = {}
    index = -1
    kept = written = 0
    fps = 30.0
    was_cancelled = False
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not 0.0 < fps < 1000.0:
            fps = 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        runs = list((previous or {}).get("runs") or [])
        track_offset = 0
        if previous is not None:
            # The file, not the meta: a render that was killed never got to
            # record where it stopped, but every frame it tracked is in there.
            last = last_frame(out / TRACKS_FILE)
            start_frame = 0 if last is None else last + 1
            # The tracker starts again from 1 in a new run. Its ids are moved past
            # the ones already used, so two vehicles never share a number.
            kept, track_offset = _trim_tracks(out / TRACKS_FILE, start_frame)
        else:
            start_frame = 0
            (out / TRACKS_FILE).unlink(missing_ok=True)
        record = {
            **meta,
            "format": FORMAT,
            "status": "running",
            "settings": asdict(options),
            "source_fps": fps,
            "fps": fps / options.stride,
            "stride": options.stride,
            "source_frames": total,
            "width": width,
            "height": height,
            "duration_ms": round(total * 1000.0 / fps, 3),
            "runs": [
                *runs,
                {"start_frame": start_frame, "track_id_offset": track_offset},
            ],
            "next_frame": start_frame,
            "frames_written": kept,
            "elapsed_s": float((previous or {}).get("elapsed_s") or 0.0),
            "created_at": (previous or {}).get("created_at") or _now(),
            "updated_at": _now(),
            "tracks_file": TRACKS_FILE,
        }
        _write_json(out / META_FILE, record)

        if start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        index = start_frame - 1
        tracker = tracker_config(out, options.confidence)
        publish(
            {"kind": "started", "start_frame": start_frame, "run": len(runs), "total": total}
        )
        publish({"kind": "progress", "done": start_frame, "total": total, "fps": 0.0})
        last_published = time.perf_counter()
        with (out / TRACKS_FILE).open("a", encoding="utf-8") as tracks:
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
                    tracker=tracker,
                )
                boxes = [_plain(item, track_offset) for item in detections]
                spans["decode"] += decoded - step
                spans["track"] += time.perf_counter() - decoded
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
                # Per frame: a capture taken while this runs reads the file
                # for the boxes of the frame on screen.
                tracks.flush()
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
    except BaseException:
        if record:
            _account(record, started, spans, kept, written)
            with suppress(OSError):
                _write_json(out / META_FILE, record)
        settle_render(out)
        raise
    finally:
        capture.release()

    _account(record, started, spans, kept, written)
    if was_cancelled:
        _write_json(out / META_FILE, record)
        left = settle_render(out)
        publish({"kind": "cancelled", "kept": left == "partial"})
        return

    source_frames = index + 1
    record.update(
        status="complete",
        source_frames=source_frames,
        duration_ms=round(source_frames * 1000.0 / fps, 3),
        next_frame=source_frames,
    )
    if last_frame(out / TRACKS_FILE) is None:
        settle_render(out)
        raise RuntimeError("The video yielded no frames to render")
    _write_json(out / META_FILE, record)
    publish({"kind": "complete", "meta": record})


def _can_continue(previous: dict[str, Any], meta: dict[str, Any], options: RenderSettings) -> bool:
    return (
        render_format(previous) in {FORMAT, SEGMENTED_FORMAT}
        and previous.get("status") in {"running", "partial"}
        and previous.get("video_fingerprint") == meta.get("video_fingerprint")
        and previous.get("model_fingerprint") == meta.get("model_fingerprint")
        and previous.get("settings") == asdict(options)
    )


def _trim_tracks(path: Path, before_frame: int) -> tuple[int, int]:
    """Keep only the frames before one; return how many were kept and their highest track id.

    Anything from that frame on is tracked again when the render continues, so it
    goes rather than appearing twice -- as does a last line a killed render left
    half written.
    """
    if not path.is_file():
        return 0, 0
    kept = highest = 0
    temporary = path.with_name(path.name + ".tmp")
    with path.open("rb") as source, temporary.open("wb") as target:
        for line in source:
            match = _FRAME_INDEX.match(line)
            if not line.endswith(b"\n") or match is None or int(match.group(1)) >= before_frame:
                continue
            target.write(line)
            kept += 1
            for found in _TRACK_ID.findall(line):
                highest = max(highest, int(found))
    os.replace(temporary, path)
    return kept, highest


def _clear_render(directory: Path) -> None:
    """Remove every file a render writes, and nothing else, leaving the folder."""
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_dir():
            if _RUN_DIR.fullmatch(child.name):
                shutil.rmtree(child, ignore_errors=True)
        elif child.name in {
            TRACKS_FILE,
            META_FILE,
            TRACKER_FILE,
            LEGACY_VIDEO_FILE,
        } or child.suffix in {
            ".part",
            ".tmp",
        }:
            with suppress(OSError):
                child.unlink()


def _account(
    record: dict[str, Any], started: float, spans: dict[str, float], kept: int, written: int
) -> None:
    record.update(
        frames_written=kept + written,
        elapsed_s=round(float(record.get("elapsed_s") or 0.0) + time.perf_counter() - started, 2),
        stages_ms=_averages(spans, written),
        updated_at=_now(),
    )


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
        run_render(
            video=job["video"],
            output_dir=job["output_dir"],
            settings=job["settings"],
            meta=job["meta"],
            detector=detector,
            publish=publish,
            cancelled=cancel.is_set,
            fresh=bool(job.get("fresh")),
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
    """At most one render at a time, and the renders already on disk."""

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
        self._tracks: dict[str, TrackIndex] = {}

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
        fresh: bool = False,
    ) -> dict[str, Any]:
        """Start rendering, continuing a partial render of the same files if there is one."""
        settings.validate()
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
                "fresh": fresh,
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
                "start_frame": 0,
                "run": None,
                "kept": None,
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
                settle_render(self.directory(job["render_id"]))

    def renders(self, video_id: str | None = None) -> list[dict[str, Any]]:
        """Every render with something to play: finished, partial, or being made."""
        if not self.root.is_dir():
            return []
        with self._lock:
            active = self._active_render_locked()
        found = []
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            described = self._describe(directory, active)
            if described is None:
                continue
            if video_id is not None and described.get("video") != video_id:
                continue
            found.append(described)
        return found

    def tracks(
        self, render_id: str, from_ms: float = 0.0, to_ms: float | None = None
    ) -> dict[str, Any] | None:
        """How far a render has got, and the boxes of the frames between two times.

        A window rather than the whole file: a 25-minute render is several
        megabytes of boxes, and the page needs the next half-minute of them.
        """
        directory = self.directory(render_id)
        with self._lock:
            active = self._active_render_locked()
            job = self._job if active == render_id else None
            written_per_s = float(job.get("fps") or 0.0) if job is not None else 0.0
        meta = read_meta(directory)
        if meta is None or not (directory / TRACKS_FILE).is_file():
            return None
        state = render_state(directory, meta)
        state["render_id"] = render_id
        if state["status"] == "running" and render_id != active:
            state["status"] = "partial"
        # Seconds of video tracked per second, averaged over the run: what the
        # page works out how long to wait from.
        state["rate"] = (
            round(written_per_s * state["stride"] / state["source_fps"], 4)
            if state["status"] == "running"
            else None
        )
        fps = state["source_fps"]
        first = max(0, int(from_ms * fps / 1000.0))
        # One frame past the window: a box at its edge is placed toward it.
        last = int(to_ms * fps / 1000.0) + 1 if to_ms is not None else None
        state["from_frame"] = first
        state["frames"] = self._index(render_id, directory).between(first, last)
        return state

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
        """The boxes a render found on one source frame, or None if it has not got to it."""
        index = self._index(render_id, self.directory(render_id))
        try:
            return index.at(frame_index)
        except OSError:
            return None

    def _index(self, render_id: str, directory: Path) -> TrackIndex:
        with self._lock:
            index = self._tracks.get(render_id)
            if index is None:
                # Few renders are looked at in one sitting; keep the index of the
                # last couple rather than one per render ever opened.
                if len(self._tracks) >= 2:
                    self._tracks.pop(next(iter(self._tracks)))
                index = TrackIndex(directory / TRACKS_FILE)
                self._tracks[render_id] = index
            with suppress(OSError):
                index.refresh()
            return index

    # -- helpers ----------------------------------------------------------------

    def _describe(self, directory: Path, active: str | None) -> dict[str, Any] | None:
        meta = read_meta(directory)
        if meta is None or not (directory / TRACKS_FILE).is_file():
            return None
        status = str(meta.get("status") or "")
        if status == "running" and directory.name != active:
            # Left "running" by a render that did not get to say otherwise.
            status = "partial"
        if status not in {"complete", "partial", "running"}:
            return None
        state = render_state(directory, meta)
        if state["last_frame"] is None and status != "running":
            return None
        return {
            **meta,
            "format": render_format(meta),
            "status": status,
            "processed_ms": state["processed_ms"],
            # Everything in the folder, the video of older renders included: what
            # deleting it would give back.
            "size_bytes": _folder_size(directory),
        }

    def _active_render_locked(self) -> str | None:
        self._poll_locked()
        if self._job is not None and self._job["status"] in _ACTIVE:
            return str(self._job["render_id"])
        return None

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
            elif kind == "started":
                job["start_frame"] = int(message.get("start_frame", 0))
                job["run"] = message.get("run")
                job["total"] = int(message.get("total", 0))
            elif kind == "progress":
                if job["status"] == "loading":
                    job["status"] = "running"
                job["done"] = int(message.get("done", 0))
                job["total"] = int(message.get("total", 0))
                job["fps"] = float(message.get("fps", 0.0))
                if isinstance(message.get("stages_ms"), dict):
                    job["stages_ms"] = message["stages_ms"]
            elif kind == "complete":
                job["status"] = "complete"
                job["done"] = job["total"] = int(message["meta"].get("source_frames", job["total"]))
                job["meta"] = message.get("meta")
            elif kind == "cancelled":
                job["status"] = "cancelled"
                job["kept"] = bool(message.get("kept"))
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
                # The render settles itself, unless it died before it could.
                left = settle_render(self.directory(job["render_id"]))
                if job.get("kept") is None:
                    job["kept"] = left == "partial"

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
    """Byte offsets of each frame's line in tracks.jsonl, so a lookup reads one line.

    The file grows while a render runs, so refresh() indexes only what was added
    since the last look, and starts over when the file was replaced -- which a
    render continuing after a crash does when it drops the frames it redoes.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offsets: dict[int, int] = {}
        # The same frames in file order, which is frame order: what a range
        # lookup bisects.
        self._frames: list[int] = []
        self._end = 0
        self._identity: tuple[int, int] | None = None

    def refresh(self) -> None:
        stat = self._path.stat()
        identity = (int(stat.st_ino), int(stat.st_dev))
        if identity != self._identity or stat.st_size < self._end:
            self._offsets.clear()
            self._frames.clear()
            self._end = 0
            self._identity = identity
        if stat.st_size == self._end:
            return
        with self._path.open("rb") as stream:
            stream.seek(self._end)
            offset = self._end
            for line in stream:
                # A line still being written has no newline yet; it is read on
                # the next refresh, whole.
                if not line.endswith(b"\n"):
                    break
                # The frame index is the first key written; reading just that
                # much avoids parsing forty thousand detection lists up front.
                match = _FRAME_INDEX.match(line)
                if match:
                    frame = int(match.group(1))
                    self._offsets[frame] = offset
                    if not self._frames or frame > self._frames[-1]:
                        self._frames.append(frame)
                offset += len(line)
        self._end = offset

    def at(self, frame_index: int) -> list[dict[str, Any]] | None:
        offset = self._offsets.get(frame_index)
        if offset is None:
            return None
        with self._path.open("rb") as stream:
            stream.seek(offset)
            record = json.loads(stream.readline())
        detections = record.get("detections")
        return detections if isinstance(detections, list) else []

    def between(self, first: int, last: int | None = None) -> list[list[Any]]:
        """Every frame from ``first`` to ``last`` inclusive, as [frame, boxes], compactly.

        A box is [x1, y1, x2, y2, confidence, class_id, track_id, label]: the page
        draws thousands of them, and keys repeated on each would be most of the
        bytes.
        """
        start = bisect.bisect_left(self._frames, first)
        stop = len(self._frames) if last is None else bisect.bisect_right(self._frames, last)
        if start >= stop:
            return []
        found: list[list[Any]] = []
        with self._path.open("rb") as stream:
            stream.seek(self._offsets[self._frames[start]])
            for _ in range(stop - start):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    break
                record = json.loads(line)
                found.append(
                    [
                        int(record["frame_index"]),
                        [
                            [
                                box["x1"], box["y1"], box["x2"], box["y2"],
                                box.get("confidence"), box.get("class_id"),
                                box.get("track_id"), box.get("label"),
                            ]
                            for box in record.get("detections") or []
                        ],
                    ]
                )
        return found


def _averages(spans: dict[str, float], frames: int) -> dict[str, float]:
    if not frames:
        return {}
    return {name: round(total * 1000.0 / frames, 1) for name, total in spans.items()}


def _plain(detection: Any, track_offset: int = 0) -> dict[str, Any]:
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
        "track_id": None if track_id is None else int(track_id) + track_offset,
    }


def with_track_ids(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labelled = []
    for box in boxes:
        if box.get("track_id") is None:
            labelled.append(box)
        else:
            labelled.append({**box, "label": f"{box['label']} #{box['track_id']}"})
    return labelled


def _folder_size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        with suppress(OSError):
            if path.is_file():
                total += int(path.stat().st_size)
    return total


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe(value: str) -> str:
    return _UNSAFE.sub("-", value).strip("-.") or "x"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
