"""Run a whole video through a detector once, and let the page watch it as it goes.

The page used to post a frame to the detector every hundred milliseconds or so
and draw the answer over the playing video. Two things were wrong with that and
neither was the drawing. BoT-SORT was fed whatever frames the GPU had time for,
ten to sixteen a second out of thirty and never evenly spaced, so it lost
identities whenever the camera moved; and every box was already a frame or
three old by the time it was painted over a picture that had moved on.

Here the tracker sees every frame of the file (or every second or third, evenly
spaced), in order, once. The boxes are drawn into the frames themselves and
encoded as H.264, so what is on screen is exactly what the model saw at that
instant. The per-frame detections are kept beside it in tracks.jsonl, which is
what a capture reads its boxes from rather than asking the model again.

The video is written as two-second fragmented-MP4 segments under an HLS playlist
that ffmpeg rewrites as each segment finishes, not as one MP4 whose index only
exists once the last frame is in. The page feeds finished segments to Media
Source as they appear, so a render is watched while it is being made; neither
Chrome nor Edge plays a still-growing HLS playlist on its own. A render that
stops early keeps the segments it finished and continues later in a new run
folder: continuing ffmpeg's own playlist rewrites the init segment and shifts
the timestamps of everything already written.

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

TRACKS_FILE = "tracks.jsonl"
META_FILE = "meta.json"
PLAYLIST_FILE = "index.m3u8"
INIT_FILE = "init.mp4"
# The layout of renders made before segments: one MP4, playable once finished.
# Still listed and played, so nothing rendered that way has to be made again.
LEGACY_VIDEO_FILE = "annotated.mp4"
FORMAT = "hls"
LEGACY_FORMAT = "mp4"
SEGMENT_SECONDS = 2

_PROGRESS_INTERVAL_S = 0.5
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_RENDER_ID = re.compile(r"[A-Za-z0-9_.-]{1,200}")
_RUN_DIR = re.compile(r"run-(\d{3})")
_SEGMENT_NAME = re.compile(r"seg_\d{5}\.m4s")
_FRAME_INDEX = re.compile(rb'\{"frame_index": (\d+)')
_TRACK_ID = re.compile(rb'"track_id": (\d+)')
_ACTIVE = frozenset({"loading", "running", "cancelling"})

# Offered in the page. The trained sizes of the checkpoints in use, plus 1280 for
# anyone who trains larger; anything else is a typo rather than a choice.
IMAGE_SIZES = (416, 640, 960, 1280)
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


def read_playlist(path: Path) -> list[tuple[str, float]]:
    """The finished segments an HLS playlist lists, in order, with their lengths.

    ffmpeg writes a segment to a .tmp file, renames it once it is whole, and only
    then rewrites the playlist to name it, so the playlist never lists half of
    one. A read that lands on that rename on Windows fails and returns nothing;
    the page asks again a moment later.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    segments: list[tuple[str, float]] = []
    duration: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#EXTINF:"):
            try:
                duration = float(line[len("#EXTINF:"):].split(",", 1)[0])
            except ValueError:
                duration = None
        elif line and not line.startswith("#"):
            if duration is not None and _SEGMENT_NAME.fullmatch(line):
                segments.append((line, duration))
            duration = None
    return segments


def stream_index(directory: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """What the page needs to play a render: its runs and their finished segments.

    Times are on the source video's timeline, so the rendered video and the
    original switch at the same moment. A run starts at the source frame it began
    from; the timestamps inside its segments start at zero, and the page offsets
    them by the run's start.
    """
    source_fps = float(meta.get("source_fps") or 0.0) or 30.0
    runs: list[dict[str, Any]] = []
    processed_ms = 0.0
    for record in meta.get("runs") or []:
        name = str(record.get("dir", ""))
        if not _RUN_DIR.fullmatch(name):
            continue
        start_ms = int(record.get("start_frame", 0)) * 1000.0 / source_fps
        elapsed = 0.0
        segments = []
        for file, duration in read_playlist(directory / name / PLAYLIST_FILE):
            segments.append(
                {
                    "file": file,
                    "start_ms": round(start_ms + elapsed * 1000.0, 3),
                    "duration_ms": round(duration * 1000.0, 3),
                }
            )
            elapsed += duration
        if segments:
            runs.append({"run": name, "start_ms": round(start_ms, 3), "segments": segments})
            processed_ms = max(processed_ms, start_ms + elapsed * 1000.0)
    status = str(meta.get("status") or "")
    return {
        "render_id": meta.get("render_id"),
        "format": FORMAT,
        "status": status,
        "complete": status == "complete",
        "source_fps": source_fps,
        "stride": int(meta.get("stride") or 1),
        "duration_ms": meta.get("duration_ms"),
        "processed_ms": round(processed_ms, 3),
        "runs": runs,
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
    if meta is None or render_format(meta) != FORMAT or not stream_index(directory, meta)["runs"]:
        _clear_render(directory)
        with suppress(OSError):
            directory.rmdir()
        return None
    next_frame = _next_frame(directory, meta)
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
    encoder_factory: Callable[[Path, int, int, float], Any],
    publish: Callable[[dict[str, Any]], None],
    cancelled: Callable[[], bool],
    fresh: bool = False,
) -> None:
    """Render a video into ``output_dir``, continuing a partial render there if it can.

    It continues only a render made from the same files with the same settings;
    anything else in the folder is replaced, and ``fresh`` replaces it regardless.
    """
    import cv2

    from .annotate import annotate_frame

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
    # Where a frame's time goes, averaged over the frames written. Reported with
    # the progress, because "slow" has a different fix for each of these.
    spans = {"decode": 0.0, "track": 0.0, "draw": 0.0, "encode": 0.0}
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
        # yuv420p has half-resolution chroma, so x264 refuses odd dimensions.
        out_width, out_height = width - width % 2, height - height % 2
        out_fps = fps / options.stride

        runs = _finished_runs(out, previous)
        track_offset = 0
        if previous is not None and runs:
            start_frame = _next_frame(out, {**previous, "runs": runs})
            # The tracker starts again from 1 in a new run. Its ids are moved past
            # the ones already used, so two vehicles never share a number.
            kept, track_offset = _trim_tracks(out / TRACKS_FILE, start_frame)
        else:
            start_frame = 0
            (out / TRACKS_FILE).unlink(missing_ok=True)
        run_number = max((_run_number(run["dir"]) for run in runs), default=-1) + 1
        run_dir = out / f"run-{run_number:03d}"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir()
        record = {
            **meta,
            "format": FORMAT,
            "status": "running",
            "settings": asdict(options),
            "source_fps": fps,
            "fps": out_fps,
            "stride": options.stride,
            "source_frames": total,
            "width": out_width,
            "height": out_height,
            "duration_ms": round(total * 1000.0 / fps, 3),
            "runs": [
                *runs,
                {"dir": run_dir.name, "start_frame": start_frame, "track_id_offset": track_offset},
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
        encoder = encoder_factory(run_dir, out_width, out_height, out_fps)
        publish(
            {"kind": "started", "start_frame": start_frame, "run": run_dir.name, "total": total}
        )
        publish({"kind": "progress", "done": start_frame, "total": total, "fps": 0.0})
        last_published = time.perf_counter()
        encoder_failed = False
        try:
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
                        tracker="botsort.yaml",
                    )
                    tracked = time.perf_counter()
                    boxes = [_plain(item, track_offset) for item in detections]
                    marked = annotate_frame(frame, with_track_ids(boxes), width, height)
                    if (out_width, out_height) != (width, height):
                        marked = marked[:out_height, :out_width]
                    drawn = time.perf_counter()
                    # Blocks when ffmpeg falls behind, so this also measures
                    # whether the encoder is keeping up.
                    try:
                        encoder.write(marked)
                    except BaseException:
                        encoder_failed = True
                        raise
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
        finally:
            # Closed normally even when the model failed, so ffmpeg finishes the
            # segment it holds and every frame tracked so far stays watchable.
            encoder.close(abort=encoder_failed)
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
        runs=_finished_runs(out, record),
    )
    if not record["runs"]:
        settle_render(out)
        raise RuntimeError("The video yielded no frames to render")
    _write_json(out / META_FILE, record)
    publish({"kind": "complete", "meta": record})


def _can_continue(previous: dict[str, Any], meta: dict[str, Any], options: RenderSettings) -> bool:
    return (
        render_format(previous) == FORMAT
        and previous.get("status") in {"running", "partial"}
        and previous.get("video_fingerprint") == meta.get("video_fingerprint")
        and previous.get("model_fingerprint") == meta.get("model_fingerprint")
        and previous.get("settings") == asdict(options)
    )


def _finished_runs(directory: Path, meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The runs that finished at least one segment.

    A run that stopped before its first segment added nothing to keep, so its
    folder goes and the next run takes its number.
    """
    runs = []
    for run in (meta or {}).get("runs") or []:
        name = str(run.get("dir", ""))
        if not _RUN_DIR.fullmatch(name):
            continue
        if read_playlist(directory / name / PLAYLIST_FILE):
            runs.append(run)
        else:
            shutil.rmtree(directory / name, ignore_errors=True)
    return runs


def _next_frame(directory: Path, meta: dict[str, Any]) -> int:
    """The first source frame that no finished segment holds: where a render continues."""
    runs = meta.get("runs") or []
    if not runs:
        return 0
    last = runs[-1]
    out_fps = float(meta.get("fps") or 0.0)
    stride = int(meta.get("stride") or 1)
    playlist = directory / str(last.get("dir", "")) / PLAYLIST_FILE
    seconds = sum(duration for _, duration in read_playlist(playlist))
    return int(last.get("start_frame", 0)) + round(seconds * out_fps) * stride


def _trim_tracks(path: Path, before_frame: int) -> tuple[int, int]:
    """Keep only the frames before one; return how many were kept and their highest track id.

    A render that was killed can have recorded frames whose segment it never
    finished. Those frames are rendered again when it continues, so their lines
    go rather than appearing twice.
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
        elif child.name in {TRACKS_FILE, META_FILE, LEGACY_VIDEO_FILE} or child.suffix in {
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


def _run_number(name: str) -> int:
    match = _RUN_DIR.fullmatch(name)
    return int(match.group(1)) if match else -1


class FfmpegEncoder:
    """Raw BGR frames in on stdin; two-second fMP4 segments and their playlist out.

    ffmpeg runs inside the run folder, so the playlist names the init segment
    and every media segment by bare file name.
    """

    def __init__(
        self,
        ffmpeg: str,
        run_dir: Path,
        width: int,
        height: int,
        fps: float,
        codec: tuple[str, ...] = SOFTWARE_ENCODER[1],
    ) -> None:
        # A file rather than a pipe: nothing reads ffmpeg's stderr while frames
        # are going in, and a full pipe would stall the encoder mid-render.
        self._log = tempfile.TemporaryFile()  # noqa: SIM115 - closed in close()
        self._process = subprocess.Popen(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
                "-an", *codec, *KEYFRAME_ARGS,
                "-pix_fmt", "yuv420p",
                "-f", "hls",
                "-hls_time", str(SEGMENT_SECONDS),
                "-hls_playlist_type", "event",
                "-hls_segment_type", "fmp4",
                "-hls_fmp4_init_filename", INIT_FILE,
                "-hls_segment_filename", "seg_%05d.m4s",
                # Each segment goes to a .tmp file renamed when whole, and is
                # listed only after that, so a reader never sees half of one.
                "-hls_flags", "independent_segments+temp_file",
                PLAYLIST_FILE,
            ],
            cwd=str(run_dir),
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
            raise EncoderFailed(self._failure("ffmpeg stopped accepting frames")) from exc

    def close(self, abort: bool = False) -> None:
        """Finish the last segment and the playlist, or with ``abort`` just stop."""
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
        if not abort and code != 0:
            raise EncoderFailed(failure)

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
            encoder_factory=lambda run_dir, width, height, fps: FfmpegEncoder(
                ffmpeg, run_dir, width, height, fps, codec
            ),
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

    def stream(self, render_id: str) -> dict[str, Any] | None:
        """The segments of a render to play, or how to play a render made before segments."""
        directory = self.directory(render_id)
        with self._lock:
            active = self._active_render_locked()
        meta = read_meta(directory)
        if meta is None:
            return None
        if render_format(meta) == LEGACY_FORMAT:
            if meta.get("status") != "complete" or not (directory / LEGACY_VIDEO_FILE).is_file():
                return None
            return {
                "render_id": render_id,
                "format": LEGACY_FORMAT,
                "status": "complete",
                "complete": True,
                "duration_ms": meta.get("duration_ms"),
                "processed_ms": meta.get("duration_ms"),
                "runs": [],
            }
        index = stream_index(directory, meta)
        index["render_id"] = render_id
        if index["status"] == "running" and render_id != active:
            index["status"] = "partial"
        return index

    def segment_path(self, render_id: str, run: str, file: str) -> Path | None:
        """One file of one run, if the names are ones a render writes and it exists."""
        if not _RUN_DIR.fullmatch(run):
            return None
        if file != INIT_FILE and not _SEGMENT_NAME.fullmatch(file):
            return None
        path = self.directory(render_id) / run / file
        return path if path.is_file() else None

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
        """The boxes a render drew on one source frame, or None if it has not got to it."""
        path = self.directory(render_id) / TRACKS_FILE
        with self._lock:
            index = self._tracks.get(render_id)
            if index is None:
                # Few renders are looked at in one sitting; keep the index of the
                # last couple rather than one per render ever opened.
                if len(self._tracks) >= 2:
                    self._tracks.pop(next(iter(self._tracks)))
                index = TrackIndex(path)
                self._tracks[render_id] = index
            try:
                index.refresh()
                return index.at(frame_index)
            except OSError:
                return None

    # -- helpers ----------------------------------------------------------------

    def _describe(self, directory: Path, active: str | None) -> dict[str, Any] | None:
        meta = read_meta(directory)
        if meta is None:
            return None
        if render_format(meta) == LEGACY_FORMAT:
            if meta.get("status") != "complete" or not (directory / LEGACY_VIDEO_FILE).is_file():
                return None
            return {**meta, "format": LEGACY_FORMAT, "processed_ms": meta.get("duration_ms")}
        status = str(meta.get("status") or "")
        if status == "running" and directory.name != active:
            # Left "running" by a render that did not get to say otherwise.
            status = "partial"
        if status not in {"complete", "partial", "running"}:
            return None
        index = stream_index(directory, meta)
        if not index["runs"] and status != "running":
            return None
        return {**meta, "status": status, "processed_ms": index["processed_ms"]}

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
            elif kind == "encoder":
                job["encoder"] = str(message.get("name") or "")
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
        self._end = 0
        self._identity: tuple[int, int] | None = None

    def refresh(self) -> None:
        stat = self._path.stat()
        identity = (int(stat.st_ino), int(stat.st_dev))
        if identity != self._identity or stat.st_size < self._end:
            self._offsets.clear()
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
                    self._offsets[int(match.group(1))] = offset
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


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe(value: str) -> str:
    return _UNSAFE.sub("-", value).strip("-.") or "x"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
