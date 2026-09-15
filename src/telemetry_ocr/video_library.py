"""The flight videos the page can choose from, and single frames read out of them.

The page no longer opens a file from the browser. It picks from the project's
video folder, the same `asset_dirs.videos` the CLI resolves bare names against,
because the server has to read the file itself: to render it through the model,
and to pull the clean original frame a capture recognises. A browser file
picker never tells the server where a file is.

Videos are referred to by their path relative to that folder, and only ever as
an id matched against a listing this module made -- never opened because a
request spelled out a path.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# What the browser can be expected to play once served. The render output is
# always H.264 MP4, so this only limits which originals can be previewed.
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm"})


@dataclass(frozen=True, slots=True)
class VideoInfo:
    id: str
    name: str
    path: Path
    size_bytes: int
    fps: float
    frames: int
    width: int
    height: int
    duration_ms: float


class VideoLibrary:
    def __init__(self, directory: Path | None) -> None:
        self.directory = directory.resolve() if directory is not None else None
        self._lock = threading.Lock()
        # Probing opens the container, which for a 400 MB file is not free.
        self._probes: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}

    def videos(self) -> list[VideoInfo]:
        if self.directory is None or not self.directory.is_dir():
            return []
        listed = []
        for path in sorted(self.directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(self.directory):
                continue
            stat = resolved.stat()
            probe = self._probe(resolved, (stat.st_size, stat.st_mtime_ns))
            listed.append(
                VideoInfo(
                    id=resolved.relative_to(self.directory).as_posix(),
                    name=resolved.name,
                    path=resolved,
                    size_bytes=int(stat.st_size),
                    **probe,
                )
            )
        return listed

    def get(self, video_id: str) -> VideoInfo | None:
        return next((video for video in self.videos() if video.id == video_id), None)

    def _probe(self, path: Path, stamp: tuple[int, int]) -> dict[str, Any]:
        key = str(path)
        with self._lock:
            cached = self._probes.get(key)
            if cached is not None and cached[0] == stamp:
                return cached[1]
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()
        if not 0.0 < fps < 1000.0:
            fps = 30.0
        probe = {
            "fps": fps,
            "frames": frames,
            "width": width,
            "height": height,
            "duration_ms": round(frames * 1000.0 / fps, 3) if frames else 0.0,
        }
        with self._lock:
            self._probes[key] = (stamp, probe)
        return probe


class SourceFrames:
    """Reads one frame at a time from the original file a capture is taken from.

    Kept open between captures: holding Capture down asks for a frame every
    400 ms, usually the next few frames along, and reopening and seeking a
    1080p H.264 file each time would cost more than recognising the frame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._capture: Any = None
        self._next_index = 0

    def read(self, path: Path, frame_index: int) -> Any:
        import cv2

        with self._lock:
            if self._capture is None or self._path != path:
                self._release_locked()
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise ValueError(f"Cannot open video: {path.name}")
                self._capture = capture
                self._path = path
                self._next_index = 0
            if frame_index != self._next_index:
                # A short step forward is decoded rather than sought: a seek
                # lands on the keyframe before and decodes up anyway, and for a
                # few frames reading on is both cheaper and certain.
                gap = frame_index - self._next_index
                if 0 < gap <= 30:
                    for _ in range(gap):
                        if not self._capture.grab():
                            raise ValueError(f"Frame {frame_index} is past the end of the video")
                else:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = self._capture.read()
            if not ok or frame is None:
                self._next_index = 0
                self._release_locked()
                raise ValueError(f"Frame {frame_index} could not be read from the video")
            self._next_index = frame_index + 1
            return frame

    def close(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._path = None
