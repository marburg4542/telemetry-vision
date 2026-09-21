"""Choose frames from this project's own flights to label, and export them with draft labels.

The checkpoints were trained on VisDrone and a Roboflow set that is not this
camera. On this camera's footage they miss most parked motorcycles, now and then
call a bush a motorcycle, and give a small vehicle a second number when its score
flickers. The fix is more training data from this camera -- and the renders on
disk already say where the model struggles, so the frames can be chosen rather
than sampled blindly:

  flagged     frames an operator marked from a capture ("the model got this
              wrong", "the model missed something") -- all of them
  motorcycle  frames with a motorcycle in them, the class the model is weakest on
  relabel     frames where a vehicle got a second number: a new track of a class
              starting close to where one of the same class vanished moments
              before -- the model lost it, which is what uncertain looks like
  coverage    frames spread evenly through every flight, including empty ones:
              what the model misses entirely is only found by looking

Each frame is written with a draft label from the render's own tracks.jsonl, in
the YOLOv8 layout Roboflow imports with its splits: correcting a draft is
several times faster than drawing every box. Flights named as held out go to
the test split whole, so frames of one flight never sit on both sides of it.

Imported by the CLI only; like render_export it needs OpenCV and nothing else.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .render_export import boxes_at
from .render_job import TRACKS_FILE, read_meta

_LOG = logging.getLogger(__name__)

FEEDBACK_FILE = "feedback.jsonl"
REASONS = ("flagged", "motorcycle", "relabel", "coverage")
# Shares of the frames asked for, after every flagged frame is taken.
SHARES = {"motorcycle": 0.35, "relabel": 0.20}
RELABEL_WINDOW_MS = 3000.0
RELABEL_SIZES = 1.5


@dataclass(slots=True)
class Pick:
    video: str
    frame_index: int
    reason: str
    boxes: list[dict[str, Any]]
    split: str = "train"
    note: str = ""


@dataclass(slots=True)
class Flight:
    """One video and the render its draft labels come from."""

    video: str
    path: Path
    render_dir: Path
    fps: float
    stride: int
    frames: list[int] = field(default_factory=list)
    boxes: dict[int, list[dict[str, Any]]] = field(default_factory=dict)


def load_flights(renders_root: Path, videos: dict[str, Path]) -> list[Flight]:
    """Each video that has a render, with that render's tracks.

    The newest render of a video is used: renders are made with better models
    and settings as the project goes on, and its boxes are the better draft.
    """
    newest: dict[str, tuple[str, Path, dict[str, Any]]] = {}
    for directory in sorted(Path(renders_root).glob("*")):
        meta = read_meta(directory)
        if meta is None or not (directory / TRACKS_FILE).is_file():
            continue
        video = str(meta.get("video") or "")
        if video not in videos:
            continue
        stamp = str(meta.get("updated_at") or meta.get("created_at") or "")
        if video not in newest or stamp > newest[video][0]:
            newest[video] = (stamp, directory, meta)
    flights = []
    for video, (_, directory, meta) in sorted(newest.items()):
        flight = Flight(
            video=video,
            path=videos[video],
            render_dir=directory,
            fps=float(meta.get("source_fps") or 30.0),
            stride=int(meta.get("stride") or 1),
        )
        with (directory / TRACKS_FILE).open("r", encoding="utf-8") as tracks:
            for line in tracks:
                if not line.endswith("\n"):
                    break
                record = json.loads(line)
                index = int(record["frame_index"])
                flight.frames.append(index)
                flight.boxes[index] = list(record.get("detections") or [])
        if flight.frames:
            flights.append(flight)
    return flights


def load_feedback(sessions_root: Path) -> list[dict[str, Any]]:
    """Every capture an operator marked, newest mark per shot."""
    marks: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(Path(sessions_root).glob(f"*/{FEEDBACK_FILE}")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            marks[(path.parent.name, int(record.get("shot", -1)))] = record
    return [record for record in marks.values() if record.get("kind") in {"wrong", "missed"}]


def select(
    flights: list[Flight],
    feedback: Iterable[dict[str, Any]],
    *,
    count: int,
    holdout: set[str],
    test_count: int,
    spacing_s: float = 2.0,
) -> list[Pick]:
    """The frames to label, flagged ones first, then by share, spaced apart."""
    picks: list[Pick] = []
    taken: dict[str, list[int]] = defaultdict(list)
    by_video = {flight.video: flight for flight in flights}

    def free(flight: Flight, frame: int) -> bool:
        gap = spacing_s * flight.fps
        return all(abs(frame - other) >= gap for other in taken[flight.video])

    def take(flight: Flight, frame: int, reason: str, note: str = "") -> None:
        taken[flight.video].append(frame)
        picks.append(
            Pick(
                video=flight.video,
                frame_index=frame,
                reason=reason,
                boxes=boxes_at(frame, flight.frames, flight.boxes, flight.stride),
                split="test" if flight.video in holdout else "train",
                note=note,
            )
        )

    for mark in feedback:
        flight = by_video.get(str(mark.get("video", "")))
        if flight is None:
            continue
        frame = round(float(mark.get("source_time_ms", 0.0)) * flight.fps / 1000.0)
        take(flight, frame, "flagged", str(mark.get("kind", "")))

    for split, total in (("train", count), ("test", test_count)):
        group = [flight for flight in flights if (flight.video in holdout) == (split == "test")]
        if not group or total <= 0:
            continue
        already = sum(1 for pick in picks if pick.split == split)
        left = max(0, total - already)
        for reason in ("motorcycle", "relabel"):
            quota = round(left * SHARES[reason])
            candidates = [
                (flight, frame)
                for flight in group
                for frame in (
                    _motorcycle_frames(flight)
                    if reason == "motorcycle"
                    else _relabel_frames(flight)
                )
            ]
            _take_evenly(candidates, quota, free, take, reason)
        remaining = max(0, total - sum(1 for pick in picks if pick.split == split))
        coverage = [
            (flight, frame)
            for flight in group
            for frame in flight.frames[:: max(1, int(flight.fps * spacing_s / flight.stride))]
        ]
        _take_evenly(coverage, remaining, free, take, "coverage")
    return picks


def _take_evenly(
    candidates: list[tuple[Flight, int]],
    quota: int,
    free: Callable[[Flight, int], bool],
    take: Callable[[Flight, int, str], None],
    reason: str,
) -> None:
    """Up to `quota` of the candidates, spread through them rather than bunched.

    First every step-th one; then, when spacing ruled some of those out, the
    free ones left in order, so a quota is met whenever there is room for it.
    """
    if quota <= 0 or not candidates:
        return
    step = max(1.0, len(candidates) / quota)
    chosen = 0
    tried: set[int] = set()
    for slot in range(quota):
        index = int(slot * step)
        if index >= len(candidates):
            break
        tried.add(index)
        flight, frame = candidates[index]
        if free(flight, frame):
            take(flight, frame, reason)
            chosen += 1
    for index, (flight, frame) in enumerate(candidates):
        if chosen >= quota:
            break
        if index not in tried and free(flight, frame):
            take(flight, frame, reason)
            chosen += 1


def _motorcycle_frames(flight: Flight) -> list[int]:
    return [
        frame
        for frame in flight.frames
        if any(box.get("label") == "motorcycle" for box in flight.boxes[frame])
    ]


def _relabel_frames(flight: Flight) -> list[int]:
    """Frames where a track starts close to where one of its class just vanished."""
    first: dict[int, tuple[int, dict[str, Any]]] = {}
    last: dict[int, tuple[int, dict[str, Any]]] = {}
    for frame in flight.frames:
        for box in flight.boxes[frame]:
            track = box.get("track_id")
            if track is None:
                continue
            first.setdefault(int(track), (frame, box))
            last[int(track)] = (frame, box)
    window = RELABEL_WINDOW_MS * flight.fps / 1000.0
    found = []
    for track, (start, box) in first.items():
        for other, (end, other_box) in last.items():
            if other == track or other_box.get("label") != box.get("label"):
                continue
            if not 0 < start - end <= window:
                continue
            size = max(
                float(other_box["x2"]) - float(other_box["x1"]),
                float(other_box["y2"]) - float(other_box["y1"]),
                1.0,
            )
            (x, y), (ox, oy) = _centre(box), _centre(other_box)
            if ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 <= RELABEL_SIZES * size:
                found.append(start)
                break
    return sorted(found)


def class_names(flights: list[Flight]) -> list[str]:
    """The class names by id, as the render's model numbered them."""
    names: dict[int, str] = {}
    for flight in flights:
        for boxes in flight.boxes.values():
            for box in boxes:
                names.setdefault(int(box.get("class_id", 0)), str(box.get("label", "")))
    if not names:
        return []
    return [names.get(index, f"class{index}") for index in range(max(names) + 1)]


def export(
    picks: list[Pick], flights: list[Flight], output: Path, names: list[str]
) -> dict[str, Any]:
    """Write the frames and their draft labels as a YOLOv8 dataset Roboflow imports."""
    import cv2

    output = Path(output)
    by_video = {flight.video: flight for flight in flights}
    ids = {name: index for index, name in enumerate(names)}
    rows = []
    per_video: dict[str, list[Pick]] = defaultdict(list)
    for pick in picks:
        per_video[pick.video].append(pick)
    for video, wanted in sorted(per_video.items()):
        flight = by_video[video]
        wanted.sort(key=lambda item: item.frame_index)
        capture = cv2.VideoCapture(str(flight.path))
        try:
            # Read forward rather than seeking: a seek in OpenCV can land a
            # frame off, and a label drawn on the neighbour of its frame is a
            # wrong label.
            position = -1
            frame = None
            for pick in wanted:
                while position < pick.frame_index:
                    if not capture.grab():
                        break
                    position += 1
                if position != pick.frame_index:
                    _LOG.warning("%s has no frame %d", video, pick.frame_index)
                    continue
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    continue
                height, width = frame.shape[:2]
                stem = f"{_safe(Path(video).stem)}__f{pick.frame_index:06d}"
                images = output / pick.split / "images"
                labels = output / pick.split / "labels"
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(images / f"{stem}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                lines = [
                    _yolo_line(box, ids, width, height)
                    for box in pick.boxes
                    if str(box.get("label", "")) in ids
                ]
                (labels / f"{stem}.txt").write_text(
                    "".join(line + "\n" for line in lines), encoding="utf-8"
                )
                rows.append(
                    {
                        "file": f"{pick.split}/images/{stem}.jpg",
                        "video": video,
                        "frame_index": pick.frame_index,
                        "time_s": round(pick.frame_index / flight.fps, 2),
                        "reason": pick.reason,
                        "note": pick.note,
                        "split": pick.split,
                        "draft_boxes": len(lines),
                    }
                )
        finally:
            capture.release()
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
        "# Draft labels from this project's renders. Correct them before training.\n"
        "train: train/images\n"
        "val: test/images\n"
        "test: test/images\n"
        f"nc: {len(names)}\n"
        f"names: {json.dumps(names)}\n",
        encoding="utf-8",
    )
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as sheet:
        writer = csv.DictWriter(
            sheet,
            fieldnames=[
                "file", "video", "frame_index", "time_s", "reason", "note", "split", "draft_boxes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{row['split']}:{row['reason']}"] += 1
    return {"frames": len(rows), "by": dict(sorted(counts.items())), "names": names}


def _yolo_line(box: dict[str, Any], ids: dict[str, int], width: int, height: int) -> str:
    x1 = min(max(float(box["x1"]), 0.0), width)
    x2 = min(max(float(box["x2"]), 0.0), width)
    y1 = min(max(float(box["y1"]), 0.0), height)
    y2 = min(max(float(box["y2"]), 0.0), height)
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return (
        f"{ids[str(box['label'])]} {cx:.6f} {cy:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def _centre(box: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(box["x1"]) + float(box["x2"])) / 2.0,
        (float(box["y1"]) + float(box["y2"])) / 2.0,
    )


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in value)
