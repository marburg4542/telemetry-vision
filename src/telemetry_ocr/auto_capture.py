"""Capture, and send to LINE, the vehicles a render finds -- without anyone pressing a button.

While a render is being made its tracker writes one line per processed frame to
tracks.jsonl. The server reads the new lines as they land, and a Selector decides
which vehicles are worth a message:

* Only the classes asked for, each at its own confidence: a motorcycle may be
  worth reporting at 0.5 where a car is not below 0.7.
* Only one seen for at least `min_seconds` in a row (1 s by default). A box that
  flickers for a frame is far more often a false detection than a vehicle, and
  the second of watching also gives a choice of frames: the capture is taken of
  the one the model was surest about.
* Once per vehicle, even when the tracker gives it a second number. Measured on
  this project's flights, a small motorcycle's number changes a few times a
  minute; a new number of the same class that starts within `same_vehicle_s` of
  one already reported vanishing, and near where it vanished, is taken to be the
  same vehicle and not reported again.

What happens to a vehicle that qualifies -- the capture, the message, the quota
check -- is given to AutoLineRun by the web app, so this module does no OCR, no
drawing and no network, and the selection can be tested on its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

MAX_CLASSES = 20
MAX_EVENTS = 50
_DEFAULT_MIN_SECONDS = 1.0
_DEFAULT_RESERVE = 10


@dataclass(frozen=True, slots=True)
class AutoRules:
    # Label -> the confidence a detection needs to count. Only these are watched.
    classes: dict[str, float]
    min_seconds: float = _DEFAULT_MIN_SECONDS
    # A new number within this long of a reported one vanishing, and within
    # this many of that vehicle's box sizes of where it vanished, is the same one.
    same_vehicle_s: float = 3.0
    same_vehicle_sizes: float = 1.5
    # A run of sightings survives gaps this long -- a frame or two the model
    # missed -- before it counts as seen again from the start.
    max_gap_s: float = 0.5
    # False: capture only, send nothing. For trying the filter on a real flight
    # before spending the month's messages on it.
    send: bool = False
    # Messages left unspent for sending by hand: automatic sending stops here.
    reserve: int = _DEFAULT_RESERVE

    @classmethod
    def from_request(cls, body: Any) -> AutoRules:
        if not isinstance(body, dict):
            raise ValueError("auto must be an object")
        raw = body.get("classes")
        if not isinstance(raw, dict) or not raw:
            raise ValueError("auto.classes must name at least one class and its confidence")
        if len(raw) > MAX_CLASSES:
            raise ValueError(f"auto.classes takes at most {MAX_CLASSES} classes")
        classes: dict[str, float] = {}
        for label, threshold in raw.items():
            name = str(label).strip()
            if not name or len(name) > 40:
                raise ValueError("auto.classes has a class name that is empty or too long")
            try:
                value = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"auto confidence for {name!r} must be a number") from exc
            if not 0.05 <= value <= 1.0:
                raise ValueError(f"auto confidence for {name!r} must be between 0.05 and 1")
            classes[name] = round(value, 3)
        try:
            reserve = int(body.get("reserve", _DEFAULT_RESERVE))
            min_seconds = float(body.get("min_seconds", _DEFAULT_MIN_SECONDS))
        except (TypeError, ValueError) as exc:
            raise ValueError("auto.reserve and auto.min_seconds must be numbers") from exc
        if not 0 <= reserve <= 1000:
            raise ValueError("auto.reserve must be between 0 and 1000")
        if not 0.0 <= min_seconds <= 30.0:
            raise ValueError("auto.min_seconds must be between 0 and 30")
        return cls(
            classes=classes,
            min_seconds=min_seconds,
            send=bool(body.get("send", False)),
            reserve=reserve,
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """A vehicle to capture: the frame of its run the model was surest about."""

    track_id: int
    label: str
    confidence: float
    frame_index: int
    time_ms: float
    box: dict[str, Any]

    @property
    def note(self) -> str:
        return f"{self.label} #{self.track_id} {self.confidence:.2f}"


@dataclass(slots=True)
class _Track:
    label: str
    first_ms: float
    first_box: dict[str, Any]
    last_ms: float
    last_box: dict[str, Any]
    run_start_ms: float | None = None
    run_last_ms: float = 0.0
    best: tuple[float, int, float, dict[str, Any]] | None = None
    # Reported, or recognised as a vehicle that already was.
    settled: bool = False


class Selector:
    """Fed the render's frames in order; says which vehicles to capture."""

    def __init__(self, rules: AutoRules) -> None:
        self.rules = rules
        self._tracks: dict[int, _Track] = {}
        self.reported = 0
        self.same_vehicle = 0

    def feed(
        self, frame_index: int, time_ms: float, detections: list[dict[str, Any]]
    ) -> list[Candidate]:
        found: list[Candidate] = []
        for box in detections:
            track_id = box.get("track_id")
            label = str(box.get("label", ""))
            threshold = self.rules.classes.get(label)
            # Unnumbered boxes are a tracker's first sight of something; the
            # same thing comes back numbered a frame later if it is real.
            if track_id is None or threshold is None:
                continue
            track = self._tracks.get(int(track_id))
            if track is None:
                track = _Track(label, time_ms, box, time_ms, box)
                self._tracks[int(track_id)] = track
            track.last_ms, track.last_box = time_ms, box
            confidence = float(box.get("confidence") or 0.0)
            if track.settled or confidence < threshold:
                continue
            gap = time_ms - track.run_last_ms
            if track.run_start_ms is None or gap > self.rules.max_gap_s * 1000.0:
                track.run_start_ms = time_ms
                track.best = None
            track.run_last_ms = time_ms
            if track.best is None or confidence > track.best[0]:
                track.best = (confidence, frame_index, time_ms, box)
            if time_ms - track.run_start_ms + 1e-6 < self.rules.min_seconds * 1000.0:
                continue
            track.settled = True
            if self._seen_before(int(track_id), track):
                self.same_vehicle += 1
                continue
            self.reported += 1
            best_confidence, best_frame, best_time, best_box = track.best
            found.append(
                Candidate(
                    int(track_id), label, round(best_confidence, 3), best_frame, best_time, best_box
                )
            )
        return found

    def _seen_before(self, track_id: int, track: _Track) -> bool:
        """Whether this number is a vehicle already settled under another one."""
        window = self.rules.same_vehicle_s * 1000.0
        x, y = _centre(track.first_box)
        for other_id, other in self._tracks.items():
            if other_id == track_id or not other.settled or other.label != track.label:
                continue
            # Gone before this one appeared, and not long before: two numbers on
            # screen at once are two vehicles.
            if not 0.0 < track.first_ms - other.last_ms <= window:
                continue
            ox, oy = _centre(other.last_box)
            reach = self.rules.same_vehicle_sizes * _size(other.last_box)
            if ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 <= reach:
                return True
        return False


def _centre(box: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(box["x1"]) + float(box["x2"])) / 2.0,
        (float(box["y1"]) + float(box["y2"])) / 2.0,
    )


def _size(box: dict[str, Any]) -> float:
    return max(float(box["x2"]) - float(box["x1"]), float(box["y2"]) - float(box["y1"]), 1.0)


# ---------------------------------------------------------------------------
# Following one render as it is made.
# ---------------------------------------------------------------------------

Capture = Callable[[Candidate], Awaitable[Any]]
# Returns ("sent" | "skipped" | "failed", detail).
Send = Callable[[Any, Candidate], Awaitable[tuple[str, str]]]


@dataclass(slots=True)
class AutoEvent:
    at: str
    label: str
    track_id: int
    confidence: float
    time_ms: float
    shot: int | None
    outcome: str  # "captured" | "sent" | "skipped" | "failed"
    detail: str = ""


@dataclass(slots=True)
class _Counts:
    captured: int = 0
    sent: int = 0
    failed: int = 0


class AutoLineRun:
    """Reads a render's tracks as they are written, and acts on what qualifies."""

    def __init__(
        self,
        render_id: str,
        tracks_path: Path,
        rules: AutoRules,
        *,
        session: str,
        capture: Capture,
        send: Send,
        is_active: Callable[[], bool],
        poll_s: float = 1.0,
    ) -> None:
        self.render_id = render_id
        self.tracks_path = Path(tracks_path)
        self.rules = rules
        self.session = session
        self._capture = capture
        self._send = send
        self._is_active = is_active
        self._poll_s = poll_s
        self.selector = Selector(rules)
        self.counts = _Counts()
        self.events: list[AutoEvent] = []
        self.sending = rules.send
        self.stopped_sending = ""
        self.finished = False
        self.error = ""
        self._offset = 0

    async def run(self) -> None:
        try:
            while True:
                active = self._is_active()
                for candidate in await asyncio.to_thread(self._read_new):
                    await self._act(candidate)
                if not active:
                    # One last read above caught what the render wrote before it
                    # stopped; nothing more is coming.
                    break
                await asyncio.sleep(self._poll_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _LOG.exception("Automatic capture of %s stopped", self.render_id)
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.finished = True

    def _read_new(self) -> list[Candidate]:
        """The candidates in the lines added since the last look, whole lines only."""
        try:
            size = self.tracks_path.stat().st_size
        except OSError:
            return []
        if size < self._offset:
            # Rewritten from the start, as a render that continues after a crash
            # does when it drops the lines it redoes.
            self._offset = 0
        if size == self._offset:
            return []
        found: list[Candidate] = []
        with self.tracks_path.open("rb") as stream:
            stream.seek(self._offset)
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                self._offset += len(line)
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                found.extend(
                    self.selector.feed(
                        int(record.get("frame_index", 0)),
                        float(record.get("time_ms", 0.0)),
                        record.get("detections") or [],
                    )
                )
        return found

    async def _act(self, candidate: Candidate) -> None:
        event = AutoEvent(
            at=time.strftime("%H:%M:%S"),
            label=candidate.label,
            track_id=candidate.track_id,
            confidence=candidate.confidence,
            time_ms=candidate.time_ms,
            shot=None,
            outcome="captured",
        )
        self._remember(event)
        try:
            shot = await self._capture(candidate)
        except Exception as exc:
            _LOG.exception("Automatic capture of %s failed", candidate.note)
            event.outcome, event.detail = "failed", f"capture: {exc}"
            self.counts.failed += 1
            return
        event.shot = getattr(shot, "frame_index", None)
        self.counts.captured += 1
        if not self.sending:
            return
        outcome, detail = await self._send(shot, candidate)
        event.outcome, event.detail = outcome, detail
        if outcome == "sent":
            self.counts.sent += 1
        elif outcome == "failed":
            self.counts.failed += 1
        elif outcome == "skipped" and detail.startswith("quota"):
            # Captures carry on; the messages left are kept for sending by hand.
            self.sending = False
            self.stopped_sending = detail

    def _remember(self, event: AutoEvent) -> None:
        self.events.append(event)
        del self.events[:-MAX_EVENTS]

    def status(self) -> dict[str, Any]:
        return {
            "render_id": self.render_id,
            "session": self.session,
            "rules": asdict(self.rules),
            "active": not self.finished,
            "sending": self.sending,
            "stopped_sending": self.stopped_sending,
            "captured": self.counts.captured,
            "sent": self.counts.sent,
            "failed": self.counts.failed,
            "same_vehicle": self.selector.same_vehicle,
            "error": self.error,
            "events": [asdict(event) for event in reversed(self.events)],
        }

