"""Capture single frames on request, and keep them with what they read.

The rest of this application answers while the video plays and keeps nothing.
That is right for watching and wrong for recording: when an operator sees a value
worth keeping, the frame it came from is gone a fraction of a second later.

This is the other half. A shot writes the picture to disk and the telemetry
beside it, in the session format `extract` already understands, so a value can
always be checked against the exact frame it was read from.

Where the telemetry comes from depends on who is holding the video:

- a live stream is already being read by the server, so a shot reuses the
  reading it has just produced. Running OCR again would put a second thread
  inside PaddleOCR, which is the collision the stream lock exists to prevent,
  and would produce a slightly different answer for the same frame.
- a file playing in the browser is not being read by anyone until asked, so the
  posted frame is recognised here.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .capture import FRAME_DIR, MANIFEST_NAME, SESSION_NAME, FrameRecord, SessionInfo
from .errors import TelemetryOCRError

_LOG = logging.getLogger(__name__)

# Session names end up as a directory, so they are restricted rather than
# escaped: a name is chosen by whoever is holding the browser, and "../.." is a
# perfectly ordinary thing for a text field to contain.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_JPEG_QUALITY = 95

# How many times the verifier reads the same saved image. Five fills the widest
# temporal window in use (`stable` is 3 of 5), so a field is not left unconfirmed
# merely because the check ran out of samples rather than because the picture was
# unclear. On a manual capture the extra passes cost a fraction of a second.
# See verify_shot for why repeating one image is not double counting.
_VERIFY_PASSES = 5


@dataclass(slots=True)
class DetectionSummary:
    """What YOLO had to say about this frame, recorded when it was taken.

    Counted at capture time rather than when something later asks. A count taken
    on demand would describe whatever the detector happens to be looking at then,
    which for a shot reviewed minutes afterwards is a different scene entirely.

    `status` separates three things a bare count of zero cannot: no model loaded,
    a model that has not produced a result for this source yet, and a model that
    looked and found nothing. They are fixed in completely different ways.
    """

    status: str
    counts: dict[str, int] = field(default_factory=dict)
    # Per class: mean and lowest confidence across that class's boxes. The lowest
    # is the one that decides whether a count can be trusted -- three cars at
    # 0.9 and three cars where one sits at 0.31 are different claims, and a mean
    # alone hides which of the two this is.
    confidence: dict[str, dict[str, float]] = field(default_factory=dict)
    total: int = 0
    # How far behind the captured frame the detection result was. Detection runs
    # at its own pace, so this is rarely zero, and a large value is the signal
    # that the counts describe a different moment than the picture.
    age_ms: float | None = None


@dataclass(slots=True)
class VerifiedField:
    """One field re-read from the image that was actually written to disk."""

    field_id: str
    value: Any
    raw_text: str
    confidence: float
    confirmed: bool
    # Whether this matches what the live pipeline reported for the same frame.
    agrees: bool
    live_value: Any = None


@dataclass(slots=True)
class Verification:
    """A second reading, taken from the saved file rather than from memory.

    Worth doing because the live reading and the saved picture are not
    guaranteed to be the same frame: on a stream the telemetry published last and
    the frame decoded last can be one apart. Re-reading the file is the only way
    to say that the numbers describe *this* image.

    It is also the reading that actually belongs to the image. The live value is
    whatever temporal voting had last confirmed, which on a fast-moving field is
    deliberately behind what the current frame says: measured on a real stream,
    the OCR text matched on 10 of 11 fields while the published values differed
    on 8, because voting was still holding the previous confirmation. So for a
    capture -- a picture and the numbers describing it -- this reading is the
    authoritative one, and the live value is kept beside it for the fields where
    voting had not caught up.
    """

    status: str
    fields: list[VerifiedField] = field(default_factory=list)
    # Fields whose OCR text matched the live reading: the same picture.
    agreed: int = 0
    disagreed: int = 0
    # Fields where the text matched but the published value did not, because
    # temporal voting had not yet confirmed what this frame says and was still
    # publishing an older one. Not an error -- it is what voting is for -- but it
    # is the reason a capture's live telemetry can trail its own picture.
    lagging: int = 0
    duration_ms: float = 0.0


@dataclass(slots=True)
class ShotResult:
    session: str
    frame_index: int
    image: str
    source_time_ms: float
    captured_at: str
    bytes: int
    records: list[dict[str, Any]]
    detections: DetectionSummary = field(default_factory=lambda: DetectionSummary("unknown"))
    verification: Verification = field(default_factory=lambda: Verification("skipped"))


class CaptureSession:
    """One directory of shots, appended to as they are taken.

    Appends rather than rewrites: a session is open for as long as someone is
    watching, and losing an hour of captures because the last write failed would
    be a poor trade for a tidier file.
    """

    def __init__(
        self, root: Path, name: str, source: str, field_ids: list[str], is_stream: bool = False
    ) -> None:
        if not _SAFE_NAME.match(name):
            # ValueError rather than TelemetryOCRError: this is the caller's
            # mistake, and the web layer already turns a ValueError into 400.
            raise ValueError(
                "Session name may use letters, digits, dot, dash and underscore "
                f"only, up to 64 characters: {name!r}"
            )
        self.name = name
        self.directory = root / name
        self.frame_dir = self.directory / FRAME_DIR
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_index = self._resume_index()
        self.info = SessionInfo(
            source=source,
            is_stream=is_stream,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # Shots are taken by hand, so there is no rate to declare. Zero says
            # that plainly rather than implying a cadence nothing enforced.
            capture_hz=0.0,
            image_format="jpeg",
            config_fields=list(field_ids),
        )
        self.bytes_written = 0
        self.last_shot: ShotResult | None = None
        self._write_session()

    def _resume_index(self) -> int:
        """Continue numbering after whatever is already here.

        Reopening a session by name is the normal way to add to a recording
        after a page reload, and restarting at zero would overwrite the frames
        already in it while the manifest still described them.
        """
        existing = sorted(self.frame_dir.glob("*.jpg"))
        if not existing:
            return 0
        try:
            return int(existing[-1].stem) + 1
        except ValueError:
            return len(existing)

    @property
    def frame_count(self) -> int:
        return self._next_index

    def save(
        self,
        frame: Any,
        source_time_ms: float,
        records: list[Any],
        detections: DetectionSummary | None = None,
        verify: Any = None,
    ) -> ShotResult:
        import cv2

        with self._lock:
            index = self._next_index
            self._next_index += 1
        name = f"{index:08d}.jpg"
        path = self.frame_dir / name
        if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]):
            raise TelemetryOCRError(f"Could not write {path}")
        size = path.stat().st_size
        height, width = frame.shape[:2]
        relative = f"{FRAME_DIR}/{name}"
        captured_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        summary = detections or DetectionSummary("unknown")
        # Between writing the picture and writing the line that describes it, so
        # the manifest records the second reading in the same entry rather than
        # needing the file rewritten afterwards.
        shot_verification = Verification("skipped")
        if verify is not None:
            try:
                shot_verification = verify(path)
            except Exception:
                _LOG.exception("Could not verify %s", path)
                shot_verification = Verification("failed")
        entry = FrameRecord(
            frame_index=index,
            file=relative,
            source_time_ms=float(source_time_ms),
            captured_at=captured_at,
            width=int(width),
            height=int(height),
            bytes=int(size),
        )
        payload = [_record_payload(item, relative) for item in records]
        with self._lock:
            self.bytes_written += size
            self.info.frame_count = self._next_index
            # The manifest carries the counts alongside the frame, so a session
            # read back later still knows what was in the picture without having
            # to run the detector over it again.
            _append_json(
                self.directory / MANIFEST_NAME,
                {
                    **_as_dict(entry),
                    "detections": _as_dict(summary),
                    "verification": _as_dict(shot_verification),
                },
            )
            for item in payload:
                _append_json(self.directory / "telemetry.jsonl", item)
            self._write_session()

        shot = ShotResult(
            session=self.name,
            frame_index=index,
            image=relative,
            source_time_ms=float(source_time_ms),
            captured_at=captured_at,
            bytes=int(size),
            records=payload,
            detections=summary,
            verification=shot_verification,
        )
        with self._lock:
            # Remembered so sending it somewhere afterwards does not have to
            # re-read and re-parse the session to find the frame just taken.
            self.last_shot = shot
        return shot

    def record_verification(self, shot: ShotResult) -> None:
        """Append the second reading to the session's own telemetry.

        Written as its own lines rather than replacing the live ones. Both
        readings are evidence, and a record that quietly overwrote the first with
        the second would destroy the disagreement that makes the check useful.
        """
        if shot.verification.status != "ok" or not shot.verification.fields:
            return
        with self._lock:
            for item in shot.verification.fields:
                _append_json(
                    self.directory / "telemetry-verified.jsonl",
                    {**_as_dict(item), "image": shot.image, "frame_index": shot.frame_index},
                )

    def shot_at(self, frame_index: int) -> ShotResult | None:
        """Rebuild an earlier shot from what was written.

        Only the last one is held in memory; anything older is reconstructed
        from the manifest and the telemetry beside it, which is the same source
        `extract` reads and therefore cannot drift from it.
        """
        import json

        manifest = self.directory / MANIFEST_NAME
        telemetry = self.directory / "telemetry.jsonl"
        if not manifest.is_file():
            return None
        entry = None
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            candidate = json.loads(line)
            if int(candidate.get("frame_index", -1)) == frame_index:
                entry = candidate
                break
        if entry is None:
            return None
        records = []
        if telemetry.is_file():
            for line in telemetry.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("image") == entry["file"]:
                    records.append(record)
        raw = entry.get("detections") or {}
        raw_verify = entry.get("verification") or {}
        return ShotResult(
            session=self.name,
            frame_index=int(entry["frame_index"]),
            image=str(entry["file"]),
            source_time_ms=float(entry["source_time_ms"]),
            captured_at=str(entry["captured_at"]),
            bytes=int(entry["bytes"]),
            records=records,
            detections=DetectionSummary(
                status=str(raw.get("status", "unknown")),
                counts=dict(raw.get("counts") or {}),
                confidence=dict(raw.get("confidence") or {}),
                total=int(raw.get("total", 0)),
                age_ms=raw.get("age_ms"),
            ),
            verification=Verification(
                status=str(raw_verify.get("status", "skipped")),
                fields=[
                    VerifiedField(**item) for item in (raw_verify.get("fields") or [])
                ],
                agreed=int(raw_verify.get("agreed", 0)),
                disagreed=int(raw_verify.get("disagreed", 0)),
                lagging=int(raw_verify.get("lagging", 0)),
                duration_ms=float(raw_verify.get("duration_ms", 0.0)),
            ),
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session": self.name,
                "directory": str(self.directory),
                "frames": self._next_index,
                "bytes": self.bytes_written,
                "started_at": self.info.started_at,
                "source": self.info.source,
            }

    def _write_session(self) -> None:
        (self.directory / SESSION_NAME).write_text(
            _dumps(_as_dict(self.info)), encoding="utf-8"
        )


def verify_shot(
    image_path: Path, live_records: list[dict[str, Any]], pipeline: Any
) -> Verification:
    """Re-read the saved image and say where it agrees with the live reading.

    The pipeline handed in must be a fresh one. Reusing the pipeline that
    produced the live reading would feed this frame into the same temporal
    voting, which is the opposite of an independent check.
    """
    import time as _time

    import cv2

    started = _time.perf_counter()
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None or getattr(frame, "size", 0) == 0:
        return Verification("unreadable")

    from .pipeline import FramePacket

    # The same image, read _VERIFY_PASSES times.
    #
    # Not to gather evidence -- reading one picture repeatedly tells you nothing
    # new -- but because the pipeline only publishes a typed, validated value
    # once temporal voting has enough agreeing samples, and a `stable` field
    # needs three. A single pass leaves every value None, which would make every
    # field look like a disagreement and the whole check worthless.
    #
    # The vote is unanimous by construction, so what comes out is exactly what
    # this image says, with the same validation and typing the live path applies.
    produced: list[Any] = []
    for index in range(_VERIFY_PASSES):
        produced, _ = pipeline.process_frame(
            FramePacket(frame, index, index * 100.0, _time.perf_counter())
        )
    live = {item.get("field_id"): item for item in live_records}

    # Two different questions, and conflating them made the first run of this
    # look catastrophic when it was working.
    #
    # `agrees` compares the OCR text, which answers "is this the same picture the
    # live reading came from". `lagging` compares the published values, which
    # differ whenever temporal voting has not yet confirmed what this frame says
    # and is still publishing an older confirmed value -- by design, and common
    # on fields that move quickly. Measured on a real stream: 10 of 11 texts
    # identical while 8 of 11 values differed. Reporting that as disagreement
    # would have been badly wrong.
    fields: list[VerifiedField] = []
    agreed = disagreed = lagging = 0
    for record in produced:
        live_record = live.get(record.field_id) or {}
        live_text = str(live_record.get("raw_text") or "").strip()
        live_value = live_record.get("value")
        if live_text:
            if live_text == str(record.raw_text or "").strip():
                agreed += 1
                same_picture = True
            else:
                disagreed += 1
                same_picture = False
        else:
            same_picture = False
        if live_value is not None and not _same_value(record.value, live_value):
            lagging += 1
        fields.append(
            VerifiedField(
                field_id=record.field_id,
                value=record.value,
                raw_text=record.raw_text,
                confidence=record.confidence,
                confirmed=record.confirmed,
                agrees=same_picture,
                live_value=live_value,
            )
        )
    return Verification(
        status="ok",
        fields=fields,
        agreed=agreed,
        disagreed=disagreed,
        lagging=lagging,
        duration_ms=(_time.perf_counter() - started) * 1000.0,
    )


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        # Floats that came from the same digits can still differ in the last
        # place after two separate parses, which is not a disagreement about
        # what the overlay said.
        return abs(float(left) - float(right)) < 1e-6
    return str(left).strip() == str(right).strip()


def _record_payload(record: Any, image: str) -> dict[str, Any]:
    # Either a TelemetryRecord, when this frame was recognised here, or a plain
    # dict lifted from the stream's published snapshot. Both end up as one line
    # of telemetry.jsonl naming the same image.
    data = dict(record) if isinstance(record, dict) else _as_dict(record)
    data["image"] = image
    return data


def _as_dict(value: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(value)


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False)


def _append_json(path: Path, value: dict[str, Any]) -> None:
    import json

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
