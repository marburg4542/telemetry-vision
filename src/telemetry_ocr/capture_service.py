"""Capture single frames on request, and keep them with what they read.

The rest of this application answers while the video plays and keeps nothing.
That is right for watching and wrong for recording: when an operator sees a value
worth keeping, the frame it came from is gone a fraction of a second later.

This is the other half. A shot writes the picture to disk and the telemetry
beside it, in the session format `extract` already understands, so a value can
always be checked against the exact frame it was read from.

The page posts the frame from the video it is playing. Nothing else is reading
that video, so the frame is recognised when the shot is taken.
"""

from __future__ import annotations

import logging
import re
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Re-exported: callers and tests import it from here.
from .annotate import annotate_frame as annotate_frame
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

# The readers a capture is checked by, and why there are nine of them.
#
# Measured against an answer key: the camera-yaw ROI of 42 saved captures, read
# by eye, so this counts *correct* answers rather than answers that merely pass
# the regex -- "3.40" for a ROI showing "33.40" passes and is wrong, and that is
# the failure being chased.
#
#     tiny + colour only (what this used to do)      35/42   4 wrong   3 blank
#     tiny, other channels for blanks only           38/42   4 wrong   0 blank
#     three models x colour                          35/42   6 wrong   1 blank
#     three models x two channels (6 reads)          37/42   5 wrong   0 blank
#     three models x three channels (9 reads)        40/42   2 wrong   0 blank
#
# Two things make the last row work, and neither does it alone. The channels
# differ because the overlay is saturated green drawn over vegetation: green on
# green has almost no luminance contrast, so a colour or grey recipe loses
# glyphs on exactly the frames worth capturing, while saturation separates them
# cleanly. The models differ because they are wrong in different places -- on
# their own the two mobile models score 26 and 21 against tiny's 35, which is
# what makes them useful as voters rather than as replacements.
#
# Nine reads cost about 30 ms per ROI against 3 ms for one, so roughly a third
# of a second on a twelve-field capture. A capture is one deliberate frame and
# can afford that; the live path never runs this.
_ENSEMBLE_MODELS = (
    "PP-OCRv6_tiny_rec",
    "en_PP-OCRv5_mobile_rec",
    "en_PP-OCRv4_mobile_rec",
)
_ENSEMBLE_CHANNELS = ("color", "hsv_s", "g")

_ensemble_lock = threading.Lock()
_ensemble_readers: dict[str, Any] = {}
_ensemble_failed: set[str] = set()


def _ensemble_recognizers(runtime: Any, primary: Any) -> dict[str, Any]:
    """The extra recognizers, built once and kept for the life of the process.

    A model that will not load is remembered as failed rather than retried on
    every capture: the usual reason is that it was never downloaded, which the
    next capture will not change. The ensemble then votes with whatever did
    load, down to the configured recognizer alone.
    """
    from dataclasses import replace as _replace

    from .ocr import create_recognizer

    with _ensemble_lock:
        # The configured recognizer votes whatever it is called. Registering it
        # only when its name appears in the list below would silence it entirely
        # on a deployment that had been pointed at some other model.
        _ensemble_readers[runtime.model_name] = primary
        for name in _ENSEMBLE_MODELS:
            if name == runtime.model_name:
                continue
            if name in _ensemble_readers or name in _ensemble_failed:
                continue
            try:
                _ensemble_readers[name] = create_recognizer(
                    _replace(runtime, model_name=name, warmup_iterations=1)
                )
            except Exception:
                _LOG.warning(
                    "Capture ensemble: %s could not be loaded; voting without it",
                    name,
                    exc_info=True,
                )
                _ensemble_failed.add(name)
        return dict(_ensemble_readers)


def close_ensemble() -> None:
    """Release the extra recognizers. Only the process shutdown needs this."""
    from .ocr import close_recognizer

    with _ensemble_lock:
        for name, recognizer in list(_ensemble_readers.items()):
            _ensemble_readers.pop(name, None)
            with suppress(Exception):
                close_recognizer(recognizer)


def _ensemble_read(frame: Any, pipeline: Any, produced: list[Any]) -> list[Any]:
    """Re-read every field with the ensemble and keep what the readers agree on.

    Majority wins; a tie goes to the higher summed confidence. The value the
    ordinary pipeline produced is kept only when the vote has nothing to say,
    which happens when no reader returns anything that validates.
    """
    from dataclasses import replace as _replace

    from .pipeline import _RECOGNIZER_LOCK
    from .preprocess import apply_recipe, extract_roi
    from .validation import validate_text

    if not produced:
        return produced
    config = pipeline.config
    readers = _ensemble_recognizers(config.runtime, pipeline.recognizer)
    if len(readers) <= 1 and len(_ENSEMBLE_CHANNELS) <= 1:
        return produced

    fields = {field.field_id: field for field in config.fields}
    # The crops, once per field per channel. Cropping and recipe work is cheap
    # next to recognition, but doing it once per model as well would triple it.
    jobs: list[tuple[int, Any, str]] = []
    images: list[Any] = []
    for index, record in enumerate(produced):
        field = fields.get(record.field_id)
        if field is None:
            continue
        padding = config.runtime.padding if field.padding is None else field.padding
        crop = extract_roi(frame, field.box, padding)
        for channel in _ENSEMBLE_CHANNELS:
            recipe = (
                field.primary
                if channel == field.primary.channel
                else _replace(
                    field.primary, name=channel, channel=channel, clahe=True, unsharp=True
                )
            )
            jobs.append((index, field, channel))
            images.append(apply_recipe(crop, recipe))
    if not images:
        return produced

    # value -> [votes, summed confidence, best (confidence, text, variant)]
    tally: list[dict[Any, list[Any]]] = [{} for _ in produced]
    with _RECOGNIZER_LOCK:
        for name, recognizer in readers.items():
            try:
                answers = recognizer.recognize(images)
            except Exception:
                _LOG.warning("Capture ensemble: %s failed to read", name, exc_info=True)
                continue
            for (index, field, channel), answer in zip(jobs, answers, strict=True):
                checked = validate_text(answer.text, field)
                if not checked.valid:
                    continue
                key = _vote_key(checked.value)
                slot = tally[index].setdefault(
                    key, [0, 0.0, (0.0, "", ""), checked.value]
                )
                slot[0] += 1
                slot[1] += answer.confidence
                if answer.confidence > slot[2][0]:
                    slot[2] = (answer.confidence, answer.text, f"{name}:{channel}")

    for index, votes in enumerate(tally):
        if not votes:
            continue
        field_config = fields.get(produced[index].field_id)
        may_override = bool(field_config and field_config.ensemble_override)
        if produced[index].value is not None and not may_override:
            # Blanks only, for a field nobody has measured this on. See
            # FieldConfig.ensemble_override.
            continue
        _key, (count, _total, best, value) = max(
            votes.items(), key=lambda item: (item[1][0], item[1][1])
        )
        confidence, raw_text, variant = best
        record = produced[index]
        if _same_value(record.value, value) and record.value is not None:
            continue
        produced[index] = _replace(
            record,
            value=value,
            raw_text=raw_text,
            confidence=confidence,
            # Agreement between independent readers is not the temporal quorum
            # `confirmed` means, and borrowing the word would misreport it.
            confirmed=record.confirmed and record.value is not None,
            preprocess_variant=f"{variant} ({count}/{len(readers) * len(_ENSEMBLE_CHANNELS)})",
        )
    return produced


def _vote_key(value: Any) -> Any:
    """Group readings that mean the same number despite two separate parses."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    return str(value).strip()


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
    # The copy with boxes drawn on it, when the detector had any. None when no
    # model was loaded or nothing was found -- an annotated file identical to the
    # original would only waste disk and invite the question of what it adds.
    annotated: str | None = None
    # "auto" when the render's watcher took it rather than someone pressing
    # Capture, with a note of which vehicle it was taken for.
    origin: str = "manual"
    note: str = ""


class CaptureSession:
    """One directory of shots, appended to as they are taken.

    Appends rather than rewrites: a session is open for as long as someone is
    watching, and losing an hour of captures because the last write failed would
    be a poor trade for a tidier file.
    """

    def __init__(
        self, root: Path, name: str, source: str, field_ids: list[str]
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
            # The page captures only from a file it is playing. The field stays in
            # the format because `telemetry-ocr capture` still records streams,
            # and extract reads sessions of both kinds.
            is_stream=False,
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
        # Only the plainly numbered frames. `frames/` also holds the marked-up
        # copies (`00000007.boxes.jpg`), and counting files instead of reading
        # the highest index would restart numbering part way through a session
        # and overwrite frames the manifest still describes.
        indices = [
            int(item.stem)
            for item in self.frame_dir.glob("*.jpg")
            if item.stem.isdigit()
        ]
        return max(indices) + 1 if indices else 0

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
        boxes: list[dict[str, Any]] | None = None,
        box_source: tuple[int, int] | None = None,
        origin: str = "manual",
        note: str = "",
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

        annotated_name: str | None = None
        if boxes and box_source:
            # After the clean frame is on disk, so a failure here costs the
            # annotation and never the record.
            try:
                marked = annotate_frame(frame, boxes, box_source[0], box_source[1])
                annotated_name = f"{FRAME_DIR}/{index:08d}.boxes.jpg"
                if not cv2.imwrite(
                    str(self.directory / annotated_name),
                    marked,
                    [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
                ):
                    annotated_name = None
            except Exception:
                _LOG.exception("Could not annotate frame %d", index)
                annotated_name = None
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
                    "annotated": annotated_name,
                    "origin": origin,
                    "note": note,
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
            annotated=annotated_name,
            origin=origin,
            note=note,
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

    def _read_manifest(self) -> list[dict[str, Any]]:
        import json

        manifest = self.directory / MANIFEST_NAME
        if not manifest.is_file():
            return []
        return [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _read_telemetry(self) -> dict[str, list[dict[str, Any]]]:
        """Every saved reading, grouped by the image it belongs to."""
        import json

        telemetry = self.directory / "telemetry.jsonl"
        grouped: dict[str, list[dict[str, Any]]] = {}
        if not telemetry.is_file():
            return grouped
        for line in telemetry.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            grouped.setdefault(str(record.get("image", "")), []).append(record)
        return grouped

    def all_shots(self) -> list[ShotResult]:
        """Every shot in the session, in the order they were taken.

        One pass over each file rather than `shot_at` per index. The per-index
        form re-parses the whole manifest *and* the whole telemetry log every
        time, so listing a session cost O(n^2) parsing: measured on synthetic
        sessions with the real 11 fields per frame, listing took 0.4 s at 50
        shots, 5.8 s at 200 and 33 s at 500 -- and the page lists after every
        capture, so holding the button made each shot slower than the last.
        """
        grouped = self._read_telemetry()
        shots = []
        for entry in self._read_manifest():
            shot = self._shot_from_entry(entry, grouped.get(str(entry.get("file", "")), []))
            if shot is not None:
                shots.append(shot)
        shots.sort(key=lambda item: item.frame_index)
        return shots

    def shot_at(self, frame_index: int) -> ShotResult | None:
        """Rebuild an earlier shot from what was written.

        Only the last one is held in memory; anything older is reconstructed
        from the manifest and the telemetry beside it, which is the same source
        `extract` reads and therefore cannot drift from it.
        """
        entry = next(
            (
                item
                for item in self._read_manifest()
                if int(item.get("frame_index", -1)) == frame_index
            ),
            None,
        )
        if entry is None:
            return None
        records = self._read_telemetry().get(str(entry.get("file", "")), [])
        return self._shot_from_entry(entry, records)

    def _shot_from_entry(
        self, entry: dict[str, Any], records: list[dict[str, Any]]
    ) -> ShotResult | None:
        if "frame_index" not in entry:
            return None
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
            annotated=entry.get("annotated"),
            origin=str(entry.get("origin") or "manual"),
            note=str(entry.get("note") or ""),
        )

    def discard(self, frame_index: int) -> bool:
        """Remove one shot: its pictures and every record that named them.

        The records go with the picture on purpose. A telemetry line whose image
        is no longer on disk is worse than no line at all -- `extract` would read
        the session back and report a frame nobody can look at.

        The frame numbering is not reused. `_next_index` only ever climbs while
        a session is open, and a reopened session takes its next number from the
        highest file still present, so a gap stays a gap rather than becoming a
        second frame with an old frame's name.
        """
        import json

        with self._lock:
            entries = self._read_manifest()
            keep = [
                entry for entry in entries
                if int(entry.get("frame_index", -1)) != frame_index
            ]
            if len(keep) == len(entries):
                return False
            gone = [
                entry for entry in entries
                if int(entry.get("frame_index", -1)) == frame_index
            ]
            images = {str(entry.get("file", "")) for entry in gone}

            for entry in gone:
                for relative in (entry.get("file"), entry.get("annotated")):
                    if not relative:
                        continue
                    # From the manifest, never from the request: this is the
                    # only thing keeping a frame index from naming a file
                    # somewhere else on the disk.
                    path = self.directory / str(relative)
                    with suppress(OSError):
                        self.bytes_written -= path.stat().st_size
                    path.unlink(missing_ok=True)
                    (self.directory / "thumbs" / path.name).unlink(missing_ok=True)

            # One object per line, like _append_json writes them. _dumps is the
            # indented form the session file uses and would break the manifest
            # into unparseable fragments.
            self._write_lines(
                MANIFEST_NAME,
                [json.dumps(entry, ensure_ascii=False) for entry in keep],
            )
            for name in ("telemetry.jsonl", "telemetry-verified.jsonl"):
                path = self.directory / name
                if not path.is_file():
                    continue
                kept_lines = [
                    line
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and json.loads(line).get("image") not in images
                ]
                self._write_lines(name, kept_lines)

            if self.last_shot is not None and self.last_shot.frame_index == frame_index:
                self.last_shot = None
            self.info.frame_count = len(keep)
            self._write_session()
            return True

    def _write_lines(self, name: str, lines: list[str]) -> None:
        """Rewrite one of the session's logs, atomically.

        Through a temporary file because the page reads these while captures are
        still being taken, and a log truncated halfway through a rewrite would
        lose shots that were never asked to be deleted.
        """
        path = self.directory / name
        staging = path.with_name(path.name + ".part")
        body = "\n".join(lines)
        staging.write_text(body + "\n" if body else "", encoding="utf-8")
        staging.replace(path)

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
    produced = _ensemble_read(frame, pipeline, list(produced))
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
