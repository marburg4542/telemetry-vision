"""Send a capture to LINE.

Named line_client rather than line_notify on purpose: LINE Notify, which was the
easy way to do this, was shut down in March 2025. What remains is the Messaging
API, which is a bot pushing to a user or group and therefore needs a channel
access token and a destination id.

Text always; images only when a public base URL is configured. The Messaging API
cannot accept an uploaded image at all -- an image message carries a URL that
LINE's own servers fetch, over HTTPS, from somewhere publicly reachable. This
server listens on 127.0.0.1, so sending a picture means putting it briefly within
reach of the internet, which is what public_share exists to bound.

Configuring that is deliberately a separate step from configuring LINE at all.
Sending telemetry as text exposes nothing; sending the frame exposes the frame,
to anyone holding the URL while it lives and to LINE's cache permanently
afterwards. Someone should choose the second on purpose.

The token never appears in a log line, an error message or a response body.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError, DependencyError

_LOG = logging.getLogger(__name__)

PUSH_URL = "https://api.line.me/v2/bot/message/push"

# What is left of the month's allowance, and how much of it has gone. Two calls
# because LINE reports the ceiling and the consumption separately; neither
# includes the other.
QUOTA_URL = "https://api.line.me/v2/bot/message/quota"
CONSUMPTION_URL = "https://api.line.me/v2/bot/message/quota/consumption"

# LINE rejects a text message above 5000 characters. A capture is nowhere near
# it, but a truncation here is far better than a push rejected in full.
MAX_TEXT = 5000

DEFAULT_CONFIG_PATH = Path("configs/line.json")

_ENV_TOKEN = "TELEMETRY_OCR_LINE_TOKEN"
_ENV_TO = "TELEMETRY_OCR_LINE_TO"
_ENV_PUBLIC_URL = "TELEMETRY_OCR_PUBLIC_URL"

# LINE's own ceilings on an image message, and the two are an order of magnitude
# apart. The full-resolution captures on disk run 400-650 KB at q95, which is
# comfortable against the first and uncomfortably close to the second: a busier
# scene crosses it, and a push whose previewImageUrl is over is refused whole,
# taking the telemetry with it. These were declared here and never read, so the
# limit they describe was never actually applied to anything.
MAX_IMAGE_BYTES = 10 * 1024**2
MAX_PREVIEW_BYTES = 1024**2

# Field order in the message. Fixed rather than taken from the config so the
# message reads the same every time and the eye can find a line by position.
_LAYOUT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ACFT", ("acft_lat_dms", "acft_lon_dms", "acft_alt_m")),
    ("TAG ", ("tag_lat_dms", "tag_lon_dms", "tag_alt_m")),
    ("CAM ", ("sensor_angle_deg", "elevation_deg")),
    ("LRF ", ("lrf_m",)),
    ("SENS", ("eo_detail", "ir_detail")),
)

# Short names for the disagreement lines, where the full layout label is not
# available because a field is listed on its own.
_SHORT_LABELS = {
    "acft_lat_dms": "ACFT N",
    "acft_lon_dms": "ACFT E",
    "acft_alt_m": "ACFT ALT",
    "tag_lat_dms": "TAG N",
    "tag_lon_dms": "TAG E",
    "tag_alt_m": "TAG ALT",
    "sensor_angle_deg": "CAM yaw",
    "elevation_deg": "CAM pitch",
    "lrf_m": "LRF",
    "eo_detail": "EO",
    "ir_detail": "IR",
}

_UNITS = {
    "acft_alt_m": " m",
    "tag_alt_m": " m",
    "lrf_m": " m",
    "sensor_angle_deg": "°",
    "elevation_deg": "°",
}

# Every value is labelled, including the ones the UI can leave unlabelled because
# their position on screen says what they are. Whoever reads this is not looking
# at the screen, and "2.72° -24.96°" on its own does not say which is which.
_PREFIXES = {
    "acft_lat_dms": "N ",
    "acft_lon_dms": "E ",
    "acft_alt_m": "ALT ",
    "tag_lat_dms": "N ",
    "tag_lon_dms": "E ",
    "tag_alt_m": "ALT ",
    "sensor_angle_deg": "yaw ",
    "elevation_deg": "pitch ",
    "eo_detail": "EO ",
    "ir_detail": "IR ",
}


@dataclass(frozen=True, slots=True)
class ImageMessage:
    """One picture as LINE takes it: a full copy, and a preview of it.

    Two URLs rather than one because the ceilings differ by a factor of ten.
    They are usually the same file -- re-encoding a frame that already fits
    would cost an encode and a second token for no gain -- and differ only when
    the original is too heavy to stand in for its own preview.
    """

    original: str
    preview: str

    @classmethod
    def same(cls, url: str) -> ImageMessage:
        return cls(url, url)


def oversized_for_line(size_bytes: int) -> str:
    """Why LINE would refuse a picture this size, or "" when it would not.

    Checked before the push rather than after: LINE answers an oversized image
    with a 400 naming the property, the whole request is rejected including the
    text, and the operator is left with no telemetry and a message about a URL.
    """
    if size_bytes > MAX_IMAGE_BYTES:
        return (
            f"รูปขนาด {size_bytes / 1024**2:.1f} MB เกินขีดจำกัด "
            f"{MAX_IMAGE_BYTES // 1024**2} MB ของ LINE"
        )
    return ""


def needs_smaller_preview(size_bytes: int) -> bool:
    """Whether this file is too heavy to be its own preview."""
    return size_bytes > MAX_PREVIEW_BYTES


@dataclass(frozen=True, slots=True)
class LineConfig:
    channel_access_token: str
    to: str
    # Where LINE should fetch capture images from, e.g. a Cloudflare Tunnel
    # hostname. Empty means text only, which is the default and the safe one.
    public_base_url: str = ""

    @property
    def masked_to(self) -> str:
        """Enough of the destination to recognise, not enough to reuse."""
        if len(self.to) <= 6:
            return "*" * len(self.to)
        return f"{self.to[:4]}...{self.to[-2:]}"


def load_line_config(path: str | Path | None = None) -> LineConfig | None:
    """Read the credentials, or None when none are set.

    The environment wins over the file. A token put in the environment never
    touches the disk, which is the safer of the two and should not be overridden
    by a file someone forgot about.

    Returns None rather than raising when nothing is configured: not having set
    LINE up is a normal state for this application, and the endpoints say so
    themselves.
    """
    token = os.environ.get(_ENV_TOKEN, "").strip()
    to = os.environ.get(_ENV_TO, "").strip()
    base = os.environ.get(_ENV_PUBLIC_URL, "").strip()
    if token and to:
        return LineConfig(token, to, base)

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path} must contain a JSON object")

    token = token or str(data.get("channel_access_token", "")).strip()
    to = to or str(data.get("to", "")).strip()
    base = base or str(data.get("public_base_url", "")).strip()
    if not token or not to:
        return None
    return LineConfig(token, to, base)


# Below this many messages left, say so before the operator finds out from a
# 429 with a capture already taken. A tenth of the plan, and never fewer than
# ten: on a 300-message plan that is 30, which is a week of ordinary use rather
# than a warning that arrives too late to act on.
QUOTA_WARN_FRACTION = 0.1
QUOTA_WARN_FLOOR = 10


def quota_warning_threshold(limit: int) -> int:
    return max(QUOTA_WARN_FLOOR, round(limit * QUOTA_WARN_FRACTION))


@dataclass(frozen=True, slots=True)
class Quota:
    """The month's allowance, as LINE counts it.

    LINE counts a push by the number of destinations, not by the number of
    message objects in it -- their own example is four message objects to a room
    of five people counting as five. One capture from this app goes to one
    destination, so it costs exactly 1 however many pictures ride along, which
    is what makes batching them into a single request worth doing.

    `limit` is None when the account is on an unmetered plan, which LINE reports
    as type "none" with no value at all rather than as a large number.
    """

    limit: int | None = None
    used: int = 0
    error: str = ""

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.used)

    @property
    def warn(self) -> bool:
        left = self.remaining
        return (
            left is not None
            and self.limit is not None
            and left <= quota_warning_threshold(self.limit)
        )

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0

    def spend(self, messages: int = 1) -> Quota:
        """The same quota with a send accounted for.

        Applied locally instead of asking LINE again. The number is known --
        one destination, one message -- and a second round trip inside the send
        would make the button that is already slow slower still.
        """
        return Quota(self.limit, self.used + messages, self.error)

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "used": self.used,
            "remaining": self.remaining,
            "unlimited": self.limit is None and not self.error,
            "warn": self.warn,
            "exhausted": self.exhausted,
            "error": self.error,
        }


async def fetch_quota(config: LineConfig, *, timeout_s: float = 8.0) -> Quota:
    """Ask LINE what is left this month.

    A failure here is reported rather than raised. Not knowing the quota is a
    worse page, not a broken one, and the send itself does not depend on it --
    LINE is the thing that enforces the limit, and this only forecasts it.
    """
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover - aiohttp ships with the web extra
        raise DependencyError(
            "Reading the LINE quota requires aiohttp; install the project with "
            "`pip install -e .[web]`."
        ) from exc

    headers = {"Authorization": f"Bearer {config.channel_access_token}"}
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            limit: int | None = None
            async with session.get(QUOTA_URL) as response:
                if response.status != 200:
                    body = await response.text()
                    _LOG.warning("LINE quota lookup failed: %s %s", response.status, body[:200])
                    return Quota(error=_quota_error(response.status, body))
                payload = await response.json()
            # "none" is an unmetered plan and carries no value; treating a
            # missing value as zero would report an account with no limit as an
            # account with no messages left.
            if str(payload.get("type", "")) != "none":
                limit = int(payload.get("value") or 0)

            async with session.get(CONSUMPTION_URL) as response:
                if response.status != 200:
                    body = await response.text()
                    _LOG.warning("LINE usage lookup failed: %s %s", response.status, body[:200])
                    return Quota(limit=limit, error=_quota_error(response.status, body))
                payload = await response.json()
            return Quota(limit=limit, used=int(payload.get("totalUsage") or 0))
    except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
        _LOG.info("Could not read the LINE quota: %r", exc)
        return Quota(error=f"อ่านโควต้าไม่ได้: {exc or type(exc).__name__}")


def _quota_error(status: int, body: str) -> str:
    if status == 401:
        return "โควต้า: token ไม่ถูกต้อง (401)"
    return f"โควต้า: LINE ตอบ {status} {_line_error(body)}".strip()


def describe_detections(summary: Any) -> str:
    """One line about what was in the picture.

    The three non-counting states are spelled out rather than collapsed into a
    zero, because "no model is loaded" and "the model looked and found nothing"
    call for completely different responses from whoever reads this.
    """
    status = getattr(summary, "status", "unknown")
    if status == "no_render":
        return "ตรวจจับ: ยังไม่ได้ประมวลผลวิดีโอด้วย AI"
    if status == "not_processed":
        return "ตรวจจับ: ยังประมวลผลไม่ถึงเฟรมนี้"
    if status == "no_model":
        return "ตรวจจับ: ยังไม่ได้โหลดโมเดล"
    if status == "no_result":
        return "ตรวจจับ: โมเดลพร้อม แต่ยังไม่มีผลสำหรับภาพนี้"
    if status == "unknown":
        return "ตรวจจับ: ไม่ทราบสถานะ"

    counts = getattr(summary, "counts", {}) or {}
    if not counts:
        return "ตรวจจับ: ไม่พบวัตถุ"
    scores = getattr(summary, "confidence", {}) or {}
    # Most numerous first, then by name, so the same scene always reads the same.
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    parts = []
    for name, count in ordered:
        stat = scores.get(name)
        if not stat:
            parts.append(f"{name} {count}")
            continue
        # The lowest score is what decides whether the count can be trusted, so
        # it is shown whenever it differs from the average; three boxes at 0.9
        # and three where one sits at 0.31 are different claims.
        if count > 1 and abs(stat.get("min", 0.0) - stat.get("mean", 0.0)) >= 0.02:
            parts.append(f"{name} {count} (conf {stat['mean']:.2f}, ต่ำสุด {stat['min']:.2f})")
        else:
            parts.append(f"{name} {count} (conf {stat.get('mean', 0.0):.2f})")
    listed = ", ".join(parts)
    total = getattr(summary, "total", sum(counts.values()))
    line = f"ตรวจจับได้: {listed} (รวม {total})"

    age = getattr(summary, "age_ms", None)
    if age is not None and age > 2000:
        # Detection runs at its own pace; a large gap means these counts belong
        # to a visibly different moment than the frame, and saying so is more
        # useful than quietly presenting them as matching.
        line += f"\n  (ผลตรวจจับเก่ากว่าภาพ {age / 1000:.1f} วินาที)"
    return line


def _coordinate_link(values: dict[str, Any], lat_field: str, lon_field: str) -> str:
    latitude = (values.get(lat_field) or {}).get("value")
    longitude = (values.get(lon_field) or {}).get("value")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return ""
    return maps_link(latitude, longitude)


def describe_verification(
    verification: Any, rescued: int = 0
) -> str:
    """How far the saved image agreed with what the live feed had published.

    The values above come from the image wherever it could be read, so this is
    a statement about the two readings agreeing rather than about which was
    used. Where the OCR text matched, the reading is about as solid as this
    system gets.
    """
    status = getattr(verification, "status", "skipped")
    if status == "unreadable":
        return "ตรวจซ้ำจากไฟล์ภาพ: อ่านไฟล์ไม่ได้"
    if status == "failed":
        return "ตรวจซ้ำจากไฟล์ภาพ: ไม่สำเร็จ"
    if status != "ok":
        return ""

    agreed = getattr(verification, "agreed", 0)
    disagreed = getattr(verification, "disagreed", 0)
    total = agreed + disagreed
    parts = [f"ตรวจซ้ำจากไฟล์ภาพ: ข้อความตรงกัน {agreed}/{total}"]
    if rescued:
        # Worth saying: the picture could not be read for these, so the number
        # shown is what voting held rather than what this frame gave.
        parts.append(f"  ({rescued} ฟิลด์อ่านจากภาพไม่ได้ ใช้ค่าจาก voting แทน)")
    elif disagreed:
        parts.append(f"  ({disagreed} ฟิลด์อ่านได้ต่างกัน ควรเปิดรูปดู)")
    return "\n".join(parts)


def format_capture_message(
    shot: Any,
    directory: str | Path | None = None,
    dms_decimals: dict[str, int] | None = None,
) -> str:
    """Turn one shot into the text that gets pushed."""
    values: dict[str, Any] = {}
    for record in getattr(shot, "records", []) or []:
        field_id = record.get("field_id")
        if not field_id:
            continue
        # An unconfirmed reading is shown with a marker rather than hidden: the
        # point of a capture is to report what was on screen, and silently
        # dropping the fields that were hardest to read would misrepresent it.
        values[field_id] = record

    # The saved image is read first, and the live reading is the fallback.
    #
    # Settled by counting, over every capture on disk, rather than by argument:
    #
    #   policy                     confirmed     shown as raw text
    #   image only, no fallback    94.5%         5.5%
    #   live only                  88.2%         1.8%
    #   image, else live           95.5%         1.8%
    #
    # The image wins the usual case because it is the picture being described,
    # and because temporal voting is deliberately behind it -- 68 of 110 fields
    # had both readings confirmed and disagreeing, which is voting still holding
    # an earlier frame. But the image is one sample, and when it fails
    # validation there is nothing to fall back on within itself. That was the
    # bug: no fallback, so a field the image misread was printed as raw OCR text
    # -- "10z°00'23.7" where voting had 102.006658 all along. Four fields across
    # these captures were being shown that way for no reason.
    verification = getattr(shot, "verification", None)
    rescued = 0
    for item in getattr(verification, "fields", []) or []:
        live_record = values.get(item.field_id) or {}
        live_value = live_record.get("value")
        if item.value is not None:
            values[item.field_id] = {
                "field_id": item.field_id,
                "value": item.value,
                "raw_text": item.raw_text,
                "confidence": item.confidence,
                "confirmed": item.confirmed,
            }
        elif live_value is not None:
            # The image could not be validated; voting had an answer. Keeping the
            # live record is what stops raw OCR text reaching the message.
            rescued += 1

    lines = [f"[Capture] {getattr(shot, 'captured_at', '')}"]
    session = getattr(shot, "session", "")
    image = getattr(shot, "image", "")
    lines.append(f"{session} · {image}")
    lines.append("")

    for label, field_ids in _LAYOUT:
        parts = [
            _format_field(field_id, values.get(field_id), dms_decimals)
            for field_id in field_ids
        ]
        rendered = [part for part in parts if part]
        if rendered:
            lines.append(f"{label}  " + "   ".join(rendered))

    # The target's position, one tap from a map. Only when both halves read --
    # a link built from a missing coordinate would point somewhere real and
    # wrong, which is worse than no link.
    tag_link = _coordinate_link(values, "tag_lat_dms", "tag_lon_dms")
    if tag_link:
        # Set off from the readings above it. The link is the one line meant to
        # be tapped rather than read, and running it straight on from SENS made
        # the block look like a sixth telemetry row.
        lines.append("")
        lines.append(f"      TAG on map: {tag_link}")

    lines.append("")
    lines.append(describe_detections(getattr(shot, "detections", None)))

    if directory is not None:
        # The annotated copy when there is one. Naming the original as well put
        # two nearly identical paths in a phone notification and left the reader
        # to work out which was which; it is still on disk beside this one.
        annotated = getattr(shot, "annotated", None)
        lines.append(f"รูป: {Path(directory) / (annotated or image)}")

    text = "\n".join(lines)
    if len(text) > MAX_TEXT:
        text = text[: MAX_TEXT - 1] + "…"
    return text


# Decimal places of arc-seconds in a printed coordinate.
#
# Two, because that is what `dms_decimal_places` says for every coordinate field
# and what the field regexes require -- the overlay prints 14°50'57.37" and the
# reading carries all of it. This was fixed at one place, so the message quietly
# dropped a digit the system had read correctly and sent 14°50'57.4" instead.
# A hundredth of an arc-second is about 0.31 m; rounding to a tenth moves the
# position by up to 1.5 m, which is not a rounding error anyone should absorb
# on a coordinate that is going to be acted on.
DEFAULT_DMS_DECIMALS = 2

# Decimal degrees in the map link. 1e-7 degrees is about 1.1 cm, two orders of
# magnitude finer than the 0.31 m the source data is quantised to, so the link
# and the printed DMS always describe the same point: converting the link's
# number back to arc-seconds lands within 0.0004" of the digits above it.
# At the previous 1e-6 they could differ by half a metre.
_MAPS_DECIMALS = 7


def format_dms(
    degrees: float, is_latitude: bool, decimals: int = DEFAULT_DMS_DECIMALS
) -> str:
    """Decimal degrees as the overlay writes them: 14°50\'42.6"N.

    The OSD shows DMS, so a message quoting the same reading in decimal makes
    the reader convert before they can compare the two. The hemisphere comes
    from the sign rather than from a fixed prefix, which a southern or western
    position would otherwise label wrongly.
    """
    hemisphere = ("N" if degrees >= 0 else "S") if is_latitude else ("E" if degrees >= 0 else "W")
    value = abs(float(degrees))
    whole = int(value)
    minutes_all = (value - whole) * 60.0
    minutes = int(minutes_all)
    seconds = (minutes_all - minutes) * 60.0
    # Seconds can round up to 60 and produce 50'60.00"; carry it rather than
    # print a value no coordinate notation allows.
    if round(seconds, decimals) >= 60.0:
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        whole += 1
    # Two digits before the point, the point itself, then the fraction.
    width = decimals + 3
    return f"{whole}°{minutes:02d}\'{seconds:0{width}.{decimals}f}\"{hemisphere}"


def maps_link(latitude: float, longitude: float) -> str:
    """A link that opens the position, in the app on a phone and the web elsewhere.

    Built from the stored reading rather than from the rounded text above it, and
    printed with enough digits that it cannot disagree with that text. See
    _MAPS_DECIMALS.
    """
    return (
        "https://www.google.com/maps?q="
        f"{float(latitude):.{_MAPS_DECIMALS}f},{float(longitude):.{_MAPS_DECIMALS}f}"
    )


def _format_field(
    field_id: str,
    record: dict[str, Any] | None,
    dms_decimals: dict[str, int] | None = None,
) -> str:
    if record is None:
        return ""
    value = record.get("value")
    if value is None:
        # Raw OCR text, marked, because nothing validated it. This is the only
        # place the marker appears now.
        raw = (record.get("raw_text") or "").strip()
        return f"{_PREFIXES.get(field_id, '')}{raw or '—'}?" if raw else ""
    unit = _UNITS.get(field_id, "")
    prefix = _PREFIXES.get(field_id, "")
    # No marker on a validated value.
    #
    # It used to follow `confirmed`, which is temporal voting having seen the
    # same reading in enough consecutive frames. That is the right test for a
    # running stream and the wrong one for a capture: a shot is a single moment,
    # so voting has usually not reached its quorum and nearly everything came
    # out marked. Worse, it marked readings that were plainly right --
    # "10.3xFOV7.2°", repaired by validation from a garbled "1D.3xFQv7.2”" --
    # which teaches the reader to ignore the mark. A value that passed the
    # field's own regex and range is reported as a value.
    marker = ""
    if "lat_dms" in field_id or "lon_dms" in field_id:
        # DMS, and no prefix: the hemisphere letter is part of the rendering.
        places = (dms_decimals or {}).get(field_id, DEFAULT_DMS_DECIMALS)
        return f"{format_dms(float(value), 'lat_dms' in field_id, places)}{marker}"
    rendered = f"{value:g}" if isinstance(value, float) else str(value)
    return f"{prefix}{rendered}{unit}{marker}"


# One push carries at most five messages, and one of them is the text, so four
# pictures is the most that can go in a single request. It is a real cap rather
# than a chosen one: a fifth picture means a second request, which spends a
# second message from the account's monthly allowance.
MAX_IMAGES_PER_PUSH = 4

_DIGEST_RULE = "\n\n" + "\u2014" * 8 + "\n\n"


def format_capture_digest(
    shots: list[Any],
    directory: str | Path | None = None,
    dms_decimals: dict[str, int] | None = None,
) -> str:
    """The text for several shots sent together.

    Each block is the message that shot would have sent on its own, so a digest
    and a single send say the same thing about the same frame. They are joined
    rather than summarised because the reason to send four frames at once is to
    compare them, and a summary would drop the numbers being compared.
    """
    if not shots:
        return ""
    if len(shots) == 1:
        return format_capture_message(
            shots[0], directory=directory, dms_decimals=dms_decimals
        )
    header = f"[Capture x{len(shots)}]\n\n"
    blocks = [
        format_capture_message(shot, directory=directory, dms_decimals=dms_decimals)
        for shot in shots
    ]
    text = header + _DIGEST_RULE.join(blocks)
    if len(text) <= MAX_TEXT:
        return text
    # Trimmed from the end, and said out loud. Silently dropping the last frames
    # would leave pictures in the chat with no numbers under them.
    kept: list[str] = []
    total = len(header)
    for block in blocks:
        if total + len(block) + len(_DIGEST_RULE) > MAX_TEXT - 60:
            break
        kept.append(block)
        total += len(block) + len(_DIGEST_RULE)
    note = f"\n\n(ข้อความยาวเกิน ตัดเหลือ {len(kept)} จาก {len(shots)} เฟรม)"
    return header + _DIGEST_RULE.join(kept) + note


async def push_capture(
    config: LineConfig,
    text: str,
    images: list[ImageMessage] | None = None,
    *,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Push the telemetry, and the pictures when there are any.

    One request carrying every message. LINE counts a push by the request rather
    than by the messages inside it, so batching keeps a capture at one message
    from the monthly allowance instead of three, and it arrives in the chat in
    one piece rather than as a text that might be followed by images.
    """
    messages: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in (images or [])[:MAX_IMAGES_PER_PUSH]:
        messages.append(
            {
                "type": "image",
                "originalContentUrl": image.original,
                "previewImageUrl": image.preview,
            }
        )
    return await _push(config, messages, timeout_s=timeout_s)


async def push_text(config: LineConfig, text: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
    """Push one text message. Returns what happened, never the token.

    LINE's failure modes are fixed in different places -- 401 is the token, 400
    is usually the destination id, 429 is the monthly quota -- so the status and
    LINE's own message are both passed back rather than flattened into "failed".
    """
    return await _push(config, [{"type": "text", "text": text}], timeout_s=timeout_s)


async def _push(
    config: LineConfig, messages: list[dict[str, Any]], *, timeout_s: float
) -> dict[str, Any]:
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover - aiohttp ships with the web extra
        raise DependencyError(
            "Sending to LINE requires aiohttp; install the project with "
            "`pip install -e .[web]`."
        ) from exc

    text = next(
        (item.get("text", "") for item in messages if item.get("type") == "text"), ""
    )
    payload = {"to": config.to, "messages": messages}
    headers = {
        "Authorization": f"Bearer {config.channel_access_token}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(PUSH_URL, json=payload, headers=headers) as response,
        ):
            body = await response.text()
            request_id = response.headers.get("x-line-request-id", "")
            ok = response.status == 200
            if not ok:
                # Logged without the headers, which are the only part carrying
                # anything secret.
                _LOG.warning("LINE push failed: %s %s", response.status, body[:300])
            return {
                "ok": ok,
                "status": response.status,
                "request_id": request_id,
                "line_message": _line_error(body) if not ok else "",
                "characters": len(text),
            }
    except (aiohttp.ClientError, TimeoutError) as exc:
        # TimeoutError as well as ClientError: aiohttp raises the former when the
        # total timeout expires, and it is not a subclass of the latter. Letting
        # it escape turned a slow LINE into a 500 carrying only the exception
        # name, which tells the operator nothing about what to do.
        _LOG.warning("LINE push could not be sent: %r", exc)
        detail = str(exc) or type(exc).__name__
        return {
            "ok": False,
            "status": 0,
            "request_id": "",
            "line_message": f"ติดต่อ LINE ไม่ได้: {detail}",
            "characters": len(text),
        }


def _line_error(body: str) -> str:
    try:
        data = json.loads(body)
    except ValueError:
        return body[:300]
    message = str(data.get("message", "")).strip()
    details = data.get("details") or []
    if details:
        extra = "; ".join(
            f"{item.get('property', '')} {item.get('message', '')}".strip()
            for item in details
            if isinstance(item, dict)
        )
        if extra:
            return f"{message} ({extra})"
    return message or body[:300]
