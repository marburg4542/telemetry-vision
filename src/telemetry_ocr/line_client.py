"""Send a capture to LINE.

Named line_client rather than line_notify on purpose: LINE Notify, which was the
easy way to do this, was shut down in March 2025. What remains is the Messaging
API, which is a bot pushing to a user or group and therefore needs a channel
access token and a destination id.

Only text is sent. The Messaging API cannot accept an uploaded image at all -- an
image message carries a URL that LINE's own servers fetch, over HTTPS, from
somewhere publicly reachable. This server listens on 127.0.0.1, so sending the
picture would mean exposing captured drone frames to anyone holding the URL, and
to LINE's cache. That is a decision about how sensitive the footage is, not a
detail of message formatting, so it is left out until someone makes it. The
message names the file on disk instead.

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

# LINE rejects a text message above 5000 characters. A capture is nowhere near
# it, but a truncation here is far better than a push rejected in full.
MAX_TEXT = 5000

DEFAULT_CONFIG_PATH = Path("configs/line.json")

_ENV_TOKEN = "TELEMETRY_OCR_LINE_TOKEN"
_ENV_TO = "TELEMETRY_OCR_LINE_TO"

# Field order in the message. Fixed rather than taken from the config so the
# message reads the same every time and the eye can find a line by position.
_LAYOUT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ACFT", ("acft_lat_dms", "acft_lon_dms", "acft_alt_m")),
    ("TAG ", ("tag_lat_dms", "tag_lon_dms", "tag_alt_m")),
    ("CAM ", ("sensor_angle_deg", "elevation_deg")),
    ("LRF ", ("lrf_m",)),
    ("SENS", ("eo_detail", "ir_detail")),
)

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
class LineConfig:
    channel_access_token: str
    to: str

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
    if token and to:
        return LineConfig(token, to)

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
    if not token or not to:
        return None
    return LineConfig(token, to)


def describe_detections(summary: Any) -> str:
    """One line about what was in the picture.

    The three non-counting states are spelled out rather than collapsed into a
    zero, because "no model is loaded" and "the model looked and found nothing"
    call for completely different responses from whoever reads this.
    """
    status = getattr(summary, "status", "unknown")
    if status == "no_model":
        return "ตรวจจับ: ยังไม่ได้โหลดโมเดล"
    if status == "no_result":
        return "ตรวจจับ: โมเดลพร้อม แต่ยังไม่มีผลสำหรับภาพนี้"
    if status == "unknown":
        return "ตรวจจับ: ไม่ทราบสถานะ"

    counts = getattr(summary, "counts", {}) or {}
    if not counts:
        return "ตรวจจับ: ไม่พบวัตถุ"
    # Most numerous first, then by name, so the same scene always reads the same.
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    listed = ", ".join(f"{name} {count}" for name, count in ordered)
    total = getattr(summary, "total", sum(counts.values()))
    line = f"ตรวจจับได้: {listed} (รวม {total})"

    age = getattr(summary, "age_ms", None)
    if age is not None and age > 2000:
        # Detection runs at its own pace; a large gap means these counts belong
        # to a visibly different moment than the frame, and saying so is more
        # useful than quietly presenting them as matching.
        line += f"\n  (ผลตรวจจับเก่ากว่าภาพ {age / 1000:.1f} วินาที)"
    return line


def format_capture_message(shot: Any, directory: str | Path | None = None) -> str:
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

    lines = [f"[Capture] {getattr(shot, 'captured_at', '')}"]
    session = getattr(shot, "session", "")
    image = getattr(shot, "image", "")
    lines.append(f"{session} · {image}")
    lines.append("")

    for label, field_ids in _LAYOUT:
        parts = [_format_field(field_id, values.get(field_id)) for field_id in field_ids]
        rendered = [part for part in parts if part]
        if rendered:
            lines.append(f"{label}  " + "   ".join(rendered))

    lines.append("")
    lines.append(describe_detections(getattr(shot, "detections", None)))

    if directory is not None:
        lines.append(f"รูป: {Path(directory) / image}")

    text = "\n".join(lines)
    if len(text) > MAX_TEXT:
        text = text[: MAX_TEXT - 1] + "…"
    return text


def _format_field(field_id: str, record: dict[str, Any] | None) -> str:
    if record is None:
        return ""
    value = record.get("value")
    if value is None:
        raw = (record.get("raw_text") or "").strip()
        return f"{_PREFIXES.get(field_id, '')}{raw or '—'}?" if raw else ""
    unit = _UNITS.get(field_id, "")
    prefix = _PREFIXES.get(field_id, "")
    if isinstance(value, float):
        rendered = f"{value:.6f}" if "dms" in field_id else f"{value:g}"
    else:
        rendered = str(value)
    marker = "" if record.get("confirmed") else "?"
    return f"{prefix}{rendered}{unit}{marker}"


async def push_text(config: LineConfig, text: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
    """Push one text message. Returns what happened, never the token.

    LINE's failure modes are fixed in different places -- 401 is the token, 400
    is usually the destination id, 429 is the monthly quota -- so the status and
    LINE's own message are both passed back rather than flattened into "failed".
    """
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover - aiohttp ships with the web extra
        raise DependencyError(
            "Sending to LINE requires aiohttp; install the project with "
            "`pip install -e .[web]`."
        ) from exc

    payload = {"to": config.to, "messages": [{"type": "text", "text": text}]}
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
