from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import FieldConfig


CONFUSION_MAP = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "Q": "0",
        "q": "0",
        "@": "0",
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedValue:
    raw_text: str
    normalized_text: str
    value: Any
    valid: bool
    corrected: bool = False
    reason: str | None = None


def validate_text(raw_text: str, config: FieldConfig) -> ValidatedValue:
    source = _normalize(raw_text)
    normalized = _normalize_for_field(source, config)
    first = _validate_candidate(raw_text, normalized, config, corrected=False)
    if first.valid:
        return first
    corrected = _normalize_for_field(source.translate(CONFUSION_MAP), config)
    if corrected != normalized:
        second = _validate_candidate(raw_text, corrected, config, corrected=True)
        if second.valid:
            return second
    return first


def _normalize(text: str) -> str:
    return "".join(text.strip().split())


def _normalize_for_field(text: str, config: FieldConfig) -> str:
    if config.normalizer == "none":
        return text
    if config.normalizer == "optics":
        cleaned = text.upper().replace(",", ".").replace(":", ".")
        cleaned = re.sub(r"(?<=\d)-(?=\d)", ".", cleaned)
        cleaned = cleaned.replace("F0V", "FOV")
        zoom = re.match(r"^(\d{1,2})\.(\d)X", cleaned)
        if zoom is None:
            return cleaned
        if "FOV" not in cleaned:
            return f"{zoom.group(1)}.{zoom.group(2)}x"
        fov = re.fullmatch(r"(\d{1,2})\.(\d)XFOV(\d{2,4})(?:°)?", cleaned)
        if fov:
            fov_digits = fov.group(3)
            cleaned = (
                f"{fov.group(1)}.{fov.group(2)}xFOV"
                f"{fov_digits[:-1]}.{fov_digits[-1]}°"
            )
        else:
            cleaned = cleaned.replace("X", "x")
        return cleaned
    if config.normalizer == "numeric":
        cleaned = text.replace(",", ".").replace("。", ".").replace("·", ".")
        cleaned = re.sub(r"(?<=\d):(?=\d)", ".", cleaned)
        if cleaned.startswith(("=", "—", "–", "_")):
            cleaned = "-" + cleaned[1:]
        cleaned = re.sub(r"(?<=\d)-(?=\d)", ".", cleaned)
        if config.decimals:
            # Repeated OSD-font confusions observed in telemetry overlays. These
            # repairs only apply to fields with a declared fixed precision.
            short_fraction = re.fullmatch(
                rf"([+-]?\d+\.\d{{{config.decimals - 1}}})%", cleaned
            )
            if short_fraction:
                cleaned = short_fraction.group(1) + "6"
            if "." not in cleaned:
                signed_digits = re.fullmatch(r"([+-]?)(\d+)", cleaned)
                if signed_digits:
                    sign, digits = signed_digits.groups()
                    if len(digits) > config.decimals:
                        resolved = _resolve_missing_point(sign, digits, config)
                        if resolved is None:
                            # Both readings are plausible and guessing would be a
                            # 10x error. Fail this frame instead; the temporal
                            # layer keeps the last confirmed value.
                            return text
                        cleaned = resolved
        match = re.search(r"[+-]?\d+(?:\.\d+)?", cleaned)
        if not match:
            return text
        numeric = match.group(0)
        if config.decimals and "." not in numeric:
            sign = "-" if numeric.startswith("-") else "+" if numeric.startswith("+") else ""
            digits = numeric[len(sign) :]
            if len(digits) > config.decimals:
                numeric = sign + digits[: -config.decimals] + "." + digits[-config.decimals :]
        if config.decimals is not None and "." in numeric:
            whole, fraction = numeric.split(".", 1)
            if config.decimals == 0:
                numeric = whole
            elif len(fraction) > config.decimals:
                numeric = whole + "." + fraction[: config.decimals]
            elif len(fraction) < config.decimals:
                numeric = whole + "." + fraction.ljust(config.decimals, "0")
        try:
            if numeric.startswith("-") and float(numeric) == 0:
                numeric = numeric[1:]
        except ValueError:
            pass
        return numeric
    if config.normalizer == "dms":
        degree_digits = config.dms_degrees_digits or 0
        decimal_places = config.dms_decimal_places
        digits = "".join(char for char in text if char in "0123456789")
        required = degree_digits + 2 + 2 + decimal_places
        # The closing double-prime glyph in this OSD font is commonly decoded
        # as a trailing "2". Drop it only when it is the sole extra digit and
        # no explicit seconds mark was recognized.
        if (
            len(digits) == required + 1
            and digits.endswith("2")
            and not text.endswith(('"', "″", "”"))
        ):
            digits = digits[:-1]
        if len(digits) != required:
            return text
        degrees = int(digits[:degree_digits])
        minutes = int(digits[degree_digits : degree_digits + 2])
        seconds_whole = digits[degree_digits + 2 : degree_digits + 4]
        seconds_fraction = digits[degree_digits + 4 :]
        if minutes >= 60 or int(seconds_whole) >= 60:
            return text
        seconds = seconds_whole
        if decimal_places:
            seconds += "." + seconds_fraction
        return f'{degrees}°{minutes:02d}\'{seconds}"'
    return text


def _resolve_missing_point(sign: str, digits: str, config: FieldConfig) -> str | None:
    """Place the decimal point in a fixed-precision read that lost it.

    Two readings compete. Either OCR dropped the point outright, or it decoded
    the point as a "7" - a repeated confusion in this OSD font. They differ by a
    factor of ten, so the choice is decided on evidence only:

    * a leading zero in the integer part rules a reading out, because the
      overlay prints `0.26`, never `07.26`
    * a reading that fails the field's regex or range rules itself out

    When both survive, the frame is genuinely ambiguous and None is returned so
    the caller can reject the read rather than emit a coin flip.
    """
    decimals = config.decimals or 0
    split = len(digits) - decimals
    plain = f"{sign}{digits[:split]}.{digits[split:]}"
    if digits[split - 1] != "7" or split < 2:
        return plain
    seven = f"{sign}{digits[: split - 1]}.{digits[split:]}"
    plain_ok = _is_plausible_reading(plain, config)
    seven_ok = _is_plausible_reading(seven, config)
    if plain_ok and seven_ok:
        return None
    if seven_ok:
        return seven
    return plain


def _is_plausible_reading(candidate: str, config: FieldConfig) -> bool:
    whole = candidate.lstrip("+-").split(".", 1)[0]
    if len(whole) > 1 and whole.startswith("0"):
        return False
    return _validate_candidate(candidate, candidate, config, corrected=False).valid


def _validate_candidate(
    raw_text: str, candidate: str, config: FieldConfig, corrected: bool
) -> ValidatedValue:
    if not candidate:
        return ValidatedValue(raw_text, candidate, None, False, corrected, "empty")
    invalid_chars = sorted(set(candidate) - set(config.allowed_chars))
    if invalid_chars:
        return ValidatedValue(
            raw_text, candidate, None, False, corrected, f"characters_not_allowed:{''.join(invalid_chars)}"
        )
    if re.fullmatch(config.regex, candidate) is None:
        return ValidatedValue(raw_text, candidate, None, False, corrected, "regex_mismatch")
    if config.data_type == "float" and config.decimals is not None and "." in candidate:
        decimal_digits = len(candidate.rsplit(".", 1)[1])
        if decimal_digits > config.decimals:
            return ValidatedValue(raw_text, candidate, None, False, corrected, "too_many_decimals")
    try:
        if config.data_type == "integer":
            value: Any = int(candidate)
        elif config.data_type == "float":
            value = float(candidate)
        elif config.data_type == "dms":
            value = _dms_to_decimal(candidate, config)
        else:
            value = candidate
    except ValueError:
        return ValidatedValue(raw_text, candidate, None, False, corrected, "type_conversion")
    if isinstance(value, (int, float)):
        if config.min_value is not None and value < config.min_value:
            return ValidatedValue(raw_text, candidate, value, False, corrected, "below_minimum")
        if config.max_value is not None and value > config.max_value:
            return ValidatedValue(raw_text, candidate, value, False, corrected, "above_maximum")
    return ValidatedValue(raw_text, candidate, value, True, corrected, None)


def _dms_to_decimal(candidate: str, config: FieldConfig) -> float:
    match = re.fullmatch(r"(\d{1,3})°(\d{2})'(\d{2}(?:\.\d+)?)\"", candidate)
    if match is None:
        raise ValueError("invalid canonical DMS")
    degrees = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("DMS minutes/seconds out of range")
    if config.hemisphere in {"N", "S"} and degrees > 90:
        raise ValueError("latitude degrees out of range")
    if config.hemisphere in {"E", "W"} and degrees > 180:
        raise ValueError("longitude degrees out of range")
    value = degrees + minutes / 60.0 + seconds / 3600.0
    return -value if config.hemisphere in {"S", "W"} else value


def prefer_candidate(
    first: tuple[ValidatedValue, float], second: tuple[ValidatedValue, float]
) -> tuple[ValidatedValue, float]:
    """Prefer validity before OCR confidence; ties retain the primary candidate."""
    first_value, first_score = first
    second_value, second_score = second
    if second_value.valid and not first_value.valid:
        return second
    if first_value.valid and not second_value.valid:
        return first
    if second_score > first_score:
        return second
    return first
