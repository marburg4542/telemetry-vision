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
        return _normalize_optics(text, config)
    if config.normalizer == "numeric":
        cleaned = text.replace(",", ".").replace("。", ".").replace("·", ".")
        cleaned = cleaned.replace("．", ".").replace("：", ".").replace("，", ".")
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
        if config.decimals and "." not in numeric and "." in cleaned:
            # The reader saw a point but the digit beside it did not survive:
            # "646.Om", "513.an", "47&.2m". Shifting the point left here would
            # turn 646.0 into 64.6 -- a number that passes the range check and
            # is reported as confirmed while being ten times wrong. Worse, it
            # pre-empts the confusion-map retry, which reads "646.Om" correctly
            # once O becomes 0. Fail the frame and let that retry happen.
            return text
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


# Letters this OSD font hands back in place of digits. Applied only inside the
# number segments of an optics reading, never to the whole string: "O" is a
# digit in "1O2" and a letter in "FOV", and one map cannot be right about both.
_OPTICS_DIGITS = str.maketrans(
    {
        "O": "0", "Q": "0", "D": "0", "U": "0",
        "I": "1", "L": "1", "|": "1", "!": "1",
        "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7",
        # Accented vowels are what this font's 6 comes back as often enough to
        # be worth naming: "1.0x FOv o6.4" is 66.4 with one 6 decoded as a vowel.
        "Ó": "6", "Ò": "6", "Ô": "6", "Õ": "6", "Ö": "6",
    }
)

# The zoom is written "N.Nx" and the field of view "N.N°": both carry exactly
# one decimal, which is what makes a lost point recoverable rather than a guess.
_OPTICS_DECIMALS = 1


def _normalize_optics(text: str, config: FieldConfig) -> str:
    """Rebuild "1.0xFOV66.4°" from what the recognizer actually returned.

    Rebuilt from the parts rather than repaired in place, because the failures
    are not spelling mistakes in one string -- they are a decimal point that
    became nothing, a degree mark that became a letter, and an "F0V" that was
    never three letters to begin with. Measured over every capture on disk, the
    in-place repair validated 26 of 74 EO readings and 40 of 74 IR readings; the
    rest were readable by eye in the saved frame, so the text was there and the
    normalizer was what threw it away.

    Nothing here invents a digit. The only thing reconstructed is *where* the
    point sits, and that is fixed by the overlay's format: one decimal place,
    always. A reading that loses a digit outright still fails.
    """
    cleaned = text.upper()
    # "F0V", "FQv", "FDV", "ROV" -- the recognizer rarely gets all three.
    cleaned = re.sub(r"[FPRE][O0QDU][VMWNY]", "FOV", cleaned)

    head, marker, tail = cleaned.partition("FOV")
    zoom = _optics_number(head)
    if zoom is None:
        return cleaned
    if not marker:
        # An IR reading is the zoom alone.
        return f"{zoom}x"
    fov = _optics_number(tail)
    if fov is None:
        return cleaned
    return f"{zoom}xFOV{fov}°"


def _optics_number(part: str) -> str | None:
    """One "N.N" out of a segment, or None when the digits do not add up.

    Every number in this overlay is followed by a mark -- the "x" of a zoom, the
    degree sign of a field of view -- and that mark is the character the
    recognizer gets wrong most often, because it is the one glyph in the ROI
    that is not a digit. It comes back as "a", as a quote, as "2", as nothing at
    all. So the trailing character is treated as the mark and dropped, and only
    kept as a digit when dropping it would leave too few to read.

    That distinction is what separates "102" from "400": the first is 1.0x with
    the x decoded as a 2, the second is 40.0x with the x decoded correctly and
    three real digits in front of it.
    """
    body = part.strip()
    if not body:
        return None
    if not body[-1].isdigit():
        # An explicit mark, however it was decoded. Drop it and read the rest.
        return _optics_digits(body[:-1])
    if "." in body or "," in body:
        # The point survived, so every digit after it is a real digit and the
        # mark was lost rather than decoded. Dropping one here turned 19.5 into
        # 1.9 and 10.3 into 1.0.
        return _optics_digits(body)
    # No point and it ends in a digit, so the mark was decoded as one. Dropping
    # it is right unless what is left is too short to be a reading at all.
    without_mark = _optics_digits(body[:-1])
    return without_mark if without_mark is not None else _optics_digits(body)


def _optics_digits(part: str) -> str | None:
    digits = "".join(
        char
        for char in part.translate(_OPTICS_DIGITS)
        # A separator the font drew as a comma, a dash or a colon is still a
        # separator; anything else in here is noise.
        if char.isdigit() or char in ".,-:"
    )
    digits = re.sub(r"[,\-:]", ".", digits)
    # Only the first point can be the real one; a second is a stray, and keeping
    # it would make the number unparseable.
    whole, point, fraction = digits.partition(".")
    fraction = fraction.replace(".", "")
    if point and whole and len(fraction) >= _OPTICS_DECIMALS:
        return f"{whole[-2:]}.{fraction[:_OPTICS_DECIMALS]}"
    run = whole + fraction
    if len(run) < 1 + _OPTICS_DECIMALS or len(run) > 2 + _OPTICS_DECIMALS:
        # Too few digits means one was lost; too many means something that is
        # not part of this number came along. Either way the point cannot be
        # placed from the count alone.
        return None
    return f"{run[:-_OPTICS_DECIMALS]}.{run[-_OPTICS_DECIMALS:]}"


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
            raw_text,
            candidate,
            None,
            False,
            corrected,
            f"characters_not_allowed:{''.join(invalid_chars)}",
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
