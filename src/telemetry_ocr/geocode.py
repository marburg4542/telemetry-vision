"""Turn the TAG coordinate into a place an officer can be sent to.

A message that says 14°51'53.1"N 102°01'15.0"E is exact and useless on the
radio. What gets acted on is "ถ.304 ต.หินดาด อ.ห้วยแถลง", so the coordinate is
put through Google's reverse geocoder and the answer is printed above the map
link, which stays: the address is for saying out loud, the link for driving.

Three things this deliberately does not do:

* It does not call the Places API for the nearby landmark. A reverse geocode
  already returns several results for one point, and the ones Google marks as
  an establishment or a premise are landmarks with coordinates of their own --
  the distance and bearing below are measured from those. Places Nearby would
  name them better and costs about six times as much per call.
* It does not invent detail. Outside town there is usually no house number in
  any database and often no road name either; the line then says so rather than
  reading as though the address were complete.
* It never holds up a message. Anything that fails -- no network, a refused
  key, the day's cap -- becomes one line of explanation next to the coordinate
  that was going to be sent anyway.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

_ENV_KEY = "TELEMETRY_OCR_GOOGLE_MAPS_KEY"
DEFAULT_CONFIG_PATH = Path("configs/line.json")

# Cached by position, rounded to about 11 m. A drone hovering over one target
# sends a dozen captures from the same spot, and asking again for a hundredth of
# an arc-second is a call spent on a road name that cannot have changed.
CACHE_DECIMALS = 4

# A ceiling on calls per day, because a runaway loop against a billed API is
# the kind of mistake that is only noticed on the invoice. 500 is far more than
# a day of flying and still under a dollar at Geocoding's $5 per 1000.
DEFAULT_DAILY_LIMIT = 500

CACHE_ENTRIES = 5000
DEFAULT_LANGUAGE = "th"
DEFAULT_TIMEOUT_S = 6.0

# Results whose own coordinates make them usable as a landmark, nearest first.
# A route or an administrative area is not one: it has a centre, but the centre
# is not a place anyone can see from the road.
_LANDMARK_TYPES = frozenset(
    {
        "point_of_interest",
        "establishment",
        "premise",
        "subpremise",
        "park",
        "school",
        "university",
        "place_of_worship",
        "hospital",
        "pharmacy",
        "police",
        "fire_station",
        "local_government_office",
        "store",
        "gas_station",
        "restaurant",
        "lodging",
        "tourist_attraction",
        "transit_station",
        "bus_station",
        "train_station",
        "airport",
        "natural_feature",
        "campground",
        "cemetery",
        "stadium",
    }
)

# Which Google component fills each slot, in the order they are trusted. Thai
# addresses come back inconsistently: a tambon is administrative_area_level_3 in
# most of the country, a sublocality inside Bangkok, and occasionally only the
# locality, so each slot takes the first of several types rather than one.
_SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("house_number", ("street_number",)),
    ("road", ("route",)),
    ("subdistrict", ("administrative_area_level_3", "sublocality_level_2", "sublocality_level_1")),
    ("district", ("administrative_area_level_2", "locality")),
    ("province", ("administrative_area_level_1",)),
)

# Google prints the Thai administrative prefix inside the name; the message adds
# its own short one, so the long form is taken off first. Otherwise a line reads
# "ต.ตำบลหินดาด".
_PREFIXES = (
    "จังหวัด",
    "อำเภอ",
    "กิ่งอำเภอ",
    "ตำบล",
    "แขวง",
    "เขต",
    "ถนน",
    "ทางหลวงแผ่นดินหมายเลข",
    "ทางหลวงหมายเลข",
    "ซอย",
)

_PLACEHOLDER_NAMES = frozenset({"unnamed road", "ถนนไม่มีชื่อ", "ไม่มีชื่อ"})

# A plus code, e.g. V2CC+MQP: Google's fallback where nothing is named.
_PLUS_CODE = re.compile(r"^[0-9A-Z]{4,8}\+[0-9A-Z]{2,3}\b")

_COMPASS = (
    "เหนือ",
    "ตะวันออกเฉียงเหนือ",
    "ตะวันออก",
    "ตะวันออกเฉียงใต้",
    "ใต้",
    "ตะวันตกเฉียงใต้",
    "ตะวันตก",
    "ตะวันตกเฉียงเหนือ",
)

# Why a lookup produced no address, in the words the message will carry.
_STATUS_NOTES = {
    "ZERO_RESULTS": "ไม่พบที่อยู่ของพิกัดนี้ในฐานข้อมูล",
    "OVER_QUERY_LIMIT": "โควตา Google เต็มหรือเกินอัตราที่กำหนด",
    "OVER_DAILY_LIMIT": "คีย์ Google ใช้ไม่ได้ (โควตา/การเรียกเก็บเงิน)",
    "REQUEST_DENIED": "Google ปฏิเสธคีย์ (ตรวจสอบว่าเปิด Geocoding API แล้ว)",
    "INVALID_REQUEST": "คำขอไม่ถูกต้อง",
    "UNKNOWN_ERROR": "Google ตอบว่าเกิดข้อผิดพลาดชั่วคราว",
}


@dataclass(frozen=True, slots=True)
class GeocodeConfig:
    """The key and the limits. Read the same way as the LINE credentials."""

    api_key: str
    language: str = DEFAULT_LANGUAGE
    daily_limit: int = DEFAULT_DAILY_LIMIT


@dataclass(frozen=True, slots=True)
class Place:
    """What one coordinate turned out to be, or why it did not.

    `note` alone, with everything else empty, is a lookup that failed; it is
    still worth printing, because the reader needs to know the address is
    missing rather than absent.
    """

    house_number: str = ""
    road: str = ""
    subdistrict: str = ""
    district: str = ""
    province: str = ""
    landmark: str = ""
    landmark_m: float = 0.0
    landmark_bearing: str = ""
    note: str = ""

    @property
    def found(self) -> bool:
        return any((self.house_number, self.road, self.subdistrict, self.district, self.province))

    def address(self) -> str:
        """The administrative line, most specific first."""
        parts = []
        if self.house_number:
            parts.append(f"บ้านเลขที่ {self.house_number}")
        if self.road:
            parts.append(_road_label(self.road))
        for prefix, name in (
            ("ต.", self.subdistrict),
            ("อ.", self.district),
            ("จ.", self.province),
        ):
            if name:
                parts.append(name if name.startswith(("แขวง", "เขต")) else f"{prefix}{name}")
        return " ".join(parts)

    def lines(self) -> list[str]:
        """The message lines, indented to sit under the map link."""
        if not self.found:
            return [f"      หาที่อยู่ไม่ได้: {self.note}"] if self.note else []
        address = self.address()
        # Said out loud, because an officer told "ต.หินดาด" with no road will
        # ask which road, and the answer is that nobody knows.
        if not self.road:
            address += " (ไม่มีชื่อถนนในฐานข้อมูล)"
        lines = [f"      ที่อยู่: {address}"]
        if self.landmark:
            lines.append(
                f"      ห่างจาก {self.landmark} {_distance_text(self.landmark_m)} "
                f"ไปทางทิศ{self.landmark_bearing}"
            )
        if self.note:
            lines.append(f"      ({self.note})")
        return lines


def load_geocode_config(path: str | Path | None = None) -> GeocodeConfig | None:
    """The key, from the environment or the LINE config file, or None.

    None is an ordinary state: without a key the messages go out exactly as
    they did before, with no line about an address at all. A key that is set but
    broken is the opposite -- that gets said in the message, because someone
    meant for the address to be there.
    """
    key = os.environ.get(_ENV_KEY, "").strip()
    language = os.environ.get("TELEMETRY_OCR_GEOCODE_LANGUAGE", "").strip()
    limit_text = os.environ.get("TELEMETRY_OCR_GEOCODE_DAILY_LIMIT", "").strip()
    data: dict[str, Any] = {}
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            data = {}
    key = key or str(data.get("google_maps_api_key", "")).strip()
    if not key:
        return None
    language = language or str(data.get("geocode_language", "")).strip() or DEFAULT_LANGUAGE
    limit = _as_int(limit_text) or _as_int(data.get("geocode_daily_limit")) or DEFAULT_DAILY_LIMIT
    return GeocodeConfig(key, language, limit)


def _as_int(value: Any) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


@dataclass
class _Cache:
    """Answers already paid for, and how many calls today has cost.

    Kept on disk so a restart -- which happens whenever the model panel loads
    something new -- does not buy the same road name again.
    """

    path: Path
    places: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> _Cache:
        cache = cls(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cache
        if isinstance(data, dict):
            places = data.get("places")
            calls = data.get("calls")
            if isinstance(places, dict):
                cache.places = {str(k): v for k, v in places.items() if isinstance(v, dict)}
            if isinstance(calls, dict):
                cache.calls = {str(k): int(v) for k, v in calls.items() if _as_int(v)}
        return cache

    def save(self) -> None:
        if len(self.places) > CACHE_ENTRIES:
            # Oldest first: dicts keep insertion order, and the oldest answers
            # are the least likely to be asked for again.
            for key in list(self.places)[: len(self.places) - CACHE_ENTRIES]:
                del self.places[key]
        today = _today()
        self.calls = {day: count for day, count in self.calls.items() if day >= today}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"places": self.places, "calls": self.calls}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            # A cache that cannot be written is slower and more expensive, not
            # broken. The address still goes out.
            pass

    def used_today(self) -> int:
        return self.calls.get(_today(), 0)

    def count_call(self) -> None:
        self.calls[_today()] = self.used_today() + 1


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def cache_key(latitude: float, longitude: float) -> str:
    return f"{latitude:.{CACHE_DECIMALS}f},{longitude:.{CACHE_DECIMALS}f}"


Fetch = Callable[[str, dict[str, str], float], Awaitable[dict[str, Any]]]


async def _fetch(url: str, params: dict[str, str], timeout_s: float) -> dict[str, Any]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.get(url, params=params) as response,
    ):
        if response.status != 200:
            return {"status": "HTTP_ERROR", "http_status": response.status}
        return dict(await response.json(content_type=None))


class Geocoder:
    """One reverse geocoder, with its cache and its daily cap."""

    def __init__(
        self,
        config: GeocodeConfig,
        cache_path: Path,
        *,
        fetch: Fetch | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._config = config
        self._cache = _Cache.load(cache_path)
        self._fetch = fetch or _fetch
        self._timeout_s = timeout_s

    @property
    def used_today(self) -> int:
        return self._cache.used_today()

    async def place(self, latitude: float, longitude: float) -> Place:
        key = cache_key(latitude, longitude)
        cached = self._cache.places.get(key)
        if cached is not None:
            return Place(**{k: v for k, v in cached.items() if k in Place.__slots__})

        if self._cache.used_today() >= self._config.daily_limit:
            return Place(note=f"ครบโควตาที่ตั้งไว้วันนี้ ({self._config.daily_limit} ครั้ง)")

        params = {
            "latlng": f"{latitude:.6f},{longitude:.6f}",
            "language": self._config.language,
            "key": self._config.api_key,
        }
        try:
            payload = await self._fetch(GEOCODE_URL, params, self._timeout_s)
        except Exception as exc:
            return Place(note=f"เชื่อมต่อ Google ไม่สำเร็จ ({type(exc).__name__})")
        self._cache.count_call()

        place = parse_response(payload, latitude, longitude)
        # Only answers worth keeping. A timeout or a refused key would otherwise
        # be cached as the address of that spot for good.
        if place.found or payload.get("status") == "ZERO_RESULTS":
            self._cache.places[key] = asdict(place)
        # Off the loop: the file is small but it is written on the path of a
        # send, and a send already has a person waiting on it.
        await asyncio.to_thread(self._cache.save)
        return place


def parse_response(payload: dict[str, Any], latitude: float, longitude: float) -> Place:
    """Read Google's answer: the address, and the nearest landmark in it."""
    status = str(payload.get("status") or "")
    results = payload.get("results")
    if status == "HTTP_ERROR":
        return Place(note=f"Google ตอบรหัส HTTP {payload.get('http_status')}")
    if status != "OK" or not isinstance(results, list) or not results:
        note = _STATUS_NOTES.get(status, f"Google ตอบสถานะ {status or 'ไม่ทราบ'}")
        message = str(payload.get("error_message") or "").strip()
        return Place(note=f"{note} · {message}" if message else note)

    # The preferred type for a slot wins over a later one wherever it appears,
    # which is why the types are the outer loop. Around Nakhon Ratchasima one
    # result carries both `locality` ("ตำบล สุรนารี") and
    # `administrative_area_level_2` ("อำเภอเมืองนครราชสีมา"), with the locality
    # first; taking whichever came first in the component list put the tambon's
    # name in the amphoe's place.
    found: dict[str, str] = {}
    for slot, types in _SLOTS:
        for wanted in types:
            for result in results:
                if not isinstance(result, dict):
                    continue
                for component in result.get("address_components") or []:
                    if not isinstance(component, dict):
                        continue
                    kinds = component.get("types") or []
                    if not isinstance(kinds, list) or wanted not in kinds:
                        continue
                    name = _strip_prefix(str(component.get("long_name") or "").strip())
                    if name and not _is_placeholder(name):
                        found[slot] = name
                        break
                if found.get(slot):
                    break
            if found.get(slot):
                break

    landmark, distance, bearing = _landmark(results, latitude, longitude)
    return Place(
        house_number=found.get("house_number", ""),
        road=found.get("road", ""),
        subdistrict=found.get("subdistrict", ""),
        district=found.get("district", ""),
        province=found.get("province", ""),
        landmark=landmark,
        landmark_m=distance,
        landmark_bearing=bearing,
    )


def _landmark(
    results: list[Any], latitude: float, longitude: float
) -> tuple[str, float, str]:
    """The nearest result that is a place rather than an area."""
    best: tuple[float, str, str] | None = None
    for result in results:
        if not isinstance(result, dict):
            continue
        types = result.get("types") or []
        if not isinstance(types, list) or not set(types) & _LANDMARK_TYPES:
            continue
        location = ((result.get("geometry") or {}).get("location")) or {}
        spot_lat, spot_lon = location.get("lat"), location.get("lng")
        if not isinstance(spot_lat, (int, float)) or not isinstance(spot_lon, (int, float)):
            continue
        name = _landmark_name(result)
        if not name:
            continue
        distance = distance_m(latitude, longitude, float(spot_lat), float(spot_lon))
        if best is None or distance < best[0]:
            # Measured from the landmark towards the target, because that is the
            # way it will be used: the officer knows where the temple is and
            # needs to be told which way to go from it.
            bearing = compass(bearing_deg(float(spot_lat), float(spot_lon), latitude, longitude))
            best = (distance, name, bearing)
    if best is None:
        return "", 0.0, ""
    return best[1], best[0], best[2]


def _landmark_name(result: dict[str, Any]) -> str:
    """The name Google gave the place, or nothing.

    Only a component Google itself marks as a place counts. The fallback used
    to be the head of `formatted_address`, which outside town is a plus code
    followed by the whole address -- "V2CC+MQP ตำบล สุรนารี อำเภอเมือง..." was
    being offered as the landmark to drive to.
    """
    for component in result.get("address_components") or []:
        if not isinstance(component, dict):
            continue
        kinds = set(component.get("types") or [])
        if kinds & _LANDMARK_TYPES:
            name = str(component.get("long_name") or "").strip()
            if name and not _is_placeholder(name):
                return name
    return ""


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Metres between two positions, on a sphere.

    Good to a few parts in a thousand over the kilometre or two that matters
    here, which is finer than the landmark's own position is known.
    """
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Degrees clockwise from north, from the first position to the second."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def compass(degrees: float) -> str:
    """One of eight Thai directions: what gets said on the radio."""
    index = int((degrees % 360.0) / 45.0 + 0.5) % 8
    return _COMPASS[index]


def _distance_text(metres: float) -> str:
    if metres >= 1000:
        return f"{metres / 1000:.1f} กม."
    return f"{round(metres / 10) * 10:.0f} ม."


def _road_label(road: str) -> str:
    """Highways keep their number, named roads get ถ. in front."""
    if road.startswith(("ถ.", "ทางหลวง", "ซ.", "ซอย")):
        return road
    if road[:1].isdigit():
        return f"ทางหลวง {road}"
    return f"ถ.{road}"


def _is_placeholder(name: str) -> bool:
    """Google's way of saying it has no name for this, dressed as a name.

    "Unnamed Road" is what a route with no name comes back as, even in a Thai
    response, and printing it gives "ถ.Unnamed Road". A plus code is a real
    locator but not something to drive towards by name; it is the map link's
    job, and the link is already in the message.
    """
    stripped = name.strip()
    if stripped.lower() in _PLACEHOLDER_NAMES:
        return True
    return bool(_PLUS_CODE.match(stripped))


def _strip_prefix(name: str) -> str:
    for prefix in _PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix) :].strip()
    return name
