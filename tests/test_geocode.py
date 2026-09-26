from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry_ocr.geocode import (
    DEFAULT_DAILY_LIMIT,
    GeocodeConfig,
    Geocoder,
    Place,
    bearing_deg,
    compass,
    distance_m,
    load_geocode_config,
    parse_response,
)

# A point on a road outside town, and a temple 420 m west of it.
TARGET = (14.851000, 102.020000)
TEMPLE = (14.851000, 102.016100)


def response(*, road: bool = True, landmark: bool = True) -> dict:
    """Google's shape, in the Thai that a reverse geocode actually returns."""
    results: list[dict] = [
        {
            "types": ["route"],
            "formatted_address": "ทางหลวงแผ่นดินหมายเลข 304, หินดาด, ห้วยแถลง, นครราชสีมา",
            "geometry": {"location": {"lat": TARGET[0], "lng": TARGET[1]}},
            "address_components": (
                [{"long_name": "ทางหลวงแผ่นดินหมายเลข 304", "types": ["route"]}] if road else []
            )
            + [
                {"long_name": "ตำบลหินดาด", "types": ["administrative_area_level_3"]},
                {"long_name": "อำเภอห้วยแถลง", "types": ["administrative_area_level_2"]},
                {"long_name": "จังหวัดนครราชสีมา", "types": ["administrative_area_level_1"]},
            ],
        }
    ]
    if landmark:
        results.insert(
            0,
            {
                "types": ["place_of_worship", "point_of_interest", "establishment"],
                "formatted_address": "วัดหินดาด, หินดาด, ห้วยแถลง",
                "geometry": {"location": {"lat": TEMPLE[0], "lng": TEMPLE[1]}},
                "address_components": [
                    {
                        "long_name": "วัดหินดาด",
                        "types": ["point_of_interest", "establishment"],
                    },
                    {"long_name": "ตำบลหินดาด", "types": ["administrative_area_level_3"]},
                ],
            },
        )
    return {"status": "OK", "results": results}


class ParseTests(unittest.TestCase):
    def test_the_address_reads_as_an_address(self) -> None:
        place = parse_response(response(), *TARGET)
        self.assertEqual(place.road, "304")
        self.assertEqual(place.subdistrict, "หินดาด")
        self.assertEqual(place.district, "ห้วยแถลง")
        self.assertEqual(place.province, "นครราชสีมา")
        # The Thai prefix Google puts inside the name is not printed twice.
        self.assertEqual(place.address(), "ทางหลวง 304 ต.หินดาด อ.ห้วยแถลง จ.นครราชสีมา")

    def test_the_landmark_is_measured_from_itself_to_the_target(self) -> None:
        # An officer who knows the temple needs to be told which way to go.
        place = parse_response(response(), *TARGET)
        self.assertEqual(place.landmark, "วัดหินดาด")
        self.assertAlmostEqual(place.landmark_m, 420, delta=15)
        self.assertEqual(place.landmark_bearing, "ตะวันออก")
        lines = place.lines()
        self.assertIn("ที่อยู่: ทางหลวง 304", lines[0])
        self.assertIn("ห่างจาก วัดหินดาด 420 ม. ไปทางทิศตะวันออก", lines[1])

    def test_a_place_with_no_road_says_so(self) -> None:
        """Most of the countryside has no road name in any database.

        The line has to read as incomplete, or it will be repeated on the radio
        as though a road had been given.
        """
        place = parse_response(response(road=False), *TARGET)
        self.assertEqual(place.road, "")
        self.assertIn("(ไม่มีชื่อถนนในฐานข้อมูล)", place.lines()[0])
        self.assertIn("ต.หินดาด", place.lines()[0])

    def test_an_area_is_not_offered_as_a_landmark(self) -> None:
        # A tambon has a centre, but nobody can stand at it.
        place = parse_response(response(landmark=False), *TARGET)
        self.assertEqual(place.landmark, "")
        self.assertEqual(len(place.lines()), 1)

    def test_a_house_number_is_kept_when_there_is_one(self) -> None:
        payload = response()
        payload["results"][-1]["address_components"].insert(
            0, {"long_name": "41/2", "types": ["street_number"]}
        )
        self.assertTrue(parse_response(payload, *TARGET).address().startswith("บ้านเลขที่ 41/2"))

    def test_every_failure_becomes_one_line_the_reader_can_act_on(self) -> None:
        for payload, expected in (
            ({"status": "ZERO_RESULTS", "results": []}, "ไม่พบที่อยู่"),
            ({"status": "REQUEST_DENIED"}, "ปฏิเสธคีย์"),
            ({"status": "OVER_QUERY_LIMIT"}, "โควตา Google"),
            ({"status": "HTTP_ERROR", "http_status": 503}, "HTTP 503"),
        ):
            with self.subTest(status=payload["status"]):
                place = parse_response(payload, *TARGET)
                self.assertFalse(place.found)
                [line] = place.lines()
                self.assertIn("หาที่อยู่ไม่ได้", line)
                self.assertIn(expected, line)

    def test_googles_own_explanation_is_carried_through(self) -> None:
        payload = {"status": "REQUEST_DENIED", "error_message": "API key not valid"}
        self.assertIn("API key not valid", parse_response(payload, *TARGET).lines()[0])


class MathTests(unittest.TestCase):
    def test_distance_and_bearing(self) -> None:
        self.assertAlmostEqual(distance_m(*TARGET, *TEMPLE), 420, delta=15)
        self.assertAlmostEqual(bearing_deg(*TEMPLE, *TARGET), 90, delta=1)
        for degrees, name in ((0, "เหนือ"), (46, "ตะวันออกเฉียงเหนือ"), (180, "ใต้"), (359, "เหนือ")):
            with self.subTest(degrees=degrees):
                self.assertEqual(compass(degrees), name)

    def test_a_kilometre_is_not_printed_as_metres(self) -> None:
        far = Place(road="x", landmark="สถานีตำรวจ", landmark_m=2350.0, landmark_bearing="ใต้")
        self.assertIn("2.4 กม.", far.lines()[1])


class GeocoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.cache = Path(self.temp.name) / "geocode-cache.json"
        self.calls: list[dict[str, str]] = []
        self.addCleanup(self.temp.cleanup)

    def geocoder(self, payload: dict, **settings) -> Geocoder:
        async def fetch(url: str, params: dict[str, str], timeout_s: float) -> dict:
            self.calls.append(params)
            return payload

        config = GeocodeConfig("key-1", **{"daily_limit": DEFAULT_DAILY_LIMIT, **settings})
        return Geocoder(config, self.cache, fetch=fetch)

    def test_the_same_spot_is_only_paid_for_once(self) -> None:
        """A drone hovering sends a dozen captures from one position.

        Rounded to about 11 m, so a hover is one call and the cache survives the
        restart that loading a model causes.
        """
        geocoder = self.geocoder(response())
        first = asyncio.run(geocoder.place(*TARGET))
        again = asyncio.run(geocoder.place(TARGET[0] + 0.00002, TARGET[1]))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(first.address(), again.address())
        self.assertEqual(again.landmark, "วัดหินดาด")

        # And a new geocoder, as after a restart, reads it off the disk.
        fresh = self.geocoder(response())
        self.assertEqual(asyncio.run(fresh.place(*TARGET)).address(), first.address())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(fresh.used_today, 1)

    def test_the_daily_cap_stops_calls_but_not_the_message(self) -> None:
        geocoder = self.geocoder(response(), daily_limit=2)
        for index in range(4):
            place = asyncio.run(geocoder.place(TARGET[0] + index / 100, TARGET[1]))
            if index < 2:
                self.assertTrue(place.found)
            else:
                self.assertFalse(place.found)
                self.assertIn("ครบโควตาที่ตั้งไว้วันนี้ (2 ครั้ง)", place.lines()[0])
        self.assertEqual(len(self.calls), 2)

    def test_a_failure_is_not_remembered_as_the_answer(self) -> None:
        # Otherwise a timeout would become that spot's address for good.
        async def fetch(url: str, params: dict[str, str], timeout_s: float) -> dict:
            self.calls.append(params)
            raise TimeoutError("no network")

        geocoder = Geocoder(GeocodeConfig("key-1"), self.cache, fetch=fetch)
        place = asyncio.run(geocoder.place(*TARGET))
        self.assertIn("เชื่อมต่อ Google ไม่สำเร็จ", place.lines()[0])
        self.assertEqual(asyncio.run(self.geocoder(response()).place(*TARGET)).road, "304")

    def test_nothing_found_is_remembered_so_it_is_not_asked_again(self) -> None:
        geocoder = self.geocoder({"status": "ZERO_RESULTS", "results": []})
        asyncio.run(geocoder.place(*TARGET))
        asyncio.run(geocoder.place(*TARGET))
        self.assertEqual(len(self.calls), 1)

    def test_the_key_is_sent_to_google_and_nowhere_else(self) -> None:
        asyncio.run(self.geocoder(response()).place(*TARGET))
        self.assertEqual(self.calls[0]["key"], "key-1")
        self.assertEqual(self.calls[0]["language"], "th")
        self.assertEqual(self.calls[0]["latlng"], "14.851000,102.020000")
        # The cache on disk holds the answer, never the key.
        self.assertNotIn("key-1", self.cache.read_text(encoding="utf-8"))


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "line.json"
        self.addCleanup(self.temp.cleanup)
        for name in ("TELEMETRY_OCR_GOOGLE_MAPS_KEY", "TELEMETRY_OCR_GEOCODE_DAILY_LIMIT"):
            os.environ.pop(name, None)

    def test_no_key_anywhere_means_the_feature_is_simply_off(self) -> None:
        self.assertIsNone(load_geocode_config(self.path))
        self.path.write_text(json.dumps({"to": "U1", "channel_access_token": "t"}), "utf-8")
        self.assertIsNone(load_geocode_config(self.path))

    def test_the_key_can_sit_beside_the_line_credentials(self) -> None:
        self.path.write_text(
            json.dumps({"google_maps_api_key": "from-file", "geocode_daily_limit": 50}), "utf-8"
        )
        settings = load_geocode_config(self.path)
        assert settings is not None
        self.assertEqual((settings.api_key, settings.daily_limit), ("from-file", 50))

    def test_the_environment_wins_so_a_key_need_never_touch_the_disk(self) -> None:
        self.path.write_text(json.dumps({"google_maps_api_key": "from-file"}), "utf-8")
        os.environ["TELEMETRY_OCR_GOOGLE_MAPS_KEY"] = "from-env"
        self.addCleanup(os.environ.pop, "TELEMETRY_OCR_GOOGLE_MAPS_KEY", None)
        settings = load_geocode_config(self.path)
        assert settings is not None
        self.assertEqual(settings.api_key, "from-env")
        self.assertEqual(settings.daily_limit, DEFAULT_DAILY_LIMIT)

    def test_a_nonsense_limit_falls_back_to_the_default(self) -> None:
        self.path.write_text(
            json.dumps({"google_maps_api_key": "k", "geocode_daily_limit": "soon"}), "utf-8"
        )
        settings = load_geocode_config(self.path)
        assert settings is not None
        self.assertEqual(settings.daily_limit, DEFAULT_DAILY_LIMIT)


if __name__ == "__main__":
    unittest.main()
