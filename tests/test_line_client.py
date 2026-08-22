"""Sending a capture to LINE.

Nothing here talks to LINE. The formatting is a pure function, and the push is
pointed at a local aiohttp server standing in for the real endpoint, so the
suite can assert on what would have been sent -- including that the token is not
in anything that comes back.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

HAS_WEB_DEPS = bool(importlib.util.find_spec("aiohttp"))

TOKEN = "test-channel-access-token-do-not-log"


@dataclass
class _Summary:
    status: str
    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0
    age_ms: float | None = None


@dataclass
class _Shot:
    session: str = "session-20260818-131353"
    frame_index: int = 0
    image: str = "frames/00000000.jpg"
    source_time_ms: float = 20478.7
    captured_at: str = "2026-08-18T05:12:23.641+00:00"
    bytes: int = 570875
    records: list[dict[str, Any]] = field(default_factory=list)
    detections: Any = field(default_factory=lambda: _Summary("ok", {"car": 3}, 3, 120.0))


def _record(field_id: str, value: Any, confirmed: bool = True, raw: str = "") -> dict[str, Any]:
    return {
        "field_id": field_id,
        "value": value,
        "raw_text": raw or str(value),
        "confidence": 0.97,
        "confirmed": confirmed,
        "image": "frames/00000000.jpg",
    }


class MessageFormattingTests(unittest.TestCase):
    def test_the_message_carries_the_readings_and_the_counts(self) -> None:
        from telemetry_ocr.line_client import format_capture_message

        shot = _Shot(
            records=[
                _record("acft_lat_dms", 14.871608),
                _record("acft_lon_dms", 102.019786),
                _record("acft_alt_m", 241.0),
                _record("lrf_m", 721.7),
                _record("sensor_angle_deg", 1.05),
            ]
        )
        text = format_capture_message(shot)
        self.assertIn("session-20260818-131353", text)
        self.assertIn("frames/00000000.jpg", text)
        # DMS, matching the overlay, rather than the decimal the pipeline stores.
        self.assertIn("14°52'17.8\"N", text)
        self.assertIn("721.7 m", text)
        self.assertIn("1.05°", text)
        self.assertIn("car 3", text)
        self.assertIn("รวม 3", text)

    def test_an_unconfirmed_reading_is_marked_not_dropped(self) -> None:
        """Hiding the hard fields would misreport what was on screen.

        The whole point of a capture is to say what the overlay showed, so a
        value temporal voting has not agreed on yet is shown with a marker
        rather than silently omitted.
        """
        from telemetry_ocr.line_client import format_capture_message

        text = format_capture_message(_Shot(records=[_record("lrf_m", 721.7, confirmed=False)]))
        self.assertIn("721.7 m?", text)

    def test_the_three_ways_of_having_no_counts_read_differently(self) -> None:
        # A bare zero cannot distinguish these, and they are fixed in completely
        # different places: load a model, wait, or accept there was nothing.
        from telemetry_ocr.line_client import describe_detections

        no_model = describe_detections(_Summary("no_model"))
        no_result = describe_detections(_Summary("no_result"))
        empty = describe_detections(_Summary("ok", {}, 0))
        self.assertNotEqual(no_model, no_result)
        self.assertNotEqual(no_result, empty)
        self.assertIn("โมเดล", no_model)
        self.assertIn("ไม่พบวัตถุ", empty)

    def test_counts_far_from_the_frame_say_so(self) -> None:
        # Detection runs at its own pace. A large gap means the numbers describe
        # a different moment than the picture, which the reader has to know.
        from telemetry_ocr.line_client import describe_detections

        fresh = describe_detections(_Summary("ok", {"car": 1}, 1, age_ms=120.0))
        stale = describe_detections(_Summary("ok", {"car": 1}, 1, age_ms=9000.0))
        self.assertNotIn("เก่ากว่า", fresh)
        self.assertIn("9.0", stale)

    def test_counts_are_ordered_so_the_same_scene_reads_the_same(self) -> None:
        from telemetry_ocr.line_client import describe_detections

        text = describe_detections(_Summary("ok", {"bus": 1, "car": 5, "person": 5}, 11))
        self.assertLess(text.index("car 5"), text.index("bus 1"))
        # Equal counts fall back to the name, so the order never wobbles.
        self.assertLess(text.index("car 5"), text.index("person 5"))

    def test_a_very_long_message_is_truncated_not_rejected(self) -> None:
        from telemetry_ocr.line_client import MAX_TEXT, format_capture_message

        shot = _Shot(records=[_record("lrf_m", "x" * 9000)])
        text = format_capture_message(shot)
        self.assertLessEqual(len(text), MAX_TEXT)


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "line.json"

    def test_nothing_configured_is_not_an_error(self) -> None:
        # Not having set LINE up is an ordinary state for this application.
        from telemetry_ocr.line_client import load_line_config

        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(load_line_config(self.path))

    def test_the_file_is_read_when_the_environment_is_empty(self) -> None:
        from telemetry_ocr.line_client import load_line_config

        self.path.write_text(
            json.dumps({"channel_access_token": TOKEN, "to": "U123456789"}), encoding="utf-8"
        )
        with mock.patch.dict("os.environ", {}, clear=True):
            config = load_line_config(self.path)
        assert config is not None
        self.assertEqual(config.to, "U123456789")

    def test_the_environment_wins_over_the_file(self) -> None:
        # A token in the environment never touches the disk, so it should not be
        # overridden by a file someone left behind.
        from telemetry_ocr.line_client import load_line_config

        self.path.write_text(
            json.dumps({"channel_access_token": "from-file", "to": "Ufile"}), encoding="utf-8"
        )
        env = {"TELEMETRY_OCR_LINE_TOKEN": TOKEN, "TELEMETRY_OCR_LINE_TO": "Uenv"}
        with mock.patch.dict("os.environ", env, clear=True):
            config = load_line_config(self.path)
        assert config is not None
        self.assertEqual(config.channel_access_token, TOKEN)
        self.assertEqual(config.to, "Uenv")

    def test_a_destination_is_masked_for_display(self) -> None:
        from telemetry_ocr.line_client import LineConfig

        self.assertEqual(LineConfig(TOKEN, "U1234567890abcd").masked_to, "U123...cd")
        self.assertEqual(LineConfig(TOKEN, "U123").masked_to, "****")


@unittest.skipUnless(HAS_WEB_DEPS, "aiohttp not installed")
class PushTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from aiohttp import web

        self.received: list[dict[str, Any]] = []
        self.reply_status = 200
        self.reply_body: dict[str, Any] = {}

        async def handler(request: Any) -> Any:
            self.received.append(
                {
                    "auth": request.headers.get("Authorization", ""),
                    "body": await request.json(),
                }
            )
            return web.json_response(
                self.reply_body,
                status=self.reply_status,
                headers={"x-line-request-id": "req-abc"},
            )

        app = web.Application()
        app.router.add_post("/v2/bot/message/push", handler)
        from aiohttp.test_utils import TestServer

        self.server = TestServer(app)
        await self.server.start_server()
        self.patch = mock.patch(
            "telemetry_ocr.line_client.PUSH_URL",
            str(self.server.make_url("/v2/bot/message/push")),
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    async def asyncTearDown(self) -> None:
        await self.server.close()

    async def _push(self, text: str = "hello") -> dict[str, Any]:
        from telemetry_ocr.line_client import LineConfig, push_text

        return await push_text(LineConfig(TOKEN, "U123"), text)

    async def test_a_successful_push_sends_a_bearer_token_and_the_text(self) -> None:
        result = await self._push("telemetry here")
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["request_id"], "req-abc")
        self.assertEqual(self.received[0]["auth"], f"Bearer {TOKEN}")
        self.assertEqual(self.received[0]["body"]["messages"][0]["text"], "telemetry here")

    async def test_the_token_is_never_in_the_result(self) -> None:
        """The result is handed back to the browser and into logs."""
        result = await self._push()
        self.assertNotIn(TOKEN, json.dumps(result))

    async def test_line_s_own_wording_reaches_the_caller(self) -> None:
        # 401, 400 and 429 are fixed in three different places. Flattening them
        # into "failed" would hide which one happened.
        self.reply_status = 401
        self.reply_body = {"message": "Invalid channel access token"}
        result = await self._push()
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], 401)
        self.assertIn("Invalid channel access token", result["line_message"])
        self.assertNotIn(TOKEN, json.dumps(result))

    async def test_a_quota_error_is_reported_as_itself(self) -> None:
        self.reply_status = 429
        self.reply_body = {"message": "You have reached your monthly limit."}
        result = await self._push()
        self.assertEqual(result["status"], 429)
        self.assertIn("monthly limit", result["line_message"])

    async def test_a_validation_error_includes_the_detail_line(self) -> None:
        self.reply_status = 400
        self.reply_body = {
            "message": "The request body has 1 error(s)",
            "details": [{"property": "to", "message": "must not be empty"}],
        }
        result = await self._push()
        self.assertIn("to", result["line_message"])
        self.assertIn("must not be empty", result["line_message"])

    async def test_an_unreachable_line_is_reported_not_raised(self) -> None:
        # The operator is standing at a browser; an exception here would surface
        # as a 500 with the type name and nothing about the cause.
        from telemetry_ocr.line_client import LineConfig, push_text

        with mock.patch(
            "telemetry_ocr.line_client.PUSH_URL", "http://127.0.0.1:1/v2/bot/message/push"
        ):
            result = await push_text(LineConfig(TOKEN, "U123"), "x", timeout_s=2.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], 0)
        self.assertTrue(result["line_message"])


if __name__ == "__main__":
    unittest.main()


class CoordinateFormattingTests(unittest.TestCase):
    """DMS, because that is what the overlay shows.

    A message quoting the same reading in decimal makes the reader convert
    before they can compare it with the screen they are looking at.
    """

    def test_degrees_minutes_seconds(self) -> None:
        from telemetry_ocr.line_client import format_dms

        self.assertEqual(format_dms(14.848431, is_latitude=True), '14°50\'54.4"N')
        self.assertEqual(format_dms(101.999197, is_latitude=False), '101°59\'57.1"E')

    def test_the_hemisphere_follows_the_sign(self) -> None:
        # A fixed "N " prefix would label a southern position wrongly, and the
        # reader would have no way to tell from the message.
        from telemetry_ocr.line_client import format_dms

        self.assertTrue(format_dms(-14.848431, is_latitude=True).endswith("S"))
        self.assertTrue(format_dms(-101.999197, is_latitude=False).endswith("W"))

    def test_seconds_rounding_to_sixty_carries(self) -> None:
        # 14.999999 is a minute short of the next degree; printing 59'60.0" is
        # arithmetically true and reads as a mistake.
        from telemetry_ocr.line_client import format_dms

        rendered = format_dms(14.99999999, is_latitude=True)
        self.assertNotIn("60.0", rendered)
        self.assertEqual(rendered, '15°00\'00.0"N')

    def test_the_map_link_carries_decimal_degrees(self) -> None:
        # Google Maps takes decimal, not DMS; the message shows one and links
        # the other.
        from telemetry_ocr.line_client import maps_link

        self.assertEqual(
            maps_link(14.845161, 102.006658),
            "https://www.google.com/maps?q=14.845161,102.006658",
        )

    def test_the_message_links_the_target(self) -> None:
        from telemetry_ocr.line_client import format_capture_message

        shot = _Shot(
            records=[
                _record("tag_lat_dms", 14.845161),
                _record("tag_lon_dms", 102.006658),
            ]
        )
        text = format_capture_message(shot)
        self.assertIn('14°50\'42.6"N', text)
        self.assertIn("google.com/maps?q=14.845161,102.006658", text)

    def test_a_half_read_position_gets_no_link(self) -> None:
        """A link built from one coordinate would point somewhere real and wrong.

        Which is worse than no link: the reader would check a location that was
        never claimed.
        """
        from telemetry_ocr.line_client import format_capture_message

        text = format_capture_message(_Shot(records=[_record("tag_lat_dms", 14.845161)]))
        self.assertNotIn("google.com/maps", text)


class ValuePolicyTests(unittest.TestCase):
    """Which of the two readings reaches the message.

    Measured over every capture on disk: the image alone confirmed 94.5% of
    fields but printed 5.5% as raw OCR text; live alone confirmed 88.2%; image
    with a live fallback confirmed 95.5% and printed 1.8% raw. The fallback is
    the part that had been missing.
    """

    def _verified(self, value, raw, live_value):
        from telemetry_ocr.capture_service import Verification, VerifiedField

        return Verification(
            status="ok",
            fields=[
                VerifiedField("lrf_m", value, raw, 0.9, value is not None,
                              agrees=False, live_value=live_value)
            ],
            agreed=0,
            disagreed=1,
        )

    def test_the_image_value_is_used_when_it_reads(self) -> None:
        from telemetry_ocr.line_client import format_capture_message

        shot = _Shot(records=[_record("lrf_m", 817.7)])
        shot.verification = self._verified(839.4, "839.4", 817.7)
        self.assertIn("839.4 m", format_capture_message(shot))

    def test_the_live_value_rescues_a_field_the_image_failed(self) -> None:
        # The exact case that reached a real message as "10z°00'23.7".
        from telemetry_ocr.line_client import format_capture_message

        shot = _Shot(records=[_record("lrf_m", 478.6)])
        shot.verification = self._verified(None, "4786n", 478.6)
        text = format_capture_message(shot)
        self.assertIn("478.6 m", text)
        self.assertNotIn("4786n", text)
        self.assertIn("ใช้ค่าจาก voting แทน", text)

    def test_raw_text_only_when_neither_could_read_it(self) -> None:
        from telemetry_ocr.line_client import format_capture_message

        shot = _Shot(records=[_record("lrf_m", None, confirmed=False, raw="4786n")])
        shot.verification = self._verified(None, "4786n", None)
        self.assertIn("4786n?", format_capture_message(shot))
