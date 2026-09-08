"""What is left of the month's allowance.

The account this was written against is on a 300-message plan. Running into that
ceiling shows up as a 429 with a capture already taken and no indication of when
the allowance comes back, so the number is worth knowing before the press rather
than after it.

LINE reports the ceiling and the consumption from two different endpoints, and
the ceiling is absent entirely -- not large -- on an unmetered plan.
"""

from __future__ import annotations

import unittest


class QuotaArithmeticTests(unittest.TestCase):
    def test_remaining_is_the_ceiling_less_what_went(self) -> None:
        from telemetry_ocr.line_client import Quota

        self.assertEqual(Quota(limit=300, used=2).remaining, 298)

    def test_an_unmetered_plan_has_no_remaining_rather_than_a_large_one(self) -> None:
        """LINE answers type "none" with no value at all.

        Reading a missing value as zero would report an account with no limit as
        an account with nothing left, and the button would refuse to send.
        """
        from telemetry_ocr.line_client import Quota

        quota = Quota(limit=None, used=4200)
        self.assertIsNone(quota.remaining)
        self.assertFalse(quota.exhausted)
        self.assertFalse(quota.warn)
        self.assertTrue(quota.as_dict()["unlimited"])

    def test_usage_past_the_ceiling_does_not_go_negative(self) -> None:
        # LINE counts sends this app did not make -- a broadcast from the OA
        # Manager, say -- so used can exceed the limit this server knows about.
        from telemetry_ocr.line_client import Quota

        self.assertEqual(Quota(limit=300, used=305).remaining, 0)
        self.assertTrue(Quota(limit=300, used=305).exhausted)

    def test_a_send_costs_one_however_many_pictures_ride_along(self) -> None:
        """LINE counts a push by destinations, not by message objects.

        Their own example is four message objects to a room of five people
        counting as five. One capture goes to one destination, so batching the
        text and up to four pictures into a single request costs 1.
        """
        from telemetry_ocr.line_client import Quota

        self.assertEqual(Quota(limit=300, used=2).spend().used, 3)

    def test_the_warning_band_is_a_tenth_of_the_plan(self) -> None:
        from telemetry_ocr.line_client import Quota, quota_warning_threshold

        self.assertEqual(quota_warning_threshold(300), 30)
        self.assertFalse(Quota(limit=300, used=269).warn)
        self.assertTrue(Quota(limit=300, used=270).warn)

    def test_a_small_plan_still_gets_a_usable_warning(self) -> None:
        # A tenth of 50 is 5, which on a plan that small is one afternoon.
        from telemetry_ocr.line_client import Quota, quota_warning_threshold

        self.assertEqual(quota_warning_threshold(50), 10)
        self.assertTrue(Quota(limit=50, used=40).warn)

    def test_an_exhausted_quota_is_also_a_warning(self) -> None:
        from telemetry_ocr.line_client import Quota

        quota = Quota(limit=300, used=300)
        self.assertTrue(quota.exhausted)
        self.assertTrue(quota.warn)

    def test_a_failed_lookup_is_not_reported_as_unlimited(self) -> None:
        """The dangerous confusion.

        "No limit" and "could not read the limit" both leave `limit` as None,
        and treating the second as the first would show an account with 3
        messages left as unmetered.
        """
        from telemetry_ocr.line_client import Quota

        payload = Quota(error="โควต้า: token ไม่ถูกต้อง (401)").as_dict()
        self.assertFalse(payload["unlimited"])
        self.assertTrue(payload["error"])
        self.assertIsNone(payload["remaining"])


class QuotaFetchTests(unittest.IsolatedAsyncioTestCase):
    """The two calls, against a local server rather than against LINE."""

    async def _serve(self, quota_body, consumption_body, status=200):
        from aiohttp import web

        async def quota(_request):
            return web.json_response(quota_body, status=status)

        async def consumption(_request):
            return web.json_response(consumption_body, status=status)

        app = web.Application()
        app.router.add_get("/v2/bot/message/quota", quota)
        app.router.add_get("/v2/bot/message/quota/consumption", consumption)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.addAsyncCleanup(runner.cleanup)
        port = runner.addresses[0][1]
        return f"http://127.0.0.1:{port}"

    async def _fetch(self, base, **kwargs):
        from unittest import mock

        from telemetry_ocr import line_client

        with (
            mock.patch.object(line_client, "QUOTA_URL", f"{base}/v2/bot/message/quota"),
            mock.patch.object(
                line_client, "CONSUMPTION_URL", f"{base}/v2/bot/message/quota/consumption"
            ),
        ):
            return await line_client.fetch_quota(
                line_client.LineConfig("token", "U1"), **kwargs
            )

    async def test_a_limited_plan_is_read_from_both_endpoints(self) -> None:
        base = await self._serve({"type": "limited", "value": 300}, {"totalUsage": 2})
        quota = await self._fetch(base)
        self.assertEqual((quota.limit, quota.used, quota.remaining), (300, 2, 298))
        self.assertEqual(quota.error, "")

    async def test_an_unmetered_plan_reports_usage_and_no_ceiling(self) -> None:
        base = await self._serve({"type": "none"}, {"totalUsage": 9001})
        quota = await self._fetch(base)
        self.assertIsNone(quota.limit)
        self.assertEqual(quota.used, 9001)
        self.assertTrue(quota.as_dict()["unlimited"])

    async def test_a_bad_token_comes_back_as_an_error_not_an_exception(self) -> None:
        # Not knowing the quota is a worse page, not a broken one: the send does
        # not depend on it, and LINE enforces the real limit either way.
        base = await self._serve({"message": "Invalid access token"}, {}, status=401)
        quota = await self._fetch(base)
        self.assertIn("401", quota.error)
        self.assertIsNone(quota.remaining)

    async def test_an_unreachable_line_does_not_raise(self) -> None:
        from telemetry_ocr import line_client

        # A port nothing is listening on, reached with a short timeout.
        quota = await self._fetch("http://127.0.0.1:9", timeout_s=1.0)
        self.assertTrue(quota.error)
        self.assertIsInstance(quota, line_client.Quota)


if __name__ == "__main__":
    unittest.main()
