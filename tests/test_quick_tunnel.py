"""Reading a quick tunnel's address back out of cloudflared.

The address is assigned fresh on every start, so it cannot be configured -- it
has to be parsed from the process output, and the parsing is the part that
breaks silently when cloudflared changes how it prints.
"""

from __future__ import annotations

import unittest
from unittest import mock

# Real cloudflared output, box drawing and all. The banner is decoration and can
# change; the URL is the thing being matched.
CLOUDFLARED_OUTPUT = """2026-08-18T09:00:00Z INF Thank you for trying Cloudflare Tunnel.
2026-08-18T09:00:00Z INF Requesting new quick Tunnel on trycloudflare.com...
+--------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time   |
|  to be reachable):                                                        |
|  https://celebrity-genesis-gpl-brian.trycloudflare.com                    |
+--------------------------------------------------------------------------+
2026-08-18T09:00:01Z INF Connection registered connIndex=0
"""


class UrlParsingTests(unittest.TestCase):
    def test_the_address_is_found_in_the_banner(self) -> None:
        from telemetry_ocr.quick_tunnel import _URL_PATTERN

        match = _URL_PATTERN.search(CLOUDFLARED_OUTPUT)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(
            match.group(0), "https://celebrity-genesis-gpl-brian.trycloudflare.com"
        )

    def test_the_trailing_box_border_is_not_swallowed(self) -> None:
        # The URL sits inside a table whose right edge is a pipe padded with
        # spaces; a greedy pattern would take them along and produce a URL that
        # fetches nothing.
        from telemetry_ocr.quick_tunnel import _URL_PATTERN

        line = "|  https://a-b-c-d.trycloudflare.com     |"
        self.assertEqual(
            _URL_PATTERN.search(line).group(0), "https://a-b-c-d.trycloudflare.com"
        )

    def test_unrelated_output_yields_nothing(self) -> None:
        from telemetry_ocr.quick_tunnel import _URL_PATTERN

        self.assertIsNone(_URL_PATTERN.search("INF Connection registered connIndex=0"))


class MissingBinaryTests(unittest.TestCase):
    def test_a_missing_cloudflared_is_reported_not_raised(self) -> None:
        """The server keeps working without it.

        Images are an optional extra; failing to find cloudflared must not take
        the telemetry send down with it.
        """
        from telemetry_ocr.quick_tunnel import QuickTunnel

        with mock.patch("telemetry_ocr.quick_tunnel.find_cloudflared", return_value=None):
            tunnel = QuickTunnel(8766)
        self.assertIsNone(tunnel.start())
        status = tunnel.status()
        self.assertFalse(status["available"])
        self.assertIn("cloudflared", str(status["error"]))

    def test_stopping_something_never_started_is_harmless(self) -> None:
        from telemetry_ocr.quick_tunnel import QuickTunnel

        with mock.patch("telemetry_ocr.quick_tunnel.find_cloudflared", return_value=None):
            QuickTunnel(8766).stop()

    def test_the_url_is_dropped_when_the_process_ends(self) -> None:
        # An address whose tunnel has gone would produce image messages that
        # fetch nothing, which is worse than sending no image at all.
        from telemetry_ocr.quick_tunnel import QuickTunnel

        with mock.patch("telemetry_ocr.quick_tunnel.find_cloudflared", return_value=None):
            tunnel = QuickTunnel(8766)

        class _Ended:
            stdout = iter(CLOUDFLARED_OUTPUT.splitlines(keepends=True))

            def poll(self) -> int:
                return 0

        tunnel._process = _Ended()  # type: ignore[assignment]
        tunnel._read_output()
        self.assertIsNone(tunnel.url)
        self.assertEqual(tunnel.status()["error"], "cloudflared exited")


class PortTargetTests(unittest.TestCase):
    def test_the_tunnel_is_aimed_at_the_share_port_not_the_api(self) -> None:
        """The whole security argument for the split.

        A quick tunnel takes no ingress rules, so whatever port it is given is
        forwarded in full. Pointing it at the application would put the entire
        API on the internet.
        """
        from telemetry_ocr.quick_tunnel import QuickTunnel

        captured: dict[str, list[str]] = {}

        class _Popen:
            def __init__(self, args: list[str], **kwargs: object) -> None:
                captured["args"] = args
                self.stdout = iter(())

            def poll(self) -> None:
                return None

        with (
            mock.patch("telemetry_ocr.quick_tunnel.find_cloudflared", return_value="cf.exe"),
            mock.patch("telemetry_ocr.quick_tunnel.subprocess.Popen", _Popen),
        ):
            QuickTunnel(8766).start(wait=False)

        self.assertIn("--url", captured["args"])
        target = captured["args"][captured["args"].index("--url") + 1]
        self.assertEqual(target, "http://127.0.0.1:8766")
        # Loopback, so nothing on the local network reaches it either.
        self.assertNotIn("8765", target)


if __name__ == "__main__":
    unittest.main()
