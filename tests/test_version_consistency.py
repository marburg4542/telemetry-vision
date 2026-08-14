"""The version is stated in four places; nothing but this joins them up.

pyproject declares it, telemetry_ocr.__init__ mirrors it, /api/health serves
that, and app.js hard-codes its own copy to compare against. Bump one and the
suite stays green while the page either warns about a mismatch that is not real
or, worse, stays silent about one that is. A comment saying "keep in step" is
not a forcing function.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import tomllib

from telemetry_ocr import __version__

ROOT = Path(__file__).resolve().parents[1]
WEB_STATIC = ROOT / "src" / "telemetry_ocr" / "web_static"


class VersionConsistencyTests(unittest.TestCase):
    def test_pyproject_matches_the_package(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            declared = tomllib.load(handle)["project"]["version"]
        self.assertEqual(declared, __version__)

    def test_the_page_knows_which_build_it_is(self) -> None:
        source = (WEB_STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn(f'const APP_BUILD = "{__version__}"', source)

    def test_the_service_worker_cache_is_versioned(self) -> None:
        # Only a name change evicts the icons and manifest, which are still
        # cache-first. It does not have to match the release, but it does have
        # to be a recognisable version rather than a bare word.
        source = (WEB_STATIC / "service-worker.js").read_text(encoding="utf-8")
        match = re.search(r'CACHE_NAME = "([^"]+)"', source)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertRegex(match.group(1), r"-v\d+$")


if __name__ == "__main__":
    unittest.main()
