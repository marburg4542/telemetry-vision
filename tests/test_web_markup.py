"""Structural checks on the page, for mistakes the eye does not catch.

Both of these come from real breakage. A patch applied twice left three buttons
with the same id sitting next to each other; getElementById returns the first, so
two of them were dead controls that looked exactly like the working one and the
page carried on without complaint. And every handle in app.js is looked up by id
at load time, so an element renamed or removed in the markup produces null and a
TypeError only on the code path that happens to touch it.
"""

from __future__ import annotations

import re
import unittest
from collections import Counter
from pathlib import Path

WEB_STATIC = Path(__file__).resolve().parents[1] / "src" / "telemetry_ocr" / "web_static"
MARKUP = (WEB_STATIC / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB_STATIC / "app.js").read_text(encoding="utf-8")

MARKUP_IDS = re.findall(r'\bid="([^"]+)"', MARKUP)


class MarkupTests(unittest.TestCase):
    def test_no_id_appears_twice(self) -> None:
        repeated = sorted(name for name, count in Counter(MARKUP_IDS).items() if count > 1)
        self.assertEqual(
            repeated,
            [],
            f"duplicated element ids: {repeated}. Only the first is reachable by "
            "getElementById; the rest are controls that look live and do nothing.",
        )

    def test_every_handle_app_js_asks_for_exists(self) -> None:
        # `const ui = { name: $("elementId"), ... }` is resolved once at load, so
        # a missing element is a null that surfaces far from the cause.
        requested = set(re.findall(r'\$\("([^"]+)"\)', APP_JS))
        missing = sorted(requested - set(MARKUP_IDS))
        self.assertEqual(missing, [], f"app.js reads ids the page does not define: {missing}")


if __name__ == "__main__":
    unittest.main()
