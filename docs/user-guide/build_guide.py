"""Build the user guide PDF from guide.html and the captured screenshots.

Two passes through headless Chrome. The first finds the page every chapter
starts on; the second writes those numbers into the table of contents. Chrome
has no target-counter(), so the pages are read back out of the first PDF
through an invisible ASCII anchor at each chapter heading -- the Thai heading
text itself does not survive text extraction reliably enough to search for.

Screenshots marked with data-marks get numbered callouts placed from the element
rectangles capture_screens.py recorded beside them, so a marker sits on the
control it names rather than where it was on the day the guide was written.

The screenshots are captured as 2x PNG and printed as JPEG copies in
assets/print/: the PNGs alone made a 24 MB PDF for no visible gain on paper.

Usage:
    .venv-cuda/Scripts/python.exe docs/user-guide/build_guide.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_lab.report import find_browser

SOURCE = HERE / "guide.html"
OUTPUT = ROOT / "docs" / "User-Guide-telemetry-vision.pdf"
SCREENS = HERE / "assets" / "screens"
PRINT = HERE / "assets" / "print"
PRINT_WIDTH = 2000
# An element covering more than this share of the frame is a region, not a
# control. It gets a pin but no outline, which would only box half the picture.
OUTLINE_MAX_AREA = 0.06


def print_copies() -> None:
    import cv2

    PRINT.mkdir(parents=True, exist_ok=True)
    for png in SCREENS.glob("*.png"):
        target = PRINT / f"{png.stem}.jpg"
        if target.exists() and target.stat().st_mtime >= png.stat().st_mtime:
            continue
        image = cv2.imread(str(png))
        height, width = image.shape[:2]
        if width > PRINT_WIDTH:
            image = cv2.resize(
                image, (PRINT_WIDTH, round(height * PRINT_WIDTH / width)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 88])


def place_marks(html: str) -> str:
    """Put numbered callouts on every <figure data-screen=... data-marks=...>."""

    def figure(match: re.Match) -> str:
        attrs, body = match.group(1), match.group(2)
        screen = re.search(r'data-screen="([^"]+)"', attrs).group(1)
        marks_attr = re.search(r'data-marks="([^"]*)"', attrs)
        image = f'<img src="assets/print/{screen}.jpg" alt="">'
        pins = []
        if marks_attr and marks_attr.group(1).strip():
            layout = json.loads((SCREENS / f"{screen}.json").read_text("utf-8"))
            view_w, view_h = layout["viewport"]
            for item in marks_attr.group(1).split(","):
                key, number = item.strip().split(":")
                if key not in layout["rects"]:
                    raise KeyError(f"{screen}: no rectangle recorded for {key}")
                x, y, w, h = layout["rects"][key]
                if (w * h) / (view_w * view_h) <= OUTLINE_MAX_AREA:
                    pins.append(
                        f'<span class="hot" style="left:{x / view_w * 100:.2f}%;'
                        f"top:{y / view_h * 100:.2f}%;width:{w / view_w * 100:.2f}%;"
                        f'height:{h / view_h * 100:.2f}%"></span>'
                    )
                # On the corner itself rather than inside the element, so the pin
                # covers the edge of a button instead of the start of its label.
                left = min(max(x / view_w * 100, 1.2), 98.0)
                top = min(max(y / view_h * 100, 1.8), 97.0)
                pins.append(
                    f'<span class="pin" style="left:{left:.2f}%;top:{top:.2f}%">{number}</span>'
                )
        return f'<figure {attrs}><div class="frame">{image}{"".join(pins)}</div>{body}</figure>'

    html = re.sub(r"<figure ([^>]*data-screen=[^>]*)>(.*?)</figure>", figure, html, flags=re.S)
    # Plain <img> figures reference the screenshots directly; print the copies.
    return re.sub(r"assets/screens/([\w-]+)\.png", r"assets/print/\1.jpg", html)


def render(html: str, pdf: Path) -> None:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", dir=HERE, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(html)
        page = Path(handle.name)
    try:
        with tempfile.TemporaryDirectory(prefix="guide-pdf-") as profile:
            subprocess.run(
                [
                    find_browser(), "--headless=new", "--disable-gpu", "--no-first-run",
                    f"--user-data-dir={profile}", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=15000",
                    f"--print-to-pdf={pdf}", page.as_uri(),
                ],
                check=True, capture_output=True, timeout=240,
            )
    finally:
        page.unlink(missing_ok=True)


def chapter_pages(pdf: Path) -> dict[str, int]:
    from pypdf import PdfReader

    found: dict[str, int] = {}
    for number, page in enumerate(PdfReader(str(pdf)).pages, start=1):
        for anchor in re.findall(r"ZZ(\w+?)ZZ", page.extract_text() or ""):
            found.setdefault(anchor, number)
    return found


def main() -> None:
    print_copies()
    html = place_marks(SOURCE.read_text("utf-8"))
    with tempfile.TemporaryDirectory() as scratch:
        first = Path(scratch) / "pass1.pdf"
        render(html, first)
        pages = chapter_pages(first)
    missing = sorted(set(re.findall(r'data-toc="(\w+)"', html)) - set(pages))
    if missing:
        print("warning: no page found for", ", ".join(missing))
    html = re.sub(
        r'<span class="toc-page" data-toc="(\w+)"></span>',
        lambda m: f'<span class="toc-page">{pages.get(m.group(1), "")}</span>',
        html,
    )
    render(html, OUTPUT)
    size = OUTPUT.stat().st_size / 1024 / 1024
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({size:.1f} MB, chapters on pages {pages})")


if __name__ == "__main__":
    main()
