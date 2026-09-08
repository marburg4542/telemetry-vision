"""Render a Markdown document to PDF through headless Chrome.

Chrome rather than ReportLab, for the same reason `vision_lab.report` gives:
Thai needs a real text shaper, and the browser already has one. ReportLab would
need the font embedded and the clusters shaped by hand.

Usage:
    python scripts/markdown_to_pdf.py docs/codebase-walkthrough.md out.pdf
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Imported after the sys.path line above, which is what makes it importable at
# all. It carries no suppression comment: E402 is already in pyproject's global
# ignore list, so one would be unused and RUF100 fails the CI lint job over it.
from vision_lab.report import find_browser

# Ships with Windows and shapes Thai correctly; the rest are fallbacks for
# anyone building this on another machine.
BODY_FONT = '"Leelawadee UI", "TH Sarabun New", Tahoma, "Noto Sans Thai", sans-serif'
MONO_FONT = '"Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace'

CSS = f"""
@page {{ size: A4; margin: 18mm 16mm 20mm; }}
* {{ box-sizing: border-box; }}
body {{
  font-family: {BODY_FONT};
  font-size: 10.5pt;
  line-height: 1.62;
  color: #14191f;
  margin: 0;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}}
h1, h2, h3, h4 {{ line-height: 1.35; break-after: avoid; page-break-after: avoid; }}
h1 {{
  font-size: 19pt; color: #12414f; margin: 0 0 12pt;
  padding-bottom: 5pt; border-bottom: 1.6pt solid #b9c6cf;
  break-before: page; page-break-before: always;
}}
h1:first-of-type {{ break-before: auto; page-break-before: auto; }}
h2 {{ font-size: 14pt; color: #12414f; margin: 17pt 0 7pt; }}
h3 {{ font-size: 11.5pt; margin: 13pt 0 5pt; }}
h4 {{ font-size: 10.5pt; margin: 10pt 0 4pt; }}
p {{ margin: 0 0 7pt; }}
ul, ol {{ margin: 0 0 8pt; padding-left: 20pt; }}
li {{ margin: 0 0 3pt; }}
code {{
  font-family: {MONO_FONT}; font-size: 9pt;
  background: #eef2f5; padding: 0.5pt 3pt; border-radius: 2pt; color: #0f3a47;
}}
pre {{
  font-family: {MONO_FONT}; font-size: 8.1pt; line-height: 1.42;
  background: #f5f7f9; border: 0.6pt solid #d3dce3;
  border-left: 2.6pt solid #12414f;
  padding: 7pt 9pt; margin: 0 0 9pt; border-radius: 3pt;
  white-space: pre; overflow-x: hidden;
  break-inside: avoid; page-break-inside: avoid;
}}
pre code {{ background: none; padding: 0; font-size: inherit; color: inherit; }}
table {{
  border-collapse: collapse; width: 100%; margin: 0 0 10pt;
  font-size: 8.8pt; break-inside: avoid; page-break-inside: avoid;
}}
th, td {{ border: 0.6pt solid #c3ced6; padding: 3.4pt 5pt; text-align: left; vertical-align: top; }}
th {{ background: #e7eef3; font-weight: 600; }}
tr:nth-child(even) td {{ background: #fafbfc; }}
blockquote {{
  margin: 0 0 9pt; padding: 6pt 10pt;
  background: #fbf6e9; border-left: 3pt solid #b08a3e; border-radius: 2pt;
  break-inside: avoid; page-break-inside: avoid;
}}
blockquote p:last-child {{ margin-bottom: 0; }}
hr {{ border: 0; border-top: 0.6pt solid #d3dce3; margin: 12pt 0; }}
a {{ color: #12414f; text-decoration: none; }}
.cover {{ text-align: center; padding-top: 62mm; break-after: page; page-break-after: always; }}
.cover .kicker {{ font-size: 12pt; color: #5d6b76; margin-bottom: 6pt; }}
.cover h1 {{
  border: 0; break-before: auto; page-break-before: auto;
  font-size: 25pt; margin: 0 0 14pt; padding: 0;
}}
.cover .meta {{ font-size: 10pt; color: #5d6b76; margin-top: 26pt; line-height: 1.9; }}
"""


def inline(text: str) -> str:
    """Bold, inline code and links, on already-escaped text."""
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    return out


def convert(markdown: str) -> str:
    lines = markdown.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r")

        if not line.strip():
            i += 1
            continue

        if line.startswith("```"):
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i].rstrip("\r"))
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(body)) + "</code></pre>")
            continue

        if line.strip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = lines[i].strip().rstrip("\r").strip("|").split("|")
                rows.append([c.strip() for c in cells])
                i += 1
            if len(rows) >= 2:
                head, body = rows[0], rows[2:]
                cols = len(head)
                out.append("<table><thead><tr>")
                out += [f"<th>{inline(c)}</th>" for c in head]
                out.append("</tr></thead><tbody>")
                for row in body:
                    out.append("<tr>")
                    out += [
                        f"<td>{inline(row[n]) if n < len(row) else ''}</td>"
                        for n in range(cols)
                    ]
                    out.append("</tr>")
                out.append("</tbody></table>")
            continue

        if line.startswith(">"):
            body = []
            while i < len(lines) and lines[i].startswith(">"):
                body.append(re.sub(r"^>\s?", "", lines[i].rstrip("\r")))
                i += 1
            paragraphs = "".join(
                f"<p>{inline(part)}</p>" for part in body if part.strip()
            )
            out.append(f"<blockquote>{paragraphs}</blockquote>")
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        if line.strip() == "---":
            out.append("<hr />")
            i += 1
            continue

        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i].rstrip("\r")):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i].rstrip("\r")))
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue

        numbered = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        if numbered:
            items = []
            while i < len(lines) and re.match(r"^\s*\d+[.)]\s+", lines[i].rstrip("\r")):
                items.append(re.sub(r"^\s*\d+[.)]\s+", "", lines[i].rstrip("\r")))
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ol>")
            continue

        body = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip("\r")
            if not nxt.strip() or re.match(r"^(#{1,4}\s|```|>|\s*[-*]\s|\s*\d+[.)]\s)", nxt):
                break
            if nxt.strip().startswith("|") or nxt.strip() == "---":
                break
            body.append(nxt)
            i += 1
        out.append(f"<p>{inline(' '.join(body))}</p>")

    return "\n".join(out)


def build(markdown_path: Path, pdf_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")

    # The first heading and the line under it become the cover.
    title = "เอกสาร"
    match = re.search(r"^#\s+(.*)$", text, re.M)
    if match:
        title = match.group(1).strip()
        text = text[match.end() :]

    cover = f"""
<div class="cover">
  <div class="kicker">เอกสารประกอบโครงการ</div>
  <h1>{html.escape(title)}</h1>
  <div class="meta">
    telemetry-vision · เวอร์ชัน 0.5.0<br />
    สร้างจาก {html.escape(markdown_path.name)}
  </div>
</div>
"""
    document = (
        "<!doctype html><html lang='th'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f"{cover}{convert(text)}</body></html>"
    )

    browser = find_browser()
    print(f"เบราว์เซอร์: {browser}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        source = tmp_dir / "document.html"
        source.write_text(document, encoding="utf-8")
        command = [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--user-data-dir={tmp_dir / 'profile'}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            source.as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    if result.returncode != 0 or not pdf_path.is_file():
        raise SystemExit(f"Chrome ล้มเหลว ({result.returncode}): {result.stderr[:400]}")
    print(f"เขียน {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    build(Path(sys.argv[1]), Path(sys.argv[2]).resolve())
