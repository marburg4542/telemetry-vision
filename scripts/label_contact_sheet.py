"""Lay a field's exported crops out on a few sheets instead of 300 files.

Labelling is the slow half of raising accuracy, and most of the slowness is not
reading -- it is opening a file, reading four characters, closing it, and doing
that nine hundred times. One sheet holds a column of crops with the frame index
beside each, so a screen of them can be read in one pass and typed into the CSV
in one pass.

Crops are drawn at their own scale, magnified, with nothing else on the sheet:
the point is to read what the OSD says, and a picture that has been resampled to
fit a grid is a picture that can be misread.

Usage:
    python scripts/label_contact_sheet.py output/labels-pass2
    python scripts/label_contact_sheet.py output/labels-pass2 --field ir_detail
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

# Big enough to read a 20-pixel-high OSD glyph on a normal screen without
# guessing, small enough that a sheet still holds a useful number of rows.
SCALE = 3
ROWS_PER_SHEET = 25
# Room for the frame index to the left of every crop.
INDEX_WIDTH = 130
MARGIN = 12


def read_manifest(root: Path) -> list[dict[str, str]]:
    manifest = root / "labels.csv"
    if not manifest.is_file():
        raise SystemExit(f"No labels.csv in {root}. Run `telemetry-ocr export-crops` first.")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_sheets(
    root: Path, field: str, rows: list[dict[str, str]], only_blank: bool
) -> list[Path]:
    import cv2
    import numpy as np

    chosen = [r for r in rows if r["field_id"] == field]
    if only_blank:
        chosen = [r for r in chosen if not (r.get("expected") or "").strip()]
    if not chosen:
        return []
    chosen.sort(key=lambda r: int(r["frame_index"]))

    out_dir = root / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for sheet_no, start in enumerate(range(0, len(chosen), ROWS_PER_SHEET), start=1):
        batch = chosen[start : start + ROWS_PER_SHEET]
        tiles: list[Any] = []
        for entry in batch:
            image = cv2.imread(str(root / entry["crop_path"]))
            if image is None:
                continue
            image = cv2.resize(
                image,
                (image.shape[1] * SCALE, image.shape[0] * SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            tiles.append((entry["frame_index"], image))
        if not tiles:
            continue

        width = INDEX_WIDTH + max(t.shape[1] for _, t in tiles) + MARGIN * 2
        height = sum(t.shape[0] + MARGIN for _, t in tiles) + MARGIN + 40
        sheet = np.full((height, width, 3), 255, np.uint8)

        # ASCII only. cv2.putText has one Hershey font and no glyph for anything
        # outside it, so a middot renders as two question marks on the sheet.
        title = f"{field}   sheet {sheet_no}   -   frame index on the left"
        cv2.putText(sheet, title, (MARGIN, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

        y = 40
        for frame_index, tile in tiles:
            cv2.putText(
                sheet, frame_index, (MARGIN, y + tile.shape[0] // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1,
            )
            sheet[y : y + tile.shape[0], INDEX_WIDTH : INDEX_WIDTH + tile.shape[1]] = tile
            # A hairline under each row, so the eye does not slip between crops
            # of nearly identical text.
            cv2.line(
                sheet, (MARGIN, y + tile.shape[0] + MARGIN // 2),
                (width - MARGIN, y + tile.shape[0] + MARGIN // 2), (215, 215, 215), 1,
            )
            y += tile.shape[0] + MARGIN

        path = out_dir / f"{field}_{sheet_no:02d}.png"
        cv2.imwrite(str(path), sheet)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="The --output-dir given to export-crops")
    parser.add_argument("--field", help="Only this field id. Default: every field present")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Include rows already labelled. Default: only the ones still blank",
    )
    args = parser.parse_args()

    rows = read_manifest(args.root)
    fields = [args.field] if args.field else sorted({r["field_id"] for r in rows})
    total = 0
    for field in fields:
        sheets = build_sheets(args.root, field, rows, only_blank=not args.all_rows)
        for sheet in sheets:
            print(sheet)
        total += len(sheets)
    if not total:
        print("Nothing left to label." if not args.all_rows else "No crops found.")
    else:
        print(f"\n{total} sheet(s) in {args.root / 'sheets'}")
        print("Read a sheet, then fill `expected` for those frame indexes in labels.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
