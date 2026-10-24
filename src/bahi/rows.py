"""Step 2 of the register pipeline: cut each page into one strip per entry.

Usage: python -m bahi.rows <preprocess_dir> <out_dir>

Reads <name>_page.jpg and <name>_ink.png from step 1 and writes, per page:
  <name>/row_NN.png       ink-only strip for each entry (black on white)
  <name>_rows.jpg         debug view: row boundaries over the page
  rows.csv                one line per strip, the template for ground truth
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def amount_anchors(ink: np.ndarray) -> tuple[list[int], int]:
    """One anchor per entry: the amount written at the start of each row.

    Every entry starts with an amount ("100)", "200)") in a column at the
    left of the text. In a narrow strip over that column the row profile is
    a clean series of separate bumps, one per entry, which the dense,
    touching middle of the page never gives. Bumps that are clearly two
    entries merged (much taller than the median) are split evenly.
    """
    _, xs = np.where(ink > 0)
    if len(xs) == 0:
        return [], 55
    left, right = np.percentile(xs, 2), np.percentile(xs, 98)
    x0, x1 = int(left), int(left + 0.14 * (right - left))
    prof = (ink[:, x0:x1] > 0).sum(axis=1).astype(np.float32)
    prof = np.convolve(prof, np.ones(7) / 7, mode="same")
    on = prof > 0.12 * float(np.percentile(prof[prof > 0], 95))
    runs, y = [], 0
    while y < len(on):
        if not on[y]:
            y += 1
            continue
        a = y
        while y < len(on) and on[y]:
            y += 1
        runs.append((a, y, float(prof[a:y].sum())))
    if not runs:
        return [], 55
    med_mass = float(np.median([m for _, _, m in runs]))
    runs = [r for r in runs if r[2] >= 0.25 * med_mass]  # drop ditto marks and specks
    med_h = float(np.median([b - a for a, b, _ in runs]))
    anchors: list[int] = []
    for a, b, _ in runs:
        parts = max(1, round((b - a) / med_h)) if (b - a) > 1.6 * med_h else 1
        for i in range(parts):
            anchors.append(int(a + (i + 0.5) * (b - a) / parts))
    pitch = int(np.median(np.diff(anchors))) if len(anchors) > 1 else 55
    return anchors, pitch


def assign_to_anchors(ink: np.ndarray, anchors: list[int]) -> list[np.ndarray]:
    """Give each ink component to the entry whose anchor is vertically nearest."""
    rows = [np.zeros_like(ink) for _ in anchors]
    if not anchors:
        return rows
    centres = np.array(anchors, np.float32)
    n, cc, _, cent = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, n):
        rows[int(np.argmin(np.abs(centres - cent[i][1])))][cc == i] = 255
    return rows


def crop_strip(row: np.ndarray, pad: int = 8) -> np.ndarray | None:
    ys, xs = np.where(row > 0)
    if len(xs) < 80:  # too little ink to be an entry
        return None
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, row.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, row.shape[1])
    return 255 - row[y0:y1, x0:x1]


def main(argv: list[str]) -> None:
    src, out = Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for page_path in sorted(src.glob("*_page.jpg")):
        stem = page_path.stem.removesuffix("_page")
        page = cv2.imread(str(page_path))
        ink_img = cv2.imread(str(src / f"{stem}_ink.png"), cv2.IMREAD_GRAYSCALE)
        if page is None or ink_img is None:
            raise ValueError(f"missing step-1 output for {stem}")
        ink = 255 - ink_img
        heads, _ = amount_anchors(ink)
        strips = [crop_strip(r) for r in assign_to_anchors(ink, heads)]
        (out / stem).mkdir(exist_ok=True)
        dbg = page.copy()
        colours = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 140, 255)]
        kept = 0
        for y, strip in zip(heads, strips):
            if strip is None:
                continue
            kept += 1
            name = f"row_{kept:02d}.png"
            cv2.imwrite(str(out / stem / name), strip)
            col = colours[kept % len(colours)]
            cv2.line(dbg, (0, y), (dbg.shape[1], y), col, 2)
            cv2.putText(dbg, str(kept), (5, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            records.append({"page": stem, "row": kept, "image": f"{stem}/{name}", "amount": "", "name": "", "place": "", "note": ""})
        cv2.imwrite(str(out / f"{stem}_rows.jpg"), dbg, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"{stem}: {len(heads)} amount anchors, {kept} entry rows")
    with open(out / "rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page", "row", "image", "amount", "name", "place", "note"])
        w.writeheader()
        w.writerows(records)
    print(f"{len(records)} rows total -> {out / 'rows.csv'}")


if __name__ == "__main__":
    main(sys.argv)
