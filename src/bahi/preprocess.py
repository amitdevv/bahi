"""Step 1 of the register pipeline: crop the page, keep the pen ink, fix orientation.

Usage: python -m bahi.preprocess <raw_dir> <out_dir>

For each photo this writes:
  <name>_page.jpg   the page, cropped, perspective-corrected and upright
  <name>_ink.png    only the handwriting (black on white)
  <name>_mask.png   debug overlay: kept ink in red over the page
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

WORK_WIDTH = 1600  # px; every page is resized to this width before analysis


@dataclass
class PageResult:
    page: np.ndarray  # BGR, upright
    ink: np.ndarray  # uint8 mask, 255 = handwriting
    rotation: int  # degrees applied to make the page upright
    ink_cluster_lab: tuple[float, float, float]


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Set every background region not reachable from the border to foreground."""
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def _page_hypotheses(img: np.ndarray) -> list[np.ndarray]:
    """Candidate page outlines (convex hulls) under several mask settings.

    The page is a bright, low-saturation, smooth region. Paper is smooth
    while the patterned cloth is not, so a local-variance test keeps the
    cloth's white flowers out. Dense handwriting also fails that test and
    punches holes, so holes are filled, then a large opening cuts thin
    bridges to the cloth. No single setting works on every photo, so each
    setting proposes a hull and the caller picks the best.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (31, 31))
    local_std = np.sqrt(np.maximum(cv2.blur(gray * gray, (31, 31)) - mean * mean, 0))
    hulls = []
    for close, max_std in ((25, 28), (51, 28), (25, 40), (51, 40)):
        paper = ((hsv[..., 1] < 70) & (mean > 140) & (local_std < max_std)).astype(np.uint8) * 255
        paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close, close)))
        paper = _fill_holes(paper)
        paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (61, 61)))
        contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hulls.append(cv2.convexHull(max(contours, key=cv2.contourArea)))
    return hulls


def _hull_score(hull: np.ndarray, image_area: float) -> float:
    """Prefer large outlines that fill their own rotated bounding rectangle.

    A clean page fills its minimum-area rectangle almost completely; a page
    that leaked into the cloth, or one cut in half, does not, or is too small.
    """
    area = cv2.contourArea(hull)
    (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
    if rw * rh == 0 or area < 0.2 * image_area or not 0.4 < rw / rh < 2.5:
        return 0.0
    rectangularity = area / (rw * rh)
    return rectangularity**4 * (area / image_area)


def crop_page(img: np.ndarray) -> np.ndarray:
    """Find the page against the background and warp it flat.

    Uses the best-scoring outline from `_page_hypotheses`. If no clean
    four-corner outline is found, falls back to the tightest rotated rectangle.
    """
    image_area = float(img.shape[0] * img.shape[1])
    scored = [(_hull_score(h, image_area), h) for h in _page_hypotheses(img)]
    scored = [s for s in scored if s[0] > 0]
    if not scored:
        return img
    hull = max(scored, key=lambda s: s[0])[1]
    c = hull
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(quad) == 4 and cv2.contourArea(quad) > 0.85 * cv2.contourArea(hull):
        src = _order_corners(quad.reshape(4, 2).astype(np.float32))
    else:
        # No trustworthy four-corner outline: use the tightest rotated rectangle.
        src = _order_corners(cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32))
    w = int(max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2])))
    h = int(max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2])))
    if not 0.4 < w / max(h, 1) < 2.5:
        x, y, bw, bh = cv2.boundingRect(c)
        return img[y : y + bh, x : x + bw]
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))


def ink_mask(page: np.ndarray, seed: int = 0) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Keep pen strokes; drop paper, ruled lines, printed text, watermark, show-through.

    Every non-paper pixel is clustered in Lab space. The pen ink is the
    cluster that is both dark (low L) and blue (most negative b). Ruled
    lines and show-through are blue-ish too, but much lighter, so they land
    in other clusters.
    """
    lab = cv2.cvtColor(page, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, b = lab[..., 0], lab[..., 2]
    # Paper is bright; everything noticeably darker than the local
    # background is a candidate (pen, print, lines, watermark, show-through).
    background = cv2.medianBlur(L.astype(np.uint8), 51).astype(np.float32)
    candidate = (background - L) > 12
    pts = lab[candidate]
    if len(pts) < 500:
        return np.zeros(L.shape, np.uint8), (0.0, 0.0, 0.0)
    sample = pts[np.random.default_rng(seed).choice(len(pts), min(len(pts), 40000), replace=False)]
    km = KMeans(n_clusters=5, n_init=4, random_state=seed).fit(sample)
    centers = km.cluster_centers_
    # Score: darker and bluer is more ink-like. b* is centred on 128 in OpenCV's 8-bit Lab.
    score = (255 - centers[:, 0]) + 2.0 * (128 - centers[:, 2])
    ink_k = int(np.argmax(score))
    labels = np.full(L.shape, -1, np.int32)
    labels[candidate] = km.predict(pts)
    strong = labels == ink_k
    c = centers[ink_k]
    # Hysteresis, as in Canny: faint parts of a pen stroke are bluish and
    # darker than the paper but fall outside the ink cluster. Keep a weak
    # pixel only if its connected region touches a strong (ink-cluster) pixel,
    # so isolated ruled lines and show-through stay out.
    weak = candidate & (b < 128 - 0.5 * (128 - c[2])) & (L < background - 20)
    region = (strong | weak).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    has_strong = np.zeros(n, bool)
    has_strong[np.unique(cc[strong])] = True
    has_strong[0] = False
    # Drop specks, and anything touching the page edge (cloth, binding, shadow).
    h, w = L.shape
    x0, y0 = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    x1, y1 = x0 + stats[:, cv2.CC_STAT_WIDTH], y0 + stats[:, cv2.CC_STAT_HEIGHT]
    edge = (x0 <= 4) | (y0 <= 4) | (x1 >= w - 4) | (y1 >= h - 4)
    keep = has_strong & (stats[:, cv2.CC_STAT_AREA] >= 12) & ~edge
    mask = (keep[cc] * 255).astype(np.uint8)
    return mask, (float(c[0]), float(c[1]), float(c[2]))


def _shirorekha_score(mask: np.ndarray) -> float:
    """Positive when text is upright, negative when upside down.

    Devanagari words hang from a headline (shirorekha) at the TOP of each
    word. In every text band, the densest pixel row should therefore sit in
    the upper half of the band. Upside down, it sits in the lower half.
    """
    prof = (mask > 0).sum(axis=1).astype(np.float32)
    if prof.max() == 0:
        return 0.0
    on = prof > 0.15 * prof.max()
    scores = []
    y = 0
    while y < len(on):
        if not on[y]:
            y += 1
            continue
        start = y
        while y < len(on) and on[y]:
            y += 1
        band = prof[start:y]
        if len(band) >= 8:
            peak = int(np.argmax(band))
            scores.append(0.5 - peak / (len(band) - 1))
    return float(np.mean(scores)) if scores else 0.0


def _central(mask: np.ndarray, margin: float = 0.08) -> np.ndarray:
    h, w = mask.shape
    return mask[int(h * margin) : int(h * (1 - margin)), int(w * margin) : int(w * (1 - margin))]


def _text_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    prof = (mask > 0).sum(axis=1).astype(np.float32)
    if prof.max() == 0:
        return []
    on = prof > 0.15 * prof.max()
    bands, y = [], 0
    while y < len(on):
        if not on[y]:
            y += 1
            continue
        start = y
        while y < len(on) and on[y]:
            y += 1
        if y - start >= 8:
            bands.append((start, y))
    return bands


def _alignment_score(mask: np.ndarray) -> float:
    """Positive when lines share a left edge (the amounts column) and end raggedly.

    Register entries start at the same x (the amount) and end wherever the
    village name ends. Upside down, the straight edge moves to the right.
    """
    lefts, rights = [], []
    for a, b in _text_bands(mask):
        cols = np.where((mask[a:b] > 0).sum(axis=0) > 1)[0]
        if len(cols) > 20:
            lefts.append(cols[0])
            rights.append(cols[-1])
    if len(lefts) < 2:
        return 0.0
    ls, rs = float(np.std(lefts)), float(np.std(rights))
    return (rs - ls) / (rs + ls + 1e-6)


def orient(page: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Rotate in 90° steps so that text lines are horizontal and upright.

    Each of the four rotations is scored with two independent cues: the
    Devanagari headline sits at the top of each line, and entries share a
    left edge (the amounts column). Both are strongest only when the page is
    upright. (A plain row-profile test barely separates 0° from 90°, because
    the aligned columns of a register also make the sideways profile spiky.)
    """
    def score(k: int) -> float:
        core = _central(np.rot90(mask, k=k))
        return 2.0 * _shirorekha_score(core) + _alignment_score(core)

    k = max(range(4), key=score)
    return np.ascontiguousarray(np.rot90(page, k=k)), np.ascontiguousarray(np.rot90(mask, k=k)), 90 * k


def process(path: Path) -> PageResult:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read {path}")
    page = crop_page(img)
    scale = WORK_WIDTH / page.shape[1]
    page = cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask, center = ink_mask(page)
    page, mask, rot = orient(page, mask)
    return PageResult(page=page, ink=mask, rotation=rot, ink_cluster_lab=center)


def main(argv: list[str]) -> None:
    raw_dir, out_dir = Path(argv[1]), Path(argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(raw_dir.glob("*.jpg")):
        r = process(path)
        stem = path.stem
        cv2.imwrite(str(out_dir / f"{stem}_page.jpg"), r.page, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(out_dir / f"{stem}_ink.png"), 255 - r.ink)
        overlay = r.page.copy()
        overlay[r.ink > 0] = (0, 0, 255)
        cv2.imwrite(str(out_dir / f"{stem}_mask.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
        ink_pct = 100.0 * (r.ink > 0).mean()
        print(f"{stem}: size={r.page.shape[1]}x{r.page.shape[0]} rotation={r.rotation} ink={ink_pct:.1f}% ink_lab={tuple(round(v) for v in r.ink_cluster_lab)}")


if __name__ == "__main__":
    main(sys.argv)
