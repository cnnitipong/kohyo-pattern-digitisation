#!/usr/bin/env python3
"""
digitize.py -- Thai Frab (fabric) pattern digitization pipeline.

Takes example.jpg (a photograph of a Thai gold-on-purple kranok border /
gold-stripe / diamond-lattice floral body textile) and produces a set of
loom-ready assets for a TC2 digital Jacquard loom:

  1.  Programmatic zone detection (border / gold stripe / body) from the
      row-brightness profile.
  2.  Repeat-unit extraction for the body zone: integer horizontal/vertical
      period search (autocorrelation for an initial estimate, then a grid
      search over candidate integer periods that MINIMIZES the RMSE of a
      tiled reconstruction against the original body zone).
  3.  Colour quantization: k-means (k=5) run independently on the body zone
      and the border zone, then the two sets of cluster centres are merged
      (weighted by pixel count) into a single shared palette of <= 6 colours.
      Every pixel in every output raster is snapped to the NEAREST palette
      colour (nearest-neighbour only, no interpolation/anti-aliasing), so
      every loom file contains exactly the palette colours and nothing else.
  4.  Output rasters: repeat_unit.png, repeat_unit_chart.png (8x + point-paper
      grid), tc2_body.tif (4x4 tiled indexed TIFF), tc2_border.tif (one
      border repeat x4, full border+stripe height, indexed TIFF),
      full_simulation.png (the whole 1920x900 layout re-rendered from the
      quantized, tiled repeats -- not a copy of the input), palette.json,
      metrics.json.
  5.  Self-verification: unique-colour-count assertions on every loom file,
      tiling-dimension assertions, and a printed summary table.

Run:  python3 pipeline/digitize.py
"""

import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUT_PATH = os.path.join(ROOT, "example.jpg")
OUTPUT_DIR = os.path.join(ROOT, "output")


# ----------------------------------------------------------------------------
# 1. Zone boundary detection
# ----------------------------------------------------------------------------
def detect_zones(gray):
    """
    Detect the three horizontal zones (A: kranok border, B: gold stripe band,
    C: diamond-lattice body) purely from the row-brightness / row-texture
    profile -- no hardcoded row numbers.

    Zone A -> B boundary: the border motifs are localized (gold occupies only
    part of each row's width), so row-mean stays moderate (< ~150). The gold
    stripe band is a *solid, full-width* band, so its row-mean jumps to a
    sustained high plateau. We detect the first row where that plateau
    begins.

    Zone B -> C boundary: naive full-row std is contaminated by a large-scale
    left-right lighting gradient (vignette) present across the whole photo,
    which alone can push a "flat" band's std above a naive texture
    threshold. To isolate real *local* texture (motifs on a ~50-130px
    scale) from that slow gradient, each row is high-pass filtered (row
    minus a wide horizontal Gaussian blur of itself) before taking std.
    Under this detrended measure the stripe band sits well below the body
    zone's texture level, so we detect the first sustained run of high
    detrended-std after the stripe starts.
    """
    row_mean = gray.mean(axis=1)
    H, W = gray.shape

    # Detrend: remove slow horizontal gradient (sigma wider than any motif)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=61, sigmaY=0)
    highpass = gray - blurred
    row_std = highpass.std(axis=1)

    BRIGHT_THRESH = 150.0   # solid gold band plateau is ~180 vs local motifs ~<130
    TEXTURE_THRESH = 30.0   # body zone detrended texture (~35-55) vs stripe band (~27.5 max)
    SUSTAIN = 3             # rows required to confirm a transition (avoid 1px noise)

    # --- Zone A/B boundary ---
    stripe_start = None
    for i in range(H):
        if row_mean[i] > BRIGHT_THRESH:
            stripe_start = i
            break
    if stripe_start is None:
        raise RuntimeError("Could not find gold stripe band (zone A/B boundary).")

    # --- Zone B/C boundary ---
    body_start = None
    consec = 0
    for i in range(stripe_start, H):
        if row_std[i] > TEXTURE_THRESH:
            consec += 1
            if consec >= SUSTAIN:
                body_start = i - consec + 1
                break
        else:
            consec = 0
    if body_start is None:
        raise RuntimeError("Could not find body zone start (zone B/C boundary).")

    return {
        "border": (0, stripe_start),
        "stripe": (stripe_start, body_start),
        "body": (body_start, H),
    }


# ----------------------------------------------------------------------------
# 2. Period detection helpers
# ----------------------------------------------------------------------------
def autocorr_estimate(profile, min_p, max_p):
    """Return the period in [min_p, max_p] with the highest normalized
    autocorrelation, as an initial estimate to center a refinement search."""
    n = len(profile)
    p = profile - profile.mean()
    full = np.correlate(p, p, mode="full")
    corr = full[n - 1:]
    corr = corr / corr[0]
    best = max(range(min_p, max_p + 1), key=lambda k: corr[k])
    return best, float(corr[best])


def averaged_tile_2d(zone, px, py):
    """
    Build the phase-averaged repeat tile for candidate period (px, py):
    every pixel of `zone` is assigned to its (row % py, col % px) phase and
    averaged there. This is the RMSE-minimizing tile for that period (the
    mean minimizes squared error), and -- provably -- its reconstruction RMSE
    does NOT depend on where the crop starts (circular-shift invariance), so
    only (px, py) needs to be searched; offset is a free choice for the final
    crop.
    Returns (tile, rmse).
    """
    H, W, C = zone.shape
    rows_idx = np.arange(H) % py
    cols_idx = np.arange(W) % px
    flat_phase = (rows_idx[:, None] * px + cols_idx[None, :]).astype(np.int64).flatten()

    tile = np.empty((py, px, C), dtype=np.float64)
    for c in range(C):
        sums = np.bincount(flat_phase, weights=zone[:, :, c].flatten(), minlength=px * py)
        tile[:, :, c] = sums.reshape(py, px)
    cnt = np.bincount(flat_phase, minlength=px * py).reshape(py, px)
    tile /= cnt[:, :, None]

    recon = tile.reshape(px * py, C)[flat_phase].reshape(H, W, C)
    rmse = float(np.sqrt(np.mean((recon - zone) ** 2)))
    return tile, rmse


def averaged_tile_1d_x(zone, px):
    """Same idea as averaged_tile_2d but periodic in x only (used for the
    border+stripe zone, which is not vertically repeating)."""
    H, W, C = zone.shape
    reps = W // px
    rem = W % px
    acc = np.zeros((H, px, C), dtype=np.float64)
    cnt = np.zeros(px, dtype=np.float64)
    for i in range(reps):
        acc += zone[:, i * px:(i + 1) * px, :]
        cnt += 1
    if rem > 0:
        acc[:, :rem, :] += zone[:, reps * px: reps * px + rem, :]
        cnt[:rem] += 1
    tile = acc / cnt[None, :, None]
    recon = np.tile(tile, (1, W // px + 1, 1))[:, :W, :]
    rmse = float(np.sqrt(np.mean((recon - zone) ** 2)))
    return tile, rmse


def best_offset_crop_2d(zone, px, py, nx=1, ny=1):
    """
    Find the offset (oy, ox) in [0,py) x [0,px) whose *direct crop*
    zone[oy:oy+py, ox:ox+px] is most self-consistent with its immediate
    neighbouring repeat instances (nx repeats to the right, ny repeats down).

    Why not just use the phase-averaged tile (averaged_tile_2d) here? That
    average is RMSE-optimal *globally*, but real woven/photographed fabric
    period drifts slightly across a wide field (perspective/lens effects
    accumulate over many repeats), so averaging over ALL repeats blurs/ghosts
    fine detail. Comparing a crop only to its *immediate* neighbours keeps the
    baseline short (little accumulated drift) and rewards offsets that sit in
    a locally well-aligned patch of fabric, giving a crisp representative
    tile made of real (not blended) pixels.
    """
    H, W, C = zone.shape
    best = None
    for oy in range(py):
        if oy + (ny + 1) * py > H:
            continue
        for ox in range(px):
            if ox + (nx + 1) * px > W:
                continue
            tile = zone[oy:oy + py, ox:ox + px, :]
            errs = []
            for j in range(ny + 1):
                for i in range(nx + 1):
                    if i == 0 and j == 0:
                        continue
                    neighbor = zone[oy + j * py:oy + (j + 1) * py,
                                    ox + i * px:ox + (i + 1) * px, :]
                    errs.append(np.sqrt(np.mean((tile - neighbor) ** 2)))
            if not errs:
                continue
            local_rmse = float(np.mean(errs))
            if best is None or local_rmse < best[0]:
                best = (local_rmse, oy, ox, tile)
    return best  # (local_rmse, oy, ox, tile)


def best_offset_crop_1d_x(zone, px, n_neighbors=3):
    """Same idea as best_offset_crop_2d but for a horizontally-periodic-only
    zone (the border+stripe band): compares a crop only to its next
    `n_neighbors` repeats to the right, keeping the comparison baseline
    short so global left-to-right period drift doesn't blur the result."""
    H, W, C = zone.shape
    best = None
    for ox in range(px):
        if ox + (n_neighbors + 1) * px > W:
            continue
        tile = zone[:, ox:ox + px, :]
        errs = []
        for i in range(1, n_neighbors + 1):
            neighbor = zone[:, ox + i * px:ox + (i + 1) * px, :]
            errs.append(np.sqrt(np.mean((tile - neighbor) ** 2)))
        local_rmse = float(np.mean(errs))
        if best is None or local_rmse < best[0]:
            best = (local_rmse, ox, tile)
    return best  # (local_rmse, ox, tile)


def search_body_period(zone_gray_c, zone_c, search_radius=15):
    """Two-stage search for the body repeat period: autocorrelation gives a
    coarse estimate near the expected ~128 x ~126, then a full grid search
    over integer periods around that estimate picks the (px, py) that
    minimizes the 2D phase-averaged tiling RMSE."""
    colprof = zone_gray_c.mean(axis=0)
    rowprof = zone_gray_c.mean(axis=1)
    est_x, _ = autocorr_estimate(colprof, 100, 150)
    est_y, _ = autocorr_estimate(rowprof, 100, 150)

    best = None
    for px in range(est_x - search_radius, est_x + search_radius + 1):
        for py in range(est_y - search_radius, est_y + search_radius + 1):
            if px < 20 or py < 20:
                continue
            _, rmse = averaged_tile_2d(zone_c, px, py)
            if best is None or rmse < best[0]:
                best = (rmse, px, py)
    rmse, px, py = best
    return px, py, rmse, (est_x, est_y)


def search_border_period(zone_gray_ab, zone_ab, search_range=(35, 90)):
    """Same idea for the border's horizontal-only period."""
    colprof = zone_gray_ab.mean(axis=0)
    est_x, _ = autocorr_estimate(colprof, 30, 80)

    best = None
    for px in range(*search_range):
        _, rmse = averaged_tile_1d_x(zone_ab, px)
        if best is None or rmse < best[0]:
            best = (rmse, px)
    rmse, px = best
    return px, rmse, est_x


# ----------------------------------------------------------------------------
# 3. Colour quantization
# ----------------------------------------------------------------------------
def kmeans_centers(pixels_rgb, k, seed=42):
    """Run cv2.kmeans on an RGB pixel array (H,W,3) -> (centers[k,3], counts[k])."""
    cv2.setRNGSeed(seed)
    Z = pixels_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.3)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float64)
    return centers.astype(np.float64), counts


def weighted_merge_agglomerative(points, weights, k_final):
    """
    Merge a small set of weighted candidate colour points (the union of the
    border- and body-zone k-means cluster centers) down to k_final colours
    using weighted-average agglomerative clustering: repeatedly merge the
    two closest remaining points (Euclidean, RGB) into their weighted
    centroid, until only k_final points remain.

    This is used instead of iterating cv2.kmeans / a from-scratch k-means
    because with only ~10 candidate points and a few very tight near-duplicate
    pairs (e.g. the body-zone and border-zone "ground purple" centres, which
    sit ~1-2 units apart while every other pair is 30-100+ units apart),
    a from-scratch weighted k-means can get stuck in a degenerate local
    minimum (two near-duplicate seeds each keep exactly one point forever,
    instead of merging). Greedy nearest-pair agglomeration has no such
    failure mode for this well-separated-pairs structure and is deterministic.
    """
    pts = [p.astype(np.float64) for p in points]
    wts = [float(w) for w in weights]

    while len(pts) > k_final:
        n = len(pts)
        best = None
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        w1, w2 = wts[i], wts[j]
        merged = (pts[i] * w1 + pts[j] * w2) / (w1 + w2)
        merged_w = w1 + w2
        # remove higher index first to keep the lower index valid
        for idx in sorted((i, j), reverse=True):
            pts.pop(idx)
            wts.pop(idx)
        pts.append(merged)
        wts.append(merged_w)

    centers = np.array(pts)
    weights_out = np.array(wts)
    return centers, weights_out


def classify_palette_names(palette_rgb):
    """
    Assign semantic names (dark purple ground / magenta / light pink /
    dark gold / bright gold) using hue + value, since the fabric's palette
    naturally splits into a gold-ish family (yellow/orange hue) and a
    purple/pink family (magenta hue). Within each family, colours are ranked
    by brightness (V in HSV).
    """
    hsv = cv2.cvtColor(palette_rgb.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
    gold_idx, pink_idx = [], []
    for i, (h, s, v) in enumerate(hsv):
        if 15 <= h <= 65:
            gold_idx.append(i)
        else:
            pink_idx.append(i)

    gold_idx.sort(key=lambda i: hsv[i][2])       # ascending brightness
    pink_idx.sort(key=lambda i: hsv[i][2])       # ascending brightness

    gold_names = ["dark gold", "bright gold", "pale gold"]
    pink_names = ["dark purple ground", "magenta/mid pink", "light pink", "pale pink"]

    names = [None] * len(palette_rgb)
    for rank, i in enumerate(gold_idx):
        names[i] = gold_names[rank] if rank < len(gold_names) else f"gold_shade_{rank}"
    for rank, i in enumerate(pink_idx):
        names[i] = pink_names[rank] if rank < len(pink_names) else f"pink_shade_{rank}"
    return names


WEAVE_SUGGESTIONS = {
    "dark purple ground": "8-end satin, weft-faced (ground weave)",
    "magenta/mid pink": "5-end satin, weft-faced accent (medallion fill)",
    "light pink": "plain weave, weft float highlight",
    "dark gold": "3/1 weft-faced twill (motif outline/fill)",
    "bright gold": "1/2 twill, weft float (metallic highlight)",
    "pale gold": "1/2 twill, weft float (metallic highlight)",
    "pale pink": "plain weave, weft float highlight",
}


def nearest_palette_index(image_rgb_float, palette_rgb):
    """Assign every pixel of image_rgb_float (H,W,3) to the index of the
    nearest palette colour (Euclidean, RGB). Returns uint8 index array."""
    H, W, _ = image_rgb_float.shape
    flat = image_rgb_float.reshape(-1, 3)
    dists = np.linalg.norm(flat[:, None, :] - palette_rgb[None, :, :], axis=2)
    idx = np.argmin(dists, axis=1).astype(np.uint8)
    return idx.reshape(H, W)


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------
def make_pil_palette(palette_rgb):
    """Build a 768-entry (256*3) flat palette list for PIL 'P' mode images."""
    flat = []
    for rgb in palette_rgb:
        flat.extend([int(rgb[0]), int(rgb[1]), int(rgb[2])])
    flat.extend([0, 0, 0] * (256 - len(palette_rgb)))
    return flat


def save_indexed_tiff(index_array, palette_rgb, path):
    im = Image.fromarray(index_array, mode="P")
    im.putpalette(make_pil_palette(palette_rgb))
    im.save(path, format="TIFF", compression="raw")


def save_indexed_png(index_array, palette_rgb, path):
    im = Image.fromarray(index_array, mode="P")
    im.putpalette(make_pil_palette(palette_rgb))
    im.save(path, format="PNG")


def tile_index_array(index_tile, reps_y, reps_x, crop_h=None, crop_w=None):
    """Tile a 2D index array reps_y times vertically and reps_x times
    horizontally (nearest-neighbour repetition -- index values are just
    repeated, no interpolation), then optionally crop to exact size."""
    out = np.tile(index_tile, (reps_y, reps_x))
    if crop_h is not None:
        out = out[:crop_h, :, ]
    if crop_w is not None:
        out = out[:, :crop_w]
    return out


def render_chart(index_tile, palette_rgb, path, scale=8, grid_every=1, heavy_every=10):
    """Nearest-neighbour enlarge by `scale`, then draw a point-paper grid:
    a light-grey line every `grid_every` threads, a heavier grey line every
    `heavy_every` threads."""
    py, px = index_tile.shape
    rgb_tile = np.array(palette_rgb, dtype=np.uint8)[index_tile]  # (py,px,3)
    big = cv2.resize(rgb_tile, (px * scale, py * scale), interpolation=cv2.INTER_NEAREST)
    img = Image.fromarray(big, mode="RGB")
    draw = ImageDraw.Draw(img)

    W, H = img.size
    light = (170, 170, 170)
    heavy = (90, 90, 90)

    # vertical lines
    for t in range(0, px + 1, grid_every):
        x = t * scale
        color = heavy if (t % heavy_every == 0) else light
        width = 2 if (t % heavy_every == 0) else 1
        draw.line([(x, 0), (x, H)], fill=color, width=width)
    # horizontal lines
    for t in range(0, py + 1, grid_every):
        y = t * scale
        color = heavy if (t % heavy_every == 0) else light
        width = 2 if (t % heavy_every == 0) else 1
        draw.line([(0, y), (W, y)], fill=color, width=width)

    img.save(path, format="PNG")


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load ----
    bgr = cv2.imread(INPUT_PATH)
    if bgr is None:
        raise FileNotFoundError(f"Could not read {INPUT_PATH}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    H, W, _ = rgb.shape
    print(f"[1] Loaded {INPUT_PATH}: {W}x{H} RGB")

    # ---- 1. Zone detection ----
    zones = detect_zones(gray)
    (ba, be) = zones["border"]
    (sa, se) = zones["stripe"]
    (bda, bde) = zones["body"]
    print(f"[1] Zone boundaries detected:")
    print(f"    Zone A (border) : rows {ba}-{be - 1}  ({be - ba} rows)")
    print(f"    Zone B (stripe) : rows {sa}-{se - 1}  ({se - sa} rows)")
    print(f"    Zone C (body)   : rows {bda}-{bde - 1}  ({bde - bda} rows)")

    body_start = bda                        # == zones["body"][0]; bde == H (image height)
    zone_ab_rgb = rgb[0:body_start, :, :]   # border+stripe, full height (0..body_start)
    zone_ab_gray = gray[0:body_start, :]
    zone_c_rgb = rgb[bda:bde, :, :]         # body zone only
    zone_c_gray = gray[bda:bde, :]

    # ---- 2. Body repeat period search ----
    body_px, body_py, body_rmse, body_est = search_body_period(zone_c_gray, zone_c_rgb, search_radius=15)
    print(f"[2] Body period search: autocorr estimate x={body_est[0]} y={body_est[1]}")
    print(f"    Refined (min tiling RMSE over integer grid): {body_px} x {body_py}px, RMSE={body_rmse:.3f}")

    # The phase-averaged tile is the RMSE-optimal reconstruction for the
    # *chosen period* (used above purely to score/select px,py). For the
    # actual pixel content of the output tile we instead pick the single
    # real crop whose immediate neighbours match it best -- real fabric
    # period drifts slightly over many repeats (perspective/lens effects),
    # so averaging over all ~5x15 body repeats or ~37 border repeats can
    # ghost/blur fine detail; a locally-validated direct crop stays crisp.
    _, body_oy, body_ox, body_tile_f = best_offset_crop_2d(zone_c_rgb, body_px, body_py, nx=1, ny=1)
    print(f"    Body tile crop offset (local self-consistency search): ox={body_ox}, oy={body_oy}")

    # ---- Border repeat period search (horizontal only) ----
    border_px, border_rmse, border_est = search_border_period(zone_ab_gray, zone_ab_rgb, search_range=(35, 90))
    print(f"[2] Border period search: autocorr estimate x={border_est}")
    print(f"    Refined (min tiling RMSE): {border_px}px, RMSE={border_rmse:.3f}")

    _, border_ox, border_tile_f = best_offset_crop_1d_x(zone_ab_rgb, border_px, n_neighbors=3)
    print(f"    Border tile crop offset (local self-consistency search): ox={border_ox}")

    # ---- 3. Colour quantization ----
    print("[3] k-means (k=5) on body zone pixels...")
    body_centers, body_counts = kmeans_centers(zone_c_rgb, 5)
    print("[3] k-means (k=5) on border+stripe zone pixels...")
    border_centers, border_counts = kmeans_centers(zone_ab_rgb, 5)

    all_points = np.vstack([body_centers, border_centers])
    all_weights = np.concatenate([body_counts, border_counts])
    palette_rgb, _ = weighted_merge_agglomerative(all_points, all_weights, k_final=5)
    palette_rgb = np.clip(np.round(palette_rgb), 0, 255).astype(np.uint8)
    n_colors = len(palette_rgb)
    print(f"[3] Merged shared palette -> {n_colors} colours")

    names = classify_palette_names(palette_rgb)

    # Overall coverage: nearest-assign every pixel of the ENTIRE image to the palette
    full_idx_for_coverage = nearest_palette_index(rgb, palette_rgb.astype(np.float64))
    total_px = H * W
    coverage = {}
    for i in range(n_colors):
        coverage[i] = 100.0 * np.sum(full_idx_for_coverage == i) / total_px

    # sort palette by coverage desc for reporting (keep index stable for files)
    order = sorted(range(n_colors), key=lambda i: -coverage[i])

    print("[3] Palette:")
    for i in order:
        r, g, b = palette_rgb[i]
        print(f"    idx {i}: RGB({r},{g},{b}) '{names[i]}'  coverage={coverage[i]:.2f}%")

    # ---- Quantize the body & border tiles to the shared palette (nearest colour) ----
    body_idx_tile = nearest_palette_index(body_tile_f, palette_rgb.astype(np.float64))   # (py,px) uint8
    border_idx_tile = nearest_palette_index(border_tile_f, palette_rgb.astype(np.float64))  # (H_border, border_px)

    # ---- 4. Outputs ----
    # 4a. repeat_unit.png -- actual size, indexed
    repeat_unit_path = os.path.join(OUTPUT_DIR, "repeat_unit.png")
    save_indexed_png(body_idx_tile, palette_rgb, repeat_unit_path)

    # 4b. repeat_unit_chart.png -- 8x nearest-neighbour + point-paper grid
    chart_path = os.path.join(OUTPUT_DIR, "repeat_unit_chart.png")
    render_chart(body_idx_tile, palette_rgb, chart_path, scale=8, grid_every=1, heavy_every=10)

    # 4c. tc2_body.tif -- repeat tiled 4x horizontally and 4x vertically
    body_tiled_idx = tile_index_array(body_idx_tile, reps_y=4, reps_x=4)
    tc2_body_path = os.path.join(OUTPUT_DIR, "tc2_body.tif")
    save_indexed_tiff(body_tiled_idx, palette_rgb, tc2_body_path)

    # 4d. tc2_border.tif -- one border repeat x4 width, full border+stripe height
    border_tiled_idx = tile_index_array(border_idx_tile, reps_y=1, reps_x=4)
    tc2_border_path = os.path.join(OUTPUT_DIR, "tc2_border.tif")
    save_indexed_tiff(border_tiled_idx, palette_rgb, tc2_border_path)

    # 4e. full_simulation.png -- entire 1920x900 re-rendered from quantized tiled repeats
    reps_y_body = (bde - bda) // body_py + 2
    reps_x_body = W // body_px + 2
    body_full_idx = tile_index_array(body_idx_tile, reps_y_body, reps_x_body,
                                      crop_h=(bde - bda), crop_w=W)

    reps_x_border = W // border_px + 2
    border_full_idx = tile_index_array(border_idx_tile, reps_y=1, reps_x=reps_x_border,
                                        crop_w=W)  # shape (body_start, W) since border_idx_tile has full height already

    full_idx = np.vstack([border_full_idx, body_full_idx])  # (H, W)
    assert full_idx.shape == (H, W), f"full_simulation index shape mismatch: {full_idx.shape}"
    full_rgb = np.array(palette_rgb, dtype=np.uint8)[full_idx]  # (H,W,3), exact palette colours only
    full_sim_path = os.path.join(OUTPUT_DIR, "full_simulation.png")
    Image.fromarray(full_rgb, mode="RGB").save(full_sim_path, format="PNG")

    # ---- palette.json ----
    palette_entries = []
    for i in order:
        r, g, b = [int(v) for v in palette_rgb[i]]
        palette_entries.append({
            "index": int(i),
            "rgb": [r, g, b],
            "name": names[i],
            "coverage_percent": round(coverage[i], 2),
            "suggested_weave": WEAVE_SUGGESTIONS.get(names[i], "8-end satin, weft-faced"),
        })
    palette_json_path = os.path.join(OUTPUT_DIR, "palette.json")
    with open(palette_json_path, "w") as f:
        json.dump(palette_entries, f, indent=2)

    # ---- metrics.json ----
    def dims_of(path):
        with Image.open(path) as im:
            return list(im.size)  # (W,H)

    metrics = {
        "image_size": [W, H],
        "zone_boundaries": {
            "border": [ba, be],
            "stripe": [sa, se],
            "body": [bda, bde],
        },
        "border_repeat_px": border_px,
        "body_repeat_px": [body_px, body_py],
        "tiling_rmse": {
            "border": round(border_rmse, 4),
            "body": round(body_rmse, 4),
        },
        "n_colors": n_colors,
        "files": {
            "repeat_unit.png": dims_of(repeat_unit_path),
            "repeat_unit_chart.png": dims_of(chart_path),
            "tc2_body.tif": dims_of(tc2_body_path),
            "tc2_border.tif": dims_of(tc2_border_path),
            "full_simulation.png": dims_of(full_sim_path),
        },
    }
    metrics_json_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- 5. Verification ----
    print("\n[5] Verification:")

    def unique_colors_of_indexed(path):
        im = Image.open(path).convert("RGB")
        arr = np.array(im)
        return len(np.unique(arr.reshape(-1, 3), axis=0))

    checks = []

    body_unique = unique_colors_of_indexed(tc2_body_path)
    checks.append(("tc2_body.tif unique colours == palette size",
                    body_unique, n_colors, body_unique == n_colors))

    border_unique = unique_colors_of_indexed(tc2_border_path)
    checks.append(("tc2_border.tif unique colours <= palette size",
                    border_unique, n_colors, border_unique <= n_colors))

    ru_unique = unique_colors_of_indexed(repeat_unit_path)
    checks.append(("repeat_unit.png unique colours <= palette size",
                    ru_unique, n_colors, ru_unique <= n_colors))

    fs_unique = unique_colors_of_indexed(full_sim_path)
    checks.append(("full_simulation.png unique colours == palette size",
                    fs_unique, n_colors, fs_unique == n_colors))

    exp_body_dims = (body_px * 4, body_py * 4)
    got_body_dims = tuple(dims_of(tc2_body_path))
    checks.append(("tc2_body.tif dims == repeat x4 x4",
                    got_body_dims, exp_body_dims, got_body_dims == exp_body_dims))

    exp_border_dims = (border_px * 4, body_start)
    got_border_dims = tuple(dims_of(tc2_border_path))
    checks.append(("tc2_border.tif dims == (border_px*4, border_height)",
                    got_border_dims, exp_border_dims, got_border_dims == exp_border_dims))

    got_fs_dims = tuple(dims_of(full_sim_path))
    checks.append(("full_simulation.png dims == input dims",
                    got_fs_dims, (W, H), got_fs_dims == (W, H)))

    print(f"{'check':55s} {'got':>18s} {'expected':>18s}  ok")
    all_ok = True
    for name, got, exp, ok in checks:
        print(f"{name:55s} {str(got):>18s} {str(exp):>18s}  {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    # hard asserts on the core loom-file requirements
    assert body_unique == n_colors, f"tc2_body.tif has {body_unique} unique colours, expected {n_colors}"
    assert fs_unique == n_colors, f"full_simulation.png has {fs_unique} unique colours, expected {n_colors}"
    assert got_body_dims == exp_body_dims, "tc2_body.tif dimension mismatch"
    assert got_border_dims == exp_border_dims, "tc2_border.tif dimension mismatch"
    assert got_fs_dims == (W, H), "full_simulation.png dimension mismatch"

    print(f"\nAll checks {'PASSED' if all_ok else 'HAD FAILURES (see above)'}")

    # ---- Summary table ----
    print("\n=== SUMMARY ===")
    print(f"Image size        : {W} x {H}")
    print(f"Zone A (border)   : rows {ba}-{be - 1}")
    print(f"Zone B (stripe)   : rows {sa}-{se - 1}")
    print(f"Zone C (body)     : rows {bda}-{bde - 1}")
    print(f"Border repeat     : {border_px}px (RMSE {border_rmse:.3f})")
    print(f"Body repeat       : {body_px} x {body_py}px (RMSE {body_rmse:.3f})")
    print(f"Palette size      : {n_colors}")
    print("Files written:")
    for name, dims in metrics["files"].items():
        print(f"  output/{name:26s} {dims[0]:5d} x {dims[1]:5d}")
    print(f"  output/palette.json")
    print(f"  output/metrics.json")


if __name__ == "__main__":
    main()
