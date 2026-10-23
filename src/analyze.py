#!/usr/bin/env python3
"""Koh Yo pattern digitisation and verification pipeline.

Run from anywhere: python3 src/analyze.py
Inputs : data/raw/P*.png   (five photographs of Koh Yo hand-woven cloth)
Outputs: data/results.json, data/loom/<id>/ (repeat PNG, point paper, TC2 TIFF, WIF),
         data/npz/<id>.npz (arrays used by the figure script)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import color as skcolor
from skimage.metrics import structural_similarity as ssim
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
(OUT / "loom").mkdir(parents=True, exist_ok=True)
(OUT / "npz").mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

PATTERNS = [
    ("P1", "P1_ratchawat_4h.png", "Ratchawat", "ราชวัตร", 4),
    ("P2", "P2_phakakrong_8h.png", "Dok Phakakrong", "ดอกผกากรอง", 8),
    ("P3", "P3_rotsukhon_lukkaeo_8h.png", "Dok Rotsukhon with Luk Kaeo", "ดอกรสสุคนธ์ผสมลูกแก้ว", 8),
    ("P4", "P4_kamphaeng_4h.png", "Kamphaeng", "กำแพง", 4),
    ("P5", "P5_lukkaeo_4h.png", "Luk Kaeo", "ลูกแก้ว", 4),
]

# ----------------------------------------------------------------------------- helpers

def load_rgb(path):
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return im


def flatten_illumination(rgb, sigma=45):
    """Divide L* by its large-scale trend (retinex style); keep chroma."""
    lab = skcolor.rgb2lab(rgb)
    L = lab[..., 0]
    trend = ndi.gaussian_filter(L, sigma)
    Lf = L / np.maximum(trend, 1e-3) * float(np.mean(L))
    lab2 = lab.copy(); lab2[..., 0] = np.clip(Lf, 0, 100)
    out = np.clip(skcolor.lab2rgb(lab2), 0, 1).astype(np.float32)
    return out, trend


def gray(rgb):
    return skcolor.rgb2gray(rgb).astype(np.float32)


def profile_variance_angle(g, angles):
    """Rotation that maximises variance of the row-mean profile (weft lines horizontal)."""
    h, w = g.shape
    c = min(h, w) // 2
    cy, cx = h // 2, w // 2
    crop = g[cy - c // 2: cy + c // 2, cx - c // 2: cx + c // 2]
    scores = []
    for a in angles:
        r = ndi.rotate(crop, a, reshape=False, order=1, mode="reflect")
        m = r.shape[0] // 6
        r = r[m:-m, m:-m]
        hp = r - ndi.gaussian_filter(r, 8)
        scores.append(np.var(hp.mean(axis=1)) + np.var(hp.mean(axis=0)))
    return np.asarray(scores)


def rectify(rgb):
    g = gray(rgb)
    coarse = np.arange(-8, 8.01, 0.5)
    s = profile_variance_angle(g, coarse)
    a0 = coarse[int(np.argmax(s))]
    fine = np.arange(a0 - 0.5, a0 + 0.51, 0.1)
    s2 = profile_variance_angle(g, fine)
    ang = float(fine[int(np.argmax(s2))])
    rot = ndi.rotate(rgb, ang, reshape=False, order=1, mode="reflect")
    h, w = g.shape
    m = int(0.5 * (abs(np.sin(np.deg2rad(ang))) * max(h, w))) + 8
    rot = rot[m:h - m, m:w - m]
    return np.clip(rot, 0, 1), ang, (coarse, s)


def autocorr2d(g):
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    f = np.fft.rfft2(g * win)
    ac = np.fft.irfft2(np.abs(f) ** 2, s=g.shape)
    ac = np.fft.fftshift(ac)
    return ac / ac.max()


def local_maxima(ac, min_dist=4, thr=0.15):
    mx = ndi.maximum_filter(ac, size=2 * min_dist + 1)
    pk = (ac == mx) & (ac > thr)
    ys, xs = np.nonzero(pk)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    return np.stack([xs - cx, ys - cy, ac[ys, xs]], axis=1)


def subpixel_peak(ac, x, y):
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    yy, xx = cy + y, cx + x
    def par(fm, f0, fp):
        d = fm - 2 * f0 + fp
        return 0.0 if abs(d) < 1e-9 else 0.5 * (fm - fp) / d
    dx = par(ac[yy, xx - 1], ac[yy, xx], ac[yy, xx + 1])
    dy = par(ac[yy - 1, xx], ac[yy, xx], ac[yy + 1, xx])
    return x + dx, y + dy


def lattice_candidates(ac, max_period, n_top=24, win=170, second_ratio=0.6):
    """Score candidate bases by the mean autocorrelation at their lattice points.
    The densest lattice whose points are all strongly correlated is the repeat lattice;
    internal motif correlations (sub-peaks) fail because their multiples are weak."""
    pk = local_maxima(ac, 4, 0.2)
    pk = pk[(np.abs(pk[:, 0]) <= max_period) & (np.abs(pk[:, 1]) <= max_period)]
    pk = pk[(pk[:, 0] != 0) | (pk[:, 1] != 0)]
    pk = pk[(pk[:, 1] > 0) | ((pk[:, 1] == 0) & (pk[:, 0] > 0))]
    pk = pk[np.argsort(-pk[:, 2])][:n_top]
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    best = []
    for i in range(len(pk)):
        for j in range(i + 1, len(pk)):
            v1, v2 = pk[i, :2], pk[j, :2]
            area = abs(v1[0] * v2[1] - v1[1] * v2[0])
            if area < 0.3 * np.hypot(*v1) * np.hypot(*v2): continue
            vals1, vals2 = [], []
            for n1 in range(-2, 3):
                for n2 in range(-2, 3):
                    if n1 == 0 and n2 == 0: continue
                    x, y = n1 * v1 + n2 * v2
                    if abs(x) < win and abs(y) < win:
                        yy, xx = int(round(cy + y)), int(round(cx + x))
                        if yy < 1 or xx < 1 or yy >= ac.shape[0] - 1 or xx >= ac.shape[1] - 1: continue
                        v = ac[yy - 1:yy + 2, xx - 1:xx + 2].max()
                        (vals1 if max(abs(n1), abs(n2)) == 1 else vals2).append(v)
            if len(vals1) < 8: continue
            s1 = float(np.min(vals1)); s2 = float(np.min(vals2)) if vals2 else s1
            best.append((s1, float(area), v1, v2, s2))
    best.sort(key=lambda t: -t[0])
    top = best[0][0]
    valid = [b for b in best if b[0] >= 0.8 * top and b[4] >= second_ratio * b[0]]
    if not valid: valid = best[:1]
    valid.sort(key=lambda t: t[1])
    return valid[0], pk, best


def axis_vectors(v1, v2, tol=0.1, nmax=4):
    """Shortest lattice vectors lying near the x axis and near the y axis."""
    X = Y = None
    for n1 in range(-nmax, nmax + 1):
        for n2 in range(-nmax, nmax + 1):
            if n1 == 0 and n2 == 0: continue
            v = n1 * np.asarray(v1, float) + n2 * np.asarray(v2, float)
            if v[0] > 0 and abs(v[1]) <= tol * abs(v[0]) and (X is None or v[0] < X[0]): X = v
            if v[1] > 0 and abs(v[0]) <= tol * abs(v[1]) and (Y is None or v[1] < Y[1]): Y = v
    if X is None: X = np.array([abs(v1[0]) + abs(v2[0]), 0.0])
    if Y is None: Y = np.array([0.0, abs(v1[1]) + abs(v2[1])])
    return X, Y


def affine_rectify(img, X, Y):
    """Affine map sending the detected axis lattice vectors onto the image axes."""
    basis = np.array([X, Y], float); ideal = np.array([[np.hypot(*X), 0], [0, np.hypot(*Y)]])
    M = np.linalg.solve(basis, ideal).T  # M @ basis_i = ideal_i
    H, W = img.shape[:2]
    c = np.array([W / 2, H / 2])
    A = np.hstack([M, (c - M @ c)[:, None]]).astype(np.float32)
    out = cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = int(0.06 * min(H, W))
    return out[m:-m, m:-m], M


def perspective_rectify(img, ipx, ipy, kind="rectangular", grid=(7, 5)):
    """Lattice-anchor homography: match the central mean tile across the image, fit a homography
    from measured anchor positions to ideal lattice positions and warp. Returns warped image,
    homography, anchors (measured, ideal) and the per-anchor residual drift in px."""
    H, W = img.shape[:2]
    g = gray(img).astype(np.float32)
    cy, cx = H // 2, W // 2
    # central template: mean of the 3x3 repeats around the centre
    k = 3 if (3 * ipy + 4 < H and 3 * ipx + 4 < W) else 1
    y0, x0 = cy - (k * ipy) // 2, cx - (k * ipx) // 2
    blk = g[y0:y0 + k * ipy, x0:x0 + k * ipx].reshape(k, ipy, k, ipx).mean(axis=(0, 2))
    T = np.tile(blk, (2, 2)).astype(np.float32)
    th, tw = T.shape
    # anchor search positions ordered by distance from centre
    ys = np.linspace(th // 2 + 2, H - th // 2 - 3, grid[1]).astype(int)
    xs = np.linspace(tw // 2 + 2, W - tw // 2 - 3, grid[0]).astype(int)
    pts = sorted([(y, x) for y in ys for x in xs], key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
    res = cv2.matchTemplate(g, T, cv2.TM_CCOEFF_NORMED)  # res[y, x] = match of T placed at top-left (x, y)
    Bp = np.array([[ipx / 2, ipy / 2], [ipx / 2, -ipy / 2]], float).T if kind == "centred" else np.array([[ipx, 0], [0, ipy]], float).T
    Bpi = np.linalg.inv(Bp)
    def collect(model):
        meas, ideal, score = [], [], []
        base = None
        for (y, x) in pts:
            p = model @ np.array([x - tw / 2, y - th / 2, 1.0]); p = p[:2] / p[2]
            px_, py_ = int(round(p[0])), int(round(p[1]))
            wy0, wy1 = max(py_ - ipy // 2, 0), min(py_ + ipy // 2 + 1, res.shape[0])
            wx0, wx1 = max(px_ - ipx // 2, 0), min(px_ + ipx // 2 + 1, res.shape[1])
            if wy1 <= wy0 or wx1 <= wx0: continue
            win = res[wy0:wy1, wx0:wx1]
            dy, dx = np.unravel_index(np.argmax(win), win.shape)
            my, mx = wy0 + dy, wx0 + dx
            if base is None: base = (mx, my)
            n = np.round(Bpi @ np.array([mx - base[0], my - base[1]], float))
            meas.append([mx, my]); ideal.append((Bp @ n + np.array(base)).tolist()); score.append(float(win.max()))
        return np.float32(meas), np.float32(ideal), np.array(score)
    meas, ideal, score = collect(np.eye(3))
    A, _ = cv2.estimateAffine2D(ideal, meas, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is not None:
        meas, ideal, score = collect(np.vstack([A, [0, 0, 1]]))
    meas, ideal = np.float32(meas), np.float32(ideal)
    Hfit = None
    if len(meas) >= 8:
        Hfit, _ = cv2.findHomography(meas, ideal, cv2.RANSAC, 3.0)
    if Hfit is None or not np.all(np.isfinite(Hfit)) or abs(np.linalg.det(Hfit[:2, :2]) - 1) > 0.3:
        Hfit = np.eye(3)
    drift = np.hypot(*(meas - ideal).T)
    warped = cv2.warpPerspective(img, Hfit, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = int(0.05 * min(H, W))
    return warped[m:-m, m:-m], Hfit, meas, ideal, drift, np.array(score)


def autocorr_window(rect, size=440):
    g = gray(rect)
    H, W = g.shape
    s = min(size, H - 4, W - 4)
    c = g[H // 2 - s // 2:H // 2 + s // 2, W // 2 - s // 2:W // 2 + s // 2]
    sm = ndi.gaussian_filter(c, 1.2)
    return autocorr2d(sm - ndi.gaussian_filter(sm, 25))


def rect_repeat(ac, max_period):
    (score, area, v1, v2, s2), pk, allc = lattice_candidates(ac, max_period)
    X, Y = axis_vectors(v1, v2)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    def refine(v):
        x, y = int(round(v[0])), int(round(v[1]))
        if abs(x) < cx - 3 and abs(y) < cy - 3:
            # snap to the nearest local maximum within 2 px, then parabolic sub-pixel
            patch = ac[cy + y - 2:cy + y + 3, cx + x - 2:cx + x + 3]
            dy, dx = np.unravel_index(np.argmax(patch), patch.shape)
            x, y = x + dx - 2, y + dy - 2
            (fx, fy) = subpixel_peak(ac, x, y)
            return np.array([fx, fy]), float(ac[cy + y, cx + x])
        return np.asarray(v, float), 0.0
    X, vx = refine(X); Y, vy = refine(Y)
    px, py = float(np.hypot(*X)), float(np.hypot(*Y))
    # centred lattice test: a peak near (X+Y)/2 ?
    h = (X + Y) / 2
    hx, hy = int(round(h[0])), int(round(h[1]))
    vh = float(ac[cy + hy - 2:cy + hy + 3, cx + hx - 2:cx + hx + 3].max()) if abs(hx) < cx - 3 and abs(hy) < cy - 3 else 0.0
    kind = "centred" if vh >= 0.5 * min(vx, vy) else "rectangular"
    return px, py, np.array([v1, v2], float), kind, score, vx, vy, pk, X, Y, vh


def fold_tile(img, px, py):
    """Phase-aligned mean tile and per-repeat residual map. img is (H,W,C) with integer periods."""
    H, W = img.shape[:2]
    ny, nx = H // py, W // px
    crop = img[:ny * py, :nx * px]
    blocks = crop.reshape(ny, py, nx, px, -1)
    tile = blocks.mean(axis=(0, 2))
    resid = np.sqrt(((blocks - tile[None, :, None, :, :]) ** 2).mean(axis=(1, 3, 4)))
    return tile, resid, blocks


def fold_tile_aligned(img, px, py, iters=2, m=6):
    """Mean tile after per-repeat alignment: every repeat block is re-registered (integer shift
    within +-m px) against the running mean tile before averaging, so residual non-planar
    distortion of the cloth does not blur the tile. Returns tile, residual map, shift field."""
    tile, resid, blocks = fold_tile(img, px, py)
    H, W = img.shape[:2]
    ny, nx = H // py, W // px
    g = gray(img).astype(np.float32)
    shifts = np.zeros((ny, nx, 2))
    for _ in range(iters):
        T = gray(tile).astype(np.float32)
        acc = np.zeros_like(tile); cnt = 0
        res = np.zeros((ny, nx)); keep = []
        for j in range(ny):
            for i in range(nx):
                y0, x0 = j * py, i * px
                ya, yb = max(y0 - m, 0), min(y0 + py + m, H)
                xa, xb = max(x0 - m, 0), min(x0 + px + m, W)
                reg = g[ya:yb, xa:xb]
                if reg.shape[0] < py or reg.shape[1] < px: continue
                r = cv2.matchTemplate(reg, T, cv2.TM_CCOEFF_NORMED)
                dy, dx = np.unravel_index(np.argmax(r), r.shape)
                yy, xx = ya + dy, xa + dx
                blk = img[yy:yy + py, xx:xx + px]
                shifts[j, i] = (yy - y0, xx - x0)
                res[j, i] = np.sqrt(((blk - tile) ** 2).mean())
                acc += blk; cnt += 1
        tile = acc / max(cnt, 1); resid = res
    return tile, resid, shifts


def thread_pitch(img, lo=3, hi=14):
    """Yarn-level period from the autocorrelation of the fine-scale detail image."""
    g = gray(img).astype(np.float32)
    H, W = g.shape
    s = min(400, H - 4, W - 4)
    c = g[H // 2 - s // 2:H // 2 + s // 2, W // 2 - s // 2:W // 2 + s // 2]
    hp = c - ndi.gaussian_filter(c, 2.5)
    ac = autocorr2d(hp)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    rowp, colp = ac[cy, cx:cx + hi + 2], ac[cy:cy + hi + 2, cx]
    def first_peak(p):
        best = None
        for k in range(lo, hi + 1):
            if p[k] > p[k - 1] and p[k] >= p[k + 1] and (best is None or p[k] > best[1]):
                best = (k, p[k])
        if best is None: return float("nan"), 0.0
        k = best[0]
        d = p[k - 1] - 2 * p[k] + p[k + 1]
        off = 0.0 if abs(d) < 1e-9 else 0.5 * (p[k - 1] - p[k + 1]) / d
        return float(k + off), float(p[k])
    (pxv, sxv), (pyv, syv) = first_peak(rowp), first_peak(colp)
    return pxv, pyv, sxv, syv, rowp, colp


def fold_refine(img, px, py, rel=0.06):
    g = gray(img)[..., None]
    def curve(axis):
        P0 = px if axis == 0 else py
        Ps = list(range(int(np.floor(P0 * (1 - rel))), int(np.ceil(P0 * (1 + rel))) + 1))
        vals = []
        for P in Ps:
            if axis == 0: vals.append(float(fold_tile(g, P, int(round(py)))[1].mean()))
            else: vals.append(float(fold_tile(g, int(round(px)), P)[1].mean()))
        k = int(np.argmin(vals))
        if 0 < k < len(Ps) - 1:
            d = vals[k - 1] - 2 * vals[k] + vals[k + 1]
            off = 0.0 if abs(d) < 1e-12 else 0.5 * (vals[k - 1] - vals[k + 1]) / d
        else: off = 0.0
        return float(Ps[k] + off), list(zip(Ps, vals))
    pxr, cx = curve(0); pyr, cy = curve(1)
    return pxr, pyr, {"x": cx, "y": cy, "acf": [float(px), float(py)]}


def resample_to_integer_period(img, px, py):
    """Scale so the sub-pixel periods become integers (nearest integer)."""
    ipx, ipy = int(round(px)), int(round(py))
    sx, sy = ipx / px, ipy / py
    H, W = img.shape[:2]
    out = cv2.resize(img, (int(round(W * sx)), int(round(H * sy))), interpolation=cv2.INTER_LINEAR)
    return out, ipx, ipy, sx, sy


def kmeans_lab(pixels_lab, k, seed=0):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    data = pixels_lab.astype(np.float32)
    comp, labels, centers = cv2.kmeans(data, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    return labels.ravel(), centers, comp / len(data)


def deltaE(lab1, lab2):
    return skcolor.deltaE_ciede2000(lab1, lab2)


def otsu_binary(L):
    L8 = np.clip(L / 100.0 * 255, 0, 255).astype(np.uint8)
    t, b = cv2.threshold(L8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (b > 0), t / 255.0 * 100


def min_thread_grid(L_tile, B_px, min_n=4, max_pitch=16.0, knee=0.01, n_phase=4):
    """Coarsest thread grid (ends x picks) whose nearest-neighbour up-sampling reproduces the
    pixel-level binary tile. The grid origin (phase) is searched as well, because cells averaged
    over a misaligned window blur the draft."""
    py, px = B_px.shape
    L = L_tile.astype(np.float32)
    def draft_at(ne, npk, ox, oy):
        Ls = np.roll(np.roll(L, -oy, 0), -ox, 1)
        small = cv2.resize(Ls, (ne, npk), interpolation=cv2.INTER_AREA)
        t = cv2.threshold(np.clip(small / 100 * 255, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
        up = cv2.resize(t.astype(np.uint8), (px, py), interpolation=cv2.INTER_NEAREST) > 0
        up = np.roll(np.roll(up, oy, 0), ox, 1)
        return t, float(np.mean(up != B_px))
    table = []; best = None
    for ne in range(max(min_n, int(px / max_pitch)), px // 3 + 1):
        fx = px / ne
        for npk in range(max(min_n, int(py / max_pitch)), py // 3 + 1):
            fy = py / npk
            errs = []
            for kx in range(n_phase):
                for ky in range(n_phase):
                    ox, oy = int(round(kx * fx / n_phase)), int(round(ky * fy / n_phase))
                    t, e = draft_at(ne, npk, ox, oy)
                    errs.append((e, ox, oy, t))
            e, ox, oy, t = min(errs, key=lambda r: r[0])
            table.append((ne, npk, e))
            if best is None or e < best[2]: best = (ne, npk, e, t)
    if best is None:
        return 1.0, 1.0, 0.0, B_px.copy(), [(px, py, 0.0)]
    emin = best[2]
    ok = [r for r in table if r[2] <= emin + knee]
    ok.sort(key=lambda r: r[0] * r[1])
    ne, npk, err = ok[0]
    fx, fy = px / ne, py / npk
    errs = []
    for kx in range(n_phase):
        for ky in range(n_phase):
            ox, oy = int(round(kx * fx / n_phase)), int(round(ky * fy / n_phase))
            t, e = draft_at(ne, npk, ox, oy); errs.append((e, ox, oy, t))
    e, ox, oy, t = min(errs, key=lambda r: r[0])
    return fx, fy, e, t, table


def draft_from_binary(B):
    """B: picks x ends, True = pattern (supplementary) float visible.
    Returns threading (ends->shaft), treadling (picks->treadle), tie-up (shaft x treadle)."""
    cols = [tuple(c) for c in B.T]
    rows = [tuple(r) for r in B]
    ucol = {c: i for i, c in enumerate(dict.fromkeys(cols))}
    urow = {r: i for i, r in enumerate(dict.fromkeys(rows))}
    threading = np.array([ucol[c] for c in cols])
    treadling = np.array([urow[r] for r in rows])
    S, T = len(ucol), len(urow)
    tieup = np.zeros((S, T), bool)
    for s, c in enumerate(ucol):
        for t, r in enumerate(urow):
            tieup[s, t] = c[list(urow.keys()).index(r)] if False else B[treadling == t][:, threading == s][0, 0]
    return threading, treadling, tieup


def drawdown(threading, treadling, tieup):
    return tieup[threading][:, treadling].T  # picks x ends


def project_to_shafts(B, H, iters=30, seed=0):
    """Cluster end-columns into at most H classes (k-medoids, Hamming) and rebuild the draft."""
    cols = B.T.astype(np.uint8)
    n = len(cols)
    uniq, inv = np.unique(cols, axis=0, return_inverse=True)
    if len(uniq) <= H:
        return B.copy(), 0.0
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(8):
        med = uniq[rng.choice(len(uniq), H, replace=False)]
        for _ in range(iters):
            d = (uniq[:, None, :] != med[None, :, :]).sum(-1)
            a = d.argmin(1)
            new = med.copy()
            for h in range(H):
                m = uniq[a == h]
                if len(m):
                    new[h] = (m.mean(0) > 0.5)
            if np.array_equal(new, med): break
            med = new
        d = (uniq[:, None, :] != med[None, :, :]).sum(-1)
        a = d.argmin(1)
        rec = med[a][inv].T.astype(bool)
        err = np.mean(rec != B)
        if best is None or err < best[1]:
            best = (rec, err)
    return best


def max_float(B):
    """Longest run of pattern floats along picks and ends."""
    def runs(a):
        best = 0
        for row in a:
            r = 0
            for v in row:
                r = r + 1 if v else 0
                best = max(best, r)
        return best
    return int(runs(B)), int(runs(B.T))


def symmetry_scores(B):
    """Best-shift agreement with mirror / rotation images of the tile (wallpaper tests)."""
    py, px = B.shape
    def best(Bt):
        s = 0
        for dy in range(py):
            for dx in range(px):
                s = max(s, np.mean(np.roll(np.roll(Bt, dy, 0), dx, 1) == B))
        return float(s)
    out = {"mirror_x": best(B[:, ::-1]), "mirror_y": best(B[::-1, :]), "rot180": best(B[::-1, ::-1])}
    if px == py:
        out["rot90"] = best(np.rot90(B))
    else:
        out["rot90"] = 0.0
    return out


def wallpaper_group(sym, thr=0.97):
    mx, my, r2, r4 = sym["mirror_x"] >= thr, sym["mirror_y"] >= thr, sym["rot180"] >= thr, sym["rot90"] >= thr
    if r4 and mx and my: return "p4m"
    if r4: return "p4"
    if mx and my: return "pmm or cmm"
    if mx or my: return "pm or cm"
    if r2: return "p2"
    return "p1"


def write_wif(path, threading, treadling, tieup, name, colors):
    S, T = tieup.shape
    L = ["[WIF]", "Version=1.1", "Date=2026-10-08", "Developers=kohyo-digitisation", "Source Program=analyze.py", "",
         "[CONTENTS]", "COLOR PALETTE=yes", "WEAVING=yes", "WARP=yes", "WEFT=yes", "TIEUP=yes", "THREADING=yes",
         "TREADLING=yes", "COLOR TABLE=yes", "TEXT=yes", "",
         "[TEXT]", f"Title={name}", "",
         "[COLOR PALETTE]", "Entries=2", "Range=0,255", "",
         "[COLOR TABLE]", f"1={colors[0][0]},{colors[0][1]},{colors[0][2]}", f"2={colors[1][0]},{colors[1][1]},{colors[1][2]}", "",
         "[WEAVING]", f"Shafts={S}", f"Treadles={T}", "Rising Shed=yes", "",
         "[WARP]", f"Threads={len(threading)}", "Color=1", "", "[WEFT]", f"Threads={len(treadling)}", "Color=2", "",
         "[TIEUP]"]
    for t in range(T):
        sh = [str(s + 1) for s in range(S) if tieup[s, t]]
        L.append(f"{t + 1}={','.join(sh) if sh else ''}")
    L += ["", "[THREADING]"] + [f"{i + 1}={s + 1}" for i, s in enumerate(threading)]
    L += ["", "[TREADLING]"] + [f"{i + 1}={t + 1}" for i, t in enumerate(treadling)]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def save_indexed(index, palette_rgb8, path):
    im = Image.fromarray(index.astype(np.uint8), "P")
    pal = list(np.asarray(palette_rgb8).astype(np.uint8).ravel()) + [0] * (768 - 3 * len(palette_rgb8))
    im.putpalette(pal)
    im.save(path, compression=None) if str(path).endswith(".tif") else im.save(path)


def point_paper(B, path, scale=10, heavy=10):
    py, px = B.shape
    img = np.ones((py * scale + 1, px * scale + 1, 3), np.uint8) * 255
    mask = np.zeros((py * scale + 1, px * scale + 1), bool)
    mask[:py * scale, :px * scale] = np.repeat(np.repeat(B, scale, 0), scale, 1)
    img[mask] = (40, 40, 40)
    for i in range(0, py * scale + 1, scale):
        img[i, :] = 160 if (i // scale) % heavy else 0
    for j in range(0, px * scale + 1, scale):
        img[:, j] = 160 if (j // scale) % heavy else 0
    Image.fromarray(img).save(path)


def synth_fabric(B, col_bg, col_fg, pitch, reps=(6, 8), angle=2.5, noise=0.04, seed=0):
    """Render a synthetic cloth image from a binary draft: yarn-like shading, tilt, vignette, noise."""
    rng = np.random.default_rng(seed)
    tile = np.where(B[..., None], col_fg, col_bg)
    img = np.tile(tile, (reps[0], reps[1], 1))
    img = np.repeat(np.repeat(img, pitch, 0), pitch, 1).astype(np.float32)
    H, W = img.shape[:2]
    yarn = 0.5 + 0.5 * np.cos(np.pi * (np.arange(W) % pitch) / pitch * 2)[None, :, None] * 0.25
    yarn2 = 0.5 + 0.5 * np.cos(np.pi * (np.arange(H) % pitch) / pitch * 2)[:, None, None] * 0.25
    img = img * (0.85 + 0.15 * yarn) * (0.85 + 0.15 * yarn2)
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.25 * (((yy - H / 2) / H) ** 2 + ((xx - W / 2) / W) ** 2) * 4
    img = img * vig[..., None]
    img = ndi.rotate(img, angle, reshape=False, order=1, mode="reflect")
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    img = ndi.gaussian_filter(img, (0.6, 0.6, 0))
    m = int(0.1 * min(H, W))
    return np.clip(img[m:-m, m:-m], 0, 1)


# ----------------------------------------------------------------------------- core

def digitise(rgb, pid, heddles, max_period=160, store=None, verbose=True):
    t0 = time.time()
    flat, trend = flatten_illumination(rgb)
    rect0, ang, angscan = rectify(flat)
    ac0 = autocorr_window(rect0)
    px0, py0, basis, kind0, lscore, _, _, pk0, X0, Y0, _ = rect_repeat(ac0, max_period)
    rect, M = affine_rectify(rect0, X0, Y0)
    ac = autocorr_window(rect, size=520)
    px, py, basis2, kind, lscore2, vx, vy, pk, X, Y, vh = rect_repeat(ac, max_period)
    rs, ipx, ipy, sx, sy = resample_to_integer_period(rect, px, py)
    tile_a, resid_a, _ = fold_tile(rs, ipx, ipy)          # before perspective correction
    rs_affine = rs
    rs, Hp, anc_m, anc_i, drift, anc_s = perspective_rectify(rs, ipx, ipy, kind)
    ac2 = autocorr_window(rs, size=520)
    px2, py2, _, _, _, _, _, _, _, _, _ = rect_repeat(ac2, max_period)
    persp_ok = abs(px2 - ipx) / ipx < 0.05 and abs(py2 - ipy) / ipy < 0.05
    if not persp_ok:
        # consistency guard: the homography must not change the period by more than 5 percent;
        # otherwise (too few anchors, small crop) keep the affine-stage image
        rs, Hp, px2, py2 = rs_affine, np.eye(3), float(ipx), float(ipy)
        drift = np.zeros(1); anc_s = np.zeros(1); anc_m = anc_i = np.zeros((1, 2))
    # fold-residual refinement: the period that minimises the per-repeat residual of the folded
    # image, searched within 6 percent of the autocorrelation estimate, with parabolic sub-pixel fit
    px2, py2, fold_curve = fold_refine(rs, px2, py2)
    rs, ipx, ipy, sx2, sy2 = resample_to_integer_period(rs, px2, py2)
    px, py = px2, py2
    tile_b, resid_b, _ = fold_tile(rs, ipx, ipy)            # plain mean after perspective correction
    tile, resid, shifts = fold_tile_aligned(rs, ipx, ipy)   # per-repeat re-registered mean
    Lt = skcolor.rgb2lab(tile)[..., 0]
    cshift = int(np.argmin(np.abs(np.diff(np.r_[Lt.mean(0), Lt.mean(0)[0]]))))
    rshift = int(np.argmin(np.abs(np.diff(np.r_[Lt.mean(1), Lt.mean(1)[0]]))))
    rs = np.roll(np.roll(rs, -cshift - 1, 1), -rshift - 1, 0)
    tile, resid, shifts = fold_tile_aligned(rs, ipx, ipy)
    tpx, tpy, tsx, tsy, _, _ = thread_pitch(rs)
    lab_tile = skcolor.rgb2lab(tile)
    # binary draft (pattern floats vs ground) from L* with Otsu
    B_px, thr = otsu_binary(lab_tile[..., 0])
    fg_frac = float(B_px.mean())
    # ensure "pattern" = minority / lighter class consistently: pattern floats = lighter (supplementary yarn)
    fx, fy, grid_err, B, grid_table = min_thread_grid(lab_tile[..., 0], B_px)
    ends, picks = B.shape[1], B.shape[0]
    threading, treadling, tieup = draft_from_binary(B)
    S, T = tieup.shape
    Bp, proj_err = project_to_shafts(B, heddles)
    thr_p, trd_p, tie_p = draft_from_binary(Bp)
    assert np.array_equal(drawdown(thr_p, trd_p, tie_p), Bp)
    assert np.array_equal(drawdown(threading, treadling, tieup), B)
    proj_curve = {}
    for Hh in range(2, 17):
        proj_curve[Hh] = round(float(project_to_shafts(B, Hh)[1]), 4)
    fl = max_float(B)
    sym = symmetry_scores(B)
    group = wallpaper_group(sym)
    # colour quantisation of the mean tile in CIELAB
    pix = lab_tile.reshape(-1, 3)
    kres = {}
    for k in range(2, 7):
        lab_, cen, _ = kmeans_lab(pix, k, seed=0)
        q = cen[lab_]
        kres[k] = {"mean_dE": float(deltaE(pix, q).mean()), "p95_dE": float(np.percentile(deltaE(pix, q), 95))}
    # two-colour palette (ground / pattern) from Otsu classes
    c_bg = np.median(tile[~B_px], axis=0); c_fg = np.median(tile[B_px], axis=0)
    pal8 = (np.array([c_bg, c_fg]) * 255).round().astype(int)
    # reconstruction fidelity: tile the quantised 2-colour tile and the mean tile over the rectified image
    H, W = rs.shape[:2]
    ny, nx = H // ipy, W // ipx
    rec_mean = np.tile(tile, (ny, nx, 1))
    Bq = cv2.resize(B.astype(np.uint8), (ipx, ipy), interpolation=cv2.INTER_NEAREST) > 0
    rec_q = np.where(Bq[..., None], c_fg, c_bg)
    rec_q = np.tile(rec_q, (ny, nx, 1))
    ref = rs[:ny * ipy, :nx * ipx]
    def fid(rec):
        g1, g2 = gray(ref), gray(rec)
        s = ssim(g1, g2, data_range=1.0)
        mse = np.mean((ref - rec) ** 2)
        psnr = 10 * np.log10(1.0 / mse)
        dE = deltaE(skcolor.rgb2lab(ref), skcolor.rgb2lab(rec))
        e1 = cv2.Canny((g1 * 255).astype(np.uint8), 60, 140) > 0
        e2 = cv2.Canny((g2 * 255).astype(np.uint8), 60, 140) > 0
        e2d = ndi.binary_dilation(e2, iterations=1); e1d = ndi.binary_dilation(e1, iterations=1)
        prec = (e2 & e1d).sum() / max(e2.sum(), 1); rec_ = (e1 & e2d).sum() / max(e1.sum(), 1)
        f1 = 2 * prec * rec_ / max(prec + rec_, 1e-9)
        gs1, gs2 = ndi.gaussian_filter(g1, 2.0), ndi.gaussian_filter(g2, 2.0)
        s_motif = ssim(gs1, gs2, data_range=1.0)
        return {"ssim": float(s), "ssim_motif_scale": float(s_motif), "psnr_db": float(psnr), "mean_dE00": float(dE.mean()), "p95_dE00": float(np.percentile(dE, 95)), "edge_f1": float(f1)}
    fid_mean, fid_q = fid(rec_mean), fid(rec_q)
    # periodicity confidence: autocorr peak value at the repeat, and ratio to the strongest off-lattice value
    res = {
        "id": pid, "heddles_label": heddles,
        "image_size_px": [int(rgb.shape[1]), int(rgb.shape[0])],
        "rectified_size_px": [int(rect.shape[1]), int(rect.shape[0])],
        "skew_deg": round(ang, 2),
        "repeat_px_subpixel": [round(px, 2), round(py, 2)],
        "repeat_px_int": [ipx, ipy],
        "resample_scale": [round(sx, 4), round(sy, 4)],
        "autocorr_peak": [round(vx, 3), round(vy, 3)],
        "lattice_score": round(float(lscore2), 3),
        "primitive_basis_px": [[round(float(basis[0][0]), 2), round(float(basis[0][1]), 2)], [round(float(basis[1][0]), 2), round(float(basis[1][1]), 2)]],
        "lattice_type": kind, "half_lattice_peak": round(vh, 3), "axis_vectors_before_affine": [np.round(X0, 2).tolist(), np.round(Y0, 2).tolist()],
        "affine_matrix": np.round(M, 4).tolist(),
        "shear_deg": round(float(np.degrees(np.arctan2(M[0, 1], M[0, 0]))), 2),
        "repeat_px_before_affine": [round(px0, 2), round(py0, 2)],
        "n_repeats_used": [int(nx), int(ny)],
        "repeat_residual_rmse_mean": round(float(resid.mean()), 4), "repeat_residual_rmse_mean_before_perspective": round(float(resid_a.mean()), 4), "repeat_residual_rmse_mean_plain_after_perspective": round(float(resid_b.mean()), 4), "local_shift_px_max": float(np.abs(shifts).max()), "local_shift_px_rms": round(float(np.sqrt((shifts ** 2).mean())), 3), "yarn_autocorr_pitch_px": [round(float(tpx), 2), round(float(tpy), 2)], "yarn_autocorr_peak": [round(tsx, 3), round(tsy, 3)], "anchor_drift_px_max": round(float(drift.max()), 2), "anchor_drift_px_mean": round(float(drift.mean()), 2), "anchor_match_score_mean": round(float(anc_s.mean()), 3), "n_anchors": int(len(drift)), "perspective_accepted": bool(persp_ok), "homography": np.round(Hp, 6).tolist(), "repeat_px_after_perspective": [round(px2, 2), round(py2, 2)], "repeat_px_acf_after_perspective": [round(fold_curve["acf"][0], 2), round(fold_curve["acf"][1], 2)], "fold_curve": fold_curve,
        "repeat_residual_rmse_max": round(float(resid.max()), 4),
        "otsu_threshold_L": round(float(thr), 2),
        "pattern_fraction": round(fg_frac, 4),
        "thread_pitch_px": [round(fx, 3), round(fy, 3)],
        "grid_error": round(float(grid_err), 4),
        "ends_per_repeat": int(ends), "picks_per_repeat": int(picks),
        "shafts_required": int(S), "treadles_required": int(T),
        "shaft_projection_error": round(float(proj_err), 4), "shaft_projection_curve": proj_curve,
        "shafts_after_projection": int(tie_p.shape[0]), "treadles_after_projection": int(tie_p.shape[1]),
        "max_float_weft": fl[0], "max_float_warp": fl[1],
        "symmetry": {k: round(v, 4) for k, v in sym.items()}, "wallpaper_group": group,
        "palette_rgb8": pal8.tolist(),
        "kmeans_dE": kres,
        "fidelity_mean_tile": fid_mean, "fidelity_two_colour": fid_q,
        "runtime_s": round(time.time() - t0, 2),
    }
    if store is not None:
        np.savez_compressed(store, rgb_small=cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5), flat_small=cv2.resize(flat, (0, 0), fx=0.5, fy=0.5),
                            trend=cv2.resize(trend.astype(np.float32), (0, 0), fx=0.5, fy=0.5), rect_small=cv2.resize(rect, (0, 0), fx=0.5, fy=0.5), rect0_small=cv2.resize(rect0, (0, 0), fx=0.5, fy=0.5), ac0=ac0[ac0.shape[0] // 2 - 180: ac0.shape[0] // 2 + 181, ac0.shape[1] // 2 - 180: ac0.shape[1] // 2 + 181], grid_table=np.array(grid_table), tile_a=tile_a, resid_a=resid_a, tile_b=tile_b, resid_b=resid_b, shifts=shifts, anc_m=anc_m, anc_i=anc_i, drift=drift, rs_small=cv2.resize(rs, (0, 0), fx=0.5, fy=0.5), basis=basis, basis2=basis2, X=X, Y=Y, X0=X0, Y0=Y0,
                            angscan_a=angscan[0], angscan_s=angscan[1], ac=ac[ac.shape[0] // 2 - 180: ac.shape[0] // 2 + 181, ac.shape[1] // 2 - 180: ac.shape[1] // 2 + 181],
                            peaks=pk, tile=tile, resid=resid, B_px=B_px, B=B, Bp=Bp, threading=threading, treadling=treadling, tieup=tieup,
                            thr_p=thr_p, trd_p=trd_p, tie_p=tie_p, ref_crop=ref[:3 * ipy, :3 * ipx], rec_q_crop=rec_q[:3 * ipy, :3 * ipx],
                            rec_mean_crop=rec_mean[:3 * ipy, :3 * ipx], c_bg=c_bg, c_fg=c_fg)
    if verbose:
        print(f"{pid}: skew {ang:+.2f} deg, {kind}, repeat {px:.2f}x{py:.2f} px -> {ends}x{picks} threads (pitch {fx:.2f},{fy:.2f}), shafts {S} (label {heddles}), "
              f"proj err {proj_err:.3f}, SSIM {fid_q['ssim']:.3f}, dE {fid_q['mean_dE00']:.2f}, {group}, {time.time()-t0:.1f}s")
    return res, dict(tile=tile, B=B, B_px=B_px, c_bg=c_bg, c_fg=c_fg, fx=fx, fy=fy, threading=threading, treadling=treadling, tieup=tieup, pal8=pal8, rs=rs, ipx=ipx, ipy=ipy)


def draft_agreement(B0, B_px, n_phase=4):
    """Pixel draft resampled onto the true cell grid with a search over the grid phase (sub-cell
    pixel offsets) and over whole-cell cyclic shifts; returns the best agreement and that draft."""
    py, px = B_px.shape; fy, fx = py / B0.shape[0], px / B0.shape[1]
    best, bestB = 0.0, None
    for kx in range(n_phase):
        for ky in range(n_phase):
            ox, oy = int(round(kx * fx / n_phase)), int(round(ky * fy / n_phase))
            Bs = np.roll(np.roll(B_px, -oy, 0), -ox, 1).astype(np.float32)
            B1r = cv2.resize(Bs, (B0.shape[1], B0.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
            for dy in range(B0.shape[0]):
                for dx in range(B0.shape[1]):
                    a = float(np.mean(np.roll(np.roll(B1r, dy, 0), dx, 1) == B0))
                    if a > best: best, bestB = a, np.roll(np.roll(B1r, dy, 0), dx, 1)
    return best, bestB


def bootstrap(rgb, heddles, n=12, frac=0.8, max_period=160):
    H, W = rgb.shape[:2]
    out = []
    for i in range(n):
        h, w = int(H * frac), int(W * frac)
        y, x = RNG.integers(0, H - h), RNG.integers(0, W - w)
        try:
            r, _ = digitise(rgb[y:y + h, x:x + w], "bs", heddles, max_period=max_period, verbose=False)
            out.append({"repeat_px": r["repeat_px_subpixel"], "threads": [r["ends_per_repeat"], r["picks_per_repeat"]], "shafts": r["shafts_required"], "ssim": r["fidelity_two_colour"]["ssim"]})
        except Exception as e:
            out.append({"error": str(e)})
    return out


def main():
    results = {"patterns": [], "roundtrip": [], "bootstrap": {}}
    arrays = {}
    for pid, fn, name_en, name_th, hed in PATTERNS:
        rgb = load_rgb(RAW / fn)
        res, arr = digitise(rgb, pid, hed, store=OUT / "npz" / f"{pid}.npz")
        res.update({"file": fn, "name_en": name_en, "name_th": name_th})
        results["patterns"].append(res)
        arrays[pid] = arr
        d = OUT / "loom" / pid; d.mkdir(exist_ok=True)
        Image.fromarray((arr["tile"] * 255).astype(np.uint8)).save(d / "repeat_mean_tile.png")
        save_indexed(arr["B"].astype(np.uint8), arr["pal8"], d / "repeat_draft_1px_per_thread.png")
        tiled = np.tile(arr["B"], (8, 8)).astype(np.uint8)
        save_indexed(tiled, arr["pal8"], d / "tc2_8x8_repeats.tif")
        point_paper(arr["B"], d / "point_paper.png")
        write_wif(d / f"{pid}.wif", arr["threading"], arr["treadling"], arr["tieup"], f"{name_en} ({name_th}) Koh Yo", arr["pal8"])
        with open(d / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    # bootstrap stability
    print("bootstrap ...")
    for pid, fn, _, _, hed in PATTERNS:
        results["bootstrap"][pid] = bootstrap(load_rgb(RAW / fn), hed)
    # round trip: synthetic cloth from each extracted draft -> pipeline -> compare draft
    print("round trip ...")
    for pid, fn, _, _, hed in PATTERNS:
        a = arrays[pid]
        for pitch, angle, noise in [(5, 2.5, 0.04), (7, -4.0, 0.08), (4, 1.0, 0.12)]:
            pitch = int(min(pitch, 150 // max(a["B"].shape)))
            reps = (int(np.ceil(760 / (a["B"].shape[0] * pitch))) + 1, int(np.ceil(860 / (a["B"].shape[1] * pitch))) + 1)
            syn = synth_fabric(a["B"], a["c_bg"], a["c_fg"], pitch, reps=reps, angle=angle, noise=noise, seed=1)
            r, b = digitise(syn, pid + "syn", hed, verbose=False)
            # compare drafts allowing cyclic shift and 1-thread tolerance in size
            B0, B1 = a["B"], b["B"]
            ok_size = B0.shape == B1.shape
            # pixel-level test independent of the cell-grid step: recovered pixel draft resampled to the true grid
            agree, B1r = draft_agreement(B0, b["B_px"])
            if (pitch, angle) == (5, 2.5) or pid != "P1" and angle == 2.5:
                np.savez_compressed(OUT / "npz" / f"rt_{pid}.npz", syn=cv2.resize(syn, (0, 0), fx=0.5, fy=0.5), true=B0, rec=B1r, rec_px=b["B_px"], tile=b["tile"])
            agree_cells = 0.0
            if ok_size:
                for dy in range(B0.shape[0]):
                    for dx in range(B0.shape[1]):
                        agree_cells = max(agree_cells, float(np.mean(np.roll(np.roll(B1, dy, 0), dx, 1) == B0)))
            results["roundtrip"].append({"id": pid, "pitch": pitch, "angle_deg": angle, "noise": noise, "recovered_skew_deg": r["skew_deg"],
                                         "recovered_pitch": r["thread_pitch_px"], "recovered_shear_deg": r["shear_deg"], "recovered_threads": [r["ends_per_repeat"], r["picks_per_repeat"]],
                                         "true_threads": [int(B0.shape[1]), int(B0.shape[0])], "size_match": bool(ok_size), "draft_agreement": round(agree, 4), "cell_draft_agreement": round(agree_cells, 4), "true_repeat_px": [int(B0.shape[1] * pitch), int(B0.shape[0] * pitch)], "recovered_repeat_px": r["repeat_px_after_perspective"],
                                         "shafts": r["shafts_required"]})
            print(f"  {pid} pitch {pitch} angle {angle}: size {ok_size}, agreement {agree:.3f}, skew {r['skew_deg']}")
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("done")


if __name__ == "__main__":
    main()
