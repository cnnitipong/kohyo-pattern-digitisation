#!/usr/bin/env python3
"""Second-round analyses for the revised manuscript (all CPU, NumPy / scikit-learn only).
1. Baseline lattice rules (shortest-vector vs shell-consistency) on full images and bootstrap crops.
2. Fold-residual cross-check of the repeat period.
3. Noise2Noise-style learned denoiser (MLP on patches, trained on pairs of repeat blocks).
4. GMM + MRF (ICM) draft versus Otsu draft.
5. Resolution experiment: cell-grid recovery against synthetic thread pitch.
6. Noise sweep for round-trip draft agreement.
7. Pattern retrieval: LBP features, SVM / kNN, cross-validated.
Writes data/results_v2.json and data/npz/v2_*.npz."""
import json, sys, time
from pathlib import Path
import numpy as np, cv2
from scipy import ndimage as ndi
from skimage import color as skcolor
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
from sklearn.neural_network import MLPRegressor
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A
from analyze import *

OUT = ROOT / "data"
R1 = json.load(open(OUT / "results.json", encoding="utf-8"))
REF = {p["id"]: np.array(p["repeat_px_after_perspective"]) for p in R1["patterns"]}
PRE = {p["id"]: np.array(p["repeat_px_before_affine"]) for p in R1["patterns"]}
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ 1. baseline lattice rule
def shortest_vector_lattice(ac, max_period):
    pk = local_maxima(ac, 4, 0.2)
    pk = pk[(np.abs(pk[:, 0]) <= max_period) & (np.abs(pk[:, 1]) <= max_period)]
    pk = pk[(pk[:, 0] != 0) | (pk[:, 1] != 0)]
    pk = pk[(pk[:, 1] > 0) | ((pk[:, 1] == 0) & (pk[:, 0] > 0))]
    pk = pk[np.argsort(np.hypot(pk[:, 0], pk[:, 1]))]
    v1 = pk[0, :2]
    for p in pk[1:]:
        if abs(v1[0] * p[1] - v1[1] * p[0]) > 0.3 * np.hypot(*v1) * np.hypot(*p[:2]):
            return v1, p[:2]
    return v1, pk[1, :2]


def repeat_with_rule(rgb, rule):
    flat, _ = flatten_illumination(rgb); rect, _, _ = rectify(flat); ac = autocorr_window(rect)
    if rule == "shell":
        (s, a, v1, v2, s2), _, _ = lattice_candidates(ac, 160)
    else:
        v1, v2 = shortest_vector_lattice(ac, 160)
    X, Y = axis_vectors(v1, v2)
    return np.array([np.hypot(*X), np.hypot(*Y)])


def baseline_rules():
    out = {}
    for pid, fn, _, _, h in PATTERNS:
        rgb = load_rgb(RAW / fn); H, W = rgb.shape[:2]
        res = {}
        for rule in ("shortest", "shell"):
            full = repeat_with_rule(rgb, rule)
            crops = []
            rng = np.random.default_rng(3)
            for _ in range(12):
                hh, ww = int(0.8 * H), int(0.8 * W); y, x = rng.integers(0, H - hh), rng.integers(0, W - ww)
                try: crops.append(repeat_with_rule(rgb[y:y + hh, x:x + ww], rule))
                except Exception: crops.append(np.array([np.nan, np.nan]))
            crops = np.array(crops)
            ok = np.all(np.abs(crops - PRE[pid]) / PRE[pid] < 0.03, axis=1)
            res[rule] = {"full_repeat_px": np.round(full, 2).tolist(), "full_ok": bool(np.all(np.abs(full - PRE[pid]) / PRE[pid] < 0.03)),
                         "crops_ok": int(np.nansum(ok)), "crops_n": 12}
        out[pid] = res
        print("rule", pid, res)
    return out


# ------------------------------------------------------------------ 2. fold-residual cross-check
def fold_curves():
    out = {}
    for pid, fn, _, _, h in PATTERNS:
        d = np.load(OUT / "npz" / f"{pid}.npz")
        rs = cv2.resize(d["rs_small"], (0, 0), fx=2, fy=2)  # stored at half size
        g = gray(rs)[..., None]
        ipx, ipy = [int(round(v)) for v in REF[pid]]
        cx = [(P, float(fold_tile(g, P, ipy)[1].mean())) for P in range(int(0.8 * ipx), int(1.2 * ipx) + 1)]
        cy = [(P, float(fold_tile(g, ipx, P)[1].mean())) for P in range(int(0.8 * ipy), int(1.2 * ipy) + 1)]
        bx, by = min(cx, key=lambda t: t[1])[0], min(cy, key=lambda t: t[1])[0]
        out[pid] = {"curve_x": cx, "curve_y": cy, "argmin_x": bx, "argmin_y": by, "acf_px": REF[pid].tolist(),
                    "diff_px": [round(bx - REF[pid][0], 2), round(by - REF[pid][1], 2)]}
        print("fold", pid, bx, by, REF[pid])
    return out


# ------------------------------------------------------------------ 3. Noise2Noise MLP denoiser
def blocks_of(rs, ipx, ipy):
    H, W = rs.shape[:2]; ny, nx = H // ipy, W // ipx
    return rs[:ny * ipy, :nx * ipx].reshape(ny, ipy, nx, ipx, 3).transpose(0, 2, 1, 3, 4).reshape(ny * nx, ipy, ipx, 3)


def aligned_blocks(rs, ipx, ipy, m=6):
    """Re-registered blocks (same registration as fold_tile_aligned)."""
    tile, resid, shifts = fold_tile_aligned(rs, ipx, ipy)
    H, W = rs.shape[:2]; ny, nx = H // ipy, W // ipx
    out = []
    for j in range(ny):
        for i in range(nx):
            dy, dx = shifts[j, i].astype(int); y0, x0 = j * ipy + dy, i * ipx + dx
            if 0 <= y0 and y0 + ipy <= H and 0 <= x0 and x0 + ipx <= W:
                out.append(rs[y0:y0 + ipy, x0:x0 + ipx])
    return np.array(out), tile


def patches(img, r=3):
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode="wrap")
    H, W = img.shape[:2]
    cols = [pad[dy:dy + H, dx:dx + W] for dy in range(2 * r + 1) for dx in range(2 * r + 1)]
    return np.stack(cols, axis=2).reshape(H * W, -1)


def n2n_denoise(blocks, n_pairs=40, r=3, seed=0):
    """Train an MLP to predict the centre pixel of block B from the patch of block A at the same
    lattice position (Noise2Noise: both are noisy views of the same clean tile)."""
    rng = np.random.default_rng(seed)
    nb = len(blocks); X, Y = [], []
    for _ in range(n_pairs):
        a, b = rng.choice(nb, 2, replace=False)
        X.append(patches(blocks[a], r)); Y.append(blocks[b].reshape(-1, 3))
    X, Y = np.vstack(X), np.vstack(Y)
    idx = rng.choice(len(X), min(60000, len(X)), replace=False)
    mlp = MLPRegressor(hidden_layer_sizes=(96, 48), max_iter=60, batch_size=512, learning_rate_init=2e-3, random_state=seed, early_stopping=True, n_iter_no_change=6)
    t0 = time.time(); mlp.fit(X[idx], Y[idx]); tt = time.time() - t0
    den = np.array([mlp.predict(patches(b, r)).reshape(b.shape) for b in blocks])
    return np.clip(den.mean(0), 0, 1), tt, mlp.n_iter_


def learned_tiles():
    out = {}
    for pid, fn, _, _, h in PATTERNS:
        d = np.load(OUT / "npz" / f"{pid}.npz")
        rs = cv2.resize(d["rs_small"], (0, 0), fx=2, fy=2)
        ipx, ipy = d["tile"].shape[1], d["tile"].shape[0]
        blk, tile_mean = aligned_blocks(rs, ipx, ipy)
        tile_med = np.median(blk, axis=0)
        tile_n2n, tt, it = n2n_denoise(blk)
        def resid(t): return float(np.sqrt(((blk - t[None]) ** 2).mean()))
        def sharp(t): g = gray(t); return float(np.mean(np.abs(np.gradient(g)[0])) + np.mean(np.abs(np.gradient(g)[1])))
        # leave-out consistency: predict held-out blocks, SSIM at motif scale against each block
        def ss(t):
            return float(np.mean([ssim(ndi.gaussian_filter(gray(b), 2), ndi.gaussian_filter(gray(t), 2), data_range=1.0) for b in blk]))
        r = {"n_blocks": int(len(blk)), "train_s": round(tt, 1), "n_iter": int(it),
             "resid": {"mean": resid(tile_mean), "median": resid(tile_med), "n2n": resid(tile_n2n)},
             "sharpness": {"mean": sharp(tile_mean), "median": sharp(tile_med), "n2n": sharp(tile_n2n)},
             "block_ssim_motif": {"mean": ss(tile_mean), "median": ss(tile_med), "n2n": ss(tile_n2n)}}
        out[pid] = r
        np.savez_compressed(OUT / "npz" / f"v2_{pid}_tiles.npz", mean=tile_mean, median=tile_med, n2n=tile_n2n, blocks=blk[:6])
        print("n2n", pid, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r.items() if isinstance(v, dict)}, "train", round(tt, 1), "s")
    return out


# ------------------------------------------------------------------ 4. GMM + MRF draft
def gmm_mrf_draft(tile, beta=0.9, iters=8):
    lab = skcolor.rgb2lab(tile); H, W = lab.shape[:2]
    feats = np.concatenate([lab, ndi.gaussian_filter(lab, (1, 1, 0))], axis=2).reshape(-1, 6)
    gm = GaussianMixture(2, covariance_type="full", random_state=0).fit(feats)
    ll = gm.predict_proba(feats).reshape(H, W, 2)
    lab_cls = np.argmax(ll, axis=2)
    # class 1 = lighter (pattern float), as in the Otsu convention
    if lab[..., 0][lab_cls == 1].mean() < lab[..., 0][lab_cls == 0].mean():
        lab_cls = 1 - lab_cls; ll = ll[..., ::-1]
    unary = -np.log(np.clip(ll, 1e-6, 1))
    x = lab_cls.copy()
    for _ in range(iters):  # ICM on a 4-neighbour Potts model with wrap-around
        nb = sum(np.roll(x, s, a) for s in (-1, 1) for a in (0, 1))
        cost0 = unary[..., 0] + beta * nb            # cost of label 0: neighbours with label 1
        cost1 = unary[..., 1] + beta * (4 - nb)
        x = (cost1 < cost0).astype(int)
    return x.astype(bool), lab_cls.astype(bool)


def isolated_fraction(B):
    nb = sum(np.roll(B.astype(int), s, a) for s in (-1, 1) for a in (0, 1))
    iso = ((B & (nb == 0)) | (~B & (nb == 4)))
    return float(iso.mean())


def drafts_compare():
    out = {}
    for pid, fn, _, _, h in PATTERNS:
        d = np.load(OUT / "npz" / f"{pid}.npz"); t2 = np.load(OUT / "npz" / f"v2_{pid}_tiles.npz")
        res = {}
        for tname in ("mean", "n2n"):
            tile = t2[tname]
            B_otsu, _ = otsu_binary(skcolor.rgb2lab(tile)[..., 0])
            B_mrf, B_gmm = gmm_mrf_draft(tile)
            res[tname] = {"otsu_isolated": isolated_fraction(B_otsu), "gmm_isolated": isolated_fraction(B_gmm), "mrf_isolated": isolated_fraction(B_mrf),
                          "otsu_vs_mrf_disagreement": float(np.mean(B_otsu != B_mrf)), "pattern_fraction_otsu": float(B_otsu.mean()), "pattern_fraction_mrf": float(B_mrf.mean())}
            if tname == "n2n":
                np.savez_compressed(OUT / "npz" / f"v2_{pid}_drafts.npz", otsu=B_otsu, gmm=B_gmm, mrf=B_mrf)
        out[pid] = res
        print("draft", pid, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in res.items()})
    return out


# ------------------------------------------------------------------ 5/6. resolution and noise sweeps (synthetic)
def synth_case(B, c_bg, c_fg, pitch, angle, noise, seed=1):
    reps = (int(np.ceil(700 / (B.shape[0] * pitch))) + 1, int(np.ceil(800 / (B.shape[1] * pitch))) + 1)
    return synth_fabric(B, c_bg, c_fg, pitch, reps=reps, angle=angle, noise=noise, seed=seed)


def agreement(B0, B_px, draft_fn=None):
    return draft_agreement(B0, B_px)[0]


def sweeps():
    res_sweep, noise_sweep = [], []
    for pid, fn, _, _, h in PATTERNS:
        d = np.load(OUT / "npz" / f"{pid}.npz"); B = d["B"]; c_bg, c_fg = d["c_bg"], d["c_fg"]
        # use a reduced draft so that large pitches keep the repeat below the search limit
        for pitch in (2, 3, 4, 5, 6, 8):
            if max(B.shape) * pitch > 150:
                res_sweep.append({"id": pid, "pitch": pitch, "skipped": True}); continue
            syn = synth_case(B, c_bg, c_fg, pitch, 2.0, 0.06)
            try:
                r, b = digitise(syn, "s", h, verbose=False)
                Bm, Bmrf = gmm_mrf_draft(b["tile"])
                res_sweep.append({"id": pid, "pitch": pitch, "repeat_err_px": float(np.abs(np.array(r["repeat_px_after_perspective"]) - np.array([B.shape[1] * pitch, B.shape[0] * pitch])).max()),
                                  "cells_exact": bool(b["B"].shape == B.shape), "cells_rel_err": float(np.abs(np.array(b["B"].shape) / np.array(B.shape) - 1).max()),
                                  "agree_otsu": agreement(B, b["B_px"]), "agree_mrf": agreement(B, Bmrf), "shafts": r["shafts_required"]})
            except Exception as e:
                res_sweep.append({"id": pid, "pitch": pitch, "error": str(e)[:60]})
            print("res", res_sweep[-1])
        pitch = int(min(4, 150 // max(B.shape)))
        for noise in (0.02, 0.06, 0.10, 0.15, 0.20, 0.25):
            syn = synth_case(B, c_bg, c_fg, pitch, 2.0, noise)
            try:
                r, b = digitise(syn, "s", h, verbose=False)
                Bm, Bmrf = gmm_mrf_draft(b["tile"])
                noise_sweep.append({"id": pid, "noise": noise, "pitch": pitch, "repeat_err_px": float(np.abs(np.array(r["repeat_px_after_perspective"]) - np.array([B.shape[1] * pitch, B.shape[0] * pitch])).max()),
                                    "agree_otsu": agreement(B, b["B_px"]), "agree_mrf": agreement(B, Bmrf)})
            except Exception as e:
                noise_sweep.append({"id": pid, "noise": noise, "error": str(e)[:60]})
            print("noise", noise_sweep[-1])
    return res_sweep, noise_sweep


# ------------------------------------------------------------------ 7. retrieval
def lbp_features(g):
    f = []
    for P, Rr in ((8, 1), (16, 2), (24, 3)):
        lbp = local_binary_pattern((g * 255).astype(np.uint8), P, Rr, method="uniform")
        hist, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True); f.append(hist)
    return np.concatenate(f)


def retrieval(n_crops=40, size=160):
    X, y, meta = [], [], []
    for k, (pid, fn, _, _, h) in enumerate(PATTERNS):
        d = np.load(OUT / "npz" / f"{pid}.npz")
        for src, lab in (("rgb_small", "raw"), ("rs_small", "rect")):
            img = cv2.resize(d[src], (0, 0), fx=2, fy=2); g = gray(img); H, W = g.shape
            for _ in range(n_crops):
                yy, xx = RNG.integers(0, H - size), RNG.integers(0, W - size)
                c = g[yy:yy + size, xx:xx + size]
                c = (c - c.mean()) / (c.std() + 1e-6) * 0.15 + 0.5
                X.append(lbp_features(np.clip(c, 0, 1))); y.append(k); meta.append(lab)
    X, y, meta = np.array(X), np.array(y), np.array(meta)
    out = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, clf in (("svm", make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))), ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(5)))):
        for sub in ("raw", "rect", "all"):
            m = np.ones(len(y), bool) if sub == "all" else meta == sub
            pred = cross_val_predict(clf, X[m], y[m], cv=cv)
            cm = np.zeros((5, 5), int)
            for a, b in zip(y[m], pred): cm[a, b] += 1
            out[f"{name}_{sub}"] = {"accuracy": float(np.mean(pred == y[m])), "confusion": cm.tolist(), "n": int(m.sum())}
            print("retrieval", name, sub, round(out[f"{name}_{sub}"]["accuracy"], 4))
    # cross-domain: train on raw crops, test on rectified crops
    clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(X[meta == "raw"], y[meta == "raw"])
    out["svm_train_raw_test_rect"] = float(np.mean(clf.predict(X[meta == "rect"]) == y[meta == "rect"]))
    pca = PCA(2).fit_transform(StandardScaler().fit_transform(X))
    np.savez_compressed(OUT / "npz" / "v2_retrieval.npz", pca=pca, y=y, meta=meta)
    print("retrieval cross-domain", out["svm_train_raw_test_rect"])
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {"baseline_rules": baseline_rules(), "fold_curves": fold_curves(), "learned_tiles": learned_tiles(), "drafts": drafts_compare()}
    res["resolution_sweep"], res["noise_sweep"] = sweeps()
    res["retrieval"] = retrieval()
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "results_v2.json", "w", encoding="utf-8"), indent=1)
    print("done", res["runtime_s"], "s")
