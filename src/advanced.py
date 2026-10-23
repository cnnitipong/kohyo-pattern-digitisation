#!/usr/bin/env python3
"""Techniques integrated from the literature survey (all CPU, NumPy / OpenCV / scikit-image / scikit-learn):
A. multi-repeat sub-pixel super-resolution of the repeat tile (shift-and-add of phase-registered repeats),
   evaluated on real cloth and on synthetic cloth at 2 and 3 px per cell;
B. pairwise-displacement vote map as an alternative to the autocorrelation for lattice selection,
   evaluated on bootstrap crops;
C. statistical symmetry test (trimmed chi-square against the repeat-stack variance) versus the 80 % rule;
D. soft draft: two-class Gaussian mixture on cell lightness giving a per-cell confidence map;
E. texture (Gram-matrix) distance and homogeneity score as additional verification metrics;
F. Hu-moment descriptors of the drafts for the archive.
Writes data/results_adv.json and data/npz/adv_*.npz."""
import json, sys, time
from pathlib import Path
import numpy as np, cv2
from scipy import ndimage as ndi, stats
from skimage import color as skcolor
from skimage.registration import phase_cross_correlation
from skimage.feature import peak_local_max
from skimage.filters import gabor_kernel
from sklearn.mixture import GaussianMixture
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A
from analyze import *
from symmetrize import cell_lightness, ops_for, symmetrise, shafts_of, fidelity_motif
from analyze_v2 import aligned_blocks

OUT = ROOT / "data"
R1 = json.load(open(OUT / "results.json", encoding="utf-8"))
RNGA = np.random.default_rng(11)


# ----------------------------------------------------------------- A. super-resolution tile
def super_tile(blocks, s=3, upsample=20):
    """Shift-and-add: every repeat block is registered to the mean tile with sub-pixel precision and
    accumulated on an s-times finer grid; blocks at different sub-pixel phases supply the missing samples."""
    ref = gray(blocks.mean(0)).astype(np.float32)
    H, W = ref.shape
    acc = np.zeros((H * s, W * s, 3), np.float32); stack = []
    shifts = []
    for b in blocks:
        sh, _, _ = phase_cross_correlation(ref, gray(b).astype(np.float32), upsample_factor=upsample)
        shifts.append(sh)
        M = np.float32([[s, 0, s * sh[1]], [0, s, s * sh[0]]])
        up = cv2.warpAffine(b.astype(np.float32), M, (W * s, H * s), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        stack.append(up)
    stack = np.array(stack)
    sr_mean = stack.mean(0); sr_med = np.median(stack, axis=0)
    return np.clip(sr_mean, 0, 1), np.clip(sr_med, 0, 1), np.array(shifts)


def sharpness(t):
    g = gray(t); return float(np.mean(np.abs(np.gradient(g)[0])) + np.mean(np.abs(np.gradient(g)[1])))


def draft_from_tile(tile, s=1, min_pitch_fullres=2.5):
    """Cell grid on an s-times super-resolved tile; the physical floor of 2.5 px per cell at full
    resolution bounds the search, otherwise the sharper tile is fitted by an over-fine grid."""
    lab = skcolor.rgb2lab(tile); B_px, _ = otsu_binary(lab[..., 0])
    py, px = B_px.shape
    max_pitch = 16.0 * s
    fx, fy, err, B, _ = min_thread_grid_bounded(lab[..., 0], B_px, min_pitch=min_pitch_fullres * s, max_pitch=max_pitch)
    return B_px, B, (fx, fy)


def min_thread_grid_bounded(L_tile, B_px, min_pitch, max_pitch, knee=0.01, n_phase=4):
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
    for ne in range(max(4, int(px / max_pitch)), int(px / min_pitch) + 1):
        fx = px / ne
        for npk in range(max(4, int(py / max_pitch)), int(py / min_pitch) + 1):
            fy = py / npk; errs = []
            for kx in range(n_phase):
                for ky in range(n_phase):
                    ox, oy = int(round(kx * fx / n_phase)), int(round(ky * fy / n_phase))
                    t, e = draft_at(ne, npk, ox, oy); errs.append((e, ox, oy, t))
            e, ox, oy, t = min(errs, key=lambda r: r[0]); table.append((ne, npk, e))
            if best is None or e < best[2]: best = (ne, npk, e, t)
    if best is None: return 1.0, 1.0, 0.0, B_px.copy(), []
    ok = [r for r in table if r[2] <= best[2] + knee]; ok.sort(key=lambda r: r[0] * r[1]); ne, npk, err = ok[0]
    fx, fy = px / ne, py / npk; errs = []
    for kx in range(n_phase):
        for ky in range(n_phase):
            ox, oy = int(round(kx * fx / n_phase)), int(round(ky * fy / n_phase)); t, e = draft_at(ne, npk, ox, oy); errs.append((e, ox, oy, t))
    e, ox, oy, t = min(errs, key=lambda r: r[0])
    return fx, fy, e, t, table


# ----------------------------------------------------------------- B. displacement vote map
def vote_map(rect, max_period=180, n_peaks=350):
    g = gray(rect).astype(np.float32)
    H, W = g.shape; s = min(520, H - 4, W - 4)
    c = g[H // 2 - s // 2:H // 2 + s // 2, W // 2 - s // 2:W // 2 + s // 2]
    sm = ndi.gaussian_filter(c, 1.2); hp = sm - ndi.gaussian_filter(sm, 25)
    chans = [hp, -ndi.gaussian_laplace(c, 2.0), -ndi.gaussian_laplace(c, 4.0)]
    acc = np.zeros((2 * max_period + 1, 2 * max_period + 1), np.float32)
    for ch in chans:
        pk = peak_local_max(ch, min_distance=3, threshold_rel=0.25, num_peaks=n_peaks)
        if len(pk) < 4: continue
        d = pk[:, None, :] - pk[None, :, :]
        d = d.reshape(-1, 2); d = d[(np.abs(d[:, 0]) <= max_period) & (np.abs(d[:, 1]) <= max_period)]
        np.add.at(acc, (d[:, 0] + max_period, d[:, 1] + max_period), 1.0 / len(pk))
    acc = ndi.gaussian_filter(acc, 1.5)
    acc[max_period - 3:max_period + 4, max_period - 3:max_period + 4] = 0
    return acc / max(acc.max(), 1e-9)


def repeat_from_map(m, max_period=160):
    (s1, area, v1, v2, s2), _, _ = lattice_candidates(m, max_period)
    X, Y = axis_vectors(v1, v2)
    return np.array([np.hypot(*X), np.hypot(*Y)]), float(s1)


def lattice_methods(rgb):
    flat, _ = flatten_illumination(rgb); rect, _, _ = rectify(flat)
    ac = autocorr_window(rect)
    vm = vote_map(rect)
    # bring the vote map to the same lag window as the autocorrelation (centre crop or pad)
    c = ac.shape[0] // 2; k = vm.shape[0] // 2
    if k >= c: vmc = vm[k - c:k - c + ac.shape[0], k - c:k - c + ac.shape[1]]
    else:
        vmc = np.zeros_like(ac); vmc[c - k:c - k + vm.shape[0], c - k:c - k + vm.shape[1]] = vm
    fused = 0.5 * (ac / ac.max() + vmc)
    out = {}
    for name, mm in (("acf", ac), ("vote", vmc), ("fused", fused)):
        try: out[name] = repeat_from_map(mm)[0]
        except Exception: out[name] = np.array([np.nan, np.nan])
    return out, ac, vmc


def lattice_bootstrap(n=12):
    res = {}
    for p in R1["patterns"]:
        pid = p["id"]; rgb = load_rgb(RAW / p["file"]); H, W = rgb.shape[:2]
        pre = np.array(p["repeat_px_before_affine"]); rng = np.random.default_rng(5)
        ok = {"acf": 0, "vote": 0, "fused": 0}; full, _, _ = lattice_methods(rgb)
        for _ in range(n):
            hh, ww = int(0.8 * H), int(0.8 * W); y, x = rng.integers(0, H - hh), rng.integers(0, W - ww)
            est, _, _ = lattice_methods(rgb[y:y + hh, x:x + ww])
            for k in ok: ok[k] += int(np.all(np.abs(est[k] - pre) / pre < 0.03))
        res[pid] = {"full": {k: np.round(v, 2).tolist() for k, v in full.items()}, "crops_ok": ok, "n": n}
        print("lattice", pid, res[pid])
    return res


# ----------------------------------------------------------------- C. statistical symmetry test
def cell_stack(blocks, B, phase):
    L = skcolor.rgb2lab(blocks)[..., 0].astype(np.float32); npk, ne = B.shape
    out = []
    for Lb in L:
        Ls = np.roll(np.roll(Lb, -phase[1], 0), -phase[0], 1)
        out.append(cv2.resize(Ls, (ne, npk), interpolation=cv2.INTER_AREA))
    return np.array(out)


def chi2_symmetry(Lc, sigma2, ops, trim=0.1):
    """Liu-Collins-Tsin style test: trimmed normalised residual between the cell tile and its transform."""
    out = {}
    N = Lc.size; Np = int((1 - trim) * N)
    for name, o in ops.items():
        r = ((Lc - o["fn"](Lc)) ** 2 / np.maximum(sigma2 + o["fn"](sigma2), 1e-3)).ravel()
        r = np.sort(r)[:Np]; d = float(r.sum())
        thr = float(stats.chi2.ppf(0.99, Np))
        out[name] = {"d_over_N": round(d / Np, 3), "chi2_pass": bool(d < thr), "rule_pass": bool(o["accepted"]), "agreement": round(o["agreement"], 4)}
    return out


# ----------------------------------------------------------------- D. soft draft
def soft_draft(Lc):
    gm = GaussianMixture(2, random_state=0).fit(Lc.reshape(-1, 1))
    post = gm.predict_proba(Lc.reshape(-1, 1)); hi = int(np.argmax(gm.means_.ravel()))
    u = post[:, hi].reshape(Lc.shape)
    conf = np.abs(2 * u - 1)
    return u, conf


# ----------------------------------------------------------------- E. texture metrics
def gabor_bank():
    return [np.real(gabor_kernel(f, theta=t, sigma_x=2.5, sigma_y=2.5)) for f in (0.15, 0.3) for t in np.deg2rad([0, 45, 90, 135])]


def gram_distance(img1, img2, patch=16):
    bank = gabor_bank(); g1, g2 = gray(img1), gray(img2)
    F1 = np.array([ndi.convolve(g1, k, mode="wrap") for k in bank]); F2 = np.array([ndi.convolve(g2, k, mode="wrap") for k in bank])
    H, W = g1.shape; ds = []
    for y in range(0, H - patch + 1, patch):
        for x in range(0, W - patch + 1, patch):
            a = F1[:, y:y + patch, x:x + patch].reshape(len(bank), -1); b = F2[:, y:y + patch, x:x + patch].reshape(len(bank), -1)
            ds.append(np.linalg.norm(a @ a.T - b @ b.T) / patch ** 2)
    return float(np.mean(ds))


def homogeneity(blocks):
    return float(np.median(gray(blocks).std(axis=0)))


def recon_from_draft(tile, B, c_bg, c_fg):
    py, px = tile.shape[:2]
    Bq = cv2.resize(B.astype(np.uint8), (px, py), interpolation=cv2.INTER_NEAREST) > 0
    return np.where(Bq[..., None], c_fg, c_bg).astype(np.float32)


# ----------------------------------------------------------------- main
def main():
    t0 = time.time(); res = {"patterns": {}}
    for p in R1["patterns"]:
        pid, hed = p["id"], p["heddles_label"]
        d = np.load(OUT / "npz" / f"{pid}.npz"); sy = np.load(OUT / "npz" / f"sym_{pid}.npz")
        tile, B, Bs, c_bg, c_fg = d["tile"], sy["B"], sy["Bs"], d["c_bg"], d["c_fg"]
        rs = cv2.resize(d["rs_small"], (0, 0), fx=2, fy=2)
        ipx, ipy = tile.shape[1], tile.shape[0]
        blocks, tile_mean = aligned_blocks(rs, ipx, ipy)
        # A. super tile
        sr_mean, sr_med, shifts = super_tile(blocks, s=3)
        Bpx_sr, B_sr, pitch_sr = draft_from_tile(sr_mean, s=3)
        Lc_sr, phase_sr, _ = cell_lightness(sr_mean, B_sr)
        ops_sr = ops_for(B_sr, p["lattice_type"] == "centred"); Bs_sr, _, acc_sr = symmetrise(Lc_sr, B_sr, ops_sr)
        S_sr, T_sr, *_ = shafts_of(Bs_sr); S_sr0, T_sr0, *_ = shafts_of(B_sr)
        cost_sr = float(project_to_shafts(Bs_sr, hed)[1])
        # C/D on the baseline cell grid
        Lc, phase, _ = cell_lightness(tile, B)
        st = cell_stack(blocks, B, phase); sigma2 = st.var(axis=0)
        ops = ops_for(B, p["lattice_type"] == "centred")
        chi = chi2_symmetry(Lc, sigma2, ops)
        u, conf = soft_draft(Lc)
        # E. metrics
        rec_o, rec_s = recon_from_draft(tile, B, c_bg, c_fg), recon_from_draft(tile, Bs, c_bg, c_fg)
        gd = {"otsu": gram_distance(tile, rec_o), "sym": gram_distance(tile, rec_s), "mean_tile_vs_block": float(np.mean([gram_distance(tile, b) for b in blocks[:8]]))}
        hom = homogeneity(blocks)
        # F. Hu moments
        hu = {k: np.sign(cv2.HuMoments(cv2.moments(v.astype(np.uint8))).ravel()) * np.log10(np.abs(cv2.HuMoments(cv2.moments(v.astype(np.uint8))).ravel()) + 1e-30) for k, v in (("otsu", B), ("sym", Bs))}
        res["patterns"][pid] = {
            "n_blocks": int(len(blocks)), "sr_scale": 3, "subpixel_shift_rms": round(float(np.sqrt((shifts ** 2).mean())), 3),
            "sharpness": {"mean_tile": sharpness(tile_mean), "super_mean": sharpness(cv2.resize(sr_mean, (ipx, ipy), interpolation=cv2.INTER_AREA)), "super_median": sharpness(cv2.resize(sr_med, (ipx, ipy), interpolation=cv2.INTER_AREA))},
            "cells_base": [int(B.shape[1]), int(B.shape[0])], "cells_super": [int(B_sr.shape[1]), int(B_sr.shape[0])], "pitch_super_px_fullres": [round(pitch_sr[0] / 3, 3), round(pitch_sr[1] / 3, 3)],
            "shafts_super_otsu": int(S_sr0), "shafts_super_sym": int(S_sr), "treadles_super_sym": int(T_sr), "cost_super_sym_at_heddles": round(cost_sr, 4), "accepted_ops_super": acc_sr,
            "chi2_symmetry": chi, "soft_draft": {"uncertain_frac_0.4_0.6": round(float(np.mean((u > 0.4) & (u < 0.6))), 4), "mean_confidence": round(float(conf.mean()), 4), "agree_with_otsu": round(float(np.mean((u > 0.5) == B)), 4)},
            "gram_distance": {k: round(v, 4) for k, v in gd.items()}, "homogeneity_median_std": round(hom, 4),
            "hu_log": {k: np.round(v, 2).tolist() for k, v in hu.items()},
        }
        np.savez_compressed(OUT / "npz" / f"adv_{pid}.npz", sr_mean=sr_mean, sr_med=sr_med, tile_mean=tile_mean, B_sr=B_sr, Bs_sr=Bs_sr, u=u, conf=conf, Lc=Lc, vote=lattice_methods(load_rgb(RAW / p["file"]))[2], ac=lattice_methods(load_rgb(RAW / p["file"]))[1])
        print(pid, "SR cells", B_sr.shape[::-1], "pitch", np.round(pitch_sr, 2), "shafts", S_sr0, "->", S_sr, "cost", round(cost_sr, 3), "| chi2", {k: (v["chi2_pass"], v["d_over_N"]) for k, v in chi.items()}, "| uncertain", res["patterns"][pid]["soft_draft"], "| gram", gd, "hom", round(hom, 3))
    # A on synthetic cloth at 2 and 3 px per cell: does the super tile recover the cell grid?
    syn = []
    for p in R1["patterns"]:
        pid, hed = p["id"], p["heddles_label"]; d = np.load(OUT / "npz" / f"{pid}.npz"); B0 = d["B"]
        for pitch in (2, 3):
            if max(B0.shape) * pitch > 150: continue
            reps = (int(np.ceil(700 / (B0.shape[0] * pitch))) + 1, int(np.ceil(800 / (B0.shape[1] * pitch))) + 1)
            img = synth_fabric(B0, d["c_bg"], d["c_fg"], pitch, reps=reps, angle=2.0, noise=0.06, seed=1)
            try:
                r, b = digitise(img, "s", hed, verbose=False)
                blocks, _ = aligned_blocks(b["rs"], b["ipx"], b["ipy"])
                srm, _, _ = super_tile(blocks, s=3)
                _, B_sr, _ = draft_from_tile(srm, s=3)
                a0, _ = draft_agreement(B0, b["B_px"]); a1, _ = draft_agreement(B0, draft_from_tile(srm, s=3)[0])
                syn.append({"id": pid, "pitch": pitch, "cells_true": [int(B0.shape[1]), int(B0.shape[0])], "cells_base": [int(b["B"].shape[1]), int(b["B"].shape[0])], "cells_super": [int(B_sr.shape[1]), int(B_sr.shape[0])],
                            "exact_base": bool(b["B"].shape == B0.shape), "exact_super": bool(B_sr.shape == B0.shape), "agree_base": round(a0, 4), "agree_super": round(a1, 4)})
            except Exception as e:
                syn.append({"id": pid, "pitch": pitch, "error": str(e)[:60]})
            print("syn", syn[-1])
    res["synthetic_super"] = syn
    prev = OUT / "results_adv.json"
    res["lattice_bootstrap"] = json.load(open(prev))["lattice_bootstrap"] if prev.exists() and "--relattice" not in sys.argv else lattice_bootstrap()
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "results_adv.json", "w"), indent=1)
    print("done", res["runtime_s"])


if __name__ == "__main__":
    main()
