#!/usr/bin/env python3
"""Symmetry-regularised cell draft.
The cell lightness of every pattern is averaged over the orbit of the symmetry operations the
pattern itself exhibits (centred half-repeat translation, vertical mirror, horizontal mirror,
two-fold rotation), each with its best-fit shift, and re-thresholded. Writes data/results_sym.json
and data/npz/sym_<id>.npz, and loom files data/loom/<id>/sym_*."""
import json, sys, time
from pathlib import Path
import numpy as np, cv2
from scipy import ndimage as ndi
from skimage import color as skcolor
from skimage.metrics import structural_similarity as ssim
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import *

OUT = ROOT / "data"
R1 = json.load(open(OUT / "results.json", encoding="utf-8"))
THR = 0.80   # an operation is accepted when its best-shift agreement with the draft is at least this


def cell_lightness(tile, B):
    """Area-averaged L* on the cell grid, with the phase that best matches the binary draft B."""
    L = skcolor.rgb2lab(tile)[..., 0].astype(np.float32)
    py, px = L.shape; npk, ne = B.shape
    fx, fy = px / ne, py / npk
    best = None
    for kx in range(4):
        for ky in range(4):
            ox, oy = int(round(kx * fx / 4)), int(round(ky * fy / 4))
            small = cv2.resize(np.roll(np.roll(L, -oy, 0), -ox, 1), (ne, npk), interpolation=cv2.INTER_AREA)
            t = cv2.threshold(np.clip(small / 100 * 255, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
            e = np.mean(t != B)
            if best is None or e < best[0]: best = (e, small, (ox, oy))
    return best[1], best[2], best[0]


def best_shift(A, B):
    """Cyclic shift (dy, dx) maximising agreement of A with B."""
    best = (-1, 0, 0)
    for dy in range(B.shape[0]):
        for dx in range(B.shape[1]):
            a = float(np.mean(np.roll(np.roll(A, dy, 0), dx, 1) == B))
            if a > best[0]: best = (a, dy, dx)
    return best


def ops_for(B, centred):
    """Candidate operations as functions on a cell array, with their agreement on the binary draft."""
    npk, ne = B.shape
    cand = {"mirror_x": lambda X: X[:, ::-1], "mirror_y": lambda X: X[::-1, :], "rot180": lambda X: X[::-1, ::-1]}
    if centred and ne % 2 == 0 and npk % 2 == 0:
        cand["half_shift"] = lambda X: np.roll(np.roll(X, npk // 2, 0), ne // 2, 1)
    ops = {}
    for name, f in cand.items():
        a, dy, dx = best_shift(f(B), B)
        ops[name] = {"agreement": a, "dy": dy, "dx": dx, "accepted": bool(a >= THR), "fn": (lambda X, f=f, dy=dy, dx=dx: np.roll(np.roll(f(X), dy, 0), dx, 1))}
    return ops


def symmetrise(Lc, B, ops, iters=2):
    """Average cell lightness over identity plus accepted operations (applied to the current
    estimate), threshold with Otsu, repeat so the result is closed under the group."""
    acc = [name for name, o in ops.items() if o["accepted"]]
    Ls = Lc.copy()
    for _ in range(iters):
        stack = [Ls] + [ops[n]["fn"](Ls) for n in acc]
        Ls = np.mean(stack, axis=0)
    t = cv2.threshold(np.clip(Ls / 100 * 255, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
    return t, Ls, acc


def shafts_of(B):
    thr, trd, tie = draft_from_binary(B)
    return tie.shape[0], tie.shape[1], thr, trd, tie


def fidelity_motif(tile, B, c_bg, c_fg, ref_rgb):
    """Motif-scale SSIM of the two-colour reconstruction from cell draft B against the rectified image crop."""
    py, px = tile.shape[:2]
    Bq = cv2.resize(B.astype(np.uint8), (px, py), interpolation=cv2.INTER_NEAREST) > 0
    rec = np.where(Bq[..., None], c_fg, c_bg)
    H, W = ref_rgb.shape[:2]; ny, nx = H // py, W // px
    rec = np.tile(rec, (ny, nx, 1)); ref = ref_rgb[:ny * py, :nx * px]
    g1, g2 = ndi.gaussian_filter(gray(ref), 2), ndi.gaussian_filter(gray(rec), 2)
    return float(ssim(g1, g2, data_range=1.0))


def main():
    t0 = time.time(); res = {}
    for p in R1["patterns"]:
        pid, hed = p["id"], p["heddles_label"]
        d = np.load(OUT / "npz" / f"{pid}.npz")
        tile, B, c_bg, c_fg = d["tile"], d["B"], d["c_bg"], d["c_fg"]
        rs = cv2.resize(d["rs_small"], (0, 0), fx=2, fy=2)
        Lc, phase, e0 = cell_lightness(tile, B)
        ops = ops_for(B, p["lattice_type"] == "centred")
        Bs, Ls, acc = symmetrise(Lc, B, ops)
        # keep the class convention of B (pattern = lighter)
        S0, T0, *_ = shafts_of(B); S1, T1, thr, trd, tie = shafts_of(Bs)
        proj0 = {H: round(float(project_to_shafts(B, H)[1]), 4) for H in range(2, 17)}
        proj1 = {H: round(float(project_to_shafts(Bs, H)[1]), 4) for H in range(2, 17)}
        sym0 = {n: round(o["agreement"], 4) for n, o in ops.items()}
        ops1 = ops_for(Bs, p["lattice_type"] == "centred")
        sym1 = {n: round(o["agreement"], 4) for n, o in ops1.items()}
        fid0, fid1 = fidelity_motif(tile, B, c_bg, c_fg, rs), fidelity_motif(tile, Bs, c_bg, c_fg, rs)
        fl0, fl1 = max_float(B), max_float(Bs)
        changed = float(np.mean(B != Bs))
        res[pid] = {"heddles": hed, "lattice": p["lattice_type"], "cells": [int(B.shape[1]), int(B.shape[0])], "phase_px": list(map(int, phase)),
                    "ops": {n: {"agreement": sym0[n], "accepted": o["accepted"], "shift": [int(o["dy"]), int(o["dx"])]} for n, o in ops.items()},
                    "accepted_ops": acc, "cells_changed_frac": round(changed, 4),
                    "shafts_before": int(S0), "treadles_before": int(T0), "shafts_after": int(S1), "treadles_after": int(T1),
                    "proj_before": proj0, "proj_after": proj1, "cost_at_heddles_before": proj0[hed], "cost_at_heddles_after": proj1[hed],
                    "symmetry_after": sym1, "fidelity_motif_before": round(fid0, 4), "fidelity_motif_after": round(fid1, 4),
                    "floats_before": fl0, "floats_after": fl1, "pattern_fraction_before": round(float(B.mean()), 4), "pattern_fraction_after": round(float(Bs.mean()), 4)}
        np.savez_compressed(OUT / "npz" / f"sym_{pid}.npz", B=B, Bs=Bs, Lc=Lc, Ls=Ls, tile=tile)
        dd = OUT / "loom" / pid
        pal8 = (np.array([c_bg, c_fg]) * 255).round().astype(int)
        save_indexed(Bs.astype(np.uint8), pal8, dd / "sym_repeat_draft.png")
        save_indexed(np.tile(Bs, (8, 8)).astype(np.uint8), pal8, dd / "sym_tc2_8x8_repeats.tif")
        point_paper(Bs, dd / "sym_point_paper.png")
        write_wif(dd / f"{pid}_sym.wif", thr, trd, tie, f"{p['name_en']} symmetrised", pal8)
        print(pid, "ops", {n: (sym0[n], o["accepted"]) for n, o in ops.items()}, "changed", round(changed, 3), "shafts", S0, "->", S1, "cost", proj0[hed], "->", proj1[hed], "fid", round(fid0, 3), "->", round(fid1, 3))
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "results_sym.json", "w"), indent=1)


if __name__ == "__main__":
    main()
