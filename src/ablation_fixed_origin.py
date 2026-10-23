#!/usr/bin/env python3
"""Round trips without the cell-grid origin search.

Repeats the fifteen synthetic round trips of analyze.py with a single grid origin
(n_phase=1) instead of four per axis, to measure what the origin search contributes.
Writes data/ablation_fixed_origin.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

ROUND_TRIPS = [(5, 2.5, 0.04), (7, -4.0, 0.08), (4, 1.0, 0.12)]  # pitch (px), tilt (deg), noise


def best_cyclic_agreement(rec, true):
    best = 0.0
    for dy in range(true.shape[0]):
        for dx in range(true.shape[1]):
            best = max(best, float(np.mean(np.roll(np.roll(rec, dy, 0), dx, 1) == true)))
    return best


def main():
    full_search = A.min_thread_grid
    A.min_thread_grid = lambda L, B, **kw: full_search(L, B, n_phase=1)
    out = []
    for pid, _fn, _name, _thai, heddles in A.PATTERNS:
        d = np.load(A.OUT / "npz" / f"{pid}.npz")
        B, c_bg, c_fg = d["B"], d["c_bg"], d["c_fg"]
        for pitch, angle, noise in ROUND_TRIPS:
            pitch = int(min(pitch, 150 // max(B.shape)))
            reps = (int(np.ceil(760 / (B.shape[0] * pitch))) + 1, int(np.ceil(860 / (B.shape[1] * pitch))) + 1)
            syn = A.synth_fabric(B, c_bg, c_fg, pitch, reps=reps, angle=angle, noise=noise, seed=1)
            _r, b = A.digitise(syn, pid + "syn", heddles, verbose=False)
            exact = B.shape == b["B"].shape
            agree = best_cyclic_agreement(b["B"], B) if exact else 0.0
            row = {"id": pid, "pitch": pitch, "exact_grid": bool(exact),
                   "recovered_cells": list(b["B"].shape), "true_cells": list(B.shape), "agreement": round(agree, 4)}
            out.append(row)
            print(row, flush=True)
    with open(A.OUT / "ablation_fixed_origin.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
