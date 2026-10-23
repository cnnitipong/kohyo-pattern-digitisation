#!/usr/bin/env python3
"""Symmetry noise test.

For each pattern and each candidate symmetry operation T, compares the cell-lightness
tile L with T(L) cell by cell. The variance of the averaged tile is estimated as the
variance across the n registered repeats divided by n, so the statistic
    z = (L - T(L))^2 / ((s2 + T(s2)) / n)
is about 1 on average if the departures from symmetry are averaging noise.
Reports the mean of z and the fraction of cells above the 1% quantile of chi-square(1).
Writes data/symmetry_noise_test.json.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import advanced as A  # noqa: E402

CHI2_1DOF_99 = 6.63


def main():
    out = {}
    for p in A.R1["patterns"]:
        pid = p["id"]
        d = np.load(A.OUT / "npz" / f"{pid}.npz")
        sy = np.load(A.OUT / "npz" / f"sym_{pid}.npz")
        tile, B = d["tile"], sy["B"]
        rs = cv2.resize(d["rs_small"], (0, 0), fx=2, fy=2)
        blocks, _ = A.aligned_blocks(rs, tile.shape[1], tile.shape[0])
        n = len(blocks)
        Lc, phase, _ = A.cell_lightness(tile, B)
        s2 = A.cell_stack(blocks, B, phase).var(axis=0, ddof=1)
        ops = A.ops_for(B, p["lattice_type"] == "centred")
        res = {}
        for name, o in ops.items():
            var_mean = (s2 + o["fn"](s2)) / n
            z = (Lc - o["fn"](Lc)) ** 2 / np.maximum(var_mean, 1e-6)
            res[name] = {"mean_ratio": round(float(z.mean()), 2),
                         "frac_cells_sig": round(float(np.mean(z > CHI2_1DOF_99)), 3),
                         "agreement": round(o["agreement"], 3)}
        out[pid] = {"n_repeats": n, "ops": res}
        print(pid, n, res)
    with open(A.OUT / "symmetry_noise_test.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
