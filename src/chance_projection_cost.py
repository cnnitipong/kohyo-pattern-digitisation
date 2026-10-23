#!/usr/bin/env python3
"""Chance baseline for the shaft-projection cost.

Projects random drafts with the same size and float density as each measured draft
onto the weaver's heddle count (five draws each) and reports the mean and minimum
fraction of cells changed. Writes data/chance_projection_cost.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402


def main(draws=5, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for pid, _fn, _name, _thai, heddles in A.PATTERNS:
        B = np.load(A.OUT / "npz" / f"{pid}.npz")["B"]
        density = B.mean()
        costs = [float(A.project_to_shafts(rng.random(B.shape) < density, heddles)[1]) for _ in range(draws)]
        out[pid] = {"density": round(float(density), 3), "heddles": heddles,
                    "random_cost_mean": round(float(np.mean(costs)), 4), "random_cost_min": round(float(np.min(costs)), 4)}
        print(pid, out[pid])
    with open(A.OUT / "chance_projection_cost.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
