#!/usr/bin/env python3
"""Panels for the symmetry-regularised draft: fig22_<id> strips and fig23 bars/curves."""
import json
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figures import save, imshow_rgb, legend_below, GREYS, LS, MK, HATCH, HALF, FULL, IDS, D

S = json.load(open(D / "results_sym.json"))
x = np.arange(5)


def strips():
    for pid in IDS:
        d = np.load(D / "npz" / f"sym_{pid}.npz"); B, Bs, tile = d["B"], d["Bs"], d["tile"]
        h = B.shape[0]; w = B.shape[1]
        tile_c = cv2.resize(tile, (w * 8, h * 8), interpolation=cv2.INTER_LINEAR)
        fig, axs = plt.subplots(1, 4, figsize=(FULL, FULL / 4 * h / w * 1.15 + 0.35))
        imshow_rgb(axs[0], tile_c, "mean tile")
        for ax, arr, t in ((axs[1], B, "Otsu cell draft"), (axs[2], Bs, "symmetrised draft")):
            ax.imshow(arr, cmap="gray_r", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t)
        diff = np.zeros((h, w, 3)); diff[...] = 1.0; diff[B & ~Bs] = (0.75, 0.75, 0.75); diff[~B & Bs] = (0.0, 0.0, 0.0)
        axs[3].imshow(diff, interpolation="nearest"); axs[3].set_xticks([]); axs[3].set_yticks([]); axs[3].set_title(f"changed cells, {100 * S[pid]['cells_changed_frac']:.1f} %")
        for ax in axs:
            for s in ax.spines.values(): s.set_visible(False)
        fig.subplots_adjust(wspace=0.05)
        save(fig, f"fig22_{pid.lower()}")


def bars():
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.bar(x - 0.2, [S[p]["shafts_before"] for p in IDS], 0.4, color=GREYS[3], edgecolor="black", lw=0.5, hatch="//", label="Otsu draft")
    ax.bar(x + 0.2, [S[p]["shafts_after"] for p in IDS], 0.4, color=GREYS[0], edgecolor="black", lw=0.5, label="symmetrised draft")
    ax.plot(x, [S[p]["heddles"] for p in IDS], "k_", ms=16, mew=1.5, label="weaver's heddles")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Shafts required"); legend_below(ax, ncol=3, fontsize=7)
    save(fig, "fig23_a")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.bar(x - 0.2, [100 * S[p]["cost_at_heddles_before"] for p in IDS], 0.4, color=GREYS[3], edgecolor="black", lw=0.5, hatch="//", label="Otsu draft")
    ax.bar(x + 0.2, [100 * S[p]["cost_at_heddles_after"] for p in IDS], 0.4, color=GREYS[0], edgecolor="black", lw=0.5, label="symmetrised draft")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Cells changed at weaver's heddles (%)"); legend_below(ax)
    save(fig, "fig23_b")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    keys = ["mirror_x", "mirror_y", "rot180", "half_shift"]; labs = ["vertical mirror", "horizontal mirror", "two-fold rotation", "half-repeat shift"]
    for k, (key, lab) in enumerate(zip(keys, labs)):
        vals = [100 * S[p]["ops"][key]["agreement"] if key in S[p]["ops"] else 0 for p in IDS]
        ax.bar(x + (k - 1.5) * 0.2, vals, 0.2, color=GREYS[[0, 2, 3, 4][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.axhline(80, color="0.5", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Best-shift agreement (%)"); ax.set_ylim(60, 102); legend_below(ax, ncol=2, fontsize=7)
    save(fig, "fig23_c")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, p in enumerate(IDS):
        c = S[p]["proj_after"]; hs = sorted(int(h) for h in c)
        ax.plot(hs, [100 * c[str(h)] for h in hs], color=GREYS[min(k, 3)], ls=LS[k], marker=MK[k], ms=3.5, label=f"{p} (label {S[p]['heddles']})")
        ax.plot([S[p]["heddles"]], [100 * c[str(S[p]["heddles"])]], marker=MK[k], ms=8, mfc="none", mec="black", ls="none")
    ax.set_xlabel("Shafts allowed H"); ax.set_ylabel("Cells changed (%)"); ax.set_xticks(range(2, 17, 2)); legend_below(ax, ncol=3, fontsize=7)
    save(fig, "fig23_d")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.bar(x - 0.2, [S[p]["fidelity_motif_before"] for p in IDS], 0.4, color=GREYS[3], edgecolor="black", lw=0.5, hatch="//", label="Otsu draft")
    ax.bar(x + 0.2, [S[p]["fidelity_motif_after"] for p in IDS], 0.4, color=GREYS[0], edgecolor="black", lw=0.5, label="symmetrised draft")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Motif-scale SSIM to photograph"); ax.set_ylim(0, 1); legend_below(ax)
    save(fig, "fig23_e")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.bar(x - 0.2, [S[p]["shafts_before"] * S[p]["treadles_before"] for p in IDS], 0.4, color=GREYS[3], edgecolor="black", lw=0.5, hatch="//", label="Otsu draft")
    ax.bar(x + 0.2, [S[p]["shafts_after"] * S[p]["treadles_after"] for p in IDS], 0.4, color=GREYS[0], edgecolor="black", lw=0.5, label="symmetrised draft")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Tie-up size (shafts $\\times$ treadles)"); ax.set_yscale("log"); legend_below(ax)
    save(fig, "fig23_f")


if __name__ == "__main__":
    strips(); bars(); print("sym panels done")
