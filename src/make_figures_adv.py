#!/usr/bin/env python3
"""Panels for the literature-integrated techniques: fig24 strips, fig25/fig26 charts."""
import json
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figures import save, imshow_rgb, legend_below, GREYS, LS, MK, HATCH, HALF, FULL, IDS, D

A = json.load(open(D / "results_adv.json")); P = A["patterns"]
x = np.arange(5)


def strips():
    for pid in IDS:
        d = np.load(D / "npz" / f"adv_{pid}.npz"); tm, sr, conf = d["tile_mean"], d["sr_mean"], d["conf"]
        h, w = tm.shape[:2]
        fig, axs = plt.subplots(1, 3, figsize=(FULL, FULL / 3 * h / w * 1.1 + 0.35))
        imshow_rgb(axs[0], cv2.resize(tm, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC), "mean tile, cubic $\\times$3")
        imshow_rgb(axs[1], sr, "super-resolved tile, $\\times$3")
        im = axs[2].imshow(conf, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); axs[2].set_xticks([]); axs[2].set_yticks([]); axs[2].set_title("cell confidence")
        for ax in axs:
            for s in ax.spines.values(): s.set_visible(False)
        fig.subplots_adjust(wspace=0.05)
        save(fig, f"fig24_{pid.lower()}")


def charts():
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, (key, lab) in enumerate([("mean_tile", "re-registered mean"), ("super_mean", "super-resolved mean"), ("super_median", "super-resolved median")]):
        ax.bar(x + (k - 1) * 0.27, [1000 * P[p]["sharpness"][key] for p in IDS], 0.27, color=GREYS[[3, 0, 2][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Gradient magnitude at full resolution ($\\times 10^{-3}$)"); legend_below(ax, ncol=2, fontsize=7)
    save(fig, "fig25_a")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    syn = [s for s in A["synthetic_super"] if "exact_base" in s]
    for k, (key, lab) in enumerate([("exact_base", "mean tile"), ("exact_super", "super-resolved tile")]):
        vals = [100 * np.mean([s[key] for s in syn if s["pitch"] == pt]) for pt in (2, 3)]
        ax.bar(np.arange(2) + (k - 0.5) * 0.38, vals, 0.38, color=GREYS[[3, 0][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["2 px per cell", "3 px per cell"]); ax.set_ylabel("Exact cell grids (% of 5 patterns)"); ax.set_ylim(0, 105); legend_below(ax)
    save(fig, "fig25_b")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    L = A["lattice_bootstrap"]
    for k, (key, lab) in enumerate([("acf", "autocorrelation"), ("vote", "displacement vote"), ("fused", "fused")]):
        ax.bar(x + (k - 1) * 0.27, [100 * L[p]["crops_ok"][key] / L[p]["n"] for p in IDS], 0.27, color=GREYS[[0, 3, 2][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Crops with correct repeat (%)"); ax.set_ylim(0, 105); legend_below(ax, ncol=3, fontsize=7)
    save(fig, "fig25_c")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    keys = ["mirror_x", "mirror_y", "rot180", "half_shift"]; labs = ["vertical mirror", "horizontal mirror", "two-fold rotation", "half-repeat shift"]
    for k, (key, lab) in enumerate(zip(keys, labs)):
        vals = [P[p]["chi2_symmetry"][key]["d_over_N"] if key in P[p]["chi2_symmetry"] else np.nan for p in IDS]
        ax.bar(x + (k - 1.5) * 0.2, vals, 0.2, color=GREYS[[0, 2, 3, 4][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.axhline(1, color="0.4", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Trimmed residual $d/N'$"); ax.set_ylim(0, 1.1); legend_below(ax, ncol=2, fontsize=7)
    save(fig, "fig25_d")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, (key, lab) in enumerate([("otsu", "Otsu draft"), ("sym", "symmetrised draft"), ("mean_tile_vs_block", "single repeat")]):
        ax.bar(x + (k - 1) * 0.27, [1000 * P[p]["gram_distance"][key] for p in IDS], 0.27, color=GREYS[[3, 0, 2][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Gram distance to mean tile ($\\times 10^{-3}$)"); legend_below(ax, ncol=3, fontsize=7)
    save(fig, "fig26_a")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.bar(x - 0.2, [100 * P[p]["soft_draft"]["uncertain_frac_0.4_0.6"] for p in IDS], 0.4, color=GREYS[3], edgecolor="black", lw=0.5, hatch="//", label="uncertain cells (%)")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.bar(x + 0.2, [P[p]["soft_draft"]["mean_confidence"] for p in IDS], 0.4, color=GREYS[0], edgecolor="black", lw=0.5, label="mean confidence")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Uncertain cells (%)"); ax2.set_ylabel("Mean confidence"); ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, "fig26_b")
    d = np.load(D / "npz" / "adv_P3.npz")
    for key, name in (("ac", "fig26_c"), ("vote", "fig26_d")):
        m = d[key]; c = m.shape[0] // 2
        fig, ax = plt.subplots(figsize=(HALF, HALF)); ax.imshow(m, cmap="gray", extent=[-c, c, c, -c], vmin=0, vmax=max(m.max(), 1e-6))
        ax.set_xlabel("Lag x (px)"); ax.set_ylabel("Lag y (px)"); save(fig, name)
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.bar(x - 0.2, [P[p]["shafts_super_otsu"] for p in IDS], 0.4, color=GREYS[3], edgecolor="black", lw=0.5, hatch="//", label="super tile, Otsu")
    ax.bar(x + 0.2, [P[p]["shafts_super_sym"] for p in IDS], 0.4, color=GREYS[0], edgecolor="black", lw=0.5, label="super tile, symmetrised")
    S = json.load(open(D / "results_sym.json"))
    ax.plot(x, [S[p]["heddles"] for p in IDS], "k_", ms=16, mew=1.5, label="weaver's heddles")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Shafts required"); legend_below(ax, ncol=3, fontsize=7)
    save(fig, "fig26_e")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.bar(x - 0.2, [100 * S[p]["cost_at_heddles_after"] for p in IDS], 0.4, color=GREYS[3], edgecolor="black", lw=0.5, hatch="//", label="mean tile, symmetrised")
    ax.bar(x + 0.2, [100 * P[p]["cost_super_sym_at_heddles"] for p in IDS], 0.4, color=GREYS[0], edgecolor="black", lw=0.5, label="super tile, symmetrised")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Cells changed at weaver's heddles (%)"); legend_below(ax)
    save(fig, "fig26_f")


if __name__ == "__main__":
    strips(); charts(); print("adv panels done")
