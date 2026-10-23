#!/usr/bin/env python3
"""Second-round panels: baseline rules, fold-residual curves, learned tiles, drafts, sweeps,
retrieval, learned lattice confidence, rectified photographs."""
import json
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figures import save, imshow_rgb, legend_below, fig_for, GREYS, LS, MK, HATCH, HALF, FULL, IDS, PAT, NPZ, D, P

V = json.load(open(D / "results_v2.json", encoding="utf-8"))
M = json.load(open(D / "results_ml.json", encoding="utf-8"))
x = np.arange(5)


def fig_rules():
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, (rule, lab) in enumerate([("shortest", "shortest-vector rule"), ("shell", "shell-consistency rule")]):
        ax.bar(x + (k - 0.5) * 0.38, [100 * V["baseline_rules"][p][rule]["crops_ok"] / V["baseline_rules"][p][rule]["crops_n"] for p in IDS], 0.38, color=GREYS[[3, 0][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Crops with correct repeat (%)"); ax.set_ylim(0, 105); legend_below(ax)
    save(fig, "fig14_a")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, (key, lab) in enumerate([("rule_ok", "shell rule"), ("ml_ok", "learned selector"), ("hybrid_ok", "hybrid")]):
        ax.bar(x + (k - 1) * 0.27, [100 * M["real_bootstrap"][p][key] / M["real_bootstrap"][p]["crops_n"] for p in IDS], 0.27, color=GREYS[[0, 2, 4][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(x, [M["real_bootstrap"][p]["p_mean"] for p in IDS], "k_", ms=14, mew=1.5, label="mean confidence")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Selector confidence")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Crops with correct repeat (%)"); ax.set_ylim(0, 105)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, "fig14_b")


def fig_fold():
    for axis, name in (("x", "fig15_a"), ("y", "fig15_b")):
        fig, ax = plt.subplots(figsize=(HALF, 2.4))
        for k, p in enumerate(PAT):
            c = np.array(p["fold_curve"][axis]); acf = p["fold_curve"]["acf"][0 if axis == "x" else 1]
            ax.plot(c[:, 0] / acf * 100, c[:, 1] / c[:, 1].min(), color=GREYS[min(k, 3)], ls=LS[k], marker=MK[k], ms=3, label=p["id"])
        ax.axvline(100, color="0.6", lw=0.8)
        ax.set_xlabel(f"Candidate period, % of autocorrelation estimate ({axis})"); ax.set_ylabel("Fold residual / minimum"); legend_below(ax, ncol=5)
        save(fig, name)


def fig_tiles_v2():
    for pid in ("P1", "P4"):
        t = np.load(D / "npz" / f"v2_{pid}_tiles.npz")
        for key, suffix in (("mean", "a"), ("median", "b"), ("n2n", "c")):
            img = t[key]
            fig, ax = fig_for(img, max_h=2.0); imshow_rgb(ax, img); save(fig, f"fig16_{pid.lower()}_{suffix}")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, (key, lab) in enumerate([("mean", "re-registered mean"), ("median", "re-registered median"), ("n2n", "learned denoiser")]):
        ax.bar(x + (k - 1) * 0.27, [V["learned_tiles"][p]["block_ssim_motif"][key] for p in IDS], 0.27, color=GREYS[[0, 2, 4][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Motif-scale SSIM to repeat blocks"); ax.set_ylim(0.5, 1.0); legend_below(ax, ncol=3, fontsize=7)
    save(fig, "fig16_d")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, (key, lab) in enumerate([("mean", "re-registered mean"), ("median", "re-registered median"), ("n2n", "learned denoiser")]):
        ax.bar(x + (k - 1) * 0.27, [1000 * V["learned_tiles"][p]["sharpness"][key] for p in IDS], 0.27, color=GREYS[[0, 2, 4][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Mean gradient magnitude ($\\times 10^{-3}$)"); legend_below(ax, ncol=3, fontsize=7)
    save(fig, "fig16_e")


def fig_drafts_v2():
    d = np.load(D / "npz" / "v2_P5_drafts.npz")
    for key, suffix in (("otsu", "a"), ("mrf", "b")):
        B = d[key]; fig, ax = fig_for(B, max_h=2.2); ax.imshow(B, cmap="gray_r", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([]); save(fig, f"fig17_{suffix}")
    rs, ns = V["resolution_sweep"], V["noise_sweep"]
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, pid in enumerate(IDS):
        pts = [(r["pitch"], r["agree_otsu"], r["agree_mrf"]) for r in rs if r["id"] == pid and "agree_otsu" in r]
        if pts:
            a = np.array(pts); ax.plot(a[:, 0], 100 * a[:, 1], color=GREYS[min(k, 3)], ls=LS[k], marker=MK[k], ms=4, label=pid)
    ax.set_xlabel("Synthetic cell pitch (px)"); ax.set_ylabel("Draft agreement, Otsu (%)"); ax.set_ylim(80, 101); legend_below(ax, ncol=5)
    save(fig, "fig17_c")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, pid in enumerate(IDS):
        pts = [(r["noise"], r["agree_otsu"], r["agree_mrf"]) for r in ns if r["id"] == pid and "agree_otsu" in r]
        a = np.array(pts); ax.plot(a[:, 0], 100 * a[:, 1], color=GREYS[min(k, 3)], ls=LS[k], marker=MK[k], ms=4, label=f"{pid} Otsu")
        ax.plot(a[:, 0], 100 * a[:, 2], color=GREYS[min(k, 3)], ls=LS[k], marker=MK[k], ms=4, mfc="white", alpha=0.6)
    ax.set_xlabel("Synthetic noise standard deviation"); ax.set_ylabel("Draft agreement (%)"); ax.set_ylim(55, 101); legend_below(ax, ncol=5, fontsize=7)
    ax.text(0.02, 58, "filled: Otsu, open: GMM + MRF", fontsize=7)
    save(fig, "fig17_d")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, pid in enumerate(IDS):
        pts = [(r["pitch"], r["cells_rel_err"], r["repeat_err_px"]) for r in rs if r["id"] == pid and "agree_otsu" in r]
        if pts:
            a = np.array(pts); ax.plot(a[:, 0], 100 * a[:, 1], color=GREYS[min(k, 3)], ls=LS[k], marker=MK[k], ms=4, label=pid)
    ax.set_xlabel("Synthetic cell pitch (px)"); ax.set_ylabel("Cell-count error (%)"); legend_below(ax, ncol=5)
    save(fig, "fig17_e")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, pid in enumerate(IDS):
        pts = [(r["pitch"], r["repeat_err_px"]) for r in rs if r["id"] == pid and "agree_otsu" in r]
        if pts:
            a = np.array(pts); ax.plot(a[:, 0], a[:, 1], color=GREYS[min(k, 3)], ls=LS[k], marker=MK[k], ms=4, label=pid)
    ax.set_xlabel("Synthetic cell pitch (px)"); ax.set_ylabel("Repeat error (px)"); legend_below(ax, ncol=5)
    save(fig, "fig17_f")


def fig_retrieval():
    r = np.load(D / "npz" / "v2_retrieval.npz")
    fig, ax = plt.subplots(figsize=(HALF, 2.6))
    for k, pid in enumerate(IDS):
        for src, mfc in (("raw", GREYS[min(k, 3)]), ("rect", "white")):
            m = (r["y"] == k) & (r["meta"] == src)
            ax.plot(r["pca"][m, 0], r["pca"][m, 1], ls="none", marker=MK[k], ms=4, mfc=mfc, mec=GREYS[min(k, 3)], label=pid if src == "raw" else None)
    ax.set_xlabel("PCA component 1"); ax.set_ylabel("PCA component 2"); legend_below(ax, ncol=5)
    ax.text(0.02, 0.97, "filled: raw crops, open: rectified crops", transform=ax.transAxes, fontsize=7, va="top")
    save(fig, "fig18_a")
    cm = np.array(V["retrieval"]["svm_all"]["confusion"])
    fig, ax = plt.subplots(figsize=(HALF, 2.6))
    ax.imshow(cm, cmap="gray_r"); ax.set_xticks(x); ax.set_yticks(x); ax.set_xticklabels(IDS); ax.set_yticklabels(IDS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    save(fig, "fig18_b")


def fig_rectified():
    for p in PAT:
        d = NPZ[p["id"]]; img = d["rs_small"]
        fig, ax = fig_for(img, max_h=2.0); imshow_rgb(ax, img); save(fig, f"fig19_{p['id'].lower()}")


if __name__ == "__main__":
    fig_rules(); fig_fold(); fig_tiles_v2(); fig_drafts_v2(); fig_retrieval(); fig_rectified()
    print("v2 panels done")


def fig_roundtrip_all():
    """fig20: synthetic cloth for every pattern; fig21: true cell draft beside the recovered one."""
    for pid in IDS:
        f = D / "npz" / f"rt_{pid}.npz"
        if not f.exists(): continue
        d = np.load(f)
        fig, ax = fig_for(d["syn"], max_h=2.0); imshow_rgb(ax, d["syn"]); save(fig, f"fig20_{pid.lower()}")
        t, r = d["true"], d["rec"]
        gap = np.ones((t.shape[0], max(2, t.shape[1] // 12)), bool) * np.nan
        both = np.hstack([t.astype(float), gap, r.astype(float)])
        fig, ax = fig_for(both, max_h=2.2)
        cm = matplotlib.colormaps["gray_r"].copy(); cm.set_bad("white")
        ax.imshow(np.ma.masked_invalid(both), cmap=cm, interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks([t.shape[1] / 2, t.shape[1] + gap.shape[1] + r.shape[1] / 2]); ax.set_xticklabels(["true draft", "recovered"]); ax.set_yticks([])
        diff = float(np.mean(t != r))
        ax.set_title(f"{pid}: {100 * (1 - diff):.1f} % cells agree", fontsize=8)
        save(fig, f"fig21_{pid.lower()}")


if __name__ == "__main__":
    fig_roundtrip_all()
    print("rt panels done")
