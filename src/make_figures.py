#!/usr/bin/env python3
"""Build every manuscript panel from data/results.json and data/npz/*.npz.
Photographs and tiles are in colour; all plots are greyscale with line style / marker / hatch."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
P = ROOT / "figures" / "panels"
P.mkdir(parents=True, exist_ok=True)
R = json.load(open(D / "results.json", encoding="utf-8"))
PAT = R["patterns"]
IDS = [p["id"] for p in PAT]
NPZ = {p["id"]: np.load(D / "npz" / f"{p['id']}.npz") for p in PAT}

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.titlesize": 9, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
GREYS = ["0.0", "0.35", "0.55", "0.7", "0.85"]
LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]
MK = ["o", "s", "^", "D", "v"]
HATCH = ["", "//", "..", "xx", "\\\\"]
HALF, FULL = 3.1, 6.4


def save(fig, name, extra=None):
    arts = [a for a in fig.findobj(matplotlib.text.Text)] + [a for a in fig.findobj(matplotlib.legend.Legend)]
    if extra: arts += extra
    fig.savefig(P / f"{name}.pdf", bbox_inches="tight", bbox_extra_artists=arts, pad_inches=0.02)
    fig.savefig(P / f"{name}.png", bbox_inches="tight", bbox_extra_artists=arts, pad_inches=0.02, dpi=150)
    plt.close(fig)


def legend_below(ax, ncol=2, **kw):
    return ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=ncol, **kw)


def fig_for(img, width=HALF, max_h=2.4):
    """Figure sized to the image aspect, capped in height."""
    h = width * img.shape[0] / img.shape[1]
    if h > max_h:
        width, h = max_h * img.shape[1] / img.shape[0], max_h
    return plt.subplots(figsize=(width, h))


def imshow_rgb(ax, img, title=None):
    ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if title: ax.set_title(title)


# --------------------------------------------------------------------- Fig 1: flowchart
def fig_flowchart():
    fig = plt.figure(figsize=(FULL, 4.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
    boxes = {
        "in": (3, 50, 20, 11, "Photograph of\nhand-woven cloth"),
        "ill": (27, 50, 20, 11, "Illumination\nflattening (L* trend)"),
        "rot": (51, 50, 20, 11, "Rotation search\n(profile variance)"),
        "lat": (75, 50, 22, 11, "Autocorrelation\nlattice selection"),
        "aff": (75, 31, 22, 11, "Affine rectification\n(axis lattice vectors)"),
        "per": (51, 31, 20, 11, "Lattice-anchor\nhomography"),
        "reg": (27, 31, 20, 11, "Per-repeat\nre-registration"),
        "tile": (3, 31, 20, 11, "Mean repeat tile\n(archival record)"),
        "bin": (3, 12, 20, 11, "Otsu draft and\ncell-grid search"),
        "wif": (27, 12, 20, 11, "Threading, tie-up,\ntreadling (WIF, TC2)"),
        "ver": (51, 12, 20, 11, "Verification: fidelity,\nstability, round trip"),
        "out": (75, 12, 22, 11, "Archive package:\ntile, draft, metrics"),
    }
    arts = []
    for k, (x, y, w, h, t) in boxes.items():
        fc = "0.92" if k in ("in", "out") else "white"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec="black", lw=0.9))
        arts.append(ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8))
    def arrow(a, b, rad=0.0):
        xa, ya, wa, ha, _ = boxes[a]; xb, yb, wb, hb, _ = boxes[b]
        ca, cb = np.array([xa + wa / 2, ya + ha / 2]), np.array([xb + wb / 2, yb + hb / 2])
        d = cb - ca
        if abs(d[0]) > abs(d[1]):
            s = (xa + wa + 0.6, ca[1]) if d[0] > 0 else (xa - 0.6, ca[1]); e = (xb - 0.6, cb[1]) if d[0] > 0 else (xb + wb + 0.6, cb[1])
        else:
            s = (ca[0], ya - 0.6) if d[1] < 0 else (ca[0], ya + ha + 0.6); e = (cb[0], yb + hb + 0.6) if d[1] < 0 else (cb[0], yb - 0.6)
        ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=16, lw=0.9, color="black", connectionstyle=f"arc3,rad={rad}"))
    for a, b in [("in", "ill"), ("ill", "rot"), ("rot", "lat"), ("lat", "aff"), ("aff", "per"), ("per", "reg"), ("reg", "tile"), ("tile", "bin"), ("bin", "wif"), ("wif", "ver"), ("ver", "out")]:
        arrow(a, b)
    ax.text(50, 64, "Geometry recovery", ha="center", fontsize=8, style="italic")
    ax.text(50, 45, "Repeat extraction", ha="center", fontsize=8, style="italic")
    ax.text(50, 26, "Loom representation and verification", ha="center", fontsize=8, style="italic")
    save(fig, "fig01_flowchart", arts)


# --------------------------------------------------------------------- Fig 2: photographs
def fig_photos():
    for p in PAT:
        d = NPZ[p["id"]]
        fig, ax = plt.subplots(figsize=(HALF, HALF * d["rgb_small"].shape[0] / d["rgb_small"].shape[1]))
        imshow_rgb(ax, d["rgb_small"])
        save(fig, f"fig02_{p['id'].lower()}")


# --------------------------------------------------------------------- Fig 3: preprocessing on P5
def fig_preprocess():
    p = [q for q in PAT if q["id"] == "P5"][0]; d = NPZ["P5"]
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    im = ax.imshow(d["trend"], cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("L* trend")
    save(fig, "fig03_a")
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    a, s = d["angscan_a"], d["angscan_s"]
    ax.plot(a, s / s.max(), "k-", marker="o", ms=3)
    ax.axvline(p["skew_deg"], color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("Rotation angle (deg)"); ax.set_ylabel("Normalised profile variance")
    ax.set_ylim(min(s / s.max()) - 0.05, 1.05)
    save(fig, "fig03_b")
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    imshow_rgb(ax, d["rect_small"])
    m, i = d["anc_m"] / 2, d["anc_i"] / 2
    ax.scatter(m[:, 0], m[:, 1], s=8, facecolors="none", edgecolors="white", lw=0.8)
    ax.scatter(i[:, 0], i[:, 1], s=8, marker="x", c="black", lw=0.8)
    save(fig, "fig03_c")
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    c = d["anc_i"].mean(axis=0)
    dist = np.hypot(*(d["anc_i"] - c).T)
    ax.plot(dist, d["drift"], "ko", ms=4, mfc="none")
    ax.set_xlabel("Anchor distance from image centre (px)"); ax.set_ylabel("Residual drift (px)")
    save(fig, "fig03_d")


# --------------------------------------------------------------------- Fig 4: autocorrelation lattices
def fig_autocorr():
    for pid, name in [("P1", "fig04_a"), ("P3", "fig04_b"), ("P5", "fig04_c"), ("P2", "fig04_d")]:
        p = [q for q in PAT if q["id"] == pid][0]; d = NPZ[pid]
        ac = d["ac"]; c = ac.shape[0] // 2
        fig, ax = plt.subplots(figsize=(HALF, HALF))
        ax.imshow(ac, cmap="gray", extent=[-c, c, c, -c], vmin=-0.3, vmax=1)
        for v, lab in zip(d["basis"], ["$v_1$", "$v_2$"]):
            ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2, mutation_scale=12))
            ax.text(v[0] * 1.15, v[1] * 1.15, lab, color="white", fontsize=8, ha="center", va="center")
        X, Y = d["X"], d["Y"]
        ax.plot([X[0], X[0]], [-c, c], color="white", lw=0.6, ls=":"); ax.plot([-c, c], [Y[1], Y[1]], color="white", lw=0.6, ls=":")
        ax.set_xlabel("Lag x (px)"); ax.set_ylabel("Lag y (px)")
        ax.set_xlim(-c, c); ax.set_ylim(c, -c)
        save(fig, name)


# --------------------------------------------------------------------- Fig 5: residual maps and stage bars
def fig_residual():
    d = NPZ["P1"]
    vmax = max(d["resid_a"].max(), d["resid_b"].max(), d["resid"].max())
    for arr, name in [(d["resid_a"], "fig05_a"), (d["resid_b"], "fig05_b"), (d["resid"], "fig05_c")]:
        fig, ax = plt.subplots(figsize=(HALF, 2.0))
        im = ax.imshow(arr, cmap="gray_r", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("Repeat column"); ax.set_ylabel("Repeat row")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("RMSE")
        save(fig, name)
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    x = np.arange(len(PAT)); w = 0.26
    keys = ["repeat_residual_rmse_mean_before_perspective", "repeat_residual_rmse_mean_plain_after_perspective", "repeat_residual_rmse_mean"]
    labs = ["after affine", "after homography", "after re-registration"]
    for k, (key, lab) in enumerate(zip(keys, labs)):
        ax.bar(x + (k - 1) * w, [p[key] for p in PAT], w, color=GREYS[[0, 2, 4][k]], edgecolor="black", lw=0.6, hatch=HATCH[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Mean per-repeat RMSE")
    ax.legend(frameon=False, loc="upper right", ncol=1)
    ax.set_ylim(0, 0.27)
    save(fig, "fig05_d")


# --------------------------------------------------------------------- Fig 6: repeat tiles (colour) and Fig 7: drafts
def fig_tiles():
    for p in PAT:
        d = NPZ[p["id"]]; t = d["tile"]
        fig, ax = fig_for(np.tile(t, (2, 2, 1)))
        imshow_rgb(ax, np.tile(t, (2, 2, 1)))
        ax.add_patch(Rectangle((-0.5, -0.5), t.shape[1], t.shape[0], fill=False, ec="white", lw=1.0, ls="--"))
        save(fig, f"fig06_{p['id'].lower()}")
        B = d["B"]
        fig, ax = fig_for(B, max_h=2.6)
        ax.imshow(B, cmap="gray_r", interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, B.shape[1], 1), minor=True); ax.set_yticks(np.arange(-0.5, B.shape[0], 1), minor=True)
        ax.grid(which="minor", color="0.6", lw=0.3); ax.tick_params(which="minor", length=0)
        ax.set_xticks([0, B.shape[1] - 1]); ax.set_yticks([0, B.shape[0] - 1])
        ax.set_xticklabels(["1", str(B.shape[1])]); ax.set_yticklabels(["1", str(B.shape[0])])
        ax.set_xlabel("End (cell)"); ax.set_ylabel("Pick (cell)")
        save(fig, f"fig07_{p['id'].lower()}")


# --------------------------------------------------------------------- Fig 8: weaving draft for P4
def fig_draft():
    d = NPZ["P4"]; thr, trd, tie = d["threading"], d["treadling"], d["tieup"]
    S, T = tie.shape; E, K = len(thr), len(trd)
    fig = plt.figure(figsize=(FULL, 3.0))
    gap = 1.5
    ax = fig.add_axes([0.04, 0.05, 0.94, 0.9]); ax.axis("off")
    ax.set_xlim(-1, E + gap + T + 1); ax.set_ylim(-(K + 1), S + gap + 1); ax.set_aspect("equal")
    def cells(x0, y0, M, lab=None):
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                ax.add_patch(Rectangle((x0 + c, y0 + r), 1, 1, fc="black" if M[r, c] else "white", ec="0.5", lw=0.3))
    Th = np.zeros((S, E), bool); Th[thr, np.arange(E)] = True
    Tr = np.zeros((K, T), bool); Tr[np.arange(K), trd] = True
    cells(0, gap, Th[::-1]); cells(E + gap, gap, tie.T[::-1].T if False else tie[::-1]); cells(E + gap, -K, Tr[::-1]); cells(0, -K, d["B"][::-1])
    arts = [ax.text(E / 2, S + gap + 0.3, f"Threading ({E} ends, {S} shafts)", ha="center", fontsize=8),
            ax.text(E + gap + T / 2, S + gap + 0.3, "Tie-up", ha="center", fontsize=8),
            ax.text(E + gap + T + 0.4, -K / 2, f"Treadling\n({K} picks)", va="center", fontsize=8),
            ax.text(E / 2, -K - 0.9, "Drawdown (cell draft)", ha="center", fontsize=8)]
    save(fig, "fig08_draft", arts)


# --------------------------------------------------------------------- Fig 9: fidelity
def fig_fidelity():
    x = np.arange(len(PAT)); w = 0.2
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    series = [("fidelity_mean_tile", "ssim", "mean tile, pixel"), ("fidelity_mean_tile", "ssim_motif_scale", "mean tile, motif scale"),
              ("fidelity_two_colour", "ssim", "two-colour, pixel"), ("fidelity_two_colour", "ssim_motif_scale", "two-colour, motif scale")]
    for k, (a, b, lab) in enumerate(series):
        ax.bar(x + (k - 1.5) * w, [p[a][b] for p in PAT], w, color=GREYS[[0, 2, 3, 4][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("SSIM"); ax.set_ylim(0, 1.0); legend_below(ax, fontsize=7)
    save(fig, "fig09_a")
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    ax.bar(x - 0.2, [p["fidelity_two_colour"]["mean_dE00"] for p in PAT], 0.4, color=GREYS[1], edgecolor="black", lw=0.5, label="mean")
    ax.bar(x + 0.2, [p["fidelity_two_colour"]["p95_dE00"] for p in PAT], 0.4, color=GREYS[4], edgecolor="black", lw=0.5, hatch="//", label="95th percentile")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("$\\Delta E_{00}$ (two-colour)"); legend_below(ax)
    save(fig, "fig09_b")
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    for k, p in enumerate(PAT):
        ks = sorted(int(q) for q in p["kmeans_dE"]); ax.plot(ks, [p["kmeans_dE"][str(q)]["mean_dE"] for q in ks], color=GREYS[min(k, 3)], ls=LS[k], marker=MK[k], ms=4, label=p["id"])
    ax.set_xlabel("Number of palette colours k"); ax.set_ylabel("Mean $\\Delta E_{00}$ of tile"); ax.legend(frameon=False, ncol=2)
    save(fig, "fig09_c")
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    ax.bar(x - 0.2, [p["fidelity_two_colour"]["psnr_db"] for p in PAT], 0.4, color=GREYS[1], edgecolor="black", lw=0.5, label="PSNR (dB)")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.bar(x + 0.2, [p["fidelity_two_colour"]["edge_f1"] for p in PAT], 0.4, color=GREYS[4], edgecolor="black", lw=0.5, hatch="//", label="Edge F1")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("PSNR (dB)"); ax2.set_ylabel("Edge F1"); ax2.set_ylim(0, 1)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, "fig09_d")


# --------------------------------------------------------------------- Fig 10: shaft projection and symmetry
def fig_shafts():
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for k, p in enumerate(PAT):
        c = p["shaft_projection_curve"]; hs = sorted(int(h) for h in c)
        ax.plot(hs, [100 * c[str(h)] for h in hs], color=GREYS[min(k, 3)], ls=LS[k], marker=MK[k], ms=3.5, label=f"{p['id']} (label {p['heddles_label']})")
        ax.plot([p["heddles_label"]], [100 * c[str(p["heddles_label"])]], marker=MK[k], ms=8, mfc="none", mec="black", ls="none")
    ax.set_xlabel("Shafts allowed H"); ax.set_ylabel("Cells changed (%)"); legend_below(ax, ncol=3, fontsize=7)
    ax.set_xticks(range(2, 17, 2))
    save(fig, "fig10_a")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    x = np.arange(len(PAT)); w = 0.27
    for k, (key, lab) in enumerate([("mirror_x", "vertical mirror"), ("mirror_y", "horizontal mirror"), ("rot180", "two-fold rotation")]):
        ax.bar(x + (k - 1) * w, [100 * p["symmetry"][key] for p in PAT], w, color=GREYS[[0, 2, 4][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Best-shift agreement (%)"); ax.set_ylim(70, 102); legend_below(ax, ncol=3, fontsize=7)
    save(fig, "fig10_b")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.bar(x - 0.2, [p["shafts_required"] for p in PAT], 0.4, color=GREYS[1], edgecolor="black", lw=0.5, label="distinct end classes")
    ax.bar(x + 0.2, [p["treadles_required"] for p in PAT], 0.4, color=GREYS[4], edgecolor="black", lw=0.5, hatch="//", label="distinct pick classes")
    ax.plot(x, [p["heddles_label"] for p in PAT], "k_", ms=14, mew=1.5, label="weaver's heddle count")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Count"); legend_below(ax, fontsize=7)
    save(fig, "fig10_c")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.bar(x - 0.2, [p["max_float_weft"] for p in PAT], 0.4, color=GREYS[1], edgecolor="black", lw=0.5, label="along picks")
    ax.bar(x + 0.2, [p["max_float_warp"] for p in PAT], 0.4, color=GREYS[4], edgecolor="black", lw=0.5, hatch="//", label="along ends")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Longest pattern float (cells)"); legend_below(ax)
    save(fig, "fig10_d")


# --------------------------------------------------------------------- Fig 11: bootstrap stability
def fig_bootstrap():
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    data = []
    for p in PAT:
        bs = [b for b in R["bootstrap"][p["id"]] if "error" not in b]
        ref = np.array(p["repeat_px_after_perspective"])
        dev = [100 * np.abs(np.array(b["repeat_px"]) - ref) / ref for b in bs]
        data.append(np.array(dev).max(axis=1))
    bp = ax.boxplot(data, tick_labels=IDS, widths=0.5, patch_artist=True, showfliers=True, flierprops=dict(marker="o", ms=3, mfc="none"))
    for b in bp["boxes"]: b.set(facecolor="0.85", edgecolor="black")
    for m in bp["medians"]: m.set(color="black")
    ax.set_yscale("symlog", linthresh=5); ax.set_ylabel("Repeat deviation from full image (%)")
    ax.axhline(3, color="0.5", ls="--", lw=0.8)
    save(fig, "fig11_a")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    frac = []
    for p in PAT:
        bs = [b for b in R["bootstrap"][p["id"]] if "error" not in b]
        ref = np.array(p["repeat_px_after_perspective"])
        ok = [np.all(np.abs(np.array(b["repeat_px"]) - ref) / ref < 0.03) for b in bs]
        frac.append(100 * np.mean(ok))
    ax.bar(np.arange(len(PAT)), frac, 0.55, color=GREYS[2], edgecolor="black", lw=0.5)
    ax.set_xticks(np.arange(len(PAT))); ax.set_xticklabels(IDS); ax.set_ylabel("Crops within 3 % of full-image repeat (%)"); ax.set_ylim(0, 105)
    for i, f in enumerate(frac): ax.text(i, f + 2, f"{f:.0f}", ha="center", fontsize=7)
    save(fig, "fig11_b")
    return frac


# --------------------------------------------------------------------- Fig 12: round trip
def fig_roundtrip():
    import sys; sys.path.insert(0, str(ROOT / "src"))
    from analyze import synth_fabric
    d = NPZ["P2"]; B = d["B"]
    syn = synth_fabric(B, d["c_bg"], d["c_fg"], 5, reps=(5, 6), angle=2.5, noise=0.04, seed=1)
    fig, ax = fig_for(syn); imshow_rgb(ax, syn); save(fig, "fig12_a")
    rt = R["roundtrip"]
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    t = np.array([x["true_repeat_px"] for x in rt]).ravel(); r = np.array([x["recovered_repeat_px"] for x in rt]).ravel()
    ax.plot([50, 160], [50, 160], color="0.6", lw=0.8); ax.plot(t, r, "ko", ms=4, mfc="none")
    ax.set_xlabel("True repeat (px)"); ax.set_ylabel("Recovered repeat (px)"); ax.set_aspect("equal")
    ax.text(55, 150, f"max |error| = {np.abs(r - t).max():.2f} px", fontsize=8)
    save(fig, "fig12_b")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    x = np.arange(len(PAT)); w = 0.26
    cases = sorted(set((abs(x_["angle_deg"]), x_["noise"]) for x_ in rt), key=lambda c: c[1])
    for k, (ang, nz) in enumerate(cases):
        vals = [[100 * x_["draft_agreement"] for x_ in rt if x_["id"] == p["id"] and x_["noise"] == nz][0] for p in PAT]
        ax.bar(x + (k - 1) * w, vals, w, color=GREYS[[0, 2, 4][k]], edgecolor="black", lw=0.5, hatch=HATCH[k], label=f"tilt {ang:.1f} deg, noise {nz:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(IDS); ax.set_ylabel("Draft agreement (%)"); ax.set_ylim(80, 101); legend_below(ax, ncol=1, fontsize=7)
    save(fig, "fig12_c")
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    a = np.array([x_["angle_deg"] for x_ in rt]); rs_ = np.array([x_["recovered_skew_deg"] for x_ in rt])
    ax.plot([-5, 5], [5, -5], color="0.6", lw=0.8); ax.plot(a, rs_, "ks", ms=4, mfc="none")
    ax.set_xlabel("Applied tilt (deg)"); ax.set_ylabel("Recovered correction (deg)"); ax.set_aspect("equal")
    save(fig, "fig12_d")


# --------------------------------------------------------------------- Fig 13: reconstruction crops (P2) + graphical abstract
def fig_reconstruction():
    d = NPZ["P2"]
    ref, rm, rq = d["ref_crop"], d["rec_mean_crop"], d["rec_q_crop"]
    for arr, name in [(ref, "fig13_a"), (rm, "fig13_b"), (rq, "fig13_c")]:
        fig, ax = fig_for(arr); imshow_rgb(ax, arr); save(fig, name)
    diff = np.abs(ref - rq).mean(axis=2)
    fig, ax = fig_for(diff)
    im = ax.imshow(diff, cmap="gray_r", vmin=0, vmax=0.5); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("|difference|")
    save(fig, "fig13_d")


def graphical_abstract():
    fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    for k, pid in enumerate(["P1", "P2", "P5"]):
        d = NPZ[pid]
        a1 = fig.add_axes([0.03 + 0.11 * k, 0.56, 0.1, 0.36]); a1.imshow(d["rgb_small"]); a1.axis("off")
        a2 = fig.add_axes([0.03 + 0.11 * k, 0.1, 0.1, 0.36]); a2.imshow(d["B"], cmap="gray_r", interpolation="nearest"); a2.axis("off")
    ax.text(21, 38.2, "Photographs of Koh Yo cloth", fontsize=12, ha="center")
    ax.text(21, 1.2, "Cell drafts for the loom", fontsize=12, ha="center")
    steps = ["Illumination and\nrotation", "Lattice selection\nand affine", "Lattice-anchor\nhomography", "Re-registered\nmean tile", "Draft, WIF\nand TC2 files"]
    for k, s in enumerate(steps):
        x = 40 + 12 * k
        ax.add_patch(FancyBboxPatch((x - 5, 24), 10, 9, boxstyle="round,pad=0.3", fc="0.93", ec="black", lw=1))
        ax.text(x, 28.5, s, ha="center", va="center", fontsize=10)
        if k < 4: ax.add_patch(FancyArrowPatch((x + 5.4, 28.5), (x + 6.6, 28.5), arrowstyle="-|>", mutation_scale=18, lw=1.2, color="black"))
    ax.text(64, 17, "Verification of the digital record", fontsize=12, ha="center", style="italic")
    checks = ["Fidelity: SSIM, PSNR, $\\Delta E_{00}$", "Stability: bootstrap crops", "Round trip: synthetic cloth",
              "Weavability: shafts, floats", "Symmetry: wallpaper tests"]
    for k, c in enumerate(checks):
        ax.text(40 + 12 * (k % 5), 11 if k < 5 else 6, c, ha="center", fontsize=10)
    fig.savefig(ROOT / "figures" / "graphical_abstract.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    fig_flowchart(); fig_photos(); fig_preprocess(); fig_autocorr(); fig_residual(); fig_tiles(); fig_draft()
    fig_fidelity(); fig_shafts(); frac = fig_bootstrap(); fig_roundtrip(); fig_reconstruction(); graphical_abstract()
    json.dump({"bootstrap_within3pct": dict(zip(IDS, frac))}, open(D / "figure_derived.json", "w"), indent=1)
    print("panels:", len(list(P.glob("*.pdf"))))
