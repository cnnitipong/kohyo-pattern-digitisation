#!/usr/bin/env python3
"""Uniform image figures for the Studies in Conservation version.
Every photograph, tile and draft is shown in an equal-size cell: photographs are centre-cropped to a
common aspect, periodic tiles and drafts are repeated to fill a box of the same physical size
(in rectified-image pixels) so that all five patterns are seen at one scale, with one repeat outlined."""
import json
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"; P = ROOT / "figures" / "panels"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
IDS = ["P1", "P2", "P3", "P4", "P5"]
NAMES = {"P1": "P1 Ratchawat", "P2": "P2 Dok Phakakrong", "P3": "P3 Dok Rotsukhon", "P4": "P4 Kamphaeng", "P5": "P5 Luk Kaeo"}
RAW = {"P1": "P1_ratchawat_4h.png", "P2": "P2_phakakrong_8h.png", "P3": "P3_rotsukhon_lukkaeo_8h.png", "P4": "P4_kamphaeng_4h.png", "P5": "P5_lukkaeo_4h.png"}
W_FULL = 6.6  # inches, 168 mm page width


def crop_aspect(img, aspect):
    """Centre crop to width/height = aspect."""
    h, w = img.shape[:2]
    if w / h > aspect: nw = int(h * aspect); x0 = (w - nw) // 2; return img[:, x0:x0 + nw]
    nh = int(w / aspect); y0 = (h - nh) // 2; return img[y0:y0 + nh]


def periodic_fill(tile, H, W):
    ry, rx = int(np.ceil(H / tile.shape[0])) + 1, int(np.ceil(W / tile.shape[1])) + 1
    reps = (ry, rx) + (1,) * (tile.ndim - 2)
    return np.tile(tile, reps)[:H, :W]


def draft_px(B, tile_shape):
    return cv2.resize(B.astype(np.uint8), (tile_shape[1], tile_shape[0]), interpolation=cv2.INTER_NEAREST).astype(float)


def grid(nrows, ncols, cell_aspect=1.0, top=0.22, left=0.0, gap=0.04, width=W_FULL, bottom=0.02):
    cw = (width - left - gap * (ncols - 1)) / ncols; ch = cw / cell_aspect
    height = top + nrows * ch + gap * (nrows - 1) + bottom
    fig = plt.figure(figsize=(width, height)); axs = []
    for r in range(nrows):
        row = []
        for c in range(ncols):
            x = (left + c * (cw + gap)) / width; y = 1 - (top + r * (ch + gap) + ch) / height
            ax = fig.add_axes([x, y, cw / width, ch / height]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color("0.6")
            row.append(ax)
        axs.append(row)
    return fig, axs


def label_rows(fig, axs, labels):
    for row, lab in zip(axs, labels):
        row[0].set_ylabel(lab, fontsize=8)


def col_titles(axs):
    for c, pid in enumerate(IDS): axs[0][c].set_title(NAMES[pid], fontsize=8, pad=3)


def outline(ax, w, h, color="white"):
    ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, ec=color, lw=1.0, ls="--"))


def save(fig, name):
    fig.savefig(P / f"{name}.pdf", dpi=300); fig.savefig(P / f"{name}.png", dpi=110); plt.close(fig)


def photos():
    fig, axs = grid(1, 5, cell_aspect=0.75)
    for c, pid in enumerate(IDS):
        im = np.asarray(Image.open(D / "raw" / RAW[pid]).convert("RGB"))
        axs[0][c].imshow(cv2.resize(crop_aspect(im, 0.75), (450, 600), interpolation=cv2.INTER_AREA))
    col_titles(axs); save(fig, "sic_photos")


def rectified():
    fig, axs = grid(1, 5, cell_aspect=0.75)
    for c, pid in enumerate(IDS):
        im = np.load(D / "npz" / f"{pid}.npz")["rs_small"]
        axs[0][c].imshow(cv2.resize(crop_aspect(im, 0.75), (300, 400), interpolation=cv2.INTER_AREA))
    col_titles(axs); save(fig, "sic_rect")


BOX = 300  # physical box in rectified-image pixels


def tiles_drafts():
    fig, axs = grid(2, 5, left=0.18, bottom=0.2)
    for c, pid in enumerate(IDS):
        d = np.load(D / "npz" / f"{pid}.npz"); t, B = d["tile"], d["B"]
        axs[0][c].imshow(np.clip(periodic_fill(t, BOX, BOX), 0, 1)); outline(axs[0][c], t.shape[1], t.shape[0])
        axs[1][c].imshow(periodic_fill(draft_px(B, t.shape), BOX, BOX), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        outline(axs[1][c], t.shape[1], t.shape[0], color="tab:orange")
        axs[1][c].set_xlabel(f"{B.shape[1]} $\\times$ {B.shape[0]} cells", fontsize=8, labelpad=2)
    col_titles(axs); label_rows(fig, axs, ["mean repeat tile", "measured draft"])
    save(fig, "sic_tiles")


def symmetry():
    fig, axs = grid(4, 5, left=0.18, bottom=0.32)
    S = json.load(open(D / "results_sym.json"))
    for c, pid in enumerate(IDS):
        d = np.load(D / "npz" / f"sym_{pid}.npz"); t, B, Bs = d["tile"], d["B"], d["Bs"]
        axs[0][c].imshow(np.clip(periodic_fill(t, BOX, BOX), 0, 1))
        for r, A in ((1, B), (2, Bs)):
            axs[r][c].imshow(periodic_fill(draft_px(A, t.shape), BOX, BOX), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        diff = np.ones(B.shape + (3,)); diff[B & ~Bs] = (0.7, 0.7, 0.7); diff[~B & Bs] = (0, 0, 0)
        dp = cv2.resize(diff, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST)
        axs[3][c].imshow(periodic_fill(dp, BOX, BOX), interpolation="nearest")
        for r in range(4): outline(axs[r][c], t.shape[1], t.shape[0], color="white" if r == 0 else "tab:orange")
        axs[3][c].set_xlabel(f"{100 * S[pid]['cells_changed_frac']:.1f}% of cells changed\n{S[pid]['shafts_before']} to {S[pid]['shafts_after']} shafts", fontsize=7, labelpad=2)
    col_titles(axs); label_rows(fig, axs, ["mean tile", "measured draft", "regularised draft", "changed cells"])
    save(fig, "sic_sym")


def super_res():
    box = 150
    fig, axs = grid(3, 5, left=0.18)
    for c, pid in enumerate(IDS):
        d = np.load(D / "npz" / f"adv_{pid}.npz"); tm, sr, conf = d["tile_mean"], d["sr_mean"], d["conf"]
        up = cv2.resize(tm, (tm.shape[1] * 3, tm.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
        axs[0][c].imshow(np.clip(periodic_fill(up, 3 * box, 3 * box), 0, 1))
        axs[1][c].imshow(np.clip(periodic_fill(sr, 3 * box, 3 * box), 0, 1))
        cp = cv2.resize(conf.astype(np.float32), (tm.shape[1], tm.shape[0]), interpolation=cv2.INTER_NEAREST)
        axs[2][c].imshow(periodic_fill(cp, box, box), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    col_titles(axs); label_rows(fig, axs, ["mean tile, cubic $\\times$3", "super-resolved $\\times$3", "cell confidence"])
    save(fig, "sic_sr")


def round_trip():
    fig, axs = grid(3, 5, left=0.18, bottom=0.2)
    R = json.load(open(D / "results.json"))
    for c, pid in enumerate(IDS):
        d = np.load(D / "npz" / f"rt_{pid}.npz"); syn, T, Rr, tl = d["syn"], d["true"], d["rec"], d["tile"]
        axs[0][c].imshow(np.clip(crop_aspect(syn, 1.0), 0, 1))
        for r, A in ((1, T), (2, Rr)):
            axs[r][c].imshow(periodic_fill(draft_px(A, tl.shape), int(1.6 * max(tl.shape[:2])), int(1.6 * max(tl.shape[:2]))), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            outline(axs[r][c], tl.shape[1], tl.shape[0], color="tab:orange")
        agree = 100 * float(np.mean(T == Rr))
        axs[2][c].set_xlabel(f"{agree:.1f}% of cells agree", fontsize=8, labelpad=2)
    col_titles(axs); label_rows(fig, axs, ["synthetic cloth", "true draft", "recovered draft"])
    save(fig, "sic_rt")


def learned_tiles():
    box = 170
    fig, axs = grid(2, 3, left=0.18, width=4.6)
    for r, pid in enumerate(["P1", "P4"]):
        t = np.load(D / "npz" / f"v2_{pid}_tiles.npz")
        for c, k in enumerate(["mean", "median", "n2n"]):
            axs[r][c].imshow(np.clip(periodic_fill(t[k], box, box), 0, 1))
    for c, lab in enumerate(["re-registered mean", "re-registered median", "Noise2Noise denoiser"]): axs[0][c].set_title(lab, fontsize=8, pad=3)
    label_rows(fig, axs, [NAMES["P1"], NAMES["P4"]]); save(fig, "sic_learned")


def recon():
    d = np.load(D / "npz" / "P2.npz")
    ref, rm, rq = d["ref_crop"], d["rec_mean_crop"], d["rec_q_crop"]
    s = min(ref.shape[:2]); sl = (slice(0, s), slice(0, s))
    diff = 1 - np.clip(np.abs(ref - rq).mean(axis=2) / 0.5, 0, 1)
    fig, axs = grid(1, 4)
    for c, (img, lab) in enumerate([(ref, "rectified photograph"), (rm, "mean tile"), (rq, "two-colour draft"), (np.dstack([diff] * 3), "difference")]):
        axs[0][c].imshow(np.clip(img[sl], 0, 1)); axs[0][c].set_title(lab, fontsize=8, pad=3)
    save(fig, "sic_recon")


def synth_square():
    d = np.load(D / "npz" / "rt_P2.npz")
    fig, axs = grid(1, 1, width=3.1, top=0.02)
    axs[0][0].imshow(np.clip(crop_aspect(d["syn"], 1.0), 0, 1)); save(fig, "sic_syn_p2")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1: [globals()[f]() for f in sys.argv[1:]]
    else: photos(); rectified(); tiles_drafts(); symmetry(); super_res(); round_trip(); learned_tiles(); recon(); synth_square()
    print("sic panels done")
