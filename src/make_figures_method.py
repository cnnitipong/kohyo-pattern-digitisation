#!/usr/bin/env python3
"""Schematic figures that explain the method: shell rule, tile to draft, symmetry interpretation,
shaft projection. Built from the real data of P1 and P4."""
import sys
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"; P = ROOT / "figures" / "panels"
sys.path.insert(0, str(ROOT / "src"))
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
ORANGE, BLUE, GREY = "#d55e00", "#0072b2", "0.55"


def save(fig, name):
    fig.savefig(P / f"{name}.pdf", dpi=300, bbox_inches="tight"); fig.savefig(P / f"{name}.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color("0.6")


def shell_rule():
    d = np.load(D / "npz" / "P1.npz"); ac = d["ac"]; c = ac.shape[0] // 2; R = 110
    win = ac[c - R:c + R + 1, c - R:c + R + 1]
    cands = {"(a) two shortest peaks": (np.array([21, 0]), np.array([0, 24])),
             "(b) true repeat vectors": (np.array([41, 43]), np.array([-41, 43]))}
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1, 1, 0.8], "wspace": 0.35})
    vals = {}
    for ax, (name, (v1, v2)) in zip(axs[:2], cands.items()):
        ax.imshow(win, cmap="gray", vmin=0, vmax=1, extent=[-R, R, R, -R]); clean(ax)
        s1, s2 = [], []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0: continue
                p = i * v1 + j * v2
                if abs(p[0]) > R or abs(p[1]) > R: continue
                val = float(ac[c + p[1], c + p[0]])
                shell = max(abs(i), abs(j))
                (s1 if shell == 1 else s2).append(val)
                ax.plot(p[0], p[1], "o" if shell == 1 else "s", ms=5 if shell == 1 else 4, mfc="none", mec=ORANGE if shell == 1 else BLUE, mew=1.2)
        ax.set_title(name, fontsize=8); vals[name] = (min(s1), min(s2) if s2 else 0)
        ax.set_xlabel("shift in pixels", fontsize=7)
    ax = axs[2]; names = list(vals); x = np.arange(2)
    ax.bar(x - 0.18, [vals[n][0] for n in names], 0.34, color="black", label="first shell")
    ax.bar(x + 0.18, [vals[n][1] for n in names], 0.34, color="white", edgecolor="black", hatch="///", label="second shell")
    ax.set_xticks(x); ax.set_xticklabels(["(a)", "(b)"], fontsize=7); ax.set_ylim(0, 1); ax.set_title("(c) shell minima", fontsize=8)
    ax.set_ylabel("minimum autocorrelation", fontsize=7); ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    save(fig, "method_shell")
    print("shell", vals)


def tile_to_draft():
    import analyze as A
    d = np.load(D / "npz" / "P4.npz"); rs = cv2.resize(d["rs_small"], (0, 0), fx=2, fy=2); tile = d["tile"]; Bpx = d["B_px"]; B = d["B"]
    py, px = tile.shape[:2]
    # six single repeats from the rectified image
    x0 = (rs.shape[1] - 6 * px) // 2; y0 = (rs.shape[0] - py) // 2
    reps = [np.clip(rs[y0:y0 + py, x0 + k * px:x0 + (k + 1) * px], 0, 1) for k in range(6)]
    from skimage import color as skc
    L = skc.rgb2lab(tile)[..., 0]
    fx, fy = px / B.shape[1], py / B.shape[0]
    # phase from the same search as the pipeline
    best = None
    for kx in range(4):
        for ky in range(4):
            ox, oy = int(round(kx * fx / 4)), int(round(ky * fy / 4))
            small = cv2.resize(np.roll(np.roll(L, -oy, 0), -ox, 1), (B.shape[1], B.shape[0]), interpolation=cv2.INTER_AREA)
            t = cv2.threshold(np.clip(small / 100 * 255, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
            e = np.mean(t != B)
            if best is None or e < best[0]: best = (e, ox, oy)
    _, ox, oy = best
    fig = plt.figure(figsize=(6.6, 2.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.45], hspace=0.22)
    top = outer[0].subgridspec(1, 6, wspace=0.08); bot = outer[1].subgridspec(1, 5, wspace=0.12)
    for k, r in enumerate(reps):
        ax = fig.add_subplot(top[0, k]); ax.imshow(r); clean(ax)
        if k == 0: ax.set_title("(a) six single repeats of P4 from the straightened photograph", fontsize=8, loc="left")
    # zoom window for (e): cells 12 to 22 across, 2 to 7 down
    c0, c1, r0, r1 = 12, 20, 3, 6
    zx0, zx1 = ox + c0 * fx - 0.5, ox + c1 * fx - 0.5
    zy0, zy1 = oy + r0 * fy - 0.5, oy + r1 * fy - 0.5
    panels = [("(b) mean tile", np.clip(tile, 0, 1), None),
              ("(c) lightness", L, "gray"),
              ("(d) two colours", Bpx, "gray_r"),
              ("(e) cell grid on (d)", Bpx, "grid"),
              ("(f) measured draft", B, "gray_r")]
    for k, (name, img, cm) in enumerate(panels):
        ax = fig.add_subplot(bot[0, k])
        if cm == "grid":
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest", alpha=0.85)
            for i in range(B.shape[1] + 1): ax.axvline(ox + i * fx - 0.5, color=ORANGE, lw=0.7)
            for j in range(B.shape[0] + 1): ax.axhline(oy + j * fy - 0.5, color=ORANGE, lw=0.7)
            ax.set_xlim(zx0, zx1); ax.set_ylim(zy1, zy0); ax.set_aspect((zx1 - zx0) / (zy1 - zy0) * py / px, adjustable='box')
        elif cm == "gray_r" and img.shape == B.shape:
            ax.imshow(img, cmap=cm, vmin=0, vmax=1, interpolation="nearest", extent=[0, px, py, 0])
        else:
            ax.imshow(img, cmap=cm, interpolation="nearest")
            if k == 2:
                ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=ORANGE, lw=1.0))
        clean(ax); ax.set_title(name, fontsize=7)
    save(fig, "method_tile2draft"); print("phase", ox, oy)


def symmetry():
    s = np.load(D / "npz" / "sym_P4.npz"); B, Bs, Lc, Ls = s["B"], s["Bs"], s["Lc"], s["Ls"]
    sys.path.insert(0, str(ROOT / "src")); import symmetrize as S
    ops = S.ops_for(B, False)
    fig, axs = plt.subplots(2, 4, figsize=(6.6, 2.9))
    items = [("(a) measured draft", B, "gray_r"), ("(b) vertical mirror, aligned", ops["mirror_x"]["fn"](B), "gray_r"),
             ("(c) horizontal mirror, aligned", ops["mirror_y"]["fn"](B), "gray_r"), ("(d) two-fold rotation, aligned", ops["rot180"]["fn"](B), "gray_r"),
             ("(e) cell lightness", Lc, "gray"), ("(f) averaged over (a) to (d)", Ls, "gray"),
             ("(g) regularised draft", Bs, "gray_r"), ("(h) cells changed", None, None)]
    for ax, (name, img, cm) in zip(axs.ravel(), items):
        if img is None:
            diff = np.ones(B.shape + (3,)); diff[B & ~Bs] = (0.7, 0.7, 0.7); diff[~B & Bs] = (0, 0, 0)
            ax.imshow(diff, interpolation="nearest")
        else:
            ax.imshow(img, cmap=cm, interpolation="nearest")
        clean(ax); ax.set_title(name, fontsize=7)
    for ax, (name, o) in zip(axs[0, 1:], ops.items()):
        ax.set_xlabel(f"agrees with (a) on {100 * o['agreement']:.1f}% of cells", fontsize=6.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, "method_symmetry")


def projection():
    import analyze as A
    d = np.load(D / "npz" / "P4.npz"); B = d["B"]; Bp = d["Bp"]; thr = d["threading"]; thr_p = d["thr_p"]
    # P4 regularised draft and its projection
    s = np.load(D / "npz" / "sym_P4.npz"); Bs = s["Bs"]
    Bsp, cost_s = A.project_to_shafts(Bs, 4)
    cost = float(np.mean(Bp != B))
    cmap = ListedColormap(["#ffffff", "#000000"])
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 1.9))
    def draw(ax, X, classes, title, mark=None):
        ax.imshow(X, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        # colour strip of threading classes above the draft
        pal = plt.cm.tab10(np.linspace(0, 1, 10))
        for j, cl in enumerate(classes):
            ax.add_patch(Rectangle((j - 0.5, -1.6), 1, 0.9, color=pal[cl % 10], lw=0))
        if mark is not None:
            ys, xs = np.where(mark)
            ax.plot(xs, ys, "s", ms=3, mfc="none", mec=ORANGE, mew=0.9)
        ax.set_ylim(X.shape[0] - 0.5, -1.8); clean(ax); ax.set_title(title, fontsize=7)
    draw(axs[0], B, thr, f"(a) measured draft, {len(set(thr))} shafts")
    draw(axs[1], Bp, thr_p, f"(b) on 4 shafts, {100 * cost:.1f}% changed", mark=Bp != B)
    thr_s = np.unique(Bs.T, axis=0, return_inverse=True)[1].ravel()
    thr_sp = np.unique(Bsp.T, axis=0, return_inverse=True)[1].ravel()
    draw(axs[2], Bs, thr_s, f"(c) regularised draft, {len(set(thr_s))} shafts")
    draw(axs[3], Bsp, thr_sp, f"(d) on 4 shafts, {100 * cost_s:.1f}% changed", mark=Bsp != Bs)
    fig.tight_layout(w_pad=0.5)
    save(fig, "method_projection"); print("proj", cost, cost_s)


if __name__ == "__main__":
    shell_rule(); tile_to_draft(); symmetry(); projection(); print("method panels done")
