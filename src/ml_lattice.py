#!/usr/bin/env python3
"""Learned lattice selection: a gradient-boosted classifier scores candidate lattices from
autocorrelation features. Trained on synthetic cloth with known repeat (rendered from the five
extracted drafts), evaluated leave-one-pattern-out on synthetic crops and on the real bootstrap
crops against the full-image reference. Writes data/results_ml.json."""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import *

OUT = ROOT / "data"
R1 = json.load(open(OUT / "results.json", encoding="utf-8"))
REF = {p["id"]: np.array(p["repeat_px_after_perspective"]) for p in R1["patterns"]}


def candidate_table(ac, max_period=160, n_top=24, win=170):
    """All candidate bases with features (same enumeration as lattice_candidates)."""
    pk = local_maxima(ac, 4, 0.2)
    pk = pk[(np.abs(pk[:, 0]) <= max_period) & (np.abs(pk[:, 1]) <= max_period)]
    pk = pk[(pk[:, 0] != 0) | (pk[:, 1] != 0)]
    pk = pk[(pk[:, 1] > 0) | ((pk[:, 1] == 0) & (pk[:, 0] > 0))]
    pk = pk[np.argsort(-pk[:, 2])][:n_top]
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    rows, vecs = [], []
    for i in range(len(pk)):
        for j in range(i + 1, len(pk)):
            v1, v2 = pk[i, :2], pk[j, :2]
            area = abs(v1[0] * v2[1] - v1[1] * v2[0])
            if area < 0.3 * np.hypot(*v1) * np.hypot(*v2): continue
            vals1, vals2 = [], []
            for n1 in range(-2, 3):
                for n2 in range(-2, 3):
                    if n1 == 0 and n2 == 0: continue
                    x, y = n1 * v1 + n2 * v2
                    if abs(x) < win and abs(y) < win:
                        yy, xx = int(round(cy + y)), int(round(cx + x))
                        if yy < 1 or xx < 1 or yy >= ac.shape[0] - 1 or xx >= ac.shape[1] - 1: continue
                        v = ac[yy - 1:yy + 2, xx - 1:xx + 2].max()
                        (vals1 if max(abs(n1), abs(n2)) == 1 else vals2).append(v)
            if len(vals1) < 8: continue
            X, Y = axis_vectors(v1, v2)
            s1, s2 = np.min(vals1), (np.min(vals2) if vals2 else np.nan)
            rows.append([s1, np.mean(vals1), s2 if vals2 else -1, np.mean(vals2) if vals2 else -1, len(vals2), area / win ** 2,
                         np.hypot(*v1) / win, np.hypot(*v2) / win, pk[i, 2], pk[j, 2], np.hypot(*X) / win, np.hypot(*Y) / win,
                         abs(v1[0] * v2[0] + v1[1] * v2[1]) / (np.hypot(*v1) * np.hypot(*v2))])
            vecs.append((v1, v2, X, Y))
    rows = np.array(rows)
    # relative features: rank of s1 among candidates and ratio to best
    if len(rows):
        rows = np.hstack([rows, (rows[:, :1] / rows[:, 0].max()), (np.argsort(np.argsort(-rows[:, 0]))[:, None] / len(rows))])
    return rows, vecs


def acf_of(rgb):
    flat, _ = flatten_illumination(rgb); rect, _, _ = rectify(flat)
    return autocorr_window(rect)


def label_rows(vecs, true_xy, tol=0.03):
    return np.array([np.all(np.abs(np.array([np.hypot(*X), np.hypot(*Y)]) - true_xy) / true_xy < tol) for (_, _, X, Y) in vecs], int)


def synth_dataset(n_per=6, seed=0):
    rng = np.random.default_rng(seed)
    data = []
    for pid, fn, _, _, h in PATTERNS:
        d = np.load(OUT / "npz" / f"{pid}.npz"); B = d["B"]
        for k in range(n_per):
            pitch = int(min(rng.integers(3, 6), 150 // max(B.shape)))
            reps = (int(np.ceil(700 / (B.shape[0] * pitch))) + 1, int(np.ceil(800 / (B.shape[1] * pitch))) + 1)
            syn = synth_fabric(B, d["c_bg"], d["c_fg"], pitch, reps=reps, angle=float(rng.uniform(-4, 4)), noise=float(rng.uniform(0.03, 0.15)), seed=k)
            H, W = syn.shape[:2]; f = rng.uniform(0.5, 1.0); hh, ww = int(H * f), int(W * f)
            y, x = rng.integers(0, H - hh + 1), rng.integers(0, W - ww + 1)
            crop = syn[y:y + hh, x:x + ww]
            true = np.array([B.shape[1] * pitch, B.shape[0] * pitch], float)
            try:
                rows, vecs = candidate_table(acf_of(crop))
            except Exception:
                continue
            if len(rows) == 0: continue
            data.append((pid, rows, label_rows(vecs, true), vecs, true))
    return data


def choose(clf, rows, vecs):
    p = clf.predict_proba(rows)[:, 1]
    k = int(np.argmax(p))
    return vecs[k], float(p[k])


def rule_choice(rows, vecs, second_ratio=0.6):
    """Exactly the shell-consistency rule of analyze.lattice_candidates on the candidate table."""
    s1, s2, area = rows[:, 0], rows[:, 2], rows[:, 5]
    top = s1.max()
    valid = [i for i in range(len(rows)) if s1[i] >= 0.8 * top and (s2[i] if rows[i, 4] > 0 else s1[i]) >= second_ratio * s1[i]]
    i = min(valid, key=lambda i: area[i]) if valid else int(np.argmax(s1))
    return vecs[i]


def hybrid_choice(clf, rows, vecs, pmin=0.5):
    (v, p) = choose(clf, rows, vecs)
    return (v if p >= pmin else rule_choice(rows, vecs)), p


def main():
    t0 = time.time()
    data = synth_dataset()
    print("synthetic crops:", len(data), "positives:", sum(int(l.sum() > 0) for _, _, l, _, _ in data))
    res = {"n_synthetic_crops": len(data)}
    # leave-one-pattern-out on synthetic
    lopo = {}
    for pid, *_ in PATTERNS:
        tr = [x for x in data if x[0] != pid]; te = [x for x in data if x[0] == pid]
        X = np.vstack([r for _, r, _, _, _ in tr]); y = np.concatenate([l for _, _, l, _, _ in tr])
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, random_state=0).fit(X, y)
        ok_ml = ok_rule = ok_hy = 0
        for _, rows, lab, vecs, true in te:
            (_, _, X_, Y_), _ = choose(clf, rows, vecs)
            ok_ml += int(np.all(np.abs(np.array([np.hypot(*X_), np.hypot(*Y_)]) - true) / true < 0.03))
            (_, _, X_, Y_), _ = hybrid_choice(clf, rows, vecs)
            ok_hy += int(np.all(np.abs(np.array([np.hypot(*X_), np.hypot(*Y_)]) - true) / true < 0.03))
            _, _, X_, Y_ = rule_choice(rows, vecs)
            ok_rule += int(np.all(np.abs(np.array([np.hypot(*X_), np.hypot(*Y_)]) - true) / true < 0.03))
        lopo[pid] = {"n": len(te), "ml_ok": ok_ml, "hybrid_ok": ok_hy, "rule_ok": ok_rule}
        print("LOPO", pid, lopo[pid])
    res["lopo_synthetic"] = lopo
    # train on all synthetic, test on real bootstrap crops
    X = np.vstack([r for _, r, _, _, _ in data]); y = np.concatenate([l for _, _, l, _, _ in data])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, random_state=0).fit(X, y)
    real = {}
    for pid, fn, _, _, h in PATTERNS:
        rgb = load_rgb(RAW / fn); H, W = rgb.shape[:2]; rng = np.random.default_rng(3)
        pre = np.array([p for p in R1["patterns"] if p["id"] == pid][0]["repeat_px_before_affine"])
        ok_ml = ok_rule = ok_hy = n = 0; full = None; probs = []
        for k in range(13):
            if k == 0: crop = rgb
            else:
                hh, ww = int(0.8 * H), int(0.8 * W); yy, xx = rng.integers(0, H - hh), rng.integers(0, W - ww); crop = rgb[yy:yy + hh, xx:xx + ww]
            try:
                rows, vecs = candidate_table(acf_of(crop))
            except Exception:
                continue
            if len(rows) == 0: continue
            (_, _, X_, Y_), pr = choose(clf, rows, vecs); est_ml = np.array([np.hypot(*X_), np.hypot(*Y_)])
            (_, _, X_, Y_), _ = hybrid_choice(clf, rows, vecs); est_hy = np.array([np.hypot(*X_), np.hypot(*Y_)])
            _, _, X_, Y_ = rule_choice(rows, vecs); est_rule = np.array([np.hypot(*X_), np.hypot(*Y_)])
            if k == 0: full = {"ml": np.round(est_ml, 2).tolist(), "hybrid": np.round(est_hy, 2).tolist(), "rule": np.round(est_rule, 2).tolist(), "p": round(pr, 3)}
            else:
                n += 1; ok_ml += int(np.all(np.abs(est_ml - pre) / pre < 0.03)); ok_rule += int(np.all(np.abs(est_rule - pre) / pre < 0.03)); ok_hy += int(np.all(np.abs(est_hy - pre) / pre < 0.03)); probs.append(pr)
        real[pid] = {"full": full, "crops_n": n, "ml_ok": ok_ml, "hybrid_ok": ok_hy, "rule_ok": ok_rule, "p_mean": round(float(np.mean(probs)), 3)}
        print("REAL", pid, real[pid])
    res["real_bootstrap"] = real
    res["feature_names"] = ["s1_min", "s1_mean", "s2_min", "s2_mean", "n_shell2", "area_rel", "len_v1", "len_v2", "peak_v1", "peak_v2", "len_X", "len_Y", "cos_angle", "s1_rel", "s1_rank"]
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "results_ml.json", "w"), indent=1)
    print("done", res["runtime_s"])


if __name__ == "__main__":
    main()
