"""SD3 analysis (pre-registered plan in SMALLDATA.md): paired quantum-vs-classical AUC differences per training size N,
hierarchical bootstrap (resample folds, then subsets within fold; 5000 resamples), one-sided superiority tests with Holm correction across all
(quantum x classical x size) comparisons, non-inferiority against the fixed margin -0.02, win fractions, generalisation-gap comparison,
tables and figures.  Numpy only (run with C:/bwvenv python for matplotlib).
  python sd_analyze.py [K]     -> results/sd_analysis_k{K}.json, results/sd_tables_k{K}.md, results/figs/sd_*.png"""
import glob, json, sys
from pathlib import Path
import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
B = 5000; MARGIN = -0.02; SIZES = (25, 50, 100, 200, 500)
QN = ["Q1_vqc", "Q2_vqc_trained_scales", "Q3_qkernel_svm", "Q4_brickwall8"]
CN = ["C1_lr", "C2_svm_rbf", "C3_gpc", "C4_knn", "C5_rf", "C6_xgb", "C7_tabpfn"]
SHORT = {"Q1_vqc": "Q1 angle VQC", "Q2_vqc_trained_scales": "Q2 VQC trained scales", "Q3_qkernel_svm": "Q3 quantum-kernel SVM", "Q4_brickwall8": "Q4 brick-wall n=8",
         "C1_lr": "C1 LR", "C2_svm_rbf": "C2 SVM-RBF", "C3_gpc": "C3 GP classifier", "C4_knn": "C4 kNN", "C5_rf": "C5 random forest", "C6_xgb": "C6 XGBoost", "C7_tabpfn": "C7 TabPFN"}


def load(k, seedoff=0, tag=""):
    """-> dict[(model, N)] = {"auc": (5,R) array, "train": ..., "gap": ...} with NaN where missing"""
    recs = {}
    for f in glob.glob(f"results/sd_runs_*{tag}.w*.jsonl") + glob.glob(f"results/sd_runs_*{tag}.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["k"] == k and r.get("seedoff", 0) == seedoff:
                recs[(r["model"], r["N"], r["fold"], r["r"])] = r
    out = {}
    for (m, N, f, r), v in recs.items():
        d = out.setdefault((m, N), {x: np.full((5, 40), np.nan) for x in ("auc", "train", "gap")})
        d["auc"][f, r] = v["test_auc"]; d["train"][f, r] = v["train_auc"]; d["gap"][f, r] = v["gap"]
    return out


def boot_indices(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5, (B, 5)), rng.integers(0, 40, (B, 5, 40))


def boot_mean(D, ix):
    """hierarchical bootstrap means of D (5, R<=40): resample folds, then subsets within fold; NaN-aware (uses available subsets)."""
    f, r = ix
    G = D[f[:, :, None], r]                      # (B, 5, 40)
    return np.nanmean(G.reshape(B, -1), axis=1)


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def main():
    data = load(K); ix = boot_indices()
    models = sorted({m for m, _ in data}); qs = [q for q in QN if q in models]; cs = [c for c in CN if c in models]
    sizes = [N for N in SIZES if any((m, N) in data for m in models)]
    res = dict(k=K, sizes=sizes, quantum=qs, classical=cs, mean_auc={}, pairs=[], n_comparisons=len(qs) * len(cs) * len(sizes))
    for m in models:
        for N in sizes:
            if (m, N) in data:
                D = data[(m, N)]["auc"]; bm = boot_mean(D, ix); n = int(np.isfinite(D).sum())
                res["mean_auc"][f"{m}|{N}"] = dict(mean=float(np.nanmean(D)), ci=ci(bm), n=n, gap=float(np.nanmean(data[(m, N)]["gap"])), train=float(np.nanmean(data[(m, N)]["train"])))
    rows = []
    for q in qs:
        for c in cs:
            for N in sizes:
                if (q, N) not in data or (c, N) not in data: continue
                Dq, Dc = data[(q, N)], data[(c, N)]
                d = Dq["auc"] - Dc["auc"]; ok = np.isfinite(d)
                if ok.sum() < 20: continue
                bm = boot_mean(d, ix); lo, hi = ci(bm)
                dg = Dq["gap"] - Dc["gap"]; bg = boot_mean(dg, ix); glo, ghi = ci(bg)
                rows.append(dict(q=q, c=c, N=N, n=int(ok.sum()), mean_d=float(np.nanmean(d)), ci=[lo, hi], win_frac=float((d[ok] > 0).mean()),
                                 p_sup=float((np.sum(bm <= 0) + 1) / (B + 1)), p_ni=float((np.sum(bm <= MARGIN) + 1) / (B + 1)),
                                 gap_d=float(np.nanmean(dg)), gap_ci=[glo, ghi]))
    if rows:
        adj = holm([r["p_sup"] for r in rows]); adjn = holm([r["p_ni"] for r in rows])
        for r, a, an in zip(rows, adj, adjn):
            r["p_sup_holm"] = float(a); r["p_ni_holm"] = float(an)
            r["superiority"] = bool(a < 0.05); r["noninferior_ci"] = bool(r["ci"][0] > MARGIN); r["noninferior_holm"] = bool(an < 0.05)
            r["smaller_gap"] = bool(r["gap_ci"][1] < 0); r["larger_gap"] = bool(r["gap_ci"][0] > 0)
    res["pairs"] = rows
    # best-vs-best per size (best = highest mean AUC at that size, chosen after seeing results)
    bb = []
    for N in sizes:
        mq = {q: res["mean_auc"].get(f"{q}|{N}", {}).get("mean", -1) for q in qs}; mc = {c: res["mean_auc"].get(f"{c}|{N}", {}).get("mean", -1) for c in cs}
        if not mq or not mc: continue
        q = max(mq, key=mq.get); c = max(mc, key=mc.get)
        pr = [r for r in rows if r["q"] == q and r["c"] == c and r["N"] == N]
        if pr: bb.append(dict(N=N, best_q=q, best_c=c, **{k: pr[0][k] for k in ("mean_d", "ci", "win_frac", "p_sup_holm", "superiority", "noninferior_ci", "gap_d", "gap_ci")}))
    res["best_vs_best"] = bb
    Path("results").mkdir(exist_ok=True); json.dump(res, open(f"results/sd_analysis_k{K}.json", "w"), indent=1)
    # ---- tables
    L = [f"### k = {K}: mean test AUC [95% hierarchical-bootstrap CI] (5 folds x up to 40 subsets), generalisation gap in parentheses\n", "| model | " + " | ".join(f"N={N}" for N in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for m in qs + cs:
        L.append(f"| {SHORT[m]} | " + " | ".join((lambda a: f"{a['mean']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}] ({a['gap']:+.2f})" if a else "–")(res["mean_auc"].get(f"{m}|{N}")) for N in sizes) + " |")
    L += ["", f"### Best quantum vs best classical per size (best = highest mean AUC at that size)\n", "| N | best quantum | best classical | mean d [95% CI] | wins | Holm p (superiority) | superiority | non-inferior (CI lower > −0.02) |", "|---|---|---|---|---|---|---|---|"]
    for b in bb:
        L.append(f"| {b['N']} | {SHORT[b['best_q']]} | {SHORT[b['best_c']]} | {b['mean_d']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] | {b['win_frac']:.2f} | {b['p_sup_holm']:.3f} | {b['superiority']} | {b['noninferior_ci']} |")
    L += ["", f"### All {len(rows)} quantum × classical × size comparisons (Holm over {len(rows)}; d = quantum − classical)\n", "| N | quantum | classical | mean d [95% CI] | wins | p_sup (Holm) | superior | non-inf. | gap diff [CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["N"], r["q"], r["c"])):
        L.append(f"| {r['N']} | {SHORT[r['q']]} | {SHORT[r['c']]} | {r['mean_d']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['win_frac']:.2f} | {r['p_sup_holm']:.3f} | {r['superiority']} | {r['noninferior_ci']} | {r['gap_d']:+.3f} [{r['gap_ci'][0]:+.3f}, {r['gap_ci'][1]:+.3f}] |")
    Path(f"results/sd_tables_k{K}.md").write_text("\n".join(L), encoding="utf-8")
    # ---- figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print("no matplotlib:", e); return
    Path("results/figs").mkdir(parents=True, exist_ok=True)
    col = {"Q1_vqc": "#cc3311", "Q2_vqc_trained_scales": "#ee7733", "Q3_qkernel_svm": "#aa3377", "Q4_brickwall8": "#882255", "C1_lr": "#888888", "C2_svm_rbf": "#228833",
           "C3_gpc": "#009988", "C4_knn": "#bbbb33", "C5_rf": "#4477aa", "C6_xgb": "#66ccee", "C7_tabpfn": "#000000"}
    for fname, key, ylab in (("sd_auc_vs_N", "mean", "mean test AUC"), ("sd_gap_vs_N", "gap", "generalisation gap (train AUC − test AUC)")):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for m in qs + cs:
            xs = [N for N in sizes if f"{m}|{N}" in res["mean_auc"]]
            if not xs: continue
            ys = np.array([res["mean_auc"][f"{m}|{N}"][key] for N in xs])
            ax.plot(xs, ys, marker="o", ms=3, lw=2 if m.startswith("Q") else 1.1, ls="-" if m.startswith("Q") else "--", color=col[m], label=SHORT[m])
            if key == "mean":
                lo = np.array([res["mean_auc"][f"{m}|{N}"]["ci"][0] for N in xs]); hi = np.array([res["mean_auc"][f"{m}|{N}"]["ci"][1] for N in xs]); ax.fill_between(xs, lo, hi, color=col[m], alpha=0.12)
        ax.set_xscale("log"); ax.set_xlabel("training ECGs N (one per patient)"); ax.set_ylabel(ylab); ax.set_title(f"k = {K} features"); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
        plt.tight_layout(); plt.savefig(f"results/figs/{fname}{'' if K == 8 else f'_k{K}'}.png", dpi=150); plt.close()
    if bb:
        fig, ax = plt.subplots(figsize=(6.5, 4.5)); xs = [b["N"] for b in bb]
        ax.errorbar(xs, [b["mean_d"] for b in bb], yerr=[[b["mean_d"] - b["ci"][0] for b in bb], [b["ci"][1] - b["mean_d"] for b in bb]], marker="o", capsize=3, color="#cc3311")
        ax.axhline(0, color="k", lw=1); ax.axhline(MARGIN, color="k", lw=1, ls="--"); ax.set_xscale("log"); ax.set_xlabel("training ECGs N"); ax.set_ylabel("best quantum − best classical (test AUC)")
        ax.set_title(f"k = {K}: paired difference, 95% hierarchical-bootstrap CI\n(dashed: non-inferiority margin −0.02)"); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"results/figs/sd_diff_vs_N{'' if K == 8 else f'_k{K}'}.png", dpi=150); plt.close()
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
