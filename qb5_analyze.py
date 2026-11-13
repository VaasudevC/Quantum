"""QB5 analysis: per setting, quantum (Q1, Q3) vs the best classical model chosen by inner CV (from the SD clean runs: highest mean inner-CV AUC among C1, C2, C3, C5)
and vs each dequantized twin (TW_LR, TW_LR2, TW_SVM).  Two paired quantities per (fold, r): perturbed-AUC difference (quantum - comparator) and drop difference
(comparator drop - quantum drop; > 0 = quantum loses less).  Hierarchical bootstrap: 3 discovery folds, then the 10 subsets within fold, 5000 resamples.
Candidate for QB6 (pre-registered): perturbed-AUC difference vs best classical >= +0.005 with CI lower bound > 0, AND > 0 vs every twin.
  python qb5_analyze.py -> results/qb5_table.md, results/qb5_analysis.json"""
import glob, json
import numpy as np

R = [json.loads(l) for l in open("results/qb5.jsonl")]
B = 5000; rng = np.random.default_rng(0)
SETTINGS = [("testnoise_0.1", "clean"), ("testnoise_0.3", "clean"), ("testnoise_0.5", "clean"), ("leaddrop_mean", "clean"), ("leaddrop_worst", "clean"),
            ("labelnoise_0.1", "clean"), ("labelnoise_0.2", "clean"), ("labelnoise_0.3", "clean"), ("imbalance_0.05", "clean"), ("imbalance_0.1", "clean"),
            ("site", "site_cleanref"), ("date", "date_cleanref"), ("subtype_IMI", None)]
inner = {}
for f in glob.glob("results/sd_runs_main.w*.jsonl"):
    for l in open(f):
        q = json.loads(l)
        if q["k"] == 8 and q["N"] == 100 and q["fold"] in (0, 1, 2) and q["r"] < 10: inner.setdefault(q["model"], []).append(q["inner_cv_auc"])
best_c = max(("C1_lr", "C2_svm_rbf", "C3_gpc", "C5_rf"), key=lambda m: np.mean(inner[m]))
folds = sorted({r["fold"] for r in R}); fi = rng.integers(0, len(folds), (B, len(folds))); ri = rng.integers(0, 10, (B, len(folds), 10))


def arr(model, key):
    A = np.full((len(folds), 10), np.nan)
    for r in R:
        v = r["res"][model].get(key)
        if v is not None: A[folds.index(r["fold"]), r["r"]] = v
    return A


def boot(D):
    G = D[fi[:, :, None], ri]; m = np.nanmean(G.reshape(B, -1), 1); return float(np.nanmean(D)), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


out = dict(best_classical_by_inner_cv=best_c, n_units=len(R), rows=[]); cand = []
L = [f"Best classical by inner CV (clean SD subsets, folds 0–2, r 0–9): **{best_c}**. {len(R)} units (fold × subset). d = quantum − comparator on the perturbed AUC; Δdrop = comparator drop − quantum drop (> 0: quantum loses less).", "",
     "| setting | quantum | mean AUC quantum / best classical | d vs best classical [95% CI] | Δdrop vs best classical [95% CI] | d vs TW_LR / TW_LR2 / TW_SVM | candidate |", "|---|---|---|---|---|---|---|"]
for key, ref in SETTINGS:
    for q in ("Q1_vqc", "Q3_qkernel_svm"):
        Aq, Ac = arr(q, key), arr(best_c, key)
        if np.isnan(Aq).all(): continue
        d, dci = boot(Aq - Ac)
        if ref:
            dd, ddci = boot((arr(best_c, ref) - Ac) - (arr(q, ref) - Aq))
        else:
            dd, ddci = None, None
        tw = {t: boot(Aq - arr(t, key)) for t in ("TW_LR", "TW_LR2", "TW_SVM")}
        is_c = d >= 0.005 and dci[0] > 0 and all(v[0] > 0 for v in tw.values())
        if is_c: cand.append(dict(setting=key, quantum=q, d=d, ci=dci))
        out["rows"].append(dict(setting=key, quantum=q, auc_q=float(np.nanmean(Aq)), auc_c=float(np.nanmean(Ac)), d=d, d_ci=dci, ddrop=dd, ddrop_ci=ddci, twins={k: v for k, v in tw.items()}, candidate=is_c))
        L.append(f"| {key} | {q.split('_')[0]} | {np.nanmean(Aq):.3f} / {np.nanmean(Ac):.3f} | {d:+.4f} [{dci[0]:+.4f}, {dci[1]:+.4f}] | " + (f"{dd:+.4f} [{ddci[0]:+.4f}, {ddci[1]:+.4f}]" if dd is not None else "–") +
                 " | " + " / ".join(f"{v[0]:+.4f}" for v in tw.values()) + f" | {'**yes**' if is_c else 'no'} |")
out["candidates"] = cand
json.dump(out, open("results/qb5_analysis.json", "w"), indent=1); open("results/qb5_table.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L)); print("candidates:", cand)
