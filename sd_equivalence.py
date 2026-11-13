"""EXPLORATORY (added after the results; not pre-registered): equivalence check, best quantum vs best classical chosen by INNER-CV AUC, at every N (k=8).
Equivalent within +-0.005 if the 95% hierarchical-bootstrap CI of the paired difference lies inside [-0.005, +0.005].  -> results/sd_equivalence_k8.md"""
import glob, json
import numpy as np
import sd_analyze as A
data = A.load(8); ix = A.boot_indices(); inner = {}
for f in glob.glob("results/sd_runs_*.w*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["k"] == 8 and r.get("seedoff", 0) == 0:
            inner.setdefault((r["model"], r["N"]), np.full((5, 40), np.nan))[r["fold"], r["r"]] = r["inner_cv_auc"]
L = ["| N | best quantum (inner CV) | best classical (inner CV) | mean d [95% CI] | CI inside ±0.005 | note |", "|---|---|---|---|---|---|"]
for N in (25, 50, 100, 200, 500):
    qs = [m for m in A.QN if (m, N) in data]; cs = [m for m in A.CN if (m, N) in data]
    iq = max(qs, key=lambda m: np.nanmean(inner[(m, N)])); ic = max(cs, key=lambda m: np.nanmean(inner[(m, N)]))
    d = data[(iq, N)]["auc"] - data[(ic, N)]["auc"]; bm = A.boot_mean(d, ix); lo, hi = A.ci(bm)
    L.append(f"| {N} | {A.SHORT[iq]} | {A.SHORT[ic]} | {np.nanmean(d):+.4f} [{lo:+.4f}, {hi:+.4f}] | {'yes' if lo > -0.005 and hi < 0.005 else 'no'} | {'reduced sample' if N == 500 else ''} |")
open("results/sd_equivalence_k8.md", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
