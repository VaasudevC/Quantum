"""SD sensitivity analyses at N = 100, k = 8 (added AFTER the primary result; the primary result and claim level are unchanged).
 (1) bootstrap levels + per-fold mean differences for the primary pair, with a fold-level CI (t-interval over the 5 fold means) and a
     three-level bootstrap that also resamples test ECGs (patients) inside each test fold;
 (2) "best" quantum / classical chosen by mean INNER-CV AUC instead of test AUC;
 (3) train-test gap of the angle VQC (Q1) vs LR (C1) with parameter counts.
  C:/bwvenv/Scripts/python.exe sd_sensitivity.py   -> results/sd_sensitivity_N100.json / .md"""
import glob, json
from pathlib import Path
import numpy as np
import sd_analyze as A

N, K = 100, 8


def load_inner():
    out = {}
    for f in glob.glob("results/sd_runs_*.w*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["N"] == N and r["k"] == K and r.get("seedoff", 0) == 0:
                out.setdefault(r["model"], np.full((5, 40), np.nan))[r["fold"], r["r"]] = r["inner_cv_auc"]
    return out


def pair_stats(data, q, c, ix):
    d = data[(q, N)]["auc"] - data[(c, N)]["auc"]
    bm = A.boot_mean(d, ix); lo, hi = A.ci(bm)
    fm = np.nanmean(d, axis=1)                                   # per-fold mean differences
    t975 = 2.776                                                 # t quantile, 4 df
    se = fm.std(ddof=1) / np.sqrt(5)
    return dict(q=q, c=c, mean_d=float(np.nanmean(d)), ci_hier=[lo, hi], p_sup_raw=float((np.sum(bm <= 0) + 1) / (A.B + 1)),
                per_fold=[float(x) for x in fm], ci_fold_t=[float(fm.mean() - t975 * se), float(fm.mean() + t975 * se)],
                win_frac=float((d[np.isfinite(d)] > 0).mean()))


def main():
    data = A.load(K); ix = A.boot_indices(); inner = load_inner(); res = {}
    qs = [m for m in A.QN if (m, N) in data]; cs = [m for m in A.CN if (m, N) in data]
    test_mean = {m: float(np.nanmean(data[(m, N)]["auc"])) for m in qs + cs}
    bq, bc = max(qs, key=test_mean.get), max(cs, key=test_mean.get)
    res["bootstrap_levels"] = ("hierarchical bootstrap (sd_analyze.boot_mean): level 1 = the 5 outer folds, resampled with replacement; level 2 = the 40 subsets "
                               "within each resampled fold, resampled with replacement; level 3 = test ECGs: NOT resampled (each subset's test AUC on the full outer test fold is "
                               "treated as fixed). All 40 subsets of a fold share the same test ECGs, so test-set sampling noise enters as a common fold-level shift, "
                               "which the fold level of the bootstrap resamples (with only 5 folds, coarsely).")
    res["primary_by_test_auc"] = pair_stats(data, bq, bc, ix)
    inner_mean = {m: float(np.nanmean(inner[m])) for m in qs + cs}
    iq, ic = max(qs, key=inner_mean.get), max(cs, key=inner_mean.get)
    res["inner_cv_means"] = inner_mean
    res["best_by_inner_cv"] = pair_stats(data, iq, ic, ix)
    g = pair_stats(data, "Q1_vqc", "C1_lr", ix)
    dg = data[("Q1_vqc", N)]["gap"] - data[("C1_lr", N)]["gap"]; bg = A.boot_mean(dg, ix)
    res["gap_Q1_vs_C1"] = dict(gap_Q1=float(np.nanmean(data[("Q1_vqc", N)]["gap"])), gap_C1=float(np.nanmean(data[("C1_lr", N)]["gap"])),
                               gap_diff=float(np.nanmean(dg)), gap_diff_ci=list(A.ci(bg)), gap_diff_per_fold=[float(x) for x in np.nanmean(dg, 1)],
                               test_auc_diff=g["mean_d"], test_auc_ci=g["ci_hier"],
                               params_Q1="81 (72 circuit angles + 8 readout weights + 1 bias), plus 0 trained encoding scales",
                               params_C1="9 (8 weights + 1 bias)")
    json.dump(res, open("results/sd_sensitivity_N100.json", "w"), indent=1)
    S = A.SHORT; p = res["primary_by_test_auc"]; s = res["best_by_inner_cv"]; gg = res["gap_Q1_vs_C1"]
    L = [f"**(1) Bootstrap levels.** {res['bootstrap_levels']}", "",
         f"Primary pair {S[p['q']]} − {S[p['c']]}: per-fold mean differences " + ", ".join(f"fold {i}: {x:+.4f}" for i, x in enumerate(p["per_fold"])) +
         f"; mean {p['mean_d']:+.4f}; hierarchical-bootstrap 95% CI [{p['ci_hier'][0]:+.4f}, {p['ci_hier'][1]:+.4f}]; fold-level t-interval over the 5 fold means (4 df) [{p['ci_fold_t'][0]:+.4f}, {p['ci_fold_t'][1]:+.4f}].", "",
         f"**(2) Best models chosen by inner-CV AUC** (on the N-ECG subsets, no test data): best quantum = {S[iq]} (mean inner-CV AUC {inner_mean[iq]:.4f}), best classical = {S[ic]} ({inner_mean[ic]:.4f}). "
         f"Difference on the test folds {s['mean_d']:+.4f}, 95% CI [{s['ci_hier'][0]:+.4f}, {s['ci_hier'][1]:+.4f}], fold-level t-interval [{s['ci_fold_t'][0]:+.4f}, {s['ci_fold_t'][1]:+.4f}], quantum wins {s['win_frac']:.2f} of subsets, "
         f"per-fold " + ", ".join(f"{x:+.4f}" for x in s["per_fold"]) + ". Inner-CV means: " + ", ".join(f"{S[m]} {v:.4f}" for m, v in sorted(inner_mean.items(), key=lambda kv: -kv[1])) + ".", "",
         f"**(3) Train–test gap, angle VQC (Q1) vs logistic regression (C1).** Parameters: Q1 {gg['params_Q1']}; C1 {gg['params_C1']} — the VQC has 9× more trainable parameters. "
         f"Gap Q1 {gg['gap_Q1']:+.4f}, C1 {gg['gap_C1']:+.4f}; difference {gg['gap_diff']:+.4f} [{gg['gap_diff_ci'][0]:+.4f}, {gg['gap_diff_ci'][1]:+.4f}] (per fold " + ", ".join(f"{x:+.4f}" for x in gg["gap_diff_per_fold"]) + "); "
         f"test AUC difference {gg['test_auc_diff']:+.4f} [{gg['test_auc_ci'][0]:+.4f}, {gg['test_auc_ci'][1]:+.4f}]."]
    Path("results/sd_sensitivity_N100.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
