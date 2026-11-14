"""Extended analysis of IBM real-hardware brick-wall + VQC results.

Usage:
    ..\.venv\Scripts\python.exe hw_analyze_extended.py [--tag hw] [--seed 0] [--B 2000]

Outputs:
    results/hw/ibm_bw_{tag}/extended_analysis.json
    results/hw/ibm_bw_{tag}/hw_report.md

No qiskit dependency; uses only numpy/scipy/sklearn.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _auc(y, s):
    """Rank-based AUC (ties averaged); pure numpy."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_ci(y, s, B=2000, seed=0):
    """Bootstrap percentile CI for AUC.  Returns (auc, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    N = len(y); base = _auc(y, s)
    boot = []
    for _ in range(B):
        idx = rng.integers(0, N, N)
        yb, sb = y[idx], s[idx]
        if yb.sum() in (0, N):
            continue
        boot.append(_auc(yb, sb))
    boot = np.array(boot)
    return base, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_diff_ci(y, s1, s2, B=2000, seed=0):
    """Bootstrap percentile CI for paired difference AUC(s1) - AUC(s2)."""
    rng = np.random.default_rng(seed)
    N = len(y); base = _auc(y, s1) - _auc(y, s2)
    boot = []
    for _ in range(B):
        idx = rng.integers(0, N, N)
        yb = y[idx]
        if yb.sum() in (0, N):
            continue
        boot.append(_auc(yb, s1[idx]) - _auc(yb, s2[idx]))
    boot = np.array(boot)
    return base, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def features_from_counts(counts, n, k=None):
    """Duplicate of hw_bw.features_from_counts — no qiskit."""
    tot = sum(counts.values()); acc = np.zeros(n); kept = 0
    for bs, c in counts.items():
        bits = [int(ch) for ch in bs.replace(" ", "")[::-1]]
        if k is not None and sum(bits) != k:
            continue
        kept += c
        for j in range(n):
            if bits[j] and bits[(j + 1) % n]:
                acc[j] += c
    return (4 * acc / max(kept, 1)), kept / tot


def z_from_counts(counts):
    """Pure numpy <Z_j> from counts dict (qiskit little-endian bitstrings)."""
    NQ = 8
    shots = sum(counts.values()); z = np.zeros(NQ)
    for bs, c in counts.items():
        bits = bs.replace(" ", "")[::-1]
        z += c * np.array([1 - 2 * int(bits[j]) for j in range(NQ)])
    return z / shots


def balanced_accuracy_optimal(y, logit):
    """Find threshold that maximises balanced accuracy on the given (y, logit)."""
    thresholds = np.unique(logit)
    best_thr, best_ba = thresholds[0] - 1e-9, 0.0
    for t in thresholds:
        pred = (logit >= t).astype(int)
        tpr = float((pred[y == 1] == 1).mean()) if y.sum() > 0 else 0.0
        tnr = float((pred[y == 0] == 0).mean()) if (len(y) - y.sum()) > 0 else 0.0
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best_ba = ba; best_thr = t
    return best_thr, best_ba


def classify_at(y, logit, thr):
    """Return (acc, sens, spec) at threshold thr."""
    pred = (logit >= thr).astype(int)
    acc = float((pred == y).mean())
    sens = float((pred[y == 1] == 1).mean()) if y.sum() > 0 else float("nan")
    spec = float((pred[y == 0] == 0).mean()) if (len(y) - y.sum()) > 0 else float("nan")
    return acc, sens, spec


def global_rescale(hw_logit, nl_logit):
    """s = argmin ||s * hw - nl||^2 = cov(hw, nl) / var(hw)."""
    hw = np.asarray(hw_logit, dtype=float); nl = np.asarray(nl_logit, dtype=float)
    var_hw = np.var(hw)
    if var_hw < 1e-15:
        return 1.0
    return float(np.cov(hw, nl, ddof=1)[0, 1] / var_hw)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(tag, B, seed):
    d = Path("results/hw") / f"ibm_bw_{tag}"
    meta = json.load(open(d / "meta.json"))
    n_ecg = meta["n_ecg"]

    result = {}

    for model_name in ("bw8", "vqc32"):
        counts_path = d / f"counts_{model_name}.json"
        if not counts_path.exists():
            print(f"  {model_name}: no counts, skip")
            continue
        info = json.load(open(d / f"info_{model_name}.json"))
        e = np.load(d / f"ecgs_{model_name}.npz")
        y = np.array(e["y"], dtype=int)
        nl_score = np.array(e["noiseless_score"], dtype=float)
        cnt = json.load(open(counts_path))

        mr = {}

        # --- AUC with bootstrap CIs ---
        auc_nl, ci_nl_lo, ci_nl_hi = bootstrap_ci(y, nl_score, B=B, seed=seed)
        mr["auc_noiseless"] = auc_nl
        mr["auc_noiseless_ci"] = [ci_nl_lo, ci_nl_hi]

        if info["kind"] == "bw":
            n_qubits = info["n"]; k = info["k"]
            h = np.array(info["h"]); b_bias = info["b"]

            # raw (no postselection)
            raw_pairs = [features_from_counts(c, n_qubits) for c in cnt]
            f_raw = np.array([f for f, _ in raw_pairs])
            hw_raw_score = f_raw @ h + b_bias
            auc_raw, ci_raw_lo, ci_raw_hi = bootstrap_ci(y, hw_raw_score, B=B, seed=seed)
            mr["auc_raw"] = auc_raw
            mr["auc_raw_ci"] = [ci_raw_lo, ci_raw_hi]

            # postselected (k-particle)
            ps_pairs = [features_from_counts(c, n_qubits, k) for c in cnt]
            f_ps = np.array([f for f, _ in ps_pairs])
            kept_fracs = np.array([q for _, q in ps_pairs])
            hw_ps_score = f_ps @ h + b_bias
            auc_ps, ci_ps_lo, ci_ps_hi = bootstrap_ci(y, hw_ps_score, B=B, seed=seed)
            mr["auc_postselected"] = auc_ps
            mr["auc_postselected_ci"] = [ci_ps_lo, ci_ps_hi]
            mr["kept_fraction_mean"] = float(np.mean(kept_fracs))
            mr["kept_fraction_std"] = float(np.std(kept_fracs))

            # paired diff: raw - noiseless
            d_raw, d_raw_lo, d_raw_hi = bootstrap_diff_ci(y, hw_raw_score, nl_score, B=B, seed=seed)
            mr["diff_raw_minus_noiseless"] = d_raw
            mr["diff_raw_minus_noiseless_ci"] = [d_raw_lo, d_raw_hi]

            # --- Per-ECG agreement ---
            # noiseless logit
            nl_logit = nl_score  # already linear combination before sigmoid
            hw_logit = hw_raw_score
            hw_ps_logit = hw_ps_score

            pr, pr_p = pearsonr(hw_logit, nl_logit)
            sp, sp_p = spearmanr(hw_logit, nl_logit)
            mr["pearson_r_raw"] = float(pr)
            mr["pearson_p_raw"] = float(pr_p)
            mr["spearman_rho_raw"] = float(sp)
            mr["spearman_p_raw"] = float(sp_p)
            mr["mae_logit_raw"] = float(np.mean(np.abs(hw_logit - nl_logit)))

            pr_ps, _ = pearsonr(hw_ps_logit, nl_logit)
            sp_ps, _ = spearmanr(hw_ps_logit, nl_logit)
            mr["pearson_r_postselected"] = float(pr_ps)
            mr["spearman_rho_postselected"] = float(sp_ps)
            mr["mae_logit_postselected"] = float(np.mean(np.abs(hw_ps_logit - nl_logit)))

            # Z-contraction factor: mean(|f_raw|) / mean(|f_noiseless|) where f_noiseless = nl_score recomputed
            nl_f = (nl_score - b_bias) / (np.linalg.norm(h) + 1e-15)  # projection only; use feature magnitude
            # Better: use raw feature magnitudes
            mean_abs_f_hw = float(np.mean(np.abs(f_raw)))
            # Reconstruct noiseless features: nl_score = f_nl @ h + b, so f_nl @ h = nl_score - b
            # We can't reconstruct individual f_nl components, but we can compare |hw_logit - b| / |h| vs |nl_logit - b| / |h|
            mean_abs_logit_hw = float(np.mean(np.abs(hw_raw_score - b_bias)))
            mean_abs_logit_nl = float(np.mean(np.abs(nl_score - b_bias)))
            mr["contraction_factor_raw"] = mean_abs_logit_hw / max(mean_abs_logit_nl, 1e-15)
            mr["mean_abs_hw_logit_centered"] = mean_abs_logit_hw
            mr["mean_abs_nl_logit_centered"] = mean_abs_logit_nl

            # --- Calibration ---
            thr_nl, ba_nl = balanced_accuracy_optimal(y, nl_logit)
            acc_nl, sens_nl, spec_nl = classify_at(y, nl_logit, thr_nl)
            mr["calibration"] = {
                "noiseless_optimal_threshold": float(thr_nl),
                "noiseless": {"acc": acc_nl, "sens": sens_nl, "spec": spec_nl, "bal_acc": ba_nl},
            }
            acc_hw, sens_hw, spec_hw = classify_at(y, hw_logit, thr_nl)
            ba_hw = (sens_hw + spec_hw) / 2
            mr["calibration"]["hw_raw_at_nl_thr"] = {"acc": acc_hw, "sens": sens_hw, "spec": spec_hw, "bal_acc": ba_hw}

            # Hardware-optimal threshold (for reference, not the main number)
            thr_hw, ba_hw_opt = balanced_accuracy_optimal(y, hw_logit)
            acc_hw_opt, sens_hw_opt, spec_hw_opt = classify_at(y, hw_logit, thr_hw)
            mr["calibration"]["hw_raw_optimal_threshold"] = float(thr_hw)
            mr["calibration"]["hw_raw_at_hw_thr"] = {
                "acc": acc_hw_opt, "sens": sens_hw_opt, "spec": spec_hw_opt, "bal_acc": ba_hw_opt}

            # Global rescale: s * hw_logit ~ nl_logit
            s_raw = global_rescale(hw_logit, nl_logit)
            hw_rescaled = s_raw * hw_logit
            thr_rs, ba_rs = balanced_accuracy_optimal(y, hw_rescaled)
            acc_rs, sens_rs, spec_rs = classify_at(y, hw_rescaled, thr_nl)
            mr["calibration"]["rescale_factor_raw"] = s_raw
            mr["calibration"]["hw_raw_rescaled_at_nl_thr"] = {
                "acc": acc_rs, "sens": sens_rs, "spec": spec_rs, "bal_acc": float((sens_rs + spec_rs) / 2)}
            mr["calibration"]["hw_raw_rescaled_at_rs_thr"] = {
                "thr": float(thr_rs), "acc": float(classify_at(y, hw_rescaled, thr_rs)[0]),
                "sens": float(classify_at(y, hw_rescaled, thr_rs)[1]),
                "spec": float(classify_at(y, hw_rescaled, thr_rs)[2]),
                "bal_acc": ba_rs}

        else:  # vqc32
            w = np.array(info["w"]); b_bias = info["b"]
            z_hw = np.array([z_from_counts(c) for c in cnt])
            hw_logit = z_hw @ w + b_bias
            nl_logit = nl_score

            auc_raw, ci_raw_lo, ci_raw_hi = bootstrap_ci(y, hw_logit, B=B, seed=seed)
            mr["auc_raw"] = auc_raw
            mr["auc_raw_ci"] = [ci_raw_lo, ci_raw_hi]

            d_raw, d_raw_lo, d_raw_hi = bootstrap_diff_ci(y, hw_logit, nl_logit, B=B, seed=seed)
            mr["diff_raw_minus_noiseless"] = d_raw
            mr["diff_raw_minus_noiseless_ci"] = [d_raw_lo, d_raw_hi]

            # Per-ECG agreement
            pr, pr_p = pearsonr(hw_logit, nl_logit)
            sp, sp_p = spearmanr(hw_logit, nl_logit)
            mr["pearson_r_raw"] = float(pr)
            mr["pearson_p_raw"] = float(pr_p)
            mr["spearman_rho_raw"] = float(sp)
            mr["spearman_p_raw"] = float(sp_p)
            mr["mae_logit_raw"] = float(np.mean(np.abs(hw_logit - nl_logit)))

            # Z-contraction: mean(|z_hw|) / mean(|z_nl|); nl z = (nl_score - b) needs w to invert
            # Use logit magnitude proxy
            mean_abs_logit_hw = float(np.mean(np.abs(hw_logit - b_bias)))
            mean_abs_logit_nl = float(np.mean(np.abs(nl_logit - b_bias)))
            mr["contraction_factor_raw"] = mean_abs_logit_hw / max(mean_abs_logit_nl, 1e-15)
            mr["mean_abs_hw_logit_centered"] = mean_abs_logit_hw
            mr["mean_abs_nl_logit_centered"] = mean_abs_logit_nl

            # Calibration
            thr_nl, ba_nl = balanced_accuracy_optimal(y, nl_logit)
            acc_nl, sens_nl, spec_nl = classify_at(y, nl_logit, thr_nl)
            mr["calibration"] = {
                "noiseless_optimal_threshold": float(thr_nl),
                "noiseless": {"acc": acc_nl, "sens": sens_nl, "spec": spec_nl, "bal_acc": ba_nl},
            }
            acc_hw, sens_hw, spec_hw = classify_at(y, hw_logit, thr_nl)
            ba_hw = (sens_hw + spec_hw) / 2
            mr["calibration"]["hw_raw_at_nl_thr"] = {"acc": acc_hw, "sens": sens_hw, "spec": spec_hw, "bal_acc": ba_hw}

            # Hardware-optimal threshold (for reference)
            thr_hw2, ba_hw_opt2 = balanced_accuracy_optimal(y, hw_logit)
            acc_hw_opt2, sens_hw_opt2, spec_hw_opt2 = classify_at(y, hw_logit, thr_hw2)
            mr["calibration"]["hw_raw_optimal_threshold"] = float(thr_hw2)
            mr["calibration"]["hw_raw_at_hw_thr"] = {
                "acc": acc_hw_opt2, "sens": sens_hw_opt2, "spec": spec_hw_opt2, "bal_acc": ba_hw_opt2}

            s_raw = global_rescale(hw_logit, nl_logit)
            hw_rescaled = s_raw * hw_logit
            thr_rs2, ba_rs2 = balanced_accuracy_optimal(y, hw_rescaled)
            acc_rs, sens_rs, spec_rs = classify_at(y, hw_rescaled, thr_nl)
            mr["calibration"]["rescale_factor_raw"] = s_raw
            mr["calibration"]["hw_raw_rescaled_at_nl_thr"] = {
                "acc": acc_rs, "sens": sens_rs, "spec": spec_rs, "bal_acc": float((sens_rs + spec_rs) / 2)}
            mr["calibration"]["hw_raw_rescaled_at_rs_thr"] = {
                "thr": float(thr_rs2), "acc": float(classify_at(y, hw_rescaled, thr_rs2)[0]),
                "sens": float(classify_at(y, hw_rescaled, thr_rs2)[1]),
                "spec": float(classify_at(y, hw_rescaled, thr_rs2)[2]),
                "bal_acc": ba_rs2}

        result[model_name] = mr
        print(f"\n{model_name}:", flush=True)
        for k2, v2 in mr.items():
            if isinstance(v2, dict):
                print(f"  {k2}:")
                for k3, v3 in v2.items():
                    print(f"    {k3}: {v3}")
            elif isinstance(v2, list):
                print(f"  {k2}: [{v2[0]:.4f}, {v2[1]:.4f}]")
            elif isinstance(v2, float):
                print(f"  {k2}: {v2:.4f}")
            else:
                print(f"  {k2}: {v2}")

    # Save JSON
    out_json = d / "extended_analysis.json"
    json.dump(result, open(out_json, "w"), indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    # Save markdown report
    _write_report(d, tag, result, n_ecg, meta)
    print(f"Saved: {d / 'hw_report.md'}", flush=True)


def _write_report(d, tag, result, n_ecg, meta):
    lines = [f"# Hardware report — ibm_bw_{tag}",
             "",
             f"**Backend:** {meta['backend']}  **ECGs:** {n_ecg}  **Shots:** {meta['shots']}",
             "",
             "## AUC with 95% bootstrap CIs (B=2000, seed=0, percentile)",
             "",
             "| model | AUC noiseless [95% CI] | AUC raw [95% CI] | AUC postselected [95% CI] | kept fraction |",
             "|---|---|---|---|---|"]
    for mname in ("bw8", "vqc32"):
        if mname not in result:
            continue
        mr = result[mname]
        def fmt_ci(key):
            v = mr.get(key)
            ci = mr.get(key + "_ci")
            if v is None:
                return "–"
            if ci:
                return f"{v:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
            return f"{v:.4f}"
        auc_nl = fmt_ci("auc_noiseless")
        auc_raw = fmt_ci("auc_raw")
        auc_ps = fmt_ci("auc_postselected")
        kf = f"{mr['kept_fraction_mean']:.3f}" if "kept_fraction_mean" in mr else "–"
        lines.append(f"| {mname} | {auc_nl} | {auc_raw} | {auc_ps} | {kf} |")
    lines += ["",
              "## Per-ECG agreement (hardware vs noiseless logit)",
              "",
              "| model | Pearson r | Spearman ρ | MAE logit | contraction factor |",
              "|---|---|---|---|---|"]
    for mname in ("bw8", "vqc32"):
        if mname not in result:
            continue
        mr = result[mname]
        pr = f"{mr.get('pearson_r_raw', float('nan')):.3f}"
        sp = f"{mr.get('spearman_rho_raw', float('nan')):.3f}"
        mae = f"{mr.get('mae_logit_raw', float('nan')):.3f}"
        cf = f"{mr.get('contraction_factor_raw', float('nan')):.3f}"
        lines.append(f"| {mname} | {pr} | {sp} | {mae} | {cf} |")
    lines += ["",
              "## Calibration (noiseless-optimal threshold)",
              "",
              "| model | condition | bal. acc | sens | spec | acc |",
              "|---|---|---|---|---|---|"]
    for mname in ("bw8", "vqc32"):
        if mname not in result:
            continue
        cal = result[mname].get("calibration", {})
        for cond, key in [("noiseless", "noiseless"), ("hw raw", "hw_raw_at_nl_thr"), ("hw rescaled", "hw_raw_rescaled_at_nl_thr")]:
            c = cal.get(key, {})
            if not c:
                continue
            lines.append(f"| {mname} | {cond} | {c.get('bal_acc', float('nan')):.3f} | {c.get('sens', float('nan')):.3f} | {c.get('spec', float('nan')):.3f} | {c.get('acc', float('nan')):.3f} |")
    lines += ["",
              f"*Rescale factors: bw8 s = {result.get('bw8', {}).get('calibration', {}).get('rescale_factor_raw', 'n/a')}, "
              f"vqc32 s = {result.get('vqc32', {}).get('calibration', {}).get('rescale_factor_raw', 'n/a')}*",
              "",
              "Note: threshold is noiseless-optimal (balanced-accuracy) — not from a held-out validation set.",
              ""]
    (d / "hw_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    run(a.tag, a.B, a.seed)
