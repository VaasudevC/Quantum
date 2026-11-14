"""Phase 4 (Exploratory, added 2026-11-14): per-ECG full-data paired bootstrap for the k=32 comparison.
Logged in RESULTS.md under 'Exploratory (added 2026-11-14)' BEFORE running.
Uses pre-tuned classical models from results/p1_classical.jsonl (k=32) and VQC scores
from results/bw/vqc_scores.json. Bootstrap seed=0.
  ..\.venv\Scripts\python.exe p4_full_bootstrap.py -> results/p4_bootstrap.json"""
import json
import numpy as np
from pathlib import Path
import sd_prep as P

B = 2000
rng = np.random.default_rng(0)


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s, float)
    if y.sum() == 0 or (1 - y).sum() == 0: return float('nan')
    return float(roc_auc_score(y, s))


def delong_cov(y, s1, s2):
    """DeLong 1988: covariance of two AUCs for paired test. Returns (auc1, auc2, var1, var2, cov)."""
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    idx1 = np.where(y == 1)[0]; idx0 = np.where(y == 0)[0]
    def pv10(s):
        v = np.zeros(n1)
        for i, p in enumerate(idx1):
            v[i] = np.mean(s[p] > s[idx0]) + 0.5 * np.mean(s[p] == s[idx0])
        return v
    def pv01(s):
        v = np.zeros(n0)
        for i, p in enumerate(idx0):
            v[i] = np.mean(s[idx1] > s[p]) + 0.5 * np.mean(s[idx1] == s[p])
        return v
    v10_1 = pv10(s1); v10_2 = pv10(s2)
    v01_1 = pv01(s1); v01_2 = pv01(s2)
    a1 = roc_auc(y, s1); a2 = roc_auc(y, s2)
    s10 = np.cov(v10_1, v10_2); s01 = np.cov(v01_1, v01_2)
    var1 = s10[0, 0] / n1 + s01[0, 0] / n0
    var2 = s10[1, 1] / n1 + s01[1, 1] / n0
    cov = s10[0, 1] / n1 + s01[0, 1] / n0
    return a1, a2, var1, var2, cov


def fold_delong_z(y, s1, s2):
    a1, a2, v1, v2, c = delong_cov(y, s1, s2)
    var_d = v1 + v2 - 2 * c
    if var_d <= 0: return float('nan'), a1 - a2
    return (a1 - a2) / np.sqrt(var_d), a1 - a2


def bootstrap_ci(y_folds, s1_folds, s2_folds, B=2000):
    """Fold-stratified bootstrap CI for mean AUC difference."""
    n_folds = len(y_folds)
    fold_sizes = [len(y) for y in y_folds]
    all_y = np.concatenate(y_folds)
    all_s1 = np.concatenate(s1_folds)
    all_s2 = np.concatenate(s2_folds)
    fold_idx = np.concatenate([np.full(n, i) for i, n in enumerate(fold_sizes)])

    boot_diffs = np.zeros(B)
    for b in range(B):
        d_sum = 0.0
        for fi in range(n_folds):
            mask = fold_idx == fi
            yf = all_y[mask]; s1f = all_s1[mask]; s2f = all_s2[mask]
            idx = rng.integers(0, len(yf), len(yf))
            yb = yf[idx]; s1b = s1f[idx]; s2b = s2f[idx]
            d_sum += roc_auc(yb, s1b) - roc_auc(yb, s2b)
        boot_diffs[b] = d_sum / n_folds

    mean_d = np.mean([roc_auc(y, s1) - roc_auc(y, s2)
                      for y, s1, s2 in zip(y_folds, s1_folds, s2_folds)])
    return float(mean_d), float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))


def main():
    import scale_cv as S
    X, y, g, _, feats = S.load()

    # --- Load VQC scores (pre-computed) ---
    vqc_scores = json.load(open('results/bw/vqc_scores.json'))
    s_pi6_folds = vqc_scores['vqc_pi6_k32']        # list of 5 fold-score arrays
    s_trained_folds = vqc_scores['vqc_trained_k32']

    # --- Load tuned classical scores from p1_classical.jsonl (k=32) ---
    p1_rows = [json.loads(l) for l in open('results/p1_classical.jsonl')]
    p1_k32 = {m: sorted([r for r in p1_rows if r.get('k') == 32 and r['model'] == m],
                         key=lambda x: x['fold'])
              for m in ('lr', 'lr2', 'xgb')}

    # Build per-fold arrays
    fold_data = {}
    for fold in range(5):
        _, te = P.outer(fold)
        y_te = y[te]

        s_lr = np.array(p1_k32['lr'][fold]['test_scores'])
        s_lr2 = np.array(p1_k32['lr2'][fold]['test_scores'])
        s_xgb = np.array(p1_k32['xgb'][fold]['test_scores'])
        s_pi6 = np.array(s_pi6_folds[fold])
        s_tr = np.array(s_trained_folds[fold])

        fold_data[fold] = dict(
            y=y_te.tolist(),
            s_lr=s_lr.tolist(),
            s_lr2=s_lr2.tolist(),
            s_xgb=s_xgb.tolist(),
            s_pi6=s_pi6.tolist(),
            s_trained=s_tr.tolist(),
            auc_lr=roc_auc(y_te, s_lr),
            auc_lr2=roc_auc(y_te, s_lr2),
            auc_xgb=roc_auc(y_te, s_xgb),
            auc_pi6=roc_auc(y_te, s_pi6),
            auc_trained=roc_auc(y_te, s_tr),
        )
        print(f"fold {fold}: LR {fold_data[fold]['auc_lr']:.4f}  LR2 {fold_data[fold]['auc_lr2']:.4f}"
              f"  XGB {fold_data[fold]['auc_xgb']:.4f}"
              f"  VQC_pi6 {fold_data[fold]['auc_pi6']:.4f}"
              f"  VQC_tr {fold_data[fold]['auc_trained']:.4f}", flush=True)

    # Collect per-fold arrays
    folds = list(range(5))
    get = lambda k: [np.array(fold_data[f][k]) for f in folds]
    ys = [np.array(fold_data[f]['y']) for f in folds]

    models = {
        'LR': get('s_lr'), 'LR2': get('s_lr2'), 'XGB': get('s_xgb'),
        'VQC_pi6': get('s_pi6'), 'VQC_trained': get('s_trained'),
    }

    # Pairwise comparisons
    pairs = [
        ('VQC_pi6', 'LR'), ('VQC_pi6', 'LR2'), ('VQC_pi6', 'XGB'),
        ('VQC_trained', 'LR'), ('VQC_trained', 'LR2'), ('VQC_trained', 'XGB'),
        ('LR2', 'LR'), ('LR2', 'VQC_pi6'),
    ]

    results = dict(fold_aucs=fold_data, pairs=[], note=(
        "Classical LR, LR2, XGB are tuned models from p1_classical.jsonl (k=32). "
        "LR2 is tuned deg-2 LR (mean AUC 0.9583). "
        "VQC scores from results/bw/vqc_scores.json. Bootstrap seed=0."
    ))
    table_rows = [
        "| comparison | mean AUC A | mean AUC B | mean d [95% CI] |",
        "|---|---|---|---|",
    ]

    for a_name, b_name in pairs:
        sa = models[a_name]; sb = models[b_name]
        mean_d, ci_lo, ci_hi = bootstrap_ci(ys, sa, sb, B=B)

        mean_a = np.mean([roc_auc(ys[f], sa[f]) for f in range(5)])
        mean_b = np.mean([roc_auc(ys[f], sb[f]) for f in range(5)])
        results['pairs'].append(dict(a=a_name, b=b_name, mean_a=mean_a, mean_b=mean_b,
                                     mean_d=mean_d, ci=[ci_lo, ci_hi]))

        row = (f"| {a_name} vs {b_name} | {mean_a:.4f} | {mean_b:.4f} | "
               f"{mean_d:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] |")
        table_rows.append(row)
        print(row, flush=True)

    json.dump(results, open('results/p4_bootstrap.json', 'w'), indent=1)
    with open('results/p4_bootstrap_table.md', 'w') as f:
        f.write('\n'.join(table_rows) + '\n')
    print('\nDone -> results/p4_bootstrap.json, results/p4_bootstrap_table.md', flush=True)


if __name__ == '__main__':
    main()
