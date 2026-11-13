"""Phase 4 (Exploratory, added 2026-11-14): per-ECG full-data paired bootstrap for the k=32 comparison.
Logged in RESULTS.md under 'Exploratory (added 2026-11-14)' BEFORE running.
  ..\.venv\Scripts\python.exe p4_full_bootstrap.py -> results/p4_bootstrap.json"""
import json, numpy as np
import scale_cv as S

B = 2000
rng = np.random.default_rng(42)

def roc_auc(y, s):
    s = np.asarray(s, float)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0: return float('nan')
    r = np.empty_like(s)
    order = np.argsort(s)
    ties = np.concatenate([[True], s[order[1:]] != s[order[:-1]], [True]])
    start = np.where(ties[:-1])[0]; end = np.where(ties[1:])[0]
    for s_, e_ in zip(start, end): r[order[s_:e_+1]] = (s_ + e_) / 2.0 + 1
    n1f = y.sum()
    return float((r[y == 1].sum() - n1f * (n1f + 1) / 2) / (n1f * n0))


def delong_cov(y, s1, s2):
    """DeLong 1988: covariance of two AUCs for paired test. Returns (auc1, auc2, var1, var2, cov)."""
    n = len(y); n1 = y.sum(); n0 = n - n1
    # placement values
    def pv10(s):  # V10: for each pos, fraction of negs it beats
        idx1 = np.where(y == 1)[0]; idx0 = np.where(y == 0)[0]
        v = np.zeros(n1)
        for i, p in enumerate(idx1):
            v[i] = np.mean(s[p] > s[idx0]) + 0.5 * np.mean(s[p] == s[idx0])
        return v
    def pv01(s):  # V01: for each neg, fraction of pos it's beaten by
        idx1 = np.where(y == 1)[0]; idx0 = np.where(y == 0)[0]
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
    """Z statistic for paired AUC difference from DeLong's method."""
    a1, a2, v1, v2, c = delong_cov(y, s1, s2)
    var_d = v1 + v2 - 2 * c
    if var_d <= 0: return float('nan'), a1 - a2
    return (a1 - a2) / np.sqrt(var_d), a1 - a2


def bootstrap_ci(y_folds, s1_folds, s2_folds, B=2000):
    """Fold-stratified bootstrap CI for mean AUC difference. Returns (mean_d, ci_lo, ci_hi)."""
    n_folds = len(y_folds)
    fold_sizes = [len(y) for y in y_folds]
    # Build combined array with fold index
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

    mean_d = np.mean([roc_auc(y, s1) - roc_auc(y, s2) for y, s1, s2 in zip(y_folds, s1_folds, s2_folds)])
    return float(mean_d), float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))


def get_scores_from_jsonl(jsonl_file, k, flt_extra=None):
    """Get per-ECG test scores by refitting using stored params from a scale_cv run."""
    # Load all rows, pick per fold
    rows = [json.loads(l) for l in open(jsonl_file)]
    rows = [r for r in rows if r.get('k') == k and r.get('selector') == 'l1'
            and r.get('split', 'grouped5') == 'grouped5'
            and r.get('steps', 1000) == 1000 and r.get('readout', 'z') == 'z'
            and not r.get('train_scale', False) and r.get('seed', 0) == 0]
    if flt_extra:
        rows = [r for r in rows if all(r.get(a) == b for a, b in flt_extra.items())]
    return {r['fold']: r for r in rows}


def main():
    import Vqc_fast8 as V

    X, y, g, _, feats = S.load()

    # Load VQC scores (already computed)
    vqc_scores = json.load(open('results/bw/vqc_scores.json'))
    s_pi6 = vqc_scores['vqc_pi6_k32']        # list of 5 fold-score arrays
    s_trained = vqc_scores['vqc_trained_k32']

    # For each fold: refit LR, deg-2 LR, XGBoost using stored feature list from stage2 pi/6 run
    rows_vqc = get_scores_from_jsonl('results/stage2_steps1000.jsonl', 32, {'scale': 'pi/6'})
    # fold 4: use n0_baseline_fold4 (deterministic selector)
    rows_vqc[4] = json.loads(open('results/n0_baseline_fold4.jsonl').readline())

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    import xgboost as xgb
    import warnings; warnings.filterwarnings('ignore')

    fold_data = {}
    for fold in range(5):
        r = rows_vqc[fold]
        Z, tr, va, te, _ = S.prep('grouped5', fold, 'l1')
        sel = [feats.index(nm) for nm in r['features']]
        Zk = Z[:, sel]  # n_samples x k (k=32 for all folds)

        X_tr = Zk[tr]; y_tr = y[tr]
        X_te = Zk[te]; y_te = y[te]

        # --- LR (plain, same as stage2 baseline) ---
        best_c, best_v = None, -np.inf
        for c in (0.001, 0.01, 0.1, 1, 10):
            m = LogisticRegression(C=c, max_iter=3000).fit(X_tr, y_tr)
            v = roc_auc(y_tr, m.decision_function(X_tr))
            if v > best_v: best_c, best_v = c, v
        lr_m = LogisticRegression(C=best_c, max_iter=3000).fit(X_tr, y_tr)
        s_lr_fold = lr_m.decision_function(X_te)

        # --- Degree-2 LR (bounded inputs, L1 penalty, up to 81 non-zero params as in QB3) ---
        sc = StandardScaler().fit(np.tanh(X_tr))
        Xb_tr = sc.transform(np.tanh(X_tr))
        Xb_te = sc.transform(np.tanh(X_te))
        pf = PolynomialFeatures(2, include_bias=False).fit(Xb_tr)
        Xp_tr = pf.transform(Xb_tr); Xp_te = pf.transform(Xb_te)
        best_c2, best_v2 = None, -np.inf
        for c in (0.001, 0.01, 0.1, 1):
            m = LogisticRegression(C=c, max_iter=5000, penalty='l1', solver='liblinear').fit(Xp_tr, y_tr)
            v = roc_auc(y_tr, m.decision_function(Xp_tr))
            if v > best_v2: best_c2, best_v2 = c, v
        lr2_m = LogisticRegression(C=best_c2, max_iter=5000, penalty='l1', solver='liblinear').fit(Xp_tr, y_tr)
        s_lr2_fold = lr2_m.decision_function(Xp_te)

        # --- XGBoost ---
        best_xgb, best_vx = None, -np.inf
        for d in (2, 3):
            for n in (50, 200):
                for lam in (1, 10):
                    m = xgb.XGBClassifier(n_estimators=n, max_depth=d, learning_rate=0.1, reg_lambda=lam,
                                          tree_method='hist', n_jobs=1, random_state=0, eval_metric='logloss').fit(X_tr, y_tr)
                    v = roc_auc(y_tr, m.predict_proba(X_tr)[:, 1])
                    if v > best_vx: best_xgb, best_vx = m, v
        s_xgb_fold = best_xgb.predict_proba(X_te)[:, 1]

        fold_data[fold] = dict(
            y=y_te.tolist(),
            s_lr=s_lr_fold.tolist(),
            s_lr2=s_lr2_fold.tolist(),
            s_xgb=s_xgb_fold.tolist(),
            s_pi6=list(s_pi6[fold]),
            s_trained=list(s_trained[fold]),
            auc_lr=roc_auc(y_te, s_lr_fold),
            auc_lr2=roc_auc(y_te, s_lr2_fold),
            auc_xgb=roc_auc(y_te, s_xgb_fold),
            auc_pi6=roc_auc(y_te, np.array(s_pi6[fold])),
            auc_trained=roc_auc(y_te, np.array(s_trained[fold])),
        )
        print(f"fold {fold}: LR {fold_data[fold]['auc_lr']:.4f}  LR2 {fold_data[fold]['auc_lr2']:.4f}  XGB {fold_data[fold]['auc_xgb']:.4f}  VQC_pi6 {fold_data[fold]['auc_pi6']:.4f}  VQC_tr {fold_data[fold]['auc_trained']:.4f}", flush=True)

    # Collect per-fold arrays
    folds = list(range(5))
    get = lambda k: [np.array(fold_data[f][k]) for f in folds]
    ys = [np.array(fold_data[f]['y']) for f in folds]

    models = {'LR': get('s_lr'), 'LR2': get('s_lr2'), 'XGB': get('s_xgb'),
              'VQC_pi6': get('s_pi6'), 'VQC_trained': get('s_trained')}

    # Pairwise comparisons: VQC_pi6 vs {LR, LR2, XGB, VQC_trained} and VQC_trained vs {LR, LR2, XGB}
    pairs = [('VQC_pi6', 'LR'), ('VQC_pi6', 'LR2'), ('VQC_pi6', 'XGB'),
             ('VQC_trained', 'LR'), ('VQC_trained', 'LR2'), ('VQC_trained', 'XGB'),
             ('LR2', 'LR'), ('LR2', 'VQC_pi6')]

    results = dict(fold_aucs=fold_data, pairs=[])
    table_rows = ["| comparison | mean AUC A | mean AUC B | mean d [95% CI] | DeLong Z per fold |",
                  "|---|---|---|---|---|"]

    for a_name, b_name in pairs:
        sa = models[a_name]; sb = models[b_name]
        mean_d, ci_lo, ci_hi = bootstrap_ci(ys, sa, sb, B=B)
        delong_per_fold = []
        for fold in range(5):
            yf = ys[fold]; saf = sa[fold]; sbf = sb[fold]
            z, d = fold_delong_z(yf, saf, sbf)
            delong_per_fold.append(dict(fold=fold, z=z, d=d))

        mean_a = np.mean([roc_auc(ys[f], sa[f]) for f in range(5)])
        mean_b = np.mean([roc_auc(ys[f], sb[f]) for f in range(5)])
        results['pairs'].append(dict(a=a_name, b=b_name, mean_a=mean_a, mean_b=mean_b,
                                     mean_d=mean_d, ci=[ci_lo, ci_hi], delong=delong_per_fold))

        z_str = ', '.join(f"f{r['fold']}:{r['z']:+.2f}" for r in delong_per_fold)
        table_rows.append(f"| {a_name} vs {b_name} | {mean_a:.4f} | {mean_b:.4f} | {mean_d:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] | {z_str} |")
        print(table_rows[-1], flush=True)

    json.dump(results, open('results/p4_bootstrap.json', 'w'), indent=1)
    print('\n'.join(table_rows))
    # Save table
    with open('results/p4_bootstrap_table.md', 'w') as f:
        f.write('\n'.join(table_rows) + '\n')
    print('Done -> results/p4_bootstrap.json, results/p4_bootstrap_table.md', flush=True)


if __name__ == '__main__':
    main()
