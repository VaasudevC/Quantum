"""P1: tuned classical benchmarks (LR, degree-2 LR, RF, XGBoost, MLP) on the same CV protocol as scale_cv.py.

One JSON line per (model, selector, k, fold) in results/p1_classical.jsonl (resumable).
All hyperparameters are chosen on VALIDATION AUC; test AUC and sens@90%spec are only reported.

  python -u p1_classical.py --jobs 4
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scale_cv as S

C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
OUT = "results/p1_classical.jsonl"


def sens90(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= 0.10].max())


def fit_lr(F, y, tr, va, te, grid):
    best = None
    for C in grid:
        m = LogisticRegression(C=C, max_iter=5000).fit(F[tr], y[tr])
        v = roc_auc_score(y[va], m.decision_function(F[va]))
        if best is None or v > best[0]:
            best = (v, C, m)
    v, C, m = best
    return v, dict(C=C), m.decision_function(F[te])


def m_lr(Zs, y, tr, va, te):
    F = StandardScaler().fit(Zs[tr]).transform(Zs)
    return fit_lr(F, y, tr, va, te, C_GRID)


def m_lr2(Zs, y, tr, va, te):
    P = PolynomialFeatures(2, include_bias=False).fit_transform(Zs)
    P = StandardScaler().fit(P[tr]).transform(P)
    return fit_lr(P, y, tr, va, te, C_GRID[:8])


def m_rf(Zs, y, tr, va, te):
    from sklearn.ensemble import RandomForestClassifier
    best = None
    for depth in (None, 8, 16):
        for leaf in (1, 5):
            rf = RandomForestClassifier(800, max_depth=depth, min_samples_leaf=leaf, n_jobs=1, random_state=0,
                                        class_weight="balanced_subsample").fit(Zs[tr], y[tr])
            for ne in (300, 800):         # first 300 trees of an 800-tree forest == a 300-tree forest (same seeding)
                pv = np.mean([t.predict_proba(Zs[va])[:, 1] for t in rf.estimators_[:ne]], 0)
                v = roc_auc_score(y[va], pv)
                if best is None or v > best[0]:
                    pt = np.mean([t.predict_proba(Zs[te])[:, 1] for t in rf.estimators_[:ne]], 0)
                    best = (v, dict(n_estimators=ne, max_depth=depth, min_samples_leaf=leaf), pt)
    return best


def m_xgb(Zs, y, tr, va, te):
    import xgboost as xgb
    best = None
    pos = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
    for depth in (3, 5, 7):
        for lr in (0.03, 0.1):
            m = xgb.XGBClassifier(n_estimators=800, max_depth=depth, learning_rate=lr, subsample=0.8,
                                  colsample_bytree=0.8, tree_method="hist", n_jobs=1, random_state=0,
                                  eval_metric="auc", early_stopping_rounds=50, scale_pos_weight=pos)
            m.fit(Zs[tr], y[tr], eval_set=[(Zs[va], y[va])], verbose=False)
            v = roc_auc_score(y[va], m.predict_proba(Zs[va])[:, 1])
            if best is None or v > best[0]:
                best = (v, dict(max_depth=depth, learning_rate=lr, n_estimators=int(m.best_iteration) + 1),
                        m.predict_proba(Zs[te])[:, 1])
    return best


def m_mlp(Zs, y, tr, va, te):
    from sklearn.neural_network import MLPClassifier
    best = None
    for h in (8, 16, 32):
        for alpha in (1e-4, 1e-2):
            for seed in (0, 1):
                m = MLPClassifier((h,), alpha=alpha, max_iter=300, early_stopping=True, n_iter_no_change=15,
                                  random_state=seed).fit(Zs[tr], y[tr])
                v = roc_auc_score(y[va], m.predict_proba(Zs[va])[:, 1])
                if best is None or v > best[0]:
                    best = (v, dict(hidden=h, alpha=alpha, seed=seed), m.predict_proba(Zs[te])[:, 1])
    return best


MODELS = dict(lr=m_lr, lr2=m_lr2, rf=m_rf, xgb=m_xgb, mlp=m_mlp)


def run_one(cfg):
    X, y, _, _, feats = S.load()
    Z, tr, va, te, order = S.prep("grouped5", cfg["fold"], cfg["selector"])
    k = cfg["k"]
    sel = list(order[:k]) if k < Z.shape[1] else list(range(Z.shape[1]))
    Zs = Z[:, sel]
    t0 = time.time()
    v, hp, s = MODELS[cfg["model"]](Zs, y, tr, va, te)
    return dict(cfg, val_auc=float(v), test_auc=float(roc_auc_score(y[te], s)), sens90=sens90(y[te], s),
                hp=hp, secs=round(time.time() - t0, 1), features=[feats[j] for j in sel] if k <= 32 else "all",
                test_scores=[round(float(a), 5) for a in s])


def key(c):
    return (c["model"], c["selector"], c["k"], c["fold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    cfgs = []
    for fold in range(5):
        for k in (8, 12, 16, 32, 249):
            for m in MODELS:
                if m == "lr2" and k > 32:
                    continue
                cfgs.append(dict(model=m, selector="l1", k=k, fold=fold))
        for k in (8, 12):
            for m in MODELS:
                cfgs.append(dict(model=m, selector="cohen_dedup", k=k, fold=fold))
    # slow ones (k=249 trees) first for load balance
    cfgs.sort(key=lambda c: -(c["k"] if c["model"] in ("rf", "xgb") else 0))
    done = {key(json.loads(l)) for l in open(OUT)} if Path(OUT).exists() else set()
    todo = [c for c in cfgs if key(c) not in done]
    print(f"{len(cfgs)} configs, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run_one, c) for c in todo]
        for fu in as_completed(futs):
            r = fu.result()
            with open(OUT, "a") as fo:
                fo.write(json.dumps(r) + "\n")
            print(f"{r['model']:4s} {r['selector']:11s} k={r['k']:3d} f{r['fold']} val={r['val_auc']:.4f} "
                  f"test={r['test_auc']:.4f} {r['secs']}s", flush=True)


if __name__ == "__main__":
    main()
