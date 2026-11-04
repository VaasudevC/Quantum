"""Stage 4b: compare the VQC at s* with a degree-2 polynomial logistic regression.

For each finished VQC run (one per fold) in --vqc that matches --scale/--k/--steps:
  * rebuild the exact split, imputation, scaling and feature selection used by scale_cv.py
  * recompute the VQC test probabilities from the stored params
  * fit on the inner-train part only, choosing C by VALIDATION AUC (never test):
      lr1        linear LR on z                      (same as lr_auc in scale_cv)
      poly2      degree-2 LR on z                    (k + k(k+1)/2 terms)
      poly2_ang  degree-2 LR on the VQC angles s*tanh(z)
  * report test AUC for each, and how closely each tracks the VQC:
      spearman_*   rank correlation of test scores with the VQC logit
      r2_deg1/2    R^2 (test rows) of a ridge fit of the VQC logit on degree-1 / degree-2
                   features of the angles (fit on train rows), i.e. how much of the
                   learned function is linear / quadratic in the encoded angles

  python -u stage4b_poly.py --vqc results/stage4_real.jsonl --scale pi/6 --k 32 --out results/stage4b_poly.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scale_cv as S
import Vqc_fast8 as V

C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]


def poly(F, tr):
    P = PolynomialFeatures(2, include_bias=False).fit_transform(F)
    return StandardScaler().fit(P[tr]).transform(P)


def fit_lr(F, y, tr, va, te, grid):
    best = None
    for C in grid:
        m = LogisticRegression(C=C, max_iter=5000).fit(F[tr], y[tr])
        v = roc_auc_score(y[va], m.decision_function(F[va]))
        if best is None or v > best[0]:
            best = (v, C, m)
    v, C, m = best
    s = m.decision_function(F)
    return dict(val_auc=float(v), test_auc=float(roc_auc_score(y[te], s[te])), C=C), s


def ridge_r2(F, t, tr, te):
    best = -np.inf
    for a in (1e-3, 1e-1, 10.0):
        best = max(best, r2_score(t[te], Ridge(alpha=a).fit(F[tr], t[tr]).predict(F[te])))
    return float(best)


def run(row):
    _, y, _, _, feats = S.load()
    Z, tr, va, te, order = S.prep(row["split"], row["fold"], row["selector"])
    k, s = row["k"], eval(row["scale"], {"pi": np.pi})
    sel = list(order[:k])
    assert [feats[j] for j in sel] == row["features"], "feature selection mismatch"
    Zs = Z[:, sel]
    A = np.zeros((len(Zs), V.N_FEATURES)); A[:, :k] = s * np.tanh(Zs)
    p = np.asarray(row["params"])
    zexp = V.expectations(V.forward(A, p[:V.N_ANSATZ]), "z")
    vqc = zexp @ p[V.N_ANSATZ:V.N_ANSATZ + 8] + p[-1]           # VQC logit
    vqc_auc = float(roc_auc_score(y[te], vqc[te]))

    out = dict(fold=row["fold"], k=k, scale=row["scale"], steps=row["steps"], split=row["split"],
               selector=row["selector"], seed=row["seed"], vqc_test_auc=vqc_auc,
               vqc_test_auc_logged=row["test_auc"])
    Zstd = StandardScaler().fit(Zs[tr]).transform(Zs)
    models = {"lr1": (Zstd, [1.0]), "poly2": (poly(Zs, tr), C_GRID), "poly2_ang": (poly(A[:, :k], tr), C_GRID)}
    for name, (F, grid) in models.items():
        res, sc = fit_lr(F, y, tr, va, te, grid)
        out.update({f"{name}_{a}": b for a, b in res.items()})
        out[f"spearman_{name}"] = float(spearmanr(sc[te], vqc[te])[0])
    Ang = A[:, :k]
    out["r2_deg1"] = ridge_r2(StandardScaler().fit(Ang[tr]).transform(Ang), vqc, tr, te)
    out["r2_deg2"] = ridge_r2(poly(Ang, tr), vqc, tr, te)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqc", nargs="+", required=True); ap.add_argument("--scale", default="pi/6")
    ap.add_argument("--k", type=int, default=32); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for f in a.vqc for l in open(f)]
    rows = [r for r in rows if r["scale"] == a.scale and r["k"] == a.k and "params" in r
            and not r.get("shuffle_later", False) and (a.steps == 0 or r["steps"] == a.steps)]
    if not rows:
        sys.exit("no matching VQC runs with stored params")
    with open(a.out, "w") as fo:
        for r in sorted(rows, key=lambda r: (r["seed"], r["fold"])):
            o = run(r); fo.write(json.dumps(o) + "\n")
            print(f"fold {o['fold']} VQC={o['vqc_test_auc']:.4f} LR={o['lr1_test_auc']:.4f} "
                  f"poly2={o['poly2_test_auc']:.4f} (C={o['poly2_C']}) poly2_ang={o['poly2_ang_test_auc']:.4f} "
                  f"rho(VQC,LR)={o['spearman_lr1']:.3f} rho(VQC,poly2)={o['spearman_poly2']:.3f} "
                  f"R2 deg1={o['r2_deg1']:.3f} deg2={o['r2_deg2']:.3f}", flush=True)


if __name__ == "__main__":
    main()
