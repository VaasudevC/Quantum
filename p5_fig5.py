"""Phase 5 (Exploratory, added 2026-11-14): Figure 5 — QB3 coefficient heatmap.
Recomputes the symmetric 32×32 quadratic coefficient matrix Q for the degree-2
surrogate of (a) the pi/6 VQC and (b) the best degree-2 L1 LR on the same fold-0
features.  Plots both side by side with their eigenvalue spectra.
  ..\\.venv\\Scripts\\python.exe p5_fig5.py -> results/figs/fig5_qb3_coef_heatmap.png"""
import json, os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scale_cv as S
import Vqc_fast8 as V

OUT = Path("results/figs/fig5_qb3_coef_heatmap.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

FOLD = 0


def build_Q(coef, pf):
    """Build symmetric 32×32 quadratic coefficient matrix from deg-2 feature coefficients."""
    names = pf.get_feature_names_out([f"x{i}" for i in range(32)])
    Q = np.zeros((32, 32))
    for c, nm in zip(coef, names):
        parts = nm.split(" ")
        if len(parts) == 2:
            i, j = int(parts[0][1:]), int(parts[1][1:])
            Q[i, j] += c / 2
            Q[j, i] += c / 2
        elif "^2" in nm:
            i = int(nm.split("^")[0][1:])
            Q[i, i] += c
    return Q


def main():
    X, y, _, _, feats = S.load()
    Z, tr, va, te, _ = S.prep("grouped5", FOLD, "l1")

    # Load stored pi/6 VQC row for fold 0
    rows_pi6 = [json.loads(l) for l in open("results/stage2_steps1000.jsonl")]
    rows_pi6 = [r for r in rows_pi6 if r["fold"] == FOLD and r["k"] == 32
                and r["scale"] == "pi/6" and r["selector"] == "l1"][0]
    feature_names = rows_pi6["features"]
    sel = [feats.index(n) for n in feature_names]

    # Compute VQC logit and circuit angle activations
    T = np.zeros((len(Z), V.N_FEATURES))
    T[:, :32] = np.tanh(Z[:, sel])
    vqc_model = S.Linear(32, np.pi / 6, rows_pi6.get("readout", "z"),
                          rows_pi6.get("train_scale", False))
    th, w, b, s = vqc_model.split(np.asarray(rows_pi6["params"]))
    A = vqc_model.angles(T, s)
    logit = V.expectations(V.forward(A, th), "z") @ w + b
    A32 = A[:, :32]  # shape (n_samples, 32)

    # --- VQC surrogate Q ---
    pf_vqc = PolynomialFeatures(2, include_bias=False)
    P_vqc = pf_vqc.fit_transform(A32)
    sc_vqc = StandardScaler().fit(P_vqc[tr])
    Ps_vqc = sc_vqc.transform(P_vqc)

    best_ridge = None
    for al in (1e-3, 1e-2, 1e-1, 1, 10):
        m_r = Ridge(alpha=al).fit(Ps_vqc[tr], logit[tr])
        pred_te = m_r.predict(Ps_vqc[te])
        r2 = float(np.corrcoef(logit[te], pred_te)[0, 1] ** 2)
        if best_ridge is None or r2 > best_ridge[0]:
            best_ridge = (r2, al, m_r)
    r2_vqc, al_vqc, ridge_m = best_ridge
    coef_vqc = ridge_m.coef_ / sc_vqc.scale_
    Q_vqc = build_Q(coef_vqc, pf_vqc)

    # --- L1 deg-2 LR Q (on z-scored inputs, same 32 features) ---
    Zs = Z[:, sel]
    pf_lr = PolynomialFeatures(2, include_bias=False)
    P_lr = pf_lr.fit_transform(Zs)
    sc_lr = StandardScaler().fit(P_lr[tr])
    Pls = sc_lr.transform(P_lr)

    best_lr = None
    for C in np.logspace(-4, -1, 16):
        ml = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                max_iter=3000, random_state=0).fit(Pls[tr], y[tr])
        nnz = int((np.abs(ml.coef_) > 1e-10).sum())
        val_auc = roc_auc_score(y[va], ml.decision_function(Pls[va]))
        if nnz <= 81 and (best_lr is None or val_auc > best_lr[0]):
            best_lr = (val_auc, C, ml, nnz)
    _, lr_C, lr2_m, lr_nnz = best_lr
    coef_lr2 = lr2_m.coef_[0] / sc_lr.scale_
    Q_lr2 = build_Q(coef_lr2, pf_lr)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                              gridspec_kw={"height_ratios": [3, 1]})

    for ax, Q, title in [
        (axes[0, 0], Q_vqc,
         f"VQC π/6 surrogate (fold {FOLD}, α={al_vqc}, R²={r2_vqc:.3f})"),
        (axes[0, 1], Q_lr2,
         f"L1 deg-2 LR (fold {FOLD}, C={lr_C:.4g}, nnz={lr_nnz})")
    ]:
        vmax = np.abs(Q).max() or 1.0
        im = ax.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature index")
        fig.colorbar(im, ax=ax, shrink=0.8)

    for ax, Q, label in [
        (axes[1, 0], Q_vqc, "VQC surrogate"),
        (axes[1, 1], Q_lr2, "L1 deg-2 LR")
    ]:
        ev = np.sort(np.abs(np.linalg.eigvalsh(Q)))[::-1]
        total = ev.sum() or 1.0
        ax.bar(range(len(ev)), ev / total, color="steelblue", alpha=0.7)
        ax.set_xlabel("Eigenvalue rank")
        ax.set_ylabel("Fraction of |λ| sum")
        ax.set_title(f"{label} eigenspectrum")

    fig.suptitle(
        f"QB3 Nonlinearity — Degree-2 Coefficient Heatmaps (fold {FOLD}, exploratory)",
        fontsize=12
    )
    plt.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()

    ev_vqc = np.sort(np.abs(np.linalg.eigvalsh(Q_vqc)))[::-1]
    print(f"Saved -> {OUT}")
    print(f"VQC surrogate R² = {r2_vqc:.4f},  top-eigenvalue fraction = {ev_vqc[0] / (ev_vqc.sum() or 1):.3f}")
    print(f"LR2 nonzero coefs = {lr_nnz}")


if __name__ == "__main__":
    main()
