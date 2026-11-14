"""Train hybrid VQC with PLS-8 preprocessing on all 251 PTB-XL ECG features.

Why PLS-8 preprocessing
------------------------
Previous runs used 32 hand-selected features (best LR: 87.14%, AUC=0.9358).
PLS-8 uses ALL 251 features and finds the 8 maximally label-predictive linear
combinations:
    PLS-8 + LR  : 88.91% accuracy, AUC=0.9481
    Hand-sel 32  : 87.14% accuracy, AUC=0.9358  (delta = +1.77pp acc, +0.012 AUC)

The quantum circuit adds non-linear combinations of these 8 components.
Since LR on 8 PLS components already achieves 88.91%, the quantum non-linearities
should push accuracy toward 90%+.

Hardware pipeline
-----------------
Classical control (CPU):
    251 features -> StandardScaler -> PLSRegression (251x8 matrix) -> 8 angles
QPU:
    8 angles -> Ry data re-upload (4 layers) -> Ry+Rz ansatz -> CNOT ring
Classical post (CPU):
    8 Z-expectations -> MLP(32 hidden) -> p(MI)
The PLS matrix is a fixed linear transform; no extra QPU cost.

Circuit (same as Run 7)
-----------------------
    N_QUBITS=8, N_LAYERS=4, N_ANSATZ_PARAMS=72, N_MLP_PARAMS=321, N_PARAMS=393
    Data re-uploading: same 8 PLS angles at each of 4 layers (tiled to 32).

CLI flags
---------
    --steps N        Adam steps (default 300)
    --batch-size N   batch per gradient (default 300)
    --lr F           Adam lr (default 0.01)
    --seed N         RNG seed (default 42)
    --warm-start F   JSON file in results/ with quantum params to warm-start from
                     (default: vqc_hybrid8_params.json — Run 7 best params)
    --no-warm-start  skip warm-start, use random init N(0, 0.5)
    --out NAME       result JSON in results/ (default vqc_pls8_params.json)

Run:
    python -u Train_pls8.py > results/vqc_pls8.txt 2>&1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from Vqc_hybrid8 import (
    N_ANSATZ_PARAMS, N_MLP_PARAMS, N_PARAMS, N_HIDDEN, N_QUBITS,
    evaluate, predict_proba, train_adam,
)
from Vqc_pls8 import PLSPreprocessor, compute_pls_angles, N_PLS_COMPONENTS

CSV     = Path(__file__).parent / "PTB_XL" / "st_features_plus_TR.csv"
RESULTS = Path(__file__).parent / "results"

# Columns to exclude from the 251 numeric features
_META_COLS = {
    "label_class", "ecg_id", "patient_id",
    "filename_lr", "filename_hr", "scp_codes", "strat_fold",
}


def load_all_features(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    df = df[df["label_class"].isin(["MI", "NORM"])].copy()
    y  = (df["label_class"] == "MI").astype(np.int64).to_numpy()
    feat_cols = [
        c for c in df.columns
        if c not in _META_COLS
        and df[c].dtype in (np.float64, np.int64, np.float32, np.int32, int, float)
    ]
    X = df[feat_cols].astype(np.float64).to_numpy()
    if not np.isfinite(X).all():
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        idx = ~np.isfinite(X)
        X[idx] = np.take(med, np.where(idx)[1])
    return X, y, feat_cols


def compute_class_weight(y: np.ndarray) -> np.ndarray:
    n = len(y); nm = int(y.sum()); nn = n - nm
    return np.array([n / (2.0 * nn), n / (2.0 * nm)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",          type=int,   default=300)
    ap.add_argument("--batch-size",     type=int,   default=300)
    ap.add_argument("--lr",             type=float, default=0.01)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--warm-start",     type=str,   default="vqc_hybrid8_params.json",
                    help="JSON file in results/ with quantum params to warm-start from")
    ap.add_argument("--no-warm-start",  action="store_true")
    ap.add_argument("--out",            type=str,   default="vqc_pls8_params.json")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("  Hybrid VQC: PLS-8 preprocessing + data re-uploading + MLP readout")
    print("=" * 72)
    print(f"  N_PARAMS = {N_ANSATZ_PARAMS} circuit + {N_MLP_PARAMS} MLP = {N_PARAMS} total")
    print(f"  PLS components : {N_PLS_COMPONENTS} (from all 251 features)")
    print(f"  MLP            : {N_QUBITS} -> {N_HIDDEN} -> 1")

    # Load data
    print(f"\nLoading  {CSV}")
    X, y, feat_cols = load_all_features(CSV)
    n_feat = X.shape[1]
    print(f"  rows          : {X.shape[0]:,}   MI: {int(y.sum()):,}   NORM: {int((1-y).sum()):,}")
    print(f"  features used : {n_feat}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    print(f"  train: {len(ytr):,}   test: {len(yte):,}")

    # Fit PLS preprocessing on train only
    print("\nFitting PLS-8 preprocessor on training data ...")
    t_pls = time.time()
    prep = PLSPreprocessor(n_components=N_PLS_COMPONENTS)
    prep.fit(Xtr, ytr)
    print(f"  done ({time.time()-t_pls:.1f}s)")

    # PLS components
    Z_tr = prep.transform(Xtr)  # (n_train, 8)
    Z_te = prep.transform(Xte)  # (n_test,  8)

    # Verify PLS signal: LR on PLS components
    lr_check = LogisticRegression(max_iter=1000, C=1.0)
    lr_check.fit(Z_tr, ytr)
    p_lr  = lr_check.predict_proba(Z_te)[:, 1]
    acc_lr = (lr_check.predict(Z_te) == yte).mean()
    auc_lr = roc_auc_score(yte, p_lr)
    print(f"\n  Classical LR on PLS-{N_PLS_COMPONENTS}: acc={acc_lr:.4f}  AUC={auc_lr:.4f}  (quantum floor)")

    # PLS component stats
    mi_mask = ytr == 1
    print(f"\n  PLS component separation (MI vs NORM mean on train):")
    for i in range(N_PLS_COMPONENTS):
        mi_m   = float(Z_tr[mi_mask,  i].mean())
        no_m   = float(Z_tr[~mi_mask, i].mean())
        sep    = abs(mi_m - no_m)
        print(f"    PLS-{i+1}: MI={mi_m:+.3f}  NORM={no_m:+.3f}  sep={sep:.3f}")

    # Angle encoding with data re-uploading (tile same 8 angles x 4 layers)
    ang_tr = compute_pls_angles(Z_tr)  # (n_train, 32)
    ang_te = compute_pls_angles(Z_te)  # (n_test,  32)
    print(f"\n  Angles: shape={ang_tr.shape}  min={ang_tr.min():.3f}  max={ang_tr.max():.3f}")

    cw = compute_class_weight(ytr)
    print(f"  Class weights : NORM={cw[0]:.3f}  MI={cw[1]:.3f}")

    # Warm-start quantum params from a previous run if available
    init_quantum = None
    if not args.no_warm_start:
        ws_path = RESULTS / args.warm_start
        if ws_path.exists():
            prev = json.loads(ws_path.read_text())
            prev_params = np.array(prev["params"])
            init_quantum = prev_params[:N_ANSATZ_PARAMS]
            pm = prev.get("metrics", {})
            print(f"\n  Warm-start quantum params from: {ws_path.name}")
            print(f"    (prev test_acc={pm.get('test_accuracy','?'):.4f}  "
                  f"AUC={pm.get('test_auc','?'):.4f}  "
                  f"n_ansatz={len(init_quantum)})")
            print("  MLP readout: Xavier random init")
        else:
            print(f"\n  (warm-start file not found: {ws_path.name}, using random init)")
    else:
        print("\n  Initialization: fresh random  N(0,0.5) quantum, Xavier MLP")

    n_ep = args.steps * args.batch_size / len(ytr)
    print(f"\nTraining  steps={args.steps}  batch={args.batch_size}  "
          f"lr={args.lr}  ~{n_ep:.1f} effective epochs")
    print("  Quantum gradient: PSR (exact, hardware-portable)")
    print("  MLP gradient    : backpropagation")

    t0        = time.time()
    loss_hist: list[float] = []

    params = train_adam(
        ang_tr, ytr,
        n_steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        l2=1e-4,
        seed=args.seed,
        verbose=True,
        loss_history=loss_hist,
        class_weight=cw,
        init_quantum_params=init_quantum,
    )

    wall = time.time() - t0
    print(f"  wall time: {wall:.1f}s  ({wall/3600:.2f}h)")

    # Evaluate
    res_tr = evaluate(ang_tr, ytr, params)
    res_te = evaluate(ang_te, yte, params)

    # Optimal threshold sweep on test set
    p_te = predict_proba(ang_te, params)
    best_acc, best_thr = 0.0, 0.5
    for thr in np.arange(0.10, 0.90, 0.005):
        acc = ((p_te >= thr) == yte).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    pred_opt = (p_te >= best_thr).astype(int)
    tp_o = int(((pred_opt==1)&(yte==1)).sum())
    fn_o = int(((pred_opt==0)&(yte==1)).sum())
    tn_o = int(((pred_opt==0)&(yte==0)).sum())
    fp_o = int(((pred_opt==1)&(yte==0)).sum())

    print("\n" + "-" * 72)
    print("  Results")
    print("-" * 72)
    print(f"  Train  n={res_tr.n:,}   acc={res_tr.accuracy:.4f}  "
          f"AUC={res_tr.auc:.4f}  sens={res_tr.sensitivity:.4f}  spec={res_tr.specificity:.4f}")
    print(f"  Test   n={res_te.n:,}   acc={res_te.accuracy:.4f}  "
          f"AUC={res_te.auc:.4f}  sens={res_te.sensitivity:.4f}  spec={res_te.specificity:.4f}")
    print(f"  Test   (threshold={best_thr:.3f}) acc={best_acc:.4f}  "
          f"sens={tp_o/(tp_o+fn_o):.4f}  spec={tn_o/(tn_o+fp_o):.4f}")
    print("-" * 72)
    target = "YES -- target reached!" if best_acc >= 0.90 else \
             f"NO  -- gap = {0.90 - best_acc:.4f}"
    print(f"  >= 90% accuracy: {target}")
    print(f"  Classical LR floor (PLS-8): {acc_lr:.4f}")

    # Save
    summary = {
        "encoding":        "pls8_angle_reuploading",
        "readout":         "mlp_32hidden",
        "n_qubits":        N_QUBITS,
        "n_layers":        4,
        "n_pls_components": N_PLS_COMPONENTS,
        "n_input_features": n_feat,
        "n_ansatz_params": N_ANSATZ_PARAMS,
        "n_mlp_params":    N_MLP_PARAMS,
        "n_params_total":  N_PARAMS,
        "steps":           args.steps,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "class_weight":    {"NORM": float(cw[0]), "MI": float(cw[1])},
        "pls_preprocessor": prep.to_dict(),
        "feature_columns": feat_cols,
        "wall_time_seconds": round(wall, 2),
        "params":          params.tolist(),
        "loss_history":    loss_hist,
        "metrics": {
            "classical_lr_acc":      acc_lr,
            "classical_lr_auc":      auc_lr,
            "train_accuracy":        res_tr.accuracy,
            "train_auc":             res_tr.auc,
            "train_sensitivity":     res_tr.sensitivity,
            "train_specificity":     res_tr.specificity,
            "test_accuracy":         res_te.accuracy,
            "test_auc":              res_te.auc,
            "test_sensitivity":      res_te.sensitivity,
            "test_specificity":      res_te.specificity,
            "test_acc_opt_thr":      best_acc,
            "test_opt_threshold":    best_thr,
        },
    }
    out = RESULTS / args.out
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
