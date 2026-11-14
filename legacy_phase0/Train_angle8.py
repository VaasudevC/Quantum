"""Train the angle-encoded 8-qubit VQC with data re-uploading on PTB-XL.

Companion to Vqc_angle8.py.

Why angle encoding over amplitude encoding
-------------------------------------------
Amplitude encoding L2-normalises the full 251-feature vector, destroying the
absolute millivolt scale that makes ST-elevation and T-wave amplitude clinically
meaningful.  Angle encoding maps 8 hand-picked discriminative features directly
to Ry rotation angles via pi*tanh(z), preserving sign and relative magnitude
through the quantum circuit without any normalisation loss.

Selected features (by |MI_mean - NORM_mean| / pooled_std on full PTB-XL):
    V4_t_inverted       sep=0.735   T-wave inversion in V4
    I_st_dev_j60        sep=0.716   ST deviation at J+60ms in lead I
    aVL_st_dev_j60      sep=0.712   ST deviation at J+60ms in aVL
    II_t_inverted       sep=0.711   T-wave inversion in II
    V3_t_inverted       sep=0.582   T-wave inversion in V3
    max_st_depression   sep=0.577   worst ST depression across leads
    V3_t_amp            sep=0.540   T-wave amplitude V3 (lower in MI)
    aVF_t_inverted      sep=0.497   T-wave inversion in aVF

Data re-uploading
    Re-applies encoding Ry gates at the start of every HEA layer.
    Expressibility grows with depth; PSR gradient is still exact.

CLI flags:
    --steps N        Adam gradient steps (default 200)
    --batch-size N   samples per PSR gradient eval (default 500)
    --lr F           Adam learning rate (default 0.02)
    --seed N         RNG seed (default 42)
    --warm-start     load results/vqc_angle8_params.json as init
    --out NAME       JSON filename in results/ (default vqc_angle8_params.json)

Run:
    python Train_angle8.py
    python Train_angle8.py --steps 150 --batch-size 500 --lr 0.02
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Vqc_angle8 import (
    ANGLE_FEATURES,
    N_ANSATZ_PARAMS,
    N_FEATURES_TOTAL,
    N_LAYERS,
    N_PARAMS,
    N_QUBITS,
    AngleScaler,
    EvalResult,
    compute_angles,
    evaluate,
    predict_proba,
    train_adam,
)

CSV     = Path(__file__).parent / "PTB_XL" / "st_features_plus_TR.csv"
RESULTS = Path(__file__).parent / "results"


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    df = df[df["label_class"].isin(["MI", "NORM"])].copy()
    y  = (df["label_class"] == "MI").astype(np.int64).to_numpy()

    # Verify all 8 selected features are present
    missing = [f for f in ANGLE_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing {len(missing)} required features: {missing}\n"
            f"Available columns (first 30): {list(df.columns[:30])}"
        )

    X = df[ANGLE_FEATURES].astype(np.float64).to_numpy()

    # Replace NaN / inf with column median
    if not np.isfinite(X).all():
        col_med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        idx = ~np.isfinite(X)
        X[idx] = np.take(col_med, np.where(idx)[1])

    return X, y, ANGLE_FEATURES


def compute_class_weight(y: np.ndarray) -> np.ndarray:
    n      = len(y)
    n_mi   = int(y.sum())
    n_norm = n - n_mi
    return np.array([n / (2.0 * n_norm), n / (2.0 * n_mi)])   # [w_NORM, w_MI]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",      type=int,   default=200)
    ap.add_argument("--batch-size", type=int,   default=500)
    ap.add_argument("--lr",         type=float, default=0.02)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--out",        type=str,   default="vqc_angle8_params.json")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("  Angle-Encoded VQC on 8 qubits with data re-uploading")
    print("=" * 72)
    print(f"Loading  {CSV}")
    X, y, cols = load_dataset(CSV)
    print(f"  rows          : {X.shape[0]:,}")
    print(f"  features used : {X.shape[1]}  ({N_QUBITS} per layer x {N_LAYERS} layers)")
    print(f"  MI vs NORM    : {int(y.sum()):,} / {int((1 - y).sum()):,}")
    print()
    print("  Feature loading (sequential, 8 per layer):")
    for L in range(N_LAYERS):
        print(f"    Layer {L}: {', '.join(ANGLE_FEATURES[L*N_QUBITS:(L+1)*N_QUBITS])}")

    # 80/20 stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    print(f"\n  train: {len(y_tr):,}   test: {len(y_te):,}")

    # Fit scaler on train only
    scaler = AngleScaler.fit(X_tr)
    X_tr_std = scaler.transform(X_tr)
    X_te_std = scaler.transform(X_te)

    angles_tr = compute_angles(X_tr_std)     # (n_train, N_QUBITS), in (-pi, pi)
    angles_te = compute_angles(X_te_std)

    print("\n  Angle encoding sanity check:")
    print(f"    angles_tr  shape={angles_tr.shape}  "
          f"min={angles_tr.min():.3f}  max={angles_tr.max():.3f}")
    print(f"    angles_te  shape={angles_te.shape}")

    # Per-feature angle stats (verify signal is present)
    print("\n  Per-feature angle stats (MI vs NORM mean angles):")
    mi_mask   = y_tr == 1
    norm_mask = ~mi_mask
    for L in range(N_LAYERS):
        print(f"  -- Layer {L} --")
        for i in range(N_QUBITS):
            fi    = L * N_QUBITS + i
            fname = ANGLE_FEATURES[fi]
            mi_mean   = float(angles_tr[mi_mask,   fi].mean())
            norm_mean = float(angles_tr[norm_mask,  fi].mean())
            sep       = abs(mi_mean - norm_mean)
            print(f"    {fname:<30s}  MI={mi_mean:+.3f}  NORM={norm_mean:+.3f}  sep={sep:.3f}")

    # Class weights
    cw = compute_class_weight(y_tr)
    print(f"\n  class weights : NORM={cw[0]:.3f}  MI={cw[1]:.3f}  "
          f"(MI/NORM ratio = {int(y_tr.sum()):,}/{int((1-y_tr).sum()):,})")

    # Warm-start
    init_params = None
    if args.warm_start:
        pf = RESULTS / args.out
        if pf.exists():
            saved = json.loads(pf.read_text())
            init_params = np.array(saved["params"])
            prev = saved.get("metrics", {})
            print(f"  warm-start: loaded {len(init_params)} params  "
                  f"(prev test_AUC={prev.get('test_auc','?'):.4f}  "
                  f"test_acc={prev.get('test_accuracy','?'):.4f})")
        else:
            print("  (--warm-start: no saved params found, using random init)")

    # ---------- Train ----------
    n_epochs_approx = args.steps * args.batch_size / len(y_tr)
    print(f"\nTraining VQC  N_QUBITS={N_QUBITS}  N_LAYERS={N_LAYERS}  "
          f"N_FEATURES={N_FEATURES_TOTAL}  "
          f"N_PARAMS={N_PARAMS} ({N_ANSATZ_PARAMS} circuit + {N_PARAMS-N_ANSATZ_PARAMS} readout)")
    print("  gradient:   parameter shift rule (exact, hardware-portable)")
    print("  encoding:   sequential -- 8 different features loaded per layer")
    print(f"  optimizer:  Adam   steps={args.steps}  batch={args.batch_size}  "
          f"lr={args.lr}   ~{n_epochs_approx:.1f} effective epochs")

    t0 = time.time()
    loss_hist: list[float] = []

    params = train_adam(
        angles_train=angles_tr,
        y_train=y_tr,
        n_steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        l2=1e-4,
        seed=args.seed,
        verbose=True,
        loss_history=loss_hist,
        class_weight=cw,
        init_params=init_params,
    )

    wall = time.time() - t0
    print(f"  wall time: {wall:.1f}s  ({wall/3600:.2f}h)")

    # ---------- Evaluate ----------
    res_tr = evaluate(angles_tr, y_tr, params)
    res_te = evaluate(angles_te, y_te, params)

    print("\n" + "-" * 72)
    print("  Results")
    print("-" * 72)
    print(f"  Train  n={res_tr.n:,}   accuracy={res_tr.accuracy:.4f}   AUC={res_tr.auc:.4f}")
    print(f"  Test   n={res_te.n:,}   accuracy={res_te.accuracy:.4f}   AUC={res_te.auc:.4f}")
    print("-" * 72)

    target_hit = "YES -- target reached!" if res_te.accuracy >= 0.90 else \
                 f"NO  -- gap = {0.90 - res_te.accuracy:.4f}"
    print(f"  >= 90% accuracy target: {target_hit}")

    # ---------- Per-class breakdown ----------
    p_te = predict_proba(angles_te, params)
    pred_te = (p_te >= 0.5).astype(int)
    tp = int(((pred_te == 1) & (y_te == 1)).sum())
    fn = int(((pred_te == 0) & (y_te == 1)).sum())
    tn = int(((pred_te == 0) & (y_te == 0)).sum())
    fp = int(((pred_te == 1) & (y_te == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"\n  Confusion matrix (test):")
    print(f"    TP={tp:4d}  FN={fn:4d}  sensitivity={sensitivity:.4f} (recall)")
    print(f"    FP={fp:4d}  TN={tn:4d}  specificity={specificity:.4f}")

    # ---------- Loss curve ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
        ax.plot(range(1, len(loss_hist) + 1), loss_hist,
                marker="o", ms=3, lw=1.2, color="steelblue")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("Binary cross-entropy (batch)")
        ax.set_title("VQC-Angle8 training loss (data re-uploading)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png = RESULTS / "vqc_angle8_training.png"
        fig.savefig(png)
        plt.close(fig)
        print(f"\nSaved loss curve -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    # ---------- Save ----------
    summary = {
        "encoding":           "angle_sequential",
        "sequential_loading": True,
        "n_qubits":           N_QUBITS,
        "n_layers":           N_LAYERS,
        "n_ansatz_params":    N_ANSATZ_PARAMS,
        "n_readout_params":   N_PARAMS - N_ANSATZ_PARAMS,
        "n_params_total":     N_PARAMS,
        "gradient":           "parameter_shift_rule",
        "optimizer":          "adam",
        "steps":              args.steps,
        "batch_size":         args.batch_size,
        "lr":                 args.lr,
        "class_weight":       {"NORM": float(cw[0]), "MI": float(cw[1])},
        "csv":                str(CSV),
        "train_rows":         int(len(y_tr)),
        "test_rows":          int(len(y_te)),
        "mi_train":           int(y_tr.sum()),
        "norm_train":         int((1 - y_tr).sum()),
        "wall_time_seconds":  round(wall, 2),
        "feature_columns":    cols,
        "angle_features":     ANGLE_FEATURES,
        "scaler": {
            "mean": scaler.mean.tolist(),
            "std":  scaler.std.tolist(),
        },
        "params":       params.tolist(),
        "loss_history": loss_hist,
        "metrics": {
            "train_accuracy": res_tr.accuracy,
            "train_auc":      res_tr.auc,
            "test_accuracy":  res_te.accuracy,
            "test_auc":       res_te.auc,
            "sensitivity":    sensitivity,
            "specificity":    specificity,
        },
    }
    out_path = RESULTS / args.out
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved trained parameters -> {out_path}")


if __name__ == "__main__":
    main()
