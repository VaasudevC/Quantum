"""Train the quantum-classical hybrid VQC (angle encoding + MLP readout).

Uses the same sequential angle encoding as Vqc_angle8 / Train_angle8 but
replaces the 9-param linear readout with a 32-hidden-unit MLP (321 params).

Why this might reach 90%
------------------------
Run 6 (sequential angle, linear readout): test_acc=80.45%, AUC=0.8699
Classical LR ceiling (32 features):        87.30%,          AUC=0.9368

The linear readout forces the 8 quantum expectations to be linearly separable.
The MLP can learn: combination(combination(quantum_features)).  This mirrors
what classical non-linear classifiers do, but the FEATURES come from the
quantum circuit (non-linear transformations of 32 clinical measurements).

Warm-start
----------
By default loads the quantum ansatz params from results/vqc_angle32_params.json
(Run 6). The quantum circuit already encodes useful clinical patterns; only the
MLP readout is trained from scratch. Use --no-warm-start to train from random.

CLI flags:
    --steps N        Adam steps (default 300)
    --batch-size N   batch per PSR gradient (default 500)
    --lr F           Adam learning rate (default 0.01)
    --seed N         RNG seed (default 42)
    --no-warm-start  skip loading Run 6 quantum params
    --out NAME       result JSON filename in results/ (default vqc_hybrid8_params.json)

Run:
    python Train_hybrid8.py
    python Train_hybrid8.py --steps 300 --lr 0.01
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Vqc_angle8  import ANGLE_FEATURES, N_FEATURES_TOTAL, N_LAYERS, N_QUBITS, AngleScaler, compute_angles
from Vqc_hybrid8 import (
    N_ANSATZ_PARAMS,
    N_MLP_PARAMS,
    N_PARAMS,
    N_HIDDEN,
    evaluate,
    predict_proba,
    train_adam,
)

CSV     = Path(__file__).parent / "PTB_XL" / "st_features_plus_TR.csv"
RESULTS = Path(__file__).parent / "results"


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df[df["label_class"].isin(["MI", "NORM"])].copy()
    y  = (df["label_class"] == "MI").astype(np.int64).to_numpy()
    missing = [f for f in ANGLE_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = df[ANGLE_FEATURES].astype(np.float64).to_numpy()
    if not np.isfinite(X).all():
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        idx = ~np.isfinite(X)
        X[idx] = np.take(med, np.where(idx)[1])
    return X, y


def compute_class_weight(y: np.ndarray) -> np.ndarray:
    n = len(y); nm = int(y.sum()); nn = n - nm
    return np.array([n / (2.0 * nn), n / (2.0 * nm)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",          type=int,   default=300)
    ap.add_argument("--batch-size",     type=int,   default=500)
    ap.add_argument("--lr",             type=float, default=0.01)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--no-warm-start",  action="store_true")
    ap.add_argument("--out",            type=str,   default="vqc_hybrid8_params.json")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("  Hybrid VQC: sequential angle encoding (32 feat) + MLP readout")
    print("=" * 72)
    print(f"  N_PARAMS = {N_ANSATZ_PARAMS} circuit + {N_MLP_PARAMS} MLP = {N_PARAMS} total")
    print(f"  MLP: {N_QUBITS} -> {N_HIDDEN} -> 1  (ReLU hidden layer)")

    print(f"\nLoading  {CSV}")
    X, y = load_dataset(CSV)
    print(f"  rows   : {X.shape[0]:,}   MI: {int(y.sum()):,}   NORM: {int((1-y).sum()):,}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    print(f"  train  : {len(ytr):,}   test: {len(yte):,}")

    scaler   = AngleScaler.fit(Xtr)
    ang_tr   = compute_angles(scaler.transform(Xtr))
    ang_te   = compute_angles(scaler.transform(Xte))

    cw = compute_class_weight(ytr)
    print(f"  weights: NORM={cw[0]:.3f}  MI={cw[1]:.3f}")

    # Warm-start: load quantum ansatz from Run 6 (sequential angle VQC)
    init_quantum = None
    if not args.no_warm_start:
        prev_path = RESULTS / "vqc_angle32_params.json"
        if prev_path.exists():
            prev = json.loads(prev_path.read_text())
            prev_params = np.array(prev["params"])
            # params[:N_ANSATZ_PARAMS] are the quantum circuit params
            init_quantum = prev_params[:N_ANSATZ_PARAMS]
            pm = prev.get("metrics", {})
            print(f"\n  Warm-start: loaded quantum params from {prev_path.name}")
            print(f"  (Run 6 test_acc={pm.get('test_accuracy','?'):.4f}  "
                  f"AUC={pm.get('test_auc','?'):.4f})")
            print("  MLP readout: Xavier random init")
        else:
            print(f"\n  (no warm-start: {prev_path.name} not found)")

    n_ep = args.steps * args.batch_size / len(ytr)
    print(f"\nTraining  steps={args.steps}  batch={args.batch_size}  "
          f"lr={args.lr}  ~{n_ep:.1f} effective epochs")
    print("  Quantum gradient: PSR (exact)")
    print("  MLP gradient:     backpropagation")

    t0         = time.time()
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

    res_tr = evaluate(ang_tr, ytr, params)
    res_te = evaluate(ang_te, yte, params)

    print("\n" + "-" * 72)
    print("  Results")
    print("-" * 72)
    print(f"  Train  n={res_tr.n:,}   acc={res_tr.accuracy:.4f}  "
          f"AUC={res_tr.auc:.4f}  sens={res_tr.sensitivity:.4f}  spec={res_tr.specificity:.4f}")
    print(f"  Test   n={res_te.n:,}   acc={res_te.accuracy:.4f}  "
          f"AUC={res_te.auc:.4f}  sens={res_te.sensitivity:.4f}  spec={res_te.specificity:.4f}")
    print("-" * 72)
    target = "YES -- target reached!" if res_te.accuracy >= 0.90 else \
             f"NO  -- gap = {0.90 - res_te.accuracy:.4f}"
    print(f"  >= 90% accuracy: {target}")

    # Save
    summary = {
        "encoding":          "angle_sequential",
        "readout":           "mlp_32hidden",
        "n_qubits":          N_QUBITS,
        "n_layers":          N_LAYERS,
        "n_features":        N_FEATURES_TOTAL,
        "n_ansatz_params":   N_ANSATZ_PARAMS,
        "n_mlp_params":      N_MLP_PARAMS,
        "n_params_total":    N_PARAMS,
        "steps":             args.steps,
        "batch_size":        args.batch_size,
        "lr":                args.lr,
        "warm_start":        (init_quantum is not None),
        "angle_features":    ANGLE_FEATURES,
        "class_weight":      {"NORM": float(cw[0]), "MI": float(cw[1])},
        "wall_time_seconds": round(wall, 2),
        "scaler":            {"mean": scaler.mean.tolist(), "std": scaler.std.tolist()},
        "params":            params.tolist(),
        "loss_history":      loss_hist,
        "metrics": {
            "train_accuracy":    res_tr.accuracy,
            "train_auc":         res_tr.auc,
            "train_sensitivity": res_tr.sensitivity,
            "train_specificity": res_tr.specificity,
            "test_accuracy":     res_te.accuracy,
            "test_auc":          res_te.auc,
            "test_sensitivity":  res_te.sensitivity,
            "test_specificity":  res_te.specificity,
        },
    }
    out = RESULTS / args.out
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
