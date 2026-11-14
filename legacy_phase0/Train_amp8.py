"""Train the amplitude-encoded 8-qubit VQC on PTB_XL/st_features_plus_TR.csv.

Companion to `Vqc_amp8.py`. Loads the ST + T/R/Q feature CSV, performs MI vs
NORM binary classification, and prints train/test metrics.

CLI flags:
    --subset N        : use only N training rows (fast demo mode)
    --iter N          : number of L-BFGS-B iterations   (default 200)
    --seed N          : RNG seed                        (default 42)

Run:
    python Train_amp8.py
    python Train_amp8.py --subset 3000 --iter 60
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Vqc_amp8 import (
    DIM,
    N_ANSATZ_PARAMS,
    N_PARAMS,
    N_QUBITS,
    Standardizer,
    encode_amplitudes,
    encode_amplitudes_with_scale,
    evaluate,
    predict_proba,
    train,
    train_adam,
)


CSV     = Path(__file__).parent / "PTB_XL" / "st_features_plus_TR.csv"
RESULTS = Path(__file__).parent / "results"


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)

    # Binary label: MI = 1, NORM = 0. Drop any rows that are neither.
    df = df[df["label_class"].isin(["MI", "NORM"])].copy()
    y = (df["label_class"] == "MI").astype(np.int64).to_numpy()

    # Numeric matrix, minus the ID column and the slope categorical duplicates
    drop_cols = {"ecg_id"}
    feature_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in drop_cols
    ]
    X = df[feature_cols].astype(np.float64).to_numpy()

    # Any NaN or inf?  Replace with column median then 0.
    if not np.isfinite(X).all():
        col_med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        idx = ~np.isfinite(X)
        X[idx] = np.take(col_med, np.where(idx)[1])

    return X, y, feature_cols


def compute_class_weight(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency balanced weights: [w_NORM, w_MI].

    Balanced formula: w_c = n_total / (n_classes * n_c).
    With two classes and PTB-XL: w_MI ≈ 1.37, w_NORM ≈ 0.79.
    The weighted mean over the dataset equals 1 by construction.
    """
    n       = len(y)
    n_mi    = int(y.sum())
    n_norm  = n - n_mi
    w_mi    = n / (2.0 * n_mi)
    w_norm  = n / (2.0 * n_norm)
    return np.array([w_norm, w_mi])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0,
                    help="if > 0, subsample the training set to this many rows")
    ap.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam",
                    help="adam (mini-batch, full dataset) or lbfgs (full-batch subset)")
    ap.add_argument("--steps", type=int, default=200,
                    help="Adam: number of gradient steps (default 200)")
    ap.add_argument("--batch-size", type=int, default=300,
                    help="Adam: mini-batch size per gradient call (default 300)")
    ap.add_argument("--lr", type=float, default=0.02,
                    help="Adam: learning rate (default 0.02)")
    ap.add_argument("--iter", type=int, default=200,
                    help="L-BFGS-B: max iterations (default 200)")
    ap.add_argument("--warm-start", action="store_true",
                    help="load saved params from results/vqc_amp8_params.json as init")
    ap.add_argument("--scale-feature", action="store_true",
                    help="append log(1+||z||) scale feature before L2-norm")
    ap.add_argument("--out", type=str, default="vqc_amp8_params.json",
                    help="output JSON filename inside results/ (default: vqc_amp8_params.json)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("  Amplitude-Encoded VQC on 8 qubits (2**8 = 256 amplitude slots)")
    print("=" * 72)
    print(f"Loading  {CSV}")
    X, y, cols = load_dataset(CSV)
    print(f"  rows       : {X.shape[0]:,}")
    print(f"  features   : {X.shape[1]}  (pad to {DIM} for amplitude encoding)")
    print(f"  MI vs NORM : {int(y.sum()):,} / {int((1 - y).sum()):,}")

    # 80/20 stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    # Optional demo subsample of the TRAIN split
    if args.subset and args.subset < len(y_tr):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(y_tr), args.subset, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        print(f"  ** subset  : training on {args.subset:,} rows for a fast demo **")

    print(f"  train      : {len(y_tr):,}   test : {len(y_te):,}")

    # Class weights from training split only (never look at test labels)
    cw = compute_class_weight(y_tr)
    print(f"  class weights : NORM={cw[0]:.3f}  MI={cw[1]:.3f}  "
          f"(MI/NORM ratio = {int(y_tr.sum()):,}/{int((1-y_tr).sum()):,})")

    # Warm-start: load previously saved params
    import json as _json
    init_params = None
    if args.warm_start:
        pf = RESULTS / "vqc_amp8_params.json"
        if pf.exists():
            saved = _json.loads(pf.read_text())
            init_params = np.array(saved["params"])
            print(f"  warm-start: loaded {len(init_params)} params  "
                  f"(prev test_AUC={saved['metrics']['test_auc']:.4f}  "
                  f"test_acc={saved['metrics']['test_accuracy']:.4f})")
        else:
            print("  (--warm-start: no saved params found, using random init)")

    # Fit standardizer on train only, then encode
    print("\nEncoding amplitudes ...")
    scaler = Standardizer.fit(X_tr)
    if args.scale_feature:
        print("  encoding: amplitude + log-scale feature (preserves magnitude info)")
        psi_tr = encode_amplitudes_with_scale(scaler.transform(X_tr))
        psi_te = encode_amplitudes_with_scale(scaler.transform(X_te))
    else:
        psi_tr = encode_amplitudes(scaler.transform(X_tr))
        psi_te = encode_amplitudes(scaler.transform(X_te))
    print(f"  train state tensor : {psi_tr.shape}   ||psi||_2 = 1  [ok]")
    print(f"  test  state tensor : {psi_te.shape}")

    tr_norm_err = float(np.max(np.abs(np.linalg.norm(psi_tr, axis=1) - 1.0)))
    te_norm_err = float(np.max(np.abs(np.linalg.norm(psi_te, axis=1) - 1.0)))
    print(f"  max norm error     : train {tr_norm_err:.2e}  test {te_norm_err:.2e}")

    # ---------- Train ----------
    print(f"\nTraining VQC   N_QUBITS={N_QUBITS}  N_PARAMS={N_PARAMS} "
          f"(circuit={N_ANSATZ_PARAMS} + readout={N_PARAMS-N_ANSATZ_PARAMS})")
    print("  gradient: parameter shift rule (exact, hardware-portable)")
    t0 = time.time()
    loss_hist: list[float] = []

    if args.optimizer == "adam":
        n_epochs_approx = args.steps * args.batch_size / len(y_tr)
        print(f"  optimizer : Adam   steps={args.steps}   "
              f"batch={args.batch_size}   lr={args.lr}   "
              f"≈{n_epochs_approx:.1f} effective epochs")
        params = train_adam(
            psi_tr, y_tr,
            n_steps=args.steps, lr=args.lr, batch_size=args.batch_size,
            l2=1e-4, seed=args.seed,
            verbose=True, loss_history=loss_hist,
            class_weight=cw,
            init_params=init_params,
        )
    else:
        print(f"  optimizer : L-BFGS-B   max_iter={args.iter}")
        params = train(
            psi_tr, y_tr,
            n_iter=args.iter, l2=1e-4, seed=args.seed,
            verbose=True, loss_history=loss_hist,
            class_weight=cw,
        )

    wall = time.time() - t0
    print(f"  wall time : {wall:.1f}s")

    # ---------- Evaluate ----------
    res_tr = evaluate(psi_tr, y_tr, params)
    res_te = evaluate(psi_te, y_te, params)

    print("\n" + "-" * 72)
    print("  Results")
    print("-" * 72)
    print(f"  Train  n={res_tr.n:,}   accuracy={res_tr.accuracy:.4f}   AUC={res_tr.auc:.4f}")
    print(f"  Test   n={res_te.n:,}   accuracy={res_te.accuracy:.4f}   AUC={res_te.auc:.4f}")
    print("-" * 72)

    # ---------- Plot training loss ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
        ax.plot(range(1, len(loss_hist) + 1), loss_hist, marker="o", ms=3, lw=1.2)
        ax.set_xlabel("L-BFGS-B iteration")
        ax.set_ylabel("Binary cross-entropy")
        ax.set_title("VQC-Amp8 training loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        loss_png = RESULTS / "vqc_amp8_training.png"
        fig.savefig(loss_png)
        plt.close(fig)
        print(f"Saved loss curve -> {loss_png}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    # ---------- Save trained params + summary ----------
    import json

    summary = {
        "encoding": "amplitude",
        "n_qubits": N_QUBITS,
        "n_layers": 4,
        "n_ansatz_params": N_ANSATZ_PARAMS,
        "n_readout_params": N_PARAMS - N_ANSATZ_PARAMS,
        "n_params_total": N_PARAMS,
        "dim": DIM,
        "gradient": "parameter_shift_rule",
        "optimizer": args.optimizer,
        "scale_feature": args.scale_feature,
        "warm_start": args.warm_start,
        "class_weight": {"NORM": float(cw[0]), "MI": float(cw[1])},
        "csv": str(CSV),
        "train_rows": int(len(y_tr)),
        "test_rows": int(len(y_te)),
        "mi_train": int(y_tr.sum()),
        "norm_train": int((1 - y_tr).sum()),
        "iterations": len(loss_hist),
        "wall_time_seconds": round(wall, 2),
        "feature_columns": cols,
        "standardizer": {
            "mean": scaler.mean.tolist(),
            "std": scaler.std.tolist(),
            "clip": scaler.clip,
        },
        "params": params.tolist(),
        "loss_history": loss_hist,
        "metrics": {
            "train_accuracy": res_tr.accuracy,
            "train_auc": res_tr.auc,
            "test_accuracy": res_te.accuracy,
            "test_auc": res_te.auc,
        },
    }
    out = RESULTS / args.out
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved trained parameters -> {out}")


if __name__ == "__main__":
    main()
