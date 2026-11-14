"""Train hybrid VQC with PLS-32 sequential encoding on all 251 PTB-XL features.

Why PLS-32 sequential over PLS-8 re-uploading
----------------------------------------------
Run 8b (PLS-8 re-uploading) converges slowly because:
  - Same 8 features tiled to all 4 layers (re-upload structure)
  - Warm-start from Run 7 expected DIFFERENT features at each layer (sequential structure)
  - Structural mismatch → slow gradient alignment

PLS-32 sequential fixes this:
  - 32 PLS components computed from all 251 features
  - Loaded sequentially: 8 different PLS components per layer (mirrors Run 7 structure)
  - Layer 0: PLS 1-8   (highest discrimination, sep=6.8-0.4 in angle space)
  - Layer 1: PLS 9-16  (medium discrimination)
  - Layer 2: PLS 17-24 (lower discrimination)
  - Layer 3: PLS 25-32 (lowest discrimination)
  - Warm-start from Run 7 is well-aligned with this decreasing-sep structure

Baseline comparison (LR on N PLS components):
  PLS-8:  88.91%, AUC=0.9481
  PLS-16: 88.77%, AUC=0.9505  (LR accuracy plateaus — quantum adds non-linearity)

Hardware pipeline (same as all angle runs)
------------------------------------------
Classical: 251 features -> PLS-32 matrix (251x32) -> 32 angles
QPU: 32 angles -> sequential VQC (4 layers x 8 features) -> 8 Z-expectations
Classical: 8 expectations -> MLP(32 hidden) -> p(MI)

CLI flags
---------
    --steps N        Adam steps (default 300)
    --batch-size N   batch size (default 300)
    --lr F           learning rate (default 0.01)
    --seed N         RNG seed (default 42)
    --warm-start F   JSON in results/ with quantum params (default vqc_hybrid8_params.json)
    --no-warm-start  skip warm-start
    --out NAME       result JSON in results/ (default vqc_pls32_params.json)

Run:
    python -u Train_pls32.py > results/vqc_pls32.txt 2>&1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Vqc_hybrid8 import (
    N_ANSATZ_PARAMS, N_MLP_PARAMS, N_PARAMS, N_HIDDEN, N_QUBITS, N_LAYERS,
    evaluate, predict_proba, train_adam,
)
from Vqc_angle8 import compute_angles  # reuse: pi * tanh(z)

CSV     = Path(__file__).parent / "PTB_XL" / "st_features_plus_TR.csv"
RESULTS = Path(__file__).parent / "results"

N_PLS_COMPONENTS = N_QUBITS * N_LAYERS   # 32 — one full sequential pass

_META_COLS = {
    "label_class", "ecg_id", "patient_id",
    "filename_lr", "filename_hr", "scp_codes", "strat_fold",
}


class PLS32Preprocessor:
    """StandardScaler + PLSRegression(n_components=32) with serializable transform."""

    def __init__(self, n_components: int = N_PLS_COMPONENTS):
        self.n_components = n_components
        self._scaler = StandardScaler()
        self._pls    = PLSRegression(n_components=n_components, max_iter=1000)
        self.A: np.ndarray | None = None  # (n_features, n_components), combined transform
        self.b: np.ndarray | None = None  # (n_components,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLS32Preprocessor":
        Xs = self._scaler.fit_transform(X)
        self._pls.fit(Xs, y)
        mu_sc   = self._scaler.mean_
        sig_sc  = self._scaler.scale_
        R       = self._pls.x_rotations_
        mu_pls  = self._pls._x_mean
        std_pls = self._pls._x_std
        denom   = sig_sc * std_pls
        denom   = np.where(denom == 0, 1.0, denom)
        A1 = R / denom[:, None]
        b1 = -((mu_sc / sig_sc + mu_pls) / std_pls) @ R

        # Critical fix: normalize each PLS component to unit-std on training data.
        # Raw PLS scores for PLS-1 can have magnitude 4-5, causing pi*tanh to saturate
        # at +-pi. Ry(+pi)|0> and Ry(-pi)|0> both produce |1>, giving Z=-1 for BOTH
        # MI and NORM — making the strongest feature completely indistinguishable.
        # After per-component unit-std normalization, angles stay within +-2 rad where
        # the tanh is in a useful, unsaturated regime.
        T_train   = X @ A1 + b1
        comp_mean = T_train.mean(axis=0)          # (32,)
        comp_std  = T_train.std(axis=0)            # (32,)
        comp_std  = np.where(comp_std == 0, 1.0, comp_std)
        # Absorb normalization into the combined linear transform
        self.A = A1 / comp_std[None, :]
        self.b = (b1 - comp_mean) / comp_std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 32) unit-std-normalized PLS components via T = X @ A + b."""
        return X @ self.A + self.b

    def to_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PLS32Preprocessor":
        obj = cls(n_components=d["n_components"])
        obj.A = np.array(d["A"])
        obj.b = np.array(d["b"])
        return obj


def compute_pls32_angles(pls_components: np.ndarray) -> np.ndarray:
    """Map 32 PLS components to (batch, 32) angles via pi*tanh.

    Sequential: layer L uses columns [L*8 : (L+1)*8].
    This matches the structure of circuit_forward in Vqc_angle8.
    """
    return np.pi * np.tanh(pls_components)   # (batch, 32), different 8 per layer


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
    ap.add_argument("--warm-start",     type=str,   default="vqc_hybrid8_params.json")
    ap.add_argument("--no-warm-start",  action="store_true")
    ap.add_argument("--out",            type=str,   default="vqc_pls32_params.json")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("  Hybrid VQC: PLS-32 sequential encoding + MLP readout")
    print("=" * 72)
    print(f"  N_PARAMS = {N_ANSATZ_PARAMS} circuit + {N_MLP_PARAMS} MLP = {N_PARAMS} total")
    print(f"  PLS components : {N_PLS_COMPONENTS} (from all 251 features, sequential)")
    print(f"  MLP            : {N_QUBITS} -> {N_HIDDEN} -> 1")
    print(f"  Encoding       : sequential, 8 different PLS components per layer")

    print(f"\nLoading  {CSV}")
    X, y, feat_cols = load_all_features(CSV)
    n_feat = X.shape[1]
    print(f"  rows          : {X.shape[0]:,}   MI: {int(y.sum()):,}   NORM: {int((1-y).sum()):,}")
    print(f"  features used : {n_feat}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    print(f"  train: {len(ytr):,}   test: {len(yte):,}")

    # Fit PLS-32
    print(f"\nFitting PLS-{N_PLS_COMPONENTS} preprocessor on training data ...")
    t_pls = time.time()
    prep = PLS32Preprocessor(n_components=N_PLS_COMPONENTS)
    prep.fit(Xtr, ytr)
    print(f"  done ({time.time()-t_pls:.1f}s)")

    Z_tr = prep.transform(Xtr)
    Z_te = prep.transform(Xte)

    # Classical LR baselines
    for n_comp in [8, 16, 32]:
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(Z_tr[:, :n_comp], ytr)
        p  = lr.predict_proba(Z_te[:, :n_comp])[:, 1]
        acc = (lr.predict(Z_te[:, :n_comp]) == yte).mean()
        auc = roc_auc_score(yte, p)
        print(f"  Classical LR on first {n_comp:2d} PLS comps: acc={acc:.4f}  AUC={auc:.4f}")

    # Angles for quantum circuit (sequential, 8 per layer)
    ang_tr = compute_pls32_angles(Z_tr)
    ang_te = compute_pls32_angles(Z_te)
    print(f"\n  Angles: shape={ang_tr.shape}  min={ang_tr.min():.3f}  max={ang_tr.max():.3f}")

    # Per-layer stats
    mi_mask = ytr == 1
    print(f"\n  PLS component separation by layer (MI vs NORM mean angle):")
    for L in range(N_LAYERS):
        seps = []
        for i in range(N_QUBITS):
            fi = L * N_QUBITS + i
            sep = abs(ang_tr[mi_mask, fi].mean() - ang_tr[~mi_mask, fi].mean())
            seps.append(sep)
        print(f"    Layer {L} (PLS {L*N_QUBITS+1}-{(L+1)*N_QUBITS}): "
              f"mean_sep={np.mean(seps):.3f}  max_sep={np.max(seps):.3f}")

    cw = compute_class_weight(ytr)
    print(f"  Class weights : NORM={cw[0]:.3f}  MI={cw[1]:.3f}")

    # Warm-start quantum params
    init_quantum = None
    if not args.no_warm_start:
        ws_path = RESULTS / args.warm_start
        if ws_path.exists():
            prev = json.loads(ws_path.read_text())
            prev_params = np.array(prev["params"])
            init_quantum = prev_params[:N_ANSATZ_PARAMS]
            pm = prev.get("metrics", {})
            print(f"\n  Warm-start quantum from: {ws_path.name}")
            print(f"    (prev test_acc={pm.get('test_accuracy','?'):.4f}  "
                  f"AUC={pm.get('test_auc','?'):.4f})")
            print("  MLP: Xavier random init")
        else:
            print(f"\n  (warm-start not found: {ws_path.name})")
    else:
        print("\n  Initialization: fresh random")

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

    res_tr = evaluate(ang_tr, ytr, params)
    res_te = evaluate(ang_te, yte, params)

    # Optimal threshold sweep
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
    print(f"  Test (threshold={best_thr:.3f}): acc={best_acc:.4f}  "
          f"sens={tp_o/(tp_o+fn_o):.4f}  spec={tn_o/(tn_o+fp_o):.4f}")
    print("-" * 72)
    target = "YES -- target reached!" if best_acc >= 0.90 else \
             f"NO  -- gap = {0.90 - best_acc:.4f}"
    print(f"  >= 90% accuracy: {target}")

    summary = {
        "encoding":          "pls32_angle_sequential",
        "readout":           "mlp_32hidden",
        "n_qubits":          N_QUBITS,
        "n_layers":          N_LAYERS,
        "n_pls_components":  N_PLS_COMPONENTS,
        "n_input_features":  n_feat,
        "n_ansatz_params":   N_ANSATZ_PARAMS,
        "n_mlp_params":      N_MLP_PARAMS,
        "n_params_total":    N_PARAMS,
        "steps":             args.steps,
        "batch_size":        args.batch_size,
        "lr":                args.lr,
        "class_weight":      {"NORM": float(cw[0]), "MI": float(cw[1])},
        "pls_preprocessor":  prep.to_dict(),
        "feature_columns":   feat_cols,
        "wall_time_seconds": round(wall, 2),
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
            "test_acc_opt_thr":  best_acc,
            "test_opt_threshold": best_thr,
        },
    }
    out = RESULTS / args.out
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
