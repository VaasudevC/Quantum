"""Quantum-classical hybrid VQC: sequential angle encoding + MLP readout.

Architecture
------------
Quantum circuit (same as Vqc_angle8, 72 ansatz params):
    Sequential angle encoding of 32 features (8 per layer, 4 layers).
    Each layer applies per-sample Ry gates from a different clinical group,
    followed by Ry+Rz+CNOT ansatz block.  Output: 8 Z-expectation values.

Classical MLP readout (replaces linear layer, N_MLP_PARAMS=321 params):
    z_all (batch, 8)  --[Linear(8->32)+ReLU]--> h1 --[Linear(32->1)+sigmoid]--> p

    W1: (8, 32)   b1: (32,)   W2: (32, 1)   b2: scalar
    N_MLP_PARAMS = 8*32 + 32 + 32*1 + 1 = 321
    N_PARAMS_TOTAL = 72 + 321 = 393

Why MLP readout over linear
---------------------------
Linear readout (`logit = w @ z_all + b`) forces the 8 quantum expectations to
be LINEARLY separable.  The quantum circuit can produce non-linear functions of
the input features, but the final classification is still linear in the quantum
output space.

A 2-layer MLP on the 8 expectations learns arbitrary combinations:
    p = sigma( W2 @ ReLU(W1 @ z_all + b1) + b2 )

This is equivalent to classifying with 32 derived features, where each derived
feature is a ReLU activation of a different linear combination of the 8 quantum
expectations.  The net result: up to 32 effective discriminant dimensions
extracted from the 8-qubit quantum state.

Hardware note
-------------
The quantum circuit (VQC part) runs on the QPU: produces 8 bitstrings per shot,
marginalised to 8 Z-expectation values (mean of many shots).  The MLP runs on
the classical CPU/GPU immediately after measurement -- no extra QPU cost.
The gradient through the quantum params uses PSR (hardware-exact); the MLP
gradient is standard backpropagation.

Param layout
------------
params[:N_ANSATZ_PARAMS]       = ansatz (72 circuit gates)
params[N_ANSATZ_PARAMS:]       = MLP params (321)
    [0   : N_QUBITS*N_HIDDEN]  = W1 flattened  (256)
    [256 : 256+N_HIDDEN]       = b1             (32)
    [288 : 288+N_HIDDEN]       = W2 flattened   (32)
    [320]                      = b2             (1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Re-use circuit primitives from Vqc_angle8
from Vqc_angle8 import (
    ANGLE_FEATURES,
    N_FEATURES_TOTAL,
    N_QUBITS,
    N_LAYERS,
    DIM,
    N_ANSATZ_PARAMS,
    AngleScaler,
    compute_angles,
    circuit_forward,
    expectation_all,
)

# ---------------------------------------------------------------------------
#  MLP readout dimensions
# ---------------------------------------------------------------------------
N_HIDDEN      = 32
N_MLP_W1      = N_QUBITS * N_HIDDEN       # 256
N_MLP_B1      = N_HIDDEN                  # 32
N_MLP_W2      = N_HIDDEN                  # 32  (32×1 flattened)
N_MLP_B2      = 1
N_MLP_PARAMS  = N_MLP_W1 + N_MLP_B1 + N_MLP_W2 + N_MLP_B2   # 321

N_PARAMS      = N_ANSATZ_PARAMS + N_MLP_PARAMS                 # 393


# ---------------------------------------------------------------------------
#  MLP helpers
# ---------------------------------------------------------------------------
def _unpack_mlp(mlp_params: np.ndarray):
    W1 = mlp_params[:N_MLP_W1].reshape(N_QUBITS, N_HIDDEN)            # (8, 32)
    b1 = mlp_params[N_MLP_W1 : N_MLP_W1 + N_MLP_B1]                   # (32,)
    W2 = mlp_params[N_MLP_W1+N_MLP_B1 : N_MLP_W1+N_MLP_B1+N_MLP_W2]  # (32,)
    b2 = mlp_params[-1]                                                  # scalar
    return W1, b1, W2, b2


def _mlp_forward(z_all: np.ndarray, mlp_params: np.ndarray) -> np.ndarray:
    """z_all (batch, 8) -> p (batch,)."""
    W1, b1, W2, b2 = _unpack_mlp(mlp_params)
    h1    = np.maximum(0.0, z_all @ W1 + b1)          # (batch, 32)  ReLU
    logit = h1 @ W2 + b2                               # (batch,)
    return 1.0 / (1.0 + np.exp(-logit))


def _mlp_backward(
    z_all:        np.ndarray,
    mlp_params:   np.ndarray,
    residual:     np.ndarray,
    class_weight: np.ndarray,
    y:            np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mlp_grad, dL_dz) where dL_dz flows to PSR gradient."""
    W1, b1, W2, b2 = _unpack_mlp(mlp_params)
    sw = np.where(y == 1, class_weight[1], class_weight[0])  # (batch,)
    wr = sw * residual                                         # weighted residual

    h1 = np.maximum(0.0, z_all @ W1 + b1)   # (batch, 32)

    # Backprop through W2, b2
    dL_dh1     = wr[:, None] * W2[None, :]                    # (batch, 32)
    dL_dpre    = dL_dh1 * (h1 > 0)                            # ReLU: (batch, 32)
    dW1        = (z_all[:, :, None] * dL_dpre[:, None, :]).mean(axis=0)  # (8, 32)
    db1        = dL_dpre.mean(axis=0)                          # (32,)
    dW2        = (h1 * wr[:, None]).mean(axis=0)               # (32,)
    db2        = float(wr.mean())

    mlp_grad = np.concatenate([dW1.ravel(), db1, dW2, [db2]])  # (321,)

    # Gradient w.r.t. z_all — passed to PSR loop
    dL_dz = dL_dpre @ W1.T                                     # (batch, 8)

    return mlp_grad, dL_dz


# ---------------------------------------------------------------------------
#  Forward pass
# ---------------------------------------------------------------------------
def predict_proba(angles: np.ndarray, params: np.ndarray) -> np.ndarray:
    """p(MI=1).

    angles: (batch, N_FEATURES_TOTAL=32)
    params: (N_PARAMS=393,)
    """
    state    = circuit_forward(angles, params[:N_ANSATZ_PARAMS])
    z_all    = expectation_all(state)
    mlp_p    = params[N_ANSATZ_PARAMS:]
    return _mlp_forward(z_all, mlp_p)


# ---------------------------------------------------------------------------
#  Loss
# ---------------------------------------------------------------------------
def _bce(
    params:       np.ndarray,
    angles:       np.ndarray,
    y:            np.ndarray,
    l2:           float,
    class_weight: Optional[np.ndarray] = None,
) -> float:
    p   = predict_proba(angles, params)
    p   = np.clip(p, 1e-7, 1.0 - 1e-7)
    bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if class_weight is not None:
        bce = bce * np.where(y == 1, class_weight[1], class_weight[0])
    loss = bce.mean()
    if l2 > 0.0:
        loss += l2 * (params[:N_ANSATZ_PARAMS] ** 2).mean()
    return float(loss)


# ---------------------------------------------------------------------------
#  Gradient: PSR for quantum params + backprop for MLP params
# ---------------------------------------------------------------------------
def _gradient(
    params:       np.ndarray,
    angles:       np.ndarray,
    y:            np.ndarray,
    l2:           float,
    class_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Combined PSR + MLP backprop gradient.

    For quantum params: PSR with dL/dz from MLP backprop (chain rule).
    For MLP params:    standard backprop (analytic, no circuit eval needed).
    """
    if class_weight is None:
        class_weight = np.ones(2, dtype=np.float64)

    grad = np.zeros(N_PARAMS)
    eps  = np.pi / 2.0

    state = circuit_forward(angles, params[:N_ANSATZ_PARAMS])
    z_all = expectation_all(state)                # (batch, 8)
    mlp_p = params[N_ANSATZ_PARAMS:]

    p        = np.clip(_mlp_forward(z_all, mlp_p), 1e-7, 1.0 - 1e-7)
    residual = p - y                               # (batch,)

    # MLP gradients (cheap: no circuit evals)
    mlp_grad, dL_dz = _mlp_backward(z_all, mlp_p, residual, class_weight, y)
    grad[N_ANSATZ_PARAMS:] = mlp_grad

    # Quantum ansatz gradients via PSR + chain rule through MLP
    for i in range(N_ANSATZ_PARAMS):
        ap, am = params.copy(), params.copy()
        ap[i] += eps;  am[i] -= eps
        z_p = expectation_all(circuit_forward(angles, ap[:N_ANSATZ_PARAMS]))
        z_m = expectation_all(circuit_forward(angles, am[:N_ANSATZ_PARAMS]))
        grad[i] = (dL_dz * (z_p - z_m) / 2.0).sum(axis=1).mean()

    if l2 > 0.0:
        grad[:N_ANSATZ_PARAMS] += 2.0 * l2 * params[:N_ANSATZ_PARAMS] / N_ANSATZ_PARAMS

    return grad


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------
def train_adam(
    angles_train: np.ndarray,
    y_train:      np.ndarray,
    n_steps:      int   = 300,
    lr:           float = 0.01,
    batch_size:   int   = 500,
    l2:           float = 1e-4,
    seed:         int   = 0,
    verbose:      bool  = True,
    loss_history: Optional[list] = None,
    class_weight: Optional[np.ndarray] = None,
    init_quantum_params: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Adam + mini-batch gradient (PSR + backprop).

    init_quantum_params: if provided, initialise ansatz from a previous run
    (e.g. from vqc_angle32_params.json) and only randomly initialise the MLP.
    This warm-starts the circuit from an already-trained encoding.
    """
    rng = np.random.default_rng(seed)

    # MLP initialisation: Xavier uniform
    W1_std = np.sqrt(2.0 / N_QUBITS)
    W2_std = np.sqrt(2.0 / N_HIDDEN)
    mlp_init = np.concatenate([
        rng.normal(scale=W1_std, size=N_MLP_W1),
        np.zeros(N_MLP_B1),
        rng.normal(scale=W2_std, size=N_MLP_W2),
        [0.0],
    ])

    if init_quantum_params is not None and len(init_quantum_params) == N_ANSATZ_PARAMS:
        ansatz_init = init_quantum_params.copy()
    else:
        ansatz_init = rng.normal(scale=0.1, size=N_ANSATZ_PARAMS)

    params = np.concatenate([ansatz_init, mlp_init])

    beta1, beta2, eps_a = 0.9, 0.999, 1e-8
    m = np.zeros(N_PARAMS);  v = np.zeros(N_PARAMS)

    hist      = loss_history if loss_history is not None else []
    n         = len(y_train)
    eff_batch = min(batch_size, n)

    for t in range(1, n_steps + 1):
        idx   = rng.choice(n, size=eff_batch, replace=False)
        ang_b = angles_train[idx]
        y_b   = y_train[idx]

        g = _gradient(params, ang_b, y_b, l2, class_weight)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps_a)

        loss = float(_bce(params, ang_b, y_b, 0.0, class_weight))
        hist.append(loss)

        if verbose and (t == 1 or t % 10 == 0):
            print(f"  step {t:3d}/{n_steps}   batch_BCE={loss:.4f}   "
                  f"lr={lr:.4f}   batch={eff_batch}")

    return params


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    accuracy:    float
    auc:         float
    sensitivity: float
    specificity: float
    n:           int


def evaluate(
    angles:    np.ndarray,
    y:         np.ndarray,
    params:    np.ndarray,
    threshold: float = 0.5,
) -> EvalResult:
    from sklearn.metrics import roc_auc_score
    p    = predict_proba(angles, params)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    acc  = (tp + tn) / len(y)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return EvalResult(accuracy=acc, auc=auc, sensitivity=sens, specificity=spec, n=len(y))
