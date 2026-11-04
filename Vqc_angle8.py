"""Angle-encoded 8-qubit VQC with sequential per-layer feature loading.

Encoding strategy
-----------------
The circuit processes 32 = N_QUBITS * N_LAYERS features without any L2
normalisation.  Features are divided into 4 groups of 8 and loaded at the
START of each HEA layer via per-sample Ry gates:

    Layer 0: top  8 features (T-wave amplitudes)       sep 0.965 - 1.477
    Layer 1: top  9-16 features (Q-wave depths/amps)   sep 0.854 - 0.959
    Layer 2: top 17-24 features (Q-durations/ST amps)  sep 0.775 - 0.850
    Layer 3: top 25-32 features (ST deviations/means)  sep 0.750 - 0.774

All 32 features are ranked by |MI_mean - NORM_mean| / pooled_std on the
training set.  Classical LR ceiling on these 32 features: 87.30%, AUC 0.937.

Circuit structure (per layer L)
    1. Ry(angle_{L*8+i}(b)) on qubit i  -- batched, feature group L
    2. Ry(theta_i)           on qubit i  -- shared ansatz gate
    3. Rz(phi_i)             on qubit i  -- proper complex Rz (hardware VZ)
    4. CNOT ring

After 4 layers: final Ry layer (ansatz only, no encoding).
Readout: trainable linear layer on <Z_all>: logit = sum_j w_j <Z_j> + b.

Why sequential loading beats re-uploading the same 8 features
--------------------------------------------------------------
Re-uploading the same features N_LAYERS times expands expressibility but
does not increase information content.  Loading 4 different groups uses the
entangling blocks to progressively fuse T-wave, Q-wave, and ST information
into a joint quantum state.  This mirrors clinical MI diagnosis: T-wave
changes, Q-wave patterns, and ST deviations are COMPLEMENTARY, not redundant.

Gradient: PSR (exact) on ansatz params, classical chain rule on readout.
Sequential encoding does not affect PSR — encoding angles are fixed input-
dependent constants; perturbing ansatz param theta_i only changes its gate.

Hardware note: loading different features per layer is standard on real QPUs
(different pulse sequences per layer), with no extra gate overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
#  Constants  (same circuit size as Vqc_amp8 for fair comparison)
# ---------------------------------------------------------------------------
N_QUBITS          = 8
DIM               = 1 << N_QUBITS                         # 256
N_LAYERS          = 4
N_FEATURES_TOTAL  = N_QUBITS * N_LAYERS                   # 32 (8 per layer)
N_ANSATZ_PARAMS   = 2 * N_QUBITS * N_LAYERS + N_QUBITS   # 72
N_READOUT_PARAMS  = N_QUBITS + 1                          # 9
N_PARAMS          = N_ANSATZ_PARAMS + N_READOUT_PARAMS    # 81

# 32 features ranked by |MI_mean - NORM_mean| / pooled_std on standardised PTB-XL
# (training set only).  Groups of 8 are loaded at layer 0, 1, 2, 3 respectively.
ANGLE_FEATURES = [
    # Layer 0 -- T-wave amplitudes (best single-feature discriminators, sep 0.965-1.477)
    "aVR_t_amp",          # sep=1.477  reciprocal T-wave in aVR → elevated in MI
    "V6_t_amp",           # sep=1.299  lateral T-wave depression
    "I_t_amp",            # sep=1.258  lateral T-wave depression
    "V5_t_amp",           # sep=1.243  lateral T-wave depression
    "II_t_amp",           # sep=1.163  inferior T-wave
    "aVR_t_inverted",     # sep=1.048  binary inversion flag for aVR
    "V5_t_inverted",      # sep=0.972  binary inversion flag for V5
    "V6_t_inverted",      # sep=0.965  binary inversion flag for V6

    # Layer 1 -- Q-wave depths + more T-wave (old MI markers, sep 0.854-0.959)
    "III_q_depth",        # sep=0.959  inferior MI (Q in III)
    "I_t_inverted",       # sep=0.935  lateral T inversion
    "II_r_amp",           # sep=0.910  reduced R in II → poor R progression
    "V4_t_amp",           # sep=0.896  anterior T-wave
    "V6_q_duration",      # sep=0.889  lateral Q-wave duration
    "V2_q_depth",         # sep=0.885  anterior Q-wave (old MI)
    "V3_q_depth",         # sep=0.880  anterior Q-wave
    "V1_q_depth",         # sep=0.854  anterior Q-wave

    # Layer 2 -- Q-wave durations + ST amplitudes (sep 0.775-0.850)
    "aVF_q_depth",        # sep=0.850  inferior Q-wave
    "I_q_duration",       # sep=0.810  lateral Q-wave duration
    "aVL_st_amp_j80",     # sep=0.801  ST amplitude aVL at J+80ms
    "V4_t_inverted",      # sep=0.796  anterior T inversion
    "aVR_q_duration",     # sep=0.796  aVR Q-wave duration
    "I_st_dev_j80",       # sep=0.794  ST deviation I at J+80ms
    "aVL_t_inverted",     # sep=0.788  lateral T inversion
    "aVL_st_mean",        # sep=0.775  mean ST level aVL

    # Layer 3 -- ST segment features (deviations/areas/means, sep 0.750-0.774)
    "aVL_st_area",        # sep=0.774  ST area aVL
    "V5_q_duration",      # sep=0.773  lateral Q-wave duration
    "aVL_st_dev_j80",     # sep=0.765  ST deviation aVL J+80ms
    "aVL_st_amp_j60",     # sep=0.763  ST amplitude aVL J+60ms
    "II_t_inverted",      # sep=0.760  inferior T inversion
    "I_st_dev_j60",       # sep=0.751  ST deviation I J+60ms
    "aVL_st_dev_j60",     # sep=0.750  ST deviation aVL J+60ms
    "I_st_mean",          # sep=0.750  mean ST level I
]


# ---------------------------------------------------------------------------
#  Feature scaling
# ---------------------------------------------------------------------------
@dataclass
class AngleScaler:
    """Column-wise z-score for all N_FEATURES_TOTAL angle-encoding features."""
    mean: np.ndarray
    std:  np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "AngleScaler":
        mean = X.mean(axis=0)
        std  = X.std(axis=0)
        std[std < 1e-9] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def compute_angles(X_std: np.ndarray) -> np.ndarray:
    """Map standardised features to Ry angles in (-pi, pi) via pi*tanh.

    X_std: (batch, N_FEATURES_TOTAL=32)
    Returns (batch, N_FEATURES_TOTAL) angles.  tanh squashes outliers while
    preserving sign and relative magnitude of clinical values.
    """
    return np.pi * np.tanh(X_std)          # (batch, N_FEATURES_TOTAL)


# ---------------------------------------------------------------------------
#  Gate helpers
# ---------------------------------------------------------------------------
def _ry_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _rz_matrix(phi: float) -> np.ndarray:
    h = phi / 2.0
    return np.array(
        [[np.exp(-1j * h), 0.0],
         [0.0,             np.exp(1j * h)]],
        dtype=np.complex128,
    )


def _apply_single_qubit(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a shared 2×2 gate to `qubit` across the entire batch."""
    batch = state.shape[0]
    shape = (batch,) + (2,) * N_QUBITS
    s     = state.reshape(shape)
    axis  = 1 + qubit
    s     = np.moveaxis(s, axis, 1)
    orig  = s.shape
    s     = s.reshape(batch, 2, -1)
    s     = np.einsum("ij,bjk->bik", gate, s)
    s     = s.reshape(orig)
    s     = np.moveaxis(s, 1, axis)
    return s.reshape(batch, DIM)


def _apply_ry_batched(state: np.ndarray, qubit: int, angles: np.ndarray) -> np.ndarray:
    """Apply Ry(angles[b]) on `qubit` for each batch sample b.

    angles: (batch,) — one rotation angle per sample.
    This is how data re-uploading works: each sample gets its own encoding gate.
    On hardware this is per-qubit pulse calibration for each shot.
    """
    batch = state.shape[0]
    c = np.cos(angles / 2)                 # (batch,)
    s = np.sin(angles / 2)                 # (batch,)
    shape = (batch,) + (2,) * N_QUBITS
    st    = state.reshape(shape)
    axis  = 1 + qubit
    st    = np.moveaxis(st, axis, 1)       # (batch, 2, ...)
    orig  = st.shape
    st    = st.reshape(batch, 2, -1)       # (batch, 2, 128)
    out   = np.empty_like(st)
    out[:, 0, :] = c[:, None] * st[:, 0, :] - s[:, None] * st[:, 1, :]
    out[:, 1, :] = s[:, None] * st[:, 0, :] + c[:, None] * st[:, 1, :]
    out = out.reshape(orig)
    out = np.moveaxis(out, 1, axis)
    return out.reshape(batch, DIM)


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    ctrl_bit     = 1 << (N_QUBITS - 1 - control)
    tgt_bit      = 1 << (N_QUBITS - 1 - target)
    idx          = np.arange(DIM)
    swap_mask    = (idx & ctrl_bit) != 0
    swap_partner = idx ^ tgt_bit
    perm = np.where(swap_mask, swap_partner, idx)
    return state[:, perm]


# ---------------------------------------------------------------------------
#  Forward pass with data re-uploading
# ---------------------------------------------------------------------------
def circuit_forward(angles: np.ndarray, ansatz_params: np.ndarray) -> np.ndarray:
    """8-qubit HEA circuit with sequential feature loading.

    angles:        (batch, N_FEATURES_TOTAL=32) — 8 different features per layer
    ansatz_params: (N_ANSATZ_PARAMS=72,) trainable parameters

    Layer L loads angles[:, L*N_QUBITS : (L+1)*N_QUBITS] via batched Ry gates,
    then applies the trained Ry+Rz+CNOT block.  Starting state is |0^8>.

    This sequentially fuses 4 clinical groups (T-wave, Q-wave, ST-amplitude,
    ST-deviation) into a joint entangled state without any L2 normalisation.
    """
    batch = angles.shape[0]
    state = np.zeros((batch, DIM), dtype=np.complex128)
    state[:, 0] = 1.0                      # |000...0>

    stride = 2 * N_QUBITS

    for L in range(N_LAYERS):
        # Sequential loading: different 8 features per layer
        layer_enc = angles[:, L * N_QUBITS : (L + 1) * N_QUBITS]   # (batch, 8)
        for i in range(N_QUBITS):
            state = _apply_ry_batched(state, i, layer_enc[:, i])

        # Ansatz: shared Ry + Rz on each qubit
        base = stride * L
        for i in range(N_QUBITS):
            state = _apply_single_qubit(state, _ry_matrix(ansatz_params[base + i]), i)
            state = _apply_single_qubit(state, _rz_matrix(ansatz_params[base + N_QUBITS + i]), i)

        # Entanglement: CNOT ring
        for i in range(N_QUBITS):
            state = _apply_cnot(state, i, (i + 1) % N_QUBITS)

    # Final Ry layer (ansatz only)
    final_base = stride * N_LAYERS
    for i in range(N_QUBITS):
        state = _apply_single_qubit(state, _ry_matrix(ansatz_params[final_base + i]), i)

    return state


def expectation_all(state: np.ndarray) -> np.ndarray:
    """<Z_j> for all j. Returns (batch, N_QUBITS) real."""
    idx   = np.arange(DIM)
    probs = (state * state.conj()).real
    out   = np.empty((state.shape[0], N_QUBITS), dtype=np.float64)
    for j in range(N_QUBITS):
        sign      = np.where((idx >> (N_QUBITS - 1 - j)) & 1, -1.0, +1.0)
        out[:, j] = probs @ sign
    return out


def predict_proba(angles: np.ndarray, params: np.ndarray) -> np.ndarray:
    """p(MI=1) = sigmoid(<Z_all> @ w + b).

    angles: (batch, N_FEATURES_TOTAL=32) pre-computed from compute_angles().
    params layout: [:72] circuit, [72:80] readout weights, [80] bias.
    """
    state = circuit_forward(angles, params[:N_ANSATZ_PARAMS])
    z_all = expectation_all(state)
    w     = params[N_ANSATZ_PARAMS : N_ANSATZ_PARAMS + N_QUBITS]
    b     = params[-1]
    return 1.0 / (1.0 + np.exp(-(z_all @ w + b)))


# ---------------------------------------------------------------------------
#  Loss and gradient
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


def _gradient_psr(
    params:       np.ndarray,
    angles:       np.ndarray,
    y:            np.ndarray,
    l2:           float,
    class_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Parameter shift rule gradient.

    Data re-uploading does not affect PSR correctness: encoding angles are
    fixed (functions of input data, not trained params), so perturbing ansatz
    param theta_i only changes its own gate.
    """
    grad = np.zeros(N_PARAMS)
    eps  = np.pi / 2.0

    state = circuit_forward(angles, params[:N_ANSATZ_PARAMS])
    z_all = expectation_all(state)
    w     = params[N_ANSATZ_PARAMS : N_ANSATZ_PARAMS + N_QUBITS]
    b     = params[-1]
    p     = np.clip(1.0 / (1.0 + np.exp(-(z_all @ w + b))), 1e-7, 1.0 - 1e-7)

    sw       = np.where(y == 1, class_weight[1], class_weight[0]) if class_weight is not None \
               else np.ones(len(y), dtype=np.float64)
    residual = sw * (p - y)

    grad[N_ANSATZ_PARAMS : N_ANSATZ_PARAMS + N_QUBITS] = (z_all * residual[:, None]).mean(axis=0)
    grad[-1] = residual.mean()

    dL_dz = residual[:, None] * w[None, :]    # (batch, N_QUBITS)

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
    n_steps:      int   = 200,
    lr:           float = 0.02,
    batch_size:   int   = 500,
    l2:           float = 1e-4,
    seed:         int   = 0,
    verbose:      bool  = True,
    loss_history: Optional[list] = None,
    class_weight: Optional[np.ndarray] = None,
    init_params:  Optional[np.ndarray] = None,
) -> np.ndarray:
    """Adam + mini-batch PSR gradient for angle-encoded VQC."""
    rng = np.random.default_rng(seed)

    if init_params is not None and len(init_params) == N_PARAMS:
        params = init_params.copy()
    else:
        x0a = rng.normal(scale=0.1,  size=N_ANSATZ_PARAMS)
        x0r = np.append(rng.normal(scale=0.01, size=N_QUBITS), 0.0)
        params = np.concatenate([x0a, x0r])

    beta1, beta2, eps_a = 0.9, 0.999, 1e-8
    m = np.zeros(N_PARAMS);  v = np.zeros(N_PARAMS)

    hist      = loss_history if loss_history is not None else []
    n         = len(y_train)
    eff_batch = min(batch_size, n)

    for t in range(1, n_steps + 1):
        idx     = rng.choice(n, size=eff_batch, replace=False)
        ang_b   = angles_train[idx]
        y_b     = y_train[idx]

        g = _gradient_psr(params, ang_b, y_b, l2, class_weight)

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
#  Metrics
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    accuracy:  float
    auc:       float
    n:         int
    threshold: float = 0.5


def evaluate(
    angles:    np.ndarray,
    y:         np.ndarray,
    params:    np.ndarray,
    threshold: float = 0.5,
) -> EvalResult:
    from sklearn.metrics import roc_auc_score
    p    = predict_proba(angles, params)
    pred = (p >= threshold).astype(int)
    acc  = float((pred == y).mean())
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return EvalResult(accuracy=acc, auc=auc, n=len(y), threshold=threshold)
