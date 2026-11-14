"""Amplitude-encoded VQC on 8 qubits (256-dim state space).

Complement to `vqc.py` (which uses angle encoding on 8 qubits, one qubit per
feature).  This module encodes up to **256 features** (2**8) directly into the
amplitudes of an 8-qubit state, then runs a hardware-efficient ansatz
(Ry + Rz per qubit + CNOT ring, N_LAYERS deep) and reads out a trainable
weighted sum of all N_QUBITS qubit expectations <Z_0 .. Z_7>.

Encoding contract
-----------------
Input: a (batch, D) real feature matrix with D <= 256.
Steps  (see `encode_amplitudes`):
    1. Standardize each column to zero mean / unit variance (fit stats supplied).
    2. Clip to +-6 sigma to bound amplitudes.
    3. Pad to length 256 with zeros.
    4. L2-normalize each row so it is a valid pure state:  ||psi||_2 == 1.

The resulting state vector ``psi in C^256`` IS the 8-qubit register state.
On real quantum hardware this is prepared via a state-preparation circuit
(e.g. Microsoft.Quantum.Preparation.PrepareArbitraryStateD or IBM initialize()).
The classical simulator writes the vector directly.  Rz gates in the variational
ansatz introduce complex phases, so all state vectors are complex128 -- matching
hardware behaviour exactly and required for correct Rz simulation.

Ansatz
------
Same design as `vqc.py`:
    * per layer: Ry(theta) then Rz(phi) on each qubit  ->  2 * N params
    * per layer: CNOT ring q0 -> q1 -> ... -> q_{N-1} -> q0
    * final Ry layer:  N params
Circuit params: N_ANSATZ_PARAMS = 2 * N * L + N = 2*8*4 + 8 = 72 for defaults.

Rz is the correct diagonal gate diag(e^{-i*phi/2}, e^{+i*phi/2}).  On IBM / IonQ
hardware Rz is a virtual frame-rotation with zero gate time, so using it properly
costs nothing on hardware but gives Ry+Rz the full SU(2) coverage per qubit.

Readout
-------
All N_QUBITS qubit expectations are combined with a trainable linear layer:
    logit = <Z_all> @ w + b        (9 params: 8 weights + 1 bias)
    p(MI = 1 | x, theta, w, b) = sigmoid(logit)

Using all qubits at readout is hardware-efficient: one shot yields a full N-bit
bitstring, so reading 8 qubits costs the same shots as reading 1.

Total params: N_ANSATZ_PARAMS (72) + N_READOUT_PARAMS (9) = N_PARAMS (81).

Gradient
--------
Ansatz params:  exact parameter shift rule  (shift = pi/2; no finite differences).
Readout params: classical chain rule -- no extra circuit evaluations.
Combined gradient is passed as `jac` to SciPy L-BFGS-B.
Parameter shift is the only gradient method available on real quantum hardware,
so this simulation is directly portable without change.

Class imbalance
---------------
BCE loss accepts inverse-frequency class weights [w_NORM, w_MI] so the minority
class (MI ~5 k) is not dominated by the majority class (NORM ~9 k).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
N_QUBITS         = 8
DIM              = 1 << N_QUBITS                        # 256
N_LAYERS         = 4
N_ANSATZ_PARAMS  = 2 * N_QUBITS * N_LAYERS + N_QUBITS  # 72  (circuit gates)
N_READOUT_PARAMS = N_QUBITS + 1                         # 9   (weights + bias)
N_PARAMS         = N_ANSATZ_PARAMS + N_READOUT_PARAMS   # 81

# QDK convention: qubit 0 is the MOST-significant bit of the basis index.
# For an amplitude vector indexed 0..255, bit i of the index (with bit 0 = MSB,
# qubit 0) selects the computational-basis label on that qubit.


# ---------------------------------------------------------------------------
#  Encoding
# ---------------------------------------------------------------------------
@dataclass
class Standardizer:
    """Column-wise z-score + clip; L2-normalize pads to a valid state vector."""
    mean: np.ndarray
    std:  np.ndarray
    clip: float = 6.0

    @classmethod
    def fit(cls, X: np.ndarray, clip: float = 6.0) -> "Standardizer":
        mean = X.mean(axis=0)
        std  = X.std(axis=0)
        std[std < 1e-12] = 1.0    # avoid div-by-zero on constant columns
        return cls(mean=mean, std=std, clip=clip)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return np.clip(Z, -self.clip, self.clip)


def encode_amplitudes_with_scale(X_std: np.ndarray, dim: int = DIM) -> np.ndarray:
    """Like encode_amplitudes but appends log(1 + ‖z‖) as an extra feature.

    Amplitude encoding L2-normalises the state, destroying all magnitude info —
    a 2 mV ST elevation looks identical to 0.5 mV if other features scale alike.
    Appending the log-norm gives the circuit a 'pattern strength' handle:
    MI records with elevated ST / large R / T-inversions have distinctly larger
    ‖z‖, and this extra slot encodes that after normalisation.

    With D=251 clinical features the augmented vector has D+1=252 entries;
    the remaining 4 slots are zero-padded to reach dim=256.
    The log_ref normalises so the scale feature is O(1), same as unit-variance
    clinical features.
    """
    batch, d = X_std.shape
    if d + 1 > dim:
        raise ValueError(f"Cannot fit {d}+1 features into {dim}-dim state.")
    norms_d  = np.linalg.norm(X_std, axis=1, keepdims=True)          # (batch,1)
    log_ref  = float(np.log1p(np.sqrt(d)))                            # ≈ log(1+√251)≈3.2
    log_scale = np.log1p(norms_d) / log_ref                           # (batch,1), O(1)
    psi = np.zeros((batch, dim), dtype=np.complex128)
    psi[:, :d]    = X_std
    psi[:, d:d+1] = log_scale
    norms_full = np.linalg.norm(psi.real, axis=1, keepdims=True)
    zero_mask  = (norms_full < 1e-12).reshape(-1)
    if zero_mask.any():
        psi[zero_mask, :] = 1.0 / np.sqrt(dim)
        norms_full[zero_mask] = 1.0
    return psi / norms_full


def encode_amplitudes(X_std: np.ndarray, dim: int = DIM) -> np.ndarray:
    """Turn a (batch, D) standardized matrix into (batch, dim) unit-norm complex states.

    Pads with zeros to `dim` and L2-normalizes each row.  Rows whose norm is
    numerically zero are given a uniform-superposition fallback so the state
    remains valid (probability distributed evenly over basis states).

    Output is complex128: the initial state is real, but Rz gates introduced in
    the ansatz rotate amplitudes into the complex plane.  Casting here avoids
    repeated promotion inside the hot loop.
    """
    batch, d = X_std.shape
    if d > dim:
        raise ValueError(f"Cannot fit {d} features into {dim}-dim state.")
    psi = np.zeros((batch, dim), dtype=np.complex128)
    psi[:, :d] = X_std
    norms = np.linalg.norm(psi.real, axis=1, keepdims=True)
    zero_mask = (norms < 1e-12).reshape(-1)
    if zero_mask.any():
        psi[zero_mask, :] = 1.0 / np.sqrt(dim)
        norms[zero_mask] = 1.0
    psi = psi / norms
    return psi


# ---------------------------------------------------------------------------
#  Batched state-vector simulator (NumPy, complex128)
# ---------------------------------------------------------------------------
def _apply_single_qubit(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a 2x2 gate to `qubit` (0=MSB) on a batched state (batch, 2**N).

    Reshape trick: view state as (batch, 2, ..., 2) with N axes of size 2,
    where axis (1 + qubit) is the target qubit's bit. Then matmul on that axis.
    Works for both real (Ry) and complex (Rz) gates on complex128 states.
    """
    batch = state.shape[0]
    shape = (batch,) + (2,) * N_QUBITS
    s     = state.reshape(shape)
    axis  = 1 + qubit
    s     = np.moveaxis(s, axis, 1)    # (batch, 2, other...)
    orig_shape = s.shape
    s     = s.reshape(batch, 2, -1)    # (batch, 2, rest)
    s     = np.einsum("ij,bjk->bik", gate, s)
    s     = s.reshape(orig_shape)
    s     = np.moveaxis(s, 1, axis)
    return s.reshape(batch, DIM)


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    """Batched CNOT (batch, 2**N) -> (batch, 2**N).

    Swap the target-qubit component of amplitudes where the control bit == 1.
    CNOT is a column permutation -- works identically for complex128 states.
    """
    ctrl_bit     = 1 << (N_QUBITS - 1 - control)
    tgt_bit      = 1 << (N_QUBITS - 1 - target)
    idx          = np.arange(DIM)
    swap_mask    = (idx & ctrl_bit) != 0
    swap_partner = idx ^ tgt_bit
    perm = np.where(swap_mask, swap_partner, idx)
    return state[:, perm]


def _ry_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _rz_matrix(phi: float) -> np.ndarray:
    """Proper Rz gate: diag(e^{-i*phi/2}, e^{+i*phi/2}).

    On IBM / IonQ hardware this is a virtual frame-rotation (zero gate time).
    Together with Ry it spans all of SU(2) per qubit, giving the ansatz full
    single-qubit expressiveness -- something two consecutive Ry rotations cannot
    achieve (Ry*Ry = Ry at the combined angle, a redundant parameterisation).
    """
    h = phi / 2.0
    return np.array(
        [[np.exp(-1j * h), 0.0        ],
         [0.0,             np.exp(1j * h)]],
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
#  Forward pass
# ---------------------------------------------------------------------------
def circuit_forward(psi0: np.ndarray, ansatz_params: np.ndarray) -> np.ndarray:
    """Return the state vector after applying the HEA ansatz to psi0.

    psi0:          (batch, 256) complex128 amplitude-encoded initial state
    ansatz_params: (N_ANSATZ_PARAMS,) = (72,) trainable circuit parameters
    Returns:       (batch, 256) complex128 final state
    """
    state  = psi0.copy()
    stride = 2 * N_QUBITS

    for L in range(N_LAYERS):
        base = stride * L
        for i in range(N_QUBITS):
            state = _apply_single_qubit(state, _ry_matrix(ansatz_params[base + i]), i)
            state = _apply_single_qubit(state, _rz_matrix(ansatz_params[base + N_QUBITS + i]), i)
        for i in range(N_QUBITS):
            state = _apply_cnot(state, i, (i + 1) % N_QUBITS)

    final_base = stride * N_LAYERS
    for i in range(N_QUBITS):
        state = _apply_single_qubit(state, _ry_matrix(ansatz_params[final_base + i]), i)

    return state


def expectation_all(state: np.ndarray) -> np.ndarray:
    """<Z_j> for every qubit j = 0 .. N_QUBITS-1.

    Returns a (batch, N_QUBITS) real array; each value in [-1, +1].

    On hardware <Z_j> is estimated by recording the j-th bit of each shot
    bitstring (+1 for |0>, -1 for |1>) and averaging over shots.  Measuring all
    qubits per shot costs no extra shots -- the full bitstring is read once.
    """
    idx   = np.arange(DIM)
    probs = (state * state.conj()).real          # |amp_k|^2, shape (batch, DIM)
    out   = np.empty((state.shape[0], N_QUBITS), dtype=np.float64)
    for j in range(N_QUBITS):
        sign      = np.where((idx >> (N_QUBITS - 1 - j)) & 1, -1.0, +1.0)
        out[:, j] = probs @ sign
    return out


def predict_proba(psi0: np.ndarray, params: np.ndarray) -> np.ndarray:
    """p(MI = 1) = sigmoid(<Z_all> @ w + b) using full N-qubit readout.

    params layout (length N_PARAMS = 81):
        params[:N_ANSATZ_PARAMS]                          -> 72 circuit params
        params[N_ANSATZ_PARAMS : N_ANSATZ_PARAMS+N_QUBITS] -> 8 readout weights
        params[-1]                                        -> 1 readout bias
    """
    state = circuit_forward(psi0, params[:N_ANSATZ_PARAMS])
    z_all = expectation_all(state)                           # (batch, N_QUBITS)
    w     = params[N_ANSATZ_PARAMS : N_ANSATZ_PARAMS + N_QUBITS]
    b     = params[-1]
    logit = z_all @ w + b
    return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------------------
#  Loss
# ---------------------------------------------------------------------------
def _bce(
    params: np.ndarray,
    psi0:   np.ndarray,
    y:      np.ndarray,
    l2:     float,
    class_weight: Optional[np.ndarray] = None,
) -> float:
    """Weighted BCE + L2 regularization (on circuit params only).

    class_weight: [w_NORM, w_MI] array.  compute_class_weight() in the training
                  script produces balanced inverse-frequency weights.
    """
    p = predict_proba(psi0, params)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if class_weight is not None:
        sw  = np.where(y == 1, class_weight[1], class_weight[0])
        bce = bce * sw
    loss = bce.mean()
    if l2 > 0.0:
        loss += l2 * (params[:N_ANSATZ_PARAMS] ** 2).mean()
    return float(loss)


# ---------------------------------------------------------------------------
#  Gradient — parameter shift rule (circuit) + chain rule (readout)
# ---------------------------------------------------------------------------
def _gradient_psr(
    params: np.ndarray,
    psi0:   np.ndarray,
    y:      np.ndarray,
    l2:     float,
    class_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exact gradient: parameter shift rule for circuit params, chain rule for readout.

    Parameter shift rule for Ry / Rz gates (generator eigenvalues ±1/2):
        d<O>/d(theta_i) = [ <O>(theta_i + pi/2) - <O>(theta_i - pi/2) ] / 2

    This is exact (not an approximation) and is the gradient method used on
    real quantum hardware (IBM, IonQ, Quantinuum).  Finite differences would
    require the same number of circuit evals but give only approximate gradients.

    Cost: 2 * N_ANSATZ_PARAMS = 144 circuit evaluations per gradient call.
    Readout gradients are classical -- no additional circuit evals needed.
    """
    grad = np.zeros(N_PARAMS)
    eps  = np.pi / 2.0

    # Current forward pass (shared with _bce but recomputed to keep interface clean)
    state = circuit_forward(psi0, params[:N_ANSATZ_PARAMS])
    z_all = expectation_all(state)                            # (batch, N_QUBITS)
    w     = params[N_ANSATZ_PARAMS : N_ANSATZ_PARAMS + N_QUBITS]
    b     = params[-1]
    p     = np.clip(1.0 / (1.0 + np.exp(-(z_all @ w + b))), 1e-7, 1.0 - 1e-7)

    if class_weight is not None:
        sw = np.where(y == 1, class_weight[1], class_weight[0])
    else:
        sw = np.ones(len(y), dtype=np.float64)

    # d(weighted_BCE) / d(logit_j)  =  sw_j * (p_j - y_j)
    residual = sw * (p - y)                                   # (batch,)

    # Readout gradients (analytical, no circuit evals)
    # d(loss)/d(w_k) = mean_j[ residual_j * z_all_{j,k} ]
    grad[N_ANSATZ_PARAMS : N_ANSATZ_PARAMS + N_QUBITS] = (
        (z_all * residual[:, None]).mean(axis=0)
    )
    # d(loss)/d(b) = mean(residual)
    grad[-1] = residual.mean()

    # Ansatz gradients via parameter shift
    # d(loss)/d(theta_i) = mean_j[ sum_k( dL_dz_{j,k} * d<Z_k>/d(theta_i)_j ) ]
    dL_dz = residual[:, None] * w[None, :]                    # (batch, N_QUBITS)

    for i in range(N_ANSATZ_PARAMS):
        ap, am = params.copy(), params.copy()
        ap[i] += eps
        am[i] -= eps
        z_p = expectation_all(circuit_forward(psi0, ap[:N_ANSATZ_PARAMS]))
        z_m = expectation_all(circuit_forward(psi0, am[:N_ANSATZ_PARAMS]))
        dz_dtheta = (z_p - z_m) / 2.0                        # (batch, N_QUBITS)
        # sum over qubits k for each sample j, then mean over j
        grad[i] = (dL_dz * dz_dtheta).sum(axis=1).mean()

    if l2 > 0.0:
        grad[:N_ANSATZ_PARAMS] += (
            2.0 * l2 * params[:N_ANSATZ_PARAMS] / N_ANSATZ_PARAMS
        )

    return grad


# ---------------------------------------------------------------------------
#  Training — Adam mini-batch  (recommended for full-dataset simulation)
# ---------------------------------------------------------------------------
def train_adam(
    psi0_train:   np.ndarray,
    y_train:      np.ndarray,
    n_steps:      int   = 200,
    lr:           float = 0.02,
    batch_size:   int   = 300,
    l2:           float = 1e-4,
    seed:         int   = 0,
    verbose:      bool  = True,
    loss_history: Optional[list] = None,
    class_weight: Optional[np.ndarray] = None,
    init_params:  Optional[np.ndarray] = None,
) -> np.ndarray:
    """Adam optimizer with mini-batch PSR gradient for full-dataset training.

    L-BFGS-B works well when the gradient is deterministic (full-batch).  With
    mini-batches the gradient is stochastic and L-BFGS-B's curvature estimate
    breaks down.  Adam is the standard choice for stochastic gradients.

    Each step samples `batch_size` rows at random, computes the exact PSR
    gradient on that mini-batch, then applies an Adam update.  Over n_steps
    steps the model sees n_steps * batch_size / n_train effective epochs.

    With batch_size=300 on 12 k training rows each step takes ~14 s
    (144 PSR circuit evals on a 300-row batch); 200 steps ≈ 45 min.

    The PSR gradient is unchanged -- this function is still hardware-portable.
    On real hardware each of the 144 circuit evals is a QPU job; classical
    post-processing assembles the gradient and Adam runs on a classical co-proc.
    """
    rng = np.random.default_rng(seed)

    if init_params is not None:
        if len(init_params) != N_PARAMS:
            raise ValueError(f"init_params length {len(init_params)} != N_PARAMS {N_PARAMS}")
        params = init_params.copy()
    else:
        x0_ansatz  = rng.normal(scale=0.1,  size=N_ANSATZ_PARAMS)
        x0_readout = np.append(rng.normal(scale=0.01, size=N_QUBITS), 0.0)
        params = np.concatenate([x0_ansatz, x0_readout])

    beta1, beta2, eps_a = 0.9, 0.999, 1e-8
    m = np.zeros(N_PARAMS)
    v = np.zeros(N_PARAMS)

    hist      = loss_history if loss_history is not None else []
    n         = len(y_train)
    eff_batch = min(batch_size, n)

    for t in range(1, n_steps + 1):
        idx   = rng.choice(n, size=eff_batch, replace=False)
        psi_b = psi0_train[idx]
        y_b   = y_train[idx]

        g = _gradient_psr(params, psi_b, y_b, l2, class_weight)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps_a)

        loss = float(_bce(params, psi_b, y_b, 0.0, class_weight))
        hist.append(loss)

        if verbose and (t == 1 or t % 10 == 0):
            print(f"  step {t:3d}/{n_steps}   batch_BCE={loss:.4f}   "
                  f"lr={lr:.4f}   batch={eff_batch}")

    return params


# ---------------------------------------------------------------------------
#  Training — L-BFGS-B full-batch  (use when batch gradient is feasible)
# ---------------------------------------------------------------------------
def train(
    psi0_train:   np.ndarray,
    y_train:      np.ndarray,
    n_iter:       int  = 200,
    l2:           float = 1e-4,
    seed:         int  = 0,
    verbose:      bool = True,
    loss_history: Optional[list] = None,
    class_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit params via L-BFGS-B with exact parameter-shift gradients.

    Returns the trained parameter vector (length N_PARAMS = 81).

    class_weight: [w_NORM, w_MI] from compute_class_weight() in the training
                  script.  Pass None for unweighted training (not recommended
                  given the 1:1.74 MI/NORM imbalance in PTB-XL).
    """
    rng = np.random.default_rng(seed)

    # Circuit params: small random to break weight-symmetry (Barren plateau risk).
    # Readout weights: very small (z_all is O(1), logit magnitude stays controlled).
    # Readout bias: 0 (L-BFGS-B will set it to match the class prior).
    x0_ansatz  = rng.normal(scale=0.1,  size=N_ANSATZ_PARAMS)
    x0_readout = np.append(rng.normal(scale=0.01, size=N_QUBITS), 0.0)
    x0 = np.concatenate([x0_ansatz, x0_readout])

    hist: list[float] = loss_history if loss_history is not None else []

    def _callback(xk: np.ndarray) -> None:
        hist.append(_bce(xk, psi0_train, y_train, 0.0, class_weight))
        if verbose and (len(hist) == 1 or len(hist) % 10 == 0):
            print(f"  iter {len(hist):3d}   BCE = {hist[-1]:.4f}")

    result = minimize(
        _bce,
        x0,
        args=(psi0_train, y_train, l2, class_weight),
        method="L-BFGS-B",
        jac=_gradient_psr,                   # exact analytic gradient
        options={"maxiter": n_iter, "ftol": 1e-6},
        callback=_callback,
    )
    if verbose:
        print(f"  converged: {result.success}   final BCE = {result.fun:.4f}")
    return result.x


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
    psi0:      np.ndarray,
    y:         np.ndarray,
    params:    np.ndarray,
    threshold: float = 0.5,
) -> EvalResult:
    from sklearn.metrics import roc_auc_score

    p    = predict_proba(psi0, params)
    pred = (p >= threshold).astype(int)
    acc  = float((pred == y).mean())
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return EvalResult(accuracy=acc, auc=auc, n=len(y), threshold=threshold)
