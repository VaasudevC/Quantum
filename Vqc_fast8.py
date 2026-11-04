"""Fast exact simulator for the 8-qubit sequential-angle VQC with adjoint gradients.

Circuit is IDENTICAL to Vqc_angle8.circuit_forward (Run 7) -- verify() checks the
state vector and the gradient against the original PSR implementation.

What is different is only how it is computed:
  * encoding Ry and ansatz Ry on the same qubit are merged into one rotation
  * the 8 Rz gates of a layer are applied as one diagonal phase vector
  * the CNOT ring is one precomposed basis permutation
  * gradient by adjoint differentiation: one forward sweep + one reverse sweep,
    instead of 2 * 72 = 144 extra forward passes for the parameter-shift rule.
    For Ry/Rz gates the adjoint gradient IS the analytic gradient that PSR
    estimates, so training finds the same parameters; deployment (inference on
    a QPU) is unchanged.

Readout
  "z"  : <Z_j>                          8 observables (Run 7 layout)
  "zz" : <Z_j> and <Z_i Z_j> for i<j   36 observables.  Every one is diagonal in
         the computational basis, so on hardware all 36 are estimated from the
         SAME Z-basis shots as the 8 single-qubit values -- no extra circuits.

Angle scale
  angle = ANGLE_SCALE * tanh(z).  Run 7 used pi.  With pi, |z| > 2 maps to
  |angle| > 0.96 pi where Ry(+a)|0> and Ry(-a)|0> are nearly indistinguishable
  (sign lives in sin(a), which -> 0).  pi/2 keeps every angle in a regime where
  cos and sin are both informative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

N_QUBITS   = 8
DIM        = 1 << N_QUBITS
N_LAYERS   = 4
N_FEATURES = N_QUBITS * N_LAYERS            # 32
N_ANSATZ   = 2 * N_QUBITS * N_LAYERS + N_QUBITS   # 72
_STRIDE    = 2 * N_QUBITS

_IDX   = np.arange(DIM)
_BITS  = np.stack([(_IDX >> (N_QUBITS - 1 - j)) & 1 for j in range(N_QUBITS)])   # (8, 256)
_ZSIGN = (1 - 2 * _BITS).astype(np.float64)                                       # (8, 256)
_PAIRS = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]        # 28
_ZZSIGN = np.stack([_ZSIGN[i] * _ZSIGN[j] for i, j in _PAIRS])                    # (28, 256)

READOUT_SIGNS = {"z": _ZSIGN, "zz": np.vstack([_ZSIGN, _ZZSIGN])}


def _cnot_perm(control: int, target: int) -> np.ndarray:
    cb = 1 << (N_QUBITS - 1 - control)
    tb = 1 << (N_QUBITS - 1 - target)
    return np.where((_IDX & cb) != 0, _IDX ^ tb, _IDX)

_RING = _IDX.copy()
for _i in range(N_QUBITS):
    _RING = _RING[_cnot_perm(_i, (_i + 1) % N_QUBITS)]
_RING_INV = np.empty(DIM, dtype=np.int64)
_RING_INV[_RING] = _IDX


# ---------------------------------------------------------------------------
#  Scaling / encoding
# ---------------------------------------------------------------------------
@dataclass
class AngleScaler:
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


def compute_angles(X_std: np.ndarray, scale: float = np.pi) -> np.ndarray:
    return scale * np.tanh(X_std)


# ---------------------------------------------------------------------------
#  Gate primitives (batched, in the (B, left, 2, right) view)
# ---------------------------------------------------------------------------
def _ry(state: np.ndarray, q: int, angles: np.ndarray) -> np.ndarray:
    """Ry(angles[b]) on qubit q for every sample b.  angles: (B,)"""
    B = state.shape[0]
    st = state.reshape(B, 1 << q, 2, 1 << (N_QUBITS - 1 - q))
    c = np.cos(angles / 2)[:, None, None]
    s = np.sin(angles / 2)[:, None, None]
    s0, s1 = st[:, :, 0, :], st[:, :, 1, :]
    out = np.empty_like(st)
    out[:, :, 0, :] = c * s0 - s * s1
    out[:, :, 1, :] = s * s0 + c * s1
    return out.reshape(B, DIM)


def _ry_grad(lam: np.ndarray, psi: np.ndarray, q: int) -> float:
    """sum_b 2 Re <lam_b | (-i/2) Y_q | psi_b>  =  Re sum (conj(l1) p0 - conj(l0) p1)."""
    B = psi.shape[0]
    p = psi.reshape(B, 1 << q, 2, 1 << (N_QUBITS - 1 - q))
    l = lam.reshape(B, 1 << q, 2, 1 << (N_QUBITS - 1 - q))
    return float(np.real(np.vdot(l[:, :, 1, :], p[:, :, 0, :]) - np.vdot(l[:, :, 0, :], p[:, :, 1, :])))


def _rz_phase(phis: np.ndarray) -> np.ndarray:
    """Diagonal of prod_i Rz_i(phis[i]) over the 256-dim basis."""
    return np.exp(-0.5j * (phis @ _ZSIGN))


# ---------------------------------------------------------------------------
#  Forward / expectations / adjoint backward
# ---------------------------------------------------------------------------
def forward(angles: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """angles (B, 32), theta (72,) -> final state (B, 256) complex128."""
    B = angles.shape[0]
    psi = np.zeros((B, DIM), dtype=np.complex128)
    psi[:, 0] = 1.0
    for L in range(N_LAYERS):
        base = _STRIDE * L
        a = angles[:, L * N_QUBITS:(L + 1) * N_QUBITS] + theta[base:base + N_QUBITS][None, :]
        for i in range(N_QUBITS):
            psi = _ry(psi, i, a[:, i])
        psi *= _rz_phase(theta[base + N_QUBITS: base + 2 * N_QUBITS])[None, :]
        psi = psi[:, _RING]
    fb = _STRIDE * N_LAYERS
    for i in range(N_QUBITS):
        psi = _ry(psi, i, np.full(B, theta[fb + i]))
    return psi


def expectations(psi: np.ndarray, readout: str = "zz") -> np.ndarray:
    probs = (psi.real ** 2 + psi.imag ** 2)
    return probs @ READOUT_SIGNS[readout].T           # (B, N_OBS)


def backward(psi: np.ndarray, g: np.ndarray, angles: np.ndarray, theta: np.ndarray,
             readout: str = "zz") -> np.ndarray:
    """Adjoint gradient of  sum_b sum_j g[b,j] <O_j>_b  w.r.t. theta (72,).

    psi: final state from forward().  g: (B, N_OBS) = dL/d<O_j> per sample
    (include any 1/B and class weights in g).
    """
    B = psi.shape[0]
    D = g @ READOUT_SIGNS[readout]                    # (B, 256) real
    lam = D * psi                                     # lambda = O psi
    psi = psi.copy()
    grad = np.zeros(N_ANSATZ)

    fb = _STRIDE * N_LAYERS
    for i in range(N_QUBITS):
        grad[fb + i] = _ry_grad(lam, psi, i)
    for i in range(N_QUBITS):
        a = np.full(B, -theta[fb + i])
        psi = _ry(psi, i, a); lam = _ry(lam, i, a)

    for L in reversed(range(N_LAYERS)):
        base = _STRIDE * L
        psi = psi[:, _RING_INV]; lam = lam[:, _RING_INV]
        # Rz grads: Im <lam | Z_i | psi>
        M = (np.conj(lam) * psi) @ _ZSIGN.T           # (B, 8) complex
        grad[base + N_QUBITS: base + 2 * N_QUBITS] = M.imag.sum(axis=0)
        ph = np.conj(_rz_phase(theta[base + N_QUBITS: base + 2 * N_QUBITS]))[None, :]
        psi = psi * ph; lam = lam * ph
        for i in range(N_QUBITS):
            grad[base + i] = _ry_grad(lam, psi, i)
        a_all = angles[:, L * N_QUBITS:(L + 1) * N_QUBITS] + theta[base:base + N_QUBITS][None, :]
        for i in range(N_QUBITS):
            psi = _ry(psi, i, -a_all[:, i]); lam = _ry(lam, i, -a_all[:, i])
    return grad


# ---------------------------------------------------------------------------
#  MLP readout  (N_OBS -> N_HIDDEN -> 1)
# ---------------------------------------------------------------------------
def mlp_sizes(readout: str, n_hidden: int) -> tuple[int, int]:
    n_in = READOUT_SIGNS[readout].shape[0]
    return n_in, n_in * n_hidden + n_hidden + n_hidden + 1


def _unpack(mp: np.ndarray, n_in: int, n_h: int):
    W1 = mp[: n_in * n_h].reshape(n_in, n_h)
    b1 = mp[n_in * n_h: n_in * n_h + n_h]
    W2 = mp[n_in * n_h + n_h: n_in * n_h + 2 * n_h]
    b2 = mp[-1]
    return W1, b1, W2, b2


def _mlp_forward(z: np.ndarray, mp: np.ndarray, n_in: int, n_h: int) -> np.ndarray:
    W1, b1, W2, b2 = _unpack(mp, n_in, n_h)
    h = np.maximum(0.0, z @ W1 + b1)
    return 1.0 / (1.0 + np.exp(-(h @ W2 + b2)))


def _mlp_backward(z, mp, wr, n_in, n_h):
    """wr: (B,) weighted residual already divided by B.  Returns (mlp_grad, dL_dz)."""
    W1, b1, W2, b2 = _unpack(mp, n_in, n_h)
    pre = z @ W1 + b1
    h = np.maximum(0.0, pre)
    dh = wr[:, None] * W2[None, :]
    dpre = dh * (pre > 0)
    dW1 = z.T @ dpre
    db1 = dpre.sum(axis=0)
    dW2 = h.T @ wr
    db2 = wr.sum()
    return np.concatenate([dW1.ravel(), db1, dW2, [db2]]), dpre @ W1.T


# ---------------------------------------------------------------------------
#  Model API
# ---------------------------------------------------------------------------
@dataclass
class Model:
    readout:  str = "zz"
    n_hidden: int = 32

    @property
    def n_in(self) -> int:  return mlp_sizes(self.readout, self.n_hidden)[0]
    @property
    def n_mlp(self) -> int: return mlp_sizes(self.readout, self.n_hidden)[1]
    @property
    def n_params(self) -> int: return N_ANSATZ + self.n_mlp

    def init_params(self, rng: np.random.Generator, ansatz: Optional[np.ndarray] = None) -> np.ndarray:
        n_in, _ = mlp_sizes(self.readout, self.n_hidden)
        mlp = np.concatenate([
            rng.normal(scale=np.sqrt(2.0 / n_in), size=n_in * self.n_hidden),
            np.zeros(self.n_hidden),
            rng.normal(scale=np.sqrt(2.0 / self.n_hidden), size=self.n_hidden),
            [0.0],
        ])
        a = ansatz.copy() if ansatz is not None else rng.normal(scale=0.1, size=N_ANSATZ)
        return np.concatenate([a, mlp])

    def predict_proba(self, angles: np.ndarray, params: np.ndarray, chunk: int = 4000) -> np.ndarray:
        out = []
        for s in range(0, len(angles), chunk):
            z = expectations(forward(angles[s:s + chunk], params[:N_ANSATZ]), self.readout)
            out.append(_mlp_forward(z, params[N_ANSATZ:], self.n_in, self.n_hidden))
        return np.concatenate(out)

    def loss_and_grad(self, params, angles, y, class_weight, l2: float = 1e-4):
        theta, mp = params[:N_ANSATZ], params[N_ANSATZ:]
        B = len(y)
        psi = forward(angles, theta)
        z = expectations(psi, self.readout)
        p = np.clip(_mlp_forward(z, mp, self.n_in, self.n_hidden), 1e-7, 1 - 1e-7)
        sw = np.where(y == 1, class_weight[1], class_weight[0])
        loss = float((sw * -(y * np.log(p) + (1 - y) * np.log(1 - p))).mean())
        wr = sw * (p - y) / B
        mlp_grad, g = _mlp_backward(z, mp, wr, self.n_in, self.n_hidden)
        qgrad = backward(psi, g, angles, theta, self.readout)
        if l2 > 0:
            loss += l2 * float((theta ** 2).mean())
            qgrad += 2.0 * l2 * theta / N_ANSATZ
        return loss, np.concatenate([qgrad, mlp_grad])


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------
def train(model: Model, params: np.ndarray, ang_tr, y_tr, ang_val, y_val, *,
          n_steps=1500, batch_size=2000, lr=0.01, lr_min_frac=0.05, warmup=20,
          l2=1e-4, class_weight=None, seed=0, eval_every=25, patience=None,
          log=print) -> tuple[np.ndarray, dict]:
    """Adam with warmup + cosine decay; keeps the best-val-AUC checkpoint."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    if class_weight is None:
        class_weight = np.ones(2)
    m = np.zeros_like(params); v = np.zeros_like(params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(y_tr); bs = min(batch_size, n)
    best = {"val_auc": -1.0, "step": 0, "params": params.copy()}
    hist = {"step": [], "loss": [], "val_acc": [], "val_auc": [], "lr": []}
    ema = None; since_best = 0
    for t in range(1, n_steps + 1):
        if t <= warmup:
            lr_t = lr * t / warmup
        else:
            frac = (t - warmup) / max(1, n_steps - warmup)
            lr_t = lr * (lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + np.cos(np.pi * frac)))
        idx = rng.choice(n, size=bs, replace=False)
        loss, g = model.loss_and_grad(params, ang_tr[idx], y_tr[idx], class_weight, l2)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        params = params - lr_t * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        ema = loss if ema is None else 0.9 * ema + 0.1 * loss
        if t % eval_every == 0 or t == n_steps:
            p = model.predict_proba(ang_val, params)
            acc = float(((p >= 0.5) == y_val).mean()); auc = float(roc_auc_score(y_val, p))
            hist["step"].append(t); hist["loss"].append(ema); hist["val_acc"].append(acc)
            hist["val_auc"].append(auc); hist["lr"].append(lr_t)
            flag = ""
            if auc > best["val_auc"]:
                best = {"val_auc": auc, "val_acc": acc, "step": t, "params": params.copy()}; flag = " *"; since_best = 0
            else:
                since_best += eval_every
            log(f"  step {t:5d}/{n_steps}  loss={ema:.4f}  lr={lr_t:.5f}  val_acc={acc:.4f}  val_auc={auc:.4f}{flag}")
            if patience and since_best >= patience:
                log(f"  early stop: no val-AUC gain for {patience} steps"); break
    return best["params"], {"best_step": best["step"], "best_val_auc": best["val_auc"],
                            "best_val_acc": best.get("val_acc"), "history": hist}


# ---------------------------------------------------------------------------
#  Verification against the original PSR implementation
# ---------------------------------------------------------------------------
def verify(B: int = 64, seed: int = 0) -> None:
    import time
    from Vqc_angle8 import circuit_forward as ref_forward, expectation_all
    from Vqc_hybrid8 import _gradient as ref_gradient, _bce as ref_bce, N_PARAMS as REF_N
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=(B, N_FEATURES))
    theta = rng.normal(scale=0.7, size=N_ANSATZ)
    psi_ref = ref_forward(angles, theta); psi = forward(angles, theta)
    print(f"state   max|diff| = {np.abs(psi - psi_ref).max():.2e}")
    print(f"<Z>     max|diff| = {np.abs(expectations(psi, 'z') - expectation_all(psi_ref)).max():.2e}")

    model = Model(readout="z", n_hidden=32)
    params = model.init_params(rng, ansatz=theta); assert len(params) == REF_N
    y = rng.integers(0, 2, size=B); cw = np.array([0.787, 1.372])
    loss, grad = model.loss_and_grad(params, angles, y, cw, l2=1e-4)
    t0 = time.perf_counter(); g_ref = ref_gradient(params, angles, y, 1e-4, cw); t_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); model.loss_and_grad(params, angles, y, cw, l2=1e-4); t_new = time.perf_counter() - t0
    print(f"loss    ref={ref_bce(params, angles, y, 1e-4, cw):.10f}  new={loss:.10f}")
    print(f"grad    max|diff| = {np.abs(grad - g_ref).max():.2e}   (max|grad| = {np.abs(g_ref).max():.3e})")
    print(f"time    PSR={t_ref:.2f}s  adjoint={t_new:.3f}s  speedup={t_ref / t_new:.0f}x   (batch={B})")

    # finite-difference check of the zz-readout gradient (no reference implementation)
    model2 = Model(readout="zz", n_hidden=32); p2 = model2.init_params(rng, ansatz=theta)
    _, g2 = model2.loss_and_grad(p2, angles, y, cw, l2=0.0)
    errs = []
    for k in list(range(0, N_ANSATZ, 7)) + [N_ANSATZ + 3, len(p2) - 1]:
        e = np.zeros_like(p2); e[k] = 1e-5
        lp, _ = model2.loss_and_grad(p2 + e, angles, y, cw, 0.0); lm, _ = model2.loss_and_grad(p2 - e, angles, y, cw, 0.0)
        errs.append(abs((lp - lm) / 2e-5 - g2[k]))
    print(f"zz-readout finite-diff max|err| = {max(errs):.2e}")


if __name__ == "__main__":
    verify()
