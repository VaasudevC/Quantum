"""Qiskit circuits for the magic-state brick-wall QNN (Antonio's model, our reconstruction) + verification against bw_torch/bw_utils.

Qubit j = wire j (little-endian qiskit index bit j).  Gate set: X, H, CX (magic prep), Z (sign layer of the spreading),
RBS = SingleExcitation(phi) = XXPlusYYGate(phi, beta=pi/2) on adjacent qubits (spreading Givens network and the trainable columns),
RZ (data phases + trainable phases), measure all qubits.

Spreading: the single-particle matrix W (Walsh-Hadamard n=8 / Hadamard-12) is compiled EXACTLY into n(n-1)/2 nearest-neighbour RBS
gates (Givens/QR) plus a +-1 diagonal (Z gates on the input):  W = G_1^-1 ... G_M^-1 D.  The ring wrap pair (n-1, 0) is only used by
odd trainable columns (as in the notebook).
"""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import XXPlusYYGate
import bw_utils as B


def givens_decompose(W):
    """returns (gates, signs): gates = list of (a, phi) in circuit order (apply first -> last), and signs d (+-1) s.t. the many-body
    circuit  [Z on qubits with d=-1] then SingleExcitation(phi_m) on (a_m, a_m+1)  implements U_had on a fixed-particle-number input."""
    n = len(W); A = W.copy(); rots = []
    for j in range(n - 1):
        for i in range(n - 1, j, -1):
            a = i - 1
            phi = 2 * np.arctan2(A[i, j], A[a, j])
            G = B.givens_single_particle(n, a, i, phi)
            A = G @ A
            rots.append((a, phi))
    d = np.sign(np.diag(A)); assert np.allclose(np.abs(np.diag(A)), 1, atol=1e-9) and np.allclose(A, np.diag(np.diag(A)), atol=1e-9)
    gates = [(a, -phi) for (a, phi) in reversed(rots)]           # G_M^-1 first ... G_1^-1 last
    return gates, d


def rbs(qc, phi, a, b):
    """SingleExcitation(phi) on (a, b) in PennyLane's |q_a q_b> basis."""
    qc.append(XXPlusYYGate(phi, np.pi / 2), [b, a])              # orientation fixed by numerical test (hw_bw.check_rbs)


def check_rbs():
    from qiskit.quantum_info import Operator
    phi = 0.7; c, s = np.cos(phi / 2), np.sin(phi / 2)
    SE = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], complex)      # pennylane basis |q_a q_b>, a = MSB
    qc = QuantumCircuit(2); rbs(qc, phi, 0, 1)
    U = Operator(qc).data                                           # qiskit little-endian: index = q1*2 + q0 ; q0 = a
    perm = [0, 2, 1, 3]                                             # swap bit order to get |q_a q_b> ordering (a MSB)
    return float(np.abs(U[np.ix_(perm, perm)] - SE).max())


def build(n, columns, k, W=None, data_params=True, measure=True):
    """parametrised circuit: data phases theta_j (ParameterVector 'x') ; trainable w bound later via bind_weights."""
    W = B.spreading_matrix(n) if W is None else W
    gates, d = givens_decompose(W)
    q = QuantumRegister(n, "q"); c = ClassicalRegister(n, "c")
    qc = QuantumCircuit(q, c) if measure else QuantumCircuit(q)
    for blk in range(k // 2):
        a, b_, cc, dd = range(4 * blk, 4 * blk + 4)
        qc.x(a); qc.x(b_); qc.h(cc)
        for t in (dd, a, b_):
            qc.cx(cc, t)
    for j in range(n):
        if d[j] < 0:
            qc.z(j)
    for (a, phi) in gates:
        rbs(qc, phi, a, a + 1)
    th = ParameterVector("x", n)
    for j in range(n):
        qc.rz(th[j], j)                                              # th = -phi_data = RZ angle
    wv = ParameterVector("w", columns * 3 * n // 2)
    for col in range(columns):
        for j in range(n):
            qc.rz(-wv[col * 3 * n // 2 + n // 2 + j], j)
        for j, (a, b_) in enumerate(B.pairs(n, col)):
            rbs(qc, -2 * wv[col * 3 * n // 2 + j], a, b_)
    if measure:
        qc.measure(q, c)
    return qc, th, wv


def bind(qc, th, wv, theta, w):
    vals = {**{th[j]: float(theta[j]) for j in range(len(th))}, **{wv[i]: float(v) for i, v in enumerate(np.asarray(w).ravel())}}
    return qc.assign_parameters(vals)


def features_from_probs(p, n):
    """p: probability vector over little-endian bitstrings (length 2^n) -> f_j = 4 P(q_j = 1 and q_{j+1} = 1)."""
    idx = np.arange(len(p)); bit = lambda q: (idx >> q) & 1
    return np.array([4 * (p * bit(j) * bit((j + 1) % n)).sum() for j in range(n)])


def features_from_counts(counts, n, k=None):
    """counts: dict bitstring (qiskit order, c[n-1]...c[0]) -> shots.  Returns (f, kept_fraction); k -> particle-number post-selection."""
    tot = sum(counts.values()); acc = np.zeros(n); kept = 0
    for bs, c in counts.items():
        bits = [int(ch) for ch in bs.replace(" ", "")[::-1]]          # bits[j] = qubit j
        if k is not None and sum(bits) != k:
            continue
        kept += c
        for j in range(n):
            if bits[j] and bits[(j + 1) % n]:
                acc[j] += c
    return (4 * acc / max(kept, 1)), kept / tot


def check(n, columns, k, trials=5, seed=0):
    """compare Qiskit statevector features to the torch/numpy sector simulator.  Returns max abs diff."""
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(seed)
    qc, th, wv = build(n, columns, k, measure=False)
    S = B.Sector(n, columns, k)
    worst = 0.0
    for _ in range(trials):
        x = rng.uniform(-1, 1, n); w = rng.normal(0, 0.8, (columns, 3 * n // 2))
        bc = bind(qc, th, wv, -np.pi * x, w)
        p = Statevector(bc).probabilities()
        f_q = features_from_probs(p, n); f_s = S.features(x, w)
        worst = max(worst, float(np.abs(f_q - f_s).max()))
    return worst


if __name__ == "__main__":
    print("rbs orientation error", check_rbs())
    for n, ks in ((8, (2, 4)), (12, (2, 4, 6))):
        for k in ks:
            print(n, k, "max |f_qiskit - f_sim| =", check(n, 2, k, trials=3))
