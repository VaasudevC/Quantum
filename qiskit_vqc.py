"""Qiskit (and PennyLane) versions of the Vqc_fast8 circuit, plus helpers for the hardware stages.

Circuit (identical to Vqc_fast8.forward), 8 qubits, 4 layers:
    for L in 0..3:  Ry(angle[8L+i] + theta[16L+i]) on qubit i
                    Rz(theta[16L+8+i])             on qubit i
                    CNOT i -> (i+1) % 8, applied in order i = 0, 1, ..., 7
    final:          Ry(theta[64+i]) on qubit i;  measure all
Gate conventions match: Ry = [[c, -s], [s, c]], Rz = diag(e^{-i phi/2}, e^{+i phi/2}).

Bit ordering. Vqc_fast8 is big-endian: qubit 0 is the MOST significant bit of the 256-dim state index.
Qiskit is little-endian: qubit q is bit q of the index, and the printed bitstring reads c[7]...c[0].
We keep the qubit labels identical (Vqc qubit j == Qiskit qubit j, measured into clbit j), and convert
only at the boundary:
  * statevector:  Statevector(qc).reverse_qargs().data  ==  Vqc_fast8.forward(...)[0]
  * counts:       bit of qubit j = bitstring[::-1][j]  (equivalently bitstring[7 - j])
PennyLane uses the same big-endian order as Vqc_fast8 (wire 0 = MSB), so no conversion is needed there.

The 6 hardware models: fold 0, selector l1, 1000 steps, k in {8, 32} x scale in {pi, pi/2, pi/6},
taken from results/stage2_steps1000.jsonl. No retraining: params = 72 ansatz + 8 weights + 1 bias.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import Vqc_fast8 as V

NQ, NL = V.N_QUBITS, V.N_LAYERS
SCALES = {"pi": np.pi, "pi/2": np.pi / 2, "pi/6": np.pi / 6}
MODEL_KEYS = [(k, s) for k in (8, 32) for s in ("pi", "pi/2", "pi/6")]


# ----------------------------------------------------------------------------- models / data
def load_models(path=ROOT / "results/stage2_steps1000.jsonl", fold=0):
    """{(k, scale): row} for the 6 hardware models."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["fold"] == fold and r["selector"] == "l1" and r["steps"] == 1000 and not r["shuffle_later"] \
                and (r["k"], r["scale"]) in MODEL_KEYS:
            out[(r["k"], r["scale"])] = r
    assert len(out) == 6, f"expected 6 models, found {sorted(out)}"
    return out


def model_data(row):
    """Angles (N, 32) for every ECG, labels y, and index arrays tr/va/te, rebuilt exactly as in scale_cv.py.
    Uses the run's logged feature list (the l1 order can swap near-tied features between runs)."""
    import scale_cv as S
    _, y, _, _, feats = S.load()
    Z, tr, va, te, _ = S.prep(row["split"], row["fold"], row["selector"])
    sel = [feats.index(n) for n in row["features"]]
    A = np.zeros((len(y), V.N_FEATURES))
    A[:, :row["k"]] = eval(row["scale"], {"pi": np.pi}) * np.tanh(Z[:, sel])
    return A, y, tr, va, te


def split_params(row):
    p = np.asarray(row["params"])
    return p[:V.N_ANSATZ], p[V.N_ANSATZ:V.N_ANSATZ + 8], p[-1]


def logit_from_z(z, w, b):
    return np.asarray(z) @ w + b


def stratified_subset(y_te, n, seed=0):
    """n test positions (indices into the test array), class-stratified, fixed seed."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_te == 1), np.flatnonzero(y_te == 0)
    n_pos = int(round(n * len(pos) / len(y_te)))
    return np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n - n_pos, replace=False)]))


def z_from_counts(counts):
    """<Z_j> for j = 0..7 from a Qiskit counts dict (little-endian bitstrings)."""
    shots = sum(counts.values()); z = np.zeros(NQ)
    for bs, c in counts.items():
        bits = bs.replace(" ", "")[::-1]          # bits[j] = qubit j
        z += c * np.array([1 - 2 * int(bits[j]) for j in range(NQ)])
    return z / shots


def z_from_bitarray(bitarray):
    """<Z_j> from a SamplerV2 BitArray (one PUB). Rows of get_bitstrings() are little-endian."""
    return z_from_counts(bitarray.get_counts())


# ----------------------------------------------------------------------------- Qiskit
def build_circuit(angles, theta, measure=True):
    """Concrete circuit for one ECG. angles (32,), theta (72,)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(NQ, NQ if measure else 0)
    for L in range(NL):
        base = 2 * NQ * L
        for i in range(NQ):
            qc.ry(float(angles[NQ * L + i] + theta[base + i]), i)
        for i in range(NQ):
            qc.rz(float(theta[base + NQ + i]), i)
        for i in range(NQ):
            qc.cx(i, (i + 1) % NQ)
    for i in range(NQ):
        qc.ry(float(theta[2 * NQ * NL + i]), i)
    if measure:
        qc.measure(range(NQ), range(NQ))
    return qc


def build_template(theta, k=V.N_FEATURES, measure=True):
    """Parametrised circuit: theta is fixed, the k used angles are Parameters x[0..k-1]
    (unused angles are 0 and folded in). Transpile once, then bind one row of angles per ECG."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    x = ParameterVector("x", k)
    qc = QuantumCircuit(NQ, NQ if measure else 0)
    for L in range(NL):
        base = 2 * NQ * L
        for i in range(NQ):
            j = NQ * L + i
            qc.ry((x[j] + float(theta[base + i])) if j < k else float(theta[base + i]), i)
        for i in range(NQ):
            qc.rz(float(theta[base + NQ + i]), i)
        for i in range(NQ):
            qc.cx(i, (i + 1) % NQ)
    for i in range(NQ):
        qc.ry(float(theta[2 * NQ * NL + i]), i)
    if measure:
        qc.measure(range(NQ), range(NQ))
    return qc, x


def statevector_bigendian(qc):
    """Statevector of an unmeasured circuit, reordered to Vqc_fast8's big-endian convention."""
    from qiskit.quantum_info import Statevector
    return Statevector(qc).reverse_qargs().data


# ----------------------------------------------------------------------------- PennyLane
def pennylane_qnode(theta, device="lightning.qubit", shots=None):
    """QNode returning the 8 <Z_j>. Wire 0 = MSB, same as Vqc_fast8. Supports parameter broadcasting
    (angles of shape (B, 32)) on default.qubit."""
    import pennylane as qml
    dev = qml.device(device, wires=NQ)

    def circuit(angles):
        for L in range(NL):
            base = 2 * NQ * L
            for i in range(NQ):
                qml.RY(angles[..., NQ * L + i] + theta[base + i], wires=i)
            for i in range(NQ):
                qml.RZ(theta[base + NQ + i], wires=i)
            for i in range(NQ):
                qml.CNOT(wires=[i, (i + 1) % NQ])
        for i in range(NQ):
            qml.RY(theta[2 * NQ * NL + i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(NQ)]

    qn = qml.QNode(circuit, dev)
    return qml.set_shots(qn, shots) if shots else qn
