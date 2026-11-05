# HARDWARE — does the pi/6 advantage survive shots, noise and real devices?

Models ("the 6 models"): fold 0, selector l1, 1000 steps, k ∈ {8, 32} × scale ∈ {pi, pi/2, pi/6}, trained parameters taken
from `results/stage2_steps1000.jsonl` (72 ansatz + 8 readout weights + 1 bias). **Nothing is retrained.** Test set = fold-0 test rows
(2,993 ECGs). Code: `qiskit_vqc.py` (Qiskit + PennyLane circuits, model loader, count decoding).

## H0 — framework choice: Qiskit vs PennyLane (2026-11-05)

PennyLane was the mentor's recommendation, so both were installed (qiskit 2.6.1, qiskit-aer 0.18.2, qiskit-ibm-runtime 0.51.0,
pennylane 0.45.1), checked for exact equivalence (H1), and timed on the real workload: the k=32, pi/6 circuit on 1,000 test ECGs, 4 CPUs.

| engine | exact ⟨Z_j⟩, 1000 ECGs | 1000 shots each, 1000 ECGs |
|---|---|---|
| `Vqc_fast8` (our numpy, batched) | **0.13 s** | – |
| PennyLane `default.qubit` (parameter broadcasting) | 0.32 s | – |
| Qiskit Aer `EstimatorV2` | 0.93 s | – |
| PennyLane `lightning.qubit` (one call per ECG) | 4.4 s | 7.6 s |
| Qiskit Aer `SamplerV2` | – | **3.1 s** |

**Reading.** For exact noiseless simulation PennyLane's broadcasting is about 3× faster than Aer, but our own simulator is faster still. With shots,
Aer is about 2.4× faster than PennyLane-lightning. Speed is a minor factor anyway (every stage runs in seconds to minutes). The deciding factor is capability:

| need | Qiskit | PennyLane |
|---|---|---|
| compile to IBM Heron heavy-hex (SWAP count, depth, 2q depth) | native transpiler | none of its own; calls Qiskit via `pennylane-qiskit` |
| noise model from real-device calibration (FakeFez, …), no account needed | native (`fake_provider` + Aer) | only via Qiskit Aer through the plugin, or hand-written channels |
| real IBM job + a local dry run of the exact same job | native (Qiskit Runtime, local mode) | wraps Qiskit Runtime via the plugin |
| differentiable training | possible | **stronger** (adjoint, autodiff) |

**Decision.** H2 and H5 use `Vqc_fast8` (fastest, exact). H3 and H4 use Qiskit, because they are IBM-specific and PennyLane would
call Qiskit underneath. The PennyLane QNode (`qiskit_vqc.pennylane_qnode`) is kept as a verified equivalent for the
mentor, for any future training, and for non-IBM hardware (Amazon Braket, IonQ) through PennyLane plugins. Real
hardware needs a provider account whichever framework is used.

## H1 — Qiskit circuit + equivalence check (2026-11-05)

`python -u hw_h1_verify.py` → `results/hw/h1_verify.json`.

- Circuit: per layer, Ry(angle+θ) on each qubit, then Rz(θ), then CNOT i→(i+1) mod 8 in order i = 0…7; a final Ry layer; measure all.
  Gate matrices match Vqc_fast8 exactly (Ry = [[c, −s], [s, c]], Rz = diag(e^{−iφ/2}, e^{iφ/2})).
- **Bit ordering.** Vqc_fast8 is big-endian (qubit 0 = most significant bit of the state index); Qiskit is little-endian. Qubit labels are kept
  identical (Vqc qubit j = Qiskit qubit j, measured into clbit j), and only the boundary is converted:
  `Statevector(qc).reverse_qargs()` for statevectors and `bitstring[::-1][j]` for counts. PennyLane is big-endian like Vqc_fast8.
- Statevector, 20 random inputs and random θ: **max|diff| = 1.5e-16** (Qiskit), ⟨Z⟩ max|diff| = 5.9e-16 (PennyLane),
  1.5e-15 (Aer EstimatorV2).
- The 6 models, noiseless Qiskit predictions on all 2,993 fold-0 test ECGs, reproduce the logged test AUC **exactly (|diff| = 0)**:

| model | k=8 | k=32 |
|---|---|---|
| pi   | 0.9243 | 0.8612 |
| pi/2 | 0.9347 | 0.9251 |
| pi/6 | 0.9349 | **0.9598** |

**H1 PASS.**
