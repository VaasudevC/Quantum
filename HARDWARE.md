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

## H2 — finite-shot robustness (2026-11-05)

`python -u hw_h2_shots.py` → `results/hw/h2_shots.{jsonl,md}`, `results/figs/hw_shots.png`. Exact probabilities come from Vqc_fast8;
counts are drawn with a numpy multinomial and turned into ⟨Z_j⟩ and then the logged linear readout. All 2,993 fold-0 test ECGs, 10 repeats per shot count
(mean ± SD over repeats). The runbook grid {100, 300, 1000, 4000, 10000} was extended with 30 and 2000. "Shots for loss ≤ x" is log-interpolated on the grid.

| model | noiseless | 30 shots | 100 shots | 300 shots | 1000 shots | 2000 shots | 4000 shots | 10000 shots | shots for loss ≤ 0.01 | ≤ 0.005 | ‖w‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k=8 pi | 0.9243 | 0.8613 ± 0.0045 | 0.9018 ± 0.0026 | 0.9160 ± 0.0021 | 0.9216 ± 0.0010 | 0.9232 ± 0.0004 | 0.9239 ± 0.0004 | 0.9241 ± 0.0002 | 243 | 512 | 8.09 |
| k=8 pi/2 | 0.9347 | 0.8994 ± 0.0033 | 0.9233 ± 0.0014 | 0.9302 ± 0.0016 | 0.9339 ± 0.0006 | 0.9340 ± 0.0003 | 0.9344 ± 0.0004 | 0.9346 ± 0.0002 | 118 | 265 | 6.55 |
| k=8 pi/6 | 0.9349 | 0.8775 ± 0.0066 | 0.9152 ± 0.0018 | 0.9286 ± 0.0015 | 0.9328 ± 0.0006 | 0.9336 ± 0.0005 | 0.9342 ± 0.0002 | 0.9346 ± 0.0002 | 192 | 386 | 6.13 |
| k=32 pi | 0.8612 | 0.7981 ± 0.0075 | 0.8416 ± 0.0025 | 0.8539 ± 0.0017 | 0.8591 ± 0.0010 | 0.8599 ± 0.0006 | 0.8608 ± 0.0003 | 0.8610 ± 0.0002 | 213 | 438 | 6.19 |
| k=32 pi/2 | 0.9251 | 0.8250 ± 0.0047 | 0.8903 ± 0.0021 | 0.9118 ± 0.0024 | 0.9205 ± 0.0013 | 0.9231 ± 0.0007 | 0.9246 ± 0.0007 | 0.9249 ± 0.0004 | 414 | 909 | 10.69 |
| k=32 pi/6 | 0.9598 | 0.9038 ± 0.0028 | 0.9413 ± 0.0027 | 0.9539 ± 0.0017 | 0.9578 ± 0.0005 | 0.9590 ± 0.0006 | 0.9592 ± 0.0005 | 0.9596 ± 0.0002 | 179 | 355 | 8.08 |

**Key question: does pi/6 lose more AUC than pi/2 under shot noise? At k=32, no; it loses less.** pi/6 needs **≈180 shots** to stay
within 0.01 of noiseless (≈360 for 0.005), against ≈410 (≈910) for pi/2. After training, the k=32 pi/6 model actually has a *larger*
class gap in ⟨Z⟩ (max 0.32 vs 0.23 for pi/2) and a smaller readout norm (‖w‖ 8.1 vs 10.7), so shot noise is amplified less.
Small input angles do not mean small output differences, because the trained ansatz re-spreads them. At k=8, pi/6 is slightly more
shot-sensitive than pi/2 (≈190 vs ≈120 shots for 0.01). At every shot count, including 30, k=32 pi/6 is the best of the 6 models.
With 300 shots it reaches 0.954, above the *noiseless* AUC of every other model. **1000 shots per ECG costs ≤ 0.002 AUC for all 6 models.**
