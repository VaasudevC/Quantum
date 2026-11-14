# HARDWARE — does the pi/6 advantage survive shots, noise and real devices?

> Part of a three-phase study: [`RESULTS.md`](RESULTS.md) (classical) · [`HARDWARE.md`](HARDWARE.md) (hardware readiness) · [`NEXT.md`](NEXT.md) (improvements). Chronological log, decisions and problems: [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md); commands: [`docs/REPRODUCE.md`](docs/REPRODUCE.md); code and file formats: [`docs/CODE_AND_DATA.md`](docs/CODE_AND_DATA.md).

Models ("the 6 models"): fold 0, selector l1, 1000 steps, k ∈ {8, 32} × scale ∈ {pi, pi/2, pi/6}, trained parameters taken
from `results/stage2_steps1000.jsonl` (72 ansatz + 8 readout weights + 1 bias). **Nothing is retrained.** Test set = fold-0 test rows
(2,993 ECGs). Code: `qiskit_vqc.py` (Qiskit + PennyLane circuits, model loader, count decoding).

## Headline

**Does the pi/6 advantage survive? In simulation of shots and IBM-Heron noise, yes. On real ibm_marrakesh hardware (Heron r2, 200 ECGs, 1000 shots, 2026-11-14): bw8 AUC 0.949 raw (0.925 noiseless), vqc32 AUC 0.956 raw (0.967 noiseless). Both well above 0.90; AUC survives real hardware noise.**
The k=32 pi/6 model stays the best of the 6 at every step of the hardware chain (fold-0 test ECGs):
- **Shot noise (H2, 2,993 ECGs):** 0.960 noiseless → 0.958 at 1000 shots → 0.954 at 300 → 0.941 at 100. It needs ≈180 shots to stay
  within 0.01 of noiseless, *fewer* than pi/2 (≈410), because the trained model has a larger ⟨Z⟩ class gap and a smaller readout norm. Even at
  300 shots it beats the noiseless AUC of every other model (pi/2 0.925, pi 0.861).
- **Compilation (H3a):** the 8-qubit CNOT ring needs 16 SWAPs on heavy-hex (32 → 80 CZ, 12 physical qubits). The estimated success
  probability is 0.705 on Heron r2 (FakeFez), 0.578 on Torino and 0.481 on Eagle. An all-to-all ion-trap-like target needs no SWAPs.
- **Device noise (H3b, FakeFez noise model, 300 ECGs, 2000 shots):** AUC 0.961 noiseless → **0.960 noisy** (pi/2: 0.934 → 0.937; pi: 0.874 → 0.871).
  On the 100-ECG best-compile check: 0.983 → 0.984. Noise contracts ⟨Z⟩ by 0.71–0.88×, which preserves ranking (AUC) but breaks the 0.5 threshold
  (balanced accuracy 0.883 → 0.857, Brier 0.080 → 0.126). One global rescale restores both. Readout mitigation changes neither AUC nor calibration much.
- **Real hardware (H6, 2026-11-14): bw8 AUC 0.949, vqc32 AUC 0.956 on ibm_marrakesh (200 ECGs, 1000 shots).** Both models survive real hardware noise. See H6 for full results.
- **H4 plan (not run, superseded by H6):** The original H4 plan (`hw_run_ibm.py`, 100 ECGs, k=32 pi/6 and pi) was not run. H6 uses `hw_run_ibm_brickwall.py` with both bw8 and vqc32 on 200 ECGs.
- **Training-free encoding score (H5):** trace distance between class-mean encoded states is a weak predictor of trained AUC
  (Spearman 0.34; 0.71 within k=32), so it is not a usable selector. Post hoc, the purity of the class-mean states tracks AUC (0.80) and flags over-large scales.
- **Framework (H0):** Qiskit is used for the device stages (transpiler, calibrated noise models, Runtime). The PennyLane circuit is kept and verified identical.
  PennyLane is about 3× faster than Aer for exact simulation, Aer about 2.4× faster with shots, and our numpy simulator is faster than both.

**Gate counts, k=32 pi/6 circuit** (from H3a):

| target | 1q | 2q | depth | 2q depth | SWAPs | est. success |
|---|---|---|---|---|---|---|
| logical | 72 | 32 (CX) | 42 | 32 | – | – |
| IBM Heron r2, FakeFez | 371 | 80 (CZ) | 205 | 80 | 16 | 0.705 |
| IBM Heron r1, FakeTorino | 371 | 80 (CZ) | 207 | 80 | 16 | 0.578 |
| IBM Eagle, FakeSherbrooke | 551 | 80 (ECR) | 296 | 80 | 16 | 0.481 |
| IonQ-like all-to-all | 193 | 32 (RXX) | 106 | 32 | 0 | 0.743 (assumed fidelities) |

**Figures** (`results/figs/`, 150 dpi): `hw_shots.png` (AUC vs shots, log-x, 6 models); `hw_noisy.png` (noiseless / shots-only / noisy /
mitigated per model); `encoding_score.png` (trace distance vs scale, and vs trained AUC).

**Limitations.** Fold 0 only; 300 ECGs (H3) and 100 ECGs (best-compile check, H4 plan), so AUC differences below about 0.01 are within sampling error.
Noise comes from one *simulated* device (FakeFez calibration snapshot of 2025-02-26), i.e. one device and one calibration day, and simulated noise models
omit crosstalk, drift and leakage. The 300-ECG noisy run uses an 8-qubit-line compile (109–117 CZ) that is more pessimistic than the best compile.
The estimated success probability ignores idle decoherence. The global rescale used for calibration relies on noiseless simulation of the same circuits.
H5 correlations rest on 13 (7) configurations with mixed training lengths, and the purity result is post hoc. **No real-hardware data yet.**

---

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

## H3a — compilation to real device layouts, k=32 pi/6 circuit (2026-11-05)

`python -u hw_h3_compile.py` → `results/hw/h3_compile.{json,md}`. optimization_level=3, 5 transpiler seeds, keep the fewest 2q gates.
Estimated success = product of (1 − error) over every gate and measurement, from the fake backend's calibration snapshot
(idle decoherence is not included).

| target | native 2q | 1q gates | 2q gates | depth | 2q depth | SWAPs added | est. success |
|---|---|---|---|---|---|---|---|
| logical circuit | cx | 72 | 32 | 42 | 32 | – | – |
| fake_fez | cz | 371 | 80 | 205 | 80 | 16 | 0.705 |
| fake_torino | cz | 371 | 80 | 207 | 80 | 16 | 0.578 |
| fake_sherbrooke | ecr | 551 | 80 | 296 | 80 | 16 | 0.481 |
| ionq_like_all_to_all | rxx | 193 | 32 | 106 | 32 | 0 | 0.743 (assumed fidelities) |

- **SWAPs.** The CNOT ring 0→1→…→7→0 is an 8-cycle, and heavy-hex has no 8-cycles (its smallest ring has 12 qubits), so the router adds **16 SWAPs**
  (4 per layer; 1 SWAP = 3 CZ/ECR). The 2q-gate count goes from 32 to **80**, and the circuit uses 12 physical qubits. The 2q depth equals the
  2q count because the ring is inherently sequential: each CNOT shares a qubit with the next.
- Heron r2 (FakeFez) is clearly the best IBM target (estimated success 0.705 vs 0.578 Torino, 0.481 Eagle/Sherbrooke). Eagle's ECR basis needs 50% more
  1q gates.
- An IonQ-like all-to-all target needs **no SWAPs** (32 RXX). Its success figure uses *assumed* fidelities (1q 0.9995, 2q 0.995, readout 0.995),
  not calibration data, so it is illustrative only.

## H3b — noisy simulation, IBM Heron noise model (FakeFez) (2026-11-05)

`python -u hw_h3_noisy.py` (and `--best12 --n 100 --models "32:pi/6,32:pi"`), then `python hw_h3_plot.py` → `results/hw/h3_noisy.md`,
`results/figs/hw_noisy.png`. Uses `AerSimulator.from_backend(FakeFez())` (density matrix, which includes gate, thermal-relaxation and readout noise), on 300 class-stratified
fold-0 test ECGs (the same 300 for every model), 2000 shots each, on transpiled circuits. Readout mitigation uses a per-qubit confusion matrix measured on the
same physical qubits (all-|0⟩ and all-|1⟩ calibration circuits, 20k shots), applied to ⟨Z_j⟩ as
(⟨Z⟩_raw − (e1 − e0)) / (1 − e0 − e1).

**Compile used for the 300-ECG run.** The best compile (H3a: 80 CZ, 12 active qubits) takes about 8 s per circuit to simulate as a density matrix, about 4 h for this run.
The circuits were instead routed on the best 8-qubit *line* of FakeFez (qubits 126-125-124-123-136-143-142-141, chosen by lowest summed CZ + readout error),
which needs 109–117 CZ instead of 80 and simulates in 0.1 s per circuit. This is a **pessimistic** proxy with about 40% more 2q gates than the best compile. The best compile
itself was then run on the 100 H4 ECGs for the two k=32 models, as a check.

| model | 2q gates (CZ) | noiseless | 2000 shots, ideal | noisy raw | noisy mitigated | corr P(MI) raw vs noiseless | ⟨Z⟩ shrink | bal. acc noiseless → noisy → rescaled | Brier noiseless → noisy → rescaled |
|---|---|---|---|---|---|---|---|---|---|
| k=8 pi | 109 | 0.9229 | 0.9241 | 0.9280 | 0.9277 | 0.989 | 0.76 | 0.829 → 0.840 → 0.832 | 0.110 → 0.127 → 0.107 |
| k=8 pi/2 | 109 | 0.9383 | 0.9373 | 0.9433 | 0.9433 | 0.990 | 0.72 | 0.847 → 0.850 → 0.858 | 0.097 → 0.112 → 0.094 |
| k=8 pi/6 | 109 | 0.9341 | 0.9334 | 0.9314 | 0.9315 | 0.991 | 0.71 | 0.849 → 0.839 → 0.843 | 0.097 → 0.115 → 0.102 |
| k=32 pi | 117 | 0.8743 | 0.8707 | 0.8715 | 0.8713 | 0.986 | 0.75 | 0.792 → 0.785 → 0.802 | 0.142 → 0.173 → 0.145 |
| k=32 pi/2 | 117 | 0.9340 | 0.9309 | 0.9373 | 0.9374 | 0.981 | 0.76 | 0.843 → 0.841 → 0.872 | 0.104 → 0.126 → 0.103 |
| k=32 pi/6 | 117 | 0.9609 | 0.9598 | 0.9602 | 0.9601 | 0.969 | 0.71 | 0.883 → 0.857 → 0.883 | 0.080 → 0.126 → 0.079 |

Best-compile check (H3a compile: 12 active qubits, fewest CZ), same noise model, 100 ECGs (the H4 set), 2000 shots:

| model | 2q gates | noiseless | noisy raw | noisy mitigated | corr P(MI) raw vs noiseless | ⟨Z⟩ shrink |
|---|---|---|---|---|---|---|
| k=32 pi/6 | 80 | 0.9831 | 0.9844 | 0.9844 | 0.981 | 0.81 |
| k=32 pi | 80 | 0.8563 | 0.8659 | 0.8659 | 0.993 | 0.88 |

(Rescaled = mitigated ⟨Z⟩ divided by one global shrink factor. This uses the noiseless *simulation* of the same circuits but no labels.
On hardware the factor would come from simulation or from mirror or calibration circuits.)

**Reading.** **AUC survives the Heron noise model essentially unchanged for all 6 models** (|noisy − noiseless| ≤ 0.006 on 300 ECGs,
≤ 0.010 on the 100-ECG best-compile check, within the sampling error of AUC on this many ECGs, which is about ±0.01). k=32 pi/6 stays the best model
(0.960 noisy vs 0.937 for pi/2 and 0.871 for pi). Device noise acts mainly as a near-uniform **contraction of ⟨Z⟩ by 0.71–0.88×**,
smaller with fewer CZ. Ranking survives this, so AUC is untouched and the predicted P(MI) stays correlated with noiseless at 0.97–0.99.
What breaks is **calibration and the fixed 0.5 threshold**. For k=32 pi/6, balanced accuracy falls 0.883 → 0.857 and Brier score worsens 0.080 → 0.126.
Readout mitigation alone barely helps, because the loss comes from gate noise, but **a single global rescaling restores both (0.883, 0.079)**.
Readout mitigation does nothing for AUC either, because it is a per-qubit affine map that preserves the ranking almost exactly.

## H5 — training-free encoding score (2026-11-05)

`python -u hw_h5_encoding.py` → `results/hw/h5_encoding.{json,md}`, `results/figs/encoding_score.png`. Encoding only (θ = 0: Ry(angle) layers plus
CNOT ring), fold-0 train rows, fold-0 l1 feature order, 1000 ECGs per class. From these, the class-mean density matrices ρ_MI and ρ_NORM (256×256) give
the trace distance ½‖ρ_MI − ρ_NORM‖₁ and the Uhlmann fidelity. Trained AUC = l1 / grouped5 / seed-0 runs in `results/*.jsonl`, at the longest steps available per
(k, scale): 13 configs, of which 7 are at k=32.

| k | scale | trace distance | fidelity | purity | trained steps | test AUC (5-fold mean) | test AUC fold 0 |
|---|---|---|---|---|---|---|---|
| 8 | pi/8 | 0.2934 | 0.8991 | 0.8519 | – | – | – |
| 8 | pi/6 | 0.3742 | 0.8352 | 0.7570 | 2000 | 0.9300 | 0.9370 |
| 8 | pi/4 | 0.4969 | 0.7009 | 0.5569 | – | – | – |
| 8 | pi/3 | 0.5734 | 0.5859 | 0.3879 | – | – | – |
| 8 | pi/2 | 0.6471 | 0.4391 | 0.1892 | 1000 | 0.9284 | 0.9347 |
| 8 | 2*pi/3 | 0.6756 | 0.3707 | 0.1057 | – | – | – |
| 8 | pi | 0.6237 | 0.4572 | 0.0575 | 1000 | 0.9103 | 0.9243 |
| 16 | pi/8 | 0.3584 | 0.8464 | 0.7394 | – | – | – |
| 16 | pi/6 | 0.4431 | 0.7637 | 0.5962 | 2000 | 0.9415 | 0.9483 |
| 16 | pi/4 | 0.5506 | 0.6241 | 0.3496 | – | – | – |
| 16 | pi/3 | 0.5995 | 0.5404 | 0.1933 | – | – | – |
| 16 | pi/2 | 0.6063 | 0.5110 | 0.0619 | 1000 | 0.9233 | 0.9383 |
| 16 | 2*pi/3 | 0.5628 | 0.5653 | 0.0247 | – | – | – |
| 16 | pi | 0.4749 | 0.6834 | 0.0096 | 1000 | 0.8922 | 0.9099 |
| 32 | pi/8 | 0.4129 | 0.7928 | 0.5498 | 600 | 0.9468 | 0.9554 |
| 32 | pi/6 | 0.4916 | 0.7071 | 0.3638 | 2000 | 0.9530 | 0.9603 |
| 32 | pi/4 | 0.5504 | 0.6245 | 0.1353 | 600 | 0.9429 | 0.9489 |
| 32 | pi/3 | 0.5240 | 0.6452 | 0.0471 | 600 | 0.9331 | 0.9430 |
| 32 | pi/2 | 0.4072 | 0.7700 | 0.0091 | 1000 | 0.9103 | 0.9251 |
| 32 | 2*pi/3 | 0.3375 | 0.8425 | 0.0055 | 600 | 0.8502 | 0.8586 |
| 32 | pi | 0.3052 | 0.8716 | 0.0049 | 1000 | 0.7910 | 0.8612 |

| Spearman with trained test AUC (5-fold mean) | trace distance | fidelity |
|---|---|---|
| all 13 configs | **+0.34** (p = 0.26) | −0.24 (p = 0.44) |
| k=32 only (7 scales) | **+0.71** (p = 0.07) | −0.61 (p = 0.15) |

**Reading.** **Trace distance is not a reliable training-free selector.** Across k the correlation is weak (ρ = 0.34). Within k=32
it tracks the collapse (it ranks 2π/3 and pi last), but it peaks at pi/4 rather than pi/6, and at k=8 it peaks at 2π/3, where training shows no
benefit. Trace distance measures distinguishability under the *best possible* measurement (the Helstrom bound), not what a linear readout on 8 ⟨Z⟩ can extract,
and it ignores whether the trained ansatz can reach that measurement. *Post-hoc observation, not pre-registered:* the **purity** of the class-mean states
tracks AUC much better (Spearman +0.80, p = 0.001 over 13 configs; +0.96 at k=32). At large scales the encoded states scatter so widely that the class means
are close to maximally mixed (purity 0.005 ≈ 1/256 at k=32 pi), the same saturation identified in the classical study. Purity, however, rises monotonically as the scale
shrinks, so on its own it would pick pi/8 rather than pi/6 (0.947 vs 0.953). It is a cheap way to *rule out* over-large scales, not to pick the optimum.

## H4 — real IBM hardware: script ready and dry-run tested; NOT run (2026-11-05; superseded by the real run H6, 2026-11-14)

**Not run, for two independent reasons.** (1) No `QISKIT_IBM_TOKEN` is set. (2) This cloud environment's network policy **blocks IBM Quantum**: the
proxy answers `403` to CONNECT for `quantum.cloud.ibm.com`, `iam.cloud.ibm.com` and `us-east.quantum-computing.cloud.ibm.com`. So even
with a token, the job cannot be sent from this session unless those hosts are allowed in the environment's network settings. No hardware results are reported or
simulated in their place.

**What is ready.** `hw_run_ibm.py` implements the runbook plan exactly:
- `QiskitRuntimeService(channel="ibm_quantum_platform", token=$QISKIT_IBM_TOKEN, instance=$QISKIT_IBM_INSTANCE)`, then `least_busy(operational=True, simulator=False, min_num_qubits=8)`.
- 2 models (k=32 pi/6, k=32 pi), the **same 100** class-stratified fold-0 test ECGs (the set used in the H3 best-compile check), 1000 shots.
- **One SamplerV2 job per model inside a `Batch`**; each PUB = one ECG (the transpiled template + that ECG's 32 angles), plus 2 readout-calibration PUBs
  on the same physical qubits. Transpiled once with the backend's preset pass manager (optimization_level=3).
- Dynamical decoupling (XpXm) and gate + measurement twirling switched on (options checked as valid in qiskit-ibm-runtime 0.51).
- Budget: 2 × 102 PUBs × 1000 shots = 204k shots, expected to need about 1–2 of the ~10 QPU minutes/month on the open plan. The script prints the plan and asks
  for confirmation before submitting.
- **Saved before any analysis** in `results/hw/ibm_hw/`: `meta.json` (job ids written immediately after submission, backend name, calibration
  timestamp, circuit stats, physical qubits), `ecgs_*.npz` (ECG indices + labels), `counts_*.json` (raw counts of every PUB).
- Then it runs Aer with **the same backend's noise model** on the same circuits, and `--analyze` reports AUC (hardware raw, hardware mitigated, Aer noisy raw/mitigated,
  noiseless) and the correlation of hardware vs noiseless P(MI) per ECG. `--retrieve` re-fetches results from the saved job ids if the session drops.

**Dry run** (`python -u hw_run_ibm.py --local --n 10`): the full pipeline ran through Qiskit Runtime *local mode* on FakeFez: transpile (80 CZ,
depth 205–209, physical qubits 146-143-136-123-126-127-137-147), submit, save, Aer comparison, analysis. Output is in `results/hw/ibm_local/`. With 10 ECGs the AUCs
(0.958 for pi/6, 0.875 for pi, the same across all columns) only show that the plumbing works, not the noise level.

**To run it** (from a laptop, or from this environment after allowing the IBM hosts):
```bash
pip install -r requirements.txt -r requirements-hw.txt
export QISKIT_IBM_TOKEN=...          # from quantum.cloud.ibm.com; do not commit or paste it
export QISKIT_IBM_INSTANCE=crn:...   # only for newer IBM Cloud accounts
python -u hw_run_ibm.py --submit     # shows the plan, asks y/N, submits, saves, analyses
```
The Aer comparison step simulates the 12-qubit compile as a density matrix, so expect about 10–25 min of CPU after the jobs return. Results land in
`results/hw/ibm_hw/analysis.json`. The H3 best-compile numbers on the same 100 ECGs (noiseless 0.983 / FakeFez-noisy 0.984 for pi/6;
0.856 / 0.866 for pi) are the simulation baseline to compare against.

---

## H6 — Real hardware, ibm_marrakesh (Heron r2) (2026-11-14)

`python -u hw_run_ibm_brickwall.py --analyze --tag hw` → `results/hw/ibm_bw_hw/analysis.json`;
`python -u hw_analyze_extended.py --tag hw` → `results/hw/ibm_bw_hw/extended_analysis.json`.

### Setup

Backend: `ibm_marrakesh` (ibm_quantum_platform), IBM Heron r2 processor. Two models tested:
- **bw8**: brick-wall QNN, n = 8 qubits, k = 4 particles, 136 CZ, depth 254 (transpiled to ibm_marrakesh).
- **vqc32**: angle VQC, k = 32 features, π/6 scale, 80 CZ, depth 208.

Both submitted via `SamplerV2` inside a `Batch` with dynamical decoupling (XpXm sequence) and gate + measurement twirling enabled. No readout calibration circuits (not included in this run). 200 class-stratified fold-0 test ECGs, 1000 shots each.

### Jobs

**Pilot** (10 ECGs, 500 shots):
- bw8 job id: `dbs3abhabdes73fqmv50`
- vqc32 job id: `dbs3ac7v5hds73diglr0`
- Submitted: 2026-11-14T10:17:17Z; Finished: 2026-11-14T10:17:31Z; est. QPU time 0.07 min.

**Main run** (200 ECGs, 1000 shots):
- bw8 job id: `dbs3bcvv5hds73digmrg`
- vqc32 job id: `dbs3bd7v5hds73digms0`
- Submitted: 2026-11-14T10:19:30Z; Finished: 2026-11-14T10:22:06Z; est. QPU time 2.67 min.

Note: actual QPU time (as opposed to wall-clock time) was not saved in metadata. Read from the IBM dashboard if needed.

### Main results (200 ECGs, 1000 shots, ibm_marrakesh)

| model | AUC noiseless [95% CI] | AUC raw [95% CI] | AUC postselected [95% CI] | kept fraction | paired Δ (raw − noiseless) [95% CI] |
|---|---|---|---|---|---|
| bw8 | 0.9253 [0.8846, 0.9576] | 0.9485 [0.9170, 0.9737] | 0.9483 [0.9149, 0.9750] | 0.499 ± 0.025 | +0.023 [+0.003, +0.046] |
| vqc32 | 0.9673 [0.9414, 0.9878] | 0.9561 [0.9238, 0.9822] | – | – | −0.011 [−0.024, −0.001] |

Bootstrap CIs: B = 2000, seed = 0, percentile method, paired resampling for Δ. Noiseless AUC computed from exact circuit simulation (precomputed in `ecgs_*.npz`). The 200-ECG test set has ~±0.01 AUC sampling error. 2.2 resource units consumed (ibm_marrakesh billing).

**bw8 particle-number note:** 50% of shots violated the k = 4 particle-number constraint and were discarded by postselection (`kept_fraction = 0.499`); postselected AUC (0.9483) is essentially unchanged from raw (0.9485), so postselection neither helps nor hurts here.

**Key reading:** vqc32 AUC degrades by −0.011 (CI excludes 0 marginally: [−0.024, −0.001]), consistent with ⟨Z⟩ contraction under real hardware noise. bw8 AUC is +0.023 *higher* on hardware than noiseless (paired CI [+0.003, +0.046] excludes 0): hardware ranked these 200 ECGs better than its own exact noiseless simulation. This is most likely a noise-induced change in the effective decision function (the raw bw8 logit is noisier but differently biased), not a reliable improvement — it is one run, one fold, 200 ECGs. Both models are well above 0.90 AUC on ibm_marrakesh.

### Per-ECG agreement (hardware vs noiseless logit)

| model | Pearson r | Spearman ρ | MAE (logit) | logit contraction factor |
|---|---|---|---|---|
| bw8 | 0.932 | 0.930 | 3.42 | 1.56 (hw raw larger) |
| vqc32 | 0.971 | 0.969 | 2.21 | 0.44 (hw raw smaller) |

Contraction factor = mean|hw_logit − b| / mean|nl_logit − b|. The bw8 raw contraction factor >1 indicates a systematic bias expansion under hardware noise; vqc32 shows the more typical ⟨Z⟩ contraction (~0.44×). With postselection (k=4), bw8 MAE improves to 1.86. The ranking is preserved at Pearson/Spearman r ≈ 0.93–0.97.

### Calibration (threshold tuned on the test ECGs — optimistic)

Threshold labelled "noiseless-optimal" is the balanced-accuracy-maximising threshold found on the noiseless logits + true labels (not from a held-out validation set — this is the same test set, so this overfits in principle). The hardware logit has a different offset; applying the noiseless threshold directly fails (spec ≈ 0). Using the hardware-optimal threshold or the rescaled logit with its own optimal threshold restores calibration. **All thresholds are tuned on the test ECGs themselves; these figures are optimistic and would need a separate held-out set in deployment.**

| model | condition | bal. acc | sens | spec | acc |
|---|---|---|---|---|---|
| bw8 | noiseless (nl-optimal thr) | 0.875 | 0.836 | 0.913 | 0.885 |
| bw8 | hw raw (hw-optimal thr) | 0.895 | 0.877 | 0.913 | 0.900 |
| vqc32 | noiseless (nl-optimal thr) | 0.907 | 0.932 | 0.882 | 0.900 |
| vqc32 | hw raw (hw-optimal thr) | 0.905 | 0.904 | 0.906 | 0.905 |

Global rescale factors: bw8 s = 1.97, vqc32 s = 2.23 (s * hw_logit ≈ noiseless_logit in least-squares sense). After rescaling + re-tuning threshold, calibration matches noiseless. The rescale factor would need to come from mirror circuits or calibration data in a real deployment (we used the noiseless simulation here, which is not label-free).

### Comparison with H3b (FakeFez noisy simulation)

H3b used a **different circuit compile** (best-compile 80 CZ for vqc32, pessimistic 109–117 CZ for the 300-ECG run) vs real hardware (80 CZ for vqc32 as per info_vqc32.json), different number of ECGs (300 vs 200), and different shots (2000 vs 1000).

H3b FakeFez predictions: AUC degrades by ≤ 0.006 from noiseless under the simulated noise model, with ⟨Z⟩ contraction 0.71–0.88×. Real hardware (H6): AUC degradation ≤ 0.012 for vqc32 (within ±0.01 sampling error), logit contraction 0.44× for vqc32 and expansion 1.56× for bw8. The larger bw8 "contraction" and the offset shift suggest real hardware has additional noise not captured by FakeFez (crosstalk, drift, leakage). The vqc32 AUC agrees well with FakeFez predictions (~0.960 FakeFez vs 0.956 real hardware), consistent with the same 80 CZ compile. The bw8 paired Δ (+0.023, CI [+0.003, +0.046]) is a real paired difference on the same 200 ECGs, not a sampling artefact — but it is one run on one fold and most likely reflects a noise-induced shift in the effective bw8 decision function rather than a reliable improvement; see main results note. The vqc32 AUC degradation (−0.011) and ⟨Z⟩ contraction (0.44×) are consistent with the FakeFez prediction direction (≤ 0.006 AUC loss, 0.71–0.88× contraction), though real hardware shows ~2× more contraction, suggesting additional noise sources (crosstalk, drift, leakage) beyond the FakeFez noise model. The vqc32 AUC agrees well with FakeFez (~0.960 FakeFez vs 0.956 real hardware) with the same 80 CZ compile. Overall real hardware is consistent with FakeFez for the purpose of AUC survival.

### Noisy simulation cell (see also COLAB_HW.md)

A matched FakeMarrakesh (or FakeFez if unavailable) simulation on the **same 200 ECGs** is needed for a direct comparison. See COLAB_HW.md Cell 9 for the Colab cell. Do NOT run Aer on this PC (Qiskit DLL blocked by WDAC).

### Honest framing

This experiment tests whether two trained quantum models (bw8, vqc32) retain their discriminative ability when executed on a real superconducting quantum processor. It is **not a quantum-advantage result** — both models are outperformed by classical XGBoost/RF in simulation. The 200-ECG test set has ~±0.01 AUC sampling error, so differences below 0.01 are not interpretable. The result shows that AUC survives real IBM Heron hardware noise without readout mitigation or error correction, consistent with the FakeFez simulation predictions.

---

## How the H6 run was produced (2026-11-14)

**Qiskit is blocked by WDAC on the lab PC.** The run was done from Google Colab using `hw_bundle.zip` (see `COLAB_HW.md` for the exact notebook cells). The commands below reproduce it.

**Step 1 — Colab dry run (FakeFez; no account needed):**
```bash
# In Colab after extracting hw_bundle.zip and installing requirements-colab.txt:
python hw_run_ibm_brickwall.py --local --n 4 --shots 200 --models bw8,vqc32
# Output: results/hw/ibm_bw_local/ (FakeFez, 4 ECGs)
```

**Step 2 — Pilot run (10 ECGs, 500 shots, ibm_marrakesh):**
```bash
python hw_run_ibm_brickwall.py --submit --n 10 --shots 500 --models bw8,vqc32 --tag pilot --yes
# Job ids: bw8=dbs3abhabdes73fqmv50, vqc32=dbs3ac7v5hds73diglr0
# Output: results/hw/ibm_bw_pilot/  (2.2 RU total for pilot + main)
```

**Step 3 — Main run (200 ECGs, 1000 shots, ibm_marrakesh):**
```bash
python hw_run_ibm_brickwall.py --submit --n 200 --shots 1000 --models bw8,vqc32 --tag hw --yes
# Job ids: bw8=dbs3bcvv5hds73digmrg, vqc32=dbs3bd7v5hds73digms0
# Output: results/hw/ibm_bw_hw/
```

**Step 4 — Analysis (run on lab PC after downloading results; no Qiskit needed):**
```bash
..\.venv\Scripts\python.exe hw_run_ibm_brickwall.py --analyze --tag hw
..\.venv\Scripts\python.exe hw_analyze_extended.py --tag hw
```

For the matched noisy-simulation comparison (FakeMarrakesh), see `COLAB_HW.md` Cell 9.
