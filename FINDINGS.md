# Findings: quantum classifiers for MI detection on PTB-XL

**Task:** myocardial infarction (MI) vs normal (NORM) 12-lead ECG, PTB-XL, 14,965 records (5,454 MI), 249 pre-extracted ST/T/QRS features.
**Last updated:** 2026-11-14. **Status:** all planned experiments finished, including a run on real IBM hardware.
This file is the single-page summary of the whole project. Every number links to a detailed report and a result file in this repository.

---

## 1. Bottom line

1. **A simple encoding fix is the main positive result.** Encoding each feature as `s·tanh(z)` with s = π (the original choice) makes the 8-qubit re-uploading VQC *worse* as more features are added (k = 32: AUC 0.791). Shrinking the scale to **s = π/6** removes the collapse and AUC *rises* with k: 0.928 (k=8) → 0.939 (k=16) → **0.951 (k=32)**. π/6 beats π in 5/5 folds at every k. The same lesson held in a second architecture (the brick-wall QNN).
2. **No quantum advantage was found.** Tuned classical models match or beat every quantum model on the same inputs: XGBoost 0.965, degree-2 logistic regression 0.958, best VQC 0.955 (k = 32, 5-fold CV). The VQC's small edge over plain logistic regression (+0.006) is explained by mild non-linearity, not by anything quantum: a classical quadratic model with no more parameters matches it.
3. **In small-data settings (N = 25 to 500 training ECGs), quantum models are non-inferior to the best classical model and superior at none** (pre-registered, margin 0.02 AUC).
4. **The models survive a real quantum computer.** On `ibm_marrakesh` (IBM Heron r2, 200 ECGs, 1000 shots, no error mitigation) AUC was 0.949 (brick-wall, 8 qubits) and 0.956 (angle VQC), against 0.925 and 0.967 noiselessly on the same ECGs. This shows the pipeline works on hardware; it is not an advantage result.
5. **Claims we do not make:** quantum advantage, parameter efficiency, clinical readiness. All models are classically simulable.

---

## 2. Project timeline

| Phase | What was done | Headline | Report |
|---|---|---|---|
| 0. Early VQC runs (Runs 1 to 7) | 8-qubit VQC, amplitude then angle encoding, parameter-shift gradients, single 80/20 split | Best test accuracy 81.26%, AUC 0.885 | [`legacy_phase0/`](legacy_phase0/README.md) |
| A. Angle-scale study | Scale sweep, training length, feature selectors, shuffle control, degree-2 comparison, official split | s = π/6 fixes the collapse; k = 32 AUC 0.951 | [`RESULTS.md`](RESULTS.md) |
| B. Hardware readiness (simulated) | Qiskit port, shot noise, compilation, IBM noise model | AUC survives ~180 shots and Heron noise | [`HARDWARE.md`](HARDWARE.md) |
| C. Improvements | zz readout, trained per-feature scales, kernel-target alignment, clinical metrics | Best VQC 0.955, still below degree-2 LR 0.958 | [`NEXT.md`](NEXT.md) |
| D. Brick-wall benchmark | Reimplemented the magic-state brick-wall QNN; full benchmark vs tuned classical models, external PTB set | No quantum model beats the tuned classical baselines | [`BRICKWALL.md`](BRICKWALL.md) |
| E. Small-data benchmark | Pre-registered, 25 to 500 training ECGs, k = 8 and k = 4 | Non-inferior everywhere, superior nowhere | [`SMALLDATA.md`](SMALLDATA.md) |
| F. Search for a quantum benefit (QB) | 1,200 feature maps, parameter-matched classical models, robustness, shift, imbalance, subtypes, confirmation run | No benefit found | [`QB.md`](QB.md) |
| G. Real hardware | Two trained models on `ibm_marrakesh` | AUC 0.949 and 0.956 | [`HARDWARE.md`](HARDWARE.md) section H6 |

Phase 0 numbers are accuracies from one 80/20 split; Phases A to G are AUCs from patient-grouped 5-fold cross-validation. **They are not directly comparable.** Phases A to G also use a different feature selector and scale, which is the point of the study.

---

## 3. Phase 0: early VQC runs (accuracy, single split)

Setup: 8 qubits, 4 layers, 72 ansatz parameters, CNOT ring, Adam, parameter-shift gradients (exact and hardware-portable). Train 11,972 / test 2,993.

| Run | Encoding | Readout | Test accuracy | AUC |
|---|---|---|---|---|
| 1 to 3 | Amplitude | Linear | 78 to 79% | 0.852 |
| 4 | Amplitude + log scaling | Linear | 79.45% | 0.869 |
| 5 | Angle, 8 binary-flag features | Linear | 77.18% | 0.800 |
| 6 | Angle, 32 features loaded sequentially (8 per layer) | Linear | 80.45% | 0.870 |
| **7** | Angle, 32 features sequential | MLP (8→32→1) | **81.26%** | **0.885** |
| 8a to 8e | PLS-8 or PLS-32 of all 251 features | MLP | did not converge | n/a |

What we learned:
- **Amplitude encoding plateaus near 79%** because L2 normalisation throws away the mV scale.
- **Tanh saturation destroyed the PLS features.** PLS component 1 had raw scores of +4.35 (MI) and −2.49 (NORM), so `π·tanh` pushed both to ±π, where `Ry(+π)|0⟩` and `Ry(−π)|0⟩` differ only by an unobservable global phase. This was the first sign of the problem that Phase A later solved with a smaller scale.
- The reported 83.36% (threshold 0.665) was **tuned on the test set**, so it is optimistic. Use 81.26% at threshold 0.5.
- A logistic regression on the *same 32 angles* scored 86.8%, so the circuit and optimiser, not the features, were the bottleneck. Parameter-shift training cost about 8 s per step. An exact adjoint-gradient simulator (`Vqc_fast8.py`, identical gradients to 3e-17, 126× faster) made Phase A possible.

---

## 4. Phase A: the angle-scale rule

Protocol: 8 qubits × 4 re-uploading layers, k features (L1-selected), patient-grouped 5-fold CV, choices made on validation AUC, test AUC reported. Source: [`RESULTS.md`](RESULTS.md).

| Test AUC (5-fold) | k = 8 | k = 16 | k = 32 |
|---|---|---|---|
| VQC, s = π | 0.910 | 0.892 | 0.791 |
| VQC, s = π/2 | 0.928 | 0.923 | 0.910 |
| **VQC, s = π/6** | **0.928** | **0.939** | **0.951** |
| Logistic regression, same features | 0.922 | 0.932 | 0.945 |
| Degree-2 logistic regression (k = 32) | n/a | n/a | 0.958 |

- **Why π fails:** about 10% of the angles sit near ±π and the later re-uploading rounds act as noise. A shuffle control confirms it: at π/6, shuffling the later features costs about nothing (−0.006) and the real ones add +0.023; at π, shuffled features wreck the circuit (−0.229).
- **Scale sweep (k = 32):** AUC falls monotonically as the scale grows past π/6 (π/8 0.947, π/6 0.950, π/4 0.943, π/3 0.933, π/2 0.908, 2π/3 0.850, π 0.794).
- **Feature selection matters most at small k:** L1 beats random features by +0.13 AUC at k = 8 and +0.04 at k = 32. A Cohen's-d de-duplication selector is within 0.005 of L1.
- **The VQC is 98.6% a quadratic function of its input angles**, and a classical degree-2 model on the same inputs (0.958) beats it in 5/5 folds.
- **Official PTB-XL split** (train folds 1 to 8, test fold 10): π/6 gives 0.942 at k = 32 (logistic regression 0.938). The ranking π/6 > π/2 > π holds. Every model scores a little lower on fold 10, so that fold is somewhat harder.
- **Training length:** 1000 steps used; 2000 steps adds only +0.002 to +0.003.

---

## 5. Phase C: improving the VQC

Source: [`NEXT.md`](NEXT.md). k = 32, 5-fold CV.

| Model | Test AUC | Sensitivity at 90% specificity | at 95% specificity |
|---|---|---|---|
| Logistic regression | 0.945 | 0.852 | 0.783 |
| VQC, fixed π/6 | 0.951 | 0.856 | 0.791 |
| VQC, zz readout | 0.953 | 0.863 | 0.797 |
| **VQC, trained per-feature scales** | **0.955** | 0.867 | 0.796 |
| Degree-2 logistic regression | 0.958 | 0.884 | 0.816 |

- Trained scales and the zz readout each beat fixed π/6 in 5/5 folds, but **neither beats degree-2 logistic regression, which wins every fold.**
- Under shot noise the zz gain disappears (at 180 shots zz 0.936 vs z 0.939), because pairwise correlators are noisier.
- Best training-free predictor of a good encoding: kernel-target alignment (Spearman 0.61 with trained AUC, 0.86 when centred). It flags too-large scales but does not find the exact optimum.
- Clinically, inferior MI is the hardest subtype for every model (AUC 0.943 vs 0.957 for anterior MI).

---

## 6. Phase D: full-data benchmark, including the brick-wall QNN

Source: [`BRICKWALL.md`](BRICKWALL.md). Tuned classical models (LR, degree-2 LR, MLP, random forest, XGBoost) against the angle VQC and a magic-state brick-wall QNN that we reimplemented from the collaborator's paper and verified against an independent simulator and PennyLane (agreement ≤ 1.3e-15).

| Features | Best classical (test AUC) | Best quantum (test AUC) | Gap | Quantum-advantage rule |
|---|---|---|---|---|
| 8 | XGBoost 0.9374 | Angle VQC 0.9280 | −0.009 | not met |
| 12 | XGBoost 0.9452 | Brick-wall n = 12: 0.9341 | −0.011 | not met |
| 32 | XGBoost 0.9647 | Trained-scale VQC 0.9547 | −0.010 | not met |
| External PTB set (445 records) | k = 32 XGBoost 0.946 | Angle VQC 0.934 | −0.012 | not met |

- XGBoost on all 249 features reaches 0.973.
- Exploratory paired bootstrap at k = 32 (95% CI): VQC π/6 vs plain LR **+0.0059** [+0.0042, +0.0077]; vs degree-2 LR **−0.0073** [−0.0089, −0.0056]; vs XGBoost **−0.0136** [−0.0155, −0.0118].
- The π/6 lesson transfers: small phase scales (π/6 to π/4) are best in the brick-wall too, and π is worst.
- **Feature extractor finding:** the stored ST features were computed after a 0.5 to 40 Hz band-pass, not the script's current 0.05 Hz setting. The recovered extractor reproduces all 249 columns of 50 records to 1e-13. A real V4 to V6 population shift remains between PTB-XL and the PTB healthy controls.
- The brick-wall readout is classically tractable (two-body expectations), so even a performance win there would not imply hardness.

---

## 7. Phase E: pre-registered small-data benchmark

Source: [`SMALLDATA.md`](SMALLDATA.md). The analysis plan was committed before any run. Question: with 25 to 500 training ECGs, does a quantum model match or beat the best classical model? Margin for non-inferiority: 0.02 AUC.

**k = 8 (primary):**

| N (training ECGs) | Best quantum vs best classical | Mean difference [95% CI] | Claim |
|---|---|---|---|
| 25 | Q3 quantum-kernel SVM vs GP | −0.0040 [−0.0099, +0.0012] | non-inferior |
| 50 | Q3 vs GP | −0.0033 [−0.0058, −0.0009] | non-inferior |
| 100 (primary) | Q3 vs random forest | −0.0028 [−0.0049, −0.0010] | non-inferior |
| 200 | Q1 angle VQC vs random forest | −0.0018 [−0.0036, −0.0002] | non-inferior |
| 500 (reduced sample) | Q1 vs random forest | −0.0009 [−0.0020, +0.0002] | non-inferior |

**k = 4 (secondary):** same conclusion at every N; best classical model is the GP classifier throughout. AUC is about 0.07 lower than at k = 8 for every model.

- Holm-adjusted wins exist only against weaker classical models (kNN, XGBoost, and plain LR at N = 25 and N = 500). Never against SVM-RBF, GP or random forest.
- Logistic regression with 9 parameters matches the 81-parameter angle VQC on AUC and on the train-to-test gap.
- The non-inferiority margin (0.02) is wide next to the spread of the good models (about 0.005 at N ≥ 100), so "non-inferior" is a weak statement. An exploratory ±0.005 equivalence check is in `SMALLDATA.md`.
- By construction this benchmark could not show an advantage: at k ≤ 8 the VQC loads each feature once, and the quantum-kernel SVM uses a product kernel a classical computer evaluates exactly. That motivated Phase F.

---

## 8. Phase F: is there any setting where quantum helps?

Source: [`QB.md`](QB.md), pre-registered, with every setting tried recorded in `results/qb_registry.csv` (262 rows).

- **Feature maps (QB1):** 1,200 settings screened. None has both expressivity beyond a tuned RBF kernel and adequate label alignment. Expressivity appears only at s = π/2, where alignment collapses.
- **Non-linearity (QB3):** classical models with no more parameters (rank-2 factorisation machine, sparse degree-2 LR) match the trained-scale VQC (0.9562 and 0.9561 vs 0.9561). Consistent across all 5 folds.
- **Robustness, shift, imbalance, subtypes (QB5, 30 of 30 units):** two candidates appeared, both in the inferior-MI subtype (Q1 +0.0103, Q3 +0.0091 vs SVM-RBF).
- **Confirmation (QB6): neither candidate confirmed** (Q1 CI spans zero; Q3 lower bound −0.0003). Both are recorded as hypotheses for future work.
- **Headline wording: "no benefit found".** The word "advantage" is not used.
- Incomplete by design: QB4 training (stopped under the 6-hour rule) and the QB5 full-data part.

---

## 9. Phase G: real IBM hardware

Source: [`HARDWARE.md`](HARDWARE.md) section H6. Backend `ibm_marrakesh` (Heron r2), run on 2026-11-14 from Google Colab, 200 class-stratified fold-0 test ECGs, 1000 shots each, dynamical decoupling and twirling on, **no readout mitigation**, 2.2 resource units.

| Model | Circuit | AUC noiseless | AUC on hardware [95% CI] | Paired difference | Per-ECG Pearson r |
|---|---|---|---|---|---|
| Angle VQC (k = 32, π/6) | 80 CZ, depth 208 | 0.967 | **0.956** [0.924, 0.982] | −0.011 [−0.024, −0.001] | 0.97 |
| Brick-wall (n = 8, k = 4) | 136 CZ, depth 254 | 0.925 | **0.949** [0.917, 0.974] | +0.023 [+0.003, +0.046] | 0.93 |

How to read this:
- Both models stay above 0.94 AUC on a real device with no error mitigation. This matches the FakeFez simulation (vqc32 about 0.960 simulated vs 0.956 real).
- The brick-wall's *higher* hardware AUC is almost certainly a noise-induced shift in its decision function, not an improvement: one run, one fold, 200 ECGs.
- Real hardware contracted the angle-VQC logits about 0.44×, roughly twice the contraction predicted by the noise model, so unmodelled noise (crosstalk, drift, leakage) is present.
- About 50% of brick-wall shots violated the particle-number constraint; post-selecting them away left AUC unchanged (0.9483 vs 0.9485).
- **Thresholds were tuned on the same 200 test ECGs, so the sensitivity/specificity figures in `HARDWARE.md` are optimistic.**
- Simulated hardware readiness: about 180 shots per ECG keeps π/6 within 0.01 AUC of noiseless; the n = 12 brick-wall needs about 10⁶ shots because its head amplifies shot noise.

---

## 10. Limitations

- One dataset, one task (MI vs NORM), hand-crafted features rather than raw signals.
- Hardware result is one run, one fold, 200 ECGs (±0.01 AUC sampling error), so differences below 0.01 are not interpretable.
- 5 outer folds, so a 5/5 sign test can reach at best p = 0.062. Seeds change only initialisation and minibatches.
- Folds 3 and 4 were used in earlier choices (π/6, the L1 selector), so they are only partly clean confirmation data.
- Small-data N = 500 is a reduced sample (10 of 40 subsets). The brick-wall ran in the small-data study only at N = 100.
- TabPFN, a strong small-data baseline in the pre-registration, was not run (licence login required).
- Several analyses (Phase 4 bootstrap, QB3 folds 3 and 4, the equivalence check) are exploratory and flagged as such in their reports.

## 11. Open items and suggested next steps

1. A matched FakeMarrakesh noisy simulation on the same 200 ECGs, to separate "model noise" from "device noise" (`COLAB_HW.md`, Cell 9).
2. Repeat the hardware run on other folds and with readout-error mitigation, with thresholds chosen on a held-out set.
3. Test the inferior-MI subtype hypothesis on independent data.
4. Brick-wall at N ≠ 100 and the n = 12 / n = 16 shot-robust heads (blocked locally by Windows Application Control; run from Colab).
5. Paper draft: [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) maps every claim to its supporting file and commit.

## 12. Reproducing

```bash
pip install -r requirements.txt           # classical + simulation
pip install -r requirements-hw.txt        # Qiskit / PennyLane / IBM Runtime
python -c "import Vqc_fast8; Vqc_fast8.verify()"
python summarize.py results/stage2_steps1000.jsonl
```

Full command list with run times: [`docs/REPRODUCE.md`](docs/REPRODUCE.md). File inventory: [`docs/CODE_AND_DATA.md`](docs/CODE_AND_DATA.md). Day-by-day log: [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).
Data: PTB-XL (CC BY 4.0), derived feature table in `data/`. Real-hardware runs need your own IBM Quantum token in the `QISKIT_IBM_TOKEN` environment variable; never commit it.
