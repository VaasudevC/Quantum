# SMALLDATA — do quantum models match or beat classical models when training data is small?

Branch `smalldata` (from `brickwall-overnight`). Log: [`OVERNIGHT_LOG.md`](OVERNIGHT_LOG.md). Code: `sd_run.py`, `sd_analyze.py`. This report is updated after every sub-phase.

## Pre-registered analysis plan

(Written and committed BEFORE any SD experiment was run; not to be changed. Additions go under "Exploratory (not pre-registered)".)

```
Data/splits: the existing patient-grouped, class-stratified 5 outer folds (scale_cv.py). Test set =
the full outer test fold, never subsampled, never used for any choice.
Training sizes N in {25, 50, 100, 200, 500}. For each outer fold and N, draw R = 40 subsets of N
ECGs from that fold's training part: class-stratified (keep the MI/NORM ratio, at least 5 of each
class), one ECG per patient. The SAME subsets are used for every model (paired design). Seed every
subset deterministically from (fold, N, r).
NO LEAKAGE: everything is fitted on the N-ECG subset only: median imputation, z-scoring, feature
selection, hyperparameters, early stopping. Nothing uses the rest of the training fold.
Feature selection inside each subset: L1 path order (random_state=0) on the N ECGs, top k features,
k in {4, 8}. Primary k = 8.
Hyperparameters and early stopping: inner 3-fold stratified CV on the N-ECG subset (for N=25 use
leave-group-out with 3 folds anyway); refit on all N with the chosen setting. Small fixed grids
below; log the chosen values.

Models
 Quantum:
  Q1 angle VQC, 8 qubits x 4 layers, k features in round 1 only (for k=4, the other 4 encoding
     angles are 0; no re-uploading at these k), scale grid
     {pi/6, pi/4}, linear readout on <Z>, Adam lr 0.01, steps grid {100, 300}, class-weighted BCE,
     L2 on readout {0, 1e-3}. Fast exact simulator Vqc_fast8.
  Q2 angle VQC with trained per-feature scales (init pi/6), same grid otherwise.
  Q3 quantum-kernel SVM: kernel K(x,x') = |<phi(x)|phi(x')>|^2 from the encoding-only angle circuit
     (theta = 0, 8 qubits, CNOT ring, scale grid {pi/8, pi/6, pi/4}), sklearn SVC(kernel=
     'precomputed'), C grid {0.1, 1, 10}.
  Q4 brick-wall n=8 (bw_utils.py / P3 final config: particle number 4, 4 columns), phase scale grid
     {pi/6, pi/4}, steps grid {100, 300}, L2 on head {1e-3, 1e-2}.
 Classical:
  C1 logistic regression, C grid {0.01, 0.1, 1, 10} (L2).
  C2 SVM RBF, C {0.1, 1, 10} x gamma {'scale', 0.1, 0.01}.
  C3 Gaussian-process classifier (RBF + white kernel), default optimizer.
  C4 kNN, k {3, 5, 9} (capped at N-1), distance weighting.
  C5 random forest, max_depth {2, 4, None}, n_estimators 300, min_samples_leaf {1, 3}.
  C6 XGBoost, max_depth {2, 3}, n_estimators {50, 200}, learning_rate 0.1, reg_lambda {1, 10}.
  C7 TabPFN (pip install tabpfn; CPU is fine at these sizes). If it cannot be installed or run on
     this machine, log why and continue without it; say so in the report.
Metrics per (model, fold, N, r): test AUC, train AUC on the N ECGs, generalisation gap = train AUC -
test AUC, number of trainable parameters, wall time.

Primary endpoint: test AUC at N = 100, k = 8.
Primary comparison: best quantum model (Q1-Q4, the one with the highest mean test AUC at N=100)
  versus the best classical model (C1-C7, likewise) at N = 100. Because "best" is picked after
  seeing results, ALSO report every quantum-vs-classical pair, with multiplicity correction.
Statistical method (paired): for each pair of models, per-subset AUC difference d = AUC_q - AUC_c
  over all 5 folds x 40 subsets = 200 paired values. Estimate: mean d with 95% CI by hierarchical
  bootstrap (resample folds, then subsets within fold; 5,000 resamples). Test: one-sided.
  - SUPERIORITY at size N: lower CI bound of mean d > 0, after Holm correction across the 5 sizes
    and across all quantum x classical pairs (report the number of comparisons).
  - NON-INFERIORITY at size N: lower CI bound of mean d > -0.02 (margin fixed now: 0.02 AUC).
  - Also report the fraction of subsets where the quantum model wins.
Secondary endpoints: (a) the same tests at N in {25, 50, 200, 500}; (b) generalisation gap,
  quantum vs classical, same paired method (claim "smaller gap" only if the CI excludes 0);
  (c) k = 4 results.
Claim language, fixed now:
  - Superiority met for a pair -> "the quantum model outperformed <model> at N = ... (mean
    difference, CI, adjusted p)". If met against the BEST classical model at N=100 -> headline.
  - Only non-inferiority met -> "matched classical performance within 0.02 AUC".
  - Neither -> report plainly that classical models were better at that size.
  Always add one sentence that this is an empirical result on classical ECG features and that the
  theory in brickwall/Scalable_Quantum_ML.pdf (Prop. 8) does not predict hardness at this size.
```

Execution order (SD0–SD5, then exploratory EXTRA, then DX) is as in the task specification; DX is documented separately in `DISTINGUISHER.md`.

## SD0 note on the earlier learning curves (P5 in BRICKWALL.md)

P5 refitted median imputation, z-scoring and the L1 selection on each **subsample only**, so those steps did not leak. It did, however, (i) choose hyper-parameters and the early-stopping checkpoint on the **full-size validation rows of the fold** (~1.8k ECGs, far more than the N=50–1000 training ECGs), which is information beyond the N-ECG training set,
and (ii) draw patient-grouped subsamples with several ECGs per patient rather than one per patient. P5 is therefore **superseded by SD** (all choices inside the N-ECG subset only, inner CV, one ECG per patient, 40 paired subsets per fold and size).

## Execution log (not analysis changes)

* **Execution-order change (user instruction, 2026-11-12):** after N = 100, the sizes N = 50, 200, 25, 500 are run for Q1–Q3 and all classical models FIRST, and Q4 (brick-wall, the slow family) at those sizes LAST. The analysis stays exactly as pre-registered above.
  The primary-endpoint analysis at N = 100 is run as soon as all models are finished at N = 100.
* **C7 TabPFN skipped** for now (needs an interactive licence login; v1 weights are not downloadable). If the licence login works later, C7 is run as a catch-up under the same plan; until then the classical side has C1–C6 and the multiplicity count excludes C7.
* Workers: independent processes each taking every W-th unit (`sd_launch.py`); a freshly started process that measures numpy ~60× slower than normal restarts itself (an OS-scheduling effect we could not explain). Results of one unit are written only after all its models finish.
* Hardware note: the first ~2 h ran on battery with a 40 % processor cap; later on AC power.
* Paused until N = 100 and its analysis are committed: EXTRA 6 (shot-robust heads), EXTRA 1 VQC seeds, n = 16 external.

## Deviations from the pre-registered plan (compute) — logged 2026-11-13, BEFORE any SD analysis was run

The analysis method (paired hierarchical bootstrap, one-sided superiority with Holm, non-inferiority margin −0.02, win fractions, gap comparison, claim language) is **unchanged**.
Only the amount of data at the secondary cells is reduced, because the available compute (~1–2 effective cores; N = 100 alone took ~12 h) makes the full grid infeasible:

| cell | pre-registered | executed |
|---|---|---|
| N = 100, k = 8 (primary endpoint), all models | 40 subsets × 5 folds | **unchanged**: 40 × 5 |
| Q4 brick-wall at N ∈ {25, 50, 200, 500}, k = 8 | 40 subsets × 5 folds | **first 10 subsets** (r = 0–9) × 5 folds |
| all models at N = 500 (k = 8 and k = 4) | 40 × 5 | **first 10 subsets** (r = 0–9) × 5 folds |
| Q4 brick-wall at k = 4 | all N | **dropped** |
| C7 TabPFN | all cells | **pending** (licence login; catch-up run under the same plan if it becomes available) |

Consequences, stated now: (1) the subsets were drawn before any result was seen, so r = 0–9 is a fixed, outcome-independent subsample;
(2) in reduced cells the hierarchical bootstrap resamples folds and then only the available subsets (10 per fold), so those CIs are wider and those tests have less power;
(3) the Holm family still counts every quantum × classical × size comparison that is actually run (reported with each table); at k = 4 the quantum side is Q1–Q3 only;
(4) claim language: a superiority or non-inferiority result in a reduced cell is reported with the label "reduced sample (10 of 40 subsets)"; a failure to reach either in a reduced cell is reported as "inconclusive at reduced sample size", not as evidence that classical models were better, unless the CI excludes the margin on the classical side.
(5) Implementation note (no effect on results): an exact faster brick-wall formulation (column unitary = matrix_exp of commuting generators; verified to 1e-15) was profiled and NOT adopted — it is only ~15 % faster and not bit-identical in complex64, so all Q4 runs use the same code as N = 100.

## Primary endpoint — N = 100, k = 8 (pre-registered analysis, run 2026-11-13 after the deviation log was committed)

All 200 subsets (5 folds × 40) for every model; C7 TabPFN pending (not in the comparison family). Table and figures: `results/sd_tables_k8_N100.md`, `results/sd_analysis_k8_N100.json`.

| model | mean test AUC [95% CI] | generalisation gap |
|---|---|---|
| Q1 angle VQC | 0.889 [0.882, 0.894] | +0.07 |
| Q2 angle VQC, trained scales | 0.883 [0.876, 0.889] | +0.08 |
| Q3 quantum-kernel SVM | 0.889 [0.882, 0.895] | +0.07 |
| Q4 brick-wall n=8 | 0.857 [0.846, 0.866] | +0.07 |
| C1 LR | 0.887 [0.881, 0.893] | +0.07 |
| C2 SVM-RBF | 0.889 [0.884, 0.895] | +0.07 |
| C3 GP classifier | 0.891 [0.886, 0.897] | +0.07 |
| C4 kNN | 0.863 [0.859, 0.868] | +0.14 |
| C5 random forest | 0.891 [0.885, 0.898] | +0.10 |
| C6 XGBoost | 0.875 [0.868, 0.881] | +0.12 |

**Primary comparison (best quantum vs best classical, chosen after seeing the results):** Q3 quantum-kernel SVM vs C5 random forest: mean d = **−0.0028 [−0.0049, −0.0010]**, quantum wins in 42 % of the 200 paired subsets.
Superiority: **not met** (Holm p = 1.0). Non-inferiority (lower CI bound > −0.02): **met**.
**Claim level at N = 100: the quantum models matched classical performance within 0.02 AUC** — Q1, Q2 and Q3 are non-inferior to every classical model (all 18 lower CI bounds > −0.012).
The best classical models (random forest, GP classifier) are nevertheless slightly but significantly better than the best quantum model (CI of d entirely below 0, by ~0.003 AUC).

**All 24 pairs (Holm over the 24 comparisons at N = 100; with the full pre-registered family of 120 the conclusions are the same — every significant raw p is the bootstrap floor 0.0002, i.e. ≤ 0.024 after ×120):**
* Superiority met: Q1, Q2 and Q3 each **outperformed C4 kNN** (+0.019 to +0.025) **and C6 XGBoost** (+0.008 to +0.014) at N = 100 (adjusted p = 0.005, win fractions 0.74–0.95). These are not the best classical models.
* No superiority vs C1 LR, C2 SVM-RBF, C3 GP or C5 random forest (Q1 and Q3 vs LR: +0.002, CI includes 0).
* Q4 brick-wall: classical models were better — significantly below LR, SVM-RBF, GP, RF and XGBoost (−0.018 to −0.035; not non-inferior against any of them); level with kNN (−0.007, CI includes 0).
* Generalisation gap: Q1/Q3 have a significantly **smaller** train-test gap than RF (−0.03), XGBoost (−0.05) and kNN (−0.07), and an equal gap to LR, SVM-RBF and GP (CIs include 0). Q2's gap is significantly larger than LR/SVM/GP (+0.01).

This is an empirical result on classical ECG features (8 hand-crafted features, 100 training ECGs); the theory in `brickwall/Scalable_Quantum_ML.pdf` (Prop. 8) does not predict classical hardness at this size — the angle-encoding kernel used by Q3 is a product of cos² terms that a classical computer evaluates exactly, and the models that matched the classical ones are of that kind.
