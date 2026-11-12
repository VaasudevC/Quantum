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
