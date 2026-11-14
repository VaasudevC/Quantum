# SMALLDATA — do quantum models match or beat classical models when training data is small?

Tag `v0.3-smalldata` (built on `v0.2-brickwall`; both merged into `main`). Log: [`OVERNIGHT_LOG.md`](OVERNIGHT_LOG.md). Code: `sd_run.py`, `sd_analyze.py`. This report is updated after every sub-phase.

## Headline (k = 8, final; k = 4, secondary endpoint, final)

**Claim level reached: non-inferiority at every training size, superiority at none — for both k = 8 and k = 4.**

**Real hardware result (2026-11-14, ibm_marrakesh, Heron r2):** The two best-performing quantum models from this study — bw8 (brick-wall n=8, k=4) and vqc32 (angle VQC, k=32, π/6) — were tested on 200 fold-0 test ECGs at 1000 shots each on a real IBM superconducting device. bw8 achieved AUC 0.949 [95% CI 0.917–0.974] and vqc32 AUC 0.956 [0.924–0.982] on ibm_marrakesh. Per-ECG Pearson r between hardware and noiseless logit was 0.93 (bw8) and 0.97 (vqc32). AUC survives real hardware noise without mitigation, consistent with FakeFez simulation predictions from HARDWARE.md H3b. This is a hardware validation result, not a quantum-advantage claim; classical XGBoost (AUC 0.965 on 32 features, full training set) still leads. See [`HARDWARE.md`](HARDWARE.md) H6 for full details.

### k = 8 (primary)

In the pre-registered paired comparison of the best quantum model against the best classical model,
the quantum models **matched classical performance within 0.02 AUC at N = 25, 50, 100, 200 and 500** (N = 500: reduced sample, 10 of 40 subsets) and **outperformed the best classical model at no size**:

| N | best quantum vs best classical | mean d [95% CI] | claim |
|---|---|---|---|
| 25 | Q3 quantum-kernel SVM vs C3 GP | −0.0040 [−0.0099, +0.0012] | non-inferior |
| 50 | Q3 vs C3 GP | −0.0033 [−0.0058, −0.0009] | non-inferior (GP slightly better) |
| 100 (primary) | Q3 vs C5 random forest | −0.0028 [−0.0049, −0.0010] | non-inferior (RF slightly better) |
| 200 | Q1 angle VQC vs C5 random forest | −0.0018 [−0.0036, −0.0002] | non-inferior (RF slightly better) |
| 500 | Q1 vs C5 random forest | −0.0009 [−0.0020, +0.0002] | non-inferior; reduced sample |

Choosing "best" by inner-CV AUC instead of test AUC makes the N = 100 comparison a tie with SVM-RBF (−0.0006 [−0.0024, +0.0013]). Holm-adjusted superiority (96 comparisons) holds only against weaker classical models:
the quantum model outperformed **kNN** (all sizes), **XGBoost** (all sizes for Q1/Q3) and **plain LR** at N = 25 (Q3, +0.010) and N = 500 (Q1/Q3, +0.0045, reduced sample) — never SVM-RBF, the GP classifier or random forest.
One full-sample cell meets neither claim: Q2 (trained scales) vs GP at N = 25 — the classical model was better at that size; the CI straddles the margin (d = −0.018 [−0.025, −0.012]). The brick-wall (Q4, N = 100 only) was worse than five of six classical models. LR (9 parameters) matches the angle VQC (81) on AUC and train–test gap.

### k = 4 (secondary endpoint, complete 2026-11-14)

Same finding: quantum models non-inferior at every size, superior at none (Holm over 90 comparisons). Best quantum = Q1 angle VQC (most sizes) or Q3 quantum-kernel SVM (N=25); best classical = **C3 GP classifier at every size** (shifted from random forest at k=8):

| N | best quantum vs best classical | mean d [95% CI] | claim |
|---|---|---|---|
| 25 | Q3 quantum-kernel SVM vs C3 GP | −0.0033 [−0.0104, +0.0024] | non-inferior |
| 50 | Q1 angle VQC vs C3 GP | −0.0026 [−0.0058, +0.0004] | non-inferior |
| 100 | Q1 angle VQC vs C3 GP | −0.0022 [−0.0038, −0.0007] | non-inferior (GP slightly better) |
| 200 | Q1 angle VQC vs C3 GP | −0.0011 [−0.0025, +0.0000] | non-inferior |
| 500 | Q1 angle VQC vs C3 GP | −0.0007 [−0.0018, +0.0004] | non-inferior; reduced sample |

At k = 4 the quantum models load each of 4 features once (no re-uploading), and the quantum-kernel SVM (Q3) uses a product kernel that a classical computer evaluates exactly. The k = 4 AUC is ~0.07 below k = 8 for all models, consistent with less information.

This is an empirical result on classical ECG features; the theory in `brickwall/Scalable_Quantum_ML.pdf` (Prop. 8) does not predict hardness at this size, and the best quantum models here use a classically computable product kernel (geometric difference to RBF ≪ √N, `DISTINGUISHER.md`).
**Remaining incomplete:** Q4 at N ≠ 100 (torch blocked by Windows Application Control on the lab PC; a Colab run, as used for IBM, has not been tried), C7 TabPFN (licence login required).

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
* **C7 TabPFN: not run (licence login required).** tabpfn ≥ 2 needs an interactive browser licence login, and the v1 weights are no longer downloadable; it was not attempted unattended. The classical side is C1–C6 and the multiplicity count excludes C7. (Status changed from "pending" on 2026-11-13.)
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
| C7 TabPFN | all cells | **not run (licence login required)** |

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

## Throughput check (2026-11-13, after machine clean-up: reboot, apps closed, more RAM free)

`sd_bench_workers.py`: W concurrent workers each run the same unit (k=8, N=50, fold 0, r=39); nothing is written to the SD results. Free RAM during the runs: 5.7–5.8 GB.

| family | W=1 | W=2 | W=4 | W=6 | W=8 |
|---|---|---|---|---|---|
| main (Q1–Q3, C1–C6): s/unit → units/hour | 47 s → 76 | 48 s → 148 | 51 s → 275 | 59 s → 357 | 72 s → **392** |
| brick-wall (Q4): s/unit → units/hour | 19 s → 186 | 19 s → 361 | 22 s → 636 | 24 s → 847 | 30 s → **934** |

Chosen: **8 workers per family** (best measured), never more; one worker is dropped whenever free RAM falls below 3 GB. Before the clean-up the same main-family N=50 unit took >10 min in a 10-worker run — the earlier slowness was a machine-state problem (memory pressure), not the code.

## Sensitivity analyses at N = 100 (added 2026-11-13 AFTER the primary result; the primary result and claim level above stay as recorded)

Code `sd_sensitivity.py`; numbers `results/sd_sensitivity_N100.json`. C7 TabPFN, listed as "pending" in the primary section, is **not run (licence login required)**.

**(1) Bootstrap levels.** hierarchical bootstrap (sd_analyze.boot_mean): level 1 = the 5 outer folds, resampled with replacement; level 2 = the 40 subsets within each resampled fold, resampled with replacement; level 3 = test ECGs: NOT resampled (each subset's test AUC on the full outer test fold is treated as fixed). All 40 subsets of a fold share the same test ECGs, so test-set sampling noise enters as a common fold-level shift, which the fold level of the bootstrap resamples (with only 5 folds, coarsely).

Primary pair Q3 quantum-kernel SVM − C5 random forest: per-fold mean differences fold 0: -0.0023, fold 1: -0.0028, fold 2: -0.0045, fold 3: -0.0008, fold 4: -0.0039; mean -0.0028; hierarchical-bootstrap 95% CI [-0.0049, -0.0010]; fold-level t-interval over the 5 fold means (4 df) [-0.0046, -0.0010].

**(2) Best models chosen by inner-CV AUC** (on the N-ECG subsets, no test data): best quantum = Q3 quantum-kernel SVM (mean inner-CV AUC 0.9428), best classical = C2 SVM-RBF (0.9469). Difference on the test folds -0.0006, 95% CI [-0.0024, +0.0013], fold-level t-interval [-0.0026, +0.0014], quantum wins 0.45 of subsets, per-fold +0.0007, +0.0007, -0.0026, -0.0021, +0.0004. Inner-CV means: C2 SVM-RBF 0.9469, C1 LR 0.9454, Q3 quantum-kernel SVM 0.9428, Q1 angle VQC 0.9413, C3 GP classifier 0.9391, C5 random forest 0.9296, Q2 VQC trained scales 0.9260, C4 kNN 0.9150, Q4 brick-wall n=8 0.9102, C6 XGBoost 0.9091.

**(3) Train–test gap, angle VQC (Q1) vs logistic regression (C1).** Parameters: Q1 81 (72 circuit angles + 8 readout weights + 1 bias), plus 0 trained encoding scales; C1 9 (8 weights + 1 bias) — the VQC has 9× more trainable parameters. Gap Q1 +0.0673, C1 +0.0682; difference -0.0010 [-0.0038, +0.0017] (per fold +0.0010, -0.0035, +0.0018, -0.0019, -0.0021); test AUC difference +0.0020 [-0.0003, +0.0041].

Reading: (1) every fold shows the best classical model (random forest) slightly ahead of the best quantum model; the fold-level interval, which does not rely on resampling subsets, agrees with the hierarchical CI.
(2) If "best" is chosen without looking at test data (inner-CV AUC), the comparison becomes Q3 quantum-kernel SVM vs C2 SVM-RBF and is a tie (CI includes 0) — the small significant deficit in the primary comparison depends on picking random forest by test AUC.
(3) The angle VQC generalises exactly like logistic regression (gap difference −0.001, CI includes 0) and reaches the same test AUC, with 81 trainable parameters against LR's 9; the comparison is therefore not parameter-matched, and per parameter LR is the more efficient model.

## Limitations

* Single dataset (PTB-XL, MI vs NORM, 8 hand-crafted features); results may not transfer to other tasks or raw signals.
* All quantum models are simulated exactly on a classical computer; the angle-encoding kernel (Q3) is a product of cos² terms that is classically computable in O(k) per pair, and the brick-wall (Q4) is our reimplementation of Antonio's model (`quantum_utils.py` was not available). No hardware noise or shot noise is included.
* C7 TabPFN, the strongest pre-registered small-data tabular baseline, was **not run** (licence login required).
* Subset variance is large at small N (CIs per model ±0.006 at N=100), and only 5 outer folds carry test-set variation.
* Compute deviations (logged before analysis): Q4 at secondary sizes and all models at N=500 on 10 of the 40 subsets; Q4 at k=4 dropped.

## Secondary sizes, main family (Phase 2: N = 25, 50, 200, 500; Q1–Q3 and C1–C6; analysed 2026-11-13)

All cells complete under the logged deviations: 200 paired subsets at N = 25, 50, 100, 200; **N = 500: reduced sample (10 of 40 subsets, 50 pairs)**. Q4 is shown only at N = 100 (Phase 4 adds it at the other sizes).
The Holm family used here has **96 comparisons** (Q1–Q3 × C1–C6 × 5 sizes + Q4 × C1–C6 at N = 100); the final family after Phase 4 is larger and is reported there. Files: `results/sd_tables_k8_main.md`, `results/sd_analysis_k8_main.json`;
figures `results/figs/sd_auc_vs_N_main.png` (mean test AUC vs N, 95% CI bands), `results/figs/sd_diff_vs_N_main.png` (best quantum − best classical), `results/figs/sd_gap_vs_N_main.png`.

### k = 8: mean test AUC [95% hierarchical-bootstrap CI] (5 folds x up to 40 subsets), generalisation gap in parentheses

| model | N=25 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|
| Q1 angle VQC | 0.825 [0.815, 0.833] (+0.17) | 0.866 [0.860, 0.872] (+0.12) | 0.889 [0.882, 0.894] (+0.07) | 0.905 [0.900, 0.911] (+0.04) | 0.919 [0.913, 0.925] (+0.02) |
| Q2 VQC trained scales | 0.814 [0.803, 0.823] (+0.19) | 0.854 [0.846, 0.862] (+0.13) | 0.883 [0.876, 0.889] (+0.08) | 0.901 [0.895, 0.906] (+0.04) | 0.916 [0.910, 0.922] (+0.02) |
| Q3 quantum-kernel SVM | 0.828 [0.818, 0.837] (+0.17) | 0.866 [0.860, 0.872] (+0.11) | 0.889 [0.882, 0.895] (+0.07) | 0.905 [0.899, 0.910] (+0.04) | 0.919 [0.912, 0.925] (+0.02) |
| Q4 brick-wall n=8 | – | – | 0.857 [0.846, 0.866] (+0.07) | – | – |
| C1 LR | 0.818 [0.808, 0.828] (+0.18) | 0.862 [0.855, 0.869] (+0.12) | 0.887 [0.881, 0.893] (+0.07) | 0.902 [0.896, 0.909] (+0.04) | 0.914 [0.907, 0.922] (+0.01) |
| C2 SVM-RBF | 0.826 [0.818, 0.834] (+0.17) | 0.865 [0.860, 0.871] (+0.12) | 0.889 [0.884, 0.895] (+0.07) | 0.904 [0.898, 0.911] (+0.04) | 0.919 [0.912, 0.925] (+0.02) |
| C3 GP classifier | 0.832 [0.825, 0.840] (+0.16) | 0.869 [0.864, 0.875] (+0.11) | 0.891 [0.886, 0.897] (+0.07) | 0.906 [0.900, 0.912] (+0.04) | 0.918 [0.912, 0.925] (+0.02) |
| C4 kNN | 0.786 [0.776, 0.795] (+0.21) | 0.833 [0.825, 0.839] (+0.17) | 0.863 [0.859, 0.868] (+0.14) | 0.884 [0.878, 0.890] (+0.12) | 0.903 [0.897, 0.908] (+0.10) |
| C5 random forest | 0.828 [0.818, 0.837] (+0.17) | 0.868 [0.863, 0.874] (+0.13) | 0.891 [0.885, 0.898] (+0.10) | 0.907 [0.901, 0.912] (+0.08) | 0.920 [0.913, 0.926] (+0.06) |
| C6 XGBoost | 0.785 [0.772, 0.797] (+0.21) | 0.846 [0.839, 0.851] (+0.15) | 0.875 [0.868, 0.881] (+0.12) | 0.898 [0.893, 0.904] (+0.09) | 0.916 [0.910, 0.922] (+0.06) |

### Best quantum vs best classical per size (best = highest mean AUC at that size)

| N | best quantum | best classical | mean d [95% CI] | wins | Holm p (superiority) | superiority | non-inferior (CI lower > −0.02) |
|---|---|---|---|---|---|---|---|
| 25 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0040 [-0.0099, +0.0012] | 0.45 | 1.000 | False | True |
| 50 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0033 [-0.0058, -0.0009] | 0.41 | 1.000 | False | True |
| 100 | Q3 quantum-kernel SVM | C5 random forest | -0.0028 [-0.0049, -0.0010] | 0.42 | 1.000 | False | True |
| 200 | Q1 angle VQC | C5 random forest | -0.0018 [-0.0036, -0.0002] | 0.46 | 1.000 | False | True |
| 500 | Q1 angle VQC | C5 random forest | -0.0009 [-0.0020, +0.0002] | 0.40 | 1.000 | False | True |

**Claim level per size (best quantum vs best classical; pre-registered language):**
| N | claim level | label |
|---|---|---|
| 25 | matched classical performance within 0.02 AUC (non-inferior; d = −0.0040, CI includes 0) | – |
| 50 | matched classical performance within 0.02 AUC (d = −0.0033, CI below 0: GP slightly better) | – |
| 100 | matched classical performance within 0.02 AUC (primary endpoint, recorded above) | – |
| 200 | matched classical performance within 0.02 AUC (d = −0.0018, CI below 0: RF slightly better) | – |
| 500 | matched classical performance within 0.02 AUC (d = −0.0009, CI includes 0) | **reduced sample** (10 of 40 subsets) |

**Superiority (Holm over 96):** the quantum model outperformed **C1 LR** at N = 25 (Q3 quantum-kernel SVM, +0.010 [+0.005, +0.015], adjusted p = 0.019) and at N = 500 (Q1 +0.0046 [+0.0023, +0.0068] and Q3 +0.0045 [+0.0023, +0.0069], adjusted p = 0.019; **reduced sample**);
it outperformed **C4 kNN** at every size (Q1, Q2, Q3; +0.013 to +0.042) and **C6 XGBoost** at every size for Q1 and Q3 and at N = 25, 100, 200 for Q2 (+0.002 to +0.043). There is **no superiority over C2 SVM-RBF, C3 GP or C5 random forest at any size**, and none over the best classical model at any size.
Q1 vs LR has small raw-p advantages at N = 50 and 200 (+0.004, +0.002) that do not survive Holm.

**Cells meeting neither claim:** Q2 (trained scales) vs C3 GP at N = 25 (a full-sample cell, 200 pairs): d = −0.0183 [−0.0251, −0.0121] — pre-registered wording: the classical model was better at that size; the CI straddles the margin. (The "inconclusive" label is reserved by the logged deviation for reduced cells only.)
(Q4 at N = 100 vs LR, SVM-RBF, GP, RF and XGBoost also meets neither, as recorded in the primary section.)

**Generalisation gap:** Q1 and Q3 have a significantly smaller train–test gap than kNN and XGBoost at every size and than random forest at every size (Q3) or at N ≥ 50 (Q1); smaller than LR at N = 25 only; equal to SVM-RBF everywhere; Q1's gap is larger than GP's at N = 25 (equal elsewhere); the gaps of Q1/Q3 and LR, SVM-RBF and GP converge from ~0.17 at N = 25 to ~0.02 at N = 500.

**Reading.** The learning curves of Q1, Q3, SVM-RBF, GP and random forest lie on top of each other at every size (within ~0.005 AUC); the quantum models' consistent wins are against the two weakest classical models (kNN, XGBoost) and, at the extremes N = 25 and N = 500, against plain LR.
This is an empirical result on classical ECG features; the theory in `brickwall/Scalable_Quantum_ML.pdf` (Prop. 8) does not predict hardness at this size, and the Q1/Q3 feature map is a classically computable product kernel.

## Deviation logged 2026-11-13 09:00 (before the affected runs): brick-wall (Q4) at secondary sizes blocked

Starting at ~08:50 on 2026-11-13, Windows Application Control blocks torch's `_C` extension on every import ("DLL load failed while importing _C: An Application Control policy has blocked this file"), in `C:\bwvenv` and in a freshly created venv at another path (`C:\tvenv`).
The same environment ran the Q4 throughput benchmark at ~01:50 the same day. Q4 at N = 25, 50, 200, 500 (planned: subsets 0–9) therefore **could not be run**; 0 of 200 units produced results (failed units were not written).
No security setting was changed. Q4 remains available only at N = 100 (the primary endpoint, already recorded). The final Holm family is reported with this cell missing. If torch becomes loadable again, the Q4 runs are executed as a catch-up under the same plan and deviations.
The main family at k = 4 does not need torch and proceeds now.

## Status 2026-11-13 10:38: brick-wall (Q4) secondary sizes — incomplete (torch blocked by Windows Application Control)

Checked whether the exact faster brick-wall variant (`bw_torch.psi_fast`: column unitary = matrix_exp of commuting generators) can run without torch: **it cannot** — it uses `torch.linalg.matrix_exp` and torch autograd for the gradients.
A numpy port would also not meet the required verification (test AUC of existing N = 100 Q4 units reproduced to 1e-10): those units were trained with complex64 torch arithmetic, and even the exact torch fast variant already differed by 0.0007 AUC on one subset.
Therefore, with no security setting changed and torch not retried: **Q4 at N = 25, 50, 200, 500: incomplete (torch blocked by Windows Application Control)**. Q4 exists only at N = 100.

## Exploratory equivalence check (exploratory, added after the results; not pre-registered; no pre-registered claim changes)

Best quantum vs best classical, both chosen by **inner-CV AUC** at each N (k = 8); "equivalent" if the 95% hierarchical-bootstrap CI of the paired difference lies inside ±0.005 AUC. Code `sd_equivalence.py`.

| N | best quantum (inner CV) | best classical (inner CV) | mean d [95% CI] | CI inside ±0.005 | note |
|---|---|---|---|---|---|
| 25 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0025 [-0.0032, +0.0080] | no |  |
| 50 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0008 [-0.0019, +0.0035] | yes |  |
| 100 | Q3 quantum-kernel SVM | C2 SVM-RBF | -0.0006 [-0.0024, +0.0013] | yes |  |
| 200 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0003 [-0.0010, +0.0016] | yes |  |
| 500 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0003 [-0.0009, +0.0015] | yes | reduced sample |

Reading: inner-CV picks the quantum-kernel SVM (Q3) and SVM-RBF at every size; they are **equivalent within ±0.005 AUC at N = 50, 100, 200 and 500** (N = 500 reduced sample) and the check is not met at N = 25 only because the CI is wider (+0.0025 [−0.0032, +0.0080]).

## Secondary endpoint — k = 4 (complete 2026-11-14; Q1–Q3 and C1–C6; 850/850 units)

Runs: 6 workers × 5 folds × (N=100: 40 subsets, N=50: 40 subsets, N=25: 40 subsets, N=200: 40 subsets, N=500: 10 subsets) = 850 units. Q4 not run at k=4 (logged deviation: Q4 at k=4 dropped due to torch block).
Files: `results/sd_analysis_k4_main.json`, `results/sd_tables_k4_main.md`, `results/sd_analysis_k4_main_k4N100.json`, figures `results/figs/sd_{auc,diff,gap}_vs_N_k4_main*.png`.

### k = 4: mean test AUC [95% hierarchical-bootstrap CI] (5 folds x up to 40 subsets), generalisation gap in parentheses

| model | N=25 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|
| Q1 angle VQC | 0.804 [0.792, 0.814] (+0.18) | 0.850 [0.844, 0.856] (+0.11) | 0.874 [0.868, 0.880] (+0.06) | 0.890 [0.886, 0.894] (+0.03) | 0.903 [0.895, 0.910] (+0.01) |
| Q2 VQC trained scales | 0.799 [0.788, 0.808] (+0.19) | 0.842 [0.834, 0.849] (+0.12) | 0.869 [0.864, 0.875] (+0.07) | 0.886 [0.883, 0.890] (+0.04) | 0.901 [0.893, 0.908] (+0.02) |
| Q3 quantum-kernel SVM | 0.810 [0.798, 0.820] (+0.17) | 0.850 [0.843, 0.856] (+0.11) | 0.873 [0.866, 0.880] (+0.06) | 0.888 [0.883, 0.892] (+0.03) | 0.902 [0.894, 0.910] (+0.01) |
| C1 LR | 0.807 [0.797, 0.815] (+0.17) | 0.850 [0.844, 0.856] (+0.11) | 0.874 [0.868, 0.881] (+0.05) | 0.889 [0.885, 0.893] (+0.03) | 0.901 [0.893, 0.909] (+0.01) |
| C2 SVM-RBF | 0.810 [0.802, 0.818] (+0.17) | 0.849 [0.842, 0.856] (+0.11) | 0.873 [0.865, 0.880] (+0.06) | 0.887 [0.883, 0.892] (+0.03) | 0.903 [0.894, 0.911] (+0.01) |
| C3 GP classifier | 0.814 [0.806, 0.821] (+0.17) | 0.853 [0.845, 0.860] (+0.11) | 0.876 [0.870, 0.883] (+0.05) | 0.891 [0.887, 0.895] (+0.03) | 0.903 [0.896, 0.911] (+0.01) |
| C4 kNN | 0.778 [0.769, 0.786] (+0.22) | 0.824 [0.817, 0.832] (+0.18) | 0.852 [0.845, 0.859] (+0.15) | 0.870 [0.865, 0.875] (+0.13) | 0.886 [0.879, 0.894] (+0.11) |
| C5 random forest | 0.803 [0.793, 0.813] (+0.19) | 0.847 [0.840, 0.854] (+0.14) | 0.871 [0.864, 0.877] (+0.10) | 0.886 [0.882, 0.890] (+0.08) | 0.902 [0.893, 0.910] (+0.05) |
| C6 XGBoost | 0.775 [0.763, 0.786] (+0.22) | 0.830 [0.823, 0.837] (+0.16) | 0.860 [0.853, 0.866] (+0.12) | 0.882 [0.878, 0.885] (+0.08) | 0.901 [0.892, 0.909] (+0.05) |

### Best quantum vs best classical per size (k = 4; Holm over 90 comparisons)

| N | best quantum | best classical | mean d [95% CI] | wins | Holm p (superiority) | superiority | non-inferior |
|---|---|---|---|---|---|---|---|
| 25 | Q3 quantum-kernel SVM | C3 GP classifier | −0.0033 [−0.0104, +0.0024] | 0.45 | 1.000 | False | True |
| 50 | Q1 angle VQC | C3 GP classifier | −0.0026 [−0.0058, +0.0004] | 0.44 | 1.000 | False | True |
| 100 | Q1 angle VQC | C3 GP classifier | −0.0022 [−0.0038, −0.0007] | 0.43 | 1.000 | False | True |
| 200 | Q1 angle VQC | C3 GP classifier | −0.0011 [−0.0025, +0.0000] | 0.43 | 1.000 | False | True |
| 500 | Q1 angle VQC | C3 GP classifier | −0.0007 [−0.0018, +0.0004] | 0.46 | 1.000 | False | True (**reduced sample**) |

**Superiority (Holm over 90):** outperformed **C4 kNN** at every size (Q1, Q2, Q3; +0.021 to +0.035) and **C6 XGBoost** at every size (Q1, Q2, Q3; +0.015 to +0.035). At N = 25, Q3 also outperformed **C5 random forest** (+0.0071 [+0.0027, +0.0117]). No superiority over C1 LR, C2 SVM-RBF or C3 GP classifier at any size. No superiority over the best classical model at any size.

**Reading.** At k = 4, all nine models (Q1–Q3, C1–C6) lie within about 0.015 AUC of each other at every size (Q2 is lowest, GP is highest). The pattern exactly replicates k = 8: quantum non-inferior, no superiority over the strong classical models. The k = 4 AUC is 0.06–0.08 below k = 8 uniformly across models.

This is an empirical result on classical ECG features; the theory in `brickwall/Scalable_Quantum_ML.pdf` (Prop. 8) does not predict hardness at this size.
