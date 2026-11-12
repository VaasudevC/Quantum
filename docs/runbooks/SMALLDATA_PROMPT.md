# Small-data study: prompt for Claude Code (queue after the current run)

```
New phase: SMALL-DATA STUDY (SD). Run unattended on branch `brickwall-overnight` (or a new branch
`smalldata` from it). Nobody will answer questions: decide, log, continue. No end time. Use the same
environment as before (C:\bwvenv for torch/quantum work, C:\bwcache for cached arrays; see
docs/REPRODUCE.md). Keep OVERNIGHT_LOG.md and a new report SMALLDATA.md updated after every step;
commit after every sub-phase; do not push.

GOAL: test rigorously whether quantum models match or beat classical models when training data is
small. The analysis plan below is PRE-REGISTERED: write it verbatim into SMALLDATA.md section
"Pre-registered analysis plan" and COMMIT IT BEFORE running any SD experiment. Do not change it
afterwards; anything added later goes under "Exploratory (not pre-registered)".

--------------------------------------------------------------------------------------------------
PRE-REGISTERED ANALYSIS PLAN
--------------------------------------------------------------------------------------------------
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
--------------------------------------------------------------------------------------------------

EXECUTION
SD0 - Write the plan into SMALLDATA.md, commit ("SD0: pre-registered plan"). Check that P5's
      learning curves fitted selection/scaling on the full inner-train; if they did, note in
      SMALLDATA.md that P5 is superseded by SD because of that leakage.
SD1 - Build sd_run.py: subset generator (save subset ids to results/sd_subsets.json so the study is
      exactly reproducible), per-subset preprocessing + selection + inner-CV tuning, all models,
      one JSON line per (model, fold, N, r, k) to results/sd_runs.jsonl, resumable, parallel over
      (fold, N, r) with (cores - 1) workers within the RAM limit. Smoke-test 1 fold x N=50 x 2
      subsets for every model and check timings; log the projected total time.
SD2 - Run primary k = 8, all N, all models. Order: N=100 first (primary endpoint), then 50, 200, 25,
      500, so the primary answer exists as early as possible. Commit after each N.
SD3 - Analysis script sd_analyze.py: all tests in the plan, Holm correction, tables, and figures:
      results/figs/sd_auc_vs_N.png (mean test AUC vs N, log-x, 95% CI bands, every model),
      results/figs/sd_diff_vs_N.png (best quantum minus best classical, CI, with 0 and -0.02
      lines), results/figs/sd_gap_vs_N.png (generalisation gap vs N).
SD4 - k = 4 runs (secondary), then rerun the analysis.
SD5 - Report: SMALLDATA.md headline that states exactly which claim level was reached (superiority
      / non-inferiority / neither) at each N, against which models, with numbers; tables; figures;
      limitations (single dataset, simulated quantum models, subset variance); exploratory notes.
      Update MORNING_SUMMARY.md with a 10-line SD summary. Commit.
EXTRA (exploratory, only after SD5, clearly labelled not pre-registered): 3 seeds for the quantum
      models' initialisation at N=100; external PTB validation of models trained on N=100 subsets
      (mean over subsets, patient-cluster bootstrap).
```
