# QB — is there any setting where a quantum model gives a real benefit on this task?

Tag `v0.3-smalldata` (merged into `main`). **This pre-registration was written and committed before any QB run.** Nothing below may be changed after a QB result has been seen; later additions go under "Exploratory (not pre-registered)".
Every setting tried, including failures, is recorded in `results/qb_registry.csv`.

## Headline (updated 2026-11-14) — claim level: **no benefit found on the main analysis; 2 discovery candidates for QB6 confirmation**

* **Tried** (262 rows in `results/qb_registry.csv`): QB1 screened 1,200 feature-map settings (product, re-uploading, IQP/ZZ, clinically paired ZZ, Heisenberg evolution, brick-wall encoding; 5 scales; N = 100/500; k = 8/4; 3 discovery folds);
  QB3 compared the trained VQCs with 6 parameter-matched classical models on raw and bounded inputs; QB4 screened 213 feature placements per fold; QB5 tested 13 perturbation/shift/imbalance/subtype settings (30/30 units complete).
* **Passed the gate:** none of the feature maps (QB1). Expressivity beyond an RBF kernel (g ≥ √N/2) appears only at s = π/2, where label alignment collapses; QB2 was skipped as pre-registered.
* **Non-linearity hypothesis (QB3): supported.** At equal or fewer parameters, classical quadratic models (rank-2 factorization machine, sparse degree-2 LR) on the same bounded inputs match the trained-scale VQC (0.9562 / 0.9561 vs 0.9561) and exceed the π/6 VQC; the win over LR is explained by the encoding plus mild non-linearity, not by the circuit.
* **Other kinds of benefit (QB5, 30/30 units complete):** 2 candidates found — both in the inferior-MI subtype (subtype_IMI) setting: Q1 angle VQC +0.0103 [+0.0008, +0.0236] and Q3 quantum-kernel SVM +0.0091 [+0.0016, +0.0193] vs SVM-RBF. All other 28 settings: no quantum model ≥ +0.005 above the best classical model with CI > 0.
* **Confirmation (QB6): RUN (2026-11-14). NO BENEFIT CONFIRMED.** Two candidates sent; neither met the full Holm-adjusted criterion (Q1: CI spans 0 and point < +0.005; Q3: CI barely spans 0, lo = −0.0003). Both classified as "hypothesis for future work".
* **Incomplete:** QB4 training (stopped under the 6-hour rule); QB5 full-data part (logged in advance as incomplete).
* Wording: the main result (no benefit on the standard analysis) is **"no benefit found"**; the two subtype_IMI candidates are **"hypothesis for future work"** unless QB6 confirms. The word "advantage" is not used.

## Working hypothesis being tested

The quantum models beat plain logistic regression (full data k=32: VQC π/6 0.951 vs LR 0.945, mean +0.0060 [+0.0043, +0.0076] pooled over 3 seeds, 13/15 runs positive; brick-wall n=12 +0.005; small data Q3 vs LR +0.010 at N=25 and +0.0045 at N=500),
but degree-2 LR (0.958) beats every VQC and a degree-2 function of the angles explains 98.6% of the VQC output. **Hypothesis: the win over LR is "the model is mildly non-linear", not "the model is quantum".**
QB tests this and looks for a setting that breaks it. The π/6 product kernel (effective dimension ≈ 1.8, g(RBF‖Q) ≪ √N) closes the search for that map only; entangling and re-uploading maps have not been screened.

## Data roles (fixed)

* **Discovery:** outer folds 0, 1, 2 only (training parts; test folds 0–2 for discovery tests). All exploration, gating, tuning and candidate choice happen here.
* **Confirmation:** outer folds 3 and 4 and the external PTB set, used **once**, at the end (QB6).
* **Caveat:** folds 3 and 4 were already used in the main 5-fold studies (choice of π/6, the L1 selector, the SD study), so they are clean only for choices made new in QB. **The external PTB set is the stronger confirmation.**
* "Best classical" is always chosen by **inner-CV AUC**, never by test AUC.

## Feature maps (QB1), each at s ∈ {π/8, π/6, π/4, π/3, π/2}, angles a = s·tanh(z), first k ∈ {8, 4} L1 features of the fold, 8 qubits (k = 4: the other 4 angles are 0)

* **F0** product map: Ry(a_i) on |0⟩ (reference; must reproduce the X2 numbers in `DISTINGUISHER.md`).
* **F1** re-uploading map: the same 8 angles loaded L ∈ {2, 3, 4} times as Ry layers, with the fixed CNOT ring (i → i+1 mod 8) between loads; no trained gates.
* **F2** IQP / ZZ map: H on all qubits, Rz(a_i), then RZZ(c·a_i·a_j) on ring neighbours (i, i+1 mod 8); 1 and 2 repetitions; c ∈ {0.5, 1, 2}.
* **F3** F2 with RZZ only on **clinically paired** features (pairing table below) instead of ring neighbours.
* **F4** Hamiltonian-evolution map: Heisenberg ring H = Σ_i (X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1}) + Σ_i a_i Z_i, initial state |+⟩^⊗8 (H on all qubits), evolution time t = 1 split into T ∈ {1, 2, 3} first-order Trotter steps (exact 2-qubit exponentials per bond, even then odd bonds).
* **F5** brick-wall encoding state D(φ)|φ0⟩ from `bw_utils.py` (numpy, untrained, n = 8, particle number 4), phase φ = s·tanh(z) — run only if it works without torch (it does: bw_utils is numpy).

Kernels: fidelity kernel K_Q(x, x') = |⟨ψ(x)|ψ(x')⟩|²; projected kernel K_P = exp(−γ Σ_q ‖ρ_q(x) − ρ_q(x')‖²_F) from 1-qubit reduced density matrices (Huang et al. 2021), γ tuned like the RBF γ.
(F5's 1-qubit RDMs are diagonal and data-independent — its projected kernel is degenerate and is reported as such.)

### F3 pairing table (fixed now, applied by feature NAME to whichever 8 (or 4) features the fold's L1 order selects)

Two selected features are paired (get an RZZ) if any rule holds:
1. **Same lead**: both features belong to the same lead (e.g. `aVR_t_amp` – `aVR_t_inverted`; ST with T of the same lead).
2. **Neighbouring precordial leads**: leads V_i and V_{i±1} (V1–V2, V2–V3, V3–V4, V4–V5, V5–V6).
3. **Reciprocal territories**: an inferior lead (II, III, aVF) with a high-lateral lead (I, aVL).
4. **Same inferior territory**: two different inferior leads among II, III, aVF.
aVR is paired only by rule 1. If a fold's selection yields no pair, F3 reduces to Rz-only encoding and is reported as "no clinical pairs among the selected features".

## Screen and gate (QB1)

On N = 100 and N = 500 ECGs per discovery fold (class-stratified random sample of the fold's training part, seed (fold, N)), for k = 8 and k = 4, per map and scale:
g(K_RBF‖K_Q) at λ ∈ {1e-3, 1e-2, 1e-1} (RBF γ tuned by 5-fold CV AUC of a precomputed SVC, C = 1), centered kernel–target alignment, effective dimension (spectral participation ratio), mean and variance of the off-diagonal entries, 5-fold CV AUC of a precomputed-kernel SVC (C = 1); the same for the projected kernel.

**Gate (fixed):** a map (at a given scale) passes to QB2 only if, at **N = 500 and k = 8**, in **at least 2 of 3 discovery folds**: g(K_RBF‖K_Q) ≥ √N / 2 at **every** λ ≥ 1e-3 tested, AND the mean off-diagonal kernel value > 0.05, AND its centered alignment ≥ 90 % of the RBF kernel's.
At most 4 (map, scale) settings pass, ranked by mean centered alignment. If none pass: write that result, skip QB2, go to QB3.

## Models, controls (dequantized twins) and tests

* **Quantum (QB2):** fidelity-kernel SVM and projected-kernel SVM for each passing map; C ∈ {0.1, 1, 10} by inner CV.
* **Classical:** C1 LR, C2 SVM-RBF, C3 GP, C5 random forest (SD grids), degree-2 LR (C grid {0.01, 0.1, 1, 10}).
* **Dequantized twins (required for every quantum model):** (i) the same bounded inputs a = s·tanh(z) fed to LR, degree-2 LR and SVM-RBF;
  (ii) a classical kernel with the same frequency content: the trigonometric product kernel K_L(a, a') = Π_i cos^{2L}((a_i − a'_i)/2) with L = number of data loads (F0: L=1, F1: L, F2/F3: 2·repetitions, F4: T), plus random Fourier features of the bounded inputs (512 features, bandwidth by inner CV).
  A quantum win that its twin also achieves is reported as **"explained by the encoding, not by the circuit"**.
* **QB2 protocol:** discovery folds 0–2, N ∈ {100, 500}, cached SD subsets r = 0–9, k = 8, inner 3-fold CV exactly as SD, refit on the subset, score the full test fold. Paired hierarchical bootstrap (folds, then subsets; 5000 resamples) as `sd_analyze.py`.
* **QB3 (full data, k = 32, discovery folds, existing trained VQCs):** degree-2 surrogate of the trained π/6 VQC and trained-scale VQC (rank, spectrum, top feature pairs of the quadratic-coefficient matrix);
  parameter-matched classical models with ≤ 113 parameters: LR (33), factorization machines rank 1 and 2 (65, 97), MLP 32-2-1 and 32-3-1 tanh (69, 103), L1-penalised degree-2 LR with ≤ 81 non-zero coefficients; tuned on validation AUC like the VQC.
* **QB4 (k = 32, π/6, discovery folds):** feature placements (a) L1 order, (b) 10 random, (c) clinically paired (F3 pairs share a qubit in consecutive layers), (d) best centered alignment among 200 random; training-free score for all, train (a), (c), (d) and best-2/worst-2 of (b), 1000 steps. "Placement does not matter" if the test-AUC spread < 0.002.
* **QB5 (only Q1, Q3, QB2 winners and their twins; discovery folds; N = 100 and full data):** label noise 10/20/30 %; test-time Gaussian input noise 0.1–0.5 SD and single-lead dropout; shift by recording site/device and by recording date; MI prevalence 5 % / 10 % in training; inferior MI vs NORM. Report the DROP vs the clean setting, paired CIs.
* **QB6 confirmation:** at most **3** candidates — the largest discovery effects that also beat their twin. One paired test each on folds 3–4 (hierarchical bootstrap) and on the external PTB set (patient-cluster bootstrap). Holm over the candidates sent.

## Claim rules (fixed)

* Smallest effect of interest: **+0.005 AUC**. A benefit claim needs, on confirmation, the lower 95% CI bound of the paired difference vs the best classical model (chosen by inner CV) **> 0** after Holm over the candidates sent, **and** a point estimate **≥ +0.005**, **and** a win over its dequantized twin.
* Wording: **"benefit confirmed"** (passes confirmation and beats its twin), **"hypothesis for future work"** (discovery only), or **"no benefit found"**. The word "advantage" is not used unless a benefit is confirmed against the best classical model and the twin.

## Execution and stop rules

numpy / scipy / scikit-learn / xgboost only (no torch); ≤ 4 workers, 1 BLAS thread each, RAM guard on; every script resumable. Order: QB1 → QB3 → QB4 → QB2 → QB5 → QB6.
Any phase projected to exceed 6 hours on this machine is stopped, logged as incomplete with a time estimate, and the next phase starts.

---

## Results (added after the runs; the pre-registration above is unchanged)

### QB1 — training-free screen (run 2026-11-13; `qb1_screen.py`, `qb1_gate.py`, `results/qb1_screen.jsonl`, `results/qb1_table.md`, `results/paper/table4_feature_maps.md`, `results/paper/fig4_g_all_maps.png`)

* 1200 settings screened (6 maps / 20 parameterisations × 5 scales × N ∈ {100, 500} × k ∈ {8, 4} × 3 discovery folds; F5 = the brick-wall encoding state, numpy). All states verified (unit norm 1e-15; F0 = product formula 1e-15; F4 = dense matrix exponential 2e-16). **F0 at π/6 reproduces the X2 numbers exactly** (g and CV AUC in all 6 fold × N cells).
* **Gate result: no (map, scale) setting passes** (0 of 100 at N = 500, k = 8). Over the 300 fold-settings: g ≥ √N/2 at all three λ fails in 286, the alignment criterion fails in 160, the concentration criterion in 15.
* **Trade-off (the paper result):** g ≥ √N/2 at all λ is reached only at the largest scale **s = π/2** (F1, F2, F3, F4, F5), where the kernel's centered alignment with the labels falls to 30–60 % of the RBF kernel's and CV AUC drops to 0.85–0.92 (RBF 0.92–0.94). Across all maps at N = 500 (k = 8, λ = 1e-2), median g and median alignment ratio (quantum/RBF) move together with the scale: π/8 4.4 / 1.03; π/6 5.9 / 0.97; π/4 10.8 / 0.90; π/3 17.3 / 0.75; π/2 27.8 / 0.49 — g crosses √N/2 = 11.2 just where the alignment falls below the 90 % gate.
  Entangling (IQP/ZZ, clinically paired ZZ, Heisenberg evolution), re-uploading and the brick-wall encoding all behave like the product map: **expressivity beyond an RBF kernel appears only together with a loss of label alignment.** This extends the scale rule from "which scale" to "which feature map".
* Near-misses, reported for completeness: 11 settings meet the three criteria in 1–2 folds **at λ = 1e-2 only**; at λ = 1e-1 their g is 4.9–7.0 (< 11.2), and their CV AUC is at most +0.004 above RBF (most are equal or below). Under the pre-registered rule they do not pass.
* Projected (1-RDM) kernels: CV AUC within the same range as the fidelity kernels; F5's projected kernel is degenerate (particle-number sector), as stated in advance.
* **QB2 is skipped** (no map passed the gate), as pre-registered.

### QB3 — is the win over LR more than non-linearity? (run 2026-11-13; `qb3_nonlinearity.py`, `results/qb3.jsonl`; full data, k = 32, discovery folds 0–2, test AUC means over the 3 folds)

| model | trainable parameters | on z-scored inputs | on bounded inputs a = (π/6)·tanh z (twin) |
|---|---|---|---|
| angle VQC, s = π/6 (logged runs) | 81 | **0.9519** | – |
| angle VQC, trained scales (logged runs) | 113 | **0.9561** | – |
| LR | 33 | 0.9460 | 0.9485 |
| factorization machine, rank 1 | 65 | 0.9532 | 0.9558 |
| factorization machine, rank 2 | 97 | 0.9544 | **0.9562** |
| MLP 32-2-1 (tanh) | 69 | 0.9476 | 0.9484 |
| MLP 32-3-1 (tanh) | 103 | 0.9516 | 0.9532 |
| L1 degree-2 LR, ≤ 81 non-zero (66–70 used) | 67–71 | 0.9551 | **0.9561** |

**Answer (one sentence): yes — at equal or fewer parameters, classical quadratic models match the VQC: a rank-2 factorization machine (97 parameters) and a sparse degree-2 LR (67–71 parameters) on the same bounded inputs reach 0.9562 / 0.9561, equal to the trained-scale VQC (0.9561, 113 parameters), and already exceed the π/6 VQC (0.952) on z-scored inputs.**
The VQC therefore behaves like a quadratic classifier on its encoded inputs; its win over LR is explained by the encoding plus mild non-linearity, not by the circuit (wording fixed in the pre-registration). No QB3 setting is a confirmation candidate.
Degree-2 surrogates: R² = 0.984–0.986 for the π/6 VQC (linear-only 0.91–0.93) and 0.971–0.975 for the trained-scale VQC (linear-only 0.87–0.88). The π/6 VQC's quadratic-coefficient matrix is spread (effective rank ≈ 21–22 of 32, top eigen-direction 14–17 % of the energy, half of it on the diagonal);
the trained-scale VQC's is concentrated (effective rank 1.0–9.1; top direction 58–100 %), with very large weights on a few pairs (e.g. fold 1: aVF–III ST curvature) — the trained scales let the model emphasise a handful of feature products.

### Deviation logged 2026-11-13 12:39 (before any QB5 run)

1. **Hyper-parameters under perturbation:** at N = 100 every model reuses the configuration chosen by inner CV on the **clean** version of the same SD subset (Q1, Q3, C1, C2, C3, C5: the `chosen` field of `results/sd_runs_main.w*.jsonl`; twins: inner CV on the clean subset) and is retrained **once** on the perturbed training data. Re-tuning per perturbation (×13 trainings for Q1) does not fit the compute budget.
2. **Subsets:** clean, label-noise, test-noise and lead-dropout settings use the cached SD subsets r = 0–9 of discovery folds 0–2 (30 units). Site shift, date shift, class imbalance and the inferior-MI subtype draw new N = 100 subsets (10 per fold, one ECG per patient, seed (fold, setting, r)) from the restricted pools, with median/z-score/L1 selection fitted on each new subset as in SD.
3. **Definitions:** site shift = train on the largest recording site only, test on test-fold ECGs from all other sites; date shift = train on recordings before the median date of the fold's training part, test on test-fold recordings after it; the "clean" reference for both is the model trained on the ordinary subset r, tested on the same restricted test ECGs. Lead dropout = all selected features of one lead set to the subset median (0 in z-units after centring on the median) at test time, averaged over the leads present. Inferior MI = label_subclass "IMI" vs NORM. Twins use s = π/6.
4. **Full data (QB5):** Q1 at full data needs ~27 training runs (≈ 9 CPU-h at the current contended speed) and Q3 at full data a ~11k × 11k Gram matrix (≈ 1 GB, not available next to the user's applications). **The full-data part of QB5 is marked incomplete** (estimate ≈ 9 CPU-h, i.e. ≈ 3–4 h with 3 free cores); QB5 reports N = 100 only.

### QB4 — feature placement (k = 32, π/6, discovery folds 0–2; `qb4_placement.py`, `results/qb4_screen.jsonl`)

**Training-free screen (complete):** centered alignment of the encoding-only re-uploading kernel (θ = 0) on 500 training ECGs per fold, for 213 placements per fold:

| fold | L1 order | clinically paired | best of 200 random | 10 random: min – max | 200 random: min – max |
|---|---|---|---|---|---|
| 0 | 0.418 | 0.408 | 0.433 | 0.401 – 0.425 | 0.387 – 0.433 |
| 1 | 0.385 | 0.396 | 0.396 | 0.371 – 0.386 | 0.360 – 0.396 |
| 2 | 0.346 | 0.356 | 0.374 | 0.347 – 0.372 | 0.343 – 0.374 |

The spread of the training-free score over placements is ±0.02–0.03 around the L1 order. The clinically paired placement is above the L1 order in folds 1–2 and below it in fold 0. The best of 200 random placements is above it in all folds (by 0.01–0.03).

**Training part: INCOMPLETE (stopped 2026-11-13 13:29 under the 6-hour rule).** Each 1000-step k = 32 run took > 72 min wall at ~20–66 % of a core while the machine was in use (1.46 s/step at batch 1000 vs 0.38 s originally). None of the 21 planned runs finished. Projected total ≈ 21 runs × 40–60 CPU-min ≈ 14–21 CPU-h (≈ 5–7 h with 3 free cores on an idle machine).
**Resume:** `sh qb4_run.sh 0`, `sh qb4_run.sh 1`, `sh qb4_run.sh 2` (resumable; the L1 order must reproduce `results/stage2_steps1000.jsonl`). Whether placement changes test AUC by more than 0.002 is therefore **not answered**.

### QB5 — other kinds of benefit (N = 100, discovery folds 0–2; `qb5_robust.py`, `qb5_analyze.py`, `results/qb5.jsonl`) — **COMPLETE: 30/30 units** (updated 2026-11-14)

Best classical by inner CV (clean SD subsets, folds 0–2, r 0–9): **C2_svm_rbf**. 30 units (fold × subset). d = quantum − comparator on the perturbed AUC; Δdrop = comparator drop − quantum drop (> 0: quantum loses less).

| setting | quantum | mean AUC quantum / best classical | d vs best classical [95% CI] | Δdrop vs best classical [95% CI] | d vs TW_LR / TW_LR2 / TW_SVM | candidate |
|---|---|---|---|---|---|---|
| testnoise_0.1 | Q1 | 0.893 / 0.895 | -0.0020 [-0.0050, +0.0008] | -0.0001 [-0.0005, +0.0005] | +0.0035 / +0.0018 / +0.0008 | no |
| testnoise_0.1 | Q3 | 0.893 / 0.895 | -0.0014 [-0.0040, +0.0015] | -0.0005 [-0.0015, -0.0001] | +0.0041 / +0.0024 / +0.0014 | no |
| testnoise_0.3 | Q1 | 0.883 / 0.886 | -0.0032 [-0.0067, +0.0006] | -0.0013 [-0.0030, +0.0018] | +0.0031 / +0.0029 / +0.0007 | no |
| testnoise_0.3 | Q3 | 0.882 / 0.886 | -0.0039 [-0.0075, -0.0004] | -0.0031 [-0.0052, -0.0015] | +0.0024 / +0.0022 / -0.0000 | no |
| testnoise_0.5 | Q1 | 0.865 / 0.872 | -0.0069 [-0.0100, -0.0023] | -0.0049 [-0.0080, -0.0003] | +0.0025 / +0.0042 / +0.0007 | no |
| testnoise_0.5 | Q3 | 0.864 / 0.872 | -0.0079 [-0.0120, -0.0042] | -0.0070 [-0.0097, -0.0047] | +0.0014 / +0.0031 / -0.0004 | no |
| leaddrop_mean | Q1 | 0.884 / 0.886 | -0.0026 [-0.0057, +0.0002] | -0.0006 [-0.0015, +0.0002] | +0.0033 / +0.0010 / +0.0005 | no |
| leaddrop_mean | Q3 | 0.885 / 0.886 | -0.0012 [-0.0042, +0.0022] | -0.0003 [-0.0014, +0.0006] | +0.0047 / +0.0023 / +0.0019 | no |
| leaddrop_worst | Q1 | 0.862 / 0.868 | -0.0064 [-0.0114, -0.0027] | -0.0045 [-0.0075, -0.0023] | +0.0019 / -0.0034 / -0.0024 | no |
| leaddrop_worst | Q3 | 0.865 / 0.868 | -0.0036 [-0.0094, +0.0009] | -0.0027 [-0.0071, +0.0003] | +0.0047 / -0.0006 / +0.0004 | no |
| labelnoise_0.1 | Q1 | 0.886 / 0.893 | -0.0075 [-0.0126, -0.0034] | -0.0056 [-0.0096, -0.0021] | +0.0003 / +0.0105 / -0.0041 | no |
| labelnoise_0.1 | Q3 | 0.890 / 0.893 | -0.0031 [-0.0090, +0.0021] | -0.0022 [-0.0071, +0.0027] | +0.0046 / +0.0148 / +0.0002 | no |
| labelnoise_0.2 | Q1 | 0.869 / 0.884 | -0.0141 [-0.0227, -0.0062] | -0.0122 [-0.0213, -0.0040] | -0.0090 / +0.0251 / -0.0132 | no |
| labelnoise_0.2 | Q3 | 0.881 / 0.884 | -0.0026 [-0.0121, +0.0054] | -0.0017 [-0.0100, +0.0056] | +0.0026 / +0.0366 / -0.0016 | no |
| labelnoise_0.3 | Q1 | 0.834 / 0.862 | -0.0282 [-0.0475, -0.0099] | -0.0263 [-0.0448, -0.0084] | -0.0242 / +0.0487 / -0.0203 | no |
| labelnoise_0.3 | Q3 | 0.853 / 0.862 | -0.0094 [-0.0253, +0.0050] | -0.0085 [-0.0233, +0.0052] | -0.0053 / +0.0676 / -0.0015 | no |
| imbalance_0.05 | Q1 | 0.798 / 0.823 | -0.0249 [-0.0536, -0.0014] | -0.0230 [-0.0515, +0.0005] | +0.0020 / +0.0502 / -0.0072 | no |
| imbalance_0.05 | Q3 | 0.816 / 0.823 | -0.0074 [-0.0196, +0.0041] | -0.0065 [-0.0198, +0.0057] | +0.0194 / +0.0677 / +0.0102 | no |
| imbalance_0.1 | Q1 | 0.837 / 0.847 | -0.0100 [-0.0251, +0.0033] | -0.0081 [-0.0241, +0.0054] | +0.0060 / +0.0272 / -0.0002 | no |
| imbalance_0.1 | Q3 | 0.840 / 0.847 | -0.0073 [-0.0162, +0.0016] | -0.0065 [-0.0150, +0.0024] | +0.0086 / +0.0298 / +0.0024 | no |
| site | Q1 | 0.882 / 0.886 | -0.0040 [-0.0120, +0.0022] | +0.0001 [-0.0063, +0.0055] | +0.0035 / -0.0010 / +0.0019 | no |
| site | Q3 | 0.885 / 0.886 | -0.0016 [-0.0053, +0.0020] | -0.0001 [-0.0045, +0.0046] | +0.0060 / +0.0015 / +0.0043 | no |
| date | Q1 | 0.890 / 0.891 | -0.0006 [-0.0041, +0.0032] | +0.0010 [-0.0040, +0.0059] | +0.0048 / +0.0076 / +0.0029 | no |
| date | Q3 | 0.889 / 0.891 | -0.0019 [-0.0062, +0.0029] | -0.0019 [-0.0053, +0.0018] | +0.0035 / +0.0063 / +0.0016 | no |
| subtype_IMI | Q1 | 0.903 / 0.893 | **+0.0103 [+0.0008, +0.0236]** | – | +0.0021 / +0.0245 / +0.0080 | **yes** |
| subtype_IMI | Q3 | 0.902 / 0.893 | **+0.0091 [+0.0016, +0.0193]** | – | +0.0008 / +0.0233 / +0.0068 | **yes** |

**Final reading (30/30 units, 2026-11-14):** 2 candidates — both in the **subtype_IMI** setting (inferior MI vs NORM, N = 100 subsets drawn from the inferior-MI pool). Q1 angle VQC: +0.0103 [+0.0008, +0.0236]; Q3 quantum-kernel SVM: +0.0091 [+0.0016, +0.0193] vs SVM-RBF (best classical by inner CV). Both pass the ≥ +0.005 and CI > 0 candidate criteria. All other 28 settings: no quantum model is a candidate.
Against the dequantized twins: both candidates beat the bounded-input deg-2 LR twin (+0.0245 / +0.0233) and the bounded SVM twin (+0.0080 / +0.0068), but only by small margins vs the bounded LR twin (+0.0021 / +0.0008). The twin comparison is part of the QB6 confirmation protocol.
Note: the interim table (18/30 units, fold 0 only) showed subtype_IMI at +0.0025/+0.0037 — both below the +0.005 candidate threshold and with CIs spanning 0. Folds 1 and 2 added the remaining 20 units and shifted the estimate to the final values above. The candidate status was applied only after all 30 units completed, as pre-registered.

### QB6 — confirmation (protocol written 2026-11-14; RUN 2026-11-14)

**Protocol (written and committed BEFORE any QB6 run).**

Two candidates from QB5 are sent to confirmation: **(1) subtype_IMI Q1 angle VQC** (+0.0103 [+0.0008, +0.0236] vs SVM-RBF, discovery) and **(2) subtype_IMI Q3 quantum-kernel SVM** (+0.0091 [+0.0016, +0.0193] vs SVM-RBF, discovery).

**QB6 confirmation protocol (fixed before running):**
* **Data:** outer folds 3 and 4 (not used in any QB discovery step). External PTB set if inferior-MI labels are present and identifiable from the available metadata.
* **Subsets:** N = 100 subsets from the inferior-MI pool (inferior MI vs NORM), 10 subsets per fold (r = 0–9), seed (fold, "subtype_IMI", r) — identical construction to QB5 subtype_IMI.
* **Hyper-parameters:** reuse the configuration chosen by inner CV on the clean (non-perturbed) SD subset of the same fold and r, exactly as QB5. No re-tuning on folds 3–4.
* **Comparators:** SVM-RBF (best classical by inner CV, same as QB5) **and** the three dequantized twins: bounded-input LR, bounded-input deg-2 LR, bounded-input SVM-RBF.
* **Bootstrap:** paired hierarchical bootstrap exactly as sd_analyze.py (B = 5000, resample folds then subsets, percentile 95% CI). One run only; no re-runs.
* **Holm correction:** over the 2 candidates (not over models or twins; twins are reported separately). A candidate passes only if the Holm-adjusted CI lower bound vs SVM-RBF > 0 AND point estimate ≥ +0.005 AND it beats its twin (bounded-input SVM-RBF).
* **Wording if confirmed:** "benefit confirmed (subtype_IMI)". If not confirmed: "hypothesis for future work (subtype_IMI)". Either way, the main-analysis claim remains "no benefit found".
* **Script:** `qb6_confirm.py`. Output: `results/qb6.jsonl`, `results/qb6_table.md`, `results/qb6_analysis.json`.
  Run: `..\.venv\Scripts\python.exe qb6_confirm.py 3 4`

**Status: RUN (2026-11-14). Results: NO BENEFIT CONFIRMED.**

#### QB6 Results (20 units, folds 3–4, r = 0–9)

| candidate | mean d [95% CI] | CI lo > 0 | point >= +0.005 | Holm pass | claim |
|---|---|---|---|---|---|
| subtype_IMI Q1 angle VQC | +0.0030 [−0.0049, +0.0108] | No | No | No | hypothesis for future work |
| subtype_IMI Q3 quantum-kernel SVM | +0.0055 [−0.0003, +0.0119] | No | Yes | No | hypothesis for future work |

vs dequantized twins (Q3): vs twin LR +0.0023 [−0.0007, +0.0053]; vs twin deg-2 LR +0.0318 [+0.0196, +0.0478]; vs twin SVM +0.0087 [+0.0033, +0.0145].

Both Q1 and Q3 point estimates are positive but neither meets the full claim (CI lower bound > 0 AND point ≥ +0.005 after Holm adjustment). Q3 barely misses: CI lower bound = −0.0003 (needs > 0); Q1 point estimate = +0.0030 (below +0.005 threshold). The discovery finding (QB5, folds 0–2) did not replicate on confirmation folds 3–4.

**Overall claim (final):** the subtype_IMI finding is classified as **"hypothesis for future work"**. The main-analysis claim (**no benefit found on the standard analysis**) is unchanged. Files: `results/qb6.jsonl`, `results/qb6_table.md`, `results/qb6_analysis.json`.

---

## Exploratory (not pre-registered) — added 2026-11-14

### QB3 extended to folds 3 and 4 (descriptive only; logged BEFORE running)

**Logged 2026-11-14 before any run on folds 3–4.**

QB is closed with no candidate, so folds 3 and 4 are no longer needed for a QB confirmation run. This extension runs the existing QB3 script (`qb3_nonlinearity.py`) on folds 3 and 4 for a **descriptive 5-fold table only**. It supports no benefit claim — it is purely to characterise whether the non-linearity finding from folds 0–2 holds across all folds.

Script: `..\.venv\Scripts\python.exe qb3_nonlinearity.py 3` and `... 4` (appends to `results/qb3.jsonl`). Analysis: `p5_qb3_extended.py` → `results/qb3_5fold_table.md`.

Figure 5 (not pre-registered): heat map of the degree-2 surrogate coefficient matrix (VQC vs degree-2 LR, fold 0) with spectrum. Script: `p5_fig5.py` → `results/figs/fig5_qb3_coef_heatmap.png`.
