# QB — is there any setting where a quantum model gives a real benefit on this task?

Branch `smalldata`. **This pre-registration was written and committed before any QB run.** Nothing below may be changed after a QB result has been seen; later additions go under "Exploratory (not pre-registered)".
Every setting tried, including failures, is recorded in `results/qb_registry.csv`.

## Working hypothesis being tested

The quantum models beat plain logistic regression (full data k=32: VQC π/6 0.951 vs LR 0.945, 5/5 folds; brick-wall n=12 +0.005; small data Q3 vs LR +0.010 at N=25 and +0.0045 at N=500),
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
