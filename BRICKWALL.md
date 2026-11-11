# BRICKWALL — Antonio's magic-state brick-wall QNN vs tuned classical models on PTB-XL (MI vs NORM)

Branch `brickwall-overnight`. Chronological log: [`OVERNIGHT_LOG.md`](OVERNIGHT_LOG.md). This report is updated after every phase.
Status: **P0 ✔ · P1 ✔ · P2 ✔ · P3 … · P4 … · P5 … · P6 … · P7 …** (see the bottom of this file for what is still open).

## Headline (updated after each phase)

* **P1 (classical benchmarks, mentor request).** Tuned XGBoost is the best model at every feature count (5-fold test AUC: 0.937 at k=8, 0.945 at k=12,
  0.952 at k=16, 0.965 at k=32, **0.973 with all 249 features**). Random Forest and an 8/16/32-unit MLP are within ~0.003 of each other and of
  degree-2 LR at k ≥ 12. Our earlier angle VQC (π/6) is below XGBoost, RF and MLP at k=8 (0.928 vs 0.937) and at k=32 (0.951 / 0.955 trained scales vs 0.965).
* Remaining headline items (brick-wall results, external validation, learning curves, advantage-rule verdicts) are filled in below as the phases finish.

## P0 — setup and sanity

* Environment: `../.venv` (Python 3.14.3, numpy 2.4.6, scikit-learn 1.9.0, xgboost 3.4.1, wfdb 4.3.1) for all classical work, feature extraction and the angle VQC;
  `C:\bwvenv` (torch 2.15.0+cpu, pennylane 0.43.3, qiskit 2.6.2, qiskit-aer 0.18.2, qiskit-ibm-runtime 0.52.0) for the brick-wall simulator and hardware prep.
  (`import torch` is blocked by Windows Application Control inside the OneDrive `.venv`, and some sklearn/pandas DLLs are intermittently blocked in `C:\bwvenv`;
  the torch workers therefore use only numpy + torch and read pre-computed arrays from `C:\bwcache`.) 16 cores; only ~1 GB RAM was free, so ≤ 4 workers ran at a time.
* Sanity: fold-0 k=32 π/6 test AUC recomputed from the stored parameters with `scale_cv.prep()` = **0.959758698692389**, identical to the logged value (`p0_sanity.py`).

## P1 — classical benchmarks (5-fold patient-grouped CV, hyper-parameters chosen on validation AUC)

Models: LR (C grid), degree-2 LR (C grid), Random Forest (n_estimators {300,800} × max_depth {None,8,16} × min_samples_leaf {1,5}), XGBoost (depth {3,5,7} ×
learning rate {0.03,0.1}, ≤ 800 trees, early stopping on validation, subsample = colsample = 0.8), MLP (hidden 8/16/32 × L2 {1e-4,1e-2} × 2 seeds, early stopping).
k features come from the L1 path of the inner-train fold (`cohen_dedup` also at k = 8, 12). Degree-2 LR is skipped at k=249 (31k features).
Rows labelled "existing" are our earlier angle-VQC runs (1000 steps, same folds; `results/n4_clinical.json`). Code `p1_classical.py`, data `results/p1_classical.jsonl`.

| selector | k | model | test AUC (mean ± SD, 5 folds) | sens @ 90% spec | val AUC | folds |
|---|---|---|---|---|---|---|
| l1 | 8 | LR | 0.9222 ± 0.0067 | 0.786 | 0.9247 | 5 |
| l1 | 8 | degree-2 LR | 0.9312 ± 0.0067 | 0.806 | 0.9335 | 5 |
| l1 | 8 | Random Forest | 0.9367 ± 0.0058 | 0.820 | 0.9396 | 5 |
| l1 | 8 | XGBoost | 0.9374 ± 0.0049 | 0.817 | 0.9421 | 5 |
| l1 | 8 | MLP (8/16/32) | 0.9336 ± 0.0075 | 0.809 | 0.9365 | 5 |
| l1 | 8 | **angle VQC π/6 (ours, existing)** | 0.9280 ± 0.0063 | 0.796 | – | 5 |
| l1 | 12 | LR | 0.9293 ± 0.0064 | 0.801 | 0.9318 | 5 |
| l1 | 12 | degree-2 LR | 0.9390 ± 0.0067 | 0.828 | 0.9410 | 5 |
| l1 | 12 | Random Forest | 0.9422 ± 0.0055 | 0.836 | 0.9457 | 5 |
| l1 | 12 | XGBoost | 0.9452 ± 0.0050 | 0.838 | 0.9483 | 5 |
| l1 | 12 | MLP (8/16/32) | 0.9416 ± 0.0064 | 0.832 | 0.9442 | 5 |
| l1 | 16 | LR | 0.9320 ± 0.0075 | 0.816 | 0.9355 | 5 |
| l1 | 16 | degree-2 LR | 0.9465 ± 0.0062 | 0.853 | 0.9497 | 5 |
| l1 | 16 | Random Forest | 0.9492 ± 0.0055 | 0.862 | 0.9518 | 5 |
| l1 | 16 | XGBoost | 0.9518 ± 0.0041 | 0.866 | 0.9555 | 5 |
| l1 | 16 | MLP (8/16/32) | 0.9489 ± 0.0061 | 0.857 | 0.9526 | 5 |
| l1 | 16 | **angle VQC π/6 (ours, existing)** | 0.9395 ± 0.0056 | 0.826 | – | 5 |
| l1 | 32 | LR | 0.9451 ± 0.0082 | 0.848 | 0.9475 | 5 |
| l1 | 32 | degree-2 LR | 0.9583 ± 0.0044 | 0.884 | 0.9585 | 5 |
| l1 | 32 | Random Forest | 0.9608 ± 0.0053 | 0.896 | 0.9619 | 5 |
| l1 | 32 | XGBoost | 0.9647 ± 0.0046 | 0.903 | 0.9650 | 5 |
| l1 | 32 | MLP (8/16/32) | 0.9597 ± 0.0052 | 0.885 | 0.9620 | 5 |
| l1 | 32 | **angle VQC π/6 (ours, existing)** | 0.9510 ± 0.0059 | 0.856 | – | 5 |
| l1 | all 249 | LR | 0.9534 ± 0.0071 | 0.873 | 0.9566 | 5 |
| l1 | all 249 | Random Forest | 0.9666 ± 0.0039 | 0.902 | 0.9678 | 5 |
| l1 | all 249 | XGBoost | 0.9731 ± 0.0029 | 0.924 | 0.9745 | 5 |
| l1 | all 249 | MLP (8/16/32) | 0.9631 ± 0.0047 | 0.892 | 0.9660 | 5 |
| l1 | 32 | **angle VQC, trained scales (existing)** | 0.9547 ± 0.0051 | 0.867 | – | 5 |
| cohen_dedup | 8 | LR | 0.9150 ± 0.0076 | 0.771 | 0.9150 | 5 |
| cohen_dedup | 8 | degree-2 LR | 0.9213 ± 0.0073 | 0.789 | 0.9216 | 5 |
| cohen_dedup | 8 | Random Forest | 0.9231 ± 0.0076 | 0.796 | 0.9247 | 5 |
| cohen_dedup | 8 | XGBoost | 0.9232 ± 0.0069 | 0.791 | 0.9273 | 5 |
| cohen_dedup | 8 | MLP (8/16/32) | 0.9222 ± 0.0081 | 0.794 | 0.9237 | 5 |
| cohen_dedup | 12 | LR | 0.9235 ± 0.0060 | 0.789 | 0.9248 | 5 |
| cohen_dedup | 12 | degree-2 LR | 0.9369 ± 0.0044 | 0.822 | 0.9377 | 5 |
| cohen_dedup | 12 | Random Forest | 0.9423 ± 0.0048 | 0.838 | 0.9443 | 5 |
| cohen_dedup | 12 | XGBoost | 0.9435 ± 0.0035 | 0.830 | 0.9468 | 5 |
| cohen_dedup | 12 | MLP (8/16/32) | 0.9406 ± 0.0048 | 0.830 | 0.9414 | 5 |

**Reading.** More features help every model; the tree ensembles gain the most (XGBoost 0.937 → 0.973 from k=8 to all 249 features).
Selection by L1 beats `cohen_dedup` at k=8, and they tie at k=12. At k=8 the quantum-friendly benchmark to beat is **0.937 (XGBoost/RF)**, at k=12 **0.945 (XGBoost)**.

## P2 — the brick-wall QNN rebuilt and verified (reconstruction of `quantum_utils.py`)


`quantum_utils.py` was **not supplied**, so `bw_utils.py` is a **reconstruction** (clearly *not* Antonio's code) from the paper and the notebook's call signatures:

| function | our reconstruction | confidence |
|---|---|---|
| `particle_count(n, k)` | explicit k must be even, 2k ≤ n (k/2 magic blocks of 4 qubits). Default (k=None) guessed as `max(2, 2·(n//8))`: gives k=2 at n=8 (matches the notebook output "features=8, k=2"), k=2 at n=12 | **low** for n=12 |
| `spreading_matrix(n)` | single-particle unitary with |W_ij|² = 1/n: Walsh–Hadamard (n=8), real Hadamard matrix of order 12 (Paley construction) at n=12 | medium (see finding) |
| `UniformSpreading(W, wires)` | exact many-body (Jordan–Wigner, passive free-fermion) action of W on the k-particle space: amplitude of Fock state T = Σ_S ψ_in(S)·det W[T,S] | medium |
| `initial_weights(n, columns, rng)` | N(0, 0.1²) of shape (columns, 3n/2); first n/2 entries per column = RBS angles, last n = Rz phases (as indexed in `make_circuit`) | high (shape), low (distribution) |
| `shift_coefficients(k)` | equidistant-frequency parallel parameter-shift rule, shifts (2m−1)π/(4k); **not used** (we train with exact autograd) | low, unused |

**Finding on the spreading circuit.** The paper (Sec. III.D) says the *fixed-angle brick-wall* (n−1 columns of chain-pair RBS at π/4, boundary pair and Rz = 0) implements the
Walsh–Hadamard transform. Built literally, that circuit is orthogonal but **not** uniform-modulus (max |(|W_ij|² − 1/n)| = 0.35 at n=8, 0.40 at n=12), so it does *not* give the
Lemma-2 occupation k/n. We therefore use the exact Walsh–Hadamard / Hadamard-12 matrix and compile it exactly into n(n−1)/2 nearest-neighbour RBS gates + a ±1 sign layer
(`hw_bw.givens_decompose`; 28 RBS at n=8, 66 at n=12). Worth asking Antonio which circuit the notebook's `UniformSpreading` really uses.
n=12 is not a power of two, so a Walsh–Hadamard transform does not exist there; the order-12 Hadamard matrix is our substitute (the DFT would be the other uniform option).

**Fast simulator** (`bw_torch.py`): |φ0⟩ = U_had|Ψ_in⟩ is data-independent, so it is precomputed once in the k-particle sector (dimension C(n,k): 28–70 at n=8, 66–924 at n=12);
|ψ(x)⟩ = U_cols(w)·D(φ(x))|φ0⟩ with D an elementwise phase and U_cols the RBS/Rz columns acting inside the sector, with the notebook's conventions
(RZ(−φ), RZ(−w), SingleExcitation(−2w), ring pairs on odd columns). Readout m_j = ⟨n_j n_{j+1}⟩ (ring), f = 4m, linear head + sigmoid. complex128 for checks, complex64 for training; gradients by autograd.

**Verification** (`p2_verify.py`; 20 random (x, w) per row, 2 columns; the reference is an independent dense 2ⁿ gate-level simulation, with U_had applied as the exact
free-fermion unitary exp(Σ log W_ij a†_i a_j) by sparse exponentials, and the notebook's own `make_circuit` gate sequence in PennyLane 0.43.3):

| n | k | state vs dense (numpy sector, incl. phase) | torch c128 vs dense (up to phase) | features vs dense | PennyLane (notebook circuit) features | weight leakage | Lemma 2 max dev (after spreading / after data) | grad vs finite diff |
|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 9.1e-16 | 6.8e-16 | 1.0e-15 | 2.8e-16 | 0e+00 | 1.1e-16 / 1.1e-16 | 8.2e-11 |
| 8 | 4 | 1.3e-15 | 8.9e-16 | 5.1e-15 | 4.4e-16 | 0e+00 | 2.2e-16 / 1.7e-16 | 9.3e-11 |
| 12 | 2 | 5.6e-16 | 5.1e-16 | 8.3e-16 | 8.3e-17 | 0e+00 | 1.4e-16 / 1.4e-16 | 1.0e-10 |
| 12 | 4 | 4.1e-16 | 3.5e-16 | 2.1e-15 | 2.8e-16 | 0e+00 | 9.4e-16 / 9.4e-16 | 1.6e-10 |
| 12 | 6 | 3.1e-16 | 3.1e-16 | 4.7e-15 | 8.9e-16 | 0e+00 | 1.4e-15 / 1.4e-15 | 5.7e-10 |

All state errors ≤ 1.3e-15 (required < 1e-10), particle number exact (no weight leakage), Lemma-2 occupations k/n to 1.4e-15, gradients within 2.3e-10 of finite differences (required < 1e-6).

**Toy XOR from the notebook** (`p2_xor_facts.py`, same data generator and loop — batch 8, 100 steps, quantum LR 0.5, head LR 1.0, one column per step — but with the **exact** gradient instead of
the stochastic parallel parameter-shift estimate): default seed (42+n) 100.0% (n=8, 2 columns, k=2) and 100.0% at n=12; averaged over 10 other seeds **52.9%** (range 50–67%);
300 steps: 89.6%. The notebook's saved **72.9%** is not reproduced exactly and cannot be: our weight initialisation, particle-number handling and gradient estimator differ
(the PSR is stochastic and the initial weights come from the unknown `initial_weights`). The accuracy on this tiny task is dominated by initialisation (it is a 96-sample, 100-step problem), so 72.9% sits inside the seed spread of our rerun.

**Circuit facts** (nearest-neighbour hardware; magic preparation depth 4; Hadamard spreading compiled as n(n−1)/2 RBS gives depth ≈ n+... — depth column uses the paper's nearest-neighbour spreading depth n−1):

| qubits | columns | particle number k | magic blocks | sector dim C(n,k) | trainable RBS | trainable Rz | circuit params | depth (NN: magic 4 + spread n−1 + data 1 + 2·cols) | spreading RBS (Givens) |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 2 | 1 | 28 | 8 | 16 | 24 | 16 | 28 |
| 8 | 2 | 4 | 2 | 70 | 8 | 16 | 24 | 16 | 28 |
| 8 | 4 | 2 | 1 | 28 | 16 | 32 | 48 | 20 | 28 |
| 8 | 4 | 4 | 2 | 70 | 16 | 32 | 48 | 20 | 28 |
| 8 | 8 | 2 | 1 | 28 | 32 | 64 | 96 | 28 | 28 |
| 8 | 8 | 4 | 2 | 70 | 32 | 64 | 96 | 28 | 28 |
| 12 | 2 | 2 | 1 | 66 | 12 | 24 | 36 | 20 | 66 |
| 12 | 2 | 4 | 2 | 495 | 12 | 24 | 36 | 20 | 66 |
| 12 | 2 | 6 | 3 | 924 | 12 | 24 | 36 | 20 | 66 |
| 12 | 4 | 2 | 1 | 66 | 24 | 48 | 72 | 24 | 66 |
| 12 | 4 | 4 | 2 | 495 | 24 | 48 | 72 | 24 | 66 |
| 12 | 4 | 6 | 3 | 924 | 24 | 48 | 72 | 24 | 66 |
| 12 | 12 | 2 | 1 | 66 | 72 | 144 | 216 | 40 | 66 |
| 12 | 12 | 4 | 2 | 495 | 72 | 144 | 216 | 40 | 66 |
| 12 | 12 | 6 | 3 | 924 | 72 | 144 | 216 | 40 | 66 |

## P4 — external validation on the PTB Diagnostic ECG Database (interim: brick-wall finals = P3 stage-B/C winners trained on all PTB-XL)

**Extraction.** `ptb_external/extract_ptb_features.py` imports `extract_st_features.process_record` and `extract_tr_features.process_record_tr` (nothing copied). PTB: 12 standard leads in PTB-XL
order from the 15-lead 1000 Hz records, `resample_poly(1,2)` to 500 Hz, first 10 s, mV, exactly the 249 feature columns of `data/st_features_plus_TR.csv.gz`.
Labels from "Reason for admission": MI = *Myocardial infarction*, NORM = *Healthy control*; all other diagnoses dropped.
**Counts:** 549 records → 445 used: **80 healthy-control records (52 patients), 365 MI records (148 patients)**; 101 other-diagnosis records dropped, 3 skipped (< 3 beats detected).
The healthy group is small (52 patients), so every external CI below is wide (±0.03 AUC).

**Sanity check — and a found-and-fixed extractor mismatch.** Re-extracting 50 PTB-XL records from `records500` with the *current* scripts reproduced the T/R/Q columns and beat counts exactly but **not the ST columns**
(142 of 249 columns differ; e.g. `st_curvature` by up to 35). Cause: `extract_st_features.py` was edited on 2026-07-31, after `st_features.csv` (the source of the table's ST columns) was generated on 2026-07-19.
By testing filter variants I recovered the version that built the table: the ST features were computed after a **0.5–40 Hz band-pass**, not the 0.05 Hz `diagnostic_filter` the script now uses. Patching only that filter inside our extractor
reproduces **all 249 columns of all 50 records to 8.5e-14** (`ptb_external/xl_check.json`; original scripts untouched). All primary external numbers below use this recovered version.
Sensitivity: the first external run used the current 0.05 Hz filter; the scores changed by ≤ 0.005 AUC for every model that existed then (kept as `results/ext/*_currentfilter0p05*`), so the conclusions do not depend on it.

**Separate, real finding — population shift (not an extraction artefact).** In PTB-XL z-units (train mean/SD), the 80 healthy PTB controls differ systematically from PTB-XL NORM: mean |shift| over all 249 columns 0.27 SD, 20% of columns shifted > 0.5 SD,
concentrated in the **lateral precordial leads V4–V6** (healthy-control `V4_st_max` +1.0 SD, `V4_st_dev_j60` +0.9 SD, `V5/V6_t_hyperacute` +1.0 SD; PTB MI records: V4–V6 ST amplitudes +0.6 SD). Likely causes: different amplifier/filter chain,
younger healthy volunteers, 1990s acquisition. `results/ext/shift.json` lists the largest shifts per group. This shift, not the model class, probably explains why external AUCs (0.90–0.95) are 0.02–0.03 below the cross-validated ones.

**Protocol.** Each final model is trained on ALL PTB-XL (patient-grouped validation split of PTB-XL only; median/z-score/L1 selector fitted on the PTB-XL train part), hyper-parameters chosen on that validation AUC, and the PTB set is scored **once**.
Brick-wall finals: n=8 (k=4, 4 columns, scale π/4) and n=12 (k=6, 4 columns, scale π/4). Angle VQC: π/6, 1000 steps (k=8; k=32 fixed and trained scales). CIs: 1000 patient-cluster bootstrap resamples (records of one patient resampled together);
"one record per patient" = mean AUC over 200 random draws of one record per patient. Advantage rule on the external set = CI rule alone (AUC difference vs the best classical model *by external AUC* at the same k, 2000 paired cluster-bootstrap resamples).

| model | k | PTB-XL val AUC | external AUC [95% CI] | one-record-per-patient AUC (mean of 200 draws) | sens @ 90% spec [95% CI] |
|---|---|---|---|---|---|
| bw8 | 8 | 0.9278 | 0.9029 [0.864, 0.941] | 0.9141 | 0.789 [0.690, 0.882] |
| lr | 8 | 0.9209 | 0.9123 [0.872, 0.949] | 0.9206 | 0.825 [0.708, 0.889] |
| lr2 | 8 | 0.9306 | 0.9063 [0.865, 0.944] | 0.9173 | 0.805 [0.681, 0.878] |
| mlp | 8 | 0.9335 | 0.9113 [0.871, 0.946] | 0.9237 | 0.814 [0.690, 0.886] |
| rf | 8 | 0.9323 | 0.9049 [0.864, 0.942] | 0.9208 | 0.825 [0.683, 0.886] |
| vqc_pi6 | 8 | 0.9264 | 0.9090 [0.870, 0.946] | 0.9187 | 0.797 [0.696, 0.894] |
| xgb | 8 | 0.9366 | 0.9122 [0.871, 0.948] | 0.9299 | 0.822 [0.707, 0.888] |
| bw12 | 12 | 0.9313 | 0.9102 [0.872, 0.945] | 0.9179 | 0.792 [0.691, 0.878] |
| lr | 12 | 0.9276 | 0.9209 [0.887, 0.953] | 0.9318 | 0.841 [0.765, 0.898] |
| lr2 | 12 | 0.9361 | 0.9121 [0.875, 0.946] | 0.9263 | 0.795 [0.699, 0.884] |
| mlp | 12 | 0.9381 | 0.9207 [0.886, 0.953] | 0.9331 | 0.841 [0.741, 0.897] |
| rf | 12 | 0.9386 | 0.9073 [0.867, 0.943] | 0.9242 | 0.811 [0.732, 0.889] |
| xgb | 12 | 0.9410 | 0.9201 [0.885, 0.953] | 0.9364 | 0.830 [0.748, 0.900] |
| lr | 16 | 0.9298 | 0.9264 [0.894, 0.958] | 0.9383 | 0.863 [0.795, 0.918] |
| lr2 | 16 | 0.9457 | 0.9220 [0.888, 0.955] | 0.9357 | 0.838 [0.750, 0.913] |
| mlp | 16 | 0.9470 | 0.9259 [0.892, 0.958] | 0.9389 | 0.860 [0.799, 0.917] |
| rf | 16 | 0.9469 | 0.9123 [0.874, 0.949] | 0.9279 | 0.836 [0.756, 0.898] |
| xgb | 16 | 0.9520 | 0.9254 [0.890, 0.957] | 0.9419 | 0.855 [0.789, 0.918] |
| lr | 32 | 0.9428 | 0.9395 [0.907, 0.967] | 0.9490 | 0.868 [0.786, 0.925] |
| lr2 | 32 | 0.9578 | 0.9401 [0.906, 0.970] | 0.9434 | 0.874 [0.799, 0.931] |
| mlp | 32 | 0.9594 | 0.9365 [0.902, 0.967] | 0.9484 | 0.877 [0.812, 0.930] |
| rf | 32 | 0.9601 | 0.9319 [0.898, 0.961] | 0.9430 | 0.868 [0.822, 0.930] |
| vqc_pi6 | 32 | 0.9471 | 0.9338 [0.897, 0.964] | 0.9423 | 0.863 [0.782, 0.922] |
| vqc_pi6_trained | 32 | 0.9485 | 0.9247 [0.888, 0.958] | 0.9331 | 0.855 [0.786, 0.918] |
| xgb | 32 | 0.9661 | 0.9463 [0.916, 0.973] | 0.9564 | 0.888 [0.826, 0.939] |

| quantum model | k | best classical at same k | AUC diff [95% CI] | beats every classical | advantage rule (CI only) |
|---|---|---|---|---|---|
| vqc_pi6 | 8 | lr | -0.0032 [-0.0160, +0.0100] | False | False |
| bw8 | 8 | lr | -0.0094 [-0.0265, +0.0080] | False | False |
| bw12 | 12 | lr | -0.0108 [-0.0289, +0.0075] | False | False |
| vqc_pi6 | 32 | xgb | -0.0125 [-0.0264, -0.0006] | False | False |
| vqc_pi6_trained | 32 | xgb | -0.0217 [-0.0362, -0.0090] | False | False |

**Reading.** No quantum model outperforms the classical models on the external set: every quantum-minus-best-classical difference is negative and no CI is above 0.
At k=8 and 12 all models are statistically indistinguishable (differences ≤ 0.01, CIs ±0.02); at k=32 XGBoost is significantly better than the angle VQC (−0.013 [−0.026, −0.001]).
The quantum models are as robust as the classical ones to the dataset shift (external − validation AUC: brick-wall n=8 −0.025, LR k=8 −0.009, XGBoost k=8 −0.024).
Caveat: 52 healthy patients; these CIs cannot detect differences below ~0.02.
