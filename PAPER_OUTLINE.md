# PAPER OUTLINE — Quantum feature maps for MI detection from compact ECG features: an encoding design rule and a pre-registered small-data benchmark

Status: draft outline (branch `smalldata`). Every claim below names the file(s) and the commit that support it. Claims marked *(pending)* depend on runs that are not finished; they must not be written up until the cited result exists.

Framing: (1) compact, interpretable feature selection (extends the poster); (2) an encoding design rule — angle saturation at large scales, fixed by s = π/6, and replicated in a different architecture (the brick-wall); (3) a pre-registered small-data benchmark — quantum models are non-inferior at N = 100 but not superior; (4) hardware readiness, in simulation only.

## 1. Introduction
* Claim: Myocardial infarction (MI) vs normal ECG classification on PTB-XL with 8–32 hand-crafted ST/T/QRS features is a realistic, small-input testbed for near-term quantum classifiers. — `README.md`, `data/st_features_plus_TR.csv.gz` (commit 86f8984).
* Claim: we make no quantum-advantage claim; every model here is classically simulable, and we say where the theory (Scalable_Quantum_ML.pdf, Prop. 8) places hardness. — `BRICKWALL.md` §P7 (58e64ca), `SMALLDATA.md` primary section (6eef855).

## 2. Data and compact feature selection
* Claim: 14,965 ECGs (MI vs NORM), 249 numeric features; patient-grouped, class-stratified 5-fold CV with an inner validation split; all preprocessing fitted on inner-train. — `scale_cv.py`, `RESULTS.md` (86f8984, a4b81a7).
* Claim: L1-path feature selection beats random features by +0.13 AUC at k = 8 and +0.04 at k = 32; a Cohen's-d + de-duplication selector is within 0.005 of L1 at k ≥ 16. — `RESULTS.md` Stage 3, `results/stage3_selectors.jsonl` (16c7a67).
* Claim: the stored ST features were computed after a 0.5–40 Hz band-pass; the recovered extractor reproduces all 249 columns of 50 records to 1e-13. — `ptb_external/xl_check.json`, `BRICKWALL.md` §P4 (34e5631, 727750c).

## 3. Encoding design rule: angle saturation and s = π/6
* Claim: with angle = s·tanh(z) and s = π, ~10% of angles saturate near ±π and re-uploading more features *lowers* AUC (k = 32: 0.791). — `README.md`, `RESULTS.md` Stage 1–2 (a4b81a7, 7891a4c).
* Claim: s = π/6 removes the collapse: AUC rises with k (0.928 → 0.940 → 0.951) and π/6 beats π in 5/5 folds at every k. — `RESULTS.md` Stage 2, `results/stage2_steps1000.jsonl` (7891a4c).
* Claim: shuffling the later features costs ~0 at π/6, while real later features add +0.023 — the gain is information, the π drop is noise damage. — `RESULTS.md` Stage 4 (72b5dcf).
* Claim: at π/6 the VQC is 98.6% quadratic in its angles and sits between linear LR (0.945) and degree-2 LR (0.958). — `RESULTS.md` Stage 4b (86e66a9).
* Claim (replication in a second architecture): in the magic-state brick-wall QNN, small phase scales (π/6–π/4) are best and s = π is worst at every (n, k, columns). — `BRICKWALL.md` §P3, `results/bw/p3_stageA2.jsonl` (97e25f9).

## 4. Brick-wall reimplementation and verification
* Claim: the brick-wall model was reimplemented from the paper (quantum_utils.py was not available) and verified against an independent dense simulator and PennyLane to ≤ 1.3e-15. — `BRICKWALL.md` §P2, `p2_verify.py`, `results/bw/p2_verify.json` (9e24020).
* Claim: the paper's literal fixed-angle "Hadamard brick-wall" is not uniform-modulus; we used the exact Walsh–Hadamard (n = 8) / Hadamard-12 matrices. — `bw_utils.py`, `BRICKWALL.md` §P2 (9e24020).

## 5. Full-data benchmark (secondary context)
* Claim: tuned XGBoost is best at every feature count (0.937 k=8, 0.945 k=12, 0.965 k=32, 0.973 all 249); no quantum model meets the outperformance rule at 8, 12 or 32 features or on the external PTB set. — `BRICKWALL.md` headline + P1/P3/P4, `results/p1_classical.jsonl`, `results/bw/p3_analysis.md`, `results/ext/p4_report.md` (0db679d, 97e25f9, 727750c, e4f7ed6).

## 6. Pre-registered small-data benchmark (main result)
* Claim: the analysis plan was committed before any SD run; compute deviations were logged before any analysis. — `SMALLDATA.md` (f7ba45e plan, 66b5b5f deviations).
* Claim (primary endpoint, N = 100, k = 8): the best quantum model (quantum-kernel SVM) vs the best classical model (random forest): d = −0.0028 [−0.0049, −0.0010]; superiority not met; non-inferiority (margin 0.02) met — "matched classical performance within 0.02 AUC". — `SMALLDATA.md` primary section, `results/sd_analysis_k8_N100.json`, `results/sd_tables_k8_N100.md` (6eef855).
* Claim: Q1–Q3 outperformed kNN and XGBoost at N = 100 (Holm-adjusted), not the best classical models. — same (6eef855).
* Claim: the brick-wall was significantly worse than LR, SVM-RBF, GP, RF and XGBoost at N = 100. — same (6eef855).
* Claim (sensitivity, added after the primary result): per-fold differences are all negative; with "best" chosen by inner-CV AUC the comparison is a tie (Q3 vs SVM-RBF, −0.0006 [−0.0024, +0.0013]); the angle VQC has the same train–test gap as LR with 81 vs 9 parameters. — `SMALLDATA.md` sensitivity section, `results/sd_sensitivity_N100.json` (2af3bff).
* Claim *(pending, Phase 2/4)*: behaviour at N = 25, 50, 200, 500 and at k = 4. — `results/sd_tables_k8.md`, `results/figs/sd_auc_vs_N.png` (to be committed).

## 7. Kernel geometry (exploratory)
* Claim *(pending, Phase 3)*: geometric difference g(K_C ‖ K_Q) of the π/6 angle feature map vs RBF and linear kernels on discovery folds, compared with √N. — `dx_x2_geometry.py`, `results/dx_x2_geometry.json`, `results/dx_registry.csv` (to be committed).

## 8. Hardware readiness (simulation only)
* Claim: the angle VQC (π/6, k = 32) keeps its AUC under 1000 shots (≤ 0.002 loss) and the FakeFez noise model (0.961 → 0.960). — `HARDWARE.md` (c553b25, 3f9ec7f).
* Claim: the n = 8 brick-wall is robust to shots and FakeFez noise (0.915 → 0.911), whereas the n = 12 model needs ~10⁶ shots because its head weights reach ~400. — `BRICKWALL.md` §P6, `results/bw/p6_noise_n8.json`, `results/bw/p6_shots.json` (8c10076).
* Claim: a test-only IBM script is ready and dry-run on FakeFez; no real-device run has been made. — `hw_run_ibm_brickwall.py`, `MORNING_SUMMARY.md` (8c10076, 1ac69dd).

## 9. Discussion and limitations
* No quantum-advantage claim: at full data XGBoost leads; at N = 100 quantum models are non-inferior but not superior to the best classical model.
* All models are classically simulable: the angle kernel is a product of cos² terms, and the brick-wall two-body readout is classically tractable (paper Prop. 8, Sec. VI.B–C).
* The brick-wall is our reimplementation; Antonio's `quantum_utils.py` conventions are unknown (questions listed in `BRICKWALL.md` §P7).
* TabPFN, a strong small-data baseline in the pre-registration, was not run (licence login required).
* No real-device run yet; all hardware statements are simulations (fake backends, noise models).
* Single dataset and hand-crafted features; 5 outer folds carry the test-set variation.

## Tables
1. Feature-selection comparison (L1 / cohen_dedup / random) — `RESULTS.md` Stage 3.
2. Angle scale × k (test AUC) — `RESULTS.md` Stage 1–2.
3. Brick-wall stage A grid (scale × k × columns) — `BRICKWALL.md` §P3.
4. Full-data model × feature count — `BRICKWALL.md` §P1.
5. SD primary endpoint (N = 100) — `SMALLDATA.md`.
6. SD all sizes *(pending)*.
7. Hardware: compile / noise / shots — `HARDWARE.md`, `BRICKWALL.md` §P6.

## Figures
1. AUC vs k for s = π, π/2, π/6 — `results/figs/b_auc_vs_k.png`.
2. Shuffle control — `results/figs/c_shuffle_control.png`.
3. SD mean test AUC vs N with CIs *(pending)* — `results/figs/sd_auc_vs_N.png`.
4. SD best quantum − best classical vs N *(pending)* — `results/figs/sd_diff_vs_N.png`.
5. Shot/noise robustness — `results/figs/hw_shots.png`, `results/figs/hw_noisy.png`.
