# PAPER OUTLINE — Quantum feature maps for MI detection from compact ECG features: an encoding design rule and a pre-registered small-data benchmark

Status: final outline (branch `smalldata`). Every claim below names the file(s) and the commit that support it. Claims marked *(pending)* are not finished and must not be written up until the cited result exists; see the incomplete list at the end.

Framing: (1) compact, interpretable feature selection (extends the poster); (2) an encoding design rule — angle saturation at large scales, fixed by s = π/6, and replicated in a different architecture (the brick-wall); (3) a pre-registered small-data benchmark — quantum models are non-inferior to the best classical model at every size and superior at none; (4) hardware readiness, in simulation only.

## 1. Introduction
* Claim: Myocardial infarction (MI) vs normal ECG classification on PTB-XL with 8–32 hand-crafted ST/T/QRS features is a realistic, small-input testbed for near-term quantum classifiers. — `README.md`, `data/st_features_plus_TR.csv.gz` (commit 86f8984).
* Claim: we make no quantum-advantage claim; every model here is classically simulable, and we say where the theory (Scalable_Quantum_ML.pdf, Prop. 8) places hardness. — `BRICKWALL.md` §P7 (58e64ca), `SMALLDATA.md` primary section (6eef855).

## 2. Data and compact feature selection
* Claim: 14,965 ECGs (MI vs NORM), 249 numeric features; patient-grouped, class-stratified 5-fold CV with an inner validation split; all preprocessing fitted on inner-train. — `scale_cv.py`, `RESULTS.md` (86f8984, a4b81a7).
* Claim: L1-path feature selection beats random features by +0.13 AUC at k = 8 and +0.04 at k = 32; a Cohen's-d + de-duplication selector is within 0.005 of L1 at k ≥ 16. — `RESULTS.md` Stage 3, `results/stage3_selectors.jsonl` (16c7a67).
* Claim: the stored ST features were computed after a 0.5–40 Hz band-pass; the recovered extractor reproduces all 249 columns of 50 records to 1e-13. — `ptb_external/xl_check.json`, `BRICKWALL.md` §P4 (34e5631, 727750c).

## 3. Encoding design rule: angle saturation and s = π/6
* Claim: with angle = s·tanh(z) and s = π, ~10% of angles saturate near ±π and re-uploading more features *lowers* AUC (k = 32: 0.791). — `README.md`, `RESULTS.md` Stage 1–2 (a4b81a7, 7891a4c).
* Claim: s = π/6 removes the collapse: AUC rises with k (0.928 → 0.939 → 0.951) and π/6 beats π in 5/5 folds at every k. — `RESULTS.md` Stage 2, `results/stage2_steps1000.jsonl` (7891a4c).
* Claim: shuffling the later features costs ~0 at π/6, while real later features add +0.023 — the gain is information, the π drop is noise damage. — `RESULTS.md` Stage 4 (72b5dcf).
* Claim: at π/6 the VQC is 98.6% quadratic in its angles and sits between linear LR (0.945) and degree-2 LR (0.958). — `RESULTS.md` Stage 4b (86e66a9).
* Claim (replication in a second architecture): in the magic-state brick-wall QNN, small phase scales (π/6–π/4) are best and s = π is worst at every (n, k, columns). — `BRICKWALL.md` §P3, `results/bw/p3_stageA2.jsonl` (97e25f9).

## 4. Brick-wall reimplementation and verification
* Claim: the brick-wall model was reimplemented from the paper (quantum_utils.py was not available) and verified against an independent dense simulator and PennyLane to ≤ 1.3e-15. — `BRICKWALL.md` §P2, `p2_verify.py`, `results/bw/p2_verify.json` (9e24020).
* Claim: the paper's literal fixed-angle "Hadamard brick-wall" is not uniform-modulus; we used the exact Walsh–Hadamard (n = 8) / Hadamard-12 matrices. — `bw_utils.py`, `BRICKWALL.md` §P2 (9e24020).

## 5. Full-data benchmark (secondary context)
* Claim: tuned XGBoost is best at every feature count (0.937 k=8, 0.945 k=12, 0.965 k=32, 0.973 all 249); no quantum model meets the outperformance rule at 8, 12 or 32 features or on the external PTB set. — `BRICKWALL.md` headline + P1/P3/P4, `results/p1_classical.jsonl`, `results/bw/p3_analysis.md`, `results/ext/p4_report.md` (0db679d, 97e25f9, 727750c, e4f7ed6).

## 6. Pre-registered small-data benchmark (main result)
* Claim: the analysis plan was committed before any SD run; compute deviations (N = 500 and Q4 secondary sizes on 10 of 40 subsets; Q4 at k = 4 dropped) were logged before any analysis. — `SMALLDATA.md` (f7ba45e plan, 66b5b5f deviations).
* Claim (headline): **the quantum models were non-inferior to the best classical model at every training size (N = 25–500, margin 0.02 AUC) and superior at none**; when "best" is chosen by inner-CV AUC instead of test AUC, the best quantum model (Q3 quantum-kernel SVM) is level with SVM-RBF at N = 100 (−0.0006 [−0.0024, +0.0013]). — `SMALLDATA.md` primary, sensitivity and Phase 2 sections; `results/sd_analysis_k8_N100.json`, `results/sd_sensitivity_N100.json`, `results/sd_analysis_k8_main.json`; `results/paper/table3_smalldata.md` (6eef855, 2af3bff, a82370b, 75c799e).
* Claim (primary endpoint, N = 100, k = 8, as recorded): best quantum (Q3) vs best classical (random forest) d = −0.0028 [−0.0049, −0.0010]; non-inferiority met, superiority not. — `SMALLDATA.md` primary section (6eef855).
* Claim: Holm-adjusted wins exist only against weaker classical models — kNN and XGBoost at most sizes, plain LR at N = 25 and N = 500 (reduced sample); never against SVM-RBF, GP or random forest. — `SMALLDATA.md` Phase 2 section, `results/sd_tables_k8_main.md` (a82370b).
* Claim: logistic regression (9 parameters) matches the angle VQC (81 parameters) on test AUC and on the train–test gap at N = 100. — `SMALLDATA.md` sensitivity section (2af3bff).
* Claim: the brick-wall QNN was significantly worse than LR, SVM-RBF, GP, RF and XGBoost at N = 100; it could not be run at other sizes (torch blocked by Windows Application Control). — `SMALLDATA.md` (6eef855, 56ea635, 4c4445c).
* Claim *(pending; incomplete if not finished)*: k = 4 results. — `results/sd_analysis_k4_main.json`.
* Figures 2 and Table 3: `results/paper/fig2_learning_curves.png`, `results/paper/table3_smalldata.md` (75c799e).

## 7. Kernel geometry (exploratory)
* Claim: for the π/6 angle kernel, the geometric difference to a tuned RBF kernel is 1.7–3.3 at N = 100 (√N = 10) and 2.6–11.5 at N = 500 (√N = 22.4), so there is no room for a quantum prediction advantage with this feature map; the kernel is a classically computable product kernel with an effective spectral dimension ≈ 1.8. Exploratory, discovery folds 0–2 only; the search was stopped after this diagnostic. — `DISTINGUISHER.md`, `results/dx_x2_geometry.json`, `results/paper/fig3_geometric_difference.png` (82c52b6, 1a60f8a, 75c799e).

## 7b. Search for a quantum benefit (QB, pre-registered in `QB.md`)
* Claim: a training-free screen of 1,200 feature-map settings (product, re-uploading, IQP/ZZ, clinically paired ZZ, Heisenberg evolution, brick-wall encoding; 5 scales) found no map with both expressivity beyond a tuned RBF kernel (g ≥ √N/2) and adequate label alignment; g crosses √N/2 only where alignment falls below 90 % of RBF (π/3–π/2). This extends the scale rule from "which scale" to "which feature map". — `QB.md` QB1, `results/qb1_screen.jsonl`, `results/paper/table4_feature_maps.md`, `results/paper/fig4_g_all_maps.png` (78984b0, 4073872, bedee3a).
* Claim: the VQC's win over LR is explained by the encoding plus mild non-linearity: classical quadratic models with no more parameters on the same bounded inputs (rank-2 factorization machine 97 parameters, sparse degree-2 LR ≤ 71) match the trained-scale VQC (0.956) and exceed the π/6 VQC (0.952) on discovery folds; degree-2 surrogates explain 97–99 % of the VQC output. — `QB.md` QB3, `results/qb3.jsonl` (d76712b).
* Exploratory (not pre-registered): QB3 extended to folds 3–4 (descriptive only, logged 2026-11-14 before running); finding is consistent across all 5 folds — sparse deg-2 LR (≤65 non-zero params) matches or exceeds the VQC in every fold; degree-2 surrogate R² ≥ 0.985 on fold 0. See `results/qb3_5fold_table.md`, `results/figs/fig5_qb3_coef_heatmap.png` (8c3e6d2+).
* Claim: no robustness, shift, imbalance or subtype setting gave the quantum models a ≥ 0.005 AUC benefit over the best classical model at N = 100 (interim, 18 of 30 units). — `QB.md` QB5 (de0ffb4).
* Claim level: **no benefit found**; confirmation was not run because no candidate qualified. — `QB.md` headline.

## 8. Hardware readiness (simulation only)
* Claim: the angle VQC (π/6, k = 32) keeps its AUC under 1000 shots (≤ 0.002 loss) and the FakeFez noise model (0.961 → 0.960). — `HARDWARE.md` (c553b25, 3f9ec7f).
* Claim: the n = 8 brick-wall is robust to shots and FakeFez noise (0.915 → 0.911), whereas the n = 12 model needs ~10⁶ shots because its head weights reach ~400. — `BRICKWALL.md` §P6, `results/bw/p6_noise_n8.json`, `results/bw/p6_shots.json` (8c10076).
* Claim: a test-only IBM script is ready and dry-run on FakeFez; no real-device run has been made. — `hw_run_ibm_brickwall.py`, `MORNING_SUMMARY.md` (8c10076, 1ac69dd).

## 9. Discussion and limitations
* No quantum-advantage claim: at full data XGBoost leads; in the small-data benchmark the quantum models are non-inferior to the best classical model at every size and superior at none.
* No parameter-efficiency claim: LR with 9 parameters matches the 81-parameter angle VQC on AUC and on the train–test gap.
* All models are classically simulable: the angle kernel is a product of cos² terms, and the brick-wall two-body readout is classically tractable (paper Prop. 8, Sec. VI.B–C); the kernel-geometry diagnostic leaves no room for a prediction advantage with the π/6 feature map.
* The brick-wall is our reimplementation; `quantum_utils.py` conventions are unknown (questions in `BRICKWALL.md` §P7), and it was run in the small-data benchmark only at N = 100.
* TabPFN, a strong small-data baseline in the pre-registration, was not run (licence login required).
* No real-device run yet; all hardware statements are simulations (fake backends, noise models).
* Single dataset and hand-crafted features; 5 outer folds carry the test-set variation; N = 500 is a reduced sample (10 of 40 subsets).
* By construction the small-data study could not have shown an advantage: at k ≤ 8 the angle VQC (Q1) loads each feature once (no re-uploading), and the quantum-kernel SVM (Q3) uses a product kernel that a classical computer evaluates exactly. This motivates the QB study (`QB.md`), which screens entangling and re-uploading feature maps.
* The 0.02 non-inferiority margin is wide compared with the spread of the good models: LR, SVM-RBF, GP, random forest, Q1 and Q3 lie within about 0.005–0.006 AUC of each other at N = 100–500 (0.007 at N = 50; 0.014 at N = 25, where LR is lowest), so "non-inferior" is a weak statement; an exploratory ±0.005 equivalence check is reported in `SMALLDATA.md`.

## Tables
1. Feature-selection comparison (L1 / cohen_dedup / random) — `RESULTS.md` Stage 3.
2. Encoding study (angle scale; brick-wall replication) — `results/paper/table1_encoding.md`.
3. Brick-wall stage A grid (scale × k × columns) — `BRICKWALL.md` §P3.
4. Full-data model × feature count — `results/paper/table2_benchmark.md`.
5. SD primary endpoint (N = 100) — `SMALLDATA.md`.
6. Small-data differences with CIs by N — `results/paper/table3_smalldata.md`.
7. Hardware: compile / noise / shots — `HARDWARE.md`, `BRICKWALL.md` §P6.

## Figures
1. Scale sweep and AUC vs k — `results/paper/fig1_scale_sweep.png`.
2. Shuffle control — `results/figs/c_shuffle_control.png`.
3. Learning curves and best-vs-best difference — `results/paper/fig2_learning_curves.png`.
4. Geometric difference vs √N — `results/paper/fig3_geometric_difference.png`.
5. Shot/noise robustness — `results/figs/hw_shots.png`, `results/figs/hw_noisy.png`.

## Incomplete (as of this outline)
* QB4 training (feature placement), QB5 folds 1–2 (12 of 30 units) and the QB5 full-data part.
* k = 4 secondary endpoint of the small-data study (running when RAM allows).
* Brick-wall (Q4) at N = 25, 50, 200, 500 — torch blocked by Windows Application Control.
* EXTRA 6 (shot-robust n = 12 / n = 16 heads) and the n = 16 external run — same reason.
* C7 TabPFN — licence login required.
* Real-device (IBM) test run — not done (user's run).
