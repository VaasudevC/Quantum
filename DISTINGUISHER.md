# DISTINGUISHER — where, if anywhere, does a quantum model beat the classical models?

Branch `smalldata`. Written and committed BEFORE any DX experiment was run. Runs after SD5 (see `SMALLDATA.md`). Every setting tried (including failures) is logged in `results/dx_registry.csv`.

## Anti-cherry-picking design (fixed now)

* **DISCOVERY** = outer folds 0, 1, 2. All exploration, tuning and choosing happen here only.
* **CONFIRMATION** = outer folds 3, 4 + the external PTB set. Untouched until the end.
* Keep a running list of EVERY setting tried (including failures) in `results/dx_registry.csv`.
* At most **3 candidate settings** (the strongest discovery results with the quantum model ahead of the best classical model) go to confirmation. Each gets ONE paired test there
  (same method as SD: hierarchical bootstrap CI of the AUC difference vs the best classical model on discovery, Holm across the ≤ 3 candidates). Claim "quantum outperforms" only if confirmation passes.
  Discovery-only wins are reported as "hypotheses for future work", never as results.
* Models in every setting: the quantum models Q1–Q4 from SD (whichever are applicable) vs the classical models C1–C7 (tuned the same way). Use N ∈ {100, full} unless stated.

## Settings to explore (fast ones first, each logged in the registry)

* **X1 Very small feature sets:** k ∈ {2, 3, 4, 6} (L1 and cohen_dedup), plus single-region sets (limb leads only, precordial leads only, T-wave features only, ST features only). Low dimension is where quantum kernels and small circuits are least disadvantaged.
* **X2 Kernel geometry diagnostic** (Huang et al. 2021, "Power of data in quantum machine learning"): for the angle-encoding and brick-wall feature maps at several scales, compute the geometric difference
  g(K_C ‖ K_Q) = sqrt(‖ sqrt(K_Q) K_C⁻¹ sqrt(K_Q) ‖_∞) between a tuned RBF kernel K_C and the quantum kernel K_Q on 500–1000 discovery ECGs (regularise K_C⁻¹). Small g means no quantum-kernel advantage is possible for ANY labelling;
  large g means one is possible. Use g to rank settings and to explain the results. Also compute the projected quantum kernel (local 1-RDM features, same paper).
* **X3 Harder clinical tasks** (labels from `../PTB_XL/ptbxl_database.csv` scp_codes, via the existing label pipeline where possible): inferior MI vs NORM, AMI vs IMI, MI vs all non-MI ECGs, early/acute MI (`infarction_stadium1`) vs NORM.
* **X4 Robustness:** training-label noise (flip 10%, 20%, 30% of training labels; test on clean labels); input noise on test features (Gaussian, 0.1–0.5 SD); low-quality recordings (metadata flags `baseline_drift` / `static_noise` / `burst_noise` / `electrodes_problems` non-empty) as a separate test subset.
* **X5 Distribution shift:** (a) train PTB-XL → test PTB (external drop); (b) train on some recording sites / devices and test on others; (c) train on older recordings, test on newer (`recording_date`). Compare the performance DROP, not only the level.
* **X6 Class imbalance:** subsample MI to 5% and 10% prevalence in training (test unchanged); report AUC and sensitivity at 90% specificity.
* **X7 Parameter-matched comparison:** classical models restricted to the quantum model's parameter count (81 for the angle VQC at 8 qubits; 57 for the brick-wall n=8 head+circuit): MLP with that many weights, LR (k+1), small trees. "Per-parameter efficiency" is a legitimate, separate claim.
* **X8 Combinations** of the most promising of X1–X7 (e.g. N=50 + k=4 + label noise), only if they were individually promising on discovery.

Then: confirmation of the ≤ 3 candidates, headline (what was tried, what passed confirmation, g-diagnostic summary), 10-line addition to `MORNING_SUMMARY.md`, commit.

## X2 results — kernel geometry (EXPLORATORY; discovery folds 0–2 only; run 2026-11-13; no other DX setting was run)

`dx_x2_geometry.py` (tested end to end; 4 s), `results/dx_x2_geometry.json`, 48 rows (2 kernels × 4 λ × 2 N × 3 folds) in `results/dx_registry.csv`. Quantum kernel: the π/6 angle-encoding fidelity kernel used by Q3,
K_Q(x,x') = Π_i cos²((a_i − a_i')/2), a = (π/6)·tanh(z), on the first 8 L1 features — **a classically computable product kernel** (O(k) per pair). Classical kernels: RBF with γ tuned by 5-fold CV on the sample, and the linear kernel rescaled to Tr K = N.
g(K_C ‖ K_Q) = sqrt(‖√K_Q (K_C + λI)⁻¹ √K_Q‖_∞) (Huang et al. 2021); N ECGs sampled (class-stratified) from the outer-train part of each discovery fold.

| N | √N | fold | CV AUC: quantum π/6 / RBF / linear | g(RBF‖Q) λ=1e-4 / 1e-3 / 1e-2 / 1e-1 | g(linear‖Q) λ=1e-4 / 1e-2 / 1e-1 | eff. dim of K_Q |
|---|---|---|---|---|---|---|
| 100 | 10.0 | 0 | 0.946 / 0.950 / 0.941 | 4.3 / 2.9 / 2.1 / 1.8 | 840 / 84 / 27 | 1.77 |
| 100 | 10.0 | 1 | 0.937 / 0.929 / 0.924 | 4.8 / 3.3 / 2.0 / 1.7 | 803 / 80 / 25 | 1.86 |
| 100 | 10.0 | 2 | 0.889 / 0.895 / 0.895 | 2.8 / 2.6 / 2.5 / 2.3 | 828 / 83 / 26 | 1.81 |
| 500 | 22.4 | 0 | 0.924 / 0.922 / 0.921 | 21.2 / 11.5 / 5.5 / 2.6 | 1899 / 190 / 60 | 1.80 |
| 500 | 22.4 | 1 | 0.942 / 0.944 / 0.941 | 24.0 / 11.5 / 5.4 / 2.7 | 1885 / 188 / 60 | 1.90 |
| 500 | 22.4 | 2 | 0.945 / 0.948 / 0.940 | 5.0 / 4.1 / 3.6 / 3.3 | 1875 / 187 / 59 | 1.77 |

**Reading.** Against the tuned **RBF** kernel, g is **1.7–3.3 at N = 100 (√N = 10)** and **2.6–11.5 at N = 500 (√N = 22.4)** for λ ≥ 1e-3, i.e. well below √N: by the Huang et al. criterion a classical RBF model can match the prediction error of this quantum kernel for *any* labelling of these inputs — **there is no room for a quantum prediction advantage with this feature map**.
Only with an almost unregularised K_RBF (λ = 1e-4) does g reach ~21–24 ≈ √N at N = 500 in folds 0–1 (5.0 in fold 2); this comes from the near-singular RBF Gram matrix, not from the quantum kernel, and the CV AUCs show no gap (quantum 0.889–0.946 vs RBF 0.895–0.950).
Against the **linear** kernel g is large (26–1900) simply because the linear Gram matrix has rank 8 and cannot represent the quantum kernel's functions; since the RBF kernel already closes that gap, this does not open room for an advantage.
The quantum kernel itself is effectively low-dimensional (participation ratio of its spectrum ≈ 1.8): at the π/6 scale all angles stay small, the kernel stays close to a constant plus low-order terms, which is consistent with the earlier finding that the π/6 VQC is ~99% quadratic in its angles (`RESULTS.md` Stage 4b).
This diagnostic is exploratory (discovery folds only), makes no confirmation claim, and nothing from it goes to the confirmation stage.
