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
