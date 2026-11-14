# Table 2 — Full-data benchmark by feature count (5-fold patient-grouped CV, test AUC mean ± SD)

Built from commit 8a5214d by make_paper.py.

Sources: `results/p1_classical.jsonl` (classical, hyper-parameters by validation AUC), `results/n4_clinical.json` (angle VQC π/6), `results/bw/p3_analysis.json` (brick-wall finals).

| model | k = 8 | k = 12 | k = 16 | k = 32 | all 249 |
|---|---|---|---|---|---|
| LR | 0.922 ± 0.007 | 0.929 ± 0.006 | 0.932 ± 0.008 | 0.945 ± 0.008 | 0.953 ± 0.007 |
| degree-2 LR | 0.931 ± 0.007 | 0.939 ± 0.007 | 0.946 ± 0.006 | 0.958 ± 0.004 | – |
| Random forest | 0.937 ± 0.006 | 0.942 ± 0.005 | 0.949 ± 0.006 | 0.961 ± 0.005 | 0.967 ± 0.004 |
| XGBoost | 0.937 ± 0.005 | 0.945 ± 0.005 | 0.952 ± 0.004 | 0.965 ± 0.005 | 0.973 ± 0.003 |
| MLP | 0.934 ± 0.008 | 0.942 ± 0.006 | 0.949 ± 0.006 | 0.960 ± 0.005 | 0.963 ± 0.005 |
| Angle VQC, s = π/6 | 0.928 ± 0.006 | – | 0.939 ± 0.006 | 0.951 ± 0.006 | – |
| Angle VQC, trained scales | – | – | – | 0.955 ± 0.005 | – |
| Brick-wall QNN (n qubits = k) | 0.920 ± 0.005 | 0.934 ± 0.008 | – | – | – |

No quantum model beats every tuned classical model at any feature count (advantage rule, `BRICKWALL.md`).
