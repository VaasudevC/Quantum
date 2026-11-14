### k = 4: mean test AUC [95% hierarchical-bootstrap CI] (5 folds x up to 40 subsets), generalisation gap in parentheses

| model | N=100 |
|---|---|
| Q1 angle VQC | 0.874 [0.868, 0.880] (+0.06) |
| Q2 VQC trained scales | 0.869 [0.864, 0.875] (+0.07) |
| Q3 quantum-kernel SVM | 0.873 [0.866, 0.880] (+0.06) |
| C1 LR | 0.874 [0.868, 0.881] (+0.05) |
| C2 SVM-RBF | 0.873 [0.865, 0.880] (+0.06) |
| C3 GP classifier | 0.876 [0.870, 0.883] (+0.05) |
| C4 kNN | 0.852 [0.845, 0.859] (+0.15) |
| C5 random forest | 0.871 [0.864, 0.877] (+0.10) |
| C6 XGBoost | 0.860 [0.853, 0.866] (+0.12) |

### Best quantum vs best classical per size (best = highest mean AUC at that size)

| N | best quantum | best classical | mean d [95% CI] | wins | Holm p (superiority) | superiority | non-inferior (CI lower > −0.02) |
|---|---|---|---|---|---|---|---|
| 100 | Q1 angle VQC | C3 GP classifier | -0.0022 [-0.0038, -0.0007] | 0.43 | 1.000 | False | True |

### All 18 quantum × classical × size comparisons (Holm over 18; d = quantum − classical)

| N | quantum | classical | mean d [95% CI] | wins | p_sup (Holm) | superior | non-inf. | gap diff [CI] |
|---|---|---|---|---|---|---|---|---|
| 100 | Q1 angle VQC | C1 LR | -0.0000 [-0.0018, +0.0020] | 0.50 | 1.000 | False | True | +0.002 [-0.001, +0.004] |
| 100 | Q1 angle VQC | C2 SVM-RBF | +0.0015 [-0.0015, +0.0045] | 0.51 | 1.000 | False | True | -0.004 [-0.008, -0.000] |
| 100 | Q1 angle VQC | C3 GP classifier | -0.0022 [-0.0038, -0.0007] | 0.43 | 1.000 | False | True | +0.001 [-0.001, +0.003] |
| 100 | Q1 angle VQC | C4 kNN | +0.0224 [+0.0187, +0.0264] | 0.93 | 0.004 | True | True | -0.092 [-0.100, -0.085] |
| 100 | Q1 angle VQC | C5 random forest | +0.0037 [+0.0020, +0.0057] | 0.64 | 0.004 | True | True | -0.046 [-0.051, -0.042] |
| 100 | Q1 angle VQC | C6 XGBoost | +0.0145 [+0.0121, +0.0169] | 0.88 | 0.004 | True | True | -0.066 [-0.072, -0.060] |
| 100 | Q2 VQC trained scales | C1 LR | -0.0049 [-0.0080, -0.0016] | 0.38 | 1.000 | False | True | +0.011 [+0.006, +0.017] |
| 100 | Q2 VQC trained scales | C2 SVM-RBF | -0.0034 [-0.0073, +0.0005] | 0.39 | 1.000 | False | True | +0.006 [-0.001, +0.013] |
| 100 | Q2 VQC trained scales | C3 GP classifier | -0.0071 [-0.0099, -0.0045] | 0.32 | 1.000 | False | True | +0.011 [+0.007, +0.016] |
| 100 | Q2 VQC trained scales | C4 kNN | +0.0175 [+0.0137, +0.0211] | 0.88 | 0.004 | True | True | -0.082 [-0.089, -0.075] |
| 100 | Q2 VQC trained scales | C5 random forest | -0.0012 [-0.0037, +0.0011] | 0.53 | 1.000 | False | True | -0.036 [-0.040, -0.032] |
| 100 | Q2 VQC trained scales | C6 XGBoost | +0.0095 [+0.0071, +0.0120] | 0.77 | 0.004 | True | True | -0.056 [-0.061, -0.051] |
| 100 | Q3 quantum-kernel SVM | C1 LR | -0.0011 [-0.0037, +0.0014] | 0.48 | 1.000 | False | True | +0.001 [-0.002, +0.005] |
| 100 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0004 [-0.0024, +0.0033] | 0.53 | 1.000 | False | True | -0.004 [-0.009, -0.001] |
| 100 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0033 [-0.0056, -0.0011] | 0.36 | 1.000 | False | True | +0.001 [-0.002, +0.003] |
| 100 | Q3 quantum-kernel SVM | C4 kNN | +0.0213 [+0.0170, +0.0262] | 0.93 | 0.004 | True | True | -0.092 [-0.102, -0.084] |
| 100 | Q3 quantum-kernel SVM | C5 random forest | +0.0026 [+0.0005, +0.0048] | 0.61 | 0.086 | False | True | -0.047 [-0.052, -0.041] |
| 100 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0134 [+0.0106, +0.0165] | 0.86 | 0.004 | True | True | -0.066 [-0.073, -0.060] |