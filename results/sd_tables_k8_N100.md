### k = 8: mean test AUC [95% hierarchical-bootstrap CI] (5 folds x up to 40 subsets), generalisation gap in parentheses

| model | N=100 |
|---|---|
| Q1 angle VQC | 0.889 [0.882, 0.894] (+0.07) |
| Q2 VQC trained scales | 0.883 [0.876, 0.889] (+0.08) |
| Q3 quantum-kernel SVM | 0.889 [0.882, 0.895] (+0.07) |
| Q4 brick-wall n=8 | 0.857 [0.846, 0.866] (+0.07) |
| C1 LR | 0.887 [0.881, 0.893] (+0.07) |
| C2 SVM-RBF | 0.889 [0.884, 0.895] (+0.07) |
| C3 GP classifier | 0.891 [0.886, 0.897] (+0.07) |
| C4 kNN | 0.863 [0.859, 0.868] (+0.14) |
| C5 random forest | 0.891 [0.885, 0.898] (+0.10) |
| C6 XGBoost | 0.875 [0.868, 0.881] (+0.12) |

### Best quantum vs best classical per size (best = highest mean AUC at that size)

| N | best quantum | best classical | mean d [95% CI] | wins | Holm p (superiority) | superiority | non-inferior (CI lower > −0.02) |
|---|---|---|---|---|---|---|---|
| 100 | Q3 quantum-kernel SVM | C5 random forest | -0.0028 [-0.0049, -0.0010] | 0.42 | 1.000 | False | True |

### All 24 quantum × classical × size comparisons (Holm over 24; d = quantum − classical)

| N | quantum | classical | mean d [95% CI] | wins | p_sup (Holm) | superior | non-inf. | gap diff [CI] |
|---|---|---|---|---|---|---|---|---|
| 100 | Q1 angle VQC | C1 LR | +0.0020 [-0.0003, +0.0041] | 0.61 | 0.752 | False | True | -0.001 [-0.004, +0.002] |
| 100 | Q1 angle VQC | C2 SVM-RBF | -0.0006 [-0.0027, +0.0013] | 0.45 | 1.000 | False | True | +0.000 [-0.003, +0.003] |
| 100 | Q1 angle VQC | C3 GP classifier | -0.0028 [-0.0051, -0.0009] | 0.38 | 1.000 | False | True | +0.001 [-0.001, +0.004] |
| 100 | Q1 angle VQC | C4 kNN | +0.0251 [+0.0209, +0.0291] | 0.95 | 0.005 | True | True | -0.069 [-0.076, -0.062] |
| 100 | Q1 angle VQC | C5 random forest | -0.0029 [-0.0051, -0.0008] | 0.41 | 1.000 | False | True | -0.030 [-0.034, -0.025] |
| 100 | Q1 angle VQC | C6 XGBoost | +0.0137 [+0.0103, +0.0173] | 0.84 | 0.005 | True | True | -0.054 [-0.061, -0.048] |
| 100 | Q2 VQC trained scales | C1 LR | -0.0038 [-0.0073, -0.0004] | 0.42 | 1.000 | False | True | +0.010 [+0.005, +0.016] |
| 100 | Q2 VQC trained scales | C2 SVM-RBF | -0.0064 [-0.0096, -0.0033] | 0.28 | 1.000 | False | True | +0.011 [+0.005, +0.018] |
| 100 | Q2 VQC trained scales | C3 GP classifier | -0.0086 [-0.0113, -0.0059] | 0.24 | 1.000 | False | True | +0.013 [+0.008, +0.017] |
| 100 | Q2 VQC trained scales | C4 kNN | +0.0194 [+0.0146, +0.0240] | 0.86 | 0.005 | True | True | -0.058 [-0.065, -0.051] |
| 100 | Q2 VQC trained scales | C5 random forest | -0.0087 [-0.0110, -0.0066] | 0.23 | 1.000 | False | True | -0.019 [-0.023, -0.014] |
| 100 | Q2 VQC trained scales | C6 XGBoost | +0.0080 [+0.0057, +0.0103] | 0.74 | 0.005 | True | True | -0.043 [-0.048, -0.038] |
| 100 | Q3 quantum-kernel SVM | C1 LR | +0.0020 [-0.0005, +0.0043] | 0.55 | 0.918 | False | True | -0.001 [-0.004, +0.003] |
| 100 | Q3 quantum-kernel SVM | C2 SVM-RBF | -0.0006 [-0.0024, +0.0013] | 0.45 | 1.000 | False | True | +0.000 [-0.003, +0.003] |
| 100 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0028 [-0.0051, -0.0009] | 0.39 | 1.000 | False | True | +0.002 [-0.002, +0.005] |
| 100 | Q3 quantum-kernel SVM | C4 kNN | +0.0252 [+0.0206, +0.0294] | 0.95 | 0.005 | True | True | -0.069 [-0.077, -0.061] |
| 100 | Q3 quantum-kernel SVM | C5 random forest | -0.0028 [-0.0049, -0.0010] | 0.42 | 1.000 | False | True | -0.029 [-0.035, -0.024] |
| 100 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0138 [+0.0103, +0.0172] | 0.83 | 0.005 | True | True | -0.054 [-0.061, -0.047] |
| 100 | Q4 brick-wall n=8 | C1 LR | -0.0299 [-0.0378, -0.0225] | 0.14 | 1.000 | False | False | +0.001 [-0.004, +0.006] |
| 100 | Q4 brick-wall n=8 | C2 SVM-RBF | -0.0325 [-0.0400, -0.0257] | 0.11 | 1.000 | False | False | +0.002 [-0.003, +0.008] |
| 100 | Q4 brick-wall n=8 | C3 GP classifier | -0.0346 [-0.0426, -0.0276] | 0.09 | 1.000 | False | False | +0.004 [-0.001, +0.008] |
| 100 | Q4 brick-wall n=8 | C4 kNN | -0.0067 [-0.0153, +0.0008] | 0.51 | 1.000 | False | True | -0.067 [-0.074, -0.059] |
| 100 | Q4 brick-wall n=8 | C5 random forest | -0.0347 [-0.0421, -0.0282] | 0.07 | 1.000 | False | False | -0.027 [-0.032, -0.023] |
| 100 | Q4 brick-wall n=8 | C6 XGBoost | -0.0181 [-0.0256, -0.0114] | 0.35 | 1.000 | False | False | -0.052 [-0.058, -0.045] |