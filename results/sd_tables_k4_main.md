### k = 4: mean test AUC [95% hierarchical-bootstrap CI] (5 folds x up to 40 subsets), generalisation gap in parentheses

| model | N=25 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|
| Q1 angle VQC | 0.804 [0.792, 0.814] (+0.18) | 0.850 [0.844, 0.856] (+0.11) | 0.874 [0.868, 0.880] (+0.06) | 0.890 [0.886, 0.894] (+0.03) | 0.903 [0.895, 0.910] (+0.01) |
| Q2 VQC trained scales | 0.799 [0.788, 0.808] (+0.19) | 0.842 [0.834, 0.849] (+0.12) | 0.869 [0.864, 0.875] (+0.07) | 0.886 [0.883, 0.890] (+0.04) | 0.901 [0.893, 0.908] (+0.02) |
| Q3 quantum-kernel SVM | 0.810 [0.798, 0.820] (+0.17) | 0.850 [0.843, 0.856] (+0.11) | 0.873 [0.866, 0.880] (+0.06) | 0.888 [0.883, 0.892] (+0.03) | 0.902 [0.894, 0.910] (+0.01) |
| C1 LR | 0.807 [0.797, 0.815] (+0.17) | 0.850 [0.844, 0.856] (+0.11) | 0.874 [0.868, 0.881] (+0.05) | 0.889 [0.885, 0.893] (+0.03) | 0.901 [0.893, 0.909] (+0.01) |
| C2 SVM-RBF | 0.810 [0.802, 0.818] (+0.17) | 0.849 [0.842, 0.856] (+0.11) | 0.873 [0.865, 0.880] (+0.06) | 0.887 [0.883, 0.892] (+0.03) | 0.903 [0.894, 0.911] (+0.01) |
| C3 GP classifier | 0.814 [0.806, 0.821] (+0.17) | 0.853 [0.845, 0.860] (+0.11) | 0.876 [0.870, 0.883] (+0.05) | 0.891 [0.887, 0.895] (+0.03) | 0.903 [0.896, 0.911] (+0.01) |
| C4 kNN | 0.778 [0.769, 0.786] (+0.22) | 0.824 [0.817, 0.832] (+0.18) | 0.852 [0.845, 0.859] (+0.15) | 0.870 [0.865, 0.875] (+0.13) | 0.886 [0.879, 0.894] (+0.11) |
| C5 random forest | 0.803 [0.793, 0.813] (+0.19) | 0.847 [0.840, 0.854] (+0.14) | 0.871 [0.864, 0.877] (+0.10) | 0.886 [0.882, 0.890] (+0.08) | 0.902 [0.893, 0.910] (+0.05) |
| C6 XGBoost | 0.775 [0.763, 0.786] (+0.22) | 0.830 [0.823, 0.837] (+0.16) | 0.860 [0.853, 0.866] (+0.12) | 0.882 [0.878, 0.885] (+0.08) | 0.901 [0.892, 0.909] (+0.05) |

### Best quantum vs best classical per size (best = highest mean AUC at that size)

| N | best quantum | best classical | mean d [95% CI] | wins | Holm p (superiority) | superiority | non-inferior (CI lower > −0.02) |
|---|---|---|---|---|---|---|---|
| 25 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0033 [-0.0104, +0.0024] | 0.45 | 1.000 | False | True |
| 50 | Q1 angle VQC | C3 GP classifier | -0.0026 [-0.0058, +0.0004] | 0.44 | 1.000 | False | True |
| 100 | Q1 angle VQC | C3 GP classifier | -0.0022 [-0.0038, -0.0007] | 0.43 | 1.000 | False | True |
| 200 | Q1 angle VQC | C3 GP classifier | -0.0011 [-0.0025, +0.0000] | 0.43 | 1.000 | False | True |
| 500 | Q1 angle VQC | C3 GP classifier | -0.0007 [-0.0018, +0.0004] | 0.46 | 1.000 | False | True |

### All 90 quantum × classical × size comparisons (Holm over 90; d = quantum − classical)

| N | quantum | classical | mean d [95% CI] | wins | p_sup (Holm) | superior | non-inf. | gap diff [CI] |
|---|---|---|---|---|---|---|---|---|
| 25 | Q1 angle VQC | C1 LR | -0.0026 [-0.0084, +0.0023] | 0.47 | 1.000 | False | True | +0.003 [-0.002, +0.009] |
| 25 | Q1 angle VQC | C2 SVM-RBF | -0.0058 [-0.0142, +0.0021] | 0.42 | 1.000 | False | True | +0.004 [-0.003, +0.012] |
| 25 | Q1 angle VQC | C3 GP classifier | -0.0096 [-0.0175, -0.0024] | 0.38 | 1.000 | False | True | +0.008 [+0.001, +0.015] |
| 25 | Q1 angle VQC | C4 kNN | +0.0260 [+0.0185, +0.0330] | 0.78 | 0.018 | True | True | -0.044 [-0.052, -0.036] |
| 25 | Q1 angle VQC | C5 random forest | +0.0008 [-0.0040, +0.0053] | 0.58 | 1.000 | False | True | -0.016 [-0.021, -0.010] |
| 25 | Q1 angle VQC | C6 XGBoost | +0.0289 [+0.0228, +0.0345] | 0.83 | 0.018 | True | True | -0.041 [-0.047, -0.034] |
| 25 | Q2 VQC trained scales | C1 LR | -0.0079 [-0.0124, -0.0035] | 0.38 | 1.000 | False | True | +0.016 [+0.009, +0.023] |
| 25 | Q2 VQC trained scales | C2 SVM-RBF | -0.0111 [-0.0174, -0.0052] | 0.39 | 1.000 | False | True | +0.017 [+0.009, +0.026] |
| 25 | Q2 VQC trained scales | C3 GP classifier | -0.0149 [-0.0206, -0.0098] | 0.31 | 1.000 | False | False | +0.021 [+0.014, +0.029] |
| 25 | Q2 VQC trained scales | C4 kNN | +0.0207 [+0.0154, +0.0263] | 0.73 | 0.018 | True | True | -0.032 [-0.038, -0.025] |
| 25 | Q2 VQC trained scales | C5 random forest | -0.0045 [-0.0107, +0.0007] | 0.47 | 1.000 | False | True | -0.003 [-0.010, +0.005] |
| 25 | Q2 VQC trained scales | C6 XGBoost | +0.0236 [+0.0159, +0.0306] | 0.76 | 0.018 | True | True | -0.028 [-0.036, -0.020] |
| 25 | Q3 quantum-kernel SVM | C1 LR | +0.0037 [-0.0015, +0.0080] | 0.56 | 1.000 | False | True | -0.005 [-0.009, +0.001] |
| 25 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0005 [-0.0063, +0.0064] | 0.49 | 1.000 | False | True | -0.003 [-0.009, +0.003] |
| 25 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0033 [-0.0104, +0.0024] | 0.45 | 1.000 | False | True | +0.001 [-0.005, +0.008] |
| 25 | Q3 quantum-kernel SVM | C4 kNN | +0.0323 [+0.0255, +0.0392] | 0.83 | 0.018 | True | True | -0.052 [-0.059, -0.044] |
| 25 | Q3 quantum-kernel SVM | C5 random forest | +0.0071 [+0.0027, +0.0117] | 0.62 | 0.037 | True | True | -0.023 [-0.028, -0.018] |
| 25 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0352 [+0.0290, +0.0419] | 0.91 | 0.018 | True | True | -0.048 [-0.055, -0.040] |
| 50 | Q1 angle VQC | C1 LR | +0.0003 [-0.0022, +0.0029] | 0.51 | 1.000 | False | True | -0.001 [-0.004, +0.003] |
| 50 | Q1 angle VQC | C2 SVM-RBF | +0.0008 [-0.0022, +0.0039] | 0.51 | 1.000 | False | True | -0.003 [-0.007, +0.001] |
| 50 | Q1 angle VQC | C3 GP classifier | -0.0026 [-0.0058, +0.0004] | 0.44 | 1.000 | False | True | +0.001 [-0.002, +0.005] |
| 50 | Q1 angle VQC | C4 kNN | +0.0259 [+0.0214, +0.0314] | 0.88 | 0.018 | True | True | -0.069 [-0.078, -0.061] |
| 50 | Q1 angle VQC | C5 random forest | +0.0031 [+0.0005, +0.0057] | 0.67 | 0.496 | False | True | -0.032 [-0.037, -0.027] |
| 50 | Q1 angle VQC | C6 XGBoost | +0.0200 [+0.0161, +0.0243] | 0.88 | 0.018 | True | True | -0.056 [-0.065, -0.048] |
| 50 | Q2 VQC trained scales | C1 LR | -0.0079 [-0.0126, -0.0037] | 0.37 | 1.000 | False | True | +0.016 [+0.009, +0.023] |
| 50 | Q2 VQC trained scales | C2 SVM-RBF | -0.0074 [-0.0127, -0.0029] | 0.39 | 1.000 | False | True | +0.013 [+0.006, +0.021] |
| 50 | Q2 VQC trained scales | C3 GP classifier | -0.0109 [-0.0161, -0.0058] | 0.33 | 1.000 | False | True | +0.018 [+0.009, +0.026] |
| 50 | Q2 VQC trained scales | C4 kNN | +0.0177 [+0.0124, +0.0234] | 0.77 | 0.018 | True | True | -0.053 [-0.061, -0.045] |
| 50 | Q2 VQC trained scales | C5 random forest | -0.0052 [-0.0094, -0.0014] | 0.49 | 1.000 | False | True | -0.016 [-0.022, -0.010] |
| 50 | Q2 VQC trained scales | C6 XGBoost | +0.0118 [+0.0079, +0.0157] | 0.76 | 0.018 | True | True | -0.040 [-0.047, -0.034] |
| 50 | Q3 quantum-kernel SVM | C1 LR | -0.0003 [-0.0033, +0.0026] | 0.46 | 1.000 | False | True | -0.002 [-0.005, +0.002] |
| 50 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0002 [-0.0023, +0.0028] | 0.49 | 1.000 | False | True | -0.004 [-0.008, -0.000] |
| 50 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0033 [-0.0062, -0.0008] | 0.38 | 1.000 | False | True | +0.000 [-0.004, +0.004] |
| 50 | Q3 quantum-kernel SVM | C4 kNN | +0.0253 [+0.0205, +0.0307] | 0.86 | 0.018 | True | True | -0.070 [-0.079, -0.062] |
| 50 | Q3 quantum-kernel SVM | C5 random forest | +0.0024 [+0.0001, +0.0050] | 0.57 | 1.000 | False | True | -0.034 [-0.039, -0.029] |
| 50 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0194 [+0.0149, +0.0238] | 0.83 | 0.018 | True | True | -0.058 [-0.066, -0.050] |
| 100 | Q1 angle VQC | C1 LR | -0.0000 [-0.0018, +0.0020] | 0.50 | 1.000 | False | True | +0.002 [-0.001, +0.004] |
| 100 | Q1 angle VQC | C2 SVM-RBF | +0.0015 [-0.0015, +0.0045] | 0.51 | 1.000 | False | True | -0.004 [-0.008, -0.000] |
| 100 | Q1 angle VQC | C3 GP classifier | -0.0022 [-0.0038, -0.0007] | 0.43 | 1.000 | False | True | +0.001 [-0.001, +0.003] |
| 100 | Q1 angle VQC | C4 kNN | +0.0224 [+0.0187, +0.0264] | 0.93 | 0.018 | True | True | -0.092 [-0.100, -0.085] |
| 100 | Q1 angle VQC | C5 random forest | +0.0037 [+0.0020, +0.0057] | 0.64 | 0.018 | True | True | -0.046 [-0.051, -0.042] |
| 100 | Q1 angle VQC | C6 XGBoost | +0.0145 [+0.0121, +0.0169] | 0.88 | 0.018 | True | True | -0.066 [-0.072, -0.060] |
| 100 | Q2 VQC trained scales | C1 LR | -0.0049 [-0.0080, -0.0016] | 0.38 | 1.000 | False | True | +0.011 [+0.006, +0.017] |
| 100 | Q2 VQC trained scales | C2 SVM-RBF | -0.0034 [-0.0073, +0.0005] | 0.39 | 1.000 | False | True | +0.006 [-0.001, +0.013] |
| 100 | Q2 VQC trained scales | C3 GP classifier | -0.0071 [-0.0099, -0.0045] | 0.32 | 1.000 | False | True | +0.011 [+0.007, +0.016] |
| 100 | Q2 VQC trained scales | C4 kNN | +0.0175 [+0.0137, +0.0211] | 0.88 | 0.018 | True | True | -0.082 [-0.089, -0.075] |
| 100 | Q2 VQC trained scales | C5 random forest | -0.0012 [-0.0037, +0.0011] | 0.53 | 1.000 | False | True | -0.036 [-0.040, -0.032] |
| 100 | Q2 VQC trained scales | C6 XGBoost | +0.0095 [+0.0071, +0.0120] | 0.77 | 0.018 | True | True | -0.056 [-0.061, -0.051] |
| 100 | Q3 quantum-kernel SVM | C1 LR | -0.0011 [-0.0037, +0.0014] | 0.48 | 1.000 | False | True | +0.001 [-0.002, +0.005] |
| 100 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0004 [-0.0024, +0.0033] | 0.53 | 1.000 | False | True | -0.004 [-0.009, -0.001] |
| 100 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0033 [-0.0056, -0.0011] | 0.36 | 1.000 | False | True | +0.001 [-0.002, +0.003] |
| 100 | Q3 quantum-kernel SVM | C4 kNN | +0.0213 [+0.0170, +0.0262] | 0.93 | 0.018 | True | True | -0.092 [-0.102, -0.084] |
| 100 | Q3 quantum-kernel SVM | C5 random forest | +0.0026 [+0.0005, +0.0048] | 0.61 | 0.468 | False | True | -0.047 [-0.052, -0.041] |
| 100 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0134 [+0.0106, +0.0165] | 0.86 | 0.018 | True | True | -0.066 [-0.073, -0.060] |
| 200 | Q1 angle VQC | C1 LR | +0.0010 [-0.0005, +0.0028] | 0.58 | 1.000 | False | True | +0.003 [+0.001, +0.005] |
| 200 | Q1 angle VQC | C2 SVM-RBF | +0.0023 [+0.0002, +0.0045] | 0.57 | 0.843 | False | True | -0.004 [-0.008, -0.001] |
| 200 | Q1 angle VQC | C3 GP classifier | -0.0011 [-0.0025, +0.0000] | 0.43 | 1.000 | False | True | +0.000 [-0.001, +0.002] |
| 200 | Q1 angle VQC | C4 kNN | +0.0195 [+0.0167, +0.0222] | 0.97 | 0.018 | True | True | -0.099 [-0.105, -0.095] |
| 200 | Q1 angle VQC | C5 random forest | +0.0036 [+0.0024, +0.0048] | 0.70 | 0.018 | True | True | -0.045 [-0.051, -0.041] |
| 200 | Q1 angle VQC | C6 XGBoost | +0.0080 [+0.0064, +0.0096] | 0.84 | 0.018 | True | True | -0.054 [-0.058, -0.051] |
| 200 | Q2 VQC trained scales | C1 LR | -0.0023 [-0.0042, -0.0002] | 0.40 | 1.000 | False | True | +0.009 [+0.005, +0.013] |
| 200 | Q2 VQC trained scales | C2 SVM-RBF | -0.0010 [-0.0035, +0.0013] | 0.40 | 1.000 | False | True | +0.002 [-0.002, +0.006] |
| 200 | Q2 VQC trained scales | C3 GP classifier | -0.0044 [-0.0062, -0.0029] | 0.24 | 1.000 | False | True | +0.006 [+0.003, +0.010] |
| 200 | Q2 VQC trained scales | C4 kNN | +0.0162 [+0.0133, +0.0186] | 0.95 | 0.018 | True | True | -0.094 [-0.099, -0.089] |
| 200 | Q2 VQC trained scales | C5 random forest | +0.0003 [-0.0011, +0.0017] | 0.51 | 1.000 | False | True | -0.039 [-0.046, -0.034] |
| 200 | Q2 VQC trained scales | C6 XGBoost | +0.0047 [+0.0032, +0.0063] | 0.71 | 0.018 | True | True | -0.048 [-0.052, -0.045] |
| 200 | Q3 quantum-kernel SVM | C1 LR | -0.0011 [-0.0025, +0.0004] | 0.48 | 1.000 | False | True | +0.004 [+0.002, +0.006] |
| 200 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0002 [-0.0013, +0.0017] | 0.49 | 1.000 | False | True | -0.003 [-0.005, -0.001] |
| 200 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0032 [-0.0043, -0.0020] | 0.34 | 1.000 | False | True | +0.001 [-0.000, +0.003] |
| 200 | Q3 quantum-kernel SVM | C4 kNN | +0.0174 [+0.0154, +0.0194] | 0.96 | 0.018 | True | True | -0.098 [-0.103, -0.093] |
| 200 | Q3 quantum-kernel SVM | C5 random forest | +0.0015 [-0.0000, +0.0030] | 0.60 | 1.000 | False | True | -0.044 [-0.050, -0.038] |
| 200 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0059 [+0.0043, +0.0076] | 0.77 | 0.018 | True | True | -0.053 [-0.058, -0.049] |
| 500 | Q1 angle VQC | C1 LR | +0.0018 [+0.0001, +0.0035] | 0.66 | 0.968 | False | True | +0.002 [-0.000, +0.004] |
| 500 | Q1 angle VQC | C2 SVM-RBF | -0.0000 [-0.0015, +0.0017] | 0.50 | 1.000 | False | True | -0.001 [-0.004, +0.001] |
| 500 | Q1 angle VQC | C3 GP classifier | -0.0007 [-0.0018, +0.0004] | 0.46 | 1.000 | False | True | -0.001 [-0.002, +0.000] |
| 500 | Q1 angle VQC | C4 kNN | +0.0163 [+0.0139, +0.0187] | 1.00 | 0.018 | True | True | -0.101 [-0.105, -0.096] |
| 500 | Q1 angle VQC | C5 random forest | +0.0009 [-0.0006, +0.0025] | 0.54 | 1.000 | False | True | -0.041 [-0.047, -0.036] |
| 500 | Q1 angle VQC | C6 XGBoost | +0.0018 [+0.0002, +0.0035] | 0.74 | 0.707 | False | True | -0.034 [-0.039, -0.030] |
| 500 | Q2 VQC trained scales | C1 LR | +0.0001 [-0.0017, +0.0021] | 0.46 | 1.000 | False | True | +0.005 [+0.002, +0.008] |
| 500 | Q2 VQC trained scales | C2 SVM-RBF | -0.0017 [-0.0034, +0.0003] | 0.22 | 1.000 | False | True | +0.002 [-0.001, +0.004] |
| 500 | Q2 VQC trained scales | C3 GP classifier | -0.0024 [-0.0037, -0.0010] | 0.22 | 1.000 | False | True | +0.002 [+0.000, +0.005] |
| 500 | Q2 VQC trained scales | C4 kNN | +0.0145 [+0.0120, +0.0173] | 1.00 | 0.018 | True | True | -0.097 [-0.103, -0.092] |
| 500 | Q2 VQC trained scales | C5 random forest | -0.0008 [-0.0025, +0.0012] | 0.34 | 1.000 | False | True | -0.038 [-0.044, -0.032] |
| 500 | Q2 VQC trained scales | C6 XGBoost | +0.0001 [-0.0014, +0.0019] | 0.52 | 1.000 | False | True | -0.031 [-0.036, -0.027] |
| 500 | Q3 quantum-kernel SVM | C1 LR | +0.0016 [+0.0002, +0.0032] | 0.68 | 0.843 | False | True | +0.001 [-0.001, +0.004] |
| 500 | Q3 quantum-kernel SVM | C2 SVM-RBF | -0.0002 [-0.0015, +0.0012] | 0.48 | 1.000 | False | True | -0.002 [-0.004, +0.000] |
| 500 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0009 [-0.0023, +0.0004] | 0.46 | 1.000 | False | True | -0.001 [-0.003, +0.001] |
| 500 | Q3 quantum-kernel SVM | C4 kNN | +0.0161 [+0.0134, +0.0189] | 1.00 | 0.018 | True | True | -0.101 [-0.106, -0.096] |
| 500 | Q3 quantum-kernel SVM | C5 random forest | +0.0008 [-0.0009, +0.0024] | 0.60 | 1.000 | False | True | -0.041 [-0.047, -0.036] |
| 500 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0016 [-0.0001, +0.0034] | 0.62 | 1.000 | False | True | -0.035 [-0.039, -0.031] |