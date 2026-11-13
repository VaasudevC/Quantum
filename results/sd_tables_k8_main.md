### k = 8: mean test AUC [95% hierarchical-bootstrap CI] (5 folds x up to 40 subsets), generalisation gap in parentheses

| model | N=25 | N=50 | N=100 | N=200 | N=500 |
|---|---|---|---|---|---|
| Q1 angle VQC | 0.825 [0.815, 0.833] (+0.17) | 0.866 [0.860, 0.872] (+0.12) | 0.889 [0.882, 0.894] (+0.07) | 0.905 [0.900, 0.911] (+0.04) | 0.919 [0.913, 0.925] (+0.02) |
| Q2 VQC trained scales | 0.814 [0.803, 0.823] (+0.19) | 0.854 [0.846, 0.862] (+0.13) | 0.883 [0.876, 0.889] (+0.08) | 0.901 [0.895, 0.906] (+0.04) | 0.916 [0.910, 0.922] (+0.02) |
| Q3 quantum-kernel SVM | 0.828 [0.818, 0.837] (+0.17) | 0.866 [0.860, 0.872] (+0.11) | 0.889 [0.882, 0.895] (+0.07) | 0.905 [0.899, 0.910] (+0.04) | 0.919 [0.912, 0.925] (+0.02) |
| Q4 brick-wall n=8 | – | – | 0.857 [0.846, 0.866] (+0.07) | – | – |
| C1 LR | 0.818 [0.808, 0.828] (+0.18) | 0.862 [0.855, 0.869] (+0.12) | 0.887 [0.881, 0.893] (+0.07) | 0.902 [0.896, 0.909] (+0.04) | 0.914 [0.907, 0.922] (+0.01) |
| C2 SVM-RBF | 0.826 [0.818, 0.834] (+0.17) | 0.865 [0.860, 0.871] (+0.12) | 0.889 [0.884, 0.895] (+0.07) | 0.904 [0.898, 0.911] (+0.04) | 0.919 [0.912, 0.925] (+0.02) |
| C3 GP classifier | 0.832 [0.825, 0.840] (+0.16) | 0.869 [0.864, 0.875] (+0.11) | 0.891 [0.886, 0.897] (+0.07) | 0.906 [0.900, 0.912] (+0.04) | 0.918 [0.912, 0.925] (+0.02) |
| C4 kNN | 0.786 [0.776, 0.795] (+0.21) | 0.833 [0.825, 0.839] (+0.17) | 0.863 [0.859, 0.868] (+0.14) | 0.884 [0.878, 0.890] (+0.12) | 0.903 [0.897, 0.908] (+0.10) |
| C5 random forest | 0.828 [0.818, 0.837] (+0.17) | 0.868 [0.863, 0.874] (+0.13) | 0.891 [0.885, 0.898] (+0.10) | 0.907 [0.901, 0.912] (+0.08) | 0.920 [0.913, 0.926] (+0.06) |
| C6 XGBoost | 0.785 [0.772, 0.797] (+0.21) | 0.846 [0.839, 0.851] (+0.15) | 0.875 [0.868, 0.881] (+0.12) | 0.898 [0.893, 0.904] (+0.09) | 0.916 [0.910, 0.922] (+0.06) |

### Best quantum vs best classical per size (best = highest mean AUC at that size)

| N | best quantum | best classical | mean d [95% CI] | wins | Holm p (superiority) | superiority | non-inferior (CI lower > −0.02) |
|---|---|---|---|---|---|---|---|
| 25 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0040 [-0.0099, +0.0012] | 0.45 | 1.000 | False | True |
| 50 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0033 [-0.0058, -0.0009] | 0.41 | 1.000 | False | True |
| 100 | Q3 quantum-kernel SVM | C5 random forest | -0.0028 [-0.0049, -0.0010] | 0.42 | 1.000 | False | True |
| 200 | Q1 angle VQC | C5 random forest | -0.0018 [-0.0036, -0.0002] | 0.46 | 1.000 | False | True |
| 500 | Q1 angle VQC | C5 random forest | -0.0009 [-0.0020, +0.0002] | 0.40 | 1.000 | False | True |

### All 96 quantum × classical × size comparisons (Holm over 96; d = quantum − classical)

| N | quantum | classical | mean d [95% CI] | wins | p_sup (Holm) | superior | non-inf. | gap diff [CI] |
|---|---|---|---|---|---|---|---|---|
| 25 | Q1 angle VQC | C1 LR | +0.0066 [+0.0019, +0.0106] | 0.59 | 0.223 | False | True | -0.006 [-0.010, -0.001] |
| 25 | Q1 angle VQC | C2 SVM-RBF | -0.0009 [-0.0065, +0.0050] | 0.52 | 1.000 | False | True | +0.000 [-0.006, +0.006] |
| 25 | Q1 angle VQC | C3 GP classifier | -0.0074 [-0.0128, -0.0020] | 0.40 | 1.000 | False | True | +0.007 [+0.001, +0.012] |
| 25 | Q1 angle VQC | C4 kNN | +0.0389 [+0.0336, +0.0440] | 0.90 | 0.019 | True | True | -0.043 [-0.048, -0.038] |
| 25 | Q1 angle VQC | C5 random forest | -0.0033 [-0.0073, +0.0003] | 0.42 | 1.000 | False | True | -0.000 [-0.005, +0.004] |
| 25 | Q1 angle VQC | C6 XGBoost | +0.0394 [+0.0322, +0.0469] | 0.89 | 0.019 | True | True | -0.042 [-0.051, -0.034] |
| 25 | Q2 VQC trained scales | C1 LR | -0.0044 [-0.0105, +0.0014] | 0.43 | 1.000 | False | True | +0.008 [+0.002, +0.015] |
| 25 | Q2 VQC trained scales | C2 SVM-RBF | -0.0119 [-0.0191, -0.0050] | 0.35 | 1.000 | False | True | +0.014 [+0.007, +0.023] |
| 25 | Q2 VQC trained scales | C3 GP classifier | -0.0183 [-0.0251, -0.0121] | 0.28 | 1.000 | False | False | +0.021 [+0.014, +0.028] |
| 25 | Q2 VQC trained scales | C4 kNN | +0.0279 [+0.0215, +0.0344] | 0.77 | 0.019 | True | True | -0.029 [-0.036, -0.023] |
| 25 | Q2 VQC trained scales | C5 random forest | -0.0142 [-0.0186, -0.0098] | 0.29 | 1.000 | False | True | +0.014 [+0.009, +0.018] |
| 25 | Q2 VQC trained scales | C6 XGBoost | +0.0285 [+0.0215, +0.0353] | 0.78 | 0.019 | True | True | -0.028 [-0.035, -0.021] |
| 25 | Q3 quantum-kernel SVM | C1 LR | +0.0100 [+0.0049, +0.0150] | 0.70 | 0.019 | True | True | -0.010 [-0.015, -0.004] |
| 25 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0025 [-0.0032, +0.0080] | 0.58 | 1.000 | False | True | -0.004 [-0.010, +0.002] |
| 25 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0040 [-0.0099, +0.0012] | 0.45 | 1.000 | False | True | +0.002 [-0.003, +0.008] |
| 25 | Q3 quantum-kernel SVM | C4 kNN | +0.0423 [+0.0362, +0.0486] | 0.92 | 0.019 | True | True | -0.048 [-0.054, -0.041] |
| 25 | Q3 quantum-kernel SVM | C5 random forest | +0.0002 [-0.0031, +0.0034] | 0.52 | 1.000 | False | True | -0.005 [-0.009, -0.001] |
| 25 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0428 [+0.0360, +0.0499] | 0.89 | 0.019 | True | True | -0.047 [-0.054, -0.040] |
| 50 | Q1 angle VQC | C1 LR | +0.0036 [+0.0012, +0.0062] | 0.60 | 0.078 | False | True | -0.003 [-0.006, +0.000] |
| 50 | Q1 angle VQC | C2 SVM-RBF | +0.0006 [-0.0024, +0.0037] | 0.47 | 1.000 | False | True | -0.001 [-0.006, +0.003] |
| 50 | Q1 angle VQC | C3 GP classifier | -0.0035 [-0.0061, -0.0009] | 0.41 | 1.000 | False | True | +0.003 [-0.000, +0.007] |
| 50 | Q1 angle VQC | C4 kNN | +0.0334 [+0.0294, +0.0376] | 0.93 | 0.019 | True | True | -0.052 [-0.057, -0.047] |
| 50 | Q1 angle VQC | C5 random forest | -0.0024 [-0.0054, +0.0002] | 0.45 | 1.000 | False | True | -0.012 [-0.016, -0.008] |
| 50 | Q1 angle VQC | C6 XGBoost | +0.0204 [+0.0165, +0.0242] | 0.86 | 0.019 | True | True | -0.038 [-0.043, -0.033] |
| 50 | Q2 VQC trained scales | C1 LR | -0.0081 [-0.0119, -0.0044] | 0.39 | 1.000 | False | True | +0.014 [+0.009, +0.019] |
| 50 | Q2 VQC trained scales | C2 SVM-RBF | -0.0110 [-0.0159, -0.0062] | 0.34 | 1.000 | False | True | +0.015 [+0.009, +0.021] |
| 50 | Q2 VQC trained scales | C3 GP classifier | -0.0152 [-0.0199, -0.0107] | 0.24 | 1.000 | False | True | +0.020 [+0.015, +0.026] |
| 50 | Q2 VQC trained scales | C4 kNN | +0.0218 [+0.0168, +0.0266] | 0.82 | 0.019 | True | True | -0.035 [-0.042, -0.029] |
| 50 | Q2 VQC trained scales | C5 random forest | -0.0141 [-0.0193, -0.0095] | 0.26 | 1.000 | False | True | +0.005 [-0.001, +0.011] |
| 50 | Q2 VQC trained scales | C6 XGBoost | +0.0087 [+0.0029, +0.0142] | 0.69 | 0.078 | False | True | -0.021 [-0.029, -0.015] |
| 50 | Q3 quantum-kernel SVM | C1 LR | +0.0038 [+0.0009, +0.0067] | 0.62 | 0.244 | False | True | -0.004 [-0.009, +0.000] |
| 50 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0008 [-0.0019, +0.0035] | 0.51 | 1.000 | False | True | -0.003 [-0.007, +0.001] |
| 50 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0033 [-0.0058, -0.0009] | 0.41 | 1.000 | False | True | +0.002 [-0.001, +0.005] |
| 50 | Q3 quantum-kernel SVM | C4 kNN | +0.0336 [+0.0292, +0.0383] | 0.92 | 0.019 | True | True | -0.053 [-0.060, -0.047] |
| 50 | Q3 quantum-kernel SVM | C5 random forest | -0.0022 [-0.0056, +0.0012] | 0.48 | 1.000 | False | True | -0.013 [-0.019, -0.008] |
| 50 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0206 [+0.0161, +0.0253] | 0.85 | 0.019 | True | True | -0.040 [-0.046, -0.033] |
| 100 | Q1 angle VQC | C1 LR | +0.0020 [-0.0003, +0.0041] | 0.61 | 1.000 | False | True | -0.001 [-0.004, +0.002] |
| 100 | Q1 angle VQC | C2 SVM-RBF | -0.0006 [-0.0027, +0.0013] | 0.45 | 1.000 | False | True | +0.000 [-0.003, +0.003] |
| 100 | Q1 angle VQC | C3 GP classifier | -0.0028 [-0.0051, -0.0009] | 0.38 | 1.000 | False | True | +0.001 [-0.001, +0.004] |
| 100 | Q1 angle VQC | C4 kNN | +0.0251 [+0.0209, +0.0291] | 0.95 | 0.019 | True | True | -0.069 [-0.076, -0.062] |
| 100 | Q1 angle VQC | C5 random forest | -0.0029 [-0.0051, -0.0008] | 0.41 | 1.000 | False | True | -0.030 [-0.034, -0.025] |
| 100 | Q1 angle VQC | C6 XGBoost | +0.0137 [+0.0103, +0.0173] | 0.84 | 0.019 | True | True | -0.054 [-0.061, -0.048] |
| 100 | Q2 VQC trained scales | C1 LR | -0.0038 [-0.0073, -0.0004] | 0.42 | 1.000 | False | True | +0.010 [+0.005, +0.016] |
| 100 | Q2 VQC trained scales | C2 SVM-RBF | -0.0064 [-0.0096, -0.0033] | 0.28 | 1.000 | False | True | +0.011 [+0.005, +0.018] |
| 100 | Q2 VQC trained scales | C3 GP classifier | -0.0086 [-0.0113, -0.0059] | 0.24 | 1.000 | False | True | +0.013 [+0.008, +0.017] |
| 100 | Q2 VQC trained scales | C4 kNN | +0.0194 [+0.0146, +0.0240] | 0.86 | 0.019 | True | True | -0.058 [-0.065, -0.051] |
| 100 | Q2 VQC trained scales | C5 random forest | -0.0087 [-0.0110, -0.0066] | 0.23 | 1.000 | False | True | -0.019 [-0.023, -0.014] |
| 100 | Q2 VQC trained scales | C6 XGBoost | +0.0080 [+0.0057, +0.0103] | 0.74 | 0.019 | True | True | -0.043 [-0.048, -0.038] |
| 100 | Q3 quantum-kernel SVM | C1 LR | +0.0020 [-0.0005, +0.0043] | 0.55 | 1.000 | False | True | -0.001 [-0.004, +0.003] |
| 100 | Q3 quantum-kernel SVM | C2 SVM-RBF | -0.0006 [-0.0024, +0.0013] | 0.45 | 1.000 | False | True | +0.000 [-0.003, +0.003] |
| 100 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0028 [-0.0051, -0.0009] | 0.39 | 1.000 | False | True | +0.002 [-0.002, +0.005] |
| 100 | Q3 quantum-kernel SVM | C4 kNN | +0.0252 [+0.0206, +0.0294] | 0.95 | 0.019 | True | True | -0.069 [-0.077, -0.061] |
| 100 | Q3 quantum-kernel SVM | C5 random forest | -0.0028 [-0.0049, -0.0010] | 0.42 | 1.000 | False | True | -0.029 [-0.035, -0.024] |
| 100 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0138 [+0.0103, +0.0172] | 0.83 | 0.019 | True | True | -0.054 [-0.061, -0.047] |
| 100 | Q4 brick-wall n=8 | C1 LR | -0.0299 [-0.0378, -0.0225] | 0.14 | 1.000 | False | False | +0.001 [-0.004, +0.006] |
| 100 | Q4 brick-wall n=8 | C2 SVM-RBF | -0.0325 [-0.0400, -0.0257] | 0.11 | 1.000 | False | False | +0.002 [-0.003, +0.008] |
| 100 | Q4 brick-wall n=8 | C3 GP classifier | -0.0346 [-0.0426, -0.0276] | 0.09 | 1.000 | False | False | +0.004 [-0.001, +0.008] |
| 100 | Q4 brick-wall n=8 | C4 kNN | -0.0067 [-0.0153, +0.0008] | 0.51 | 1.000 | False | True | -0.067 [-0.074, -0.059] |
| 100 | Q4 brick-wall n=8 | C5 random forest | -0.0347 [-0.0421, -0.0282] | 0.07 | 1.000 | False | False | -0.027 [-0.032, -0.023] |
| 100 | Q4 brick-wall n=8 | C6 XGBoost | -0.0181 [-0.0256, -0.0114] | 0.35 | 1.000 | False | False | -0.052 [-0.058, -0.045] |
| 200 | Q1 angle VQC | C1 LR | +0.0024 [+0.0007, +0.0042] | 0.69 | 0.088 | False | True | +0.001 [-0.000, +0.002] |
| 200 | Q1 angle VQC | C2 SVM-RBF | +0.0004 [-0.0011, +0.0020] | 0.53 | 1.000 | False | True | -0.000 [-0.002, +0.001] |
| 200 | Q1 angle VQC | C3 GP classifier | -0.0011 [-0.0022, -0.0001] | 0.45 | 1.000 | False | True | +0.001 [-0.001, +0.002] |
| 200 | Q1 angle VQC | C4 kNN | +0.0206 [+0.0179, +0.0231] | 0.97 | 0.019 | True | True | -0.078 [-0.081, -0.074] |
| 200 | Q1 angle VQC | C5 random forest | -0.0018 [-0.0036, -0.0002] | 0.46 | 1.000 | False | True | -0.038 [-0.042, -0.035] |
| 200 | Q1 angle VQC | C6 XGBoost | +0.0066 [+0.0048, +0.0083] | 0.75 | 0.019 | True | True | -0.050 [-0.053, -0.048] |
| 200 | Q2 VQC trained scales | C1 LR | -0.0018 [-0.0042, +0.0003] | 0.42 | 1.000 | False | True | +0.007 [+0.004, +0.010] |
| 200 | Q2 VQC trained scales | C2 SVM-RBF | -0.0039 [-0.0056, -0.0021] | 0.31 | 1.000 | False | True | +0.006 [+0.003, +0.008] |
| 200 | Q2 VQC trained scales | C3 GP classifier | -0.0054 [-0.0066, -0.0043] | 0.17 | 1.000 | False | True | +0.006 [+0.004, +0.009] |
| 200 | Q2 VQC trained scales | C4 kNN | +0.0163 [+0.0137, +0.0188] | 0.94 | 0.019 | True | True | -0.072 [-0.075, -0.068] |
| 200 | Q2 VQC trained scales | C5 random forest | -0.0061 [-0.0077, -0.0045] | 0.24 | 1.000 | False | True | -0.033 [-0.036, -0.029] |
| 200 | Q2 VQC trained scales | C6 XGBoost | +0.0023 [+0.0009, +0.0038] | 0.61 | 0.040 | True | True | -0.045 [-0.047, -0.042] |
| 200 | Q3 quantum-kernel SVM | C1 LR | +0.0023 [+0.0003, +0.0042] | 0.68 | 0.840 | False | True | +0.002 [-0.001, +0.004] |
| 200 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0003 [-0.0010, +0.0016] | 0.49 | 1.000 | False | True | +0.000 [-0.001, +0.002] |
| 200 | Q3 quantum-kernel SVM | C3 GP classifier | -0.0012 [-0.0023, -0.0002] | 0.48 | 1.000 | False | True | +0.001 [-0.001, +0.003] |
| 200 | Q3 quantum-kernel SVM | C4 kNN | +0.0205 [+0.0183, +0.0226] | 0.98 | 0.019 | True | True | -0.077 [-0.081, -0.073] |
| 200 | Q3 quantum-kernel SVM | C5 random forest | -0.0019 [-0.0031, -0.0007] | 0.44 | 1.000 | False | True | -0.038 [-0.041, -0.035] |
| 200 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0065 [+0.0052, +0.0079] | 0.77 | 0.019 | True | True | -0.050 [-0.053, -0.047] |
| 500 | Q1 angle VQC | C1 LR | +0.0046 [+0.0023, +0.0068] | 0.94 | 0.019 | True | True | +0.002 [-0.001, +0.005] |
| 500 | Q1 angle VQC | C2 SVM-RBF | +0.0004 [-0.0009, +0.0017] | 0.52 | 1.000 | False | True | -0.001 [-0.003, +0.001] |
| 500 | Q1 angle VQC | C3 GP classifier | +0.0005 [-0.0007, +0.0016] | 0.60 | 1.000 | False | True | +0.000 [-0.002, +0.002] |
| 500 | Q1 angle VQC | C4 kNN | +0.0162 [+0.0141, +0.0183] | 1.00 | 0.019 | True | True | -0.081 [-0.085, -0.077] |
| 500 | Q1 angle VQC | C5 random forest | -0.0009 [-0.0020, +0.0002] | 0.40 | 1.000 | False | True | -0.046 [-0.052, -0.040] |
| 500 | Q1 angle VQC | C6 XGBoost | +0.0028 [+0.0014, +0.0043] | 0.72 | 0.019 | True | True | -0.039 [-0.043, -0.036] |
| 500 | Q2 VQC trained scales | C1 LR | +0.0014 [-0.0008, +0.0038] | 0.66 | 1.000 | False | True | +0.006 [+0.002, +0.011] |
| 500 | Q2 VQC trained scales | C2 SVM-RBF | -0.0028 [-0.0046, -0.0011] | 0.26 | 1.000 | False | True | +0.004 [+0.001, +0.007] |
| 500 | Q2 VQC trained scales | C3 GP classifier | -0.0027 [-0.0042, -0.0011] | 0.28 | 1.000 | False | True | +0.005 [+0.002, +0.008] |
| 500 | Q2 VQC trained scales | C4 kNN | +0.0130 [+0.0106, +0.0155] | 0.96 | 0.019 | True | True | -0.076 [-0.081, -0.072] |
| 500 | Q2 VQC trained scales | C5 random forest | -0.0041 [-0.0063, -0.0021] | 0.24 | 1.000 | False | True | -0.042 [-0.049, -0.034] |
| 500 | Q2 VQC trained scales | C6 XGBoost | -0.0004 [-0.0026, +0.0017] | 0.44 | 1.000 | False | True | -0.035 [-0.039, -0.031] |
| 500 | Q3 quantum-kernel SVM | C1 LR | +0.0045 [+0.0023, +0.0069] | 0.88 | 0.019 | True | True | +0.002 [-0.001, +0.005] |
| 500 | Q3 quantum-kernel SVM | C2 SVM-RBF | +0.0003 [-0.0009, +0.0015] | 0.50 | 1.000 | False | True | -0.000 [-0.002, +0.002] |
| 500 | Q3 quantum-kernel SVM | C3 GP classifier | +0.0004 [-0.0009, +0.0020] | 0.50 | 1.000 | False | True | +0.001 [-0.001, +0.003] |
| 500 | Q3 quantum-kernel SVM | C4 kNN | +0.0161 [+0.0137, +0.0185] | 1.00 | 0.019 | True | True | -0.081 [-0.085, -0.076] |
| 500 | Q3 quantum-kernel SVM | C5 random forest | -0.0010 [-0.0022, +0.0002] | 0.32 | 1.000 | False | True | -0.046 [-0.052, -0.039] |
| 500 | Q3 quantum-kernel SVM | C6 XGBoost | +0.0026 [+0.0012, +0.0041] | 0.74 | 0.019 | True | True | -0.039 [-0.043, -0.035] |