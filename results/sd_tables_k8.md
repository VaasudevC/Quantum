### k = 8: mean test AUC [95% hierarchical-bootstrap CI] (5 folds x up to 40 subsets), generalisation gap in parentheses

| model | N=100 |
|---|---|
| Q1 angle VQC | 0.894 [0.880, 0.905] (+0.06) |
| Q2 VQC trained scales | 0.889 [0.872, 0.907] (+0.07) |
| Q3 quantum-kernel SVM | 0.896 [0.883, 0.905] (+0.06) |
| Q4 brick-wall n=8 | 0.852 [0.811, 0.881] (+0.07) |
| C1 LR | 0.890 [0.876, 0.902] (+0.07) |
| C2 SVM-RBF | 0.894 [0.880, 0.905] (+0.06) |
| C3 GP classifier | 0.894 [0.880, 0.909] (+0.06) |
| C4 kNN | 0.869 [0.851, 0.885] (+0.13) |
| C5 random forest | 0.894 [0.879, 0.906] (+0.09) |
| C6 XGBoost | 0.874 [0.856, 0.887] (+0.12) |

### Best quantum vs best classical per size (best = highest mean AUC at that size)

| N | best quantum | best classical | mean d [95% CI] | wins | Holm p (superiority) | superiority | non-inferior (CI lower > −0.02) |
|---|---|---|---|---|---|---|---|

### All 0 quantum × classical × size comparisons (Holm over 0; d = quantum − classical)

| N | quantum | classical | mean d [95% CI] | wins | p_sup (Holm) | superior | non-inf. | gap diff [CI] |
|---|---|---|---|---|---|---|---|---|