# Table 3 — Pre-registered small-data benchmark: best quantum − best classical by training size N

Built from commit 78984b0 by make_paper.py.

Paired over 5 folds × 40 subsets (N = 500: **reduced sample**, 10 subsets); 95% hierarchical-bootstrap CIs (folds, then subsets); one-sided superiority with Holm; non-inferiority margin −0.02. Source: `results/sd_analysis_k8_main.json` (and `_k4_main.json`). C7 TabPFN not run (licence login required); Q4 brick-wall only at N = 100 (torch blocked at other sizes).

**k = 8** (Holm family: 96 comparisons)

| N | best quantum | best classical | mean d [95% CI] | quantum wins | superiority | non-inferior | label |
|---|---|---|---|---|---|---|---|
| 25 | Q3 quantum-kernel SVM | C3 GP | -0.0040 [-0.0099, +0.0012] | 0.45 | no | yes |  |
| 50 | Q3 quantum-kernel SVM | C3 GP | -0.0033 [-0.0058, -0.0009] | 0.41 | no | yes |  |
| 100 | Q3 quantum-kernel SVM | C5 random forest | -0.0028 [-0.0049, -0.0010] | 0.42 | no | yes |  |
| 200 | Q1 angle VQC | C5 random forest | -0.0018 [-0.0036, -0.0002] | 0.46 | no | yes |  |
| 500 | Q1 angle VQC | C5 random forest | -0.0009 [-0.0020, +0.0002] | 0.40 | no | yes | reduced sample |

k = 4: not available (incomplete)

