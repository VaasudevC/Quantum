# QB6 Confirmation Results

> Confirmation folds: [3, 4]. N=100 subsets from inferior-MI pool, r=0-9. B=5000 fold-stratified bootstrap. Holm correction over 2 candidates.

## Results vs SVM-RBF (best classical, chosen by inner CV)

| candidate | mean d [95% CI] | CI lo > 0 | point >= +0.005 | Holm pass | claim |
|---|---|---|---|---|---|
| subtype_IMI Q1 angle VQC | +0.0030 [-0.0049, +0.0108] | False | False | False | hypothesis for future work |
| subtype_IMI Q3 quantum-kernel SVM | +0.0055 [-0.0003, +0.0119] | False | True | False | hypothesis for future work |

## Results vs dequantized twins

| candidate | vs twin LR [95% CI] | vs twin deg-2 LR [95% CI] | vs twin SVM [95% CI] |
|---|---|---|---|
| subtype_IMI Q1 angle VQC | -0.0002 [-0.0070, +0.0047] | +0.0294 [+0.0177, +0.0436] | +0.0062 [-0.0046, +0.0153] |
| subtype_IMI Q3 quantum-kernel SVM | +0.0023 [-0.0007, +0.0053] | +0.0318 [+0.0196, +0.0478] | +0.0087 [+0.0033, +0.0145] |

## Overall claim

**No benefit confirmed** (Holm-adjusted). Candidates are classified as 'hypothesis for future work'.
The main-analysis claim (no benefit found on the standard analysis) is unchanged.
