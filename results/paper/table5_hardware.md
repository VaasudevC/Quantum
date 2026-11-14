# Table 5 — Real hardware results: ibm_marrakesh (Heron r2), 200 ECGs, 1000 shots, 2026-11-14

Built from commit e12d996 by make_paper.py.

Source: `results/hw/ibm_bw_hw/extended_analysis.json`.
Bootstrap CIs: B=2000, seed=0, percentile method. Noiseless AUC from exact simulation.
2.2 RU consumed (ibm_marrakesh billing). bw8: 50% of shots violated k=4 particle number; postselection left AUC unchanged.

## AUC with 95% bootstrap CIs and paired difference (hardware − noiseless)

| model | circuit (CZ, depth) | AUC noiseless [95% CI] | AUC raw [95% CI] | AUC postselected [95% CI] | kept fraction | paired Δ [95% CI] |
|---|---|---|---|---|---|---|
| bw8 | 136 CZ, depth 254 | 0.9253 [0.8846, 0.9576] | 0.9485 [0.9170, 0.9737] | 0.9483 [0.9149, 0.9750] | 0.499 ± 0.025 | +0.0233 [+0.0033, +0.0464] |
| vqc32 | 80 CZ, depth 208 | 0.9673 [0.9414, 0.9878] | 0.9561 [0.9238, 0.9822] | – | – | -0.0112 [-0.0239, -0.0005] |

## Per-ECG agreement (hardware vs noiseless logit)

| model | Pearson r | Spearman ρ | MAE logit | contraction factor |
|---|---|---|---|---|
| bw8 | 0.932 | 0.930 | 3.424 | 1.564 |
| vqc32 | 0.971 | 0.969 | 2.212 | 0.443 |

## Calibration (at hardware-optimal balanced-accuracy threshold)

| model | bal. acc | sens | spec | acc |
|---|---|---|---|---|
| bw8 hw raw | 0.895 | 0.877 | 0.913 | 0.900 |
| bw8 noiseless | 0.875 | 0.836 | 0.913 | 0.885 |
| vqc32 hw raw | 0.905 | 0.904 | 0.906 | 0.905 |
| vqc32 noiseless | 0.907 | 0.932 | 0.882 | 0.900 |

Note: thresholds tuned on the same test ECGs (optimistic; not held-out validation). The 200-ECG test set has ~±0.01 AUC sampling error.
bw8 paired Δ excludes 0 (CI [+0.003, +0.046]): hardware ranked ECGs better than noiseless simulation; likely a noise-induced shift in the decision function, not a reliable improvement (one run, one fold).
