# Hardware report — ibm_bw_hw

**Backend:** ibm_marrakesh  **ECGs:** 200  **Shots:** 1000

## AUC with 95% bootstrap CIs (B=2000, seed=0, percentile)

| model | AUC noiseless [95% CI] | AUC raw [95% CI] | AUC postselected [95% CI] | kept fraction |
|---|---|---|---|---|
| bw8 | 0.9253 [0.8846, 0.9576] | 0.9485 [0.9170, 0.9737] | 0.9483 [0.9149, 0.9750] | 0.499 |
| vqc32 | 0.9673 [0.9414, 0.9878] | 0.9561 [0.9238, 0.9822] | – | – |

## Per-ECG agreement (hardware vs noiseless logit)

| model | Pearson r | Spearman ρ | MAE logit | contraction factor |
|---|---|---|---|---|
| bw8 | 0.932 | 0.930 | 3.424 | 1.564 |
| vqc32 | 0.971 | 0.969 | 2.212 | 0.443 |

## Calibration (noiseless-optimal threshold)

| model | condition | bal. acc | sens | spec | acc |
|---|---|---|---|---|---|
| bw8 | noiseless | 0.875 | 0.836 | 0.913 | 0.885 |
| bw8 | hw raw | 0.500 | 1.000 | 0.000 | 0.365 |
| bw8 | hw rescaled | 0.500 | 1.000 | 0.000 | 0.365 |
| vqc32 | noiseless | 0.907 | 0.932 | 0.882 | 0.900 |
| vqc32 | hw raw | 0.571 | 1.000 | 0.142 | 0.455 |
| vqc32 | hw rescaled | 0.583 | 1.000 | 0.165 | 0.470 |

*Rescale factors: bw8 s = 1.9656580498346978, vqc32 s = 2.227970252758985*

Note: threshold is noiseless-optimal (balanced-accuracy) — not from a held-out validation set.
