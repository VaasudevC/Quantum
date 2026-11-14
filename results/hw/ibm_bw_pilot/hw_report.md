# Hardware report — ibm_bw_pilot

**Backend:** ibm_marrakesh  **ECGs:** 10  **Shots:** 500

## AUC with 95% bootstrap CIs (B=2000, seed=0, percentile)

| model | AUC noiseless [95% CI] | AUC raw [95% CI] | AUC postselected [95% CI] | kept fraction |
|---|---|---|---|---|
| bw8 | 0.9583 [0.7600, 1.0000] | 0.9167 [0.6400, 1.0000] | 0.9583 [0.7600, 1.0000] | 0.482 |
| vqc32 | 0.9583 [0.7600, 1.0000] | 1.0000 [1.0000, 1.0000] | – | – |

## Per-ECG agreement (hardware vs noiseless logit)

| model | Pearson r | Spearman ρ | MAE logit | contraction factor |
|---|---|---|---|---|
| bw8 | 0.907 | 0.867 | 3.636 | 1.383 |
| vqc32 | 0.975 | 0.952 | 3.029 | 0.309 |

## Calibration (noiseless-optimal threshold)

| model | condition | bal. acc | sens | spec | acc |
|---|---|---|---|---|---|
| bw8 | noiseless | 0.917 | 1.000 | 0.833 | 0.900 |
| bw8 | hw raw | 0.500 | 1.000 | 0.000 | 0.400 |
| bw8 | hw rescaled | 0.500 | 1.000 | 0.000 | 0.400 |
| vqc32 | noiseless | 0.917 | 1.000 | 0.833 | 0.900 |
| vqc32 | hw raw | 0.500 | 1.000 | 0.000 | 0.400 |
| vqc32 | hw rescaled | 0.500 | 1.000 | 0.000 | 0.400 |

*Rescale factors: bw8 s = 2.072524338982333, vqc32 s = 3.2537636085508534*

Note: threshold is noiseless-optimal (balanced-accuracy) — not from a held-out validation set.
