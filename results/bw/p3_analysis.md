| features | model | mean test AUC ± SD | wins vs best classical (folds) | AUC diff vs best classical [95% cluster-bootstrap CI] | beats every classical (mean) | advantage rule met |
|---|---|---|---|---|---|---|
| 8 | lr (classical) | 0.9222 ± 0.0067 | – | – | – | – |
| 8 | lr2 (classical) | 0.9312 ± 0.0067 | – | – | – | – |
| 8 | rf (classical) | 0.9367 ± 0.0058 | – | – | – | – |
| 8 | xgb (classical) | 0.9374 ± 0.0049 | – | – | – | – |
| 8 | mlp (classical) | 0.9336 ± 0.0075 | – | – | – | – |
| 8 | **bw8_rank0** {'columns': 4, 'k': 4, 'scale': 0.7853981633974483, 'head': 'linear', 'lr': 0.02, 'steps': 1500} | 0.9204 ± 0.0047 | 0/5 | -0.0170 [-0.0195, -0.0147] vs xgb | False | False |
| 8 | **bw8_rank1** {'columns': 4, 'k': 4, 'scale': 0.7853981633974483, 'head': 'corr', 'lr': 0.02, 'steps': 1500} | 0.9208 ± 0.0056 | 0/5 | -0.0166 [-0.0191, -0.0144] vs xgb | False | False |
| 8 | **angle_vqc_k8** {} | 0.9280 ± 0.0063 | 0/5 | -0.0094 [-0.0113, -0.0076] vs xgb | False | False |
| 12 | lr (classical) | 0.9293 ± 0.0064 | – | – | – | – |
| 12 | lr2 (classical) | 0.9390 ± 0.0067 | – | – | – | – |
| 12 | rf (classical) | 0.9422 ± 0.0055 | – | – | – | – |
| 12 | xgb (classical) | 0.9452 ± 0.0050 | – | – | – | – |
| 12 | mlp (classical) | 0.9416 ± 0.0064 | – | – | – | – |
| 12 | **bw12_rank0** {'columns': 4, 'k': 6, 'scale': 0.7853981633974483, 'head': 'linear', 'lr': 0.02, 'steps': 1500} | 0.9341 ± 0.0079 | 0/5 | -0.0111 [-0.0131, -0.0092] vs xgb | False | False |
| 12 | **bw12_rank1** {'columns': 2, 'k': 4, 'scale': 0.5235987755982988, 'head': 'linear', 'lr': 0.02, 'steps': 1500} | 0.9322 ± 0.0083 | 0/5 | -0.0130 [-0.0151, -0.0109] vs xgb | False | False |
| 32 | lr (classical) | 0.9451 ± 0.0082 | – | – | – | – |
| 32 | lr2 (classical) | 0.9583 ± 0.0044 | – | – | – | – |
| 32 | rf (classical) | 0.9608 ± 0.0053 | – | – | – | – |
| 32 | xgb (classical) | 0.9647 ± 0.0046 | – | – | – | – |
| 32 | mlp (classical) | 0.9597 ± 0.0052 | – | – | – | – |
| 32 | **vqc_pi6_k32** {} | 0.9510 ± 0.0059 | 0/5 | -0.0136 [-0.0155, -0.0117] vs xgb | False | False |
| 32 | **vqc_trained_k32** {} | 0.9547 ± 0.0051 | 0/5 | -0.0100 [-0.0117, -0.0083] vs xgb | False | False |