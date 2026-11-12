**Stage A** (folds 0–1, mean validation AUC; 1500 steps, extended to 3000 if the best checkpoint fell in the last 20%; Adam lr 0.02, batch 512, linear head)

n = 8

| k | columns | π/6 | π/4 | π/2 | π |
|---|---|---|---|---|---|
| 2 | 2 | 0.9183 | 0.9145 | 0.8385 | 0.6634 |
| 2 | 4 | 0.9211 | 0.9191 | 0.8954 | 0.8235 |
| 4 | 2 | 0.8710 | 0.8667 | 0.7325 | 0.6066 |
| 4 | 4 | 0.9168 | 0.9220 | 0.8856 | 0.7643 |

n = 12

| k | columns | π/6 | π/4 | π/2 | π |
|---|---|---|---|---|---|
| 2 | 2 | 0.8918 | 0.9160 | 0.9188 | 0.8927 |
| 2 | 4 | 0.9238 | 0.9254 | 0.9279 | 0.9104 |
| 4 | 2 | 0.9352 | 0.9328 | 0.9241 | 0.8870 |
| 4 | 4 | 0.9349 | 0.9344 | 0.9311 | 0.9153 |
| 6 | 2 | 0.9312 | 0.9315 | 0.9251 | 0.8951 |
| 6 | 4 | 0.9344 | 0.9361 | 0.9325 | 0.9151 |

**Stage B** (folds 0–1, mean validation AUC at the stage-A winner; ‘Δ’ = change vs the winner's 0.9220 (n=8) / 0.9361 (n=12))

| n | variant | mean val AUC |
|---|---|---|
| 8 | k=4, cols=4, head=linear, lr=0.02 | 0.9220 |
| 8 | k=4, cols=4, head=corr, lr=0.02 | 0.9211 |
| 8 | k=4, cols=4, head=linear, lr=0.06 | 0.9203 |
| 8 | k=4, cols=4, head=linear_score, lr=0.02 | 0.9191 |
| 12 | k=6, cols=4, head=linear, lr=0.02 | 0.9361 |
| 12 | k=6, cols=4, head=linear_score, lr=0.02 | 0.9336 |
| 12 | k=6, cols=4, head=corr, lr=0.02 | 0.9310 |

**Stage C** (5 folds, test AUC; final = best validation AUC over stages A/B, runner-up second)

| n | config | test AUC (mean ± SD) | sens @ 90% spec | val AUC | per-fold test AUC |
|---|---|---|---|---|---|
| 8 | final: k=4, cols=4, scale π/4, head linear | 0.9204 ± 0.0047 | 0.783 | 0.9247 | 0.917, 0.922, 0.916, 0.919, 0.928 |
| 8 | runner-up: k=4, cols=4, scale π/4, head corr | 0.9208 ± 0.0056 | 0.780 | 0.9249 | 0.916, 0.923, 0.918, 0.918, 0.930 |
| 12 | final: k=6, cols=4, scale π/4, head linear | 0.9341 ± 0.0079 | 0.809 | 0.9376 | 0.945, 0.935, 0.928, 0.925, 0.937 |
| 12 | runner-up: k=4, cols=2, scale π/6, head linear | 0.9322 ± 0.0083 | 0.808 | 0.9349 | 0.943, 0.930, 0.924, 0.926, 0.939 |

**Paper phase map** φ = π·x/max|x| (max from train) vs tanh map, same final (k, columns), 5 folds

| n | phase map | test AUC (mean ± SD) |
|---|---|---|
| 8 | tanh, scale π/4 (final) | 0.9204 ± 0.0047 |
| 8 | paper: π·x/max abs(x) | 0.9108 ± 0.0104 |
| 12 | tanh, scale π/4 (final) | 0.9341 ± 0.0079 |