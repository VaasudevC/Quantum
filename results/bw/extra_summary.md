**EXTRA 1 — 3 seeds × 5 folds** (mean over folds per seed; seed-to-seed SD of that mean; the SD over all 15 fold×seed values is also given)

| model | seed 0 | seed 1 | seed 2 | mean of seeds ± SD | all 15 values: mean ± SD |
|---|---|---|---|---|---|
| brick-wall n=8 | 0.9204 | 0.9209 | 0.9214 | 0.9209 ± 0.0005 | 0.9209 ± 0.0048 |
| brick-wall n=12 | 0.9341 | 0.9349 | 0.9347 | 0.9345 ± 0.0004 | 0.9345 ± 0.0068 |
| angle VQC π/6, k=8 | 0.9280 | 0.9281 | – | 0.9281 ± 0.0000 | 0.9281 ± 0.0060 |
| angle VQC π/6, k=32 | 0.9510 | 0.9518 | – | 0.9514 ± 0.0005 | 0.9514 ± 0.0054 |

**EXTRA 3 — unitary butterfly vs brick-wall, n=8** (5 folds)

| model | params (circuit) | depth (logical) | test AUC | val AUC | folds (test) |
|---|---|---|---|---|---|
| brick-wall, 4 columns (k=4, π/4) | 48 | 4 + 7 + 1 + 8 (NN) | 0.9204 ± 0.0047 | 0.9247 | 0.917, 0.922, 0.916, 0.919, 0.928 |
| butterfly, 6 layers = 2 butterflies (k=4, π/6) | 72 | 4 + 3 + 1 + 12 (all-to-all) | 0.9225 ± 0.0049 | 0.9263 | 0.917, 0.925, 0.918, 0.923, 0.929 |

**EXTRA 4 — 16 qubits** (k=4, 4 columns, π/4, sector dimension 1820; folds finished: 3/5): test AUC f0 0.9440, f1 0.9387, f2 0.9351; mean 0.9393; max|head| 9.9