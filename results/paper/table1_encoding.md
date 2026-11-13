# Table 1 — Encoding study: angle scale s in angle = s·tanh(z)

Built from commit 1a60f8a by make_paper.py.

**(a) Angle VQC (8 qubits × 4 layers, re-uploading), 5-fold test AUC (mean ± SD), 1000 steps** — `results/stage2_steps1000.jsonl`

| scale s | k = 8 | k = 16 | k = 32 |
|---|---|---|---|
| pi | 0.910 ± 0.009 | 0.892 ± 0.015 | 0.791 ± 0.078 |
| pi/2 | 0.928 ± 0.006 | 0.923 ± 0.010 | 0.910 ± 0.012 |
| pi/6 | 0.928 ± 0.006 | 0.939 ± 0.006 | 0.951 ± 0.006 |
| LR on the same features | 0.922 ± 0.007 | 0.932 ± 0.008 | 0.945 ± 0.008 |

**(b) Scale sweep at k = 32 (600 steps), 5-fold test AUC** — `results/stage1_sweep.jsonl`

| s | test AUC | val AUC |
|---|---|---|
| pi/8 | 0.947 ± 0.006 | 0.948 |
| pi/6 | 0.950 ± 0.006 | 0.950 |
| pi/4 | 0.943 ± 0.007 | 0.945 |
| pi/3 | 0.933 ± 0.007 | 0.936 |
| pi/2 | 0.908 ± 0.009 | 0.911 |
| 2*pi/3 | 0.850 ± 0.020 | 0.850 |
| pi | 0.794 ± 0.072 | 0.796 |

**(c) Replication in the magic-state brick-wall QNN (our reimplementation): mean validation AUC, folds 0–1, phase φ = s·tanh(z)** — `results/bw/p3_stageA2.jsonl`

| configuration | s = π/6 | π/4 | π/2 | π |
|---|---|---|---|---|
| n=8, particles k=4, 4 columns | 0.917 | 0.922 | 0.886 | 0.764 |
| n=8, particles k=2, 4 columns | 0.921 | 0.919 | 0.895 | 0.823 |
| n=12, particles k=6, 4 columns | 0.934 | 0.936 | 0.933 | 0.915 |
| n=12, particles k=4, 2 columns | 0.935 | 0.933 | 0.924 | 0.887 |
