# NEXT — improving the model (N1–N4)

Protocol unless stated: patient-grouped 5-fold CV, l1 selector, 8 qubits × 4 layers, k=32, scale pi/6, 1000 steps, choices on validation AUC.
**Change for all new runs:** the l1 path now uses `liblinear` with `random_state=0`, so the selection is deterministic. Checked against the
logged features: folds 0–3 select the identical ordered list, and fold 4 the same 32 features in a different order. The fold-4 baseline (z readout, fixed pi/6)
was therefore rerun with the new selector (`results/n0_baseline_fold4.jsonl`); folds 0–3 reuse the Stage 2 runs, which this code reproduces exactly.
Reference rows (5-fold test AUC): **VQC linear-Z pi/6 0.951 · LR 0.945 · degree-2 LR 0.958.**

## N3 — kernel-target alignment as a training-free encoding score (2026-11-05)

`python -u n3_kta.py` → `results/n3_kta.{json,md}`, `results/figs/kta_vs_auc.png`. K_ij = |⟨φ(x_i)|φ(x_j)⟩|² for the encoding-only circuit (θ = 0),
1000 class-stratified fold-0 train ECGs, y ∈ {−1, +1}, KTA = ⟨K, yyᵀ⟩_F / (‖K‖_F · n) as specified. *Extra, not in the runbook:* **centered** KTA
(K and yyᵀ double-centered), the standard fix for the offset that class imbalance and a large mean kernel value add to uncentered KTA.
Trained AUC = the same 13 configs as H5.

| k | scale | KTA | centered KTA | mean K_ij | trace distance (H5) | trained AUC (5-fold) |
|---|---|---|---|---|---|---|
| 8 | pi/8 | 0.1268 | 0.3956 | 0.8342 | 0.2934 | – |
| 8 | pi/6 | 0.1634 | 0.3920 | 0.7310 | 0.3742 | 0.9300 |
| 8 | pi/4 | 0.2427 | 0.3747 | 0.5214 | 0.4969 | – |
| 8 | pi/3 | 0.2996 | 0.3425 | 0.3535 | 0.5734 | – |
| 8 | pi/2 | 0.3125 | 0.2588 | 0.1663 | 0.6471 | 0.9284 |
| 8 | 2*pi/3 | 0.2744 | 0.1969 | 0.0904 | 0.6756 | – |
| 8 | pi | 0.2206 | 0.1452 | 0.0492 | 0.6237 | 0.9103 |
| 16 | pi/8 | 0.1593 | 0.4066 | 0.7194 | 0.3584 | – |
| 16 | pi/6 | 0.2113 | 0.3979 | 0.5711 | 0.4431 | 0.9415 |
| 16 | pi/4 | 0.2966 | 0.3584 | 0.3261 | 0.5506 | – |
| 16 | pi/3 | 0.3231 | 0.2965 | 0.1774 | 0.5995 | – |
| 16 | pi/2 | 0.2616 | 0.1791 | 0.0557 | 0.6063 | 0.9233 |
| 16 | 2*pi/3 | 0.1822 | 0.1142 | 0.0215 | 0.5628 | – |
| 16 | pi | 0.0991 | 0.0644 | 0.0078 | 0.4749 | 0.8922 |
| 32 | pi/8 | 0.1889 | 0.3741 | 0.5379 | 0.4129 | 0.9468 |
| 32 | pi/6 | 0.2470 | 0.3513 | 0.3518 | 0.4916 | 0.9530 |
| 32 | pi/4 | 0.3016 | 0.2702 | 0.1280 | 0.5504 | 0.9429 |
| 32 | pi/3 | 0.2595 | 0.1778 | 0.0436 | 0.5240 | 0.9331 |
| 32 | pi/2 | 0.1020 | 0.0646 | 0.0078 | 0.4072 | 0.9103 |
| 32 | 2*pi/3 | 0.0512 | 0.0377 | 0.0044 | 0.3375 | 0.8502 |
| 32 | pi | 0.0410 | 0.0321 | 0.0040 | 0.3052 | 0.7910 |

| Spearman vs trained AUC | KTA | centered KTA | trace distance |
|---|---|---|---|
| all 13 configs | +0.61 (p=0.027) | +0.86 (p=0.000) | +0.34 (p=0.263) |
| k=32 only (7) | +0.71 (p=0.071) | +0.96 (p=0.000) | +0.71 (p=0.071) |

**Reading. KTA beats trace distance**: Spearman +0.61 (p = 0.03) across all 13 configs vs +0.34 (p = 0.26), and the two tie within k=32 (+0.71).
Centered KTA is stronger still (+0.86 overall, +0.96 within k=32). It captures the mechanism found in the classical study: at large scales the kernel
concentrates (mean K_ij → 1/256 ≈ 0.004 at k=32 pi), so every pair of ECGs looks equally dissimilar. Neither score picks the trained optimum exactly.
Uncentered KTA peaks at pi/4, and centered KTA (like H5's purity) rises monotonically as the scale shrinks, so it would pick pi/8 (trained 0.947)
over pi/6 (0.953). **Use centered KTA to rule out over-large scales before training; the final pick still needs validation AUC.** Centered KTA is an
addition to the runbook, chosen before seeing its correlation, but it is still one extra metric and should be read with that in mind.
