# NEXT — improving the model (N1–N4)

Protocol unless stated: patient-grouped 5-fold CV, l1 selector, 8 qubits × 4 layers, k=32, scale pi/6, 1000 steps, choices on validation AUC.
**Change for all new runs:** the l1 path now uses `liblinear` with `random_state=0`, so the selection is deterministic. Checked against the
logged features: folds 0–3 select the identical ordered list, and fold 4 the same 32 features in a different order. The fold-4 baseline (z readout, fixed pi/6)
was therefore rerun with the new selector (`results/n0_baseline_fold4.jsonl`); folds 0–3 reuse the Stage 2 runs, which this code reproduces exactly.
Reference rows (5-fold test AUC): **VQC linear-Z pi/6 0.951 · LR 0.945 · degree-2 LR 0.958.**

## Headline

**Best model: VQC with trained per-feature input scales (N2), test AUC 0.955 ± 0.005**, up from 0.951 for fixed pi/6 (better in 5/5 folds).
The zz readout (N1) gives 0.953 (also better in 5/5 folds). **Neither beats degree-2 LR (0.958), which wins every fold against both.** N2 closes about half of the
0.007 gap and N1 about a third. Both VQCs stay clearly above linear LR (+0.010 and +0.008). Under shot noise the zz gain disappears: at 180 shots zz scores
0.936 vs z 0.939, because pairwise correlators are noisier. **Best training-free encoding score: kernel-target alignment** (Spearman +0.61 with trained AUC vs
+0.34 for H5's trace distance; centered KTA +0.86). It reliably flags over-large scales but does not pick the exact optimum. Clinically (N4), the best VQC
reaches 86.7% sensitivity at 90% specificity and 79.6% at 95% (degree-2 LR: 88.4% / 81.6%). Inferior MI is the hardest subtype for every model
(AUC 0.943 vs 0.957 for anterior MI).

| model (k=32, 5-fold grouped CV) | test AUC | Δ vs fixed pi/6 | Δ vs degree-2 LR | Sens @ 90% spec | Sens @ 95% spec |
|---|---|---|---|---|---|
| LR (same features) | 0.945 | −0.006 | −0.013 | 0.852 | 0.783 |
| VQC linear-Z, fixed pi/6 | 0.951 | – | −0.007 | 0.856 | 0.791 |
| VQC zz readout, pi/6 (N1) | 0.953 | +0.002 (5/5) | −0.005 (0/5) | 0.863 | 0.797 |
| **VQC trained scales (N2)** | **0.955** | **+0.004 (5/5)** | **−0.004 (0/5)** | **0.867** | 0.796 |
| degree-2 LR (same features) | **0.958** | +0.007 | – | **0.884** | **0.816** |

## N1 — ZZ readout, k=32, pi/6, 1000 steps, 5 folds (2026-11-05)

`scale_cv.py --scales pi/6 --ks 32 --steps 1000 --readout zz --out results/n1_zz.jsonl`. Linear readout on 8 ⟨Z_j⟩ + 28 ⟨Z_iZ_j⟩ (72 + 37 = 109 params).
All 36 observables come from the same Z-basis shots. Finite-shot check: `python -u n1_shots.py` → `results/n1_shots.json` (multinomial sampling as in H2,
10 repeats, all test ECGs).

| fold | VQC z, fixed pi/6 | **VQC zz** | LR | degree-2 LR |
|---|---|---|---|---|
| 0 | 0.9598 | 0.9612 | 0.9577 | 0.9655 |
| 1 | 0.9492 | 0.9528 | 0.9424 | 0.9575 |
| 2 | 0.9469 | 0.9512 | 0.9379 | 0.9554 |
| 3 | 0.9452 | 0.9469 | 0.9391 | 0.9541 |
| 4 | 0.9541 | 0.9550 | 0.9483 | 0.9589 |
| **mean (val AUC)** | 0.951 (0.953) | **0.953 (0.955)** | 0.945 | 0.958 |

| shots per ECG | z readout | zz readout |
|---|---|---|
| exact | 0.9510 | 0.9534 |
| 1000 | 0.9488 | 0.9503 |
| 180 | **0.9394** | 0.9361 |

**Reading.** Pairwise correlations help a little and consistently: +0.002, better in 5/5 folds (sign test p = 0.06), and the best checkpoint is no longer at the
final step (last_step_frac 0.0 vs 0.4). They close only about a third of the gap to degree-2 LR, which still wins all 5 folds (−0.005).
**Under realistic shot counts the gain disappears**: at 180 shots zz is 0.003 *below* z, because 36 noisier estimates feed a larger readout. It recovers only
around 1000 shots (+0.0015). On hardware, zz is not worth it below about 1000 shots.

## N2 — trainable per-feature input scale, k=32, z readout, 1000 steps, 5 folds (2026-11-05)

`scale_cv.py --scales pi/6 --ks 32 --steps 1000 --train-scale --out results/n2_trainscale.jsonl`. angle_f = s_f · tanh(z_f), s_f initialised at pi/6,
trained jointly with Adam (lr 0.01). Gradient dL/ds_f = Σ_b dL/dangle_bf · tanh(z_bf), where dL/dangle comes from the new per-sample adjoint
`Vqc_fast8.backward_full()`. **Finite-difference check before running:** max|error| 8.6e-11 (z) and 6.7e-11 (zz) over θ, final-layer θ,
readout weights and two scales. The per-sample angle gradients sum exactly to the θ gradient (0.0 difference). Learned scales: `results/n2_scales.json`.

| fold | VQC z, fixed pi/6 | **VQC trained s** | LR | degree-2 LR |
|---|---|---|---|---|
| 0 | 0.9598 | 0.9631 | 0.9577 | 0.9655 |
| 1 | 0.9492 | 0.9527 | 0.9424 | 0.9575 |
| 2 | 0.9469 | 0.9525 | 0.9379 | 0.9554 |
| 3 | 0.9452 | 0.9499 | 0.9391 | 0.9541 |
| 4 | 0.9541 | 0.9551 | 0.9483 | 0.9589 |
| **mean (val AUC)** | 0.951 (0.953) | **0.955 (0.957)** | 0.945 | 0.958 |

Learned scales (5 folds × 32 = 160): mean |s| 0.58 (init 0.52), median s 0.46, range −1.19 to +2.69; 28 turned negative (a sign flip is a reflection of
that feature's angle). **Mean |s| by re-uploading round: 0.69 (features 1–8), 0.57, 0.53, 0.54.** First-round features (the strongest l1 picks) are
widened, while later rounds stay near pi/6. 22/160 shrink below pi/12, which effectively prunes those features (mostly aVR/aVF ST-shape features:
aVR_st_curvature, aVR_st_slope(_cat), aVF_q_duration). The largest are III_q_depth (1.22, all 5 folds), V2_q_duration, II_r_amp, V6/V2_t_inverted
(≈0.75–0.79). Only 2/160 exceed pi/2, the region where saturation and the original collapse begin.

**Reading.** Learning the scales is the best single improvement: +0.004 over fixed pi/6, better in 5/5 folds (p = 0.06), half of the gap to degree-2
LR closed (−0.004, still 0/5 folds). The learned scales support the Stage 1 finding: the optimiser keeps nearly every angle below pi/2, and it spends
its extra range on the first-round features while switching off weak later features.

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

## N4 — clinical metrics (no training) (2026-11-05)

`python -u n4_clinical.py` → `results/n4_clinical.{json,md}`. Test rows of each fold, mean ± SD over 5 folds. Sensitivity at a fixed specificity = the highest
TPR with FPR ≤ 0.10 / 0.05 on the test ROC (this sets the threshold on the test fold, which is standard for reporting but optimistic for deployment).
Calibration slope = logistic regression of y on logit(p), where 1 is ideal and > 1 means predictions are too moderate. Subtype AUC = that MI subtype vs all NORM ECGs of the fold.
The VQCs are trained with class-weighted BCE, so they are calibrated to a 50/50 prior, while MI prevalence is 36%. LR and degree-2 LR use no class weighting.

| model | AUC | Sens @ 90% spec | Sens @ 95% spec | Brier | Cal. slope | AUC AMI | AUC IMI | AUC AMI+IMI | AUC other MI |
|---|---|---|---|---|---|---|---|---|---|
| VQC z, k=8, pi | 0.910 ± 0.009 | 0.733 ± 0.036 | 0.610 ± 0.031 | 0.119 ± 0.008 | 1.051 ± 0.074 | 0.908 ± 0.010 | 0.899 ± 0.013 | 0.953 ± 0.002 | 0.863 ± 0.033 |
| VQC z, k=8, pi/2 | 0.928 ± 0.006 | 0.798 ± 0.012 | 0.688 ± 0.026 | 0.103 ± 0.005 | 1.055 ± 0.042 | 0.923 ± 0.005 | 0.922 ± 0.009 | 0.968 ± 0.003 | 0.864 ± 0.029 |
| VQC z, k=8, pi/6 | 0.928 ± 0.006 | 0.796 ± 0.017 | 0.682 ± 0.021 | 0.104 ± 0.006 | 1.076 ± 0.055 | 0.919 ± 0.006 | 0.925 ± 0.009 | 0.969 ± 0.004 | 0.859 ± 0.027 |
| VQC z, k=16, pi | 0.892 ± 0.015 | 0.684 ± 0.058 | 0.518 ± 0.106 | 0.130 ± 0.011 | 0.990 ± 0.062 | 0.893 ± 0.016 | 0.880 ± 0.022 | 0.928 ± 0.011 | 0.853 ± 0.026 |
| VQC z, k=16, pi/2 | 0.923 ± 0.010 | 0.792 ± 0.024 | 0.671 ± 0.038 | 0.109 ± 0.008 | 1.095 ± 0.079 | 0.927 ± 0.010 | 0.908 ± 0.013 | 0.962 ± 0.005 | 0.878 ± 0.029 |
| VQC z, k=16, pi/6 | 0.939 ± 0.006 | 0.826 ± 0.017 | 0.740 ± 0.018 | 0.095 ± 0.007 | 1.111 ± 0.084 | 0.936 ± 0.006 | 0.931 ± 0.009 | 0.979 ± 0.002 | 0.879 ± 0.018 |
| VQC z, k=32, pi | 0.791 ± 0.078 | 0.398 ± 0.219 | 0.211 ± 0.152 | 0.182 ± 0.035 | 0.930 ± 0.124 | 0.795 ± 0.074 | 0.779 ± 0.088 | 0.822 ± 0.082 | 0.748 ± 0.074 |
| VQC z, k=32, pi/2 | 0.910 ± 0.012 | 0.742 ± 0.038 | 0.581 ± 0.071 | 0.120 ± 0.008 | 1.055 ± 0.083 | 0.917 ± 0.010 | 0.890 ± 0.017 | 0.951 ± 0.010 | 0.881 ± 0.029 |
| VQC z, k=32, pi/6 | 0.951 ± 0.006 | 0.856 ± 0.020 | 0.791 ± 0.020 | 0.085 ± 0.004 | 1.107 ± 0.080 | 0.953 ± 0.007 | 0.938 ± 0.008 | 0.988 ± 0.002 | 0.902 ± 0.016 |
| LR (same k=32 features) | 0.945 ± 0.008 | 0.852 ± 0.021 | 0.783 ± 0.024 | 0.083 ± 0.006 | 0.966 ± 0.090 | 0.945 ± 0.009 | 0.933 ± 0.011 | 0.988 ± 0.002 | 0.872 ± 0.013 |
| degree-2 LR (same k=32 features) | 0.958 ± 0.004 | 0.884 ± 0.014 | 0.816 ± 0.017 | 0.073 ± 0.004 | 1.032 ± 0.037 | 0.960 ± 0.005 | 0.948 ± 0.008 | 0.990 ± 0.001 | 0.901 ± 0.011 |
| VQC zz (N1), k=32, pi/6 | 0.953 ± 0.005 | 0.863 ± 0.011 | 0.797 ± 0.014 | 0.083 ± 0.006 | 1.083 ± 0.059 | 0.954 ± 0.004 | 0.943 ± 0.008 | 0.988 ± 0.001 | 0.900 ± 0.020 |
| VQC trained scale (N2), k=32 | 0.955 ± 0.005 | 0.867 ± 0.016 | 0.796 ± 0.014 | 0.080 ± 0.004 | 1.102 ± 0.092 | 0.957 ± 0.006 | 0.943 ± 0.009 | 0.988 ± 0.002 | 0.901 ± 0.015 |

Subtype sizes (all folds): {'AMI': 1986, 'IMI': 2263, 'AMI+IMI': 989, 'other MI': 216}; each subtype AUC is that subtype vs all NORM ECGs of the test fold.

**Reading.** Ranking by sensitivity at high specificity follows AUC. The best VQC (N2) detects 86.7% of MIs at 90% specificity and 79.6% at 95%,
1.1 and 0.5 points above fixed pi/6 and 1.7 and 1.8 points below degree-2 LR. VQC calibration slopes are 1.08–1.11 (slightly too moderate, partly from the
class weighting), but Brier scores (0.080–0.085) are close to LR's (0.083). **Subtypes:** combined anterior+inferior MI is almost perfectly separated
(AUC 0.988 for every k=32 model). **Inferior MI is the hardest common subtype (0.938–0.948 vs 0.953–0.960 for anterior)**, and the rare
lateral/posterior group is weakest (≈0.90). The pi-scale collapse at k=32 hurts every subtype (AMI 0.795, IMI 0.779).
