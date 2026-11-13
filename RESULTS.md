# RESULTS — ECG VQC angle-scale study (PTB-XL, MI vs NORM)

> Part of a three-phase study: [`RESULTS.md`](RESULTS.md) (classical) · [`HARDWARE.md`](HARDWARE.md) (hardware readiness) · [`NEXT.md`](NEXT.md) (improvements). Chronological log, decisions and problems: [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md); commands: [`docs/REPRODUCE.md`](docs/REPRODUCE.md); code and file formats: [`docs/CODE_AND_DATA.md`](docs/CODE_AND_DATA.md).

**Chosen scale s\* = pi/6** (Stage 1) · **STEPS\* = 1000** (Stage 2b) · all choices made on validation AUC; test AUC reported.

## Headline

The 8-qubit re-uploading VQC encodes each feature as an angle `s·tanh(z)`. At the original scale s = pi, about 10% of the angles
are pushed to ±pi, and the classifier collapses as more features are re-uploaded (k=32: test AUC 0.79). **Shrinking the scale to
s = pi/6 removes the collapse.** AUC now **rises** with the number of re-uploaded features: **0.928 (k=8) → 0.939 (k=16) → 0.951 (k=32)**.
k=32 beats k=8 in 5/5 folds, and pi/6 beats pi in 5/5 folds at every k (+0.16 at k=32). At pi/6 the VQC is **ahead of logistic regression on the same
features by +0.006–0.007 at every k** (k=32: 0.951 vs 0.945, 5/5 folds), and 2000 steps give 0.953. A shuffle control shows the
later re-uploading rounds carry information (+0.023 over k=8, while shuffled later features cost only −0.006). A degree-2 polynomial LR
on the same 32 features scores 0.958–0.960 and beats the VQC in 5/5 folds. A quadratic function of the input angles explains 98.6% of the
VQC's learned logit, so the VQC learns within the degree-2 class but does not reach its optimum.
Selector choice: l1 > cohen_dedup by 0.005 at k=32 (competitive), and both beat random features by 0.04–0.13. On the official
PTB-XL split (train folds 1–8, val fold 9, test fold 10), pi/6 gives 0.942 at k=32 vs LR 0.938, with the same ranking pi/6 > pi/2 > pi.

| key number (test AUC, 5-fold grouped CV) | k=8 | k=16 | k=32 |
|---|---|---|---|
| VQC, s=pi (1000 steps)   | 0.910 | 0.892 | 0.791 |
| VQC, s=pi/2 (1000 steps) | 0.928 | 0.923 | 0.910 |
| **VQC, s=pi/6 (1000 steps)** | **0.928** | **0.939** | **0.951** |
| LR, same features        | 0.922 | 0.932 | 0.945 |
| degree-2 LR, same features | – | – | 0.958 |

Figures (`results/figs/`, 150 dpi): `a_auc_vs_scale_k32.png` (test AUC vs scale, k=32, LR dashed);
`b_auc_vs_k.png` (AUC vs k for pi, pi/2, pi/6); `c_shuffle_control.png` (shuffle drop, plus the change vs k=8 for real and shuffled features);
`d_poly2_vs_vqc.png` (per-fold linear LR vs VQC vs degree-2 LR). Regenerate with `python make_figs.py --sstar pi/6 --steps 1000`.

---

## Stage 0 — setup and verification (2026-11-04)

- Python 3.13.18, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1; 4 CPUs, 15 GB RAM → **JOBS = 3**.
- Data: 14,965 rows, 249 ECG-only features after the join (5,454 MI).
- `Vqc_fast8.verify()`: state diff 2.1e-16, <Z> diff 7.5e-16, gradient diff 2.2e-17 (pass).
- Timing: about 225 s per 600-step k=32 run with 3 parallel workers (≈37 s per 100 steps).
- Sanity: pilot k=32 l1 300 steps reproduces pi 0.787, pi/2 0.896.

## Stage 1 — scale sweep, k=32, 600 steps, 5 folds (2026-11-04)

Command: `scale_cv.py --scales "pi/8,pi/6,pi/4,pi/3,pi/2,2*pi/3,pi" --ks 32 --steps 600 --jobs 3 --out results/stage1_sweep.jsonl`
(8 runs from `sweep_partial.jsonl` were reused.)

| scale | n | mean val AUC | test AUC ± SD | LR AUC | gap to LR | last_step_frac |
|---|---|---|---|---|---|---|
| pi/8   | 5 | 0.9483 | 0.947 ± 0.006 | 0.945 | +0.002 | 0.6 |
| **pi/6** | 5 | **0.9502** | **0.950 ± 0.006** | 0.945 | **+0.005** | 0.6 |
| pi/4   | 5 | 0.9448 | 0.943 ± 0.007 | 0.945 | −0.002 | 0.8 |
| pi/3   | 5 | 0.9360 | 0.933 ± 0.007 | 0.945 | −0.012 | 0.2 |
| pi/2   | 5 | 0.9114 | 0.908 ± 0.009 | 0.945 | −0.037 | 0.6 |
| 2*pi/3 | 5 | 0.8495 | 0.850 ± 0.020 | 0.945 | −0.095 | 0.2 |
| pi     | 5 | 0.7960 | 0.794 ± 0.072 | 0.945 | −0.151 | 0.4 |

Fold-level sign tests on test AUC: pi/6 beats pi/2 on 5/5 folds (mean +0.041, p=0.062);
pi/6 beats pi on 5/5 folds (mean +0.156, p=0.062). At pi/6 the VQC matches or beats LR on the same 32 features in 5/5 folds.

**Interpretation.** AUC falls monotonically as the scale grows past pi/6, so the k=32 collapse is an angle-scale effect.
Compressing angles to |θ| < pi/6 recovers about 0.16 AUC compared with pi, and the VQC ends up level with logistic regression
(+0.005). Validation AUC peaks at pi/6, and pi/8 is close (−0.002).

**Decision.** s\* = pi/6. It is not the smallest scale tested, so no extension to pi/12 or pi/16 is needed. pi/8 is within 0.003 val AUC, but
the rule prefers the larger scale, which is pi/6 in any case. Next: Stage 2 with `--scales "pi/6,pi/2,pi"`.

## Stage 2 — training length, 1000 steps, k ∈ {8,16,32}, 5 folds (2026-11-04)

Command: `scale_cv.py --scales "pi/6,pi/2,pi" --ks 8,16,32 --steps 1000 --jobs 3 --out results/stage2_steps1000.jsonl`

| k | scale | val AUC | test AUC ± SD | LR AUC | gap to LR | last_step_frac | 300-step pilot |
|---|---|---|---|---|---|---|---|
| 8  | pi/6 | 0.929 | 0.928 ± 0.006 | 0.922 | +0.006 | 0.4 | – |
| 8  | pi/2 | 0.931 | 0.928 ± 0.006 | 0.922 | +0.006 | 0.2 | 0.919 |
| 8  | pi   | 0.910 | 0.910 ± 0.009 | 0.922 | −0.012 | 0.2 | 0.904 |
| 16 | pi/6 | 0.941 | 0.939 ± 0.006 | 0.932 | +0.007 | 0.4 | – |
| 16 | pi/2 | 0.926 | 0.923 ± 0.010 | 0.932 | −0.009 | 0.4 | 0.908 |
| 16 | pi   | 0.891 | 0.892 ± 0.015 | 0.932 | −0.040 | 0.4 | 0.864 |
| 32 | pi/6 | 0.952 | 0.951 ± 0.006 | 0.945 | +0.006 | 0.6 | – |
| 32 | pi/2 | 0.911 | 0.910 ± 0.012 | 0.945 | −0.035 | 0.4 | 0.896 |
| 32 | pi   | 0.789 | 0.791 ± 0.078 | 0.945 | −0.154 | 0.2 | 0.787 |

**Key question: at s\* = pi/6, does AUC fall, stay flat or rise from k=8 to k=32? It RISES:** 0.928 → 0.939 → 0.951.
k=32 beats k=8 in 5/5 folds (mean +0.023). Re-uploading extra features now helps, whereas at pi it hurts (0.910 → 0.791).
pi/6 beats pi in 5/5 folds at every k, and beats pi/2 in 5/5 folds at k=16 (+0.016) and k=32 (+0.041); at k=8 the two are tied (−0.000).
Going from 300 to 1000 steps helps pi/2 (about +0.01) but does not rescue pi at k=32.

**Decision.** last_step_frac for pi/6 at k=32 is 0.6 > 0.5, so pi/6 is rerun at 2000 steps (`results/stage2_steps2000.jsonl`) to set STEPS\*.

### Stage 2b — pi/6 at 2000 steps (2026-11-04)

| k | val AUC 1000 → 2000 | Δ val | test AUC 1000 → 2000 | LR | last_step_frac @2000 |
|---|---|---|---|---|---|
| 8  | 0.9291 → 0.9315 | +0.0024 | 0.928 → 0.930 ± 0.006 | 0.922 | 0.2 |
| 16 | 0.9411 → 0.9440 | +0.0029 | 0.940 → 0.942 ± 0.005 | 0.932 | 0.2 |
| 32 | 0.9519 → 0.9550 | +0.0030 | 0.951 → 0.953 ± 0.005 | 0.945 | 0.2 |

Doubling the steps gives a small, consistent gain (test AUC improves in 5/5 folds at every k), and the rise with k is unchanged.
The mean val-AUC gain over the rerun is **+0.0028 < 0.003**. Only k=32 alone reaches the threshold, and just barely (+0.00301).
**Decision: STEPS\* = 1000.** This also keeps Stage 3 at 5 folds; at 2000 steps it would take about 10 h on 4 CPUs and need to be cut to 3 folds.
The 2000-step k=32 figure (0.953, +0.008 over LR) is the best VQC number so far.

## Stage 3 — selectors × seeds at s\*=pi/6, 1000 steps, 5 folds × 3 seeds (2026-11-05)

Command: `scale_cv.py --scales "pi/6" --ks 8,16,32 --selectors l1,cohen_dedup,random --seeds 0,1,2 --steps 1000 --jobs 3 --out results/stage3_selectors.jsonl`
(the 15 l1/seed-0 runs come from Stage 2.) Means are over 15 runs (5 folds × 3 seeds). Seed SD is the SD of the three per-seed means.

| selector | k | VQC test AUC | fold SD | seed SD | LR AUC | VQC − LR |
|---|---|---|---|---|---|---|
| l1          | 8  | 0.928 | 0.006 | 0.0002 | 0.922 | +0.006 |
| l1          | 16 | 0.939 | 0.005 | 0.0003 | 0.932 | +0.007 |
| l1          | 32 | **0.951** | 0.005 | 0.0003 | 0.945 | +0.006 |
| cohen_dedup | 8  | 0.917 | 0.007 | 0.0003 | 0.915 | +0.003 |
| cohen_dedup | 16 | 0.934 | 0.006 | 0.0004 | 0.928 | +0.006 |
| cohen_dedup | 32 | 0.947 | 0.005 | 0.0001 | 0.941 | +0.006 |
| random      | 8  | 0.802 | 0.030 | 0.0013 | 0.759 | +0.043 |
| random      | 16 | 0.872 | 0.022 | 0.0031 | 0.836 | +0.036 |
| random      | 32 | 0.912 | 0.021 | 0.0004 | 0.902 | +0.011 |

Selected-vs-random gap (VQC test AUC):

| k | l1 − random | cohen_dedup − random | l1 − cohen_dedup |
|---|---|---|---|
| 8  | +0.127 | +0.116 | +0.011 |
| 16 | +0.068 | +0.063 | +0.005 |
| 32 | +0.039 | +0.034 | +0.005 |

**Interpretation.** Feature selection matters most at small k. At k=8, selected features beat random ones by +0.12–0.13, close to the
teammate's ~+0.10 at pi, and the gap shrinks as k grows because 32 random features carry more signal. l1 beats cohen_dedup in 5/5 folds at every
k, but by only 0.005 at k ≥ 16. The VQC at pi/6 beats LR on the same features for every selector and k, and the margin is largest with random features.
Seeds barely matter (seed SD ≤ 0.003, far below fold SD). Note: in `scale_cv.py` the random selector's feature order depends
on the fold only, so the seeds change only the initialization and minibatches, never the random feature set.

**Decision.** cohen_dedup is within 0.005 of l1 at k=32 (−0.0045): **"Cohen's-d pipeline is competitive"**,
although l1 wins consistently in 5/5 folds.

## Stage 4 — shuffle control, k ∈ {16,32}, 1000 steps, 5 folds (2026-11-05)

Commands: `scale_cv.py --scales "pi/6,pi" --ks 16,32 --steps 1000 [--shuffle-later] --out results/stage4_{real,shuffled}.jsonl`
(all 20 real runs are reused from Stage 2. The shuffled run was interrupted by a worker restart at 9/20 and resumed.)
The k=8 column is the matching Stage 2 run, which has no later features.

| scale | k | real | shuffled 9..k | **drop = real − shuffled** | folds with drop > 0 | real − k=8 | shuffled − k=8 |
|---|---|---|---|---|---|---|---|
| pi/6 | 16 | 0.940 | 0.926 | **+0.014 ± 0.002** | 5/5 | +0.011 | −0.003 |
| pi/6 | 32 | 0.951 | 0.922 | **+0.029 ± 0.004** | 5/5 | +0.023 | −0.006 |
| pi   | 16 | 0.892 | 0.867 | **+0.025 ± 0.016** | 5/5 | −0.018 | −0.043 |
| pi   | 32 | 0.791 | 0.681 | **+0.110 ± 0.058** | 5/5 | −0.119 | −0.229 |

**Rule outcome.** The drop at s\* is *not* ≥ 2× the drop at pi; it is smaller (0.029 vs 0.110 at k=32).

**Interpretation.** At pi, the drop is not mainly a measure of information use. Shuffled later features destroy the circuit
(shuffled − k=8 = −0.229), and real ones only partly undo that damage (real is still −0.119 below k=8). At pi/6
the two effects separate cleanly. Shuffled later features are close to harmless (−0.006 vs k=8), so extra re-uploading rounds of noise cost little,
and real later features add +0.023. The drop of +0.029 is therefore close to the full information content of features 9..32, with low variance (SD 0.004).
In short, at pi/6 later rounds carry information and noise in them is tolerated; at pi later rounds act mostly as damaging noise.
The real − shuffled drop alone cannot tell these two situations apart; the k=8 baseline can.

## Stage 4b — VQC vs degree-2 polynomial LR, k=32, s\*=pi/6, 1000 steps (2026-11-04)

Command: `python -u stage4b_poly.py --vqc results/stage4_real.jsonl --scale pi/6 --k 32 --steps 1000 --out results/stage4b_poly.jsonl`.
Same splits and the same 32 features (each run's logged list) as the VQC. Each polynomial model has 32 + 528 = 560 terms, standardized,
with C ∈ {1e-3 … 1} chosen by **validation** AUC (C = 0.01 in 9/10 fits). The VQC test scores are recomputed from the stored
params and match the logged AUC exactly.

| model | test AUC ± SD | Spearman ρ with VQC logit (test) |
|---|---|---|
| linear LR on z | 0.945 ± 0.008 | 0.945 |
| **VQC, s = pi/6** | **0.951 ± 0.006** | 1 |
| degree-2 LR on z | 0.958 ± 0.004 | 0.949 |
| degree-2 LR on the VQC angles pi/6·tanh(z) | 0.960 ± 0.004 | 0.959 |

How much of the learned VQC logit a fixed function class explains (ridge fit on train rows, R² on test rows):
**degree-1 in the angles: R² = 0.918 ± 0.008; degree-2 in the angles: R² = 0.986 ± 0.001.**

**Interpretation.** The VQC at pi/6 is, to 98.6%, a quadratic function of its encoded angles: mostly linear, plus a small quadratic correction.
This is consistent with small-angle encoding, where each <Z_i> expands as a low-order trigonometric polynomial
dominated by terms up to second order. It beats linear LR in 5/5 folds (+0.006) but falls short of the best
degree-2 LR in 5/5 folds (−0.008). It therefore learns within the degree-2 class but does not reach its optimum
(81 trainable readout and ansatz params vs 560 coefficients). The model whose rankings it tracks most closely is the degree-2 LR on the
same angles (ρ = 0.959).

## Stage 5 — official PTB-XL split, 1000 steps (2026-11-05)

Command: `scale_cv.py --split official --scales "pi/6,pi/2,pi" --ks 8,32 --steps 1000 --jobs 3 --out results/stage5_official.jsonl`
(train strat_fold 1–8, val 9, test 10; one run per config.) z = (official − CV mean) / CV fold SD, using the Stage 2 runs.

| k | scale | official test AUC | val AUC | LR | grouped-CV mean ± SD | z | flag (>2 SD) |
|---|---|---|---|---|---|---|---|
| 8  | pi/6 | 0.918 | 0.929 | 0.913 | 0.928 ± 0.006 | −1.6 | |
| 8  | pi/2 | 0.914 | 0.925 | 0.913 | 0.928 ± 0.006 | −2.2 | **flag** |
| 8  | pi   | 0.895 | 0.901 | 0.913 | 0.910 ± 0.009 | −1.8 | |
| 32 | pi/6 | **0.942** | 0.948 | 0.938 | 0.951 ± 0.006 | −1.5 | |
| 32 | pi/2 | 0.905 | 0.916 | 0.938 | 0.910 ± 0.012 | −0.5 | |
| 32 | pi   | 0.701 | 0.710 | 0.938 | 0.791 ± 0.078 | −1.2 | |

**Interpretation.** Every model scores lower on test fold 10 than its grouped-CV mean, LR included (0.922 → 0.913 at k=8,
0.945 → 0.938 at k=32). The official test fold is therefore somewhat harder than an average CV fold, and the gap is not specific to the VQC. Only pi/2 at k=8
passes the 2-SD flag (z = −2.2). The conclusions carry over: at pi/6 the VQC beats LR at both k (+0.005, +0.004), AUC rises from
k=8 to k=32, and the ranking pi/6 > pi/2 > pi holds at k=32.

## Limitations

- **Single dataset and task**: PTB-XL MI vs NORM, using pre-extracted ST/T/QRS features (no raw-signal model).
- **Exact noiseless simulation**: state-vector simulation with adjoint gradients; no shot noise, gate noise or hardware compilation.
  Small-angle encodings may behave differently under noise, since small rotations are closer to the noise floor.
- **8 qubits × 4 layers only**: k ≤ 32 is capped by 8 qubits × 4 re-uploading rounds; deeper or wider circuits were not tested.
- **Linear readout on 8 ⟨Z⟩** (81 params). Stage 4b shows the learned function is about 98.6% quadratic in the angles, and a classical
  degree-2 LR beats it, so this study gives **no quantum advantage claim**. The VQC is competitive with linear LR and slightly better.
- **Statistics**: 5 outer folds, so a 5/5 sign test can reach at best p = 0.062. Seeds (Stage 3) vary only initialization and minibatches.
- **Reproducibility caveat**: the l1 selector called `liblinear` without `random_state` in these runs, so near-tied features could swap order
  between runs (seen at positions 27/28 in one fold). Selected sets were identical in the cases checked, but the exact
  order can differ. Each run's logged `features` list is the ground truth; `stage4b_poly.py` uses it. Fixed for all
  runs from 5 Nov (`random_state=0`, NEXT.md): folds 0–3 then give the identical ordered list and fold 4 the same set in a different order.
- **Compute**: 4 CPUs; STEPS\* = 1000 was chosen partly so Stage 3 could keep all 5 folds.
