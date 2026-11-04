# RESULTS — ECG VQC angle-scale study (PTB-XL, MI vs NORM)

**Chosen scale s\* = pi/6** (Stage 1, highest mean validation AUC at k=32, 600 steps).

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
| pi/2   | 5 | 0.9114 | 0.909 ± 0.009 | 0.945 | −0.037 | 0.6 |
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
| 16 | pi/6 | 0.941 | 0.940 ± 0.006 | 0.932 | +0.007 | 0.4 | – |
| 16 | pi/2 | 0.926 | 0.923 ± 0.010 | 0.932 | −0.009 | 0.4 | 0.908 |
| 16 | pi   | 0.891 | 0.892 ± 0.015 | 0.932 | −0.040 | 0.4 | 0.864 |
| 32 | pi/6 | 0.952 | 0.951 ± 0.006 | 0.945 | +0.006 | 0.6 | – |
| 32 | pi/2 | 0.911 | 0.910 ± 0.012 | 0.945 | −0.035 | 0.4 | 0.896 |
| 32 | pi   | 0.789 | 0.791 ± 0.078 | 0.945 | −0.154 | 0.2 | 0.787 |

**Key question: at s\* = pi/6, does AUC fall, stay flat or rise from k=8 to k=32? It RISES:** 0.928 → 0.940 → 0.951.
k=32 beats k=8 in 5/5 folds (mean +0.023). Re-uploading extra features now helps, whereas at pi it hurts (0.910 → 0.791).
pi/6 beats pi in 5/5 folds at every k, and beats pi/2 in 5/5 folds at k=16 (+0.016) and k=32 (+0.041); at k=8 the two are tied (−0.000).
Going from 300 to 1000 steps helps pi/2 (about +0.01) but does not rescue pi at k=32.

**Decision.** last_step_frac for pi/6 at k=32 is 0.6 > 0.5, so pi/6 is rerun at 2000 steps (`results/stage2_steps2000.jsonl`) to set STEPS\*.
