# Reproducing every result

Every command below was run in this order in the session. Run times are wall-clock on the session machine (4 vCPUs, 15 GB RAM, no GPU).
All `scale_cv.py` runs are **resumable**: rerun the same command with the same `--out` and finished configs are skipped. Each finished run is
appended as one JSON line, so an interrupted job loses at most the runs still in progress.

## 0. Environment

```bash
pip install -r requirements.txt                       # phase A (numpy, scipy, pandas, scikit-learn, matplotlib)
pip install -r requirements-hw.txt                    # phase B/C extras (qiskit, qiskit-aer, qiskit-ibm-runtime, networkx, pennylane)
```

Versions used: Python 3.13.18; numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1, matplotlib 3.11.2, qiskit 2.6.1,
qiskit-aer 0.18.2, qiskit-ibm-runtime 0.51.0, pennylane 0.45.1 (+ lightning), networkx 3.7.

Notes
- `scale_cv.py` sets `OMP_NUM_THREADS=1` (and the OpenBLAS/MKL equivalents) so the `--jobs` worker processes do not oversubscribe the CPUs.
  Any script that imports `scale_cv` inherits this.
- Data: `data/st_features_plus_TR.csv.gz` (feature table) and `data/ptbxl_database.csv` (`ecg_id`, `patient_id`, `strat_fold`).
  `ECG_DATA_DIR` overrides the location.
- Feature selection: from 5 Nov the l1 selector is deterministic (`random_state=0`). Phase A/B runs were made before that fix: rerunning them
  reproduces the same feature *sets*, but near-tied features can come out in a different order (see PROJECT_LOG §7.1). Every result row stores
  its exact `features` list and trained `params`, and all downstream scripts (4b, H1–H5, N1 shots, N4) use those stored values.

## Phase A: classical study (RESULTS.md)

| step | command | time | output |
|---|---|---|---|
| Stage 0 verify | `python -c "import Vqc_fast8; Vqc_fast8.verify()"` | 5 s | diffs ~1e-16 |
| Stage 0 timing | `python -u scale_cv.py --scales pi/4 --ks 32 --folds 0 --steps 50 --jobs 1 --out results/smoke.jsonl` | 20 s | `smoke.jsonl` (git-ignored) |
| Stage 0 sanity | `python summarize.py results/pilot_runs.jsonl results/sweep_partial.jsonl` | 2 s | k=32 300 steps: π 0.787, π/2 0.896 |
| Stage 1 | `cp results/sweep_partial.jsonl results/stage1_sweep.jsonl` then `python -u scale_cv.py --scales "pi/8,pi/6,pi/4,pi/3,pi/2,2*pi/3,pi" --ks 32 --steps 600 --jobs 3 --out results/stage1_sweep.jsonl` | 35 min | `stage1_sweep.jsonl` |
| Stage 2 | `python -u scale_cv.py --scales "pi/6,pi/2,pi" --ks 8,16,32 --steps 1000 --jobs 3 --out results/stage2_steps1000.jsonl` | 100 min | `stage2_steps1000.jsonl` |
| Stage 2b | `python -u scale_cv.py --scales "pi/6" --ks 8,16,32 --steps 2000 --jobs 3 --out results/stage2_steps2000.jsonl` | 65 min | `stage2_steps2000.jsonl` |
| Stage 4b | `python -u stage4b_poly.py --vqc results/stage4_real.jsonl --scale pi/6 --k 32 --steps 1000 --out results/stage4b_poly.jsonl` | 2 min | `stage4b_poly.jsonl` |
| Stages 3, 4, 5 | `results/run_3_4_5.sh` (see below) | 6 h | `stage3_selectors.jsonl`, `stage4_{real,shuffled}.jsonl`, `stage5_official.jsonl` |
| Stage 6 figures | `python make_figs.py --sstar pi/6 --steps 1000` | 10 s | `results/figs/{a,b,c,d}_*.png` |

Stage 3/4 seeding (identical configs reused, so they are skipped on resume):
```bash
grep '"scale": "pi/6"' results/stage2_steps1000.jsonl > results/stage3_selectors.jsonl
grep -E '"k": (16|32), "scale": "(pi/6|pi)"' results/stage2_steps1000.jsonl > results/stage4_real.jsonl
```
`results/run_3_4_5.sh` then runs, in order:
```bash
python -u scale_cv.py --scales "pi/6" --ks 8,16,32 --selectors l1,cohen_dedup,random --seeds 0,1,2 --steps 1000 --jobs 3 --out results/stage3_selectors.jsonl
python -u scale_cv.py --scales "pi/6,pi" --ks 16,32 --steps 1000 --jobs 3 --out results/stage4_real.jsonl
python -u scale_cv.py --scales "pi/6,pi" --ks 16,32 --steps 1000 --jobs 3 --shuffle-later --out results/stage4_shuffled.jsonl
python -u scale_cv.py --split official --scales "pi/6,pi/2,pi" --ks 8,32 --steps 1000 --jobs 3 --out results/stage5_official.jsonl
```

Tables: `python summarize.py <files> --md results/<stage>.md`. The regenerated tables are `results/stage1_sweep.md`,
`stage2_steps1000.md` (with the 300-step pilot), `stage2_steps2000.md`, `stage3_selectors.md`, `stage4.md`, `stage5_official.md` and
`n1_n2_summary.md`. Decision-rule statistics (validation AUC per scale, fold sign tests, change vs k=8) were computed inline with pandas and scipy, and their
values are in RESULTS.md.

## Phase B: hardware readiness (HARDWARE.md)

| step | command | time | output |
|---|---|---|---|
| H0+H1 | `python -u hw_h1_verify.py` (benchmark only: `--bench`) | 2 min | `results/hw/h1_verify.json` |
| H2 | `python -u hw_h2_shots.py` | 50 s | `results/hw/h2_shots.{jsonl,md}`, `h2_summary.json`, `results/figs/hw_shots.png` |
| H3a | `python -u hw_h3_compile.py` | 5 s | `results/hw/h3_compile.{json,md}` |
| H3b (300 ECGs, 8-qubit line) | `python -u hw_h3_noisy.py` | 3 min | `results/hw/h3_noisy_line8.jsonl`, `h3_z_line8_*.npz` |
| H3b check (best compile) | `python -u hw_h3_noisy.py --best12 --n 100 --models "32:pi/6,32:pi"` | 23 min | `results/hw/h3_noisy_best12.jsonl`, `h3_z_best12_*.npz` |
| H3 calibration analysis | inline (balanced accuracy and Brier, raw / mitigated / rescaled, from the `h3_z_line8_*.npz`) | – | `results/hw/h3_calibration.json` |
| H3 table + figure | `python hw_h3_plot.py` | 5 s | `results/hw/h3_noisy.md`, `results/figs/hw_noisy.png` |
| H4 dry run | `python -u hw_run_ibm.py --local --n 10` | 5 min | `results/hw/ibm_local/` |
| H4 real (not run) | `QISKIT_IBM_TOKEN=... python -u hw_run_ibm.py --submit` | queue + ~2 QPU min + 10–25 min local | `results/hw/ibm_hw/` |
| H4 re-fetch / analyze | `python -u hw_run_ibm.py --retrieve` · `python -u hw_run_ibm.py --analyze --tag hw` | – | `analysis.json` |
| H5 | `python -u hw_h5_encoding.py` | 3 min | `results/hw/h5_encoding.{json,md}`, `results/figs/encoding_score.png` |

H4 needs network access to `quantum.cloud.ibm.com`, `iam.cloud.ibm.com` and the regional `*.quantum-computing.cloud.ibm.com` hosts.
The Claude cloud environment blocked them, so allow them in the environment's network settings or run from a laptop.

## Phase C: model improvements (NEXT.md)

| step | command | time | output |
|---|---|---|---|
| N1 zz readout | `python -u scale_cv.py --scales pi/6 --ks 32 --steps 1000 --readout zz --jobs 2 --out results/n1_zz.jsonl` | 22 min | `n1_zz.jsonl` |
| N2 trained scales | `python -u scale_cv.py --scales pi/6 --ks 32 --steps 1000 --train-scale --jobs 2 --out results/n2_trainscale.jsonl` | 22 min | `n2_trainscale.jsonl` |
| fold-4 baseline (deterministic l1) | `python -u scale_cv.py --scales pi/6 --ks 32 --steps 1000 --folds 4 --jobs 1 --out results/n0_baseline_fold4.jsonl` | 6 min | `n0_baseline_fold4.jsonl` |
| N1 shots | `python -u n1_shots.py` | 2 min | `n1_shots.json` |
| N2 learned scales | inline (|s| by layer, extremes) | – | `n2_scales.json` |
| N3 KTA | `python -u n3_kta.py` | 30 s | `n3_kta.{json,md}`, `results/figs/kta_vs_auc.png` (uses `results/hw/h5_encoding.json`) |
| N4 clinical | `python -u n4_clinical.py` | 3 min | `n4_clinical.{json,md}` |

N1 and N2 ran at the same time with 2 workers each. The gradient checks done before N1/N2 (finite differences on 5 parameters per readout, per-sample
angle gradients summing to the θ gradient, and an old run's AUC reproduced through the refactored model) are described in NEXT.md N2. They can be repeated with
the snippet below:

```python
import numpy as np, scale_cv as S
rng = np.random.default_rng(1); B = 40
T = np.tanh(rng.normal(size=(B, 32))); y = rng.integers(0, 2, B); cw = np.array([1.0, 1.3])
for ro in ("z", "zz"):
    m = S.Linear(32, np.pi / 6, ro, train_scale=True); p = m.init(rng); _, g = m.loss_grad(p, T, y, cw)
    for j in (3, 70, len(p) - 27, len(p) - 1):
        e = np.zeros(len(p)); e[j] = 1e-6
        fd = (m.loss_grad(p + e, T, y, cw)[0] - m.loss_grad(p - e, T, y, cw)[0]) / 2e-6
        print(ro, j, abs(fd - g[j]))          # expect < 1e-9
```
