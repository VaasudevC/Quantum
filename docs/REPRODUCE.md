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
| H4 real (superseded by H6, which ran from Colab: see `HARDWARE.md` "How the H6 run was produced") | `QISKIT_IBM_TOKEN=... python -u hw_run_ibm.py --submit` | queue + ~2 QPU min + 10–25 min local | `results/hw/ibm_hw/` |
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


---

## Small-data benchmark (tag `v0.3-smalldata`): SD, DX, QB commands (added 2026-11-14)

Environment: `.venv` only (no torch required). Python 3.14.3; numpy 2.4.6; scikit-learn 1.9.0; xgboost 3.4.1.
CPU: Intel Core i7-13xxx (GenuineIntel, Family 6 Model 186 Stepping 2). OS: Windows 11 10.0.26300.
Lock file: `requirements-lock.txt` (generated `pip freeze` from `.venv`, 2026-11-14).

**Important:** export `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` in the calling shell before any `sd_run.py` or `sd_launch.py` call; setting them inside Python is insufficient (vcomp thread oversubscription, ~80× slowdown).

### SD: prep and run order

```bash
# 0. Prepare per-subset cache (run once; writes C:/bwcache/sd/*.npz)
..\.venv\Scripts\python.exe sd_prep.py

# 1. Primary k=8 benchmark (all N, folds 0-4; use --tag to avoid overwriting)
..\.venv\Scripts\python.exe sd_launch.py ..\.venv\Scripts\python.exe main 8 100,50,200,25 8 --reps 40 --tag _main
..\.venv\Scripts\python.exe sd_launch.py ..\.venv\Scripts\python.exe main 8 500 2 --reps 10 --tag _main

# 2. Secondary k=8 seed-offset runs (seedoff=1 for seed robustness)
..\.venv\Scripts\python.exe sd_launch.py ..\.venv\Scripts\python.exe main 8 50,200,25 8 --reps 40 --tag _s2 --seedoff 1
..\.venv\Scripts\python.exe sd_launch.py ..\.venv\Scripts\python.exe main 8 500 2 --reps 10 --tag _s2 --seedoff 1

# 3. Secondary k=4 benchmark (850 units, complete 2026-11-14)
..\.venv\Scripts\python.exe sd_launch.py ..\.venv\Scripts\python.exe main 4 100,50,200,25 6 --reps 40 --tag _k4
..\.venv\Scripts\python.exe sd_launch.py ..\.venv\Scripts\python.exe main 4 500 2 --reps 10 --tag _k4

# 4. Analyse (run after each sub-phase)
..\.venv\Scripts\python.exe sd_analyze.py 8 25,50,100,200,500 _main   # -> results/sd_analysis_k8_main.json
..\.venv\Scripts\python.exe sd_analyze.py 4 25,50,100,200,500 _k4     # -> results/sd_analysis_k4_main.json  [COMPLETE 2026-11-14]
..\.venv\Scripts\python.exe sd_equivalence.py                          # -> results/sd_equivalence_k8.md
```

### DX: distinguisher

```bash
..\.venv\Scripts\python.exe dx_x2_geometry.py   # -> results/dx_x2_geometry.json (geometric diff VQC vs RBF)
```

### QB commands (in order; all from `cloud_repo`)

```bash
# QB1: feature-map screen
..\.venv\Scripts\python.exe qb1_screen.py 0   # fold 0  -> results/qb1_screen.jsonl
..\.venv\Scripts\python.exe qb1_screen.py 1   # fold 1
..\.venv\Scripts\python.exe qb1_screen.py 2   # fold 2
..\.venv\Scripts\python.exe qb1_gate.py       # gate + table -> results/qb1_gate.json, results/qb1_table.md

# QB3: non-linearity (run on folds 0-2)
..\.venv\Scripts\python.exe qb3_nonlinearity.py 0
..\.venv\Scripts\python.exe qb3_nonlinearity.py 1
..\.venv\Scripts\python.exe qb3_nonlinearity.py 2
# (results/qb3.jsonl, results/qb3_f{0,1,2}.log)

# QB4: placement screen then training (screen: folds 0-2; training: fold 0 only stopped under 6-h rule)
..\.venv\Scripts\python.exe qb4_placement.py 0  # screen fold 0 -> results/qb4_screen.jsonl
..\.venv\Scripts\python.exe qb4_placement.py 1
..\.venv\Scripts\python.exe qb4_placement.py 2
# training (optional, ~5-7 h idle): sh qb4_run.sh 0 | 1 | 2

# QB5: robustness / shift / imbalance
..\.venv\Scripts\python.exe qb5_robust.py 0   # fold 0 -> results/qb5.jsonl  [COMPLETE]
..\.venv\Scripts\python.exe qb5_robust.py 1   # fold 1  [COMPLETE 2026-11-14]
..\.venv\Scripts\python.exe qb5_robust.py 2   # fold 2  [COMPLETE 2026-11-14]
..\.venv\Scripts\python.exe qb5_analyze.py    # final table, apply candidate rule -> results/qb5_analysis.json

# VQC seed extension (k=32, seed=3, 5 more folds; appends to stage3_selectors.jsonl)
..\.venv\Scripts\python.exe scale_cv.py --scales pi/6 --ks 32 --selectors l1 --seeds 3 --steps 1000 --jobs 2 --out results/stage3_selectors.jsonl
```

### Analysis after all runs complete (Phase 3 of overnight prompt)

```bash
..\.venv\Scripts\python.exe sd_analyze.py 4 25,50,100,200,500 _main
..\.venv\Scripts\python.exe make_paper.py
..\.venv\Scripts\python.exe qb5_analyze.py
```

---

## Brick-wall study (tag `v0.2-brickwall`): environment and workarounds

Two Python 3.14.3 environments are used (Windows 11, 16 logical / 10 physical cores). **Do not merge them**: Windows Application Control (WDAC / Smart App Control)
blocks some native DLLs, and which ones differs per location.

| env | location | use | key versions |
|---|---|---|---|
| `.venv` | `LimrickQuantumWork\.venv` (inside OneDrive) | all sklearn/xgboost work, feature extraction (wfdb), angle VQC (`Vqc_fast8`), P1, P4 prep/classical/vqc, P5 prep/classical/vqc | numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.9.0, xgboost 3.4.1, wfdb 4.3.1, matplotlib 3.11.0, pypdf 7.0.1 |
| `C:\bwvenv` | **outside OneDrive** | brick-wall simulator (torch), PennyLane/Qiskit checks, hardware prep | numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, scikit-learn 1.9.1, torch 2.15.0+cpu, pennylane 0.43.3, pennylane_lightning 0.44.0, qiskit 2.6.2, qiskit-aer 0.18.2, qiskit-ibm-runtime 0.52.0, networkx 3.7, matplotlib 3.11.2 |

Create `C:\bwvenv`:
```
python -m venv C:\bwvenv
C:\bwvenv\Scripts\python.exe -m pip install numpy scipy scikit-learn pandas xgboost matplotlib wfdb pypdf pennylane==0.43.3 qiskit qiskit-aer qiskit-ibm-runtime networkx
C:\bwvenv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```
Problems and workarounds:
* `import torch` in the OneDrive `.venv` fails with *"An Application Control policy has blocked this file … torch_global_deps.dll"* (WinError 4551). The same wheel works in `C:\bwvenv`.
* In `C:\bwvenv`, some sklearn / pandas extension modules (`_argkmin`, `_loss`, `pandas._libs.ops_dispatch`) are blocked intermittently (the same import may succeed later).
  Therefore **torch workers import only numpy + torch** (`bw_torch.py`, `p3_brickwall.py`, `p4_bw.py`, `p5_lc.py bw`): AUC and the logistic warm start are implemented in numpy/torch and the data arrays are read from `C:\bwcache`.
* `C:\bwcache` (not in git, rebuilt in minutes, keeps big arrays out of OneDrive): `cd cloud_repo && ..\.venv\Scripts\python.exe cache_data.py` (CV folds, standardised top-32 columns, `l1_f{0-4}.npz`, `cohen_dedup_f{0-4}.npz`);
  `..\.venv\Scripts\python.exe p4_external.py prep` (`full.npz`, PTB-XL + PTB); `..\.venv\Scripts\python.exe p5_lc.py prep` (`lc\*.npz`, learning-curve subsamples).
* Only ~1 GB of RAM was free during the run, so at most 3–5 workers were used (`OMP/MKL/OPENBLAS_NUM_THREADS=1`).
* **PTB-XL feature table:** `data/st_features_plus_TR.csv.gz` has ST columns built with a 0.5–40 Hz band-pass, not the 0.05 Hz `diagnostic_filter` that `extract_st_features.py` has had since 2026-07-31.
  `ptb_external/extract_ptb_features.py` patches this in (verified: all 249 columns of 50 records reproduced to 8.5e-14, `ptb_external/xl_check.json`).

Order of commands (all from `cloud_repo`; `.venv` = `..\.venv\Scripts\python.exe`, `bw` = `C:\bwvenv\Scripts\python.exe`):
```
.venv p0_sanity.py ; .venv cache_data.py ; .venv p1_classical.py --jobs 4          # P0, P1
bw p2_verify.py ; bw p2_xor_facts.py                                                # P2
bw p3_brickwall.py --stage A|B|C|ref --jobs 4 ; .venv p3_vqc_scores.py ; bw p3_analyze.py   # P3
.venv ptb_external/extract_ptb_features.py --mode xl-check|ptb ; .venv p4_external.py prep|classical|vqc ; bw p4_bw.py ; .venv p4_report.py   # P4
.venv p5_lc.py prep|classical|vqc ; bw p5_lc.py bw                                  # P5
bw hw_bw.py ; bw p6_compile.py ; (noise / IBM scripts, see BRICKWALL.md)            # P6
```
