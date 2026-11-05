# ECG VQC: angle-scale study (PTB-XL, MI vs NORM)

An 8-qubit, 4-layer data-re-uploading variational quantum classifier (VQC) for myocardial infarction (MI) vs normal 12-lead ECGs, built on
249 pre-extracted PTB-XL features. Each feature is encoded as the angle `s·tanh(z)`; the study is about the choice of the scale s.

## Results at a glance (patient-grouped 5-fold CV, test AUC)

**The angle scale caused the re-uploading collapse.** With s = π about 10% of angles saturate near ±π, and AUC falls as more features are re-uploaded.
With **s = π/6**, AUC *rises* with the number of features (0.928 → 0.940 → 0.951 for k = 8/16/32) and the VQC beats logistic regression on the
same features at every k. A classical **degree-2 polynomial LR still beats every VQC** (0.958).

| model, k = 32 features | test AUC | source |
|---|---|---|
| VQC, s = π (original) | 0.791 | RESULTS Stage 2 |
| VQC, s = π/2 | 0.910 | RESULTS Stage 2 |
| logistic regression, same features | 0.945 | RESULTS |
| **VQC, s = π/6** | **0.951** | RESULTS Stage 2 |
| VQC, zz readout (π/6) | 0.953 | NEXT N1 |
| **VQC, trained per-feature scales** (best VQC) | **0.955** | NEXT N2 |
| degree-2 polynomial LR, same features | **0.958** | RESULTS 4b, NEXT |

**Towards hardware** (simulated; HARDWARE.md):
- About 180 shots per ECG keeps π/6 within 0.01 AUC of noiseless.
- On IBM's heavy-hex chip the CNOT ring needs 16 SWAPs (32 → 80 CZ).
- Under the IBM Heron (FakeFez) noise model AUC is unchanged (0.961 → 0.960), but calibration needs a global rescale.

**Not done yet:** the real-IBM run. The script `hw_run_ibm.py` is ready and dry-run tested; it needs an IBM token and network access to IBM.

## Documentation

| document | what is in it |
|---|---|
| [`RESULTS.md`](RESULTS.md) | Phase A, classical study: scale sweep, training length, selectors × seeds, shuffle control, official split, degree-2 comparison |
| [`HARDWARE.md`](HARDWARE.md) | Phase B, hardware readiness: Qiskit vs PennyLane, circuit equivalence, shot noise, compilation, IBM noise model, encoding score, H4 status |
| [`NEXT.md`](NEXT.md) | Phase C, improvements: zz readout, trained input scales, kernel-target alignment, clinical metrics |
| [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) | chronological log: every stage, decision, deviation from the runbooks, problem encountered, open items |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | every command in order, with run times and outputs |
| [`docs/CODE_AND_DATA.md`](docs/CODE_AND_DATA.md) | file inventory, code changes and new interfaces, data, result-file formats |
| [`CLOUD_RUNBOOK.md`](CLOUD_RUNBOOK.md), [`docs/runbooks/`](docs/runbooks/) | the runbooks the three phases followed |

## Layout
```
Vqc_fast8.py         exact 8-qubit simulator, adjoint gradients (+ backward_full: per-sample input-angle gradients)
Vqc_angle8.py        original parameter-shift implementation (only needed by Vqc_fast8.verify())
Vqc_hybrid8.py       original hybrid model                    (only needed by Vqc_fast8.verify())
scale_cv.py          experiment driver: CV, selectors, scales, --readout z|zz, --train-scale, shuffle control, --jobs, resumable
summarize.py         aggregate result *.jsonl -> table (+ --md for Markdown)
stage4b_poly.py      Stage 4b: VQC vs degree-2 polynomial LR, function-class R^2
make_figs.py         Stage 6 figures
qiskit_vqc.py        Qiskit + PennyLane versions of the circuit (verified identical), 6-model loader, count decoding
hw_h1_verify.py      H0 benchmark + H1 equivalence      hw_h2_shots.py     H2 shot noise
hw_h3_compile.py     H3a compilation                    hw_h3_noisy.py     H3b noisy simulation + readout mitigation
hw_h3_plot.py        H3 table/figure                    hw_h5_encoding.py  H5 encoding score
hw_run_ibm.py        H4 real IBM run (--local dry run, --submit, --retrieve, --analyze)
n1_shots.py          N1 shot noise, zz vs z             n3_kta.py          N3 kernel-target alignment
n4_clinical.py       N4 clinical metrics
data/                st_features_plus_TR.csv.gz (features), ptbxl_database.csv (ecg_id, patient_id, strat_fold)
results/             one JSONL line per trained model (config, AUCs, features, params), tables (*.md), logs
results/hw/          hardware-stage outputs           results/figs/   all figures
reference/           teammate's Oct-2026 report (HTML)
docs/                project log, reproduction guide, code/data reference, runbooks
```

## Quick start
```bash
pip install -r requirements.txt                       # phase A
pip install -r requirements-hw.txt                    # phases B and C (Qiskit, Aer, IBM Runtime, PennyLane)
python -c "import Vqc_fast8; Vqc_fast8.verify()"      # adjoint vs parameter-shift check
python -u scale_cv.py --scales "pi/6" --ks 32 --folds 0 --steps 50 --jobs 1 --out results/smoke.jsonl
python summarize.py results/stage2_steps1000.jsonl
python -u hw_run_ibm.py --local --n 10                # hardware pipeline dry run on a fake IBM device
```

## Pre-session pilot (for reference)

Before this study, with 300 training steps: k=8 π 0.904 / π/2 0.919; k=16 0.864 / 0.908; k=32 0.787 / 0.896 (LR 0.922 / 0.932 / 0.945).
These are in `results/pilot_runs.jsonl` and `results/sweep_partial.jsonl` (k=32, 600 steps, folds 0 and 3).

Data: PTB-XL (CC BY 4.0) derived features. Keep this repo **private**.
