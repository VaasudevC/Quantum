# ECG VQC: angle-scale study (PTB-XL, MI vs NORM)

An 8-qubit, 4-layer data-re-uploading variational quantum classifier (VQC) for myocardial infarction (MI) vs normal 12-lead ECGs, built on
249 pre-extracted PTB-XL features. Each feature is encoded as the angle `s·tanh(z)`; the study is about the choice of the scale s.

## Status (2026-11-14) — current conclusions

No quantum-advantage claim and no parameter-efficiency claim. All models are classically simulable.

**Encoding rule (full data, k = 32):** Angle saturation at s = π causes re-uploading collapse (AUC 0.791). s = π/6 fixes it (0.951). Replicated in the brick-wall (Qiskit, FakeFez).

| model, k = 32 features | test AUC | source |
|---|---|---|
| VQC, s = π (original) | 0.791 | results/stage2_steps1000.jsonl |
| VQC, s = π/2 | 0.910 | results/stage2_steps1000.jsonl |
| logistic regression, same features | 0.945 | results/stage2_steps1000.jsonl |
| **VQC, s = π/6** | **0.951** | results/stage2_steps1000.jsonl |
| VQC, zz readout (π/6) | 0.953 | results/n1_zz.jsonl |
| **VQC, trained per-feature scales** | **0.955** | results/n2_trainscale.jsonl |
| degree-2 polynomial LR, same features | **0.958** | results/stage4b_poly.jsonl |
| XGBoost (best classical at every k) | — | results/p1_classical.jsonl |

**Feature-map screen (QB1):** 1,200 settings screened; none passes the expressivity gate (g ≥ √N/2 only at s = π/2 where label alignment collapses). See [`QB.md`](QB.md).

**Non-linearity (QB3):** rank-2 factorization machine and sparse degree-2 LR (≤ VQC parameters, same bounded inputs) match the trained-scale VQC. The win over LR = mild non-linearity, not quantum. See [`QB.md`](QB.md).

**Small-data benchmark (k = 8, pre-registered):** VQC non-inferior to best classical at every N (25–500); superior at none. SVM-RBF within ±0.005 at every N (exploratory). **k = 4 secondary endpoint: in progress.** See [`SMALLDATA.md`](SMALLDATA.md).

**Other benefit (QB5, 18/30 units):** no perturbation/shift/imbalance/subtype setting gives quantum ≥ +0.005 over best classical. See [`QB.md`](QB.md).

**Hardware readiness (simulated):** ~180 shots/ECG keeps π/6 within 0.01 AUC of noiseless; under IBM Heron (FakeFez) noise AUC is unchanged (0.961 → 0.960). Real-device run not yet done; see `HARDWARE.md`.

**Not claimed:** quantum advantage, parameter efficiency, real-device validation, TabPFN comparison.

## Draft paper

[`paper/PAPER_DRAFT.md`](paper/PAPER_DRAFT.md) — first draft (pending k = 4 results). Claims table: [`paper/CLAIMS_CHECK.md`](paper/CLAIMS_CHECK.md).

## Documentation

| document | what is in it |
|---|---|
| [`RESULTS.md`](RESULTS.md) | Phase A classical study: scale sweep, training length, selectors × seeds, shuffle control, official split, degree-2 comparison |
| [`QB.md`](QB.md) | QB pre-registered benchmark: headline, feature-map screen, non-linearity, robustness/shift/imbalance |
| [`SMALLDATA.md`](SMALLDATA.md) | small-data benchmark: pre-registered plan, deviations, results at k=8 (final) and k=4 (pending) |
| [`HARDWARE.md`](HARDWARE.md) | Phase B hardware readiness: Qiskit vs PennyLane, circuit equivalence, shot noise, compilation, IBM noise model, encoding score |
| [`NEXT.md`](NEXT.md) | Phase C improvements: zz readout, trained input scales, kernel-target alignment, clinical metrics |
| [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) | chronological log: every stage, decision, deviation, problem, open items |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | every command in order, with run times and outputs |
| [`docs/CODE_AND_DATA.md`](docs/CODE_AND_DATA.md) | file inventory, code changes and new interfaces, data, result-file formats |
| [`CLOUD_RUNBOOK.md`](CLOUD_RUNBOOK.md), [`docs/runbooks/`](docs/runbooks/) | runbooks for all three phases |

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
