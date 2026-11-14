# ECG VQC: quantum classifiers for myocardial infarction detection

Variational quantum classifiers (VQCs) for **MI vs normal** 12-lead ECGs on [PTB-XL](https://physionet.org/content/ptb-xl/), built on 249 pre-extracted ST/T/QRS features. The study covers the encoding design, a full classical benchmark, a pre-registered small-data benchmark, a search for any quantum benefit, and a run on real IBM hardware.

> **Read [`FINDINGS.md`](FINDINGS.md) first.** It is the single-file summary of the whole project.

## Key results

| | |
|---|---|
| **Encoding rule** | Angle `s·tanh(z)` with s = π collapses as features are re-uploaded (k = 32: AUC 0.791). **s = π/6 fixes it: AUC rises 0.928 → 0.939 → 0.951 for k = 8, 16, 32.** |
| **Best VQC** | Trained per-feature scales, k = 32: **AUC 0.955** (5-fold patient-grouped CV) |
| **Classical comparison** | The VQC **beats plain (linear) logistic regression** (0.951 vs 0.945; trained-scale 0.955). A **degree-2 logistic regression** scores 0.958 and XGBoost 0.965, so against stronger classical models there is no quantum advantage. The edge over the linear baseline is mild non-linearity, which a classical quadratic model reproduces. |
| **Small data (N = 25 to 500)** | Pre-registered: quantum non-inferior to the best classical model at every size, superior at none. |
| **Search for a quantum benefit** | 1,200 feature maps and 30 robustness/shift/subtype settings: no benefit found. |
| **Real hardware** | `ibm_marrakesh` (Heron r2), 200 ECGs, 1000 shots, no mitigation: AUC **0.956** (angle VQC) and **0.949** (brick-wall QNN). |

**Not claimed:** quantum advantage, parameter efficiency, clinical readiness. All models are classically simulable.

| Test AUC, k = 32 (5-fold CV) | |
|---|---|
| VQC, s = π (original) | 0.791 |
| VQC, s = π/2 | 0.910 |
| Logistic regression, same features | 0.945 |
| VQC, s = π/6 | 0.951 |
| VQC, trained per-feature scales | 0.955 |
| Degree-2 logistic regression | 0.958 |
| XGBoost | 0.965 |

## Documentation

| Document | What is in it |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | **Whole-project summary**: results, limitations, open items |
| [`RESULTS.md`](RESULTS.md) | Phase A: scale sweep, training length, selectors, shuffle control, degree-2 comparison, official split |
| [`NEXT.md`](NEXT.md) | Phase C: zz readout, trained scales, kernel-target alignment, clinical metrics |
| [`BRICKWALL.md`](BRICKWALL.md) | Phase D: brick-wall QNN reimplementation and full benchmark vs tuned classical models |
| [`SMALLDATA.md`](SMALLDATA.md) | Phase E: pre-registered small-data benchmark (k = 8 and k = 4) |
| [`QB.md`](QB.md) | Phase F: pre-registered search for a quantum benefit |
| [`DISTINGUISHER.md`](DISTINGUISHER.md) | Kernel-geometry diagnostic |
| [`HARDWARE.md`](HARDWARE.md) | Hardware readiness (simulated) and the real-device run (section H6) |
| [`COLAB_HW.md`](COLAB_HW.md) | Step-by-step Colab notebook for the IBM run |
| [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) | Paper outline with each claim mapped to its file and commit |
| [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) | Chronological log: decisions, deviations, problems |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | Every command in order, with run times |
| [`docs/CODE_AND_DATA.md`](docs/CODE_AND_DATA.md) | File inventory and result-file formats |
| [`docs/runbooks/`](docs/runbooks/) | The runbooks and prompts the work followed |
| [`legacy_phase0/`](legacy_phase0/README.md) | The early Runs 1 to 7 (81.26% accuracy), archival |

## Quick start

```bash
pip install -r requirements.txt                       # classical study and simulator
pip install -r requirements-hw.txt                    # Qiskit, Aer, IBM Runtime, PennyLane
python -c "import Vqc_fast8; Vqc_fast8.verify()"      # adjoint gradients vs parameter-shift check
python -u scale_cv.py --scales "pi/6" --ks 32 --folds 0 --steps 50 --jobs 1 --out results/smoke.jsonl
python summarize.py results/stage2_steps1000.jsonl
python -u hw_run_ibm.py --local --n 10                # hardware pipeline dry run on a fake IBM device
```

Real-device runs need an IBM Quantum token in the `QISKIT_IBM_TOKEN` environment variable. **Never commit a token.** `.gitignore` already excludes common secret filenames.

## Repository layout

```
Vqc_fast8.py          exact 8-qubit simulator with adjoint gradients
Vqc_angle8.py         original parameter-shift implementation (used by Vqc_fast8.verify())
Vqc_hybrid8.py        original hybrid model (used by Vqc_fast8.verify())
scale_cv.py           experiment driver: CV, selectors, scales, readouts, shuffle control, resumable
summarize.py          aggregate result *.jsonl into tables
qiskit_vqc.py         Qiskit and PennyLane versions of the circuit (verified identical)
bw_torch.py, bw_utils.py   brick-wall QNN
p0_* ... p6_*         brick-wall benchmark phases (sanity, classical, verification, grid, external, noise)
sd_*.py               small-data benchmark
qb*.py                search for a quantum benefit
hw_*.py               hardware readiness and IBM runs
hw_bundle.zip         everything the Colab hardware run needs
data/                 feature table (st_features_plus_TR.csv.gz) and patient/fold metadata
results/              one JSONL line per trained model, tables, logs, figures
results/hw/           hardware-stage outputs, including raw ibm_marrakesh data
docs/                 project log, reproduction guide, code reference, runbooks
legacy_phase0/        early Runs 1 to 7 (archival)
```

## Data and licence

PTB-XL is distributed under CC BY 4.0 (Wagner et al., *Scientific Data* 2020). `data/` holds only derived features and metadata. The raw PTB-XL records are not included; download them from PhysioNet. `brickwall/` holds working material from the collaborator whose paper defines the brick-wall model; please check its terms before redistributing.
