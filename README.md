# ECG VQC — angle-scale experiments (PTB-XL, MI vs NORM)

Minimal repo to continue the angle-scale study on a bigger machine.

**Result (Stages 0-6, see RESULTS.md):** at s = pi/6, test AUC rises with k (0.928 -> 0.940 -> 0.951 for k = 8/16/32) and beats LR on the same features at every k; a degree-2 polynomial LR (0.958) still beats it.

**Original finding:** with `angle = scale·tanh(z)`, the scale π saturates about 10% of the angles near ±π. This explains most of the AUC collapse when more than 8 features are re-uploaded on 8 qubits. Smaller scales fix it.

| k | π | π/2 | π/3 | π/4 | LR, same features |
|---|---|---|---|---|---|
| 8 (300 steps, 5 folds) | 0.904 | 0.919 | – | – | 0.922 |
| 16 (300 steps, 5 folds) | 0.864 | 0.908 | – | – | 0.932 |
| 32 (300 steps, 5 folds) | 0.787 | 0.896 | – | – | 0.945 |
| 32 (600 steps, folds 0 and 3 only) | – | 0.907 | 0.937 | 0.942 | 0.948 |

Values are test AUC with patient-grouped CV and the L1 selector.

## Layout
```
Vqc_fast8.py        exact 8-qubit simulator, adjoint gradients (circuit identical to Run 7)
Vqc_angle8.py       original PSR implementation (only needed by Vqc_fast8.verify())
Vqc_hybrid8.py      original hybrid model     (only needed by Vqc_fast8.verify())
scale_cv.py         experiment driver: CV, selectors, scales, shuffle control, parallel (--jobs)
summarize.py        aggregate *.jsonl results -> table (+ --md for Markdown)
stage4b_poly.py     Stage 4b: VQC vs degree-2 polynomial LR on the same features
make_figs.py        Stage 6 figures -> results/figs/
RESULTS.md          full write-up of Stages 0-6 (headline, tables, limitations)
NEXT.md             N1-N4: zz readout, trained input scales, KTA encoding score, clinical metrics
HARDWARE.md         hardware stages H0-H5 (shots, compilation, IBM noise model, encoding score)
qiskit_vqc.py       Qiskit + PennyLane versions of the circuit (verified identical), model loader
hw_h*.py            hardware-stage scripts (H1 verify, H2 shots, H3 compile/noisy/plot, H5 encoding)
hw_run_ibm.py       H4: real IBM run (needs QISKIT_IBM_TOKEN); --local = dry run on FakeFez
requirements-hw.txt extra packages for the hardware stages
data/               st_features_plus_TR.csv.gz (feature table), ptbxl_database.csv (ecg_id, patient_id, strat_fold only)
results/            pilot_runs.jsonl (30 runs), sweep_partial.jsonl (8 runs); new runs append here
reference/          teammate's Oct-2026 report (HTML) for context
CLOUD_RUNBOOK.md    step-by-step prompts for a Claude cloud session
```

## Quick start
```bash
pip install -r requirements.txt
python -c "import Vqc_fast8; Vqc_fast8.verify()"     # adjoint vs PSR check
python -u scale_cv.py --scales "pi/4,pi" --ks 16 --folds 0 --steps 25 --jobs 2 --out results/smoke.jsonl
python summarize.py results/*.jsonl
```

Data: PTB-XL (CC BY 4.0) derived features. Keep this repo **private**.
