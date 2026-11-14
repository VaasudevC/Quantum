# Phase 0: early VQC runs (archival)

The first set of experiments, before the angle-scale study. An 8-qubit VQC on 251 PTB-XL features, parameter-shift gradients, single 80/20 split (train 11,972 / test 2,993). The code is kept as a record; **it is not wired to the current `data/` layout and is not meant to be re-run as is**. Paths in the scripts and logs were replaced with `<project_root>`.

| Run | Encoding | Readout | Test accuracy | AUC | Log |
|---|---|---|---|---|---|
| 1 to 3 | Amplitude | Linear | 78 to 79% | 0.852 | `results/vqc_amp8_run.txt` |
| 4 | Amplitude + log scaling | Linear | 79.45% | 0.869 | `results/vqc_scale.txt` |
| 5 | Angle, 8 features | Linear | 77.18% | 0.800 | `results/vqc_angle8.txt` |
| 6 | Angle, 32 sequential | Linear | 80.45% | 0.870 | `results/vqc_angle32.txt` |
| **7** | Angle, 32 sequential | MLP | **81.26%** | **0.885** | `results/vqc_hybrid8.txt` |
| 8a to 8e | PLS-8 / PLS-32 | MLP | did not converge | n/a | `results/vqc_pls8.txt`, `vqc_pls32.txt` |

`fastA` to `fastD` in `results/` are partial ablation logs from the adjoint-gradient simulator; those runs stopped at step 975 of 1500 and have no test numbers.

The 83.36% figure (threshold 0.665) was tuned on the test set and is optimistic.

## Contents

| Path | What it is |
|---|---|
| `Vqc_amp8.py`, `Train_amp8.py` | amplitude-encoding VQC (Runs 1 to 4) |
| `Train_angle8.py`, `Train_hybrid8.py` | angle and hybrid training (Runs 5 to 7); the models themselves are `../Vqc_angle8.py` and `../Vqc_hybrid8.py` |
| `Vqc_pls8.py`, `Train_pls8.py`, `Train_pls32.py` | PLS-encoded runs (Run 8) |
| `feature_extraction/` | scripts that built the ST, T/R/Q feature table from the raw waveforms |
| `TEST7_FEATURE_SELECTION.md`, `TR_feature_spec.md` | Run 7 feature methodology and the T/R/Q feature definitions |
| `results/` | training logs and the Run 6 and Run 7 parameter files |

For what these runs taught us and how Phase A fixed the encoding, see [`../FINDINGS.md`](../FINDINGS.md), sections 3 and 4.
