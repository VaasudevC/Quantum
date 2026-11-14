# Code, data and result files

## 1. File inventory

"Original" = present in the initial commit (`86f8984`, `33168af`). "Added" = written during the 4–5 Nov session.

### Code
| file | origin | purpose |
|---|---|---|
| `Vqc_fast8.py` | original, **extended** | exact 8-qubit state-vector simulator with batched adjoint gradients. Added `backward_full()` (§2.1) |
| `Vqc_angle8.py`, `Vqc_hybrid8.py` | original, unchanged | reference parameter-shift implementations, used only by `Vqc_fast8.verify()` |
| `scale_cv.py` | original, **extended** | experiment driver: CV splits, selectors, training, resumable JSONL output. Added `--readout`, `--train-scale`, deterministic l1 (§2.2) |
| `summarize.py` | original, **fixed** | aggregates result JSONL into tables. Now groups by readout/train_scale; `last_step_frac` fixed (§2.3) |
| `stage4b_poly.py` | added | Stage 4b: degree-2 LR vs VQC, and the function-class R² of the VQC logit |
| `make_figs.py` | added | Stage 6 figures (a)–(d) |
| `results/run_3_4_5.sh` | added | the chained Stage 3 → 4 → 5 commands as run |
| `qiskit_vqc.py` | added | Qiskit circuit (`build_circuit`, `build_template`) and PennyLane QNode (`pennylane_qnode`) identical to `Vqc_fast8`; the 6-model loader; angle rebuild; count decoding (§2.4) |
| `hw_h1_verify.py` | added | H0 benchmark + H1 equivalence checks |
| `hw_h2_shots.py` | added | H2 finite-shot study (multinomial sampling of exact probabilities) |
| `hw_h3_compile.py` | added | H3a transpilation to FakeFez / FakeTorino / FakeSherbrooke / IonQ-like; gate counts, SWAPs, estimated success |
| `hw_h3_noisy.py` | added | H3b noisy simulation (Aer + FakeFez noise model), best-line search, readout calibration and mitigation |
| `hw_h3_plot.py` | added | H3 table and figure |
| `hw_run_ibm.py` | added | H4 real-hardware run: `--local` dry run, `--submit`, `--retrieve`, `--analyze` |
| `hw_h5_encoding.py` | added | H5 trace distance, fidelity and purity of class-mean encoded states |
| `n1_shots.py` | added | N1 finite-shot AUC, zz vs z readout |
| `n3_kta.py` | added | N3 kernel-target alignment (uncentered and centered) |
| `n4_clinical.py` | added | N4 sensitivity at fixed specificity, Brier score, calibration slope, AUC by MI subtype |

### Documents
| file | content |
|---|---|
| `README.md` | overview, headline numbers, layout, quick start |
| `RESULTS.md` | phase A write-up (Stages 0–6, 4b) |
| `HARDWARE.md` | phase B write-up (H0–H5) |
| `NEXT.md` | phase C write-up (N1–N4) |
| `docs/PROJECT_LOG.md` | chronological log, decisions, deviations, problems, open items |
| `docs/REPRODUCE.md` | every command, in order, with run times |
| `docs/CODE_AND_DATA.md` | this file |
| `docs/runbooks/CLOUD_RUNBOOK.md`, `docs/runbooks/HARDWARE_RUNBOOK.md`, `docs/runbooks/NEXT_RUNBOOK.md` | the three runbooks that were followed, unchanged |
| `requirements.txt`, `requirements-hw.txt` | dependencies (phase A; phase B/C extras) |

## 2. Code changes and new interfaces

### 2.1 `Vqc_fast8.py`: `backward_full()` (new; existing functions unchanged)
```python
grad_theta, grad_angles = backward_full(psi, g, angles, theta, readout)
# psi: (B,256) final states from forward(); g: (B, n_obs) = dL/d<O_j> per sample (include 1/B and class weights)
# grad_theta: (72,), identical to backward(); grad_angles: (B,32) = dL/d angle[b, f]
```
The encoding gate on qubit i in layer L is `Ry(angle[b, 8L+i] + theta[16L+i])`, so dL/dangle[b, 8L+i] is the per-sample term of the θ[16L+i]
gradient. `_ry_grad_ps()` is the per-sample version of `_ry_grad()`. Checks: θ gradient equals `backward()` (6.7e-16), per-sample terms sum
exactly to it, and finite differences agree to below 1e-10.

### 2.2 `scale_cv.py`
- `l1_order`: `LogisticRegression(..., solver="liblinear", random_state=0)`, so selection is deterministic. Before this, near-tied features could swap order
  between processes; the selected sets were unaffected in every check.
- `class Linear(k, scale, readout="z", train_scale=False)` replaces the old parameter-less `Linear`:
  - inputs are `T = tanh(z)` padded to 32 columns, and `angles(T, s) = s * T` on the first k columns;
  - `readout="z"`: 8 ⟨Z_j⟩; `"zz"`: 8 ⟨Z_j⟩ + 28 ⟨Z_iZ_j⟩ (both from `V.READOUT_SIGNS`, i.e. the same Z-basis probabilities);
  - params = [θ (72), w (8 or 36), b (1)] plus, with `train_scale`, s (k values initialised at `scale`);
  - `split(p) -> (theta, w, b, s)`, `proba(T, p)`, `loss_grad(p, T, y, cw)`; with `train_scale`,
    dL/ds_f = Σ_b dL/dangle_bf · T_bf from `backward_full`.
- `train(m, ...)` now takes the model object. The initialization order (θ, w, b) is unchanged, so default runs reproduce old results exactly (verified:
  fold-0 k=32 π/6 test AUC 0.95975870 before and after).
- New CLI flags `--readout {z,zz}` and `--train-scale`. They are stored in each result row and included in the resume key; older rows default to `z` / `False`.
- `frac_angle_gt_0p9pi` is computed from the final (possibly learned) angles.

### 2.3 `summarize.py`
- Groups by `readout` and `train_scale` as well (defaults for older rows). Markdown tables gained those two columns.
- `last_step_frac` = fraction of runs whose best checkpoint is the **final training step** (`best_step == steps`). It previously compared with the
  group's largest `best_step`. No reported number or decision changed (re-checked; see PROJECT_LOG §7.7).

### 2.4 `qiskit_vqc.py` (key functions)
| function | does |
|---|---|
| `load_models(path, fold=0)` | the 6 hardware models `{(k, scale): row}` from `results/stage2_steps1000.jsonl` |
| `model_data(row)` | angles (N, 32), labels, train/val/test indices, rebuilt exactly as in `scale_cv.py` from the row's logged features |
| `split_params(row)` | θ (72), w (8), b |
| `build_circuit(angles, theta)` / `build_template(theta, k)` | concrete circuit, or parametrised circuit with fixed θ and k angle Parameters (transpile once, bind per ECG) |
| `statevector_bigendian(qc)` | Qiskit statevector reordered to Vqc_fast8's convention |
| `z_from_counts(counts)` | ⟨Z_j⟩ from little-endian Qiskit counts |
| `stratified_subset(y_te, n, seed=0)` | fixed class-stratified subset of test positions (the H3 300 / H4 100 ECGs) |
| `pennylane_qnode(theta, device, shots)` | PennyLane version (wire 0 = MSB), supports broadcasting |

**Bit order.** Vqc_fast8 is big-endian (qubit 0 = most significant bit of the 256-dim index); Qiskit is little-endian. Qubit labels are identical
everywhere, Vqc qubit j = Qiskit qubit j = classical bit j, and only the boundary is converted: `Statevector(qc).reverse_qargs()` and
`bitstring[::-1][j]`.

## 3. Data

| file | content |
|---|---|
| `data/st_features_plus_TR.csv.gz` | 14,965 PTB-XL ECGs (5,454 MI, 9,511 NORM); `ecg_id`, `label_class`, `label_subclass`, age, sex and 249 ECG-derived numeric features |
| `data/ptbxl_database.csv` | `ecg_id`, `patient_id`, `strat_fold` (official PTB-XL folds 1–10) |

`label_subclass` (used in N4): AMI 1,986 · IMI 2,263 · AMI+IMI 989 · other (LMI / PMI combinations) 216 · NORM 9,511.
PTB-XL is CC BY 4.0. Keep the repository private, as the README asks.

## 4. Result files

### 4.1 `scale_cv.py` rows (`results/stage*.jsonl`, `pilot_runs.jsonl`, `sweep_partial.jsonl`, `n0_*.jsonl`, `n1_zz.jsonl`, `n2_trainscale.jsonl`)
One JSON object per trained model:

| field | meaning |
|---|---|
| `split`, `fold`, `selector`, `k`, `scale`, `steps`, `batch`, `lr`, `seed`, `shuffle_later`, `readout`, `train_scale` | configuration (pilot rows lack the later fields; defaults: grouped5, l1, 300 steps, seed 0, False, z, False) |
| `test_auc` | test-fold AUC at the best-validation checkpoint (**the reported metric**) |
| `val_auc`, `best_step` | best validation AUC (checked every 25 steps) and its step (**used for all choices**) |
| `lr_auc` | test AUC of `LogisticRegression(max_iter=3000)` on the same selected (and shuffled, if any) features |
| `frac_angle_gt_0p9pi` | share of training angles with \|angle\| > 0.9π (the saturation measure) |
| `features` | the exact ordered feature list (feature j goes to qubit j mod 8, layer j div 8) |
| `params` | trained parameters: θ (72), w (8 or 36), b, [+ k learned scales] |
| `secs` | wall time of the run |

### 4.2 Phase A derived
- `results/stage4b_poly.jsonl`: per fold, `vqc_test_auc`, `lr1_*`, `poly2_*`, `poly2_ang_*` (`val_auc`, `test_auc`, `C`), `spearman_*` (vs VQC logit), and `r2_deg1` / `r2_deg2`.
- `results/*.md`: `summarize.py` tables.

### 4.3 Phase B (`results/hw/`)
| file | content |
|---|---|
| `h1_verify.json` | statevector / ⟨Z⟩ differences, per-model AUC reproduction, benchmark seconds |
| `h2_shots.jsonl` | per model × shots: `auc_mean`, `auc_sd`, `aucs` (10 repeats), `noiseless`, `loss`; `h2_summary.json`: ‖w‖, class gaps, shots needed |
| `h3_compile.json` | per target: gate counts by type, n1q/n2q, depth, 2q depth, SWAPs, success estimate, transpiler seed |
| `h3_noisy_{line8,best12}.jsonl` | per model: AUC `noiseless` / `shots_only` / `noisy_raw` / `noisy_mitigated`, `corr_P_*`, `z_shrink_raw`, `n2q`, `physical_qubits`, readout errors `e0`/`e1` |
| `h3_z_{line8,best12}_k*_*.npz` | per ECG: `idx` (row index into the feature table), `y`, `z_exact`, `z_raw`, `z_mit` (8 ⟨Z⟩ each) |
| `h3_calibration.json` | balanced accuracy and Brier score: exact / raw / mitigated / rescaled |
| `ibm_local/` (dry run), `ibm_hw/` (planned real run, never used), `ibm_bw_hw/` (real run, 2026-11-14) | `meta.json` (backend, calibration time, job ids, circuit stats), `ecgs_*.npz`, `counts_*.json` (one counts dict per PUB; **the last two PUBs are readout calibration, all-0 then all-1**), `aer_counts_*.json`, `analysis.json` |
| `h5_encoding.json` | per (k, scale): trace distance, fidelity, purity, trained AUC; Spearman values; post-hoc purity correlation |
| `*.log` | console output of the long runs |

### 4.4 Phase C (`results/`)
| file | content |
|---|---|
| `n1_shots.json` | z vs zz AUC: exact, 1000 and 180 shots (mean over folds) |
| `n2_scales.json` | learned scales per fold with their features; mean \|s\| by re-uploading round |
| `n3_kta.json` | per (k, scale): KTA, centered KTA, mean off-diagonal kernel, trace distance, trained AUC; Spearman values |
| `n4_clinical.json` | per model: mean/SD over folds of AUC, sensitivity @ 90/95% specificity, Brier, calibration slope, subtype AUCs |
| `n1_n2_summary.md` | `summarize.py` table of baseline / zz / trained-scale runs |

### 4.5 Figures (`results/figs/`, 150 dpi)
`a_auc_vs_scale_k32.png`, `b_auc_vs_k.png`, `c_shuffle_control.png`, `d_poly2_vs_vqc.png` (phase A); `hw_shots.png`, `hw_noisy.png`,
`encoding_score.png` (phase B); `kta_vs_auc.png` (phase C).
