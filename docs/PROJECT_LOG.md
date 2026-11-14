# Project log: ECG VQC angle-scale study (PTB-XL, MI vs NORM)

A detailed, chronological record of everything done in the Claude cloud session of 4–5 Nov 2026: what was run, why, what came
out, every decision and how it was made, every deviation from the runbooks, and every problem encountered. The per-phase write-ups
with full tables are:

| phase | write-up | runbook it followed |
|---|---|---|
| A. Classical simulation study (Stages 0–6 + 4b) | [`RESULTS.md`](../RESULTS.md) | [`CLOUD_RUNBOOK.md`](runbooks/CLOUD_RUNBOOK.md) |
| B. Hardware readiness (H0–H5) | [`HARDWARE.md`](../HARDWARE.md) | [`docs/runbooks/HARDWARE_RUNBOOK.md`](runbooks/HARDWARE_RUNBOOK.md) |
| C. Model improvements (N1–N4) | [`NEXT.md`](../NEXT.md) | [`docs/runbooks/NEXT_RUNBOOK.md`](runbooks/NEXT_RUNBOOK.md) |

*This log covers phases A to C (4 to 5 Nov 2026). Later phases are written up in [`BRICKWALL.md`](../BRICKWALL.md), [`SMALLDATA.md`](../SMALLDATA.md), [`QB.md`](../QB.md) and [`HARDWARE.md`](../HARDWARE.md) section H6; [`FINDINGS.md`](../FINDINGS.md) summarises everything.*

How to reproduce each step: [`docs/REPRODUCE.md`](REPRODUCE.md). Code changes, file inventory and result-file formats:
[`docs/CODE_AND_DATA.md`](CODE_AND_DATA.md).

---

## 1. The question

An 8-qubit, 4-layer data-re-uploading variational quantum classifier (VQC) separates myocardial infarction (MI) from normal (NORM)
12-lead ECGs in PTB-XL, using 249 pre-extracted ST/T/QRS features. Each feature is encoded as a rotation angle `s·tanh(z)`. With the original
scale s = π, AUC *collapsed* when more than 8 features were re-uploaded (k = 32: 0.787). The starting hypothesis (README) was that s = π saturates
about 10% of angles near ±π, and that a smaller scale would fix the collapse. The session tested that hypothesis (phase A), whether the result survives realistic
quantum hardware (phase B), and whether the model can be improved further (phase C).

## 2. Final answers in one table

| question | answer | where |
|---|---|---|
| Does a smaller angle scale fix the collapse? | **Yes.** s\* = π/6 chosen on validation AUC; k=32 test AUC 0.951 vs 0.794 at π | RESULTS Stage 1 |
| Does re-uploading more features now help? | **Yes.** At π/6, AUC rises 0.928 → 0.940 → 0.951 for k = 8/16/32 (5/5 folds) | RESULTS Stage 2 |
| Is the VQC better than classical baselines? | Better than linear LR on the same features (0.945); **worse than degree-2 LR (0.958)** | RESULTS 4b, NEXT |
| What function does it learn? | 98.6% of its logit is a degree-2 function of the input angles | RESULTS 4b |
| Do the later re-uploading rounds carry information? | Yes at π/6 (+0.023 over k=8; shuffling them costs only 0.006); at π they mostly add damage | RESULTS Stage 4 |
| Does it hold on the official PTB-XL split? | Yes: π/6 k=32 0.942 vs LR 0.938, same ranking | RESULTS Stage 5 |
| Does π/6 survive shot noise? | Yes: ≈180 shots per ECG keeps it within 0.01 AUC (π/2 needs ≈410) | HARDWARE H2 |
| Does it survive IBM Heron noise (simulated)? | AUC yes (0.961 → 0.960); calibration no (needs one global rescale) | HARDWARE H3 |
| Real IBM hardware? | **Not run in this session** (no token; network blocks IBM). **Update 2026-11-14: run on `ibm_marrakesh` from Colab, AUC 0.956 (angle VQC) and 0.949 (brick-wall)** | HARDWARE H4, H6 |
| Best model overall | Trained per-feature scales (N2): **0.955**, still below degree-2 LR (0.958) | NEXT N2 |
| Training-free way to choose the scale? | Kernel-target alignment (Spearman 0.61; centered 0.86) rules out bad scales but does not pick the optimum | NEXT N3, HARDWARE H5 |

## 3. Timeline

Times are IST (UTC+5:30). Compute: 4 vCPUs, 15 GB RAM, no GPU.

| finished (IST) | step | commit | one-line result |
|---|---|---|---|
| 4 Nov ~21:15 | Stage 0: setup, verification, timing | – (in Stage 1 commit) | all checks pass; JOBS = 3 |
| 4 Nov 21:56 | Stage 1: scale sweep k=32 | `a4b81a7` | s\* = π/6 |
| 4 Nov 23:39 | Stage 2: 1000 steps, k = 8/16/32 | `7891a4c` | AUC rises with k at π/6 |
| 5 Nov 00:44 | Stage 2b: π/6 at 2000 steps | `36ced6c` | +0.0028 val AUC < 0.003 → STEPS\* = 1000 |
| 5 Nov 00:48 | Stage 4b: degree-2 LR comparison (user's addition) | `86e66a9` | VQC between linear (0.945) and degree-2 LR (0.958) |
| 5 Nov 01:38–03:38 | Stage 3 partial checkpoints | `bd8c231`, `a70f70f`, `02c93e4` | – |
| 5 Nov 04:55 | Stage 3: selectors × seeds | `16c7a67` | cohen_dedup competitive (−0.0045 vs l1) |
| 5 Nov ~05:10 | *worker restart* killed the Stage 4 shuffled run at 9/20; resumed | `97ffc59` | no data lost |
| 5 Nov 05:42 | Stage 4: shuffle control | `72b5dcf` | later rounds informative at π/6 |
| 5 Nov 05:57 | Stages 5 + 6: official split, write-up, figures | `41c1fe5` | same ranking on official split |
| 5 Nov 09:01 | H0 + H1: framework choice, Qiskit equivalence | `d206b4d` | exact match (1e-16), AUC reproduced exactly |
| 5 Nov 09:02 | H2: shot noise | `c553b25` | ≈180 shots for π/6 |
| 5 Nov 09:25 | H3a + checkpoint (scripts for H3b/H4/H5) | `34611f7` | 16 SWAPs on heavy-hex |
| 5 Nov 09:45 | H3: noisy simulation complete | `3f9ec7f` | AUC survives FakeFez noise |
| 5 Nov 09:45 | H5: encoding score | `f17734b` | trace distance weak (ρ = 0.34) |
| 5 Nov 09:50 | H4: script ready, dry run | `dcd6e73` | blocked by token + network |
| 5 Nov 09:50 | Hardware wrap-up | `d9549f9` | HARDWARE.md headline |
| 5 Nov 11:00 | N3 + code for N1/N2 | `6e88c64` | KTA beats trace distance |
| 5 Nov 11:29 | N1, N2, N4 | `25afcb0`, `d908cc2` | trained scales 0.955; degree-2 LR still ahead |
| 5 Nov ~11:45 | Documentation pass (this file, REPRODUCE, CODE_AND_DATA, summarize.py fixes) | see `git log` | – |

---

## 4. Phase A: classical simulation study (CLOUD_RUNBOOK.md, Stages 0–6)

All numbers are test AUC from patient-grouped, class-stratified 5-fold CV (`StratifiedGroupKFold`, seed 0). Within each outer training
fold, 1/8 (patient-grouped) is held out as validation for checkpoint and model choices. Median imputation, z-scoring and feature selection are fitted
on the inner-training part only. LR = `LogisticRegression(max_iter=3000)` on the same selected features. The circuit, readout and CV protocol were not changed in this phase.

### Stage 0: setup and verification
- Installed `requirements.txt`. Python 3.13.18, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1. 4 CPUs, 15 GB RAM.
- Data check: 14,965 ECGs (5,454 MI, 9,511 NORM), 249 ECG-only features after the join with `ptbxl_database.csv`.
- `Vqc_fast8.verify()`: state 2.1e-16, ⟨Z⟩ 7.5e-16, gradient 2.2e-17 vs the original parameter-shift implementation. Pass.
- Timing: 50-step k=32 run = 18.6 s, i.e. about 37 s per 100 steps (about 225 s per 600-step run with 3 parallel workers).
- **JOBS = 3** (nproc − 1); RAM was not a constraint.
- Sanity: the pilot table reproduced (k=32, 300 steps: π 0.787, π/2 0.896).

### Stage 1: scale sweep (k=32, 600 steps, 7 scales × 5 folds)
- 35 configs; the 8 in `sweep_partial.jsonl` were reused via the resume mechanism; 27 were run (about 35 min).
- **Decision rule → s\* = π/6** (highest mean val AUC, 0.9502). π/8 was within 0.003 (0.9483); the rule prefers the larger scale, and π/6
  is larger anyway. π/6 is not the smallest scale tested, so no extension to π/12 or π/16 was needed.
- Test AUC 0.950 vs LR 0.945. π/6 beats π/2 and π in 5/5 folds (sign test p = 0.0625, the minimum possible with 5 folds).

### Stage 2: training length (π/6, π/2, π × k = 8/16/32, 1000 steps)
- 45 runs (about 100 min). **Headline: at π/6, AUC rises with k** (0.928 → 0.940 → 0.951; k=32 > k=8 in 5/5 folds), while at π it collapses
  (0.910 → 0.791). π/6 beats LR at every k.
- last_step_frac for π/6 at k=32 was 0.6 > 0.5 → rule triggered a 2000-step rerun (Stage 2b, 15 runs, about 65 min).
- **Stage 2b decision → STEPS\* = 1000.** Mean val-AUC gain over the 9 rerun configs was +0.0028 < 0.003. k=32 alone reached +0.00301,
  borderline. *Judgment call:* the pooled mean was used, as the rule says "mean val AUC", and 1000 steps also kept Stage 3 at 5 folds, where 2000 steps
  would have taken about 10 h on 4 CPUs and forced the runbook's 3-fold fallback. The 2000-step numbers (k=32: 0.953) are reported as a supplement.

### Stage 4b: degree-2 polynomial LR (requested by the user mid-session)
- `stage4b_poly.py`: for each fold, degree-2 LR on the same 32 features (560 terms, standardized), C ∈ {1e-3…1} chosen on **validation**
  AUC; also degree-2 LR on the VQC's own angles; and ridge fits of the VQC logit on degree-1 and degree-2 angle features.
- Result: linear LR 0.945 < **VQC 0.951** < degree-2 LR 0.958 (0.960 on the angles). The degree-2 LR beats the VQC in 5/5 folds.
  The VQC logit is 91.8% explained by a linear function of the angles and **98.6% by a quadratic one**.
- The script initially asserted that recomputed feature selection equals the logged one. This failed on one fold, which uncovered the
  liblinear non-determinism issue (§7.1). The script was changed to use each run's logged feature list.

### Stage 3: selectors × seeds (π/6, 1000 steps, l1 / cohen_dedup / random × k × 5 folds × 3 seeds)
- 135 configs, of which 15 were reused from Stage 2 (l1, seed 0); 120 run, about 4.5 h. Partial results were committed hourly.
- l1 > cohen_dedup by 0.011 / 0.005 / 0.005 at k = 8/16/32 (5/5 folds each). **Decision: cohen_dedup is within 0.005 of l1 at k=32 (−0.0045)
  → "Cohen's-d pipeline is competitive"**, although l1 wins consistently. Selected minus random: +0.127 (k=8), +0.068, +0.039 (k=32). Seed SD ≤ 0.003.
- Note recorded in RESULTS.md: the random selector's order depends only on the fold, so the 3 seeds change only initialization and minibatches.

### Stage 4: shuffle control (π/6 and π × k = 16/32)
- All 20 "real" runs were reused from Stage 2 (identical configs); 20 shuffled runs (features 9..k permuted) were run (about 45 min, interrupted once, §7.2).
- **The runbook's rule ("drop at s\* ≥ 2× drop at π") failed**: the drop is 0.029 at π/6 vs 0.110 at π (k=32).
- *Added analysis:* comparing both real and shuffled runs with the same-scale k=8 model separates the two effects. At π/6, shuffled later features cost
  only −0.006 and real ones add +0.023, so later rounds carry information and noise is tolerated. At π, shuffled later features cost −0.229, so the
  large "drop" at π mostly measures damage from noise, not information use. Figure (c) shows both views.

### Stage 5: official PTB-XL split (strat_fold 1–8 / 9 / 10)
- 6 runs. All models, LR included, score lower on fold 10 than their CV mean (fold 10 is harder). One case exceeds the 2-SD flag
  (π/2 at k=8, z = −2.2). Ranking unchanged: π/6 k=32 0.942 vs LR 0.938.

### Stage 6: write-up
- `RESULTS.md` made self-contained (headline, key-number table, stage sections, limitations).
- `make_figs.py` → `results/figs/a_auc_vs_scale_k32.png` (log-x to avoid overlapping π/8 and π/6 labels), `b_auc_vs_k.png`,
  `c_shuffle_control.png` (two panels, see Stage 4) and `d_poly2_vs_vqc.png`.

---

## 5. Phase B: hardware readiness (HARDWARE_RUNBOOK.md, H0–H5)

The 6 models are fold 0, l1, 1000 steps, k ∈ {8, 32} × {π, π/2, π/6}, using the **stored parameters from Stage 2. Nothing was retrained.** No IBM token was set
during the session.

### H0: Qiskit vs PennyLane (added at the user's request; the mentor recommended PennyLane)
- Both installed (qiskit 2.6.1, qiskit-aer 0.18.2, qiskit-ibm-runtime 0.51.0, pennylane 0.45.1) and **timed on the real workload**
  (k=32 π/6 circuit, 1000 test ECGs). Exact ⟨Z⟩: Vqc_fast8 0.13 s, PennyLane default.qubit (broadcast) 0.32 s, Aer Estimator 0.93 s,
  PennyLane lightning.qubit 4.4 s. 1000 shots: Aer Sampler 3.1 s, PennyLane lightning 7.6 s.
- **Decision:** our numpy simulator for H2/H5 (fastest); **Qiskit for H3/H4**, because transpiling to IBM heavy-hex, calibration-based noise
  models and IBM Runtime are native there, whereas PennyLane would call Qiskit underneath through `pennylane-qiskit`. The PennyLane QNode is kept in
  `qiskit_vqc.py`, verified identical, for the mentor, for future training, and for non-IBM hardware.

### H1: circuit equivalence
- `qiskit_vqc.build_circuit` / `build_template` reproduce `Vqc_fast8.forward` exactly. Statevector max|diff| = 1.5e-16 over 20 random inputs; the
  PennyLane ⟨Z⟩ differs by 5.9e-16; the 6 models' test AUCs are reproduced with |diff| = 0.
- Bit ordering: Vqc_fast8 is big-endian (qubit 0 = MSB), Qiskit little-endian. Labels are kept identical and converted only at the boundary
  (`Statevector.reverse_qargs()`, `bitstring[::-1][j]`).
- Problem hit: Aer's raw `parameter_binds` cannot bind `x + constant` parameter expressions → the benchmark uses Aer `EstimatorV2`/`SamplerV2`.

### H2: finite shots (own simulator, multinomial sampling)
- All 2,993 fold-0 test ECGs, 10 repeats, shots ∈ {30, 100, 300, 1000, 2000, 4000, 10000}. 30 and 2000 were **added** to the runbook grid.
- π/6 at k=32 needs ≈180 shots for ≤ 0.01 loss (π/2 ≈410). The fear that small angles are more fragile is not borne out at k=32: the trained model has a larger
  ⟨Z⟩ class gap (0.32 vs 0.23) and a smaller readout norm (8.1 vs 10.7). At k=8, π/6 is slightly more shot-sensitive than π/2.

### H3a: compilation
- optimization_level=3, 5 seeds, keep the fewest 2q gates. Heron r2 (FakeFez): 80 CZ (32 CX + **16 SWAPs**, since heavy-hex has no 8-cycle),
  12 physical qubits, depth 205, estimated success 0.705. Torino 0.578, Eagle/Sherbrooke 0.481. IonQ-like all-to-all: 32 RXX, no SWAPs; its success
  figure (0.743) uses **assumed** fidelities and is labelled as such.

### H3b: noisy simulation (FakeFez noise model) + readout mitigation
- **Deviation (speed):** the best 12-qubit compile takes about 8 s per circuit as a density matrix, which would be about 4 h for 1,800 circuits. More threads did not help.
  The 300-ECG run therefore routes the circuit on the best **8-qubit line** of FakeFez (qubits 126-125-124-123-136-143-142-141, chosen by lowest summed
  CZ + readout error): 109–117 CZ, 0.1–0.2 s per circuit. This compile has more 2q gates, so it is a **pessimistic** proxy. The best compile was additionally run on
  the 100 H4 ECGs for the two k=32 models as a check.
- Readout mitigation: per-qubit confusion matrix from all-|0⟩/all-|1⟩ calibration circuits on the same physical qubits, applied to ⟨Z⟩ in closed form.
- Result: AUC unchanged within sampling error for all 6 models (π/6 k=32 0.961 → 0.960). Noise shrinks ⟨Z⟩ by 0.71–0.88×,
  which preserves ranking but breaks calibration and the 0.5 threshold (balanced accuracy 0.883 → 0.857). *Added analysis:* a single global rescale
  restores it. Note: this uses the noiseless simulation of the same circuits, but no labels.

### H4: real IBM hardware: not run (superseded by H6 on 2026-11-14)
- No `QISKIT_IBM_TOKEN`, **and** the session's network policy returns 403 to CONNECT for `quantum.cloud.ibm.com`, `iam.cloud.ibm.com` and
  `us-east.quantum-computing.cloud.ibm.com`. No results were simulated in their place.
- `hw_run_ibm.py` implements the plan (2 models, the same 100 ECGs, 1000 shots, one SamplerV2 job per model in a Batch, DD + twirling, readout
  calibration PUBs, everything saved before analysis, `--retrieve` for dropped sessions). An `--n` option was added so the dry run fits in minutes.
  The **dry run in Qiskit Runtime local mode on FakeFez ran end to end** (`results/hw/ibm_local/`).

### H5: training-free encoding score
- Trace distance between class-mean encoded states (θ = 0) is a weak predictor of trained AUC (Spearman +0.34; +0.71 within k=32). Post hoc, and clearly labelled
  as such: purity of the class-mean states tracks AUC (+0.80), but it rises monotonically as the scale shrinks, so it would pick π/8 over π/6.

---

## 6. Phase C: model improvements (NEXT_RUNBOOK.md, N1–N4)

Protocol: k=32, π/6, 1000 steps, z readout unless stated, 5 folds. **Change for all new runs: the l1 selector's liblinear fits use
`random_state=0`.** Folds 0–3 produce the identical ordered feature list as before; fold 4 gives the same set in a different order, so the fold-4 baseline was rerun
(`results/n0_baseline_fold4.jsonl`, 0.9541) and the baseline = Stage 2 folds 0–3 + this run. Before any training, finite differences checked
the new gradients: max error 8.6e-11 (z) and 6.7e-11 (zz) over θ, final-layer θ, readout weights and scales. The refactored code reproduced an old run's test
AUC exactly (0.95975870).

| step | result |
|---|---|
| N1 zz readout (36 observables, 109 params) | 0.953 ± 0.005, +0.002 vs baseline (5/5 folds); at 180 shots 0.936 < z 0.939 |
| N2 trained per-feature scales | **0.955 ± 0.005**, +0.004 (5/5 folds); mean \|s\| 0.69 for first-round features, ≈0.54 later; 22/160 → ≈0; 2/160 > π/2 |
| both vs degree-2 LR (0.958) | 0/5 folds; N2 closes about half the gap, N1 about a third |
| N3 kernel-target alignment | Spearman +0.61 (trace distance +0.34); centered KTA (added) +0.86; neither picks π/6 exactly |
| N4 clinical metrics | best VQC: sensitivity 86.7% @ 90% spec, 79.6% @ 95% (degree-2 LR 88.4% / 81.6%); inferior MI hardest subtype |

N1 and N2 ran in parallel (2 workers each, about 22 min), N3 and N4 used no training. All finished at 11:29 IST, about 70 min before the 12:40 deadline.

---

## 7. Problems encountered and how they were handled

1. **Non-deterministic l1 selection.** `LogisticRegression(solver="liblinear")` shuffles data internally unless `random_state` is fixed. Near-tied features
   could swap order between processes (seen in fold 3 at positions 27/28 and in fold 1 during Stage 4b; selected *sets* were identical in all
   checks). Effect on reported results: none on sets, and only order, i.e. which re-uploading layer a feature lands in. Handling: phase A/B scripts use each run's
   logged `features`; phase C added `random_state=0` (as the NEXT runbook asked) and reran the one affected baseline fold.
2. **Worker restart during Stage 4** (session infrastructure). The background job died at 9/20 shuffled runs. JSONL lines are written only after each run completes, so
   no partial rows were left. The finished runs were committed and the chain was restarted; the resume logic skipped the finished configs.
3. **Wait loops that never ended.** Early `pgrep -f` wait loops matched their own command line and never exited. This affected only waiting, not runs. Fixed by using
   `pgrep -f "[p]ython -u scale_cv"`.
4. **Aer `parameter_binds` with expressions** (H1 benchmark) → switched to Aer primitives.
5. **Density-matrix noise simulation too slow for 12 qubits** (H3b) → 8-qubit line compile plus a best-compile check (§5).
6. **IBM blocked by network policy** (H4) → script and dry run delivered; the user needs to allow the hosts or run from a laptop.
7. **`summarize.py` bugs found during the documentation pass** (fixed 5 Nov):
   - it did not group by the new `readout` / `train_scale` fields, so N1/N2 rows would have merged with z-readout rows. Now grouped (defaults
     `z` / `False` for older rows);
   - `last_step_frac` was computed as "best_step equals the group's *largest* best_step" instead of "equals the final step". All reported
     values were re-checked with the correct definition: **no documented number or decision changes** (Stages 1, 2, 2b and 3 are identical; NEXT.md
     already used the correct definition). Only pre-session `sweep_partial` (π/3, 2π/3) and N1 summarize output would differ. The `results/*.md` tables
     were regenerated with the fixed script (same values plus the two new columns).

## 8. Judgment calls and deviations from the runbooks (consolidated)

| where | runbook said | what was done | why |
|---|---|---|---|
| Stage 2b | use 2000 steps if mean val AUC +≥0.003 | kept 1000 (pooled +0.0028; k=32 alone +0.00301) | literal pooled mean; keeps Stage 3 at 5 folds |
| Stage 3 | run 135 configs | reused 15 identical Stage 2 runs | identical keys, deterministic training |
| Stage 4 | interpret drop real − shuffled | added comparison with k=8 baseline and second figure panel | the plain drop confounds information with noise damage |
| Stage 4b | (user request) | C chosen on validation; also ridge "function-class" R² | answers "what class does it learn" directly |
| H0 | (user request) | benchmark + capability table, Qiskit for device stages, PennyLane kept | evidence-based choice |
| H2 | shots {100…10000} | added 30 and 2000 | locate the 0.01-loss point more precisely |
| H3b | best compile, 300 ECGs | 8-qubit line compile for 300; best compile on 100 | 12-qubit density matrix ≈ 4 h |
| H3b | raw vs mitigated vs noiseless | also shots-only, calibration and threshold analysis, global rescale | explains why AUC survives but probabilities do not |
| H4 | run if token set | not run in this session (no token; network blocked), script + dry run. Done later as H6 | no fabricated hardware data |
| H5 | trace distance + fidelity | also purity, labelled post hoc | it was the informative column |
| N3 | uncentered KTA | also centered KTA, decided before seeing results | standard fix for imbalance offset |
| N4 | metrics per model | also LR and degree-2 LR references on the same features | the comparison the paper needs |

## 9. Open items / suggested next steps

1. **(Done 2026-11-14 as H6 on `ibm_marrakesh`; see `HARDWARE.md`.)** Original item: run H4 on real hardware (`python -u hw_run_ibm.py --submit`) once `QISKIT_IBM_TOKEN` is available and IBM hosts are reachable. It is expected to use about 1–2 of the ~10 free QPU
   minutes. Compare against H3's best-compile numbers on the same 100 ECGs (π/6: noiseless 0.983, FakeFez 0.984; π: 0.856 / 0.866).
2. Seed the random selector with the run seed if the random-feature control is to vary feature sets across seeds.
3. N2 with more steps or a separate learning rate for the scales; N1 + N2 combined; a quadratic-capable readout to test whether the gap to degree-2 LR
   is the readout or the circuit.
4. Re-run the full CV under the deterministic l1 selector if exact reproducibility of phase A is required (only order changes are expected).
5. Hardware-aware circuit: replace the CNOT ring with a heavy-hex-native entangler (no SWAPs: 32 → ≤ 32 CZ) and retrain.
