# Overnight run: Claude Code in VS Code on your PC

**Before you leave it**
1. Open VS Code on `C:\Users\vaasu\OneDrive\Desktop\LimrickQuantumWork\cloud_repo`. The notebook and paper are already in `brickwall\`.
2. If you get `quantum_utils.py` from Antonio, put it in `brickwall\` too. The prompt rebuilds it from the paper if it's missing.
3. Set Windows to never sleep while plugged in (Settings → System → Power). Pausing OneDrive sync for the night also helps.
4. Permissions: an overnight run stalls on the first approval prompt. Start the session in a mode that accepts edits. On the first few prompts, choose "don't ask again" for `python`, `pip` and `git`, and stay until you've seen the first phase start.
5. Paste everything inside the code block below.

```
You are running UNATTENDED overnight on my Windows PC in this repo (cloud_repo). Nobody will answer
questions: never stop to ask. Make the most reasonable choice, write it down, and continue. Work through
phases P0-P6 IN ORDER; do as many as time allows, each one fully finished (code + results + log +
commit) before starting the next. A finished small phase beats a half-finished big one.

CONTEXT (read first, briefly): README.md, RESULTS.md, HARDWARE.md, NEXT.md (our angle-encoding VQC
study: pi/6 scale fixes re-uploading; k=32 test AUC 0.951 fixed, 0.955 trained scales; LR 0.945;
degree-2 LR 0.958), docs/CODE_AND_DATA.md, scale_cv.py, Vqc_fast8.py, qiskit_vqc.py.
New material from our quantum mentor (Antonio) in brickwall/: Magic_brickwall_classifier.ipynb (a
PennyLane QNN: magic-state encoding + Hadamard spreading + Rz data phases + unitary brick-wall of RBS
gates with ring connectivity + two-body nearest-neighbour projector readout + sigmoid head) and
Scalable_Quantum_ML.pdf (the paper it implements; Sections II, III.B, V, VI are the relevant ones).
Mentor guidance for this round: benchmark against XGBoost and Random Forest; use fewer features (8 and
12); train only in simulation; real IBM hardware (10 free minutes) is for TESTING trained models only.

GLOBAL RULES
- Time: run `date` at the start of every phase and every ~30 min. Do not START a new long run
  after 07:00 local time. By 07:45 the report (P6) must be written and committed, whatever is done.
- Protocol everywhere unless stated: patient-grouped, class-stratified 5-fold CV exactly as in
  scale_cv.py (same folds, same inner validation split), L1 selector with random_state=0, all
  preprocessing and selection fitted on inner-train only. Every choice (hyperparameters, scale, k,
  columns, steps) is made on VALIDATION AUC. Test folds are only reported, never used to choose.
- Metrics per run: test AUC, sensitivity at 90% specificity, val AUC, wall time. Append one JSON line
  per run to results/<phase>.jsonl; make every runner resumable (skip finished configs).
- Parallelism: use (CPU cores - 1) worker processes; check free RAM first and lower if needed. Set
  OMP/MKL/OPENBLAS threads to 1 per worker. Keep result files small (no big arrays in OneDrive).
- Environment: use the repo's existing venv if one works, else create .venv. pip install only what is
  needed (xgboost, torch CPU, pennylane==0.43.3, qiskit/qiskit-aer/qiskit-ibm-runtime per
  requirements-hw.txt). If an install fails, use an alternative and log it.
- Git: create branch `brickwall-overnight`. Commit after every phase with a one-line finding.
  Do not push. Do not modify the existing results files; add new ones.
- Log: keep OVERNIGHT_LOG.md updated after every step (time, what ran, outcome, any problem and how
  you handled it). If something fails twice, log it, skip that item, and move on.
- Honesty: never claim quantum advantage. The paper itself (Prop. 8, Sec. VI.B-C) says two-body
  correlator readouts are classically computable; hardness needs sample-level heads and many magic
  blocks (k ~ 60). At 8-12 qubits everything here is classically simulable; say so in the report.

P0 - Setup and sanity (~20 min)
- Environment as above; print python, numpy, torch, pennylane, xgboost, qiskit versions, cores, RAM.
- Sanity: reproduce from stored params one logged number (e.g. fold-0 k=32 pi/6 test AUC from
  results/stage2_steps1000.jsonl via scale_cv.prep()). Stop the whole run and log if it doesn't match.

P1 - Classical benchmarks (~1-1.5 h)  [mentor request]
- Models: LR, degree-2 LR (as stage4b_poly.py), Random Forest, XGBoost, small MLP (hidden 8/16/32).
- Feature counts k in {8, 12, 16, 32, all 249}, L1 selector; also cohen_dedup at k in {8, 12}.
- Tune RF/XGB/MLP with a small grid on validation AUC (RF: n_estimators {300,800}, max_depth {None,8,
  16}, min_samples_leaf {1,5}; XGB: n_estimators {300,800} with early stopping on val, max_depth
  {3,5,7}, learning_rate {0.03,0.1}, subsample 0.8, colsample 0.8). 5 folds.
- Output results/p1_classical.jsonl and a table in BRICKWALL.md: model x k, mean +- SD test AUC,
  sens@90spec, plus our existing VQC numbers where they exist (k=8,16,32 from results/*.jsonl).

P2 - Rebuild and verify the brick-wall QNN (~1.5 h)
- If brickwall/quantum_utils.py exists, use it. Otherwise RECONSTRUCT it from the paper and the
  notebook's call signatures, clearly marked as a reconstruction: particle_count(n, k) (notebook's
  default gives k=2 at n=8: its output cell printed "features=8, k=2, readout=sigmoid,
  accuracy=72.9%"), spreading_matrix(n) / UniformSpreading (the fixed Hadamard spreading = Walsh-
  Hadamard in the single-particle sector, Sec. II.B and III.D; Lemma 2: every mode ends with
  occupation k/n), initial_weights(n, columns, rng), shift_coefficients(k) (parallel PSR weights,
  Sec. V; only needed to reproduce the notebook's training).
- Build a fast exact simulator in PyTorch (CPU, complex128 for checks, complex64 for training).
  KEY SHORTCUT: magic prep and spreading are data-independent, so precompute |phi0> once; then
  |psi(x)> = D(phi(x)) |phi0> is just an elementwise phase multiply. Trainable columns: per column a
  full Rz layer then RBS(theta) on disjoint ring pairs (alternating even/odd pairs, the odd column
  includes the (n-1,0) pair), exactly as make_circuit in the notebook, INCLUDING its sign
  conventions (RZ(-pi*x), RZ(-w), SingleExcitation(-2w)). Readout: m_j = <|11><11|_{j,j+1}> on the
  ring, features f = 4m, linear head + sigmoid (and "linear_score" option). Gradients by autograd.
- Verify: (a) state equals the PennyLane notebook circuit to <1e-10 for 20 random (x, w) at n=8 and
  12 for every k used; (b) particle number exactly k; (c) Lemma 2 occupations k/n; (d) autograd
  gradient equals finite differences (<1e-6). If (a) fails, fix before going on; log every
  convention issue found.
- Rerun the notebook's toy XOR task with your simulator and report accuracy next to 72.9% (exact
  match not required if quantum_utils was reconstructed; explain any difference).
- Record circuit facts: qubits, k, magic blocks, depth, #RBS, #params for n=8,12 and columns 2/4/n.

P3 - Brick-wall on ECG, 8 and 12 qubits (~3 h, the main experiment)
- One feature per qubit: n = k_features in {8, 12} (L1 top-n features). Phase phi_j = s * tanh(z_j)
  (z = train-fitted standardisation). Also test the paper's own map phi = pi * x / max|x| (max from
  train) as a reference row.
- Stage A (fold 0 + fold 1, val AUC, short training): scale s in {pi/6, pi/4, pi/2, pi} x particle k
  in {2, 4} (n=8) / {2, 4, 6} (n=12) x columns in {2, 4} -> pick best on mean val AUC per n.
- Stage B (fold 0-1): for the Stage-A winner per n try columns = n and readout "linear_score";
  also a richer head (all n(n-1)/2 two-body <n_i n_j>, Sec. VI.D eq. 63) - log which wins on val.
- Stage C (5 folds): final config per n, plus the second-best for robustness. Adam, minibatch
  ~512, class-weighted BCE, best-val checkpoint, enough steps that the best checkpoint is not the
  last (extend once if it is).
- Compare in BRICKWALL.md at n=8 and n=12: brick-wall vs LR / degree-2 LR / RF / XGB / MLP (P1) vs
  our angle VQC (k=8 exists; there is no 12-feature angle VQC - say so). Report the phase-scale
  effect explicitly: does the pi/6 lesson transfer to phase encoding?

P4 - Limited-data learning curves (~1 h)  [Antonio's "learning with few data" link]
- Training sizes {50, 100, 250, 500, 1000, full}, stratified, patient-grouped subsample of inner-
  train, 5 folds x 3 repeats for sizes <= 1000; test on the full test fold.
- Models: best brick-wall n=8 and n=12 (P3 config), LR, degree-2 LR, RF, XGB, our angle VQC k=8 pi/6.
- Plot results/figs/learning_curves.png (AUC vs training size, log-x, error bars). Question: does
  any quantum model close or reverse the gap at small training sizes? Report plainly either way.

P5 - Hardware preparation, test-only (~1 h)
- Qiskit circuits for the final n=8 and n=12 brick-wall models; check equal to the simulator.
- Transpile (opt level 3, 5 seeds, best) for IBM Heron FakeFez and FakeTorino, an IQM-like square-
  lattice/ring-friendly coupling map, and an all-to-all target: 1q, 2q, depth, 2q depth, SWAPs, est.
  success. RBS is not native: report its decomposition cost.
- Noise: AerSimulator.from_backend(FakeFez), 300 stratified fold-0 test ECGs, 2000 shots: AUC
  noiseless vs noisy vs noisy + PARTICLE-NUMBER POST-SELECTION (discard shots whose Hamming weight !=
  k; report kept fraction). Also shot sweep {100, 300, 1000, 4000} on the noiseless model.
- Write hw_run_ibm_brickwall.py (TEST ONLY, no training): loads trained params, builds circuits for
  2-3 models (best n=8, best n=12, our pi/6 angle VQC) on the same 100-200 stratified test ECGs, one
  SamplerV2 batch job per model, saves job ids + raw counts before analysis, applies post-selection,
  reports AUC. Budget check that prints estimated QPU time and refuses if > 8 min. Dry-run it end to
  end in local mode on FakeFez. Token from env var QISKIT_IBM_TOKEN (never hard-code).

P6 - Report (must exist by 07:45)
- BRICKWALL.md: headline paragraph (what worked, exact numbers, best classical vs best quantum at
  8 and 12 features), one table per phase, figures list, limitations, and a short "classical vs
  quantum in this pipeline" note citing the paper's Prop. 8 and Sec. VI.B-C.
- A section "Questions for Antonio": e.g. confirm reconstructed quantum_utils conventions, k choice,
  whether to use a sample-level head, IQM access for testing.
- MORNING_SUMMARY.md (max 25 lines): what finished, key numbers, what failed, exact command for the
  IBM test run. Commit.
```
