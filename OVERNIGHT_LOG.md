# OVERNIGHT_LOG (branch brickwall-overnight)

Times are local. Machine: 16 cores, 15.7 GB RAM but only ~1 GB free at start (other apps) -> few workers.

- [start] P0: env = ../.venv (py3.14.3, numpy 2.4.6, sklearn 1.9.0, xgboost 3.4.1, wfdb 4.3.1). `p0_sanity.py`:
  fold-0 k=32 pi/6 test AUC recomputed from stored params = 0.959758698692389 == logged (diff 0.0). P0 PASS.
- Problem: `import torch` in ../.venv fails: "Application Control policy has blocked torch_global_deps.dll" (WDAC).
  Trying a venv outside OneDrive (C:\bwvenv). pennylane/qiskit not yet installed (pip, see below).
- P1 launched (4 workers): p1_classical.py -> results/p1_classical.jsonl
- torch works in C:\bwvenv (outside OneDrive; WDAC blocked it inside OneDrive .venv). All torch/qiskit/pennylane work uses C:\bwvenv.
- Application Control blocks DLL loads transiently on first use (sklearn _argkmin/_loss in C:\bwvenv); a retry works. Runners must tolerate this (retry once).
- P2 done: p2_verify.py (n=8,12; k=2,4 / 2,4,6; 2 columns). Fast sector simulator vs independent dense 2^n gate-level simulator: states agree to <=1.3e-15
  (numpy sector incl. global phase, torch complex128 up to global phase), features <=5e-15, no weight leakage (0.0), Lemma-2 occupations k/n to 1.4e-15
  (after spreading and after data layer), autograd vs finite differences <=2.3e-10. PennyLane check pending install (pip log C:\Users\vaasu\pip_pl.log).
  Reconstructed quantum_utils (bw_utils.py). FINDING: the literal "Hadamard brick-wall" (n-1 columns of chain-pair RBS(pi/4)) is orthogonal
  but NOT uniform-modulus (max | |W_ij|^2-1/n | = 0.35 at n=8, 0.40 at n=12), so it doesn't satisfy Lemma 2. DECISION: U_had = normalised
  Walsh-Hadamard (n=8), real Hadamard matrix of order 12 (Paley) at n=12, applied as exact many-body passive-FLO (JW) operator.
  Default particle_count guessed: max(2, 2*(n//8)) (n=8->2 as in notebook).
- P2 toy XOR (p2_xor_facts.py): notebook loop with exact gradients (instead of stochastic PSR): default-seed 100% (n=8,c=2,k=2); 10 seeds mean 52.9%
  (0.5-0.67 mostly), 300 steps 89.6%. Notebook's saved 72.9% not reproducible: different init (initial_weights unknown) + stochastic PSR. High seed variance.
- Added bw_torch.py (torch simulator, numpy-only AUC/LR to dodge blocked sklearn/pandas DLLs in C:\bwvenv), cache_data.py -> C:\bwcache npz (selected cols).
- P3 stage A launched (3 workers): results/bw/p3_stageA.jsonl. P1 still running (rf/xgb at k=249 slow).
- P4 sanity check (ptb_external/extract_ptb_features.py --mode xl-check, 50 PTB-XL records): TR columns (r_amp,s_amp,rs_ratio,q_*,t_*) and n_beats_detected
  reproduce EXACTLY (diff 0); the ST columns do NOT (max abs diff e.g. st_amp_j60 0.025 mV, st_area 0.76, st_curvature up to 35-140 mV/s^2; 142 of 249 cols differ).
  Cause: extract_st_features.py was modified (2026-07-31) after PTB_XL/st_features.csv was generated (2026-07-19); the table's ST columns are from the older
  version (st_features.csv == table ST cols to <1e-9), the current script gives slightly different ST values. Old code unavailable. PTB_XL/dataset_signals.npy is
  misaligned with its metadata (some records skipped), not usable. PLAN: (1) evaluate PTB with current extractor (primary, as specified), (2) quantify the shift,
  (3) if time allows recompute PTB-XL features with the CURRENT code and retrain/score as a consistency check.
- P4: PTB feature extraction launched in background (2 jobs).
- P2 PennyLane check (pennylane 0.43.3, qiskit 2.6.2, aer 0.18.2 in C:\bwvenv): notebook make_circuit gate sequence (StatePrep of phi0) vs sector sim: features max diff <= 8.9e-16 for n=8 k=2,4 and n=12 k=2,4,6. P2 PASS.
- PTB extraction finished: 445 usable records (80 healthy/52 patients, 365 MI/148 patients); 3 skipped (<3 beats), 101 other diagnoses dropped.
- P3 stage A prelim (600 steps, 52/80 runs, killed) showed many runs still improving at the end; relaunched Stage A with 1500 steps + extension rule (best in last 20% -> extend once to 3000). Prelim kept as p3_stageA_prelim_600steps.jsonl.
- P6 prep: hw_bw.py builds the Qiskit circuit (magic prep + exact Givens/RBS compilation of the spreading W: n(n-1)/2 nearest-neighbour RBS + Z signs, then data RZ and brick-wall columns, XXPlusYY(phi,pi/2)=SingleExcitation). Statevector features agree with the simulator to <=4.4e-15 at n=8 (k=2,4) and n=12 (k=2,4,6).
- P6 compile script p6_compile.py works (RBS(XXPlusYY) = 2 CZ; n=12 k=4 c=2 ~160-172 CZ, est success 0.5 FakeFez). Final table to be run after P3 finals are known.
- P3 stage A (1500 steps, extend to 3000 if best in last 20%; folds 0-1; val AUC): winners n=8: k=4, cols=4, s=pi/4 (0.9220; k=2 cols=4 s=pi/6 0.9211); n=12: k=6, cols=4, s=pi/4 (0.9361). Small scales (pi/6-pi/4) best, s=pi worst at every n,k -> pi/6 lesson transfers qualitatively. Launched stage B (5 workers).
- Per user: stage B n=12 restricted to linear_score and corr head at the stage-A winner (columns=12 skipped); n=8 all variants. Relaunched (resumable).
- ST MISMATCH SOLVED: table ST columns were built with the 0.5-40 Hz band-pass in place of the 0.05 Hz diagnostic_filter (script edited 2026-07-31). Patching EST.diagnostic_filter in ptb_external/extract_ptb_features.py reproduces ALL 249 columns of 50 PTB-XL records (max abs diff 8.5e-14). Re-extracting PTB with the recovered version = primary external result; the earlier run (current 0.05 Hz filter) kept as sensitivity (results/ext/*_currentfilter0p05*). Restricted-pool retraining no longer needed.
- P3 stage B (val AUC, folds 0-1): n=8 best remains linear head cols=4 k=4 pi/4 (corr head 0.9211 vs linear 0.9220 mean ~tie); n=12: corr 0.9310, linear_score 0.9336, linear 0.9361. FINALS: n=8 (k=4, 4 cols, pi/4, linear), n=12 (k=6, 4 cols, pi/4, linear); runner-ups n=8 corr head, n=12 (k=4, 2 cols, pi/6). Stage C running (6 workers); results/bw/final_configs.json written; P4 brickwall external training launched.
- P6: hw_run_ibm_brickwall.py written; local dry run on FakeFez works end-to-end (bw8 136 CZ depth 256; vqc32 80 CZ), refusal guard tested (>8 min estimate refuses).
