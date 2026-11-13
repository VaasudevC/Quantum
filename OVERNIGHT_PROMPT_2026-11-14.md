# Overnight prompt for Claude Code (14 Nov 2026, about 6 to 7 hours, machine idle)

Paste everything below the line into Claude Code, started in `cloud_repo` on branch `smalldata`.

---

You have this machine to yourself for about 7 hours. Chrome, VS Code and the Claude desktop app are closed, so RAM is free. Work through the phases below in order. Do not wait for me: nobody will answer a question or a permission prompt tonight. If a step needs a permission you do not have, skip it, log it as incomplete, and go on.

## Rules (unchanged)

- Do not change, re-run or re-word any pre-registered analysis (SD primary, QB claim rules). New analyses go under "Exploratory (not pre-registered)" with a date, logged BEFORE they run.
- Do not change Windows security or system settings. Do not retry torch on Windows. No IBM contact, no token handling. Do not try to push.
- Report null and negative results as they are. List anything unfinished as incomplete, with the resume command.
- Commit after every phase. Keep `MORNING_SUMMARY.md` at 15 lines or fewer.
- **Time box:** note the start time now. At start + 6 h stop launching new work, let running units finish, then do Phase 7. Never leave the repo uncommitted.

## Phase 0. Start the queue (10 minutes)

1. The RAM-gated scheduler has been idle since 17:27 on 13 Nov; assume it is dead. Check for leftover `python.exe` workers and stale claim files in `C:\bwcache\sd\claims`, then start fresh.
2. Tonight's worker cap is **8 processes in total** (the QB limit of 4 was for a busy machine). Log this as an execution note in `OVERNIGHT_LOG.md`; it is not an analysis change. Keep 1 BLAS thread per process and the 3 GB RAM guard.
3. Launch, in this priority order, filling the 8 slots:
   - k=4 small-data units (639 of 850 left): `sd_launch.py ... main 4 100,50,200,25 ... --tag _k4 --reps 40`, then `main 4 500 ... --reps 10` (commands on `MORNING_SUMMARY.md` lines 12 to 13 of commit 35e0dd4).
   - the 5 remaining seed-2 VQC runs (`scale_cv.py`, `--jobs 2`).
   - QB5 folds 1 and 2 (12 units).
4. Every 30 minutes write one line to `OVERNIGHT_LOG.md`: time, units done per job, free RAM.

While the queue runs, do Phases 1 and 2. They need almost no CPU.

## Phase 1. Write the paper draft (no compute)

Create `paper/PAPER_DRAFT.md`: a complete first draft, not an outline. Use only numbers that exist in the repo and cite the file and commit for each in an HTML comment next to it. Leave `[PENDING: k=4]` style markers where tonight's runs will fill in, and fill them in Phase 7.

- **Working title:** "Which Features, Which Encoding, and What Does the Circuit Add? A Pre-registered Study of Compact Quantum Classifiers for ECG-Based Myocardial Infarction Detection".
- **Story, in this order:**
  1. Compact feature selection under a qubit budget (continues the poster).
  2. The encoding rule: angle saturation causes the re-uploading collapse; a small scale (π/6) fixes it; replicated in the brick-wall.
  3. The rule extends from scale to feature map (QB1): across 1,200 settings a map becomes more expressive than an RBF kernel only where it stops aligning with the labels.
  4. What the circuit adds (QB3): the win over logistic regression is mild non-linearity. Classical quadratic models with no more parameters on the same bounded inputs match the VQC.
  5. Pre-registered small-data benchmark: non-inferior at every size, superior at none; equivalent to SVM-RBF within ±0.005 at N = 50 to 500 (exploratory).
  6. Robustness, shift, imbalance (QB5): no benefit.
  7. Hardware readiness in simulation.
- **Sections:** abstract (200 words), introduction, related work (Huang 2021; Schuld 2021; Bowles, Ahmed and Schuld 2024; the Kerenidis brick-wall paper; leave `[CITE]` markers, do not invent references), data and features, models, encoding study, benchmark, small-data study, QB study, hardware, discussion, limitations, conclusion, reproducibility statement.
- State plainly: no quantum-advantage claim, no parameter-efficiency claim, everything is classically simulable, the brick-wall is a reimplementation, TabPFN was not run, no real-device run.
- Write a separate `paper/CLAIMS_CHECK.md`: one row per number in the draft with its source file, commit, and "verified by recomputing: yes/no". Recompute each from the jsonl.

## Phase 2. Clean-up (no compute)

1. `../README.md` and `../docs/TEST7_FEATURE_SELECTION.md` (parent folder, outside the repo): do not edit them. List in `MORNING_SUMMARY.md` the lines that claim quantum beats classical or that 12 to 16 qubits reach 87 to 91%, so I can remove them.
2. Repo `README.md`: update the status section to the current conclusions and link `QB.md`, `SMALLDATA.md` and `paper/PAPER_DRAFT.md`.
3. Reproducibility: write `requirements-lock.txt` from `pip freeze` of `..\.venv`; update `docs/REPRODUCE.md` with the SD, DX and QB commands in run order; record Python, numpy, scikit-learn and xgboost versions and the CPU model.
4. IBM run package, **dry run only**: make sure one command runs the k=32 π/6 angle VQC on the 300 FakeFez ECGs at 1,024 shots plus a 10-ECG calibration job, end to end on the fake backend, and writes the report fields (noiseless vs simulated-noise AUC with bootstrap CIs, per-ECG score agreement, layout, depth, 2-qubit gate count). Put the exact real-device command, which I will run myself, in `HARDWARE.md`.

## Phase 3. Analyses when the queue finishes

1. `sd_analyze.py 4 25,50,100,200,500 _main`, `make_paper.py`, `qb5_analyze.py`. Apply the QB candidate rule to the final QB5 table. Update the `SMALLDATA.md` headline (k=4 is a secondary endpoint; keep the "reduced sample" label for N=500) and the `QB.md` headline.
2. Seeds: with all 20 runs, report the seed-to-seed SD against the fold-to-fold SD at k=8 and k=32, and whether the VQC is above LR in every fold for every seed.

## Phase 4. Stronger statistics for the full-data comparisons (exploratory, about 30 minutes)

Log first in `RESULTS.md` under "Exploratory (added 14 Nov)". The current evidence for "VQC above LR" and "degree-2 LR above VQC" is 5 of 5 folds, whose smallest sign-test p is 0.062.

- Save per-ECG out-of-fold test scores at k=32 for: LR, degree-2 LR, π/6 VQC, trained-scale VQC, XGBoost. Reuse `results/bw/vqc_scores.json` where it has them.
- For each pair, report the AUC difference with a 95% CI from a patient-level bootstrap inside each test fold (2,000 resamples, fold-stratified), and DeLong's paired test per fold combined across folds.
- Add the result as one small table to the draft. Do not change any existing claim wording; add the CIs next to them.

## Phase 5. QB3 twins on all five folds (exploratory, about 30 minutes)

QB is closed with no candidate, so folds 3 and 4 are no longer needed for a QB confirmation. Log this in `QB.md` under "Exploratory (not pre-registered), 14 Nov" BEFORE running: the QB3 comparison is extended to folds 3 and 4 for a descriptive 5-fold table only; it supports no benefit claim.

- Run the rank-2 factorization machine and the sparse degree-2 LR on bounded inputs on folds 3 and 4, same tuning as folds 0 to 2.
- Report the 5-fold table next to the VQC rows, with the Phase 4 paired CIs for twin minus VQC.
- Make Figure 5: heat map of the VQC's degree-2 surrogate coefficient matrix beside the degree-2 LR matrix for fold 0, with the spectrum of each underneath.

## Phase 6. Only if time remains before the 6-hour mark, in this order

1. **QB4 placement training, fold 0 only** (7 placements, up to 4 in parallel). If the test-AUC spread on fold 0 is below 0.002, stop and report "placement does not matter at this scale (fold 0 only; folds 1 and 2 not run)". If it is 0.002 or more, run folds 1 and 2.
2. **Brick-wall under WSL.** Only if `wsl -l -v` already shows an installed Linux distribution. Do not install WSL, do not enable any Windows feature, and do not touch Application Control. Inside WSL, create a venv in the Linux home directory, `pip install` CPU torch, numpy, scikit-learn, and read the repo through `/mnt/c/...` (make the `C:/bwcache/sd` path in `sd_run.py` overridable by an environment variable; default unchanged).
   - Verification first: rerun 10 existing N=100 Q4 units. Log the per-unit AUC differences. Proceed only if every |difference| is at most 0.002 and their mean is within ±0.0005.
   - Log the platform change as a dated deviation in `SMALLDATA.md` BEFORE any new Q4 run, quoting the verification numbers.
   - Then run Q4 at N = 50, 200, 25, 500 on subsets 0 to 9, re-run `sd_analyze.py` for k=8 with Q4 included as a separate, labelled table ("reduced sample; run under WSL"), and leave the existing primary-endpoint text untouched.
   - If any step fails, mark it "incomplete (WSL route failed: reason)" and move on. Do not attempt EXTRA 6 or the n=16 external run tonight.
3. **QB5 full-data part**, only the label-noise settings.

## Phase 7. Wrap-up (start no later than start + 6 h 30 min)

1. Fill the `[PENDING]` markers in `paper/PAPER_DRAFT.md`; rebuild `results/paper/` (Tables 1 to 4, Figures 1 to 5); finish `paper/CLAIMS_CHECK.md`.
2. Update `PAPER_OUTLINE.md` incomplete list.
3. Rewrite `MORNING_SUMMARY.md` (15 lines): line 1 = what finished; then each incomplete item with its resume command; then the parent-folder lines I should delete; last line = "push not done; run `git push -u origin smalldata` and `git push -u origin brickwall-overnight`".
4. Final commit. Print the commit hash and the wall-clock time used.
