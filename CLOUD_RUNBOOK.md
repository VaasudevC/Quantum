# Cloud runbook: prompts for a Claude cloud session

How to use it: paste **Prompt 0** to start the session. After each stage finishes, Claude reports back and you paste the next prompt. Each prompt already carries the decision rule for what comes next, so you can also say "continue with the next stage per CLOUD_RUNBOOK.md" and let it follow the branches itself.

Every stage writes to its own `results/<stage>.jsonl`, can resume, and ends with a git commit and push. A session timeout therefore never loses work; just rerun the same command.

---

## Prompt 0: setup, verification and timing (paste first)

```
You are continuing a quantum-ML experiment in this repo (8-qubit variational quantum classifier,
PTB-XL ECG, MI vs NORM). Read README.md and CLOUD_RUNBOOK.md first; follow the runbook stage by
stage. Do not change the circuit, the readout or the CV protocol in scale_cv.py unless a stage says so.

Stage 0 tasks:
1. `pip install -r requirements.txt`. Report Python/numpy versions, `nproc`, RAM.
2. Check data: data/st_features_plus_TR.csv.gz and data/ptbxl_database.csv exist; pandas reads
   14,965 rows, 249 ECG-only features after the join.
3. Run `python -c "import Vqc_fast8; Vqc_fast8.verify()"`. State-vector and gradient diffs must be
   ~1e-12 or smaller. If not, STOP and report.
4. Timing: `python -u scale_cv.py --scales pi/4 --ks 32 --folds 0 --steps 50 --jobs 1 --out results/smoke.jsonl`
   and report secs per run. Then estimate secs per 100 steps and choose JOBS = nproc - 1 (min 1).
   If RAM per worker (~1.5 GB) x JOBS exceeds 80% of RAM, lower JOBS.
5. Sanity: `python summarize.py results/pilot_runs.jsonl results/sweep_partial.jsonl` must show
   k=32 l1 300 steps: pi 0.787, pi/2 0.896. If it doesn't, STOP and report.
6. Report a table: estimated wall time for Stages 1-5 at your JOBS (run counts are in each stage).
   Then proceed to Stage 1 without waiting unless something failed.

General rules for every stage:
- Launch long runs with `nohup ... > results/<stage>.log 2>&1 &` and poll the log every few minutes;
  if the session is about to time out, commit + push what exists (runs are resumable).
- After each stage: run summarize.py on that stage's jsonl (plus earlier ones if needed), save
  the Markdown with --md results/<stage>.md, append a dated section to RESULTS.md
  (what ran, the table, 2-4 sentence interpretation, the decision taken), then
  `git add results RESULTS.md && git commit -m "<stage>: <one-line finding>" && git push`.
- Report to me: the table, the decision rule outcome, and the exact next command you will run.
- Never use the test fold for any choice. Choices of scale/steps use VALIDATION AUC (val_auc in the
  jsonl), not test AUC. Report test AUC as the result.
```

---

## Prompt 1: finish the scale sweep (k=32, 600 steps, 5 folds)

Run count: 7 scales × 5 folds = 35. Eight are already done, because `sweep_partial.jsonl` is merged in first.

```
Stage 1 - scale sweep.
cp results/sweep_partial.jsonl results/stage1_sweep.jsonl   (resume: those 8 runs are skipped)
python -u scale_cv.py --scales "pi/8,pi/6,pi/4,pi/3,pi/2,2*pi/3,pi" --ks 32 --steps 600 \
    --jobs $JOBS --out results/stage1_sweep.jsonl
Then summarize, and also print mean VAL auc per scale.

Decision rule:
- s* = scale with the highest mean val_auc.
- If s* is the smallest tested (pi/8): extend with "pi/12,pi/16" (same command, same out file)
  and re-pick s*; repeat once more at most.
- If two scales are within 0.003 val AUC, prefer the larger one (less compression of the input).
- Also report: test AUC gap to LR at s*, and the fold-level sign test s* vs pi/2 and s* vs pi.
Write s* at the top of RESULTS.md ("Chosen scale s* = ...") and continue to Stage 2.
```

## Prompt 2: training length (is 600 steps enough?)

Run count: 3 scales × 3 k × 5 folds = 45 at 1,000 steps.

```
Stage 2 - training length at the chosen scale.
python -u scale_cv.py --scales "<s*>,pi/2,pi" --ks 8,16,32 --steps 1000 --jobs $JOBS \
    --out results/stage2_steps1000.jsonl
Compare with the 300-step pilot (results/pilot_runs.jsonl) for pi/2 and pi.

Decision rule:
- If last_step_frac > 0.5 for s* at any k (best checkpoint still at the final step), rerun s* only
  with --steps 2000 into results/stage2_steps2000.jsonl. Use the longer setting from here on
  if mean val AUC improves by >= 0.003; otherwise keep 1000. Call it STEPS*.
- Key question to answer in RESULTS.md: at s*, does AUC still FALL from k=8 to k=32, stay flat, or
  RISE? (Rise = re-uploading finally helps. That is the headline for the paper.)
```

## Prompt 3: selectors and seeds

Run count: 3 selectors × 3 k × 5 folds × 3 seeds = 135 at STEPS*. This is the largest stage, so start it early.

```
Stage 3 - selectors x seeds at s*.
python -u scale_cv.py --scales "<s*>" --ks 8,16,32 --selectors l1,cohen_dedup,random \
    --seeds 0,1,2 --steps <STEPS*> --jobs $JOBS --out results/stage3_selectors.jsonl
Summarize per selector x k (average over seeds and folds; also report seed-to-seed SD).

Report:
- selected-vs-random gap at each k (the teammate found ~+0.10 at k=8 with pi);
- l1 vs cohen_dedup (cohen_dedup is our abstract's selector, so its standing matters);
- VQC vs LR gap per selector/k.
Decision: if cohen_dedup is within 0.005 of l1 at k=32, note "Cohen's-d pipeline is competitive";
otherwise note which one wins and by how much. Continue.
```

## Prompt 4: shuffle control (are later-loaded features actually used?)

Run count: 2 scales × 2 k × 5 folds × 2 (real, shuffled) = 40. The "real" half may already exist from Stage 2.

```
Stage 4 - shuffle control.
python -u scale_cv.py --scales "<s*>,pi" --ks 16,32 --steps <STEPS*> --jobs $JOBS \
    --out results/stage4_real.jsonl
python -u scale_cv.py --scales "<s*>,pi" --ks 16,32 --steps <STEPS*> --jobs $JOBS --shuffle-later \
    --out results/stage4_shuffled.jsonl
(--shuffle-later permutes features 9..k, i.e. everything loaded after the first round.)

Report drop = real - shuffled for each scale x k.
Interpretation rule: if the drop at s* is clearly larger than at pi (e.g. >= 2x), later rounds are
now carrying information; at pi the teammate saw only 0.05-0.11. Put this table in RESULTS.md.
```

## Prompt 5: official PTB-XL split (comparability)

Run count: about 6 runs, 1 split each.

```
Stage 5 - official split. ptbxl_database.csv has strat_fold (folds 1-8 train, 9 val, 10 test).
python -u scale_cv.py --split official --scales "<s*>,pi/2,pi" --ks 8,32 --steps <STEPS*> \
    --jobs $JOBS --out results/stage5_official.jsonl
Report test AUC; flag if it differs from the grouped-CV mean by > 2 fold-SDs.
```

## Prompt 6: write-up and hand-back

```
Stage 6 - wrap up.
1. Make RESULTS.md self-contained: a headline paragraph (what the angle scale does, best numbers,
   VQC vs LR gap), then one section per stage with its table, then limitations
   (single dataset, exact simulation without noise, 8 qubits only, linear readout).
2. Add figures to results/figs/: (a) test AUC vs scale at k=32 with fold error bars, LR as a
   dashed line; (b) AUC vs k for pi, pi/2, s*; (c) shuffle-control bars. matplotlib, 150 dpi.
3. Print a 10-line summary I can paste into our lab notebook, plus the commit hash.
4. Commit and push.
```

---

## If something goes wrong

- **Session timed out or the process died.** Rerun the same command with the same `--out`; finished configs are skipped.
- **Too slow** (estimated time > 6 h for a stage). Drop to `--folds 0,1,2` for that stage, say so in RESULTS.md, and continue.
- **AUC near 0.5 on everything.** Check the data join (14,965 rows, 249 features) and that `ECG_DATA_DIR` points to `data/`.
- **Out of memory.** Lower `--jobs`.
