1. PUSH NOT DONE: `git push -u origin smalldata` / `brickwall-overnight` was refused by Claude Code's permission check (needs your approval); all work is committed locally.
2. QB (pre-registered, QB.md) claim level: NO BENEFIT FOUND; confirmation not run (no candidate); folds 3-4 and PTB untouched by QB.
3. QB1: 1,200 feature-map settings screened; none passes the gate — g >= sqrt(N)/2 only at s = pi/2 where label alignment collapses (Table 4, Fig 4).
4. QB3: rank-2 factorization machine / sparse deg-2 LR (<= the VQC's parameters, same bounded inputs) match the trained-scale VQC (0.956) -> win over LR = non-linearity.
5. QB5 (18/30 units): no perturbation/shift/imbalance/subtype setting gives quantum >= +0.005 over the best classical model.
6. Small-data study (k=8, final): non-inferior to the best classical model at every N, superior at none; Q3 = SVM-RBF within +-0.005 at N=50-500 (exploratory).
7. Full data: XGBoost best at every feature count; encoding rule pi/6 (k=32: 0.791 at pi -> 0.951) replicated in the brick-wall. results/paper/ = Tables 1-4, Figs 1-4.
8. INCOMPLETE: k=4 SD endpoint (207/850 units), VQC seeds (5 runs), QB5 folds 1-2 (12 units), QB4 training (6-hour rule), QB5 full-data part.
9. Why: free RAM stays < 3.5 GB while Chrome/Claude/VS Code are open; the RAM-gated queue (qb_sched.py, <= 4 processes) resumes them automatically when memory frees.
10. INCOMPLETE (blocked): brick-wall at N != 100, EXTRA 6, n=16 external — torch blocked by Windows Application Control. C7 TabPFN — licence login. IBM run — yours.
11. Check progress: `tail results/qb_sched.log`; `wc -l results/qb5.jsonl` (target 30).
12. When the queue is done: `C:\bwvenv\Scripts\python.exe sd_analyze.py 4 25,50,100,200,500 _main` then `C:\bwvenv\Scripts\python.exe make_paper.py`
13. then `..\.venv\Scripts\python.exe qb5_analyze.py` (final QB5 table; apply the candidate rule).
14. QB4 training resume (optional, ~5-7 h on an idle machine): `sh qb4_run.sh 0`, `sh qb4_run.sh 1`, `sh qb4_run.sh 2`.
15. Please delete C:\tvenv (test venv; removal was blocked by a safety check).
