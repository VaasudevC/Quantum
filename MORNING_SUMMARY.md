1. PUSH NOT DONE: `git push -u origin smalldata` needs your approval; all work is committed locally (branch smalldata).
2. QB: NO BENEFIT FOUND; no candidate; QB3 now 5-fold (exploratory folds 3-4 consistent: deg-2 LR matches/exceeds VQC in all 5 folds).
3. QB3 exploratory: degree-2 surrogate R²=0.9865 (fold 0 VQC); sparse L1 deg-2 LR (65 nonzero) ≥ VQC_pi6 in every fold; Fig5 heatmap → results/figs/fig5_qb3_coef_heatmap.png.
4. QB5: 18/30 units (fold 0 done; folds 1-2 running in queue); no perturbation/shift/subtype gives quantum >= +0.005 over best classical.
5. Phase 4 bootstrap (exploratory, running): full-data per-ECG bootstrap CI for k=32 AUC diffs (VQC vs LR, LR2, XGB); results/p4_bootstrap.json when done.
6. k=4 SD queue: 279/850 units (N=100 done 200/200; N=50 at 79/200; N=25/200/500 not started). Queue running with MAXP=8.
7. Phase 2 done: README (no-advantage statement), docs/REPRODUCE.md, requirements-lock.txt, HARDWARE.md (IBM command). IBM dry run INCOMPLETE: Qiskit DLL blocked by WDAC.
8. Phase 4 DONE: VQC_pi6 vs LR=+0.0060[+0.0042,+0.0078]; vs LR2=-0.0022[-0.0046,+0.0002] (CI spans 0); vs XGB=-0.0112[-0.0130,-0.0095]. VQC≈LR2 (confirms QB3); XGB leads. Full table: results/p4_bootstrap_table.md.
9. When k=4 queue done: `..\.venv\Scripts\python.exe sd_analyze.py 4 25,50,100,200,500 _main` then `make_paper.py` to update Table 3.
10. Then `..\.venv\Scripts\python.exe qb5_analyze.py` (apply QB5 candidate rule; target 30/30 units).
11. INCOMPLETE (blocked): brick-wall at N!=100; torch blocked by WDAC. C7 TabPFN: licence login. IBM run: yours (command in HARDWARE.md).
12. Roc_auc bug found and fixed in p4_full_bootstrap.py: tie detection was inverted (== vs !=); LR AUC was 0.5000 before fix.
13. QB4 training resume (optional, ~5-7 h idle): `sh qb4_run.sh 0`, `sh qb4_run.sh 1`, `sh qb4_run.sh 2`.
14. seed=3 VQC runs (5 runs for stage3_selectors.jsonl): queued, will start when SD workers exit.
15. Please delete C:\tvenv (test venv; removal blocked by safety check).
