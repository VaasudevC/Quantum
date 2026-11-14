1. PUSH NOT DONE: `git push -u origin smalldata` needs your approval; all work is committed locally (branch smalldata).
2. QB: NO BENEFIT FOUND on the main analysis; QB5 final (30/30 units) found 2 discovery candidates: subtype_IMI Q1 +0.0103[+0.0008,+0.0236] and Q3 +0.0091[+0.0016,+0.0193] vs SVM-RBF. QB6 confirmation protocol written; NOT YET RUN (see QB.md).
3. QB3 exploratory: degree-2 surrogate R²=0.9865 (fold 0 VQC); sparse L1 deg-2 LR (65 nonzero) ≥ VQC_pi6 in all 5 folds (folds 3-4 exploratory, logged before running); Fig5 → results/figs/fig5_qb3_coef_heatmap.png.
4. QB5: COMPLETE (30/30 units). All settings except subtype_IMI: no quantum model ≥ +0.005 over best classical. See QB.md for full final table.
5. Phase 4 bootstrap (exploratory, DONE): VQC_pi6 vs LR=+0.0060[+0.0042,+0.0078]; vs LR2=-0.0022[-0.0046,+0.0002] (CI spans 0); vs XGB=-0.0112[-0.0130,-0.0095]. Note: uses refitted (not tuned) LR2/XGB — Task 5 redo pending. Full table: results/p4_bootstrap_table.md.
6. k=4 SD queue: COMPLETE (850/850). sd_analyze.py done: all 5 sizes non-inferior, no superiority; best quantum=Q1/Q3, best classical=GP. Windows re-run verified identical.
7. Phase 2 done: README, docs/REPRODUCE.md, requirements-lock.txt, HARDWARE.md (IBM command). IBM dry run INCOMPLETE: Qiskit DLL blocked by WDAC.
8. PENDING tasks (see ordered list): (a) Task 2: stage3_selectors.jsonl fix + extra1_vqc seed=2 runs; (b) Task 4: run QB6; (c) Task 5: redo Phase 4 with tuned models; (d) Task 6: pooled CI for 5/5-fold claim; (e) Task 7: update SMALLDATA.md k=4 headline, make_paper.py, rewrite this file.
9. QB4 training resume (optional, ~5-7 h idle): `sh qb4_run.sh 0`, `sh qb4_run.sh 1`, `sh qb4_run.sh 2`.
10. INCOMPLETE (blocked): brick-wall at N!=100; torch blocked by WDAC. C7 TabPFN: licence login. IBM run: yours (command in HARDWARE.md).
11. stage3_selectors.jsonl: MISTAKE — 5 seed=3 rows added overnight; extracted to results/extra_seed3_k32.jsonl and original restored.
12. Please delete C:\tvenv (test venv; removal blocked by safety check).
