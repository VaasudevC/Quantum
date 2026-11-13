# MORNING SUMMARY (branch smalldata; details: SMALLDATA.md, DISTINGUISHER.md, BRICKWALL.md, PAPER_OUTLINE.md)
1. Full-data benchmark: XGBoost best at every feature count (0.937 k=8 … 0.973 all 249); no quantum model meets the outperformance rule (BRICKWALL.md).
2. Encoding rule: s = π/6 removes the angle-saturation collapse (k=32: 0.791 at π → 0.951); small scales also win in the brick-wall reimplementation (results/paper/table1_encoding.md).
3. Pre-registered small-data study (k=8): quantum models non-inferior to the best classical model at N = 25, 50, 100, 200, 500 (500 = reduced sample), superior at none.
4. Primary endpoint N=100: Q3 quantum-kernel SVM vs random forest d = −0.0028 [−0.0049, −0.0010]; by inner-CV choice, Q3 vs SVM-RBF is a tie (−0.0006 [−0.0024, +0.0013]).
5. Wins (Holm) only vs kNN, XGBoost and plain LR (N=25, N=500); never vs SVM-RBF, GP, random forest. LR (9 params) matches the angle VQC (81) on AUC and gap.
6. Kernel geometry (exploratory, folds 0–2): g(RBF‖π/6 kernel) = 1.7–11.5 ≪ √N → no room for advantage; distinguisher search stopped; folds 3–4 and PTB untouched.
7. Paper outputs: results/paper/ (Tables 1–3, Figures 1–3, each stamped with its build commit); PAPER_OUTLINE.md final.
8. INCOMPLETE: k=4 secondary endpoint (194 of 850 units; stalled — free RAM < 3 GB with Chrome/Claude/VS Code open).
9. INCOMPLETE: 5 VQC seed runs (EXTRA 1) — waiting for RAM.
10. INCOMPLETE: brick-wall at N≠100, EXTRA 6 (shot-robust heads), n=16 external run — torch blocked by Windows Application Control (no setting changed).
11. INCOMPLETE: C7 TabPFN (licence login required); real IBM device run (yours; command in BRICKWALL.md §P6). Please delete C:\tvenv (test venv; removal was blocked).
12. Resume (from cloud_repo, with ≥ 3.5 GB RAM free): `..\.venv\Scripts\python.exe sd_launch.py ..\.venv\Scripts\python.exe main 4 100,50,200,25 8 --tag _k4 --reps 40`
13. then `... sd_launch.py ..\.venv\Scripts\python.exe main 4 500 8 --tag _k4 --reps 10`, `C:\bwvenv\Scripts\python.exe sd_analyze.py 4 25,50,100,200,500 _main`, `C:\bwvenv\Scripts\python.exe make_paper.py`
14. and the seeds: `..\.venv\Scripts\python.exe scale_cv.py --scales pi/6 --ks 8,32 --seeds 1,2 --steps 1000 --batch 1000 --lr 0.01 --jobs 2 --out results/bw/extra1_vqc.jsonl`
