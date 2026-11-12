"""Summaries of the EXTRA runs -> results/bw/extra_summary.md (+ prints)."""
import json, numpy as np
from pathlib import Path
def rows(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []
L = []
C = rows("results/bw/p3_stageC.jsonl"); E1 = rows("results/bw/extra1_bw.jsonl"); V = rows("results/bw/extra1_vqc.jsonl"); S2 = rows("results/stage2_steps1000.jsonl"); N0 = rows("results/n0_baseline_fold4.jsonl")
L += ["**EXTRA 1 — 3 seeds × 5 folds** (mean over folds per seed; seed-to-seed SD of that mean; the SD over all 15 fold×seed values is also given)\n", "| model | seed 0 | seed 1 | seed 2 | mean of seeds ± SD | all 15 values: mean ± SD |", "|---|---|---|---|---|---|"]
def line(name, per_seed):
    ms = [np.mean(v) for v in per_seed if len(v) == 5]
    if len(ms) < 2: return
    allv = [x for v in per_seed if len(v) == 5 for x in v]
    L.append(f"| {name} | " + " | ".join(f"{np.mean(v):.4f}" if len(v) == 5 else "–" for v in per_seed) + f" | {np.mean(ms):.4f} ± {np.std(ms, ddof=1):.4f} | {np.mean(allv):.4f} ± {np.std(allv, ddof=1):.4f} |")
for n in (8, 12):
    per = [[r["test_auc"] for r in sorted([r for r in C if r["n"] == n and r.get("rank") == 0], key=lambda r: r["fold"])]]
    for sd in (1, 2): per.append([r["test_auc"] for r in sorted([r for r in E1 if r["n"] == n and r["seed"] == sd], key=lambda r: r["fold"])])
    line(f"brick-wall n={n}", per)
for k in (8, 32):
    per = []
    s0 = [r for r in S2 if r["k"] == k and r["scale"] == "pi/6" and r["selector"] == "l1" and r["steps"] == 1000 and not r["shuffle_later"] and r["seed"] == 0 and not r.get("train_scale") and r.get("readout", "z") == "z"]
    byf = {r["fold"]: r["test_auc"] for r in s0}
    if k == 32 and N0: byf[4] = N0[0]["test_auc"]
    per.append([byf[f] for f in range(5) if f in byf])
    for sd in (1, 2): per.append([r["test_auc"] for r in sorted([r for r in V if r["k"] == k and r["seed"] == sd], key=lambda r: r["fold"])])
    line(f"angle VQC π/6, k={k}", per)
B = rows("results/bw/extra3_butterfly.jsonl"); b = sorted([r for r in B if r["columns"] == 6 and r["k"] == 4 and abs(r["scale"] - 0.5236) < 1e-3], key=lambda r: r["fold"])
c8 = sorted([r for r in C if r["n"] == 8 and r.get("rank") == 0], key=lambda r: r["fold"])
if len(b) == 5:
    L += ["", "**EXTRA 3 — unitary butterfly vs brick-wall, n=8** (5 folds)\n", "| model | params (circuit) | depth (logical) | test AUC | val AUC | folds (test) |", "|---|---|---|---|---|---|",
          f"| brick-wall, 4 columns (k=4, π/4) | {4*12} | 4 + 7 + 1 + 8 (NN) | {np.mean([r['test_auc'] for r in c8]):.4f} ± {np.std([r['test_auc'] for r in c8], ddof=1):.4f} | {np.mean([r['val_auc'] for r in c8]):.4f} | " + ", ".join(f"{r['test_auc']:.3f}" for r in c8) + " |",
          f"| butterfly, 6 layers = 2 butterflies (k=4, π/6) | {6*12} | 4 + 3 + 1 + 12 (all-to-all) | {np.mean([r['test_auc'] for r in b]):.4f} ± {np.std([r['test_auc'] for r in b], ddof=1):.4f} | {np.mean([r['val_auc'] for r in b]):.4f} | " + ", ".join(f"{r['test_auc']:.3f}" for r in b) + " |"]
N = sorted(rows("results/bw/extra4_n16.jsonl"), key=lambda r: r["fold"])
if N:
    L += ["", f"**EXTRA 4 — 16 qubits** (k=4, 4 columns, π/4, sector dimension 1820; folds finished: {len(N)}/5): test AUC " + ", ".join(f"f{r['fold']} {r['test_auc']:.4f}" for r in N) + f"; mean {np.mean([r['test_auc'] for r in N]):.4f}; max|head| {max(max(abs(x) for x in r['params']['h']) for r in N):.1f}"]
open("results/bw/extra_summary.md", "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))
