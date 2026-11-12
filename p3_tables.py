import json, collections
from pathlib import Path
import numpy as np
def rows(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []
A = rows("results/bw/p3_stageA2.jsonl"); B = rows("results/bw/p3_stageB.jsonl"); C = rows("results/bw/p3_stageC.jsonl"); R = rows("results/bw/p3_ref_paper.jsonl")
sn = {0.5235987755982988: "π/6", 0.7853981633974483: "π/4", 1.5707963267948966: "π/2", 3.141592653589793: "π"}
out = []
out.append("**Stage A** (folds 0–1, mean validation AUC; 1500 steps, extended to 3000 if the best checkpoint fell in the last 20%; Adam lr 0.02, batch 512, linear head)\n")
for n in (8, 12):
    d = collections.defaultdict(list)
    for r in A:
        if r["n"] == n: d[(r["k"], r["columns"], r["scale"])].append(r["val_auc"])
    ks = sorted({k[0] for k in d}); out.append(f"n = {n}\n\n| k | columns | " + " | ".join(sn[s] for s in sn) + " |\n|---|---|" + "---|" * len(sn))
    for k in ks:
        for c in (2, 4):
            out.append(f"| {k} | {c} | " + " | ".join(f"{np.mean(d[(k, c, s)]):.4f}" for s in sn) + " |")
    out.append("")
out.append("**Stage B** (folds 0–1, mean validation AUC at the stage-A winner; ‘Δ’ = change vs the winner's 0.9220 (n=8) / 0.9361 (n=12))\n")
out.append("| n | variant | mean val AUC |\n|---|---|---|")
for n, base in ((8, None), (12, None)):
    d = collections.defaultdict(list)
    for r in A + B:
        if r["n"] == n: d[(r["k"], r["columns"], r["head"], r["lr"], r["scale"])].append(r["val_auc"])
    win = max((k for k, v in d.items() if len(v) == 2 and k[3] == 0.02 and k[2] == "linear"), key=lambda k: np.mean(d[k]))
    for k, v in sorted(d.items(), key=lambda kv: -np.mean(kv[1])):
        if len(v) == 2 and (k[:2] == win[:2] and k[4] == win[4] or False) :
            out.append(f"| {n} | k={k[0]}, cols={k[1]}, head={k[2]}, lr={k[3]} | {np.mean(v):.4f} |")
out.append("\n**Stage C** (5 folds, test AUC; final = best validation AUC over stages A/B, runner-up second)\n")
out.append("| n | config | test AUC (mean ± SD) | sens @ 90% spec | val AUC | per-fold test AUC |\n|---|---|---|---|---|---|")
for n in (8, 12):
    for rank in (0, 1):
        rs = sorted([r for r in C if r["n"] == n and r.get("rank") == rank], key=lambda r: r["fold"])
        if len(rs) < 5: continue
        r0 = rs[0]; t = [r["test_auc"] for r in rs]
        out.append(f"| {n} | {'final' if rank == 0 else 'runner-up'}: k={r0['k']}, cols={r0['columns']}, scale {sn.get(r0['scale'], r0['scale'])}, head {r0['head']} | {np.mean(t):.4f} ± {np.std(t, ddof=1):.4f} | {np.mean([r['sens90'] for r in rs]):.3f} | {np.mean([r['val_auc'] for r in rs]):.4f} | {', '.join(f'{x:.3f}' for x in t)} |")
out.append("\n**Paper phase map** φ = π·x/max|x| (max from train) vs tanh map, same final (k, columns), 5 folds\n")
out.append("| n | phase map | test AUC (mean ± SD) |\n|---|---|---|")
for n in (8, 12):
    rs = [r for r in R if r["n"] == n]; fin = [r for r in C if r["n"] == n and r.get("rank") == 0]
    if len(fin) == 5: out.append(f"| {n} | tanh, scale π/4 (final) | {np.mean([r['test_auc'] for r in fin]):.4f} ± {np.std([r['test_auc'] for r in fin], ddof=1):.4f} |")
    if len(rs) == 5: out.append(f"| {n} | paper: π·x/max abs(x) | {np.mean([r['test_auc'] for r in rs]):.4f} ± {np.std([r['test_auc'] for r in rs], ddof=1):.4f} |")
Path("results/bw/p3_tables.md").write_text("\n".join(out), encoding="utf-8"); print("\n".join(out))
