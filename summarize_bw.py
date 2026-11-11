"""Markdown tables for BRICKWALL.md.  python summarize_bw.py p1 | p3 | ext | lc | hw"""
import json, sys
from pathlib import Path
import numpy as np

def rows(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []
def ms(v): return f"{np.mean(v):.4f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.4f}"

def p1():
    R = rows("results/p1_classical.jsonl"); out = []
    names = dict(lr="LR", lr2="degree-2 LR", rf="Random Forest", xgb="XGBoost", mlp="MLP (8/16/32)")
    n4 = json.load(open("results/n4_clinical.json"))
    out.append("| selector | k | model | test AUC (mean ± SD, 5 folds) | sens @ 90% spec | val AUC | folds |")
    out.append("|---|---|---|---|---|---|---|")
    for sel in ("l1", "cohen_dedup"):
        for k in (8, 12, 16, 32, 249):
            for m in ("lr", "lr2", "rf", "xgb", "mlp"):
                r = [x for x in R if x["selector"] == sel and x["k"] == k and x["model"] == m]
                if not r: continue
                out.append(f"| {sel} | {k if k < 249 else 'all 249'} | {names[m]} | {ms([x['test_auc'] for x in r])} | "
                           f"{np.mean([x['sens90'] for x in r]):.3f} | {np.mean([x['val_auc'] for x in r]):.4f} | {len(r)} |")
            if sel == "l1":
                key = f"VQC z, k={k}, pi/6"
                if key in n4:
                    a = n4[key]["auc"]; s = n4[key]["sens_at_spec90"]
                    out.append(f"| l1 | {k} | **angle VQC π/6 (ours, existing)** | {a['mean']:.4f} ± {a['std']:.4f} | {s['mean']:.3f} | – | 5 |")
        if sel == "l1":
            out.append(f"| l1 | 32 | **angle VQC, trained scales (existing)** | {n4['VQC trained scale (N2), k=32']['auc']['mean']:.4f} ± {n4['VQC trained scale (N2), k=32']['auc']['std']:.4f} | {n4['VQC trained scale (N2), k=32']['sens_at_spec90']['mean']:.3f} | – | 5 |")
    return "\n".join(out)

if __name__ == "__main__":
    print({"p1": p1}[sys.argv[1]]())
