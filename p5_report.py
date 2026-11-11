"""P5 report: learning curves (AUC vs training size) -> results/figs/learning_curves.png, results/lc/p5_table.md, results/lc/p5_summary.json"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def rows(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []
LC = rows("results/lc/classical.jsonl") + rows("results/lc/vqc.jsonl") + rows("results/lc/bw.jsonl")
P1 = rows("results/p1_classical.jsonl"); C = rows("results/bw/p3_stageC.jsonl")
n4 = json.load(open("results/n4_clinical.json"))
SIZES = [50, 100, 250, 500, 1000]; FULL = 11700            # approx inner-train size (fold-dependent, ~ 14965*0.8*7/8)
label = {"lr": "LR", "lr2": "degree-2 LR", "rf": "Random Forest", "xgb": "XGBoost", "vqc_pi6": "angle VQC π/6 (k=8)", "bw8": "brick-wall n=8", "bw12": "brick-wall n=12"}
color = {"lr": "#888888", "lr2": "#bbbb33", "rf": "#228833", "xgb": "#ee7733", "vqc_pi6": "#4477aa", "bw8": "#cc3311", "bw12": "#aa3377"}

def curve(model, k):
    pts = {}
    for s in SIZES:
        v = [r["test_auc"] for r in LC if r["model"] == model and r["k"] == k and r["size"] == s]
        if v: pts[s] = (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0, len(v))
    # full-data point from the 5-fold runs
    if model in ("lr", "lr2", "rf", "xgb"):
        v = [r["test_auc"] for r in P1 if r["model"] == model and r["k"] == k and r["selector"] == "l1"]
    elif model == "vqc_pi6":
        v = [None]
    else:
        n = int(model[2:]); v = [r["test_auc"] for r in C if r["n"] == n and r.get("rank") == 0]
    if model == "vqc_pi6": pts[FULL] = (n4["VQC z, k=8, pi/6"]["auc"]["mean"], n4["VQC z, k=8, pi/6"]["auc"]["std"], 5)
    elif v: pts[FULL] = (np.mean(v), np.std(v, ddof=1), len(v))
    return pts

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
table = ["| features | model | " + " | ".join(str(s) for s in SIZES) + " | full (~11.7k) |", "|---|---|" + "---|" * (len(SIZES) + 1)]
summary = {}
for ax, (k, models) in zip(axs, ((8, ["lr", "lr2", "rf", "xgb", "vqc_pi6", "bw8"]), (12, ["lr", "lr2", "rf", "xgb", "bw12"]))):
    for m in models:
        c = curve(m, 8 if m == "vqc_pi6" else k)
        if not c: continue
        xs = sorted(c); ax.errorbar(xs, [c[x][0] for x in xs], yerr=[c[x][1] for x in xs], marker="o", ms=4, lw=1.6 if m.startswith(("bw", "vqc")) else 1.1,
                                    ls="-" if m.startswith(("bw", "vqc")) else "--", color=color[m], label=label[m], capsize=2)
        table.append(f"| {k} | {label[m]} | " + " | ".join(f"{c[s][0]:.3f} ± {c[s][1]:.3f}" if s in c else "–" for s in SIZES + [FULL]) + " |")
        summary[f"{m}|{k}"] = {str(s): c[s][0] for s in c}
    ax.set_xscale("log"); ax.set_xlabel("training ECGs (patient-grouped subsample)"); ax.set_title(f"{k} features / qubits"); ax.grid(alpha=.3)
axs[0].set_ylabel("test AUC (mean ± SD over folds×repeats)"); axs[1].legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig("results/figs/learning_curves.png", dpi=150)
# per-size comparison: quantum vs best classical (paired over (fold, rep))
cmp = ["", "| features | size | brick-wall / VQC | best classical | mean AUC diff | (fold,rep) wins | paired bootstrap 95% CI over (fold,rep) pairs (NOT patient-level) |", "|---|---|---|---|---|---|---|"]
rng = np.random.default_rng(0)
for k, q in ((8, "bw8"), (8, "vqc_pi6"), (12, "bw12")):
    for s in SIZES:
        def vals(m, kk): return {(r["fold"], r["rep"]): r["test_auc"] for r in LC if r["model"] == m and r["k"] == kk and r["size"] == s}
        qv = vals(q, 8 if q == "vqc_pi6" else k)
        cl = {m: vals(m, k) for m in ("lr", "lr2", "rf", "xgb")}
        if len(qv) < 15 or any(len(v) < 15 for v in cl.values()): continue
        best = max(cl, key=lambda m: np.mean(list(cl[m].values())))
        keys = sorted(qv); d = np.array([qv[x] - cl[best][x] for x in keys])
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        cmp.append(f"| {k} | {s} | {label[q]} {np.mean(list(qv.values())):.4f} | {label[best]} {np.mean(list(cl[best].values())):.4f} | {d.mean():+.4f} | {(d > 0).sum()}/{len(d)} | [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] |")
Path("results/lc/p5_table.md").write_text("\n".join(table + cmp), encoding="utf-8"); json.dump(summary, open("results/lc/p5_summary.json", "w"), indent=1)
print("\n".join(table + cmp))
