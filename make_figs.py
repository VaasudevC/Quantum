"""Stage 6 figures -> results/figs/ (150 dpi).

  python make_figs.py --sstar pi/6 --steps 1000
(a) test AUC vs scale at k=32 (Stage 1), fold SD error bars, LR dashed
(b) test AUC vs k for pi, pi/2, s* (Stage 2 at --steps), LR dashed
(c) shuffle-control drop (real - shuffled) per scale x k (Stage 4)
(d) Stage 4b: VQC vs linear LR vs degree-2 LR per fold (if results/stage4b_poly.jsonl exists)
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--sstar", default="pi/6"); ap.add_argument("--steps", type=int, default=1000)
a = ap.parse_args()
R, OUT = Path("results"), Path("results/figs"); OUT.mkdir(parents=True, exist_ok=True)


def load(*names):
    rows = [json.loads(l) for n in names if (R / n).exists() for l in open(R / n)]
    r = pd.DataFrame(rows).drop(columns=["features", "params"], errors="ignore")
    r["sv"] = [eval(s, {"pi": np.pi}) for s in r.scale]
    return r


def label(s):
    return s.replace("*", "").replace("pi", "π")


# (a)
r = load("stage1_sweep.jsonl"); g = r.groupby("sv")
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.errorbar(g.test_auc.mean().index, g.test_auc.mean(), yerr=g.test_auc.std(), marker="o", capsize=3, label="VQC (8 qubits)")
ax.axhline(r.lr_auc.mean(), ls="--", c="k", lw=1, label="LR, same 32 features")
ax.set_xscale("log"); ax.minorticks_off()
ax.set_xticks(g.test_auc.mean().index); ax.set_xticklabels([label(s) for s in g.scale.first()])
ax.set_xlabel("angle scale s  (angle = s·tanh z), log axis"); ax.set_ylabel("test AUC (5-fold grouped CV)")
ax.set_title("k = 32 features, 600 steps"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "a_auc_vs_scale_k32.png", dpi=150)

# (b)
r = load("stage2_steps1000.jsonl", f"stage2_steps{a.steps}.jsonl")
r = r[(r.steps == a.steps) | (r.scale != a.sstar)]
r = r[r.steps == r.groupby("scale").steps.transform("max")]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for sc in [a.sstar, "pi/2", "pi"]:
    x = r[r.scale == sc].groupby("k").test_auc
    ax.errorbar(x.mean().index, x.mean(), yerr=x.std(), marker="o", capsize=3,
                label=f"VQC s={label(sc)} ({int(r[r.scale == sc].steps.iloc[0])} steps)")
lr = r.groupby("k").lr_auc.mean(); ax.plot(lr.index, lr, "k--", lw=1, label="LR, same features")
ax.set_xticks([8, 16, 32]); ax.set_xlabel("k (features re-uploaded on 8 qubits)"); ax.set_ylabel("test AUC")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "b_auc_vs_k.png", dpi=150)

# (c)
re, sh = load("stage4_real.jsonl"), load("stage4_shuffled.jsonl")
if len(re) and len(sh):
    key = ["fold", "k", "scale"]
    m = re.merge(sh, on=key, suffixes=("_r", "_s")); m["drop"] = m.test_auc_r - m.test_auc_s
    d = m.groupby(["scale", "k"]).drop.agg(["mean", "std"]).reset_index()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8)); w = 0.35
    for i, sc in enumerate([a.sstar, "pi"]):
        x = d[d.scale == sc]
        ax.bar(np.arange(len(x)) + (i - .5) * w, x["mean"], w, yerr=x["std"], capsize=3, label=f"s={label(sc)}")
    ax.set_xticks(range(len(x))); ax.set_xticklabels([f"k={k}" for k in x.k])
    ax.set_ylabel("AUC drop, real − shuffled features 9..k"); ax.legend(); ax.set_title("shuffle drop")
    # right: change vs the k=8 model (no later features), real and shuffled
    b8 = load("stage2_steps1000.jsonl"); b8 = b8[b8.k == 8].set_index(["scale", "fold"]).test_auc
    labels, w2 = [], 0.2
    for j, (nm, df) in enumerate([("real", re), ("shuffled", sh)]):
        for i, sc in enumerate([a.sstar, "pi"]):
            v = [(df[(df.scale == sc) & (df.k == k)].set_index(["scale", "fold"]).test_auc - b8.loc[sc]) for k in (16, 32)]
            ax2.bar(np.arange(2) + (2 * j + i - 1.5) * w2, [x.mean() for x in v], w2, yerr=[x.std() for x in v],
                    capsize=3, label=f"{nm}, s={label(sc)}", hatch="//" if nm == "shuffled" else None,
                    color=f"C{i}", alpha=1 if nm == "real" else .55)
    ax2.axhline(0, c="k", lw=.8); ax2.set_xticks(range(2)); ax2.set_xticklabels(["k=16", "k=32"])
    ax2.set_ylabel("test AUC − same-scale k=8 model"); ax2.set_title("effect of later features vs k=8"); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "c_shuffle_control.png", dpi=150)

# (d)
p = R / "stage4b_poly.jsonl"
if p.exists():
    q = pd.DataFrame([json.loads(l) for l in open(p)]).sort_values("fold")
    fig, ax = plt.subplots(figsize=(5.5, 3.8)); w = 0.27; x = np.arange(len(q))
    for i, (c, l) in enumerate([("lr1_test_auc", "linear LR"), ("vqc_test_auc", f"VQC s={label(a.sstar)}"),
                                ("poly2_test_auc", "degree-2 LR")]):
        ax.bar(x + (i - 1) * w, q[c], w, label=l)
    ax.set_xticks(x); ax.set_xticklabels([f"fold {f}" for f in q.fold]); ax.set_ylim(0.9, 0.98)
    ax.set_ylabel("test AUC (k=32)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "d_poly2_vs_vqc.png", dpi=150)

print("figures ->", sorted(str(f) for f in OUT.glob("*.png")))
