"""H2: finite-shot robustness of the 6 models (own simulator, exact probabilities + multinomial sampling).

  python -u hw_h2_shots.py   -> results/hw/h2_shots.jsonl, results/hw/h2_shots.md, results/figs/hw_shots.png
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qiskit_vqc as Q
import Vqc_fast8 as V

SHOTS = [30, 100, 300, 1000, 2000, 4000, 10000]          # runbook grid {100,300,1000,4000,10000} + 30, 2000
REPEATS = 10
OUT = Path("results/hw"); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
rows, summ = [], {}
for key, row in Q.load_models().items():
    A, y, tr, va, te = Q.model_data(row); th, w, b = Q.split_params(row)
    psi = V.forward(A[te], th); P = psi.real ** 2 + psi.imag ** 2
    P /= P.sum(1, keepdims=True)
    z_exact = P @ V._ZSIGN.T
    auc0 = roc_auc_score(y[te], Q.logit_from_z(z_exact, w, b))
    # how far apart the classes are in readout space, relative to the shot-noise scale
    sep = abs(np.mean(z_exact[y[te] == 1] @ w) - np.mean(z_exact[y[te] == 0] @ w))
    summ[key] = dict(noiseless=auc0, w_norm=float(np.linalg.norm(w)), logit_sep=float(sep),
                     z_class_gap=float(np.abs(z_exact[y[te] == 1].mean(0) - z_exact[y[te] == 0].mean(0)).max()))
    for s in SHOTS:
        aucs = []
        for r in range(REPEATS):
            counts = rng.multinomial(s, P)                       # (N, 256)
            z = counts @ V._ZSIGN.T / s
            aucs.append(roc_auc_score(y[te], Q.logit_from_z(z, w, b)))
        rows.append(dict(k=key[0], scale=key[1], shots=s, auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs, ddof=1)),
                         noiseless=auc0, loss=float(auc0 - np.mean(aucs)), aucs=aucs))
        print(f"k={key[0]:2d} {key[1]:5s} shots={s:5d} AUC={np.mean(aucs):.4f}±{np.std(aucs, ddof=1):.4f} "
              f"(noiseless {auc0:.4f}, loss {auc0-np.mean(aucs):+.4f})", flush=True)

with open(OUT / "h2_shots.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

# shots needed to stay within 0.01 (and 0.005) of noiseless: smallest grid value, log-interpolated
def shots_needed(key, tol):
    xs = [r for r in rows if (r["k"], r["scale"]) == key]
    for a, b in zip(xs, xs[1:]):
        if a["loss"] <= tol:
            return a["shots"]
        if b["loss"] <= tol:
            t = (np.log(a["loss"]) - np.log(tol)) / (np.log(a["loss"]) - np.log(b["loss"]))
            return int(round(np.exp(np.log(a["shots"]) + t * (np.log(b["shots"]) - np.log(a["shots"])))))
    return None

md = "| model | noiseless | " + " | ".join(f"{s} shots" for s in SHOTS) + " | shots for loss ≤ 0.01 | ≤ 0.005 | ‖w‖ |\n"
md += "|---|---|" + "---|" * len(SHOTS) + "---|---|---|\n"
for key in Q.MODEL_KEYS:
    xs = {r["shots"]: r for r in rows if (r["k"], r["scale"]) == key}
    md += (f"| k={key[0]} {key[1]} | {summ[key]['noiseless']:.4f} | "
           + " | ".join(f"{xs[s]['auc_mean']:.4f} ± {xs[s]['auc_sd']:.4f}" for s in SHOTS)
           + f" | {shots_needed(key, 0.01)} | {shots_needed(key, 0.005)} | {summ[key]['w_norm']:.2f} |\n")
(OUT / "h2_shots.md").write_text(md); print(md)
json.dump({f"k{k}_{s}": v | {"shots_0.01": shots_needed((k, s), 0.01), "shots_0.005": shots_needed((k, s), 0.005)}
           for (k, s), v in summ.items()}, open(OUT / "h2_summary.json", "w"), indent=1)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
sty = {"pi": ("C2", "π"), "pi/2": ("C1", "π/2"), "pi/6": ("C0", "π/6")}
for ax, k in zip(axs, (8, 32)):
    for sc in ("pi", "pi/2", "pi/6"):
        xs = [r for r in rows if r["k"] == k and r["scale"] == sc]
        c, lab = sty[sc]
        ax.errorbar([r["shots"] for r in xs], [r["auc_mean"] for r in xs], yerr=[r["auc_sd"] for r in xs],
                    marker="o", capsize=3, color=c, label=f"s={lab}")
        ax.axhline(xs[0]["noiseless"], color=c, ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("shots per ECG"); ax.set_ylabel("test AUC (fold 0, 2,993 ECGs)")
    ax.set_title(f"k = {k}  (dotted = noiseless)"); ax.legend()
fig.tight_layout(); Path("results/figs").mkdir(exist_ok=True); fig.savefig("results/figs/hw_shots.png", dpi=150)
