"""N3: kernel-target alignment (KTA) of the encoding-only circuit as a training-free encoding score.

K_ij = |<phi(x_i)|phi(x_j)>|^2 with phi = encoding only (theta = 0: Ry(angle) layers + CNOT ring), 1000 class-stratified
fold-0 train samples, y in {-1,+1}.  KTA = <K, yy^T>_F / (||K||_F * n)   (runbook definition, uncentered).
Also reported: centered KTA (Cortes et al. 2012; K and yy^T double-centered), which removes the class-imbalance offset.
Spearman vs trained test AUC (same 13 configs as H5), compared with H5's trace distance.

  python -u n3_kta.py  -> results/n3_kta.json, results/n3_kta.md, results/figs/kta_vs_auc.png
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import Vqc_fast8 as V
import scale_cv as S

SCALES = ["pi/8", "pi/6", "pi/4", "pi/3", "pi/2", "2*pi/3", "pi"]
KS = [8, 16, 32]
N = 1000
h5 = pd.DataFrame(json.load(open("results/hw/h5_encoding.json"))["per_config"])      # trace distance + trained AUC

_, y, _, _, feats = S.load()
row = [json.loads(l) for l in open("results/stage2_steps1000.jsonl")
       if '"fold": 0,' in l and '"k": 32, "scale": "pi/6"' in l][0]
sel = [feats.index(f) for f in row["features"]]                   # fold-0 l1 order (as in H5)
Z, tr, va, te, _ = S.prep("grouped5", 0, "l1")
rng = np.random.default_rng(0)
pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
n_pos = int(round(N * len(pos) / len(tr)))
idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, N - n_pos, replace=False)])
yy = np.where(y[idx] == 1, 1.0, -1.0); Y = np.outer(yy, yy)
H = np.eye(N) - 1.0 / N


def kta(K, Yt):
    return float((K * Yt).sum() / (np.linalg.norm(K) * np.linalg.norm(Yt)))


out = []
for k in KS:
    for sc in SCALES:
        A = np.zeros((N, V.N_FEATURES)); A[:, :k] = eval(sc, {"pi": np.pi}) * np.tanh(Z[np.ix_(idx, sel[:k])])
        psi = V.forward(A, np.zeros(V.N_ANSATZ))
        K = np.abs(psi.conj() @ psi.T) ** 2
        Kc, Yc = H @ K @ H, H @ Y @ H
        out.append(dict(k=k, scale=sc, kta=kta(K, Y), kta_centered=kta(Kc, Yc),
                        mean_offdiag_K=float((K.sum() - N) / (N * N - N))))
        print(f"k={k:2d} {sc:7s} KTA={out[-1]['kta']:.4f} centered={out[-1]['kta_centered']:.4f} "
              f"mean K_ij={out[-1]['mean_offdiag_K']:.4f}", flush=True)
m = pd.DataFrame(out).merge(h5[["k", "scale", "trace_distance", "purity", "auc_mean", "auc_fold0", "steps"]], on=["k", "scale"])
have = m.dropna(subset=["auc_mean"])
res = {"per_config": m.to_dict("records"), "spearman": {}}
for name, sub in (("all 13 configs", have), ("k=32 only (7)", have[have.k == 32])):
    for metric in ("kta", "kta_centered", "trace_distance"):
        r, p = spearmanr(sub[metric], sub["auc_mean"])
        res["spearman"][f"{name} | {metric}"] = dict(rho=float(r), p=float(p), n=int(len(sub)))
        print(f"{name:16s} {metric:15s} vs AUC: rho={r:+.3f} p={p:.3f}")
Path("results/figs").mkdir(exist_ok=True)
json.dump(res, open("results/n3_kta.json", "w"), indent=1, default=float)

md = "| k | scale | KTA | centered KTA | mean K_ij | trace distance (H5) | trained AUC (5-fold) |\n|---|---|---|---|---|---|---|\n"
for _, r in m.iterrows():
    a = "–" if np.isnan(r.auc_mean) else f"{r.auc_mean:.4f}"
    md += f"| {r.k} | {r.scale} | {r.kta:.4f} | {r.kta_centered:.4f} | {r.mean_offdiag_K:.4f} | {r.trace_distance:.4f} | {a} |\n"
md += "\n| Spearman vs trained AUC | KTA | centered KTA | trace distance |\n|---|---|---|---|\n"
for name in ("all 13 configs", "k=32 only (7)"):
    g = lambda mt: res["spearman"][f"{name} | {mt}"]
    md += f"| {name} | {g('kta')['rho']:+.2f} (p={g('kta')['p']:.3f}) | {g('kta_centered')['rho']:+.2f} (p={g('kta_centered')['p']:.3f}) | {g('trace_distance')['rho']:+.2f} (p={g('trace_distance')['p']:.3f}) |\n"
Path("results/n3_kta.md").write_text(md); print(md)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, mt, ttl in ((axs[0], "kta", "KTA (runbook, uncentered)"), (axs[1], "kta_centered", "centered KTA")):
    for k, c in zip(KS, ("C0", "C1", "C2")):
        h = have[have.k == k]
        ax.scatter(h[mt], h.auc_mean, color=c, label=f"k={k}")
        for _, r in h.iterrows():
            ax.annotate(r.scale.replace("*", "").replace("pi", "π"), (r[mt], r.auc_mean), fontsize=7, xytext=(3, 2), textcoords="offset points")
    rs = res["spearman"][f"all 13 configs | {mt}"]
    ax.set_xlabel(f"{ttl}, encoding only, fold-0 train"); ax.set_ylabel("trained test AUC (5-fold mean)")
    ax.set_title(f"Spearman ρ = {rs['rho']:+.2f} (n = {rs['n']})"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/figs/kta_vs_auc.png", dpi=150)
