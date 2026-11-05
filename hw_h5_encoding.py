"""H5: training-free encoding score. Encoding only (ansatz theta = 0: Ry(angle) layers + CNOT ring), fold-0 train,
1000 samples per class -> class-mean density matrices rho_MI, rho_NORM (256x256). Report trace distance and fidelity,
and correlate (Spearman) with the trained test AUC in results/*.jsonl.

  python -u hw_h5_encoding.py   -> results/hw/h5_encoding.json, results/hw/h5_encoding.md, results/figs/encoding_score.png
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import sqrtm
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qiskit_vqc as Q
import Vqc_fast8 as V
import scale_cv as S

SCALES = ["pi/8", "pi/6", "pi/4", "pi/3", "pi/2", "2*pi/3", "pi"]
KS = [8, 16, 32]
N_PER_CLASS = 1000
OUT = Path("results/hw")

# fold-0 l1 feature order: logged list of the fold-0 l1 k=32 run (prefix = first k features)
row = Q.load_models()[(32, "pi/6")]
_, y, _, _, feats = S.load()
Z, tr, va, te, _ = S.prep("grouped5", 0, "l1")
sel = [feats.index(n) for n in row["features"]]
rng = np.random.default_rng(0)
idx = {c: rng.choice(tr[y[tr] == c], N_PER_CLASS, replace=False) for c in (0, 1)}
theta0 = np.zeros(V.N_ANSATZ)


def rho(Ang):
    psi = V.forward(Ang, theta0)
    return psi.T @ psi.conj() / len(psi)          # sum_b |psi_b><psi_b| / n


def fidelity(r1, r2):
    s = sqrtm(r1)
    return float(np.real(np.trace(sqrtm(s @ r2 @ s))) ** 2)


enc = []
for k in KS:
    for sc in SCALES:
        s = eval(sc, {"pi": np.pi})
        R = {}
        for c in (0, 1):
            A = np.zeros((N_PER_CLASS, V.N_FEATURES)); A[:, :k] = s * np.tanh(Z[np.ix_(idx[c], sel[:k])])
            R[c] = rho(A)
        D = R[1] - R[0]
        td = float(0.5 * np.abs(np.linalg.eigvalsh((D + D.conj().T) / 2)).sum())
        fid = fidelity(R[1], R[0])
        # purity of the class-mean states (how much the encoding spreads samples over Hilbert space)
        pur = float(np.real(np.trace(R[0] @ R[0]) + np.trace(R[1] @ R[1])) / 2)
        enc.append(dict(k=k, scale=sc, trace_distance=td, fidelity=fid, purity=pur))
        print(f"k={k:2d} {sc:7s} trace distance={td:.4f} fidelity={fid:.4f} purity={pur:.4f}", flush=True)
enc = pd.DataFrame(enc)

# trained AUC: longest-steps l1 / grouped5 / seed 0 / unshuffled runs, per (k, scale); fold 0 and 5-fold mean
runs = []
for f in Path("results").glob("*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("split", "grouped5") == "grouped5" and r.get("selector", "l1") == "l1" and not r.get("shuffle_later", False) \
                and r.get("seed", 0) == 0 and "test_auc" in r:
            runs.append(dict(k=r["k"], scale=r["scale"], fold=r["fold"], steps=r.get("steps", 300), test_auc=r["test_auc"]))
runs = pd.DataFrame(runs).drop_duplicates(["k", "scale", "fold", "steps"])
runs = runs[runs.steps == runs.groupby(["k", "scale"]).steps.transform("max")]
auc = runs.groupby(["k", "scale"]).agg(steps=("steps", "first"), auc_mean=("test_auc", "mean"), n_folds=("fold", "nunique"),
                                       auc_fold0=("test_auc", lambda s: s[runs.loc[s.index, "fold"] == 0].mean())).reset_index()
m = enc.merge(auc, on=["k", "scale"], how="left")
res = {"per_config": m.to_dict("records"), "spearman": {}}
have = m.dropna(subset=["auc_mean"])
for name, sub in (("all configs with trained runs", have), ("k=32 only (7 scales)", have[have.k == 32])):
    for metric in ("trace_distance", "fidelity"):
        for target in ("auc_mean", "auc_fold0"):
            rho_s, p = spearmanr(sub[metric], sub[target])
            res["spearman"][f"{name} | {metric} vs {target}"] = dict(rho=float(rho_s), p=float(p), n=int(len(sub)))
            print(f"{name:32s} {metric:15s} vs {target:9s}: Spearman rho={rho_s:+.3f} (p={p:.3f}, n={len(sub)})")
json.dump(res, open(OUT / "h5_encoding.json", "w"), indent=1, default=float)

md = "| k | scale | trace distance | fidelity | purity | trained steps | test AUC (5-fold mean) | test AUC fold 0 |\n|---|---|---|---|---|---|---|---|\n"
for _, r in m.iterrows():
    a = "–" if np.isnan(r.auc_mean) else f"{r.auc_mean:.4f}"
    a0 = "–" if np.isnan(r.auc_fold0) else f"{r.auc_fold0:.4f}"
    st = "–" if np.isnan(r.auc_mean) else int(r.steps)
    md += f"| {r.k} | {r.scale} | {r.trace_distance:.4f} | {r.fidelity:.4f} | {r.purity:.4f} | {st} | {a} | {a0} |\n"
(OUT / "h5_encoding.md").write_text(md)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
sv = lambda s: eval(s, {"pi": np.pi})
for k, c in zip(KS, ("C0", "C1", "C2")):
    e = enc[enc.k == k].assign(sv=lambda d: d.scale.map(sv)).sort_values("sv")
    axs[0].plot(e.sv, e.trace_distance, "o-", color=c, label=f"k={k}")
axs[0].set_xscale("log"); axs[0].minorticks_off(); xs = [sv(s) for s in SCALES]
axs[0].set_xticks(xs); axs[0].set_xticklabels([s.replace("*", "").replace("pi", "π") for s in SCALES])
axs[0].set_xlabel("angle scale s (log axis)"); axs[0].set_ylabel("trace distance ρ_MI vs ρ_NORM"); axs[0].legend()
axs[0].set_title("encoding only (θ = 0), fold-0 train")
for k, c in zip(KS, ("C0", "C1", "C2")):
    h = have[have.k == k]
    axs[1].scatter(h.trace_distance, h.auc_mean, color=c, label=f"k={k}")
    for _, r in h.iterrows():
        axs[1].annotate(r.scale.replace("*", "").replace("pi", "π"), (r.trace_distance, r.auc_mean), fontsize=7,
                        xytext=(3, 2), textcoords="offset points")
rs = res["spearman"]["all configs with trained runs | trace_distance vs auc_mean"]
axs[1].set_xlabel("trace distance (training-free)"); axs[1].set_ylabel("trained test AUC (5-fold mean)")
axs[1].set_title(f"Spearman ρ = {rs['rho']:+.2f} (n = {rs['n']})"); axs[1].legend()
fig.tight_layout(); fig.savefig("results/figs/encoding_score.png", dpi=150)
print(md)
