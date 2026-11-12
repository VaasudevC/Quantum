"""P6c: shot requirement of the final brick-wall models: AUC vs number of shots, sampling from the EXACT output distribution (noiseless), 300 stratified fold-0 test ECGs."""
import json, numpy as np, torch
import bw_torch as T
torch.set_num_threads(1)
d = np.load("C:/bwcache/l1_f0.npz"); y = d["y"]; te = d["te"]; rng = np.random.default_rng(0)
pos, neg = np.flatnonzero(y[te] == 1), np.flatnonzero(y[te] == 0)
npos = int(round(300 * len(pos) / len(te))); sub = te[np.sort(np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, 300 - npos, replace=False)]))]
ys = y[sub]; rows = [json.loads(l) for l in open("results/bw/p3_stageC.jsonl")]; out = {}
for n in (8, 12):
    r = [r for r in rows if r["n"] == n and r["fold"] == 0 and r.get("rank") == 0][0]
    Z = d["Z"].astype(np.float64)[:, :n]
    m = T.Model(n, r["columns"], r["k"], r["scale"], head=r["head"], dtype=torch.complex128); m.load(r["params"])
    h = m.h.detach().numpy().astype(float); b = float(m.b)
    with torch.no_grad():
        psi = m.bw.psi(m.theta(Z[sub]), m.w); P = (psi.real ** 2 + psi.imag ** 2).numpy()
    P = P / P.sum(1, keepdims=True); M = m.bw.M_adj.numpy()
    res = {"head_norm_inf": float(np.abs(h).max()), "exact": T.auc(ys, 4 * P @ M @ h + b), "feature_sd_across_ecgs_mean": float((4 * P @ M).std(0).mean())}
    for sh in (100, 300, 1000, 4000, 16000, 64000, 256000, 1000000):
        f = np.array([4 * (rng.multinomial(sh, p) @ M) / sh for p in P]); res[str(sh)] = T.auc(ys, f @ h + b)
    out[str(n)] = res; print(n, {k: round(v, 4) for k, v in res.items()}, flush=True)
json.dump(out, open("results/bw/p6_shots.json", "w"), indent=1)
