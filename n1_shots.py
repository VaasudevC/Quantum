"""N1 add-on: finite-shot AUC of the zz readout vs the z readout (k=32, pi/6), as in H2.

All 8 <Z_j> and 28 <Z_iZ_j> come from the same Z-basis shots (no extra circuits). Multinomial sampling of the exact
probabilities, 10 repeats, every test ECG of every fold; mean over folds of the per-fold mean AUC.

  python -u n1_shots.py  -> results/n1_shots.json
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import scale_cv as S
import Vqc_fast8 as V

SHOTS, REPEATS = [180, 1000], 10
_, y, _, _, feats = S.load()
rng = np.random.default_rng(0)


def rows(f):
    return {json.loads(l)["fold"]: json.loads(l) for l in open(f)}


z_rows = rows("results/stage2_steps1000.jsonl")
z_rows = {f: r for f, r in z_rows.items()} if False else {
    r["fold"]: r for r in map(json.loads, open("results/stage2_steps1000.jsonl")) if r["k"] == 32 and r["scale"] == "pi/6"}
try:
    z_rows.update(rows("results/n0_baseline_fold4.jsonl"))
except FileNotFoundError:
    pass
models = {"z": z_rows, "zz": rows("results/n1_zz.jsonl")}
out = {}
for name, rs in models.items():
    per = {s: [] for s in ["exact"] + SHOTS}
    for f, r in sorted(rs.items()):
        Z, tr, va, te, _ = S.prep(r["split"], f, r["selector"])
        sel = [feats.index(n) for n in r["features"]]
        T = np.zeros((len(y), V.N_FEATURES)); T[:, :32] = np.tanh(Z[:, sel])
        m = S.Linear(32, np.pi / 6, r.get("readout", "z"), False)
        th, w, b, s = m.split(np.asarray(r["params"]))
        psi = V.forward(m.angles(T[te], s), th); P = psi.real ** 2 + psi.imag ** 2; P /= P.sum(1, keepdims=True)
        sg = V.READOUT_SIGNS[m.ro].T
        per["exact"].append(roc_auc_score(y[te], P @ sg @ w + b))
        for sh in SHOTS:
            per[sh].append(np.mean([roc_auc_score(y[te], (rng.multinomial(sh, P) / sh) @ sg @ w + b) for _ in range(REPEATS)]))
    out[name] = {str(k): dict(mean=float(np.mean(v)), sd_folds=float(np.std(v, ddof=1)), n_folds=len(v)) for k, v in per.items()}
    print(name, {k: round(v["mean"], 4) for k, v in out[name].items()}, flush=True)
json.dump(out, open("results/n1_shots.json", "w"), indent=1)
