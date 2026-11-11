"""test-fold scores of the EXISTING angle VQCs (stored params), for paired bootstrap comparisons.  -> results/bw/vqc_scores.json"""
import json, numpy as np
import scale_cv as S, Vqc_fast8 as V
X, y, g, _, feats = S.load()
out = {}
specs = {"vqc_pi6_k8": ("results/stage2_steps1000.jsonl", dict(k=8, scale="pi/6")),
         "vqc_pi6_k16": ("results/stage2_steps1000.jsonl", dict(k=16, scale="pi/6")),
         "vqc_pi6_k32": ("results/stage2_steps1000.jsonl", dict(k=32, scale="pi/6")),
         "vqc_trained_k32": ("results/n2_trainscale.jsonl", dict(k=32, scale="pi/6"))}
for name, (f, flt) in specs.items():
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if all(r.get(a) == b for a, b in flt.items()) and r["selector"] == "l1" and r.get("split", "grouped5") == "grouped5"
            and (name != "vqc_trained_k32" or r.get("train_scale")) and (name == "vqc_trained_k32" or not r.get("train_scale", False))
            and r.get("steps", 1000) == 1000 and r.get("readout", "z") == "z"]
    byfold = {}
    for r in rows: byfold[r["fold"]] = r
    if name == "vqc_pi6_k32":      # fold 4 rerun with the deterministic selector (as in n4_clinical.py)
        byfold[4] = json.loads(open("results/n0_baseline_fold4.jsonl").readline())
    folds = []
    for fold in range(5):
        r = byfold[fold]
        Z, tr, va, te, _ = S.prep("grouped5", fold, "l1")
        sel = [feats.index(nm) for nm in r["features"]]
        T = np.zeros((len(y), V.N_FEATURES)); T[:, :r["k"]] = np.tanh(Z[:, sel])
        m = S.Linear(r["k"], eval(r["scale"], {"pi": np.pi}), r.get("readout", "z"), r.get("train_scale", False))
        s = m.proba(T[te], np.asarray(r["params"]))
        folds.append([round(float(a), 5) for a in s])
        assert abs(S.roc_auc_score(y[te], s) - r["test_auc"]) < 1e-9, (name, fold)
    out[name] = folds; print(name, "ok", flush=True)
json.dump(out, open("results/bw/vqc_scores.json", "w"))
