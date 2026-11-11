"""P4 (torch venv): train the FINAL brick-wall configs on ALL PTB-XL (validation split from PTB-XL only), score the PTB external set once.
Reads results/bw/final_configs.json (written after P3 stage C) and C:/bwcache/full.npz (made by p4_external.py prep).
  C:/bwvenv/Scripts/python.exe p4_bw.py [--seeds 0 1 2]
"""
import json, sys, argparse, time
import numpy as np, torch
import bw_torch as T
torch.set_num_threads(1)
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="*", type=int, default=[0]); ap.add_argument("--tag", default="bw")
ap.add_argument("--configs", default="results/bw/final_configs.json"); a = ap.parse_args()
d = np.load("C:/bwcache/full.npz", allow_pickle=True)
Z = d["Z"].astype(np.float64); y = d["y"]; tr, va, te = d["tr"], d["va"], d["te"]
cfgs = json.load(open(a.configs))
for key, cfg in cfgs.items():
    n = cfg["n"]
    for seed in a.seeds:
        mdl = T.Model(n, cfg["columns"], cfg["k"], cfg["scale"], head=cfg["head"], seed=seed)
        v, bstep, tot, secs = T.train_model(mdl, Z[tr][:, :n], y[tr], Z[va][:, :n], y[va], steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"], seed=seed)
        s = mdl.predict(Z[te][:, :n])
        r = dict(model=f"{a.tag}{n}" + (f"_s{seed}" if seed else ""), k=n, family="brickwall", cfg=cfg, seed=seed, val_auc=float(v), best_step=bstep,
                 ext_auc=T.auc(y[te], s), secs=round(secs, 1), scores=[round(float(x), 5) for x in s], params=mdl.state())
        open("results/ext/scores.jsonl", "a").write(json.dumps(r) + "\n")
        print(r["model"], "val", round(v, 4), "ext", round(r["ext_auc"], 4), secs, flush=True)
