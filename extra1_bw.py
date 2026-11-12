"""EXTRA 1: seeds 1,2 (5 folds) of the FINAL brick-wall configs (seed 0 = stage C rank 0).  -> results/bw/extra1_bw.jsonl"""
import json, sys
import p3_brickwall as P
cfgs = json.load(open("results/bw/final_configs.json")); out = []
for seed in (1, 2):
    for n in ("8", "12"):
        c = cfgs[n]
        for fold in range(5):
            out.append(P.base(n=c["n"], columns=c["columns"], k=c["k"], scale=c["scale"], head=c["head"], lr=c["lr"], steps=c["steps"], fold=fold, seed=seed, save_scores=True))
P.runner(out, "results/bw/extra1_bw.jsonl", int(sys.argv[1]) if len(sys.argv) > 1 else 4)
