"""EXTRA 6: shot-robust brick-wall heads.  Retrain the final config with an L2 penalty on the head and/or a max-norm clamp |h_j| <= c,
selecting (checkpoint AND variant) on VALIDATION AUC at 1000 simulated shots (multinomial, mean of 5 repeats).
  stage 1: folds 0-1, variants l2 in {1e-4,1e-3,1e-2}, maxnorm in {10,30};  stage 2: winner on 5 folds.
Reports exact and 1000-shot (5 repeats) TEST AUC for old (stage C) and new models.   -> results/bw/extra6_n{n}.jsonl
  C:/bwvenv/Scripts/python.exe extra6_shotrobust.py N JOBS"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

def run(cfg):
    import torch, bw_torch as T
    torch.set_num_threads(1)
    n, fold = cfg["n"], cfg["fold"]
    d = np.load(f"C:/bwcache/l1_f{fold}.npz"); Z = d["Z"].astype(np.float64)[:, :n]; y = d["y"]; tr, va, te = d["tr"], d["va"], d["te"]
    m = T.Model(n, cfg["columns"], cfg["k"], cfg["scale"], head="linear", seed=fold)
    t0 = time.time()
    v, bstep, tot, _ = T.train_model(m, Z[tr], y[tr], Z[va], y[va], steps=cfg["steps"], batch=512, lr=0.02, seed=fold, l2=cfg["l2"], maxnorm=cfg["maxnorm"], select="shots")
    return dict(cfg, val_shot_auc=float(v), val_exact=T.auc(y[va], m.predict(Z[va])), test_exact=T.auc(y[te], m.predict(Z[te])),
                test_shot1000=T.shot_auc(m, Z[te], y[te], 1000, 5, seed=99), max_h=float(m.h.abs().max()), best_step=bstep, total_steps=tot,
                secs=round(time.time() - t0), params=m.state())

def old_model(n, fold):
    import torch, bw_torch as T
    r = [json.loads(l) for l in open("results/bw/p3_stageC.jsonl")]; r = [x for x in r if x["n"] == n and x["fold"] == fold and x["rank"] == 0][0]
    d = np.load(f"C:/bwcache/l1_f{fold}.npz"); Z = d["Z"].astype(np.float64)[:, :n]; y = d["y"]; te = d["te"]
    m = T.Model(n, r["columns"], r["k"], r["scale"], head="linear"); m.load(r["params"])
    return dict(fold=fold, test_exact=T.auc(y[te], m.predict(Z[te])), test_shot1000=T.shot_auc(m, Z[te], y[te], 1000, 5, seed=99), max_h=float(m.h.abs().max()))

def key(c): return (c["fold"], c["l2"], c["maxnorm"])

def go(cfgs, out, jobs):
    done = {key(json.loads(l)) for l in open(out)} if Path(out).exists() else set()
    todo = [c for c in cfgs if key(c) not in done]; print(len(todo), "to run", flush=True)
    with ProcessPoolExecutor(jobs) as ex:
        for r in ex.map(run, todo):
            open(out, "a").write(json.dumps(r) + "\n")
            print(f"f{r['fold']} l2={r['l2']} mn={r['maxnorm']} val_shot={r['val_shot_auc']:.4f} test exact={r['test_exact']:.4f} shot={r['test_shot1000']:.4f} max|h|={r['max_h']:.1f} {r['secs']}s", flush=True)

def main():
    n = int(sys.argv[1]); jobs = int(sys.argv[2]); out = f"results/bw/extra6_n{n}.jsonl"
    c0 = json.load(open("results/bw/final_configs.json")).get(str(n)) or dict(n=16, columns=4, k=4, scale=0.7853981633974483, steps=1500)
    base = dict(n=n, columns=c0["columns"], k=c0["k"], scale=c0["scale"], steps=1500)
    variants = [(1e-4, None), (1e-3, None), (1e-2, None), (0.0, 10.0), (0.0, 30.0)]
    if n == 16:   # time: only the variants that won/held at n=12
        w = json.load(open("results/bw/extra6_n12_winner.json")); variants = [(w["l2"], w["maxnorm"])]
    go([dict(base, fold=f, l2=a, maxnorm=b) for f in (0, 1) for a, b in variants], out, jobs)
    rows = [json.loads(l) for l in open(out)]
    best = max(variants, key=lambda v: np.mean([r["val_shot_auc"] for r in rows if (r["l2"], r["maxnorm"]) == v and r["fold"] in (0, 1)]))
    print("winner", best, flush=True); json.dump(dict(l2=best[0], maxnorm=best[1]), open(f"results/bw/extra6_n{n}_winner.json", "w"))
    go([dict(base, fold=f, l2=best[0], maxnorm=best[1]) for f in range(5)], out, jobs)
    old = [old_model(n, f) for f in range(5)]; json.dump(old, open(f"results/bw/extra6_n{n}_old.json", "w"))

if __name__ == "__main__":
    main()
