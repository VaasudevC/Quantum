"""P3: brick-wall QNN on ECG (PTB-XL MI vs NORM), one feature per qubit.  Resumable jsonl runner.

  python -u p3_brickwall.py --stage A --jobs 3
Stages: A (folds 0-1: scale x k x columns), B (folds 0-1: columns=n / linear_score / corr head for the winner),
C (5 folds, final + runner-up), ref (paper phase map).  Stage configs for B/C are derived from earlier results (validation AUC).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bw_torch as T0
CACHE = "C:/bwcache"


def load_fold(fold, sel="l1"):
    d = np.load(f"{CACHE}/{sel}_f{fold}.npz", allow_pickle=False)
    return d["Z"].astype(np.float64), d["y"], d["tr"], d["va"], d["te"], list(d["names"])

RES = Path("results/bw")


def sens90(y, s):
    return T0.sens_at_spec(y, s, 0.90)


def run_one(cfg):
    import torch, bw_torch as T
    torch.set_num_threads(1)
    Z, y, tr, va, te, names = load_fold(cfg["fold"], cfg.get("selector", "l1"))
    n = cfg["n"]; Zs = Z[:, :n]
    xmax = np.abs(Zs[tr]).max(0)
    m = T.Model(n, cfg["columns"], cfg["k"], cfg["scale"] if cfg["phase"] == "tanh" else None, head=cfg["head"],
                phase=cfg["phase"], xmax=xmax, seed=cfg["seed"] + cfg["fold"])
    vauc, bstep, tot, secs = T.train_model(m, Zs[tr], y[tr], Zs[va], y[va], steps=cfg["steps"], batch=cfg["batch"],
                                           lr=cfg["lr"], seed=cfg["seed"] + cfg["fold"])
    s = m.predict(Zs[te])
    out = dict(cfg, val_auc=float(vauc), test_auc=float(T0.auc(y[te], s)), sens90=sens90(y[te], s),
               best_step=bstep, total_steps=tot, secs=round(secs, 1), features=names[:n], params=m.state())
    if cfg.get("save_scores"):
        out["test_scores"] = [round(float(a), 5) for a in s]
    return out


def key(c):
    return (c["n"], c["columns"], c["k"], c["scale"], c["phase"], c["head"], c["fold"], c["steps"], c["lr"], c["seed"])


def runner(cfgs, out, jobs):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {key(json.loads(l)) for l in open(out)} if out.exists() else set()
    todo = [c for c in cfgs if key(c) not in done]
    print(f"{len(cfgs)} configs, {len(todo)} to run -> {out}", flush=True)
    if jobs <= 1:
        it = (run_one(c) for c in todo)
        for r in it:
            _emit(r, out)
        return
    with ProcessPoolExecutor(jobs) as ex:
        for fu in as_completed([ex.submit(run_one, c) for c in todo]):
            _emit(fu.result(), out)


def _emit(r, out):
    with open(out, "a") as fo:
        fo.write(json.dumps(r) + "\n")
    print(f"n={r['n']} c={r['columns']} k={r['k']} s={r['scale']:.3f} {r['phase']} {r['head']} f{r['fold']} "
          f"val={r['val_auc']:.4f} test={r['test_auc']:.4f} best@{r['best_step']}/{r['total_steps']} {r['secs']}s", flush=True)


def base(**kw):
    d = dict(n=8, columns=2, k=2, scale=float(np.pi / 6), phase="tanh", head="linear", fold=0, steps=1500, batch=512, lr=0.02, seed=0)
    d.update(kw); return d


def cli():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if a.stage in ("B", "C", "ref"):
        return stage_main(a.stage, a.jobs)
    if a.stage == "A":
        cfgs = []
        for fold in (0, 1):
            for n, ks in ((8, (2, 4)), (12, (2, 4, 6))):
                for k in ks:
                    for cols in (2, 4):
                        for sc in (np.pi / 6, np.pi / 4, np.pi / 2, np.pi):
                            cfgs.append(base(n=n, k=k, columns=cols, scale=float(sc), fold=fold))
        runner(cfgs, RES / "p3_stageA2.jsonl", a.jobs)


# ---------------------------------------------------------------------------------------------------------------------
# stage B / C / ref : configs derived from earlier results (validation AUC only)
# ---------------------------------------------------------------------------------------------------------------------
def load_rows(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def cfg_of(r):
    return {k: r[k] for k in ("n", "columns", "k", "scale", "phase", "head", "steps", "batch", "lr", "seed")}


def group_mean_val(rows, folds=(0, 1)):
    g = {}
    for r in rows:
        if r["fold"] in folds:
            g.setdefault(tuple(sorted(cfg_of(r).items())), []).append(r["val_auc"])
    return {k: float(np.mean(v)) for k, v in g.items() if len(v) == len(folds)}


def stage_main(stage, jobs):
    A = load_rows(RES / "p3_stageA2.jsonl"); Bq = load_rows(RES / "p3_stageB.jsonl")
    if stage == "B":
        cfgs = []
        for n in (8, 12):
            gm = group_mean_val([r for r in A if r["n"] == n])
            best = dict(max(gm.items(), key=lambda kv: kv[1])[0])
            print("stage A winner", n, best, flush=True)
            variants = (dict(columns=n), dict(head="linear_score"), dict(head="corr"), dict(columns=n, head="corr"), dict(lr=0.06), dict(lr=0.06, columns=n)) if n == 8                 else (dict(head="linear_score"), dict(head="corr"))          # n=12: columns=12 skipped (too slow)
            for ch in variants:
                for fold in (0, 1):
                    c = dict(best); c.update(ch); c["fold"] = fold; cfgs.append(c)
        runner(cfgs, RES / "p3_stageB.jsonl", jobs)
    elif stage in ("C", "ref"):
        cfgs = []
        for n in (8, 12):
            gm = group_mean_val([r for r in A + Bq if r["n"] == n])
            ranked = sorted(gm.items(), key=lambda kv: -kv[1])
            top = [dict(c) for c, _ in ranked[:2]]
            print("n", n, "final", top[0], "runner-up", top[1], flush=True)
            if stage == "C":
                for ti, c in enumerate(top):
                    for fold in range(5):
                        d = dict(c); d.update(fold=fold, save_scores=True, rank=ti); cfgs.append(d)
            else:
                c = dict(top[0]); c.update(phase="paper", scale=0.0, columns=c["columns"])
                for fold in range(5):
                    d = dict(c); d["fold"] = fold; cfgs.append(d)
        runner(cfgs, RES / ("p3_stageC.jsonl" if stage == "C" else "p3_ref_paper.jsonl"), jobs)


if __name__ == "__main__":
    cli()
