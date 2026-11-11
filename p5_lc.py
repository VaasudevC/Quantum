"""P5: limited-data learning curves.  Sizes {50,100,250,500,1000} x 5 folds x 3 repeats (the 'full' point reuses the full-data runs).
  prep   (../.venv)   per (fold, size, rep): stratified patient-grouped subsample of inner-train; median/z-score/L1-selection fitted on the
                      SUBSAMPLE only; standardised top-16 columns of all 14,965 rows -> C:/bwcache/lc/f{fold}_s{size}_r{rep}.npz
  classical (../.venv)  LR, LR2, RF, XGB (small validation grids) at k=8 and k=12 features
  vqc       (../.venv)  angle VQC pi/6 k=8 (1000 steps, batch=min(1000,n))
  bw        (C:/bwvenv) brick-wall finals (n=8, n=12; config read from results/bw/final_configs.json)
The validation rows (full inner-val of the fold, standardised with subsample statistics) are used for hyper-parameter / checkpoint choice.
"""
import argparse, json, os, sys, time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
LC = Path("C:/bwcache/lc"); OUT = Path("results/lc")
SIZES = (50, 100, 250, 500, 1000); REPS = 3


def subsample(y, g, tr, m, seed):
    rng = np.random.default_rng(seed)
    p = y[tr].mean(); npos = max(4, int(round(m * p))); nneg = m - npos
    pats = rng.permutation(np.unique(g[tr])); take = []; cp = cn = 0
    gi = {}
    for i in tr:
        gi.setdefault(g[i], []).append(i)
    for pt in pats:
        for i in gi[pt]:
            if y[i] == 1 and cp < npos: take.append(i); cp += 1
            elif y[i] == 0 and cn < nneg: take.append(i); cn += 1
        if cp >= npos and cn >= nneg: break
    return np.array(take)


def cmd_prep(jobs):
    from concurrent.futures import ProcessPoolExecutor
    LC.mkdir(parents=True, exist_ok=True)
    cfgs = [(f, s, r) for f in range(5) for s in SIZES for r in range(REPS)]
    with ProcessPoolExecutor(jobs) as ex:
        list(ex.map(_prep_one, cfgs))


def _prep_one(a):
    fold, size, rep = a
    out = LC / f"f{fold}_s{size}_r{rep}.npz"
    if out.exists(): return
    import scale_cv as S
    X, y, g, _, feats = S.load()
    tr0, va, te = S.splits("grouped5", fold)
    sub = subsample(y, g, tr0, size, 1000 * fold + 10 * size + rep)
    med = np.nan_to_num(np.nanmedian(X[sub], axis=0)); Xi = np.where(np.isnan(X), med, X)
    mu, sd = Xi[sub].mean(0), Xi[sub].std(0); sd[sd < 1e-9] = 1; Z = (Xi - mu) / sd
    order = S.l1_order(Z[sub], y[sub])
    np.savez(out, Z=Z[:, order[:16]].astype(np.float32), order=np.array(order[:16]), tr=sub, va=va, te=te, y=y, g=g)


def load_lc(fold, size, rep):
    d = np.load(LC / f"f{fold}_s{size}_r{rep}.npz")
    return d["Z"].astype(np.float64), d["y"], d["g"], d["tr"], d["va"], d["te"]


def _jobs(models, ks):
    return [(m, k, f, s, r) for s in SIZES for r in range(REPS) for f in range(5) for m in models for k in ks]


def cmd_classical(jobs):
    from concurrent.futures import ProcessPoolExecutor
    done = _done("classical")
    todo = [j for j in _jobs(("lr", "lr2", "rf", "xgb"), (8, 12)) if j not in done]
    print(len(todo), "classical jobs", flush=True)
    with ProcessPoolExecutor(jobs) as ex:
        for r in ex.map(_classical_one, todo, chunksize=2):
            _emit("classical", r)


def _done(fam):
    p = OUT / f"{fam}.jsonl"
    if not p.exists(): return set()
    return {(r["model"], r["k"], r["fold"], r["size"], r["rep"]) for r in map(json.loads, open(p))}


def _emit(fam, r):
    OUT.mkdir(parents=True, exist_ok=True)
    open(OUT / f"{fam}.jsonl", "a").write(json.dumps(r) + "\n")
    print(fam, r["model"], r["k"], f"f{r['fold']} n={r['size']} r{r['rep']} val={r['val_auc']:.4f} test={r['test_auc']:.4f} {r['secs']}s", flush=True)


def _classical_one(job):
    m, k, fold, size, rep = job
    import p1_classical as P1
    from sklearn.metrics import roc_auc_score
    Z, y, g, tr, va, te = load_lc(fold, size, rep)
    Zs = Z[:, :k]; t0 = time.time()
    if m == "rf":
        P1_rf = P1.m_rf
        v, hp, s = P1_rf(Zs, y, tr, va, te)
    else:
        v, hp, s = P1.MODELS[m](Zs, y, tr, va, te)
    return dict(model=m, k=k, fold=fold, size=size, rep=rep, val_auc=float(v), test_auc=float(roc_auc_score(y[te], s)), hp=hp, secs=round(time.time() - t0, 1))


def cmd_vqc(jobs):
    from concurrent.futures import ProcessPoolExecutor
    done = _done("vqc")
    todo = [j for j in _jobs(("vqc_pi6",), (8,)) if j not in done]
    print(len(todo), "vqc jobs", flush=True)
    with ProcessPoolExecutor(jobs) as ex:
        for r in ex.map(_vqc_one, todo):
            _emit("vqc", r)


def _vqc_one(job):
    m, k, fold, size, rep = job
    import scale_cv as S, Vqc_fast8 as V
    from sklearn.metrics import roc_auc_score
    Z, y, g, tr, va, te = load_lc(fold, size, rep)
    T = np.zeros((len(Z), V.N_FEATURES)); T[:, :k] = np.tanh(Z[:, :k]); t0 = time.time()
    model = S.Linear(k, np.pi / 6, "z", False)
    mm, (vauc, p, bstep) = S.train(model, T[tr], y[tr], T[va], y[va], 1000, min(1000, len(tr)), 0.01, rep)
    return dict(model=m, k=k, fold=fold, size=size, rep=rep, val_auc=float(vauc), test_auc=float(roc_auc_score(y[te], mm.proba(T[te], p))), best_step=bstep, secs=round(time.time() - t0, 1))


def cmd_bw(jobs):
    from concurrent.futures import ProcessPoolExecutor
    cfgs = json.load(open("results/bw/final_configs.json"))
    done = _done("bw")
    todo = [(f"bw{n}", n, f, s, r) for s in SIZES for r in range(REPS) for f in range(5) for n in (8, 12) if (f"bw{n}", n, f, s, r) not in done]
    print(len(todo), "bw jobs", flush=True)
    with ProcessPoolExecutor(jobs) as ex:
        for r in ex.map(_bw_one, todo):
            _emit("bw", r)


def _bw_one(job):
    m, n, fold, size, rep = job
    import torch, bw_torch as T
    torch.set_num_threads(1)
    cfg = json.load(open("results/bw/final_configs.json"))[str(n)]
    Z, y, g, tr, va, te = load_lc(fold, size, rep)
    Zs = Z[:, :n]
    mdl = T.Model(n, cfg["columns"], cfg["k"], cfg["scale"], head=cfg["head"], seed=rep + 10 * fold)
    vauc, bstep, tot, secs = T.train_model(mdl, Zs[tr], y[tr], Zs[va], y[va], steps=cfg["steps"], batch=min(512, len(tr)), lr=cfg["lr"], seed=rep + 10 * fold)
    s = mdl.predict(Zs[te])
    return dict(model=m, k=n, fold=fold, size=size, rep=rep, val_auc=float(vauc), test_auc=T.auc(y[te], s), best_step=bstep, secs=round(secs, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--jobs", type=int, default=3); a = ap.parse_args()
    {"prep": cmd_prep, "classical": cmd_classical, "vqc": cmd_vqc, "bw": cmd_bw}[a.cmd](a.jobs)
