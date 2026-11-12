"""SD1a (.venv): pre-registered subset generation + per-subset preprocessing (median/z-score/L1 order fitted on the N-ECG subset ONLY).
Outer folds exactly as scale_cv.splits; training part = outer train; test = full outer test fold.
Subsets: class-stratified (MI share of the fold's training part, >= 5 per class), ONE ECG per patient, deterministic seed (fold, N, r).
Outputs: results/sd_subsets.json  {"f,N,r": [row indices]}   and  C:/bwcache/sd/f{f}_N{N}_r{r}.npz  (Zsub[:, :8], ysub, order[:8], Ztest[:, :8], yte, te)."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import scale_cv as S
from sklearn.model_selection import StratifiedGroupKFold

SIZES = (25, 50, 100, 200, 500); R = 40; OUT = Path("C:/bwcache/sd"); NF = 8


from functools import lru_cache


@lru_cache(maxsize=1)
def _outer_all():
    X, y, g, _, _ = S.load()
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))


def outer(fold):
    return _outer_all()[fold]


def draw(fold, N, r, y, g, tr_all):
    rng = np.random.default_rng([fold, N, r])
    prev = y[tr_all].mean(); npos = int(np.clip(round(N * prev), 5, N - 5)); nneg = N - npos
    chosen, used = [], set()
    for lab, cnt in ((1, npos), (0, nneg)):
        pool = tr_all[y[tr_all] == lab]; rng.shuffle(pool)
        got = 0
        for i in pool:                                  # first ECG in shuffled order of each not-yet-used patient = random ECG, one per patient
            if g[i] in used: continue
            used.add(g[i]); chosen.append(int(i)); got += 1
            if got == cnt: break
        assert got == cnt, (fold, N, r)
    return np.array(sorted(chosen))


def prep_one(a):
    fold, N, r = a
    f_ = OUT / f"f{fold}_N{N}_r{r}.npz"
    if f_.exists():
        return (f"{fold},{N},{r}", [int(i) for i in np.load(f_)["sub"]])
    X, y, g, _, feats = S.load(); tr_all, te = outer(fold)
    sub = draw(fold, N, r, y, g, tr_all)
    med = np.nan_to_num(np.nanmedian(X[sub], axis=0)); Xi = np.where(np.isnan(X), med, X)
    mu, sd = Xi[sub].mean(0), Xi[sub].std(0); sd[sd < 1e-9] = 1
    Zs = (Xi[sub] - mu) / sd
    order = S.l1_order(Zs, y[sub])[:NF]
    Zt = ((Xi[te] - mu) / sd)[:, order]
    np.savez(OUT / f"f{fold}_N{N}_r{r}.npz", Zsub=Zs[:, order].astype(np.float32), ysub=y[sub], order=np.array(order), Ztest=Zt.astype(np.float32), yte=y[te], te=te, sub=sub)
    return (f"{fold},{N},{r}", [int(i) for i in sub])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(f, N, r) for N in (100, 50, 200, 25, 500) for f in range(5) for r in range(R)]
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
        res = dict(ex.map(prep_one, jobs, chunksize=4))
    json.dump(res, open("results/sd_subsets.json", "w"))
    print(len(res), "subsets")
