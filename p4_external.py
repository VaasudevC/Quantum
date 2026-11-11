"""P4: external validation on the PTB Diagnostic ECG DB.  Subcommands (run with ../.venv unless noted):
  prep       build the combined PTB-XL + PTB matrix with PTB-XL-only preprocessing/selection -> C:/bwcache/full.npz  (+ shift analysis)
  classical  LR, LR2, RF, XGB, MLP at k in {8,12,16,32}: tuned on a PTB-XL validation split, scored once on PTB
  vqc        angle VQC pi/6 (k=8 fixed; k=32 fixed and trained scales), trained on ALL PTB-XL (val from PTB-XL), scored once on PTB
  (bw: see p4_bw.py in the torch venv)
Scores for every model -> results/ext/scores.jsonl (one line per model, with PTB-XL val AUC and the external score vector).
"""
import argparse, json, os, sys, time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scale_cv as S
import p1_classical as P1

EXT = Path("ptb_external/ptb_features.csv"); CACHE = Path("C:/bwcache/full.npz"); OUT = Path("results/ext/scores.jsonl")


def build():
    X, y, g, sf, feats = S.load()
    ext = pd.read_csv(EXT)
    Xe = ext[list(feats)].astype(float).to_numpy(); Xe[~np.isfinite(Xe)] = np.nan
    ye = ext["label"].to_numpy().astype(int); ge = ext["patient"].to_numpy()
    n = len(y)
    tr_all = np.arange(n)
    itr, iva = next(StratifiedGroupKFold(8, shuffle=True, random_state=123).split(X, y, g))
    tr, va = itr, iva
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    Xa = np.vstack([X, Xe]); Xi = np.where(np.isnan(Xa), med, Xa)
    mu, sd = Xi[tr].mean(0), Xi[tr].std(0); sd[sd < 1e-9] = 1
    Z = (Xi - mu) / sd
    order = S.l1_order(Z[tr], y[tr])
    return dict(Z=Z, order=np.array(order), tr=tr, va=va, te=np.arange(n, n + len(ye)), y=np.concatenate([y, ye]),
                g=np.concatenate([g.astype(str), ge.astype(str)]), n_xl=n, feats=np.array(feats), med=med, mu=mu, sd=sd,
                nan_frac_ext=float(np.isnan(Xe).mean()))


def cmd_prep():
    d = build()
    o = d["order"][:32]
    np.savez(CACHE, Z=d["Z"][:, o].astype(np.float32), order=d["order"], tr=d["tr"], va=d["va"], te=d["te"], y=d["y"], g=d["g"],
             names=d["feats"][o], n_xl=d["n_xl"])
    Z, y, n = d["Z"], d["y"], d["n_xl"]
    # distribution shift: PTB-XL class vs PTB class, in PTB-XL SD units (z-scored on PTB-XL train)
    feats = d["feats"]
    rep = {"nan_frac_ext": d["nan_frac_ext"], "selected_top32": list(map(str, feats[o])), "groups": {}}
    for lab, name in ((0, "NORM"), (1, "MI")):
        a = Z[:n][y[:n] == lab]; b = Z[n:][y[n:] == lab]
        dm = b.mean(0) - a.mean(0); ratio = b.std(0) / np.where(a.std(0) > 1e-9, a.std(0), 1)
        top = np.argsort(-np.abs(dm))[:12]
        rep["groups"][name] = dict(n_xl=int(len(a)), n_ptb=int(len(b)), mean_abs_shift_all=float(np.abs(dm).mean()),
                                   mean_abs_shift_top32=float(np.abs(dm[o]).mean()), frac_cols_abs_shift_gt_0p5=float((np.abs(dm) > 0.5).mean()),
                                   largest_shifts={str(feats[j]): [round(float(dm[j]), 2), round(float(ratio[j]), 2)] for j in top},
                                   top32_shift={str(feats[j]): round(float(dm[j]), 2) for j in o[:12]})
    json.dump(rep, open("results/ext/shift.json", "w"), indent=1)
    print(json.dumps({k: rep["groups"][k]["mean_abs_shift_all"] for k in rep["groups"]}), "nan_ext", rep["nan_frac_ext"])


def cmd_classical(ks, models):
    d = build(); Z, y, tr, va, te, order = d["Z"], d["y"], d["tr"], d["va"], d["te"], d["order"]
    done = {(json.loads(l)["model"], json.loads(l)["k"]) for l in open(OUT)} if OUT.exists() else set()
    for k in ks:
        Zs = Z[:, order[:k]]
        for m in models:
            if (m, k) in done or (m == "lr2" and k > 32):
                continue
            t0 = time.time(); v, hp, s = P1.MODELS[m](Zs, y, tr, va, te)
            r = dict(model=m, k=k, family="classical", val_auc=float(v), ext_auc=float(roc_auc_score(y[te], s)), hp=hp,
                     secs=round(time.time() - t0, 1), scores=[round(float(a), 5) for a in s])
            open(OUT, "a").write(json.dumps(r) + "\n"); print(m, k, r["val_auc"], r["ext_auc"], r["secs"], flush=True)


def vqc_one(arg):
    k, scale, train_scale = arg
    d = build(); Z, y, tr, va, te, order = d["Z"], d["y"], d["tr"], d["va"], d["te"], d["order"]
    import Vqc_fast8 as V
    sel = list(order[:k]); T = np.zeros((len(Z), V.N_FEATURES)); T[:, :k] = np.tanh(Z[:, sel])
    model = S.Linear(k, scale, "z", train_scale)
    t0 = time.time()
    m, (vauc, p, bstep) = S.train(model, T[tr], y[tr], T[va], y[va], 1000, 1000, 0.01, 0)
    s = m.proba(T[te], p)
    r = dict(model="vqc_pi6" + ("_trained" if train_scale else ""), k=k, family="angle_vqc", val_auc=float(vauc), best_step=bstep,
             ext_auc=float(roc_auc_score(y[te], s)), secs=round(time.time() - t0, 1), scores=[round(float(a), 5) for a in s],
             params=p.tolist())
    return r


def cmd_vqc(jobs):
    from concurrent.futures import ProcessPoolExecutor
    todo = [(8, np.pi / 6, False), (32, np.pi / 6, False), (32, np.pi / 6, True)]
    done = {(json.loads(l)["model"], json.loads(l)["k"]) for l in open(OUT)} if OUT.exists() else set()
    todo = [t for t in todo if ("vqc_pi6" + ("_trained" if t[2] else ""), t[0]) not in done]
    with ProcessPoolExecutor(jobs) as ex:
        for r in ex.map(vqc_one, todo):
            open(OUT, "a").write(json.dumps(r) + "\n"); print(r["model"], r["k"], r["val_auc"], r["ext_auc"], r["secs"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--ks", default="8,12,16,32"); ap.add_argument("--models", default="lr,lr2,rf,xgb,mlp")
    a = ap.parse_args()
    if a.cmd == "prep": cmd_prep()
    elif a.cmd == "classical": cmd_classical([int(k) for k in a.ks.split(",")], a.models.split(","))
    elif a.cmd == "vqc": cmd_vqc(a.jobs)
