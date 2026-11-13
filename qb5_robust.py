"""QB5 (pre-registered in QB.md; deviation of 2026-11-13 logged before this run): other kinds of benefit at N = 100, k = 8, DISCOVERY folds 0-2, SD subsets r = 0-9.
Models: Q1 angle VQC, Q3 quantum-kernel SVM, C1 LR, C2 SVM-RBF, C3 GP, C5 RF (configs = inner-CV choice on the clean subset, from the SD runs),
twins on bounded inputs a = (pi/6) tanh z: TW_LR, TW_LR2, TW_SVM (inner CV on the clean subset).  Every model is retrained once per perturbation.
Settings: clean; label noise 10/20/30 %; test-time Gaussian noise 0.1/0.3/0.5 SD; single-lead dropout (test); MI prevalence 5 % / 10 % (new subsets);
site shift (train site 0 only, test other sites); date shift (train before / test after the median date); inferior MI (IMI) vs NORM.
  ..\\.venv\\Scripts\\python.exe qb5_robust.py FOLD    -> results/qb5.jsonl (one line per fold x r, all settings x models)"""
import json, os, sys, time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scale_cv as S, sd_run as R, sd_prep as P
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT = Path("results/qb5.jsonl"); A6 = np.pi / 6; N = 100
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def meta():
    d = pd.read_csv("data/st_features_plus_TR.csv.gz", usecols=["ecg_id", "label_subclass"])
    m = pd.read_csv("../PTB_XL/ptbxl_database.csv", usecols=["ecg_id", "site", "recording_date"])
    x = d.merge(m, on="ecg_id", how="left")
    return x["site"].to_numpy(), pd.to_datetime(x["recording_date"]).to_numpy(), x["label_subclass"].to_numpy()


def sd_choices(fold, r):
    ch = {}
    for f in Path("results").glob("sd_runs_main.w*.jsonl"):
        for l in open(f):
            q = json.loads(l)
            if q["k"] == 8 and q["N"] == N and q["fold"] == fold and q["r"] == r: ch[q["model"]] = q["chosen"]
    return ch


# ------------------------------------------------------------------------------------------------ fixed-config fitters: fit(Z, y) -> scorer
def make_fixed(name, chosen, seed):
    if name == "Q1_vqc":
        kv = dict(x.split("=") for x in chosen.split(",")); g = (float(kv["scale"]), float(kv["l2"])); st = int(kv["steps"])
        return lambda Z, y: R.VQC(8, False).fit(Z, y, g, [st], seed)[st]
    if name == "Q3_qkernel_svm":
        kv = dict(x.split("=") for x in chosen.split(",")); g = (float(kv["scale"]), float(kv["C"]))
        return lambda Z, y: R.QKernel(8).fit(Z, y, g, [0], seed)[0]
    ms = R.make_sk_models(); M = ms[name]; cfg = json.loads(chosen)
    return lambda Z, y: R.SimpleSK(name, [cfg], M.make, M.score).fit(Z, y, 0, [0], seed)[0]


def twin_models():
    T = lambda Z: A6 * np.tanh(Z)
    def lr(C): return lambda Z, y: (lambda m: lambda X: m.decision_function(T(X)))(LogisticRegression(C=C, max_iter=3000).fit(T(Z), y))
    def lr2(C):
        def fit(Z, y):
            pf = PolynomialFeatures(2, include_bias=False); sc = StandardScaler().fit(pf.fit_transform(T(Z))); m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(pf.transform(T(Z))), y)
            return lambda X: m.decision_function(sc.transform(pf.transform(T(X))))
        return fit
    def svm(C, g): return lambda Z, y: (lambda m: lambda X: m.decision_function(T(X)))(SVC(C=C, gamma=g).fit(T(Z), y))
    return {"TW_LR": {f"C={C}": lr(C) for C in (0.01, 0.1, 1, 10)}, "TW_LR2": {f"C={C}": lr2(C) for C in (0.01, 0.1, 1, 10)},
            "TW_SVM": {f"C={C},g={g}": svm(C, g) for C in (0.1, 1, 10) for g in ("scale", 0.1, 1.0)}}


def tune_twin(grid, Z, y, seed):
    folds = R.inner_folds(y, seed); best = None
    for k, fit in grid.items():
        v = np.mean([R.auc(y[va], fit(Z[tr], y[tr])(Z[va])) for tr, va in folds])
        if best is None or v > best[0]: best = (v, k, fit)
    return best[1], best[2]


# ------------------------------------------------------------------------------------------------ subsets
def prep_subset(X, y, sub, test):
    med = np.nan_to_num(np.nanmedian(X[sub], axis=0)); Xi = lambda I: np.where(np.isnan(X[I]), med, X[I])
    xs = Xi(sub); mu, sd = xs.mean(0), xs.std(0); sd[sd < 1e-9] = 1; Zs = (xs - mu) / sd
    order = S.l1_order(Zs, y[sub])[:8]
    return Zs[:, order], y[sub], ((Xi(test) - mu) / sd)[:, order], y[test]


def draw(pool, yy, g, npos, nneg, rng):
    chosen, used = [], set()
    for lab, cnt in ((1, npos), (0, nneg)):
        cand = pool[yy[pool] == lab].copy(); rng.shuffle(cand); got = 0
        for i in cand:
            if g[i] in used: continue
            used.add(g[i]); chosen.append(int(i)); got += 1
            if got == cnt: break
        if got < cnt: return None
    return np.array(sorted(chosen))


def run_unit(fold, r, X, y, g, feats, site, date, sub_cls):
    tr_all, te = P.outer(fold); d = np.load(P.OUT / f"f{fold}_N{N}_r{r}.npz")
    Zs, ys, Zt, yt, sub, order = d["Zsub"].astype(float), d["ysub"], d["Ztest"].astype(float), d["yte"], d["sub"], list(d["order"])
    seed = 1000 * fold + r; rng = np.random.default_rng(77 + seed)
    ch = sd_choices(fold, r); models = {m: make_fixed(m, ch[m], seed) for m in ("Q1_vqc", "Q3_qkernel_svm", "C1_lr", "C2_svm_rbf", "C3_gpc", "C5_rf")}
    tw = {}
    for name, grid in twin_models().items():
        k, fit = tune_twin(grid, Zs, ys, 7 + seed); tw[name] = k; models[name] = fit
    res = {m: {} for m in models}
    # clean + test-time perturbations
    med = np.nan_to_num(np.nanmedian(X[sub], axis=0)); xs = np.where(np.isnan(X[sub]), med, X[sub]); mu, sdv = xs.mean(0), xs.std(0); sdv[sdv < 1e-9] = 1
    zmed = ((med - mu) / sdv)[order]
    noise = {s: rng.normal(0, s, Zt.shape) for s in (0.1, 0.3, 0.5)}
    leads = sorted({feats[j].split("_")[0] for j in order if feats[j].split("_")[0] in LEADS})
    restrict = {"site": np.isin(te, np.flatnonzero(site != 0)), "date": None}
    trd = date[tr_all]; dmed = np.sort(trd)[len(trd) // 2]; restrict["date"] = date[te] >= dmed
    clean_fit = {}
    for m, fit in models.items():
        sc = fit(Zs, ys); clean_fit[m] = sc; res[m]["clean"] = R.auc(yt, sc(Zt))
        for s, E in noise.items(): res[m][f"testnoise_{s}"] = R.auc(yt, sc(Zt + E))
        dl = []
        for L in leads:
            Zd = Zt.copy(); cols = [i for i, j in enumerate(order) if feats[j].split("_")[0] == L]; Zd[:, cols] = zmed[cols]; dl.append(R.auc(yt, sc(Zd)))
        res[m]["leaddrop_mean"] = float(np.mean(dl)); res[m]["leaddrop_worst"] = float(np.min(dl))
        for key, mask in restrict.items(): res[m][f"{key}_cleanref"] = R.auc(yt[mask], sc(Zt[mask]))
    # label noise
    for p in (0.1, 0.2, 0.3):
        yn = ys.copy(); flip = rng.choice(len(ys), int(round(p * len(ys))), replace=False); yn[flip] = 1 - yn[flip]
        if len(np.unique(yn)) < 2: continue
        for m, fit in models.items():
            try: res[m][f"labelnoise_{p}"] = R.auc(yt, fit(Zs, yn)(Zt))
            except Exception as e: res[m][f"labelnoise_{p}"] = None
    # new subsets: imbalance, site, date, subtype
    newsets = {}
    for prev in (0.05, 0.10):
        s_ = draw(tr_all, y, g, int(round(prev * N)), N - int(round(prev * N)), np.random.default_rng([fold, int(prev * 100), r]))
        if s_ is not None: newsets[f"imbalance_{prev}"] = (s_, te)
    pos_rate = y[tr_all].mean(); npos = int(round(N * pos_rate))
    pool = tr_all[site[tr_all] == 0]; s_ = draw(pool, y, g, npos, N - npos, np.random.default_rng([fold, 501, r]))
    if s_ is not None: newsets["site"] = (s_, te[restrict["site"]])
    pool = tr_all[date[tr_all] < dmed]; s_ = draw(pool, y, g, npos, N - npos, np.random.default_rng([fold, 502, r]))
    if s_ is not None: newsets["date"] = (s_, te[restrict["date"]])
    keep = (sub_cls == "IMI") | (sub_cls == "NORM"); pool = tr_all[keep[tr_all]]; pr = (sub_cls[pool] == "IMI").mean()
    s_ = draw(pool, y, g, int(round(N * pr)), N - int(round(N * pr)), np.random.default_rng([fold, 503, r]))
    if s_ is not None: newsets["subtype_IMI"] = (s_, te[keep[te]])
    for key, (s_, tset) in newsets.items():
        Zs2, ys2, Zt2, yt2 = prep_subset(X, y, s_, tset)
        for m, fit in models.items():
            try: res[m][key] = R.auc(yt2, fit(Zs2, ys2)(Zt2))
            except Exception as e: res[m][key] = None
    return dict(fold=fold, r=r, chosen=ch, twins=tw, leads=leads, res=res)


def main():
    folds = [int(f) for f in sys.argv[1:]]; assert all(f in (0, 1, 2) for f in folds), "discovery folds only"
    X, y, g, _, feats = S.load(); site, date, sub_cls = meta()
    done = {(json.loads(l)["fold"], json.loads(l)["r"]) for l in open(OUT)} if OUT.exists() else set()
    for fold in folds:
        for r in range(10):
            if (fold, r) in done: continue
            t0 = time.time(); rec = run_unit(fold, r, X, y, g, feats, site, date, sub_cls); rec["secs"] = round(time.time() - t0, 1)
            with open(OUT, "a") as fo: fo.write(json.dumps(rec) + "\n")
            print(fold, r, "done", rec["secs"], flush=True)


if __name__ == "__main__":
    main()
