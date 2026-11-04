"""Angle-scale experiments for the 8-qubit re-uploading VQC (PTB-XL, MI vs NORM).

Question: does the angle scale in  angle = scale * tanh(z)  drive the AUC collapse
seen when k > 8 features are re-uploaded on 8 qubits?

Protocol (matches the teammate's Oct-2026 report):
  * ECG-only pool: 249 features (all numeric minus ids, labels, age, sex)
  * split "grouped5": patient-grouped, class-stratified 5-fold CV; 1/8 of each train fold
    (grouped) is validation.  split "official": PTB-XL strat_fold 1-8 train / 9 val / 10 test
  * median impute + z-score + selector fitted on the inner-train part only
  * selectors: l1 (order of entry along an L1-logistic path), cohen_dedup (|d| desc,
    skip |r| >= 0.8 to an accepted one), random
  * 8 qubits x 4 layers (Vqc_fast8 circuit); round r of 8 features -> layer r; unused layers angle 0
  * linear readout on 8 <Z> (81 params), class-weighted BCE, Adam, adjoint gradients
  * checkpoint = best val AUC (every 25 steps); metric = test AUC; LR on same features as reference
  * --shuffle-later: permute rows of features 9..k (independently per column) = shuffle control

Results are appended one JSON line per run to --out; reruns skip finished configs (resumable).

Examples
  python -u scale_cv.py --scales "pi/4,pi/3,pi/2,2*pi/3,pi" --ks 32 --steps 600 --jobs 16 --out results/sweep_k32.jsonl
  python -u scale_cv.py --scales "pi/8,pi/6,pi/4" --ks 8,16,32 --steps 1000 --jobs 16 --out results/small_scales.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Vqc_fast8 as V

DATA_DIR = Path(os.environ.get("ECG_DATA_DIR", Path(__file__).resolve().parent / "data"))
DROP = {"ecg_id", "label_class", "label_subclass", "age", "sex", "patient_id", "strat_fold"}


def _find(name):
    for p in (DATA_DIR / name, DATA_DIR / (name + ".gz")):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}(.gz) not found in {DATA_DIR} (set ECG_DATA_DIR)")


@lru_cache(maxsize=1)
def load():
    df = pd.read_csv(_find("st_features_plus_TR.csv"))
    meta = pd.read_csv(_find("ptbxl_database.csv"), usecols=["ecg_id", "patient_id", "strat_fold"])
    df = df.merge(meta, on="ecg_id", how="left")
    feats = [c for c in df.columns if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].astype(float).to_numpy()
    X[~np.isfinite(X)] = np.nan
    y = (df["label_class"] == "MI").astype(int).to_numpy()
    return X, y, df["patient_id"].to_numpy(), df["strat_fold"].to_numpy(), tuple(feats)


def splits(split, fold):
    X, y, g, sf, _ = load()
    if split == "official":
        idx = np.arange(len(y))
        return idx[sf <= 8], idx[sf == 9], idx[sf == 10]
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    tr_all, te = outer[fold]
    itr, iva = next(StratifiedGroupKFold(8, shuffle=True, random_state=fold).split(X[tr_all], y[tr_all], g[tr_all]))
    return tr_all[itr], tr_all[iva], te


def l1_order(Z, y):
    first = np.full(Z.shape[1], np.inf)
    for i, C in enumerate(np.logspace(-4, 0, 40)):
        c = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000).fit(Z, y).coef_[0]
        new = (np.abs(c) > 1e-8) & np.isinf(first)
        first[new] = i - 1e-6 * np.abs(c[new])        # ties: larger |coef| first
        if np.isfinite(first).sum() >= 40:
            break
    rest = np.argsort(-np.abs(LogisticRegression(max_iter=3000).fit(Z, y).coef_[0]))
    order = [int(j) for j in np.argsort(first) if np.isfinite(first[j])]
    return order + [int(j) for j in rest if j not in order]


def cohen_dedup_order(Z, y, r_max=0.8):
    a, b = Z[y == 1], Z[y == 0]
    sp = np.sqrt(((len(a)-1)*a.var(0, ddof=1) + (len(b)-1)*b.var(0, ddof=1)) / (len(a)+len(b)-2))
    d = np.abs((a.mean(0) - b.mean(0)) / np.where(sp > 0, sp, 1))
    R = np.abs(np.corrcoef(Z, rowvar=False)); R = np.nan_to_num(R)
    acc, skipped = [], []
    for j in np.argsort(-d):
        (acc if all(R[j, i] < r_max for i in acc) else skipped).append(int(j))
    return acc + skipped


@lru_cache(maxsize=8)
def prep(split, fold, selector):
    X, y, _, _, _ = load()
    tr, va, te = splits(split, fold)
    med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
    Xi = np.where(np.isnan(X), med, X)
    mu, sd = Xi[tr].mean(0), Xi[tr].std(0); sd[sd < 1e-9] = 1
    Z = (Xi - mu) / sd
    if selector == "l1":
        order = l1_order(Z[tr], y[tr])
    elif selector == "cohen_dedup":
        order = cohen_dedup_order(Z[tr], y[tr])
    elif selector == "random":
        order = list(np.random.default_rng(1000 + fold).permutation(Z.shape[1]))
    else:
        raise ValueError(selector)
    return Z, tr, va, te, tuple(order)


class Linear:
    def init(self, rng):
        return np.concatenate([rng.normal(scale=0.1, size=V.N_ANSATZ), rng.normal(scale=0.1, size=8), [0.0]])

    def proba(self, A, p, chunk=4000):
        out = []
        for s in range(0, len(A), chunk):
            z = V.expectations(V.forward(A[s:s+chunk], p[:V.N_ANSATZ]), "z")
            out.append(1 / (1 + np.exp(-(z @ p[V.N_ANSATZ:V.N_ANSATZ+8] + p[-1]))))
        return np.concatenate(out)

    def loss_grad(self, p, A, y, cw):
        th, w, b = p[:V.N_ANSATZ], p[V.N_ANSATZ:V.N_ANSATZ+8], p[-1]
        psi = V.forward(A, th); z = V.expectations(psi, "z")
        q = np.clip(1 / (1 + np.exp(-(z @ w + b))), 1e-7, 1 - 1e-7)
        sw = np.where(y == 1, cw[1], cw[0]); r = sw * (q - y) / len(y)
        loss = float((sw * -(y*np.log(q) + (1-y)*np.log(1-q))).mean())
        return loss, np.concatenate([V.backward(psi, np.outer(r, w), A, th, "z"), z.T @ r, [r.sum()]])


def train(A_tr, y_tr, A_va, y_va, steps, batch, lr, seed):
    m = Linear(); rng = np.random.default_rng(seed); p = m.init(rng)
    n = len(y_tr); cw = np.array([n/(2*(n-y_tr.sum())), n/(2*y_tr.sum())])
    mo = np.zeros_like(p); vo = np.zeros_like(p); best = (-1.0, p.copy(), 0)
    for t in range(1, steps+1):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        _, g = m.loss_grad(p, A_tr[idx], y_tr[idx], cw)
        mo = .9*mo + .1*g; vo = .999*vo + .001*g*g
        p = p - lr*(mo/(1-.9**t))/(np.sqrt(vo/(1-.999**t))+1e-8)
        if t % 25 == 0 or t == steps:
            auc = roc_auc_score(y_va, m.proba(A_va, p))
            if auc > best[0]:
                best = (auc, p.copy(), t)
    return m, best


def run_one(cfg):
    X, y, _, _, feats = load()
    Z, tr, va, te, order = prep(cfg["split"], cfg["fold"], cfg["selector"])
    k, s = cfg["k"], eval(cfg["scale"], {"pi": np.pi})
    sel = list(order[:k]); Zs = Z[:, sel].copy()
    if cfg["shuffle_later"] and k > V.N_QUBITS:
        rng = np.random.default_rng(7 + cfg["fold"])
        for j in range(V.N_QUBITS, k):
            Zs[:, j] = Zs[rng.permutation(len(Zs)), j]
    t0 = time.time()
    lr_ref = LogisticRegression(max_iter=3000).fit(Zs[tr], y[tr])
    auc_lr = roc_auc_score(y[te], lr_ref.predict_proba(Zs[te])[:, 1])
    A = np.zeros((len(Zs), V.N_FEATURES)); A[:, :k] = s * np.tanh(Zs)
    m, (vauc, p, bstep) = train(A[tr], y[tr], A[va], y[va], cfg["steps"], cfg["batch"], cfg["lr"], cfg["seed"] + cfg["fold"])
    return dict(cfg, test_auc=float(roc_auc_score(y[te], m.proba(A[te], p))), val_auc=float(vauc),
                best_step=bstep, lr_auc=float(auc_lr),
                frac_angle_gt_0p9pi=float((np.abs(A[tr][:, :k]) > 0.9*np.pi).mean()),
                secs=round(time.time()-t0, 1), features=[feats[j] for j in sel], params=p.tolist())


def key(c):
    return (c["split"], c["fold"], c["selector"], c["k"], c["scale"], c["steps"], c["seed"], c["shuffle_later"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="pi,pi/2"); ap.add_argument("--ks", default="8,16,32")
    ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--selectors", default="l1")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--split", default="grouped5", choices=["grouped5", "official"])
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--shuffle-later", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    folds = [0] if a.split == "official" else [int(f) for f in a.folds.split(",")]
    cfgs = [dict(split=a.split, fold=f, selector=sel, k=int(k), scale=sc, steps=a.steps, batch=a.batch,
                 lr=a.lr, seed=int(sd), shuffle_later=a.shuffle_later)
            for sd in a.seeds.split(",") for f in folds for sel in a.selectors.split(",")
            for k in a.ks.split(",") for sc in a.scales.split(",")]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    done = {key(json.loads(l)) for l in open(a.out)} if Path(a.out).exists() else set()
    todo = [c for c in cfgs if key(c) not in done]
    print(f"{len(cfgs)} configs, {len(todo)} to run, jobs={a.jobs}, data={DATA_DIR}", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run_one, c) for c in todo]
        for fu in as_completed(futs):
            r = fu.result()
            with open(a.out, "a") as fo:
                fo.write(json.dumps(r) + "\n")
            print(f"fold {r['fold']} {r['selector']} k={r['k']:2d} scale={r['scale']:7s} seed={r['seed']} "
                  f"shuf={int(r['shuffle_later'])} test_auc={r['test_auc']:.4f} LR={r['lr_auc']:.4f} "
                  f"best_step={r['best_step']} {r['secs']}s", flush=True)


if __name__ == "__main__":
    main()
