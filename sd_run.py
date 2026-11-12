"""SD1 runner: all models of the pre-registered plan on the cached per-subset data (sd_prep.py).  Resumable, parallel over (fold, N, r).
  families:  main  (.venv)       Q1 angle VQC, Q2 angle VQC + trained scales, Q3 quantum-kernel SVM, C1-C6
             torch (C:/bwvenv)   Q4 brick-wall n=8, C7 TabPFN
  python sd_run.py --family main --k 8 --sizes 100 --jobs 5      ->  results/sd_runs_main.jsonl   (family torch -> sd_runs_torch.jsonl)
Everything (tuning by inner 3-fold stratified CV, early stopping = steps grid, refit) uses ONLY the N-ECG subset; the test fold is scored once after the refit."""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path("C:/bwcache/sd")


def auc(y, s):
    s = np.asarray(s, dtype=float); _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def inner_folds(y, seed, K=3):
    rng = np.random.default_rng(seed); fold = np.zeros(len(y), int)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % K
    return [(np.flatnonzero(fold != i), np.flatnonzero(fold == i)) for i in range(K)]


class Model:
    """groups(N): list of (group_key, [variants]);  fit(Z, y, g, variants, seed) -> {variant: scorer(Z) -> scores}."""
    name = ""; n_params = None


class SimpleSK(Model):
    def __init__(self, name, cfgs, make, score="proba"):
        self.name, self.cfgs, self.make, self.score = name, cfgs, make, score

    def groups(self, N):
        return [(i, [0]) for i in range(len(self.cfgs))]

    def fit(self, Z, y, g, variants, seed):
        m = self.make(self.cfgs[g], len(y), seed).fit(Z, y)
        return {0: (lambda X: m.predict_proba(X)[:, 1]) if self.score == "proba" else (lambda X: m.decision_function(X))}

    def describe(self, g, v):
        return json.dumps(self.cfgs[g], default=str)


def make_sk_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    ms = {}
    ms["C1_lr"] = SimpleSK("C1_lr", [dict(C=c) for c in (0.01, 0.1, 1, 10)], lambda c, n, s: LogisticRegression(C=c["C"], max_iter=3000), "dec")
    ms["C2_svm_rbf"] = SimpleSK("C2_svm_rbf", [dict(C=c, gamma=g) for c in (0.1, 1, 10) for g in ("scale", 0.1, 0.01)], lambda c, n, s: SVC(C=c["C"], gamma=c["gamma"]), "dec")
    ms["C3_gpc"] = SimpleSK("C3_gpc", [dict()], lambda c, n, s: GaussianProcessClassifier(RBF(1.0) + WhiteKernel(1.0), random_state=0))
    ms["C4_knn"] = SimpleSK("C4_knn", [dict(k=k) for k in (3, 5, 9)], lambda c, n, s: KNeighborsClassifier(min(c["k"], n - 1), weights="distance"))
    ms["C5_rf"] = SimpleSK("C5_rf", [dict(d=d, leaf=l) for d in (2, 4, None) for l in (1, 3)],
                           lambda c, n, s: RandomForestClassifier(300, max_depth=c["d"], min_samples_leaf=c["leaf"], n_jobs=1, random_state=0, class_weight="balanced_subsample"))
    ms["C6_xgb"] = SimpleSK("C6_xgb", [dict(d=d, n=n_, lam=l) for d in (2, 3) for n_ in (50, 200) for l in (1, 10)],
                            lambda c, n, s: xgb.XGBClassifier(n_estimators=c["n"], max_depth=c["d"], learning_rate=0.1, reg_lambda=c["lam"], tree_method="hist", n_jobs=1, random_state=0, eval_metric="logloss"))
    return ms


class QKernel(Model):
    """Q3: K(x,x') = |<phi(x)|phi(x')>|^2 for the encoding-only angle circuit (theta = 0).  The layers after the first encoding (Ry(0), CNOT ring) are fixed,
    data-independent unitaries, so the fidelity equals that of the first-layer product state: K = prod_i cos^2((a_i - a'_i)/2), a = scale*tanh(z).
    (Checked against Vqc_fast8 in sd_smoke.py.)"""
    name = "Q3_qkernel_svm"

    def __init__(self, k):
        self.k = k

    def groups(self, N):
        return [((s, C), [0]) for s in (np.pi / 8, np.pi / 6, np.pi / 4) for C in (0.1, 1, 10)]

    @staticmethod
    def kern(A, B):
        K = np.ones((len(A), len(B)))
        for i in range(A.shape[1]):
            K *= np.cos((A[:, i][:, None] - B[:, i][None, :]) / 2) ** 2
        return K

    def fit(self, Z, y, g, variants, seed):
        from sklearn.svm import SVC
        s, C = g; A = s * np.tanh(Z[:, :self.k]); m = SVC(kernel="precomputed", C=C).fit(self.kern(A, A), y)
        self.n_params = int(m.n_support_.sum())
        return {0: lambda X: m.decision_function(self.kern(s * np.tanh(X[:, :self.k]), A))}

    def describe(self, g, v):
        return f"scale={g[0]:.4f},C={g[1]}"


class VQC(Model):
    """Q1 (fixed scale grid) / Q2 (trained per-feature scales, init pi/6).  Adam lr 0.01, full batch, class-weighted BCE, L2 on readout; steps grid {100,300} from ONE 300-step trajectory."""
    def __init__(self, k, trained):
        self.k, self.trained = k, trained; self.name = "Q2_vqc_trained_scales" if trained else "Q1_vqc"

    def groups(self, N):
        scales = (np.pi / 6,) if self.trained else (np.pi / 6, np.pi / 4)
        return [((s, l2), [100, 300]) for s in scales for l2 in (0.0, 1e-3)]

    def fit(self, Z, y, g, variants, seed):
        import scale_cv as S, Vqc_fast8 as V
        s, l2 = g; k = self.k; model = S.Linear(k, s, "z", self.trained)
        T = np.zeros((len(Z), V.N_FEATURES)); T[:, :k] = np.tanh(Z[:, :k])
        rng = np.random.default_rng(seed); p = model.init(rng); n = len(y); cw = np.array([n / (2 * (n - y.sum())), n / (2 * y.sum())])
        mo = np.zeros_like(p); vo = np.zeros_like(p); snaps = {}
        for t in range(1, max(variants) + 1):
            _, gr = model.loss_grad(p, T, y, cw)
            if l2:
                gr[V.N_ANSATZ:V.N_ANSATZ + 8] += 2 * l2 * p[V.N_ANSATZ:V.N_ANSATZ + 8]
            mo = .9 * mo + .1 * gr; vo = .999 * vo + .001 * gr * gr
            p = p - 0.01 * (mo / (1 - .9 ** t)) / (np.sqrt(vo / (1 - .999 ** t)) + 1e-8)
            if t in variants:
                snaps[t] = p.copy()
        self.n_params = 81 + (k if self.trained else 0)

        def mk(pp):
            def f(X):
                Tt = np.zeros((len(X), V.N_FEATURES)); Tt[:, :k] = np.tanh(X[:, :k])
                th, w, b, sc = model.split(pp); z = V.expectations(V.forward(model.angles(Tt, sc), th), "z"); return z @ w + b
            return f
        return {v: mk(snaps[v]) for v in variants}

    def describe(self, g, v):
        return f"scale={g[0]:.4f},l2={g[1]},steps={v}"


class BW(Model):
    """Q4: brick-wall n=8 (particle number 4, 4 columns, linear head), Adam lr 0.02, full batch; scale {pi/6, pi/4}, steps {100,300}, head L2 {1e-3, 1e-2}."""
    name = "Q4_brickwall8"; n_params = 57

    def __init__(self, k):
        self.k = k

    def groups(self, N):
        return [((s, l2), [100, 300]) for s in (np.pi / 6, np.pi / 4) for l2 in (1e-3, 1e-2)]

    def fit(self, Z, y, g, variants, seed):
        import torch, bw_torch as T
        torch.set_num_threads(1); s, l2 = g; k = self.k
        pad = lambda X: np.concatenate([X[:, :k], np.zeros((len(X), 8 - k))], 1)      # k=4: the other 4 qubits get phase 0
        m = T.Model(8, 4, 4, s, head="linear", seed=seed)
        snaps = T.train_fixed(m, pad(Z), y, list(variants), lr=0.02, seed=seed, l2=l2, batch=512)

        def mk(st):
            mm = T.Model(8, 4, 4, s, head="linear", seed=seed); mm.load(st)
            return lambda X: mm.predict(pad(X))
        return {v: mk(snaps[v]) for v in variants}

    def describe(self, g, v):
        return f"scale={g[0]:.4f},l2={g[1]},steps={v}"


class TabPFN(Model):
    name = "C7_tabpfn"

    def groups(self, N):
        return [(0, [0])]

    def fit(self, Z, y, g, variants, seed):
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(device="cpu", random_state=0, ignore_pretraining_limits=True).fit(Z, y)
        return {0: lambda X: m.predict_proba(X)[:, 1]}

    def describe(self, g, v):
        return "default"


def models_for(family, k):
    if family == "main":
        return [VQC(k, False), VQC(k, True), QKernel(k)] + list(make_sk_models().values())
    return [BW(k)]


def run_model(M, Zs, ys, Zt, yt, seed, innerseed):
    t0 = time.time(); folds = inner_folds(ys, innerseed); acc = {}
    for g, variants in M.groups(len(ys)):
        for tri, vai in folds:
            sc = M.fit(Zs[tri], ys[tri], g, variants, seed)
            for v, f in sc.items():
                acc.setdefault((g, v), []).append(auc(ys[vai], f(Zs[vai])))
    best = max(acc, key=lambda kv: np.mean(acc[kv]))
    g, v = best; f = M.fit(Zs, ys, g, [v], seed)[v]
    tr_auc, te_auc = auc(ys, f(Zs)), auc(yt, f(Zt))
    return dict(model=M.name, test_auc=te_auc, train_auc=tr_auc, gap=tr_auc - te_auc, inner_cv_auc=float(np.mean(acc[best])), chosen=M.describe(g, v),
                n_params=getattr(M, "n_params", None), secs=round(time.time() - t0, 1))


def unit(a):
    family, k, fold, N, r, only, seedoff = a
    d = np.load(CACHE / f"f{fold}_N{N}_r{r}.npz")
    Zs, ys, Zt, yt = d["Zsub"].astype(np.float64), d["ysub"], d["Ztest"].astype(np.float64), d["yte"]
    out = []
    for M in models_for(family, k):
        if only and M.name not in only:
            continue
        try:
            rec = run_model(M, Zs[:, :k], ys, Zt[:, :k], yt, seed=1000 * fold + r + seedoff, innerseed=7 + 31 * fold + 101 * r + N)
        except Exception as e:
            rec = dict(model=M.name, error=f"{type(e).__name__}: {e}"[:300])
        rec.update(fold=fold, N=N, r=r, k=k, family=family, seedoff=seedoff); out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", required=True); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sizes", default="100,50,200,25,500"); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--only", default=""); ap.add_argument("--out", default=None); ap.add_argument("--seedoff", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out or f"results/sd_runs_{a.family}.jsonl"); only = set(filter(None, a.only.split(",")))
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l); done.add((r["model"], r["fold"], r["N"], r["r"], r["k"], r.get("seedoff", 0)))
    names = [M.name for M in models_for(a.family, a.k) if not only or M.name in only]
    units = [(a.family, a.k, f, N, r, only, a.seedoff) for N in map(int, a.sizes.split(",")) for r in range(a.reps) for f in map(int, a.folds.split(","))
             if any((n, f, N, r, a.k, a.seedoff) not in done for n in names)]
    print(len(units), "units to run", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(unit, u) for u in units]
        for fu in as_completed(futs):
            for rec in fu.result():
                if rec.get("error"):
                    print("ERROR", rec, flush=True); continue
                if (rec["model"], rec["fold"], rec["N"], rec["r"], rec["k"], rec["seedoff"]) in done:
                    continue
                open(out, "a").write(json.dumps(rec) + "\n")
            print("done unit", rec["fold"], rec["N"], rec["r"], flush=True)


if __name__ == "__main__":
    main()
