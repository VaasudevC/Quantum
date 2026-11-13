"""QB3 (pre-registered in QB.md): is the VQC's win over LR more than mild non-linearity?  Full data, k = 32, DISCOVERY folds 0-2.
 (1) degree-2 surrogate of the trained pi/6 VQC (results/stage2_steps1000.jsonl) and the trained-scale VQC (results/n2_trainscale.jsonl):
     ridge fit of the VQC logit on degree-2 features of the angles (train rows), R^2 on test rows, spectrum / rank of the quadratic-coefficient matrix,
     top feature pairs;
 (2) parameter-matched classical models (<= 113 trainable parameters) on the same 32 features: LR (33), factorization machines rank 1 and 2 (65, 97),
     MLP 32-2-1 / 32-3-1 tanh (69, 103), L1 degree-2 LR with <= 81 non-zero coefficients; each also on the bounded inputs a = (pi/6) tanh(z) (twin);
     hyper-parameters by validation AUC (inner validation split, as the VQC).
  ..\\.venv\\Scripts\\python.exe qb3_nonlinearity.py FOLD     -> results/qb3.jsonl (one line per fold x model x input)"""
import json, os, sys, time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
import scale_cv as S, Vqc_fast8 as V

OUT = Path("results/qb3.jsonl")


def vqc_rows(fold):
    a = [json.loads(l) for l in open("results/stage2_steps1000.jsonl")]
    a = [r for r in a if r["fold"] == fold and r["k"] == 32 and r["scale"] == "pi/6" and r["selector"] == "l1"][0]
    b = [json.loads(l) for l in open("results/n2_trainscale.jsonl")]
    b = [r for r in b if r["fold"] == fold and r.get("train_scale")][0]
    return a, b


def vqc_logit(row, Z, sel):
    T = np.zeros((len(Z), V.N_FEATURES)); T[:, :32] = np.tanh(Z[:, sel])
    m = S.Linear(32, eval(row["scale"], {"pi": np.pi}), row.get("readout", "z"), row.get("train_scale", False))
    th, w, b, s = m.split(np.asarray(row["params"])); A = m.angles(T, s)
    return V.expectations(V.forward(A, th), "z") @ w + b, A[:, :32]


def surrogate(logit, A, tr, te):
    pf = PolynomialFeatures(2, include_bias=False); P = pf.fit_transform(A); sc = StandardScaler().fit(P[tr]); Ps = sc.transform(P)
    best = None
    for al in (1e-3, 1e-2, 1e-1, 1, 10):
        m = Ridge(alpha=al).fit(Ps[tr], logit[tr]); r2 = r2_score(logit[te], m.predict(Ps[te]))
        if best is None or r2 > best[0]: best = (r2, al, m)
    r2, al, m = best
    coef = m.coef_ / sc.scale_                                           # back to raw angle-polynomial units
    names = pf.get_feature_names_out([f"x{i}" for i in range(32)]); Q = np.zeros((32, 32))
    for c, nm in zip(coef, names):
        parts = nm.split(" ")
        if len(parts) == 2:
            i, j = int(parts[0][1:]), int(parts[1][1:]); Q[i, j] += c / 2; Q[j, i] += c / 2
        elif "^2" in nm:
            i = int(nm.split("^")[0][1:]); Q[i, i] += c
    ev = np.sort(np.abs(np.linalg.eigvalsh(Q)))[::-1]; frac = np.cumsum(ev ** 2) / (ev ** 2).sum()
    r2lin = r2_score(logit[te], Ridge(alpha=1).fit(A[tr], logit[tr]).predict(A[te]))
    off = [(abs(Q[i, j]), i, j) for i in range(32) for j in range(i + 1, 32)]; off.sort(reverse=True)
    return dict(r2_deg2=float(r2), r2_linear=float(r2lin), ridge_alpha=al, eig_energy_top1=float(frac[0]), eig_energy_top2=float(frac[1]), eig_energy_top3=float(frac[2]),
                eff_rank=float((ev.sum() ** 2) / (ev ** 2).sum()), top_pairs=[[int(i), int(j), float(v)] for v, i, j in off[:6]], diag_share=float((np.diag(Q) ** 2).sum() / (Q ** 2).sum()))


# ------------------------------------------------------------------------------------------------ parameter-matched classical models
def auc(y, s): return float(roc_auc_score(y, s))


def fit_fm(X, y, r, l2, seed):
    n, d = X.shape; rng = np.random.default_rng(seed); w0 = np.concatenate([[0.0], np.zeros(d), rng.normal(0, 0.01, d * r)])
    pos = y.mean(); sw = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
    def f(p):
        b, w, Vm = p[0], p[1:d + 1], p[d + 1:].reshape(d, r)
        XV = X @ Vm; s = b + X @ w + 0.5 * ((XV ** 2).sum(1) - (X ** 2) @ (Vm ** 2).sum(1))
        q = 1 / (1 + np.exp(-s)); L = np.mean(sw * (np.logaddexp(0, s) - y * s)) + l2 * (w @ w + (Vm ** 2).sum())
        g = sw * (q - y) / n
        gb = g.sum(); gw = X.T @ g + 2 * l2 * w
        gV = X.T @ (g[:, None] * XV) - (X ** 2).T @ g[:, None] * Vm + 2 * l2 * Vm
        return L, np.concatenate([[gb], gw, gV.ravel()])
    res = minimize(f, w0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    p = res.x
    def score(Xn):
        b, w, Vm = p[0], p[1:d + 1], p[d + 1:].reshape(d, r); XV = Xn @ Vm
        return b + Xn @ w + 0.5 * ((XV ** 2).sum(1) - (Xn ** 2) @ (Vm ** 2).sum(1))
    return score


def models(X, y, tr, va, te):
    out = {}
    # LR (33)
    best = max(((auc(y[va], (m := LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr])).decision_function(X[va])), C, m) for C in (0.01, 0.1, 1, 10)), key=lambda t: t[0])
    out["LR"] = dict(params=33, val=best[0], test=auc(y[te], best[2].decision_function(X[te])), hp=dict(C=best[1]))
    # FM rank 1, 2
    for r, npar in ((1, 65), (2, 97)):
        cand = []
        for l2 in (1e-4, 1e-3, 1e-2):
            for seed in (0, 1):
                sc = fit_fm(X[tr], y[tr], r, l2, seed); cand.append((auc(y[va], sc(X[va])), dict(l2=l2, seed=seed), sc))
        b = max(cand, key=lambda t: t[0]); out[f"FM_rank{r}"] = dict(params=npar, val=b[0], test=auc(y[te], b[2](X[te])), hp=b[1])
    # MLP 32-h-1 tanh
    for h, npar in ((2, 69), (3, 103)):
        cand = []
        for al in (1e-4, 1e-2, 1):
            for seed in (0, 1, 2):
                m = MLPClassifier((h,), activation="tanh", alpha=al, max_iter=500, early_stopping=True, n_iter_no_change=20, random_state=seed).fit(X[tr], y[tr])
                cand.append((auc(y[va], m.predict_proba(X[va])[:, 1]), dict(alpha=al, seed=seed), m))
        b = max(cand, key=lambda t: t[0]); out[f"MLP_32-{h}-1"] = dict(params=npar, val=b[0], test=auc(y[te], b[2].predict_proba(X[te])[:, 1]), hp=b[1])
    # L1 degree-2 LR with <= 81 non-zero coefficients
    P = PolynomialFeatures(2, include_bias=False).fit_transform(X); P = StandardScaler().fit(P[tr]).transform(P); cand = []
    for C in np.logspace(-4, -1, 16):
        m = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=3000, random_state=0).fit(P[tr], y[tr]); nnz = int((np.abs(m.coef_) > 1e-10).sum())
        if nnz <= 81: cand.append((auc(y[va], m.decision_function(P[va])), dict(C=float(C), nnz=nnz), m))
    b = max(cand, key=lambda t: t[0]); out["L1_deg2_LR_le81"] = dict(params=b[1]["nnz"] + 1, val=b[0], test=auc(y[te], b[2].decision_function(P[te])), hp=b[1])
    return out


def main():
    fold = int(sys.argv[1]); assert fold in (0, 1, 2), "discovery folds only"
    X, y, _, _, feats = S.load(); Z, tr, va, te, _ = S.prep("grouped5", fold, "l1")
    ra, rb = vqc_rows(fold); sel = [feats.index(n) for n in ra["features"]]; selb = [feats.index(n) for n in rb["features"]]
    t0 = time.time(); rec = dict(fold=fold)
    la, Aa = vqc_logit(ra, Z, sel); lb, Ab = vqc_logit(rb, Z, selb)
    rec["vqc_pi6"] = dict(test=auc(y[te], la[te]), params=81, surrogate=surrogate(la, Aa, tr, te))
    rec["vqc_trained"] = dict(test=auc(y[te], lb[te]), params=113, surrogate=surrogate(lb, Ab, tr, te))
    Zs = Z[:, sel]; rec["features"] = ra["features"]
    rec["classical_z"] = models(Zs, y, tr, va, te)
    rec["classical_bounded_twin"] = models((np.pi / 6) * np.tanh(Zs), y, tr, va, te)
    rec["secs"] = round(time.time() - t0, 1)
    with open(OUT, "a") as fo: fo.write(json.dumps(rec) + "\n")
    print(fold, "done", rec["secs"], {k: round(v["test"], 4) for k, v in rec["classical_z"].items()}, round(rec["vqc_pi6"]["test"], 4), round(rec["vqc_trained"]["test"], 4), flush=True)


if __name__ == "__main__":
    main()
