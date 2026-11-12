"""DX / X2: kernel-geometry diagnostic (Huang et al. 2021, "Power of data in quantum machine learning"), DISCOVERY folds 0-2 only, no SD data touched.

For a random stratified sample of M ECGs from the outer-train part of each discovery fold (features = the first k L1-selected, standardised columns of the cached fold arrays):
  K_Q  : quantum fidelity kernels  (angle encoding a = s*tanh(z): K = prod_i cos^2((a_i-a'_i)/2);  brick-wall encoding state |psi(x)> = D(phi(x))|phi0>, n=8, k=4, phase s*tanh(z))
  K_C  : classical RBF kernel exp(-gamma |x-x'|^2) with gamma tuned by 5-fold CV AUC of an SVC (C=1) on the same sample (labels used for tuning only)
  g(K_C || K_Q) = sqrt( || sqrt(K_Q) (K_C + lam I)^-1 sqrt(K_Q) ||_inf ),  both kernels with unit diagonal (Tr K = M), lam in {1e-4, 1e-3, 1e-2}
  also the reverse g(K_Q || K_C) and the projected quantum kernels (angle encoding: 1-qubit RDMs; brick-wall: adjacent-pair 2-RDMs; the 1-RDMs of the particle-number
  sector are diagonal and data-independent, hence useless as features), with gamma tuned the same way.
Small g  => no quantum-kernel advantage possible for ANY labelling; large g => possible (necessary, not sufficient).
  python dx_x2_geometry.py [M=600]  ->  results/dx_x2_geometry.json, appended to results/dx_registry.csv"""
import json, sys, time
import numpy as np
import bw_utils as B
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

M = int(sys.argv[1]) if len(sys.argv) > 1 else 600
SCALES = (np.pi / 8, np.pi / 6, np.pi / 4, np.pi / 2)


def fid_angle(Z, s):
    A = s * np.tanh(Z); K = np.ones((len(A), len(A)))
    for i in range(A.shape[1]):
        K *= np.cos((A[:, i][:, None] - A[:, i][None, :]) / 2) ** 2
    return K


def bw_states(Z, s, sec):
    """|psi(x)> in the k=4 sector of n=8 qubits (RZ(-phi) data layer on |phi0>): amplitudes (M, D)."""
    th = -s * np.tanh(Z)
    return sec.phi0[None, :] * np.exp(1j * (th @ sec.occ.T))


def fid_bw(psi):
    G = np.conj(psi) @ psi.T
    return np.abs(G) ** 2


def rdm1_angle(Z, s):
    a = s * np.tanh(Z); c, sn = np.cos(a / 2), np.sin(a / 2)
    return np.stack([c * c, c * sn, c * sn, sn * sn], -1)          # (M, k, 4) real RDM entries


def rdm2_bw(psi, sec, n=8):
    feats = []
    idx = {tuple(r): i for i, r in enumerate(sec.occ)}
    for j in range(n):
        a, b = j, (j + 1) % n
        p = np.abs(psi) ** 2
        f00 = p[:, (sec.occ[:, a] == 0) & (sec.occ[:, b] == 0)].sum(1); f11 = p[:, (sec.occ[:, a] == 1) & (sec.occ[:, b] == 1)].sum(1)
        f01 = p[:, (sec.occ[:, a] == 0) & (sec.occ[:, b] == 1)].sum(1); f10 = p[:, (sec.occ[:, a] == 1) & (sec.occ[:, b] == 0)].sum(1)
        i01 = [i for i, r in enumerate(sec.occ) if r[a] == 0 and r[b] == 1]; i10 = [idx[tuple(np.where((np.arange(n) == a) | (np.arange(n) == b), 1 - sec.occ[i], sec.occ[i]))] for i in i01]
        coh = (psi[:, i01] * np.conj(psi[:, i10])).sum(1)
        feats.append(np.stack([f00, f01, f10, f11, coh.real, coh.imag], -1))
    return np.stack(feats, 1)


def proj_kernel(F, gamma):
    X = F.reshape(len(F), -1); sq = (X ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    return np.exp(-gamma * np.maximum(D, 0))


def tune_gamma(Kfun, gammas, y):
    best = (-1, None)
    for g in gammas:
        K = Kfun(g); aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(K, y):
            m = SVC(C=1, kernel="precomputed").fit(K[np.ix_(tr, tr)], y[tr]); aucs.append(roc_auc_score(y[te], m.decision_function(K[np.ix_(te, tr)])))
        if np.mean(aucs) > best[0]: best = (float(np.mean(aucs)), g)
    return best


def sqrtm_psd(K):
    w, V = np.linalg.eigh((K + K.conj().T) / 2); w = np.clip(w, 0, None); return (V * np.sqrt(w)) @ V.conj().T


def geom(KC, KQ, lam):
    S = sqrtm_psd(KQ); inv = np.linalg.inv(KC + lam * np.eye(len(KC)))
    return float(np.sqrt(np.linalg.eigvalsh((S @ inv @ S + (S @ inv @ S).conj().T) / 2).max()))


def main():
    sec = B.Sector(8, 0, 4); out = {}; reg = []
    for k in (4, 8):
        for fold in (0, 1, 2):
            d = np.load(f"C:/bwcache/l1_f{fold}.npz"); Z0, y0, tr = d["Z"].astype(float)[:, :k], d["y"], d["tr"]
            rng = np.random.default_rng(100 + fold); pos = rng.choice(tr[y0[tr] == 1], M // 3, replace=False); neg = rng.choice(tr[y0[tr] == 0], M - M // 3, replace=False)
            idx = np.concatenate([pos, neg]); Z = Z0[idx]; y = y0[idx]
            d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1); med = np.median(d2[d2 > 0]); gl = [0.25 / med, 0.5 / med, 1 / med, 2 / med, 4 / med, 8 / med, 16 / med]
            cv, gC = tune_gamma(lambda g: np.exp(-g * d2), gl, y); KC = np.exp(-gC * d2)
            res = dict(rbf_gamma=gC, rbf_cv_auc=cv)
            def add(name, KQ, extra=None):
                cvq = None
                try:
                    cvq = tune_gamma(lambda g: KQ, [1.0], y)[0]            # CV AUC of the quantum kernel itself (no tuning)
                except Exception: pass
                res[name] = dict(cv_auc_quantum_kernel=cvq, **{f"g_C_to_Q_lam{l}": geom(KC, KQ, l) for l in (1e-4, 1e-3, 1e-2)}, **{f"g_Q_to_C_lam{l}": geom(KQ, KC, l) for l in (1e-3,)}, **(extra or {}))
                reg.append(dict(k=k, fold=fold, kernel=name, **{kk: vv for kk, vv in res[name].items()}))
            for s in SCALES:
                add(f"angle_fid_s{s:.3f}", fid_angle(Z, s))
            if k >= 4:
                for s in SCALES[1:]:
                    Zp = np.concatenate([Z, np.zeros((len(Z), 8 - k))], 1); psi = bw_states(Zp, s, sec)
                    add(f"bw8_fid_s{s:.3f}", fid_bw(psi))
            # projected kernels (angle encoding 1-RDM; brick-wall 2-RDM)
            for s in (np.pi / 6, np.pi / 4):
                F = rdm1_angle(Z, s); Fm = F.reshape(len(F), -1); dm = ((Fm[:, None] - Fm[None]) ** 2).sum(-1); md = np.median(dm[dm > 0])
                cvp, gp = tune_gamma(lambda g: np.exp(-g * dm), [0.25 / md, 1 / md, 4 / md, 16 / md], y)
                add(f"angle_proj1rdm_s{s:.3f}", np.exp(-gp * dm), dict(proj_gamma=gp))
                Zp = np.concatenate([Z, np.zeros((len(Z), 8 - k))], 1); F2 = rdm2_bw(bw_states(Zp, s, sec), sec); Fm = F2.reshape(len(F2), -1); dm = ((Fm[:, None] - Fm[None]) ** 2).sum(-1); md = np.median(dm[dm > 0]) if (dm > 0).any() else 1
                cvp, gp = tune_gamma(lambda g: np.exp(-g * dm), [0.25 / md, 1 / md, 4 / md, 16 / md], y)
                add(f"bw8_proj2rdm_s{s:.3f}", np.exp(-gp * dm), dict(proj_gamma=gp))
            out[f"k{k}_fold{fold}"] = res; print(k, fold, {n: round(v["g_C_to_Q_lam0.001"], 2) for n, v in res.items() if isinstance(v, dict)}, flush=True)
    json.dump(out, open("results/dx_x2_geometry.json", "w"), indent=1)
    # summary: mean over folds
    names = sorted({r["kernel"] for r in reg}); lines = []
    for k in (4, 8):
        for n in names:
            rs = [r for r in reg if r["k"] == k and r["kernel"] == n]
            if rs:
                g = np.mean([r["g_C_to_Q_lam0.001"] for r in rs]); cv = np.mean([r["cv_auc_quantum_kernel"] or np.nan for r in rs])
                lines.append(f"X2_geometry_k{k}_{n},kernel-geometry,g(K_C||K_Q) lam=1e-3 mean over folds 0-2 = {g:.2f}; CV AUC of the quantum kernel SVM(C=1) = {cv:.3f},0-2,,,,,,no,diagnostic only (no quantum-vs-classical AUC test)")
    open("results/dx_registry.csv", "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
