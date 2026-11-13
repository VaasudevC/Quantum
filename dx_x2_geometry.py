"""DX / X2 (exploratory, DISCOVERY folds 0-2 only): geometric difference (Huang et al. 2021, "Power of data in quantum machine learning")
between classical kernels and the pi/6 angle-encoding quantum kernel.

  K_Q(x,x') = |<phi(x)|phi(x')>|^2 for the encoding-only angle circuit, a_i = (pi/6) tanh(z_i):  K_Q = prod_i cos^2((a_i - a'_i)/2)
             (a classically computable product kernel; unit diagonal)
  K_C      : (a) RBF exp(-gamma |x-x'|^2), gamma tuned by 5-fold CV AUC of an SVC (C=1) on the same sample; (b) linear x.x', rescaled to Tr K = N
  g(K_C || K_Q) = sqrt( || sqrt(K_Q) (K_C + lam I)^-1 sqrt(K_Q) ||_inf ),  lam in {1e-4, 1e-3, 1e-2, 1e-1}
Interpretation (Huang et al.): if g is small compared with sqrt(N), a classical model with kernel K_C can match the quantum kernel's prediction error for ANY labelling
(no room for a quantum prediction advantage); g ~ sqrt(N) is necessary (not sufficient) for an advantage.
Samples: N in {100, 500} ECGs drawn at random (class-stratified, seed per fold/N) from the outer-train part of folds 0, 1, 2; features = first 8 L1-selected,
standardised columns of the cached fold arrays.  No test fold, no folds 3-4, no external data.
  ..\\.venv\\Scripts\\python.exe dx_x2_geometry.py   ->  results/dx_x2_geometry.json, rows appended to results/dx_registry.csv"""
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

S = np.pi / 6; K = 8; LAMS = (1e-4, 1e-3, 1e-2, 1e-1)


def k_angle(Z):
    A = S * np.tanh(Z); Kq = np.ones((len(A), len(A)))
    for i in range(A.shape[1]):
        Kq *= np.cos((A[:, i][:, None] - A[:, i][None, :]) / 2) ** 2
    return Kq


def cv_auc(Kmat, y):
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Kmat, y):
        m = SVC(C=1, kernel="precomputed").fit(Kmat[np.ix_(tr, tr)], y[tr]); aucs.append(roc_auc_score(y[te], m.decision_function(Kmat[np.ix_(te, tr)])))
    return float(np.mean(aucs))


def sqrtm_psd(Km):
    w, V = np.linalg.eigh((Km + Km.T) / 2); return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def g(KC, KQ, lam):
    Sq = sqrtm_psd(KQ); M = Sq @ np.linalg.solve(KC + lam * np.eye(len(KC)), Sq)
    return float(np.sqrt(np.linalg.eigvalsh((M + M.T) / 2).max()))


def main():
    out = {}; reg = []
    for N in (100, 500):
        for fold in (0, 1, 2):
            d = np.load(f"C:/bwcache/l1_f{fold}.npz"); Z0, y0, tr = d["Z"].astype(float)[:, :K], d["y"], d["tr"]
            rng = np.random.default_rng(1000 * fold + N); npos = int(round(N * y0[tr].mean()))
            idx = np.concatenate([rng.choice(tr[y0[tr] == 1], npos, replace=False), rng.choice(tr[y0[tr] == 0], N - npos, replace=False)])
            Z, y = Z0[idx], y0[idx]
            KQ = k_angle(Z)
            d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1); med = np.median(d2[d2 > 0])
            best = max(((cv_auc(np.exp(-gm * d2), y), gm) for gm in (0.125 / med, 0.25 / med, 0.5 / med, 1 / med, 2 / med, 4 / med, 8 / med)))
            KR = np.exp(-best[1] * d2)
            KL = Z @ Z.T; KL *= N / np.trace(KL)
            rec = dict(N=N, fold=fold, sqrtN=float(np.sqrt(N)), cv_auc_quantum=cv_auc(KQ, y), cv_auc_rbf=best[0], rbf_gamma_over_median=float(best[1] * med),
                       cv_auc_linear=cv_auc(KL, y), rank_linear=int(np.linalg.matrix_rank(KL)),
                       eff_dim_quantum=float(np.trace(KQ) ** 2 / (KQ ** 2).sum()))
            for name, KC in (("rbf", KR), ("linear", KL)):
                for lam in LAMS:
                    rec[f"g_{name}_lam{lam}"] = g(KC, KQ, lam)
            out[f"N{N}_fold{fold}"] = rec
            print(N, fold, {k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()}, flush=True)
            for name in ("rbf", "linear"):
                for lam in LAMS:
                    reg.append(f"X2_g_{name}_N{N}_f{fold}_lam{lam},kernel-geometry (exploratory),g(K_{name}||K_Q angle pi/6) = {rec[f'g_{name}_lam{lam}']:.2f} vs sqrt(N) = {np.sqrt(N):.1f};"
                               f" CV AUC quantum {rec['cv_auc_quantum']:.3f} / rbf {rec['cv_auc_rbf']:.3f} / linear {rec['cv_auc_linear']:.3f},{fold},Q3-kernel,{rec['cv_auc_quantum']:.4f},"
                               f"{name},{rec['cv_auc_rbf' if name == 'rbf' else 'cv_auc_linear']:.4f},{rec['cv_auc_quantum'] - rec['cv_auc_rbf' if name == 'rbf' else 'cv_auc_linear']:+.4f},no,diagnostic only")
    json.dump(out, open("results/dx_x2_geometry.json", "w"), indent=1)
    open("results/dx_registry.csv", "a").write("\n".join(reg) + "\n")


if __name__ == "__main__":
    main()
