"""QB1 (pre-registered in QB.md): training-free screen of quantum feature maps F0-F5 on DISCOVERY folds 0-2 only.
Exact numpy statevector simulation (8 qubits, 256 amplitudes; F5 in the 70-dim particle sector).  Resumable: one JSON line per
(fold, N, k, map, params, scale) in results/qb1_screen.jsonl.
  ..\\.venv\\Scripts\\python.exe qb1_screen.py FOLD [FOLD ...]        (one process per fold; <= 3 processes)"""
import json, os, sys, time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import Vqc_fast8 as V
import bw_utils as BU

NQ, DIM = 8, 256
OUT = Path("results/qb1_screen.jsonl")
SCALES = {"pi/8": np.pi / 8, "pi/6": np.pi / 6, "pi/4": np.pi / 4, "pi/3": np.pi / 3, "pi/2": np.pi / 2}
LAMS = (1e-3, 1e-2, 1e-1)
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_IDX = np.arange(DIM)
_BIT = np.stack([(_IDX >> (NQ - 1 - q)) & 1 for q in range(NQ)])        # big-endian, as Vqc_fast8
_Z = 1 - 2 * _BIT                                                     # (8, 256)


# ------------------------------------------------------------------------------------------------ state builders (B, 256)
def ry(psi, q, a):
    B = psi.shape[0]; st = psi.reshape(B, 1 << q, 2, 1 << (NQ - 1 - q)); c = np.cos(a / 2)[:, None, None]; s = np.sin(a / 2)[:, None, None]
    out = np.empty_like(st); out[:, :, 0] = c * st[:, :, 0] - s * st[:, :, 1]; out[:, :, 1] = s * st[:, :, 0] + c * st[:, :, 1]; return out.reshape(B, DIM)


def h_all(psi):
    B = psi.shape[0]
    for q in range(NQ):
        st = psi.reshape(B, 1 << q, 2, 1 << (NQ - 1 - q)); out = np.empty_like(st)
        out[:, :, 0] = (st[:, :, 0] + st[:, :, 1]) / np.sqrt(2); out[:, :, 1] = (st[:, :, 0] - st[:, :, 1]) / np.sqrt(2); psi = out.reshape(B, DIM)
    return psi


def zero(B):
    psi = np.zeros((B, DIM), complex); psi[:, 0] = 1; return psi


def rz_layer(psi, A):          # prod_i Rz(a_i): phase exp(-i/2 sum_i a_i z_i)
    return psi * np.exp(-0.5j * (A @ _Z))


def rzz(psi, theta, i, j):     # RZZ(theta) = exp(-i theta/2 Z_i Z_j)
    return psi * np.exp(-0.5j * theta[:, None] * (_Z[i] * _Z[j])[None, :])


def swap_perm(i, j):
    bi, bj = (_IDX >> (NQ - 1 - i)) & 1, (_IDX >> (NQ - 1 - j)) & 1
    return _IDX ^ ((bi != bj) * ((1 << (NQ - 1 - i)) | (1 << (NQ - 1 - j))))


def F0(A):
    psi = zero(len(A))
    for q in range(NQ): psi = ry(psi, q, A[:, q])
    return psi


def F1(A, L):
    psi = zero(len(A))
    for l in range(L):
        for q in range(NQ): psi = ry(psi, q, A[:, q])
        if l < L - 1: psi = psi[:, V._RING]
    return psi


def F2(A, reps, c, pairs):
    psi = zero(len(A))
    for _ in range(reps):
        psi = h_all(psi); psi = rz_layer(psi, A)
        for (i, j) in pairs: psi = rzz(psi, c * A[:, i] * A[:, j], i, j)
    return psi


def F4(A, T):
    psi = h_all(zero(len(A))); dt = 1.0 / T
    for _ in range(T):
        psi = rz_layer(psi, 2 * dt * A)                                   # exp(-i dt a_i Z_i) = Rz(2 dt a_i)
        for start in (0, 1):
            for i in range(start, NQ, 2):
                j = (i + 1) % NQ; P = swap_perm(i, j)                    # exp(-i dt (XX+YY+ZZ)) = e^{i dt}(cos 2dt I - i sin 2dt SWAP)
                psi = np.exp(1j * dt) * (np.cos(2 * dt) * psi - 1j * np.sin(2 * dt) * psi[:, P])
    return psi


_BW = None
def F5(A):
    global _BW
    if _BW is None: _BW = BU.Sector(8, 0, 4)
    return _BW.phi0[None, :] * np.exp(1j * ((-A) @ _BW.occ.T.astype(float)))


def rdm1_feats(psi):
    B = psi.shape[0]; F = []
    for q in range(NQ):
        st = psi.reshape(B, 1 << q, 2, 1 << (NQ - 1 - q)); p0, p1 = st[:, :, 0].reshape(B, -1), st[:, :, 1].reshape(B, -1)
        r00 = (np.abs(p0) ** 2).sum(1); r11 = (np.abs(p1) ** 2).sum(1); r01 = (p0 * np.conj(p1)).sum(1)
        F.append(np.stack([r00, r11, np.sqrt(2) * r01.real, np.sqrt(2) * r01.imag], 1))
    return np.concatenate(F, 1)


# ------------------------------------------------------------------------------------------------ kernel diagnostics
def cv_auc(K, y):
    a = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(K, y):
        m = SVC(C=1, kernel="precomputed").fit(K[np.ix_(tr, tr)], y[tr]); a.append(roc_auc_score(y[te], m.decision_function(K[np.ix_(te, tr)])))
    return float(np.mean(a))


def sqrtm_psd(K):
    w, U = np.linalg.eigh((K + K.T) / 2); return (U * np.sqrt(np.clip(w, 0, None))) @ U.T


def gdiff(KC, KQ, lam):
    S = sqrtm_psd(KQ); M = S @ np.linalg.solve(KC + lam * np.eye(len(KC)), S); return float(np.sqrt(np.linalg.eigvalsh((M + M.T) / 2).max()))


def centered_alignment(K, y):
    n = len(K); H = np.eye(n) - 1.0 / n; Kc = H @ K @ H; t = 2.0 * y - 1; T = np.outer(t, t); Tc = H @ T @ H
    return float((Kc * Tc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Tc) + 1e-300))


def diag_stats(K, y, KR):
    off = K[~np.eye(len(K), dtype=bool)]
    r = dict(cv_auc=cv_auc(K, y), calign=centered_alignment(K, y), eff_dim=float(np.trace(K) ** 2 / (K ** 2).sum()), off_mean=float(off.mean()), off_var=float(off.var()))
    for lam in LAMS: r[f"g_lam{lam}"] = gdiff(KR, K, lam)
    return r


def proj_kernel(F, y):
    D = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1); md = np.median(D[D > 1e-12]) if (D > 1e-12).any() else None
    if md is None:
        return None, "degenerate (data-independent 1-RDMs)"
    best = max(((cv_auc(np.exp(-g * D), y), g) for g in (0.25 / md, 1 / md, 4 / md, 16 / md)))
    return np.exp(-best[1] * D), f"gamma={best[1]:.4g}"


# ------------------------------------------------------------------------------------------------ pairs
def lead(name): return name.split("_")[0]


def clinical_pairs(names):
    P = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = lead(names[i]), lead(names[j])
            same = a == b
            prec = a.startswith("V") and b.startswith("V") and abs(int(a[1:]) - int(b[1:])) == 1
            recip = (a in ("II", "III", "aVF") and b in ("I", "aVL")) or (b in ("II", "III", "aVF") and a in ("I", "aVL"))
            inf = a != b and a in ("II", "III", "aVF") and b in ("II", "III", "aVF")
            if same or prec or recip or inf: P.append((i, j))
    return P


def settings(k, names):
    ring = [(i, (i + 1) % NQ) for i in range(NQ)]; cp = clinical_pairs(list(names[:k]))
    S = [("F0", "", lambda A: F0(A))]
    S += [("F1", f"L={L}", (lambda L: lambda A: F1(A, L))(L)) for L in (2, 3, 4)]
    S += [("F2", f"reps={r},c={c}", (lambda r, c: lambda A: F2(A, r, c, ring))(r, c)) for r in (1, 2) for c in (0.5, 1, 2)]
    S += [("F3", f"reps={r},c={c},pairs={len(cp)}", (lambda r, c: lambda A: F2(A, r, c, cp))(r, c)) for r in (1, 2) for c in (0.5, 1, 2)]
    S += [("F4", f"T={T}", (lambda T: lambda A: F4(A, T))(T)) for T in (1, 2, 3)]
    S += [("F5", "", lambda A: F5(A))]
    return S, cp


def main():
    folds = [int(f) for f in sys.argv[1:]] or [0, 1, 2]
    assert all(f in (0, 1, 2) for f in folds), "discovery folds only"
    done = set()
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l); done.add((r["fold"], r["N"], r["k"], r["map"], r["params"], r["scale"]))
    for fold in folds:
        d = np.load(f"C:/bwcache/l1_f{fold}.npz"); Z0, y0, tr, names = d["Z"].astype(float), d["y"], d["tr"], list(d["names"])
        for N in (100, 500):
            rng = np.random.default_rng(1000 * fold + N); npos = int(round(N * y0[tr].mean()))
            idx = np.concatenate([rng.choice(tr[y0[tr] == 1], npos, replace=False), rng.choice(tr[y0[tr] == 0], N - npos, replace=False)])   # identical to dx_x2_geometry.py
            y = y0[idx]
            for k in (8, 4):
                Z = Z0[idx][:, :k]
                d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1); med = np.median(d2[d2 > 0])
                rb = max(((cv_auc(np.exp(-g * d2), y), g) for g in (0.125 / med, 0.25 / med, 0.5 / med, 1 / med, 2 / med, 4 / med, 8 / med)))
                KR = np.exp(-rb[1] * d2); rbf = dict(cv_auc=rb[0], calign=centered_alignment(KR, y))
                S, cp = settings(k, names)
                for (mp, params, fn) in S:
                    for sname, s in SCALES.items():
                        key = (fold, N, k, mp, params, sname)
                        if key in done: continue
                        t0 = time.time(); A = np.zeros((N, NQ)); A[:, :k] = s * np.tanh(Z)
                        psi = fn(A)
                        G = np.conj(psi) @ psi.T; KQ = np.abs(G) ** 2
                        rec = dict(fold=fold, N=N, k=k, map=mp, params=params, scale=sname, sqrtN=float(np.sqrt(N)), rbf=rbf, fid=diag_stats(KQ, y, KR),
                                   clinical_pairs=[[names[i], names[j]] for i, j in cp] if mp == "F3" else None)
                        if mp == "F5":
                            KP, note = None, "degenerate (particle-number sector: 1-RDMs diagonal and data-independent)"
                        else:
                            KP, note = proj_kernel(rdm1_feats(psi), y)
                        rec["proj"] = diag_stats(KP, y, KR) if KP is not None else None; rec["proj_note"] = note
                        rec["secs"] = round(time.time() - t0, 2)
                        with open(OUT, "a") as fo: fo.write(json.dumps(rec) + "\n")
                print(fold, N, k, "done", flush=True)


if __name__ == "__main__":
    main()
