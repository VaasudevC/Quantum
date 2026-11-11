"""Cache the CV data (selected, standardised columns) as npz in C:/bwcache so that torch workers need only numpy
(sklearn/pandas DLLs are intermittently blocked by Windows Application Control in the torch venv)."""
import numpy as np, scale_cv as S
X, y, g, sf, feats = S.load()
for fold in range(5):
    for sel in ("l1", "cohen_dedup"):
        Z, tr, va, te, order = S.prep("grouped5", fold, sel)
        o = np.array(order[:32])
        np.savez(f"C:/bwcache/{sel}_f{fold}.npz", Z=Z[:, o].astype(np.float32), order=np.array(order), tr=tr, va=va, te=te,
                 y=y, g=g, names=np.array([feats[j] for j in o]))
        print(fold, sel, Z.shape, flush=True)
