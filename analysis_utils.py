"""Shared analysis helpers (numpy only): cluster (patient-level) bootstrap of AUC and paired AUC difference, advantage rule."""
import numpy as np

def _wauc(y, s, W):
    """AUC for R weight vectors W (R,N) (multiplicities) on one score vector. Ties are broken by order (scores are continuous)."""
    o = np.argsort(s, kind="mergesort"); yo = y[o]; Wo = W[:, o]
    neg = Wo * (yo == 0); pos = Wo * (yo == 1)
    cumneg = np.cumsum(neg, 1) - neg
    num = (pos * cumneg).sum(1)
    return num / (pos.sum(1) * neg.sum(1))

def cluster_weights(groups, R, rng):
    """multiplicity weights (R, N) from resampling clusters with replacement."""
    u, inv = np.unique(groups, return_inverse=True)
    cnt = np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(R)])
    return cnt[:, inv].astype(float)

def boot_auc_diff(folds, R=2000, seed=0):
    """folds: list of dicts(y, g, sa, sb) per fold (sb may be None).  Returns point/CI of mean-over-folds AUC of a, and of a-b."""
    rng = np.random.default_rng(seed)
    A = np.zeros(R); Bm = np.zeros(R); nf = len(folds)
    for f in folds:
        W = cluster_weights(f["g"], R, rng)
        A += _wauc(f["y"], f["sa"], W) / nf
        if f.get("sb") is not None:
            Bm += _wauc(f["y"], f["sb"], W) / nf
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = dict(auc_a_ci=ci(A))
    if folds[0].get("sb") is not None:
        out.update(diff_ci=ci(A - Bm), p_diff_le0=float((A - Bm <= 0).mean()))
    return out

def auc(y, s):
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
