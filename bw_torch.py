"""Fast exact simulator of Antonio's magic-state brick-wall QNN in the k-particle sector (PyTorch, autograd).

|psi(x)> = U_cols(w) D(theta(x)) |phi0>,  |phi0> = U_had |Psi_in> precomputed once (data independent);
D = diagonal phases (elementwise multiply), U_cols = brick-wall columns of Rz layers + RBS (SingleExcitation) gates acting
inside the sector.  Conventions copied from the notebook's make_circuit: data RZ(-phi_j), column Rz(-w[col, n/2+j]),
SingleExcitation(-2 w[col, j]), ring pairs (odd columns include (n-1, 0)).  Readout m_j = <n_j n_{j+1}> (ring), f = 4 m.
Global phases are dropped (irrelevant for probabilities); bw_core verification compares full states incl. phase via bw_utils.
"""
from __future__ import annotations
import os, time
import numpy as np
import torch
import bw_utils as B

torch.set_num_threads(1)


class BW:
    def __init__(self, n, columns, k=None, dtype=torch.complex64, W=None):
        self.n, self.columns = n, columns
        self.k = B.particle_count(n, k)
        self.cdtype = dtype
        self.fdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        sec = B.Sector(n, columns, self.k, W)
        self.D = sec.D
        self.occ = torch.tensor(sec.occ, dtype=self.fdtype)                       # (D, n)
        self.phi0 = torch.tensor(sec.phi0, dtype=dtype)
        self.pidx = {key: (torch.tensor(a), torch.tensor(b)) for key, (a, b) in sec.pair_idx.items()}
        self.M_adj = torch.tensor(sec.M, dtype=self.fdtype)                       # (D, n)   <n_j n_{j+1}>
        iu = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self.M_all = torch.tensor(np.stack([sec.occ[:, i] * sec.occ[:, j] for i, j in iu], 1), dtype=self.fdtype)
        self.pairs_list = [B.pairs(n, c) for c in range(columns)]
        self.n_w = columns * 3 * n // 2

    def psi(self, theta, w):
        """theta: (B, n) rotation angles of the data RZ gates (RZ(theta_j), theta = -phi); w: (columns, 3n/2)."""
        n = self.n
        ph = theta @ self.occ.T                                                    # (B, D)
        psi = self.phi0 * torch.polar(torch.ones_like(ph), ph)
        for col in range(self.columns):
            th = -w[col, n // 2:]
            psi = psi * torch.polar(torch.ones(self.D, dtype=self.fdtype), self.occ @ th)
            for j, (a, b) in enumerate(self.pairs_list[col]):
                phi = -2 * w[col, j]
                c, s = torch.cos(phi / 2), torch.sin(phi / 2)
                i01, i10 = self.pidx[(a, b)]
                p01, p10 = psi[:, i01], psi[:, i10]
                new = psi.clone()
                new[:, i01] = c * p01 - s * p10
                new[:, i10] = s * p01 + c * p10
                psi = new
        return psi

    def features(self, theta, w, kind="adj"):
        p = self.psi(theta, w)
        pr = p.real ** 2 + p.imag ** 2
        return 4 * (pr @ (self.M_adj if kind == "adj" else self.M_all))


def auc(y, s):
    """rank-based AUC (ties averaged); numpy only (sklearn DLLs are flaky under Application Control)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sens_at_spec(y, s, spec=0.90):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    thr = np.quantile(s[y == 0], spec)             # >= spec of negatives at or below thr
    return float((s[y == 1] > thr).mean())


def lr_fit(F, y, C=1.0, iters=60):
    """balanced L2 logistic regression by LBFGS (torch), returns (w, b) for standardised features F (numpy)."""
    Ft = torch.tensor(F, dtype=torch.float64); yt = torch.tensor(y, dtype=torch.float64)
    cw = torch.where(yt == 1, len(y) / (2 * yt.sum()), len(y) / (2 * (len(y) - yt.sum())))
    w = torch.zeros(F.shape[1], dtype=torch.float64, requires_grad=True); b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); sc = Ft @ w + b
        l = (cw * (torch.nn.functional.softplus(sc) - yt * sc)).mean() + 0.5 / (C * len(y)) * (w ** 2).sum()
        l.backward(); return l
    opt.step(closure)
    return w.detach().numpy(), float(b)


class Model:
    """phase map + circuit + head.  head: 'linear' (n features), 'linear_score' (same, MSE on +-1, no sigmoid), 'corr' (n(n-1)/2)."""
    def __init__(self, n, columns, k, scale, head="linear", phase="tanh", xmax=None, dtype=torch.complex64, seed=0):
        self.bw = BW(n, columns, k, dtype)
        self.n, self.scale, self.head_kind, self.phase, self.xmax = n, scale, head, phase, xmax
        g = torch.Generator().manual_seed(seed)
        fd = self.bw.fdtype
        self.w = (0.1 * torch.randn(columns, 3 * n // 2, generator=g, dtype=fd)).requires_grad_(True)
        nf = n if head != "corr" else n * (n - 1) // 2
        self.h = (0.1 * torch.randn(nf, generator=g, dtype=fd)).requires_grad_(True)
        self.b = torch.zeros(1, dtype=fd, requires_grad=True)
        self.kind = "all" if head == "corr" else "adj"

    def theta(self, Z):
        Z = torch.as_tensor(Z, dtype=self.bw.fdtype)
        if self.phase == "tanh":
            phi = self.scale * torch.tanh(Z)
        else:                                  # paper map phi = pi x / max|x|
            phi = np.pi * Z / torch.as_tensor(self.xmax, dtype=self.bw.fdtype)
        return -phi

    def feats(self, Z):
        return self.bw.features(self.theta(Z), self.w, self.kind)

    def score(self, Z):
        return self.feats(Z) @ self.h + self.b

    @torch.no_grad()
    def predict(self, Z, chunk=2000):
        return np.concatenate([self.score(Z[i:i + chunk]).numpy() for i in range(0, len(Z), chunk)])

    def params(self):
        return [self.w, self.h, self.b]

    def state(self):
        return dict(w=self.w.detach().numpy().tolist(), h=self.h.detach().numpy().tolist(), b=float(self.b))

    def load(self, st):
        with torch.no_grad():
            self.w.copy_(torch.tensor(st["w"])); self.h.copy_(torch.tensor(st["h"])); self.b.fill_(st["b"])


def train_model(m, Ztr, ytr, Zva, yva, steps=600, batch=512, lr=0.02, seed=0, extend=True, warm=True, log=None):
    """Adam, class-weighted BCE (or MSE for linear_score), best-val-AUC checkpoint every 25 steps; if the best checkpoint lies in the last 20% of the run, the run is extended once by the same number of steps."""
    rng = np.random.default_rng(seed)
    n_tr = len(ytr); cw = np.array([n_tr / (2 * (n_tr - ytr.sum())), n_tr / (2 * ytr.sum())])
    fd = m.bw.fdtype
    ytr_t = torch.tensor(ytr, dtype=fd); swt = torch.tensor(np.where(ytr == 1, cw[1], cw[0]), dtype=fd)
    if warm:                                    # warm-start the linear head by logistic regression on the initial features
        idx = rng.choice(n_tr, size=min(2000, n_tr), replace=False)
        with torch.no_grad():
            F = m.feats(Ztr[idx]).numpy()
        sd = F.std(0) + 1e-9
        cw_, b_ = lr_fit(F / sd, ytr[idx])
        with torch.no_grad():
            m.h.copy_(torch.tensor(cw_ / sd, dtype=fd)); m.b.copy_(torch.tensor([b_], dtype=fd))
            if m.head_kind == "linear_score":
                m.h.mul_(0.25); m.b.mul_(0.25)
    opt = torch.optim.Adam(m.params(), lr=lr)
    best = (-1.0, m.state(), 0)
    t0 = time.time(); total = steps; t = 0
    while t < total:
        t += 1
        ib = torch.tensor(rng.choice(n_tr, size=min(batch, n_tr), replace=False))
        s = m.score(Ztr[ib.numpy()])
        yb, wb = ytr_t[ib], swt[ib]
        if m.head_kind == "linear_score":
            loss = (wb * 0.5 * (s - (2 * yb - 1)) ** 2).mean()
        else:
            loss = (wb * (torch.nn.functional.softplus(s) - yb * s)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if t % 25 == 0 or t == total:
            v = auc(yva, m.predict(Zva))
            if v > best[0]:
                best = (v, m.state(), t)
            if log:
                log(f"  step {t} loss {float(loss):.4f} val {v:.4f} best {best[0]:.4f}@{best[2]}")
            if t == total and extend and best[2] >= 0.8 * total and total == steps:
                total = 2 * steps                 # extend once
    m.load(best[1])
    return best[0], best[2], total, time.time() - t0
