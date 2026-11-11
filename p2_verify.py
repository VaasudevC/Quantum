"""P2 verification of the fast sector simulator (bw_torch.BW) against independent dense 2^n references.

  ref_dense  : our own gate-level statevector simulator of the notebook circuit (numpy, 2^n amplitudes; wire 0 = most significant bit
               like PennyLane); U_had applied as the exact many-body Jordan-Wigner passive-FLO unitary exp(sum_ij logW_ij a+_i a_j)
               via sparse expm_multiply (independent of the determinant formula used by the fast simulator).
  pennylane  : the notebook's make_circuit verbatim (if pennylane is importable), with UniformSpreading replaced by StatePrep of the
               spread state (so this checks the circuit assembly: sign conventions, ring pairs, column order, readout).
Checks: (a) state agreement (full complex state, global phase included for the numpy sector simulator; torch c128 up to global phase),
(b) particle number exact, (c) Lemma 2 occupations k/n after spreading and after the data layer, (d) autograd vs finite differences.
Writes results/bw/p2_verify.json.
"""
import json, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.linalg import logm
from scipy.sparse.linalg import expm_multiply
import torch
import bw_utils as B
import bw_torch as T

rng = np.random.default_rng(0)


def jw_ops(n):
    I2, Zm = sp.identity(2, format="csr"), sp.csr_matrix([[1, 0], [0, -1]])
    sig = sp.csr_matrix([[0, 0], [1, 0]])            # |1><0|  (creation); basis index 0 = |0>, 1 = |1>
    def kron_all(ms):
        out = ms[0]
        for m in ms[1:]:
            out = sp.kron(out, m, format="csr")
        return out
    return [kron_all([Zm] * j + [sig] + [I2] * (n - j - 1)) for j in range(n)]          # wire 0 = most significant


def dense_phi0(n, k, W):
    """gate-level magic prep on 2^n vector, then exact many-body spreading."""
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    def apply1(psi, U, q):
        t = psi.reshape([2] * n); t = np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q); return t.reshape(-1)
    def apply_cnot(psi, c, tg):
        t = psi.reshape([2] * n).copy(); sl = [slice(None)] * n; sl[c] = 1
        sub = t[tuple(sl)]; ax = tg if tg < c else tg - 1
        t[tuple(sl)] = np.flip(sub, axis=ax); return t.reshape(-1)
    X = np.array([[0, 1], [1, 0]]); H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for blk in range(k // 2):
        a, b, c, d = range(4 * blk, 4 * blk + 4)
        psi = apply1(psi, X, a); psi = apply1(psi, X, b); psi = apply1(psi, H, c)
        for tg in (d, a, b):
            psi = apply_cnot(psi, c, tg)
    ops = jw_ops(n)
    L = np.real_if_close(logm(W))
    G = sp.csr_matrix((2 ** n, 2 ** n), dtype=complex)
    for i in range(n):
        for j in range(n):
            if abs(L[i, j]) > 1e-14:
                G = G + L[i, j] * (ops[i] @ ops[j].T)
    return expm_multiply(G, psi)


def dense_state(n, columns, k, x, w, phi0):
    psi = phi0.copy()
    def apply1(psi, U, q):
        t = psi.reshape([2] * n); t = np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q); return t.reshape(-1)
    def apply2(psi, U, a, b):
        t = psi.reshape([2] * n); U4 = U.reshape(2, 2, 2, 2)
        t = np.tensordot(U4, t, axes=([2, 3], [a, b])); t = np.moveaxis(t, [0, 1], [a, b]); return t.reshape(-1)
    RZ = lambda th: np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2)])
    def SE(phi):
        c, s = np.cos(phi / 2), np.sin(phi / 2)
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], complex)
    for j in range(n):
        psi = apply1(psi, RZ(-np.pi * x[j]), j)
    for col in range(columns):
        for j in range(n):
            psi = apply1(psi, RZ(-w[col, n // 2 + j]), j)
        for j, (a, b) in enumerate(B.pairs(n, col)):
            psi = apply2(psi, SE(-2 * w[col, j]), a, b)
    return psi


def dense_features(n, psi):
    p = np.abs(psi) ** 2; idx = np.arange(2 ** n)
    bit = lambda q: (idx >> (n - 1 - q)) & 1
    return 4 * np.array([(p * bit(j) * bit((j + 1) % n)).sum() for j in range(n)]), np.array([(p * bit(j)).sum() for j in range(n)])


def sector_to_dense(n, occ, psi_sec):
    out = np.zeros(2 ** n, complex)
    idx = (occ * (2 ** (n - 1 - np.arange(n)))).sum(1)
    out[idx] = psi_sec
    return out


def pennylane_check(n, columns, k, phi0, xs, ws):
    import pennylane as qml
    def pairs(col):
        out = [(j, j + 1) for j in range(col % 2, n - 1, 2)]
        return out + ([(n - 1, 0)] if col % 2 else [])
    dev = qml.device("default.qubit", wires=n)
    @qml.qnode(dev, interface=None, diff_method=None)
    def circuit(x, w):
        qml.StatePrep(phi0, wires=range(n))
        for j in range(n): qml.RZ(-np.pi * x[..., j], wires=j)
        for col in range(columns):
            for j in range(n): qml.RZ(-w[..., col, n // 2 + j], wires=j)
            for j, pair in enumerate(pairs(col)):
                qml.SingleExcitation(-2 * w[..., col, j], wires=pair)
        return tuple(qml.expval(qml.Projector([1, 1], wires=[j, (j + 1) % n])) for j in range(n))
    return max(np.abs(4 * np.array(circuit(x, w)) - dense_features(n, dense_state(n, columns, k, x, w, phi0))[0]).max()
               for x, w in zip(xs, ws))


def main():
    res = {}
    try:
        import pennylane  # noqa
        have_pl = True
    except Exception as e:
        have_pl = False; res["pennylane"] = f"not importable: {e}"
    for n, ks in ((8, (2, 4)), (12, (2, 4, 6))):
        W = B.spreading_matrix(n)
        for k in ks:
            columns = 2
            t0 = time.time()
            phi0_d = dense_phi0(n, k, W)
            S = B.Sector(n, columns, k)
            bw = T.BW(n, columns, k, dtype=torch.complex128)
            occ = S.occ
            r = {}
            # (b) particle number of dense phi0, (c) occupations
            idx = np.arange(2 ** n); wt = np.array([bin(i).count("1") for i in idx])
            r["b_weight_leak_phi0"] = float(np.abs(phi0_d[wt != k]).max())
            r["a_phi0_sector_vs_dense"] = float(np.abs(sector_to_dense(n, occ, S.phi0) - phi0_d).max())
            occ_phi0 = np.abs(S.phi0) ** 2 @ occ
            r["c_lemma2_max_dev_phi0"] = float(np.abs(occ_phi0 - k / n).max())
            ea, eb, ec, ed, ee = [], [], [], [], []
            xs, ws = [], []
            for _ in range(20):
                x = rng.uniform(-1, 1, n); w = rng.normal(0, 0.8, (columns, 3 * n // 2)); xs.append(x); ws.append(w)
                dd = dense_state(n, columns, k, x, w, phi0_d)
                ss = sector_to_dense(n, occ, S.state(x, w))
                ea.append(np.abs(dd - ss).max())                                   # full state incl. global phase
                eb.append(np.abs(dd[wt != k]).max())                               # (b)
                f_d, occ_d = dense_features(n, dd)
                ec.append(np.abs(occ_d - k / n).max() if columns == 0 else 0.0)
                ps = bw.psi(torch.tensor(-np.pi * x[None]), torch.tensor(w))[0].numpy()
                ps_d = sector_to_dense(n, occ, ps)
                ph = np.vdot(ps_d, dd); ph /= abs(ph)
                ed.append(np.abs(ps_d * ph - dd).max())                            # torch c128 up to global phase
                ee.append(np.abs(4 * (np.abs(ps) ** 2 @ bw.M_adj.numpy()) - f_d).max())
            r["a_sector_numpy_vs_dense_max"] = float(max(ea))
            r["a_torch_c128_vs_dense_max_up_to_phase"] = float(max(ed))
            r["a_features_torch_vs_dense_max"] = float(max(ee))
            r["b_weight_leak_state"] = float(max(eb))
            # Lemma 2 after the data layer (before trainable columns): |amplitudes| unchanged -> same occupations (check explicitly)
            x = xs[0]; S0 = B.Sector(n, 0, k); st = S0.state(x, np.zeros((0, 3 * n // 2)))
            r["c_lemma2_max_dev_after_data_layer"] = float(np.abs(np.abs(st) ** 2 @ occ - k / n).max())
            if have_pl:
                r["a_pennylane_features_max"] = float(pennylane_check(n, columns, k, phi0_d, xs[:5], ws[:5]))
            # (d) gradients vs finite differences (complex128)
            m = T.Model(n, columns, k, np.pi / 6, head="linear", dtype=torch.complex128, seed=1)
            Z = torch.tensor(rng.normal(size=(16, n))); y = torch.tensor((rng.random(16) > .5).astype(float))
            def loss_fn():
                s = m.score(Z); return (torch.nn.functional.softplus(s) - y * s).mean()
            m.w.data += 0.3 * torch.randn_like(m.w); m.h.data += torch.randn_like(m.h)
            l = loss_fn(); l.backward()
            gw = m.w.grad.clone(); gh = m.h.grad.clone()
            fd_err = 0.0
            for (p, g) in ((m.w, gw), (m.h, gh)):
                flat = p.data.view(-1)
                for i in range(min(len(flat), 12)):
                    old = flat[i].item(); e = 1e-6
                    flat[i] = old + e; lp = loss_fn().item(); flat[i] = old - e; lm = loss_fn().item(); flat[i] = old
                    fd_err = max(fd_err, abs((lp - lm) / (2 * e) - g.view(-1)[i].item()))
            r["d_grad_vs_fd_max_abs_err"] = float(fd_err)
            r["secs"] = round(time.time() - t0, 1)
            res[f"n{n}_k{k}"] = r
            print(n, k, json.dumps(r), flush=True)
    res["uniformity_defect"] = {str(n): B.uniformity_defect(B.spreading_matrix(n)) for n in (8, 12, 16)}
    res["literal_hadamard_brickwall_defect"] = {str(n): B.uniformity_defect(B.hadamard_brickwall_matrix(n)) for n in (8, 12, 16)}
    json.dump(res, open("results/bw/p2_verify.json", "w"), indent=1)


if __name__ == "__main__":
    main()
