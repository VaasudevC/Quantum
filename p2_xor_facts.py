"""P2: notebook toy XOR re-run with the fast simulator + circuit facts table."""
import json, numpy as np, torch
import bw_utils as B, bw_torch as T

def make_data(size, n, seed=0):          # verbatim from notebook
    rng = np.random.default_rng(seed)
    corners = np.tile([[-1, -1], [-1, 1], [1, -1], [1, 1]], (size // 4, 1))
    X = np.repeat(rng.uniform(-0.06, 0.06, (size // 4, n)), 4, axis=0)
    X[:, :2] += 0.20 * corners
    y = (corners[:, 0] != corners[:, 1]).astype(int)
    order = rng.permutation(size)
    return X[order], y[order]

def notebook_train(n, columns, k, steps=100, batch=8, Q_LR=0.5, H_LR=1.0, seed=None, exact=True):
    """Same loop as the notebook (balanced mini-batch 8, plain gradient steps, head step lr 1.0, quantum step lr 0.5 on ONE column per
    step cyclically) but with the EXACT gradient (autograd) in place of the stochastic parallel parameter-shift estimate."""
    rng = np.random.default_rng(42 + n if seed is None else seed)
    X, y = make_data(96, n, seed=42 + n); Xt, yt = make_data(48, n, seed=142 + n)
    bw = T.BW(n, columns, k, dtype=torch.complex128)
    kk = bw.k
    w = torch.tensor(rng.normal(0, 0.1, (columns, 3 * n // 2)), requires_grad=True)
    head = torch.tensor(rng.normal(0, 0.1, n + 1), requires_grad=True)
    hist = []
    for step in range(steps):
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c), batch // 2, False) for c in [0, 1]])
        f = bw.features(torch.tensor(-np.pi * X[idx]), w)
        score = f @ head[:-1] + head[-1]; yy = torch.tensor(y[idx], dtype=torch.float64)
        loss = (torch.nn.functional.softplus(score) - yy * score).mean()
        w.grad = None; head.grad = None; loss.backward(); hist.append(float(loss))
        with torch.no_grad():
            head -= H_LR * head.grad
            col = step % columns
            w[col] -= Q_LR * w.grad[col]
    with torch.no_grad():
        s = bw.features(torch.tensor(-np.pi * Xt), w) @ head[:-1] + head[-1]
    return float(((s >= 0).numpy().astype(int) == yt).mean()), hist[-1]

if __name__ == "__main__":
    out = {"xor": {}, "facts": []}
    accs = [notebook_train(8, 2, None, seed=s)[0] for s in range(10)]
    out["xor"]["n8_c2_k2_exactgrad_notebook_seed42+8"] = notebook_train(8, 2, None)[0]
    out["xor"]["n8_c2_k2_10seeds_mean"] = float(np.mean(accs)); out["xor"]["n8_c2_k2_10seeds_all"] = accs
    out["xor"]["n8_c2_k2_300steps"] = notebook_train(8, 2, None, steps=300)[0]
    out["xor"]["n12_c2_default_k"] = notebook_train(12, 2, None)[0]
    for n in (8, 12):
        for cols in (2, 4, n):
            for k in [kk for kk in (2, 4, 6) if 2 * kk <= n]:
                bw = T.BW(n, cols, k)
                nrbs = cols * n // 2
                out["facts"].append(dict(qubits=n, columns=cols, k=k, magic_blocks=k // 2, sector_dim=bw.D,
                    trainable_RBS=nrbs, trainable_Rz=cols * n, params_circuit=cols * 3 * n // 2,
                    params_head_linear=n + 1, params_head_corr=n * (n - 1) // 2 + 1,
                    depth_magic=4, depth_spread_nn=n - 1, depth_spread_butterfly=int(np.log2(n)) if n & (n - 1) == 0 else None,
                    depth_data=1, depth_trainable=2 * cols, depth_total_nn=4 + (n - 1) + 1 + 2 * cols,
                    spreading_givens_rbs=n * (n - 1) // 2))
    json.dump(out, open("results/bw/p2_xor_facts.json", "w"), indent=1)
    print(json.dumps(out["xor"], indent=1))
