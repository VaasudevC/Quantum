"""EXTRA 2: sample-level head (paper Sec. VI.B).  The trained brick-wall circuit (n=8 final, per-fold parameters from stage C) is run to produce S measured
bitstrings per ECG (sampled from the exact output distribution; here S = 1000 and 100); a small permutation-invariant DeepSets model
  logit = rho( mean_i phi(bits_i) ),  phi: n -> 64 -> 64 (ReLU),  rho: 64 -> 32 -> 1
is trained on the samples (circuit frozen).  Because the samples live in a C(n,k)-state space, mean_i phi(bits_i) = sum_s freq_s phi(s) is computed exactly from
the empirical frequencies (identical to a per-sample DeepSets, much faster).  Baselines on the same samples: the original linear head (4 <n_j n_{j+1}> estimated from the samples),
and a linear head refit on the sampled features.   -> results/bw/extra2_samplehead.json
  C:/bwvenv/Scripts/python.exe extra2_samplehead.py [n]"""
import json, sys, time
import numpy as np, torch
import bw_torch as T
torch.set_num_threads(2)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rows = [json.loads(l) for l in open("results/bw/p3_stageC.jsonl")]
res = {}
for S in (1000, 100):
    per = []
    for fold in range(5):
        r = [r for r in rows if r["n"] == n and r["fold"] == fold and r.get("rank") == 0][0]
        d = np.load(f"C:/bwcache/l1_f{fold}.npz"); Z = d["Z"].astype(np.float64)[:, :n]; y = d["y"]; tr, va, te = d["tr"], d["va"], d["te"]
        m = T.Model(n, r["columns"], r["k"], r["scale"], head=r["head"], dtype=torch.complex64); m.load(r["params"])
        g = torch.Generator().manual_seed(fold)
        with torch.no_grad():
            Pm = []
            for i in range(0, len(Z), 2000):
                psi = m.bw.psi(m.theta(Z[i:i + 2000]), m.w); Pm.append((psi.real ** 2 + psi.imag ** 2))
            P = torch.cat(Pm).double(); P = P / P.sum(1, keepdim=True)
            idx = torch.multinomial(P, S, replacement=True, generator=g)                       # (N, S) sampled sector-state indices
            freq = torch.zeros_like(P).scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.double)) / S
        Madj = m.bw.M_adj.double(); occ = m.bw.occ.double(); h = m.h.detach().double(); b = float(m.b)
        f_s = 4 * freq @ Madj; yt = torch.tensor(y, dtype=torch.double)
        out = dict(fold=fold, exact_linear=T.auc(y[te], m.predict(Z[te])), sampled_linear=T.auc(y[te], (f_s @ h + b).numpy()[te]))
        w_, b_ = T.lr_fit(f_s.numpy()[tr] / (f_s.numpy()[tr].std(0) + 1e-9), y[tr]); sd = f_s.numpy()[tr].std(0) + 1e-9
        out["sampled_linear_refit"] = T.auc(y[te], (f_s.numpy() / sd) @ w_ + b_)[()] if False else T.auc(y[te], ((f_s.numpy() / sd) @ w_ + b_)[te])
        # DeepSets on frequencies
        torch.manual_seed(fold)
        phi = torch.nn.Sequential(torch.nn.Linear(n, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU()).double()
        rho = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).double()
        opt = torch.optim.Adam(list(phi.parameters()) + list(rho.parameters()), lr=2e-3)
        n_tr = len(tr); cw = torch.tensor([n_tr / (2 * (n_tr - y[tr].sum())), n_tr / (2 * y[tr].sum())], dtype=torch.double)
        def logit(ix):
            return rho(freq[ix] @ phi(occ)).squeeze(-1)
        best = (-1, None); rng = np.random.default_rng(fold)
        for ep in range(80):
            perm = rng.permutation(tr)
            for i in range(0, len(perm), 256):
                ib = torch.tensor(perm[i:i + 256]); s = logit(ib); yb = yt[ib]; wb = cw[yb.long()]
                loss = (wb * (torch.nn.functional.softplus(s) - yb * s)).mean(); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v = T.auc(y[va], logit(torch.tensor(va)).numpy())
                if v > best[0]:
                    best = (v, {k: x.clone() for k, x in list(phi.state_dict().items())}, {k: x.clone() for k, x in rho.state_dict().items()}, ep)
        phi.load_state_dict(best[1]); rho.load_state_dict(best[2])
        with torch.no_grad():
            out["samplehead_deepsets"] = T.auc(y[te], logit(torch.tensor(te)).numpy()); out["val_auc_deepsets"] = best[0]; out["best_epoch"] = best[3]
        per.append(out); print(S, out, flush=True)
    res[str(S)] = dict(per_fold=per, mean={k: float(np.mean([p[k] for p in per])) for k in per[0] if k not in ("fold", "best_epoch")})
json.dump(dict(n=n, results=res), open(f"results/bw/extra2_samplehead_n{n}.json", "w"), indent=1)
print(json.dumps({s: v["mean"] for s, v in res.items()}, indent=1))
