"""P6b: noiseless / shots / FakeFez-noisy / noisy + particle-number post-selection for the final brick-wall models (fold-0 test ECGs).
  C:/bwvenv/Scripts/python.exe p6_noise.py --n 8 --ecgs 300 --shots 2000
  C:/bwvenv/Scripts/python.exe p6_noise.py --n 12 --ecgs 60  --shots 2000       (12-qubit density matrices are ~100x slower: fewer ECGs)
Compilation for the noisy run: routed on the best n-qubit LINE of FakeFez (only n qubits active -> density-matrix simulation feasible)."""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
import networkx as nx
from qiskit import transpile
from qiskit.transpiler import CouplingMap
import bw_torch as T, hw_bw as H
torch.set_num_threads(1)


def best_line(backend, n):
    t = backend.target; g2 = "cz"; G = nx.Graph()
    for a, b in backend.coupling_map.get_edges():
        p = t[g2].get((a, b))
        if p is not None and p.error is not None:
            w = -np.log(1 - p.error)
            if not G.has_edge(a, b) or G[a][b]["w"] > w: G.add_edge(a, b, w=w)
    ro = {q: -np.log(1 - (t["measure"][(q,)].error or 0)) for q in range(backend.num_qubits)}
    best = [np.inf, None]
    def dfs(path, cost):
        if cost >= best[0]: return
        if len(path) == n:
            c = cost + sum(ro[q] for q in path)
            if c < best[0]: best[:] = [c, list(path)]
            return
        for m in G.neighbors(path[-1]):
            if m not in path: dfs(path + [m], cost + G[path[-1]][m]["w"])
    for s in G.nodes: dfs([s], 0.0)
    return best[1]


def compile_line(qc, backend, path, seeds=range(5)):
    sub = CouplingMap([[path[i], path[i + 1]] for i in range(len(path) - 1)] + [[path[i + 1], path[i]] for i in range(len(path) - 1)])
    best = None
    for s in seeds:
        tc = transpile(qc, backend=backend, initial_layout=path, coupling_map=sub, optimization_level=3, seed_transpiler=s)
        key = (tc.count_ops().get("cz", 0), tc.depth())
        if best is None or key < best[0]: best = (key, tc)
    return best[1]


def load_final(n, fold=0):
    rows = [json.loads(l) for l in open("results/bw/p3_stageC.jsonl")]
    r = [r for r in rows if r["n"] == n and r["fold"] == fold and r.get("rank") == 0][0]
    return r


def stratified(y, n, seed=0):
    rng = np.random.default_rng(seed); pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    npos = int(round(n * len(pos) / len(y)))
    return np.sort(np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, n - npos, replace=False)]))


def main():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from qiskit_aer import AerSimulator
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--ecgs", type=int, default=300)
    ap.add_argument("--shots", type=int, default=2000); a = ap.parse_args()
    n = a.n; r = load_final(n); cfg = {k: r[k] for k in ("columns", "k", "scale", "head")}
    d = np.load("C:/bwcache/l1_f0.npz"); Z = d["Z"].astype(np.float64)[:, :n]; y = d["y"]; te = d["te"]
    sub = te[stratified(y[te], a.ecgs)]; ys = y[sub]
    m = T.Model(n, cfg["columns"], cfg["k"], cfg["scale"], head=cfg["head"]); m.load(r["params"])
    h, b = m.h.detach().numpy().astype(float), float(m.b)
    with torch.no_grad():
        f_ex = m.feats(Z[sub]).numpy().astype(float)
    auc = lambda f: T.auc(ys, f @ h + b)
    out = dict(n=n, cfg=cfg, n_ecgs=len(sub), shots=a.shots, auc_noiseless_exact=auc(f_ex))
    qc, th, wv = H.build(n, cfg["columns"], cfg["k"]); w = np.asarray(r["params"]["w"])
    theta = -cfg["scale"] * np.tanh(Z[sub])                                     # RZ angle = -phi
    circs = [H.bind(qc, th, wv, theta[i], w) for i in range(len(sub))]
    # noiseless shots (statevector sampling) and shot sweep
    sv = AerSimulator(method="statevector"); t0 = time.time()
    sweep = {}
    for sh in (100, 300, 1000, 4000):
        tcs = transpile(circs, basis_gates=["rz", "sx", "x", "cz", "h", "cx", "z"], optimization_level=0)
        cnt = sv.run(tcs, shots=sh, seed_simulator=sh).result().get_counts()
        f = np.array([H.features_from_counts(c, n)[0] for c in cnt]); sweep[sh] = auc(f)
        if sh == 1000 or sh == a.shots: pass
    out["shot_sweep_noiseless_auc"] = sweep
    tcs = transpile(circs, basis_gates=["rz", "sx", "x", "cz", "h", "cx", "z"], optimization_level=0)
    cnt = sv.run(tcs, shots=a.shots, seed_simulator=7).result().get_counts()
    out["auc_noiseless_shots"] = auc(np.array([H.features_from_counts(c, n)[0] for c in cnt]))
    print("noiseless done", out, time.time() - t0, flush=True)
    # noisy
    be = FakeFez(); path = best_line(be, n); tc0 = compile_line(circs[0], be, path)
    out["compile_line_path"] = path; out["compiled_cz"] = int(tc0.count_ops().get("cz", 0)); out["compiled_depth"] = int(tc0.depth())
    qct = compile_line(qc, be, path)                                              # compile the parametrised template once
    tmpl_params = list(qct.parameters)
    nm = AerSimulator.from_backend(be, method="density_matrix")
    names = [q.name for q in th] + [q.name for q in wv]
    tcs = []
    for i in range(len(sub)):
        vals = dict(zip(names, list(theta[i]) + list(w.ravel())))
        tcs.append(qct.assign_parameters({p: float(vals[p.name]) for p in tmpl_params}))
    t0 = time.time(); fr, fp, kept = [], [], []
    CH = 10
    for i in range(0, len(tcs), CH):
        cnt = nm.run(tcs[i:i + CH], shots=a.shots, seed_simulator=11 + i).result().get_counts()
        for c in cnt:
            f0, _ = H.features_from_counts(c, n); f1, kf = H.features_from_counts(c, n, cfg["k"])
            fr.append(f0); fp.append(f1); kept.append(kf)
        print(f"  noisy {i + CH}/{len(tcs)} {time.time() - t0:.0f}s", flush=True)
    out["auc_noisy_raw"] = auc(np.array(fr)); out["auc_noisy_postselected"] = auc(np.array(fp)); out["postselect_kept_fraction_mean"] = float(np.mean(kept))
    out["noisy_secs"] = round(time.time() - t0, 1)
    Path("results/bw").mkdir(exist_ok=True)
    json.dump(out, open(f"results/bw/p6_noise_n{n}.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
