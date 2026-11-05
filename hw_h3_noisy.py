"""H3b: noisy simulation of the 6 models with the FakeFez (IBM Heron r2) noise model + readout-error mitigation.

Compilation for the noisy run: the circuit is routed on the best 8-qubit LINE of FakeFez (lowest summed CZ + readout
error), so only 8 qubits are active and the density-matrix simulation takes ~0.6 s per circuit. The globally best compile
(H3a, 12 active qubits, 80 CZ) is ~12x slower to simulate; `--best12` runs it on a subset as a check.

Readout mitigation: per-qubit confusion matrix measured on the same physical qubits (two calibration circuits,
all-|0> and all-|1>, 20k shots each); for the readout we only need <Z_j>, which corrects exactly as
    <Z_j>_mit = (<Z_j>_raw - (e1_j - e0_j)) / (1 - e0_j - e1_j),   e0 = P(1|0), e1 = P(0|1).

  python -u hw_h3_noisy.py                 -> results/hw/h3_noisy.jsonl (+ per-ECG z in results/hw/h3_z_*.npz)
  python -u hw_h3_noisy.py --best12 --n 50 --models "32:pi/6,32:pi"
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
import qiskit_vqc as Q
import Vqc_fast8 as V

OUT = Path("results/hw"); OUT.mkdir(parents=True, exist_ok=True)


def best_line(backend, n=8):
    """Simple path of n physical qubits minimising sum(-log(1-err)) over its 2q gates and readouts."""
    t = backend.target
    g2 = next(g for g in ("cz", "ecr", "cx") if g in t.operation_names)
    G = nx.Graph()
    for a, b in backend.coupling_map.get_edges():
        props = t[g2].get((a, b))
        if props is not None and props.error is not None:
            w = -np.log(1 - props.error)
            if not G.has_edge(a, b) or G[a][b]["w"] > w:
                G.add_edge(a, b, w=w)
    ro = {q: -np.log(1 - (t["measure"][(q,)].error or 0)) for q in range(backend.num_qubits)}
    best = [np.inf, None]

    def dfs(path, cost):
        if cost >= best[0]:
            return
        if len(path) == n:
            c = cost + sum(ro[q] for q in path)
            if c < best[0]:
                best[:] = [c, list(path)]
            return
        for m in G.neighbors(path[-1]):
            if m not in path:
                dfs(path + [m], cost + G[path[-1]][m]["w"])
    for s in G.nodes:
        dfs([s], 0.0)
    return best[1]


def compile_line(qc, backend, path, seeds=range(5)):
    sub = CouplingMap([[path[i], path[i + 1]] for i in range(len(path) - 1)] +
                      [[path[i + 1], path[i]] for i in range(len(path) - 1)])
    best = None
    for s in seeds:
        tc = transpile(qc, backend=backend, initial_layout=path, coupling_map=sub, optimization_level=3, seed_transpiler=s)
        key = (tc.count_ops().get("cz", 0) + tc.count_ops().get("ecr", 0), tc.depth())
        if best is None or key < best[0]:
            best = (key, tc)
    return best[1]


def readout_calibration(run_fn, tc, backend, shots=20000):
    """e0[j] = P(read 1 | prep 0), e1[j] = P(read 0 | prep 1) for the physical qubit holding virtual qubit j at the end."""
    phys = tc.layout.final_index_layout()
    cals = []
    for flip in (0, 1):
        c = QuantumCircuit(Q.NQ, Q.NQ)
        if flip:
            c.x(range(Q.NQ))
        c.measure(range(Q.NQ), range(Q.NQ))
        cals.append(transpile(c, backend=backend, initial_layout=phys, optimization_level=0))
    z0, z1 = [Q.z_from_counts(cn) for cn in run_fn(cals, shots)]
    e0 = (1 - z0) / 2          # prep 0: <Z> = 1 - 2 e0
    e1 = (1 + z1) / 2          # prep 1: <Z> = -1 + 2 e1
    return e0, e1, phys


def mitigate(z, e0, e1):
    return (z - (e1 - e0)) / (1 - e0 - e1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--models", default="all"); ap.add_argument("--best12", action="store_true")
    ap.add_argument("--backend", default="FakeFez")
    a = ap.parse_args()
    from qiskit_aer import AerSimulator
    import qiskit_ibm_runtime.fake_provider as FP
    be = getattr(FP, a.backend)()
    sim = AerSimulator.from_backend(be, method="density_matrix")
    run_fn = lambda circs, shots: sim.run(circs, shots=shots, seed_simulator=1).result().get_counts()
    models = Q.load_models()
    keys = Q.MODEL_KEYS if a.models == "all" else [(int(m.split(":")[0]), m.split(":")[1]) for m in a.models.split(",")]
    tag = "best12" if a.best12 else "line8"
    out = OUT / f"h3_noisy_{tag}.jsonl"
    done = {(json.loads(l)["k"], json.loads(l)["scale"], json.loads(l)["n"]) for l in open(out)} if out.exists() else set()
    path = None if a.best12 else best_line(be)
    print(f"backend={be.name} compile={tag} path={path} n={a.n} shots={a.shots}", flush=True)
    rng = np.random.default_rng(0)
    for key in keys:
        if (key[0], key[1], a.n) in done:
            print("skip", key); continue
        row = models[key]; A, y, tr, va, te = Q.model_data(row); th, w, b = Q.split_params(row)
        sub = te[Q.stratified_subset(y[te], a.n, seed=0)]          # same ECGs for every model (fold-0 test rows)
        qc, x = Q.build_template(th, row["k"])
        if a.best12:
            from hw_h3_compile import best_transpile
            tc = best_transpile(qc, backend=be)[0]
        else:
            tc = compile_line(qc, be, path)
        e0, e1, phys = readout_calibration(run_fn, tc, be)
        t0 = time.time()
        circs = [tc.assign_parameters(A[i, :row["k"]]) for i in sub]
        z_raw = np.array([Q.z_from_counts(c) for c in run_fn(circs, a.shots)])
        z_mit = mitigate(z_raw, e0, e1)
        psi = V.forward(A[sub], th); P = psi.real ** 2 + psi.imag ** 2; P /= P.sum(1, keepdims=True)
        z_ex = P @ V._ZSIGN.T
        z_shot = rng.multinomial(a.shots, P) @ V._ZSIGN.T / a.shots          # ideal device, same shots
        auc = {n: float(roc_auc_score(y[sub], Q.logit_from_z(z, w, b)))
               for n, z in (("noiseless", z_ex), ("shots_only", z_shot), ("noisy_raw", z_raw), ("noisy_mitigated", z_mit))}
        pc = lambda z: float(np.corrcoef(1 / (1 + np.exp(-Q.logit_from_z(z, w, b))), 1 / (1 + np.exp(-Q.logit_from_z(z_ex, w, b))))[0, 1])
        ops = tc.count_ops()
        rec = dict(k=key[0], scale=key[1], n=a.n, shots=a.shots, backend=be.name, compile=tag, physical_qubits=phys,
                   n2q=int(ops.get("cz", 0) + ops.get("ecr", 0)), depth=int(tc.depth()), **auc,
                   corr_P_raw=pc(z_raw), corr_P_mit=pc(z_mit),
                   z_shrink_raw=float(np.median(np.abs(z_raw).mean(1) / np.abs(z_ex).mean(1))),
                   readout_e0=e0.tolist(), readout_e1=e1.tolist(), secs=round(time.time() - t0, 1))
        np.savez(OUT / f"h3_z_{tag}_k{key[0]}_{key[1].replace('/', '_')}.npz", idx=sub, y=y[sub], z_exact=z_ex, z_raw=z_raw, z_mit=z_mit)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"k={key[0]:2d} {key[1]:5s} noiseless={auc['noiseless']:.4f} shots={auc['shots_only']:.4f} "
              f"raw={auc['noisy_raw']:.4f} mit={auc['noisy_mitigated']:.4f}  corrP raw={rec['corr_P_raw']:.3f} "
              f"|z| shrink={rec['z_shrink_raw']:.2f}  n2q={rec['n2q']}  {rec['secs']}s", flush=True)


if __name__ == "__main__":
    main()
