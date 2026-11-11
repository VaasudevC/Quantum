"""P6a: transpile the brick-wall circuits (generic random angles so no gate is optimised away) for several targets.
Targets: IBM Heron FakeFez, FakeTorino (opt level 3, 5 seeds, best = fewest 2q gates then smallest depth), IQM-like 4x5 square lattice,
a plain ring, all-to-all.  Reports 1q, 2q, depth, 2q depth, SWAPs (routing-only pass with abstract gate set), estimated success.
  C:/bwvenv/Scripts/python.exe p6_compile.py [--columns 2 4]
"""
import json, sys, argparse
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XXPlusYYGate
from qiskit.transpiler import CouplingMap
import hw_bw as H

NATIVE = ["rz", "sx", "x", "cz"]
ASSUMED = dict(e1=3e-4, e2=5e-3, ro=1.5e-2)           # for targets without calibration data


def targets():
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeTorino
    n_ = {}
    n_["IBM Heron r2 (FakeFez)"] = dict(backend=FakeFez())
    n_["IBM Heron r1 (FakeTorino)"] = dict(backend=FakeTorino())
    n_["IQM-like 4x5 square lattice (assumed errors)"] = dict(cm=CouplingMap.from_grid(4, 5), basis=NATIVE)
    n_["ring of n (assumed errors)"] = dict(cm="ring", basis=NATIVE)
    n_["all-to-all (assumed errors)"] = dict(cm=None, basis=NATIVE)
    return n_


def est_success(tc, backend=None):
    p = 1.0
    if backend is not None:
        tg = backend.target
        for inst in tc.data:
            nm = inst.operation.name; qs = tuple(tc.find_bit(q).index for q in inst.qubits)
            if nm in ("measure",):
                e = tg["measure"][qs].error if "measure" in tg and qs in tg["measure"] and tg["measure"][qs] else 0.0
            elif nm in tg.operation_names and qs in tg[nm] and tg[nm][qs] is not None and tg[nm][qs].error is not None:
                e = tg[nm][qs].error
            else:
                continue
            p *= (1 - e)
    else:
        for inst in tc.data:
            nm = inst.operation.name
            if nm == "cz": p *= 1 - ASSUMED["e2"]
            elif nm in ("sx", "x"): p *= 1 - ASSUMED["e1"]
            elif nm == "measure": p *= 1 - ASSUMED["ro"]
    return p


def stats(tc):
    ops = tc.count_ops()
    n2 = sum(v for k, v in ops.items() if k in ("cz", "cx", "ecr", "rzz", "swap"))
    n1 = sum(v for k, v in ops.items() if k not in ("cz", "cx", "ecr", "rzz", "swap", "measure", "barrier", "rz", "delay"))
    d2 = tc.depth(lambda i: i.operation.num_qubits == 2)
    return dict(oneq_nonvirtual=int(n1), rz=int(ops.get("rz", 0)), twoq=int(n2), depth=int(tc.depth()), twoq_depth=int(d2))


def compile_one(qc, tgt, n, seeds=5):
    kw = {}
    be = tgt.get("backend"); cm = tgt.get("cm"); basis = tgt.get("basis")
    if cm == "ring":
        cm = CouplingMap.from_ring(n)
    if be is not None:
        kw = dict(backend=be)
    else:
        kw = dict(coupling_map=cm, basis_gates=basis)
    best = None
    for sd in range(seeds):
        tc = transpile(qc, optimization_level=3, seed_transpiler=sd, **kw)
        st = stats(tc); key = (st["twoq"], st["twoq_depth"], st["depth"])
        if best is None or key < best[0]:
            best = (key, tc, st)
    _, tc, st = best
    # SWAP count: route with an abstract gate set that keeps swaps and RBS intact
    sw = None
    if be is not None or cm is not None:
        cmap = be.coupling_map if be is not None else cm
        absb = ["rz", "sx", "x", "h", "z", "cx", "swap", "xx_plus_yy"]
        sw_best = None
        for sd in range(seeds):
            r = transpile(qc.remove_final_measurements(inplace=False), coupling_map=cmap, basis_gates=absb, optimization_level=3, seed_transpiler=sd)
            c = r.count_ops().get("swap", 0)
            sw_best = c if sw_best is None else min(sw_best, c)
        sw = int(sw_best)
    else:
        sw = 0
    st["swaps_routing_only"] = sw
    st["est_success"] = est_success(tc, be)
    st["active_qubits"] = int(len({tc.find_bit(q).index for inst in tc.data for q in inst.qubits}))
    return st, tc


def rbs_cost():
    qc = QuantumCircuit(2); qc.append(XXPlusYYGate(0.7, np.pi / 2), [0, 1])
    tc = transpile(qc, basis_gates=NATIVE, optimization_level=3)
    return dict(ops={k: int(v) for k, v in tc.count_ops().items()}, depth=tc.depth())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--columns", nargs="*", type=int, default=[2]); ap.add_argument("--out", default="results/bw/p6_compile.json")
    a = ap.parse_args()
    res = dict(rbs_two_qubit_cost=rbs_cost(), circuits={})
    rng = np.random.default_rng(0)
    T = targets()
    for n, k in ((8, 2), (8, 4), (12, 2), (12, 4), (12, 6)):
        for cols in a.columns:
            qc, th, wv = H.build(n, cols, k)
            bc = H.bind(qc, th, wv, rng.uniform(-1, 1, n), rng.normal(0, 0.8, (cols, 3 * n // 2)))
            logical = H.stats if False else None
            lg = transpile(bc, basis_gates=NATIVE, optimization_level=3)       # all-to-all logical cost
            ent = dict(logical=stats(lg))
            for name, tgt in T.items():
                st, _ = compile_one(bc, tgt, n)
                ent[name] = st
                print(n, k, cols, name, st, flush=True)
            res["circuits"][f"n{n}_k{k}_c{cols}"] = ent
    json.dump(res, open(a.out, "w"), indent=1)
