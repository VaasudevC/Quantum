"""H3a: compile the k=32 pi/6 circuit for IBM Heron/Eagle (fake backends, real calibration snapshots) and an
IonQ-like all-to-all target. Reports gate counts, depth, 2q depth, SWAPs added, and estimated success probability.

  python -u hw_h3_compile.py   -> results/hw/h3_compile.json, results/hw/h3_compile.md
"""
import json
from pathlib import Path
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeTorino, FakeSherbrooke
import qiskit_vqc as Q

OUT = Path("results/hw"); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = range(5)
TWO_Q = {"cx", "cz", "ecr", "rxx", "swap", "rzz", "iswap"}


def counts(tc):
    ops = tc.count_ops()
    n2 = sum(v for k, v in ops.items() if k in TWO_Q)
    n1 = sum(v for k, v in ops.items() if k not in TWO_Q | {"measure", "barrier", "delay"})
    d2 = tc.depth(lambda ins: ins.operation.num_qubits == 2)
    return dict(ops={k: int(v) for k, v in ops.items()}, n1q=int(n1), n2q=int(n2), depth=int(tc.depth()), depth2q=int(d2),
                active_qubits=len({tc.find_bit(q).index for ins in tc.data for q in ins.qubits}))


def success(tc, backend):
    """Product of (1 - error) over every gate and measurement, from the backend's calibration target."""
    t = backend.target; logp = 0.0
    for ins in tc.data:
        name = ins.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(tc.find_bit(q).index for q in ins.qubits)
        try:
            err = t[name][qargs].error
        except Exception:
            err = None
        if err:
            logp += np.log(max(1e-12, 1 - err))
    return float(np.exp(logp))


def best_transpile(qc, seeds=SEEDS, **kw):
    """optimization_level=3 over 5 seeds; keep the fewest 2q gates (ties: smallest depth)."""
    best = None
    for s in seeds:
        tc = transpile(qc, optimization_level=3, seed_transpiler=s, **kw)
        c = counts(tc)
        if best is None or (c["n2q"], c["depth"]) < (best[1]["n2q"], best[1]["depth"]):
            best = (tc, c, s)
    return best


if __name__ == "__main__":
    row = Q.load_models()[(32, "pi/6")]
    th, w, b = Q.split_params(row)
    qc, x = Q.build_template(th, 32, measure=True)
    logical = counts(qc)
    print("logical:", logical, flush=True)
    res = {"logical": logical, "targets": {}}
    for B in (FakeFez, FakeTorino, FakeSherbrooke):
        be = B()
        tc, c, s = best_transpile(qc, backend=be)
        # SWAP count: same routing problem, but keep swap/cx as native so routing SWAPs stay visible
        _, cs, _ = best_transpile(qc, coupling_map=be.coupling_map, basis_gates=["cx", "swap", "rz", "ry", "rx", "sx", "x"])
        c.update(seed=s, success=success(tc, be), swaps_added=int(cs["ops"].get("swap", 0)),
                 n_qubits_device=be.num_qubits, native_2q=[g for g in be.operation_names if g in TWO_Q])
        res["targets"][be.name] = c
        print(be.name, {k: c[k] for k in ("n1q", "n2q", "depth", "depth2q", "swaps_added", "success", "active_qubits")}, flush=True)
    # IonQ-like: all-to-all, basis rz ry rx rxx, no coupling map
    tc, c, s = best_transpile(qc, basis_gates=["rz", "ry", "rx", "rxx"])
    # no calibration data: illustrative fidelities only (assumption, not measured)
    F1, F2, FM = 0.9995, 0.995, 0.995
    c.update(seed=s, swaps_added=0, success_assumed=float(F1 ** c["n1q"] * F2 ** c["n2q"] * FM ** 8),
             assumed_fidelities=dict(one_q=F1, two_q=F2, readout=FM))
    res["targets"]["ionq_like_all_to_all"] = c
    print("ionq-like", {k: c[k] for k in ("n1q", "n2q", "depth", "depth2q", "success_assumed")}, flush=True)
    json.dump(res, open(OUT / "h3_compile.json", "w"), indent=1)

    md = "| target | native 2q | 1q gates | 2q gates | depth | 2q depth | SWAPs added | est. success |\n|---|---|---|---|---|---|---|---|\n"
    md += f"| logical circuit | cx | {logical['n1q']} | {logical['n2q']} | {logical['depth']} | {logical['depth2q']} | – | – |\n"
    for name, c in res["targets"].items():
        succ = f"{c['success']:.3f}" if "success" in c else f"{c['success_assumed']:.3f} (assumed fidelities)"
        nat = ",".join(c.get("native_2q", ["rxx"]))
        md += f"| {name} | {nat} | {c['n1q']} | {c['n2q']} | {c['depth']} | {c['depth2q']} | {c['swaps_added']} | {succ} |\n"
    (OUT / "h3_compile.md").write_text(md); print(md)
