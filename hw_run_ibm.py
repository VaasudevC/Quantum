"""H4: run the k=32 pi/6 and k=32 pi models on REAL IBM hardware (or a local dry run), then analyse.

Budget: IBM open plan = ~10 QPU minutes / month. This script sends 2 jobs (one per model), each with
100 ECG PUBs + 2 readout-calibration PUBs at 1000 shots, i.e. about 102k shots per job, roughly 1-2 QPU minutes in total.

Setup (once, on a machine that can reach IBM):
    pip install -r requirements.txt qiskit qiskit-aer qiskit-ibm-runtime
    export QISKIT_IBM_TOKEN=...            # never commit it
    export QISKIT_IBM_INSTANCE=crn:...     # only for newer IBM Cloud accounts

Usage:
    python -u hw_run_ibm.py --local [--n 10]     # dry run of the full pipeline on FakeFez (no account, no network)
    python -u hw_run_ibm.py --submit             # real hardware: prints the plan, asks for confirmation, submits
    python -u hw_run_ibm.py --submit --yes       # same, without the prompt
    python -u hw_run_ibm.py --retrieve           # re-fetch results for the saved job ids (if the session dropped)
    python -u hw_run_ibm.py --analyze [--tag local]   # AUC: hardware raw / mitigated / Aer noisy / noiseless

Everything is saved to results/hw/ibm_<tag>/ BEFORE any analysis: job ids, backend name, calibration timestamp,
transpiled-circuit stats, ECG indices and the raw counts of every PUB.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qiskit_vqc as Q
import Vqc_fast8 as V

MODELS = [(32, "pi/6"), (32, "pi")]
N_ECG, SHOTS, CAL_SHOTS = 100, 1000, 1000


def outdir(tag):
    d = Path("results/hw") / f"ibm_{tag}"; d.mkdir(parents=True, exist_ok=True); return d


def get_backend(local):
    if local:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        return FakeFez(), None
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = os.environ.get("QISKIT_IBM_TOKEN")
    if not tok:
        sys.exit("QISKIT_IBM_TOKEN is not set")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok,
                                   instance=os.environ.get("QISKIT_IBM_INSTANCE") or None)
    be = service.least_busy(operational=True, simulator=False, min_num_qubits=8)
    return be, service


def calibration_time(be):
    try:
        p = be.properties()
        return str(p.last_update_date) if p is not None else None
    except Exception:
        return None


def build_pubs(be, key, n_ecg=None):
    """Transpile the model template once with the backend's preset pass manager; one PUB per ECG + 2 readout cals."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    row = Q.load_models()[key]
    A, y, tr, va, te = Q.model_data(row); th, w, b = Q.split_params(row)
    sub = te[Q.stratified_subset(y[te], n_ecg or N_ECG, seed=0)]
    qc, x = Q.build_template(th, row["k"])
    pm = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=0)
    tc = pm.run(qc)
    phys = tc.layout.final_index_layout()
    pubs = [(tc, A[i, :row["k"]]) for i in sub]
    pm0 = generate_preset_pass_manager(optimization_level=0, backend=be, initial_layout=phys)
    for flip in (0, 1):                                   # readout calibration on the same physical qubits
        c = QuantumCircuit(Q.NQ, Q.NQ)
        if flip:
            c.x(range(Q.NQ))
        c.measure(range(Q.NQ), range(Q.NQ))
        pubs.append((pm0.run(c),))
    ops = tc.count_ops()
    stats = dict(n2q=int(sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))), depth=int(tc.depth()),
                 ops={k: int(v) for k, v in ops.items()}, physical_qubits=[int(q) for q in phys])
    return pubs, sub, y[sub], stats


def make_sampler(mode, local):
    from qiskit_ibm_runtime import SamplerV2
    s = SamplerV2(mode=mode)
    if not local:
        try:
            s.options.dynamical_decoupling.enable = True
            s.options.dynamical_decoupling.sequence_type = "XpXm"
        except Exception as e:
            print("DD not available:", e)
        try:
            s.options.twirling.enable_gates = True
            s.options.twirling.enable_measure = True
            s.options.twirling.num_randomizations = "auto"
        except Exception as e:
            print("twirling not available:", e)
    return s


def save_counts(d, key, result):
    counts = [pub.data.c.get_counts() if hasattr(pub.data, "c") else next(iter(pub.data.values())).get_counts()
              for pub in result]
    json.dump(counts, open(d / f"counts_k{key[0]}_{key[1].replace('/', '_')}.json", "w"))
    return counts


def submit(local, yes, n_ecg=N_ECG):
    tag = "local" if local else "hw"
    d = outdir(tag)
    be, service = get_backend(local)
    meta = dict(backend=be.name, local=local, submitted_utc=datetime.now(timezone.utc).isoformat(),
                calibration_timestamp=calibration_time(be), shots=SHOTS, n_ecg=n_ecg, jobs={}, circuits={})
    plan = []
    for key in MODELS:
        pubs, sub, ysub, stats = build_pubs(be, key, n_ecg)
        plan.append((key, pubs)); meta["circuits"][f"k{key[0]}_{key[1]}"] = stats
        np.savez(d / f"ecgs_k{key[0]}_{key[1].replace('/', '_')}.npz", idx=sub, y=ysub)
        print(f"k={key[0]} {key[1]}: {len(pubs)} PUBs, {stats['n2q']} 2q gates, depth {stats['depth']}, "
              f"qubits {stats['physical_qubits']}", flush=True)
    total = sum(len(p) for _, p in plan) * SHOTS
    print(f"backend {be.name} (calibrated {meta['calibration_timestamp']}); total shots {total:,} "
          f"(open plan budget ~10 QPU min/month; expect ~1-2 min)", flush=True)
    if not local and not yes and input("Submit to real hardware? [y/N] ").strip().lower() != "y":
        sys.exit("not submitted")
    if local:
        mode_ctx = None
        sampler = make_sampler(be, local=True)
    else:
        from qiskit_ibm_runtime import Batch
        mode_ctx = Batch(backend=be)
        sampler = make_sampler(mode_ctx, local=False)
    jobs = {}
    for key, pubs in plan:
        job = sampler.run(pubs, shots=SHOTS)
        jobs[key] = job; meta["jobs"][f"k{key[0]}_{key[1]}"] = job.job_id()
        print(f"submitted k={key[0]} {key[1]}: job {job.job_id()}", flush=True)
    json.dump(meta, open(d / "meta.json", "w"), indent=1)          # job ids saved before waiting
    for key, job in jobs.items():
        res = job.result(); save_counts(d, key, res)
        print(f"saved counts k={key[0]} {key[1]}", flush=True)
    if mode_ctx is not None:
        mode_ctx.close()
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    json.dump(meta, open(d / "meta.json", "w"), indent=1)
    aer_noisy(be, tag, n_ecg=n_ecg)
    return tag


def retrieve():
    from qiskit_ibm_runtime import QiskitRuntimeService
    d = outdir("hw"); meta = json.load(open(d / "meta.json"))
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"],
                                   instance=os.environ.get("QISKIT_IBM_INSTANCE") or None)
    for name, jid in meta["jobs"].items():
        k, s = name[1:].split("_", 1)
        save_counts(d, (int(k), s), service.job(jid).result()); print("saved", name, jid)
    aer_noisy(service.backend(meta["backend"]), "hw", n_ecg=meta["n_ecg"])


def aer_noisy(be, tag, shots=SHOTS, n_ecg=N_ECG):
    """Aer simulation with this backend's own noise model, same circuits / ECGs, for comparison."""
    from qiskit_aer import AerSimulator
    d = outdir(tag)
    sim = AerSimulator.from_backend(be, method="density_matrix")
    for key in MODELS:
        pubs, sub, ysub, stats = build_pubs(be, key, n_ecg)
        circs = [p[0].assign_parameters(p[1]) if len(p) > 1 else p[0] for p in pubs]
        counts = sim.run(circs, shots=shots, seed_simulator=1).result().get_counts()
        json.dump(counts, open(d / f"aer_counts_k{key[0]}_{key[1].replace('/', '_')}.json", "w"))
        print(f"Aer noisy ({be.name}) k={key[0]} {key[1]} done ({stats['n2q']} 2q gates)", flush=True)


def analyze(tag):
    from sklearn.metrics import roc_auc_score
    from hw_h3_noisy import mitigate
    d = outdir(tag); meta = json.load(open(d / "meta.json")); out = {}
    for key in MODELS:
        nm = f"k{key[0]}_{key[1].replace('/', '_')}"
        row = Q.load_models()[key]; A, y, tr, va, te = Q.model_data(row); th, w, b = Q.split_params(row)
        e = np.load(d / f"ecgs_{nm}.npz"); sub, ys = e["idx"], e["y"]
        psi = V.forward(A[sub], th); P = psi.real ** 2 + psi.imag ** 2; z_ex = P @ V._ZSIGN.T
        res = {"noiseless": z_ex}
        for src, f in (("hardware", f"counts_{nm}.json"), ("aer_noisy", f"aer_counts_{nm}.json")):
            if not (d / f).exists():
                continue
            cnt = json.load(open(d / f)); z = np.array([Q.z_from_counts(c) for c in cnt])
            zr, c0, c1 = z[:-2], z[-2], z[-1]
            e0, e1 = (1 - c0) / 2, (1 + c1) / 2
            res[f"{src}_raw"] = zr; res[f"{src}_mitigated"] = mitigate(zr, e0, e1)
        pmi = lambda z: 1 / (1 + np.exp(-Q.logit_from_z(z, w, b)))
        out[nm] = {n: dict(auc=float(roc_auc_score(ys, Q.logit_from_z(z, w, b))),
                           corr_P_vs_noiseless=float(np.corrcoef(pmi(z), pmi(z_ex))[0, 1])) for n, z in res.items()}
        print(nm, {n: round(v["auc"], 4) for n, v in out[nm].items()}, flush=True)
    out["meta"] = meta
    json.dump(out, open(d / "analysis.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--local", action="store_true"); g.add_argument("--submit", action="store_true")
    g.add_argument("--retrieve", action="store_true"); g.add_argument("--analyze", action="store_true")
    ap.add_argument("--yes", action="store_true"); ap.add_argument("--n", type=int, default=None, help="ECGs per model (default 100; use ~10 for --local)"); ap.add_argument("--tag", default="hw")
    a = ap.parse_args()
    if a.local:
        analyze(submit(local=True, yes=True, n_ecg=a.n or 10))
    elif a.submit:
        analyze(submit(local=False, yes=a.yes, n_ecg=a.n or N_ECG))
    elif a.retrieve:
        retrieve(); analyze("hw")
    else:
        analyze(a.tag)
