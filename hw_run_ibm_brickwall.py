"""TEST-ONLY run of trained models on IBM hardware (no training on hardware).  Models: best brick-wall n=8, best brick-wall n=12,
our pi/6 angle VQC (k=32, fold 0).  Same stratified fold-0 test ECGs for all three.  One SamplerV2 job per model inside one Batch.

Setup:   set the env var QISKIT_IBM_TOKEN (never written to disk or logs); optional QISKIT_IBM_INSTANCE=crn:...
Usage (C:/bwvenv python):
    python -u hw_run_ibm_brickwall.py --local --n 4 --shots 200 --models bw8,vqc32      # dry run on FakeFez (no account, no network)
    python -u hw_run_ibm_brickwall.py --submit --n 100 --shots 1000                      # real hardware; prints the plan, asks to confirm
    python -u hw_run_ibm_brickwall.py --submit --n 100 --shots 1000 --yes
    python -u hw_run_ibm_brickwall.py --retrieve                                          # re-fetch saved job ids, then analyse
    python -u hw_run_ibm_brickwall.py --analyze [--tag hw]
Raw counts and job ids are written to results/hw/ibm_bw_<tag>/ BEFORE any analysis.  The script refuses to submit if the estimated QPU time
exceeds --max-minutes (default 8) of the 10 free minutes.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
import bw_torch as T, hw_bw as H, qiskit_vqc as Q, Vqc_fast8 as V
torch.set_num_threads(1)
PER_SHOT_S = 0.4e-3            # conservative QPU seconds per shot (rep delay ~250 us + circuit + readout, with margin)
ALL = ["bw8", "bw12", "vqc32"]


def outdir(tag):
    d = Path("results/hw") / f"ibm_bw_{tag}"; d.mkdir(parents=True, exist_ok=True); return d


def stratified(y, n, seed=0):
    rng = np.random.default_rng(seed); pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    npos = int(round(n * len(pos) / len(y)))
    return np.sort(np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, n - npos, replace=False)]))


def model_inputs(name, n_ecg):
    """returns (template circuit, {param name: values array (n_ecg,)}, info dict, labels, positions in the fold-0 test set)"""
    hw = np.load("C:/bwcache/hw_inputs.npz"); y_te = hw["y_te"]; pos = stratified(y_te, n_ecg)
    if name.startswith("bw"):
        n = int(name[2:])
        rows = [json.loads(l) for l in open("results/bw/p3_stageC.jsonl")]
        r = [r for r in rows if r["n"] == n and r["fold"] == 0 and r.get("rank") == 0][0]
        d = np.load("C:/bwcache/l1_f0.npz"); Z = d["Z"].astype(np.float64)[:, :n]; te = d["te"]
        assert (te == hw["te"]).all()
        qc, th, wv = H.build(n, r["columns"], r["k"]); w = np.asarray(r["params"]["w"]).ravel()
        theta = -r["scale"] * np.tanh(Z[te][pos])
        vals = {th[j].name: theta[:, j] for j in range(n)}
        vals.update({wv[i].name: np.full(len(pos), w[i]) for i in range(len(w))})
        m = T.Model(n, r["columns"], r["k"], r["scale"], head=r["head"]); m.load(r["params"])
        with torch.no_grad():
            f_ex = m.feats(Z[te][pos]).numpy()
        h = m.h.detach().numpy(); b = float(m.b)
        info = dict(kind="bw", n=n, k=r["k"], h=h.tolist(), b=b, noiseless_score=(f_ex @ h + b).tolist())
        return qc, vals, info, y_te[pos], pos
    row = Q.load_models()[(32, "pi/6")]; th_, w_, b_ = Q.split_params(row)
    qc, x = Q.build_template(th_, 32)
    A = hw["A_vqc_te"][pos]
    vals = {x[j].name: A[:, j] for j in range(32)}
    z_ex = V.expectations(V.forward(A, th_), "z")
    return qc, vals, dict(kind="vqc", w=w_.tolist(), b=float(b_), noiseless_score=(z_ex @ w_ + b_).tolist()), y_te[pos], pos


def get_backend(local):
    if local:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        return FakeFez(), None
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = os.environ.get("QISKIT_IBM_TOKEN")
    if not tok:
        sys.exit("QISKIT_IBM_TOKEN is not set")
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=tok, instance=os.environ.get("QISKIT_IBM_INSTANCE") or None)
    return svc.least_busy(operational=True, simulator=False, min_num_qubits=16), svc


def build_pubs(be, models, n_ecg):
    from qiskit.transpiler import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=0)
    plan = {}
    for name in models:
        qc, vals, info, y, pos = model_inputs(name, n_ecg)
        tc = pm.run(qc)
        bind = {p: np.asarray(vals[p.name], dtype=float) for p in tc.parameters}
        ops = tc.count_ops()
        info.update(n2q=int(ops.get("cz", 0)), depth=int(tc.depth()))
        plan[name] = dict(pub=(tc, bind), info=info, y=y, pos=pos)
    return plan


def estimate_minutes(plan, shots):
    return sum(len(p["y"]) * shots * PER_SHOT_S for p in plan.values()) / 60.0


def save_counts(d, name, result):
    data = result[0].data
    ba = getattr(data, list(data.keys())[0])
    counts = [ba[i].get_counts() for i in range(ba.shape[0])] if ba.shape else [ba.get_counts()]
    json.dump(counts, open(d / f"counts_{name}.json", "w"))


def run(local, yes, n_ecg, shots, models, max_min, tag):
    d = outdir(tag); be, svc = get_backend(local)
    plan = build_pubs(be, models, n_ecg)
    est = estimate_minutes(plan, shots)
    print(f"backend {be.name}; {len(models)} jobs; ECGs/model {n_ecg}; shots {shots}; ESTIMATED QPU time {est:.2f} min (limit {max_min})", flush=True)
    for k, p in plan.items():
        print(f"  {k}: {p['info']['n2q']} CZ, depth {p['info']['depth']}", flush=True)
    if est > max_min:
        sys.exit(f"REFUSING: estimated QPU time {est:.1f} min > {max_min} min")
    if not local and not yes and input("Submit to real hardware? [y/N] ").strip().lower() != "y":
        sys.exit("not submitted")
    from qiskit_ibm_runtime import SamplerV2, Batch
    ctx = None if local else Batch(backend=be)
    sampler = SamplerV2(mode=be if local else ctx)
    if not local:
        try:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XpXm"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
        except Exception as e:
            print("options:", e)
    meta = dict(backend=be.name, local=local, submitted_utc=datetime.now(timezone.utc).isoformat(), shots=shots, n_ecg=n_ecg, est_qpu_min=est, jobs={},
                info={k: {kk: vv for kk, vv in p["info"].items() if kk != "noiseless_score"} for k, p in plan.items()})
    for k, p in plan.items():                                              # ECG indices + labels saved before submission
        np.savez(d / f"ecgs_{k}.npz", pos=p["pos"], y=p["y"], noiseless_score=np.array(p["info"]["noiseless_score"]))
        json.dump(p["info"], open(d / f"info_{k}.json", "w"))
    jobs = {}
    for k, p in plan.items():
        job = sampler.run([p["pub"]], shots=shots); jobs[k] = job; meta["jobs"][k] = job.job_id()
        print("submitted", k, job.job_id(), flush=True)
    json.dump(meta, open(d / "meta.json", "w"), indent=1)                  # job ids saved before waiting
    for k, job in jobs.items():
        save_counts(d, k, job.result()); print("saved counts", k, flush=True)
    if ctx is not None:
        ctx.close()
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    json.dump(meta, open(d / "meta.json", "w"), indent=1)


def retrieve(tag):
    from qiskit_ibm_runtime import QiskitRuntimeService
    d = outdir(tag); meta = json.load(open(d / "meta.json"))
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"], instance=os.environ.get("QISKIT_IBM_INSTANCE") or None)
    for k, jid in meta["jobs"].items():
        save_counts(d, k, svc.job(jid).result()); print("saved", k, jid)


def analyze(tag):
    d = outdir(tag); out = {}
    for k in ALL:
        if not (d / f"counts_{k}.json").exists():
            continue
        info = json.load(open(d / f"info_{k}.json")); e = np.load(d / f"ecgs_{k}.npz"); y = e["y"]
        cnt = json.load(open(d / f"counts_{k}.json")); res = dict(auc_noiseless=T.auc(y, np.array(e["noiseless_score"])))
        if info["kind"] == "bw":
            n, kk, h, b = info["n"], info["k"], np.array(info["h"]), info["b"]
            fr = [H.features_from_counts(c, n) for c in cnt]; fp = [H.features_from_counts(c, n, kk) for c in cnt]
            res.update(auc_raw=T.auc(y, np.array([f for f, _ in fr]) @ h + b), auc_postselected=T.auc(y, np.array([f for f, _ in fp]) @ h + b),
                       kept_fraction_mean=float(np.mean([q for _, q in fp])))
        else:
            z = np.array([Q.z_from_counts(c) for c in cnt]); res["auc_raw"] = T.auc(y, z @ np.array(info["w"]) + info["b"])
        out[k] = res; print(k, {a: round(v, 4) for a, v in res.items()}, flush=True)
    json.dump(out, open(d / "analysis.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    for f in ("--local", "--submit", "--retrieve", "--analyze"):
        g.add_argument(f, action="store_true")
    ap.add_argument("--yes", action="store_true"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--models", default=",".join(ALL)); ap.add_argument("--max-minutes", type=float, default=8.0); ap.add_argument("--tag", default=None)
    a = ap.parse_args(); models = a.models.split(",")
    if a.local:
        run(True, True, a.n, a.shots, models, a.max_minutes, a.tag or "local"); analyze(a.tag or "local")
    elif a.submit:
        run(False, a.yes, a.n, a.shots, models, a.max_minutes, a.tag or "hw")
    elif a.retrieve:
        retrieve(a.tag or "hw"); analyze(a.tag or "hw")
    else:
        analyze(a.tag or "hw")
