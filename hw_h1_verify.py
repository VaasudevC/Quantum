"""H1: equivalence of the Qiskit / PennyLane circuits with Vqc_fast8, plus the H0 speed benchmark.

  python -u hw_h1_verify.py      -> results/hw/h1_verify.json
"""
import json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import qiskit_vqc as Q
import Vqc_fast8 as V

OUT = Path("results/hw"); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
res = {}

# 1) statevector equivalence on 20 random inputs ------------------------------------------
A = rng.uniform(-np.pi, np.pi, (20, 32)); th = rng.normal(size=V.N_ANSATZ)
ref = V.forward(A, th)
d_q = max(np.abs(Q.statevector_bigendian(Q.build_circuit(A[b], th, measure=False)) - ref[b]).max() for b in range(20))
import pennylane as qml
zq = np.array(Q.pennylane_qnode(th, "default.qubit")(A)).T            # broadcast (20, 8)
d_pl = float(np.abs(zq - V.expectations(ref, "z")).max())
res["statevector_maxdiff_qiskit"] = float(d_q); res["z_maxdiff_pennylane"] = d_pl
print(f"Qiskit statevector max|diff| = {d_q:.2e}   PennyLane <Z> max|diff| = {d_pl:.2e}", flush=True)

# 2) the 6 models: noiseless Qiskit predictions vs logged test AUC ----------------------------
import sys
models = Q.load_models(); res["models"] = {}
for key, row in ([] if "--bench" in sys.argv else models.items()):
    A, y, tr, va, te = Q.model_data(row); th, w, b = Q.split_params(row)
    qc, x = Q.build_template(th, row["k"], measure=False)
    t0 = time.time()
    z = np.array([V.expectations(Q.statevector_bigendian(qc.assign_parameters(A[i, :row["k"]]))[None], "z")[0]
                  for i in te])
    auc = roc_auc_score(y[te], Q.logit_from_z(z, w, b))
    z_np = V.expectations(V.forward(A[te], th), "z")
    res["models"][f"k{key[0]}_{key[1]}"] = dict(logged=row["test_auc"], qiskit=float(auc),
                                               absdiff=float(abs(auc - row["test_auc"])),
                                               z_maxdiff=float(np.abs(z - z_np).max()), n_test=int(len(te)))
    print(f"k={key[0]:2d} {key[1]:5s} logged={row['test_auc']:.8f} qiskit={auc:.8f} "
          f"|diff|={abs(auc-row['test_auc']):.1e}  ({len(te)} ECGs, {time.time()-t0:.0f}s)", flush=True)

ok = len(res["models"]) == 6 and d_q < 1e-10 and d_pl < 1e-10 and all(m["absdiff"] < 1e-6 for m in res["models"].values())
res["pass"] = bool(ok)
if "--bench" not in sys.argv:
    print("H1 PASS" if ok else "H1 FAIL")

# 3) H0 benchmark: exact <Z> for 1000 ECGs, k=32 pi/6 model ----------------------------------
row = models[(32, "pi/6")]; A, y, tr, va, te = Q.model_data(row); th, w, b = Q.split_params(row)
Ab = A[te[:1000]]; bench = {}
t0 = time.time(); V.expectations(V.forward(Ab, th), "z"); bench["Vqc_fast8 (numpy, batched)"] = time.time() - t0
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit.quantum_info import SparsePauliOp
qc, x = Q.build_template(th, 32, measure=False)
obs = [SparsePauliOp.from_sparse_list([("Z", [j], 1.0)], 8) for j in range(8)]
t0 = time.time()
ev = EstimatorV2().run([(qc, [[o] for o in obs], Ab)]).result()[0].data.evs      # (8, 1000)
bench["Qiskit Aer EstimatorV2 (exact)"] = time.time() - t0
res["aer_estimator_z_maxdiff"] = float(np.abs(ev.T - V.expectations(V.forward(Ab, th), "z")).max())
for dev in ("default.qubit", "lightning.qubit"):
    qn = Q.pennylane_qnode(th, dev)
    t0 = time.time()
    if dev == "default.qubit":
        qn(Ab)                                   # parameter broadcasting
    else:
        [qn(a) for a in Ab]                      # lightning: one call per ECG
    bench[f"PennyLane {dev}"] = time.time() - t0
# with shots (1000 shots each, 1000 ECGs)
qm, xm = Q.build_template(th, 32, measure=True)
t0 = time.time(); SamplerV2().run([(qm, Ab)], shots=1000).result()
bench["Qiskit Aer SamplerV2, 1000 shots"] = time.time() - t0
qn = Q.pennylane_qnode(th, "lightning.qubit", shots=1000)
t0 = time.time(); [qn(a) for a in Ab[:200]]; bench["PennyLane lightning.qubit, 1000 shots (x5 from 200)"] = 5 * (time.time() - t0)
res["benchmark_secs_1000_ecgs"] = {k: round(v, 3) for k, v in bench.items()}
for k, v in bench.items():
    print(f"  {k:55s} {v:7.2f} s")
if "--bench" in sys.argv:
    old = json.load(open(OUT / "h1_verify.json")) if (OUT / "h1_verify.json").exists() else {}
    old.update({k: v for k, v in res.items() if k.startswith(("bench", "aer"))}); res = old
json.dump(res, open(OUT / "h1_verify.json", "w"), indent=1)
