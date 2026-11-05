# Hardware stages (H1–H5): prompt for the Claude cloud session

Paste everything inside the code block into the cloud session that has the `VaasudevC/Quantum` repo, on branch `claude/focused-wright-u3dkeg`.

The credit expires at **13:29 IST on 5 Nov**, so the prompt is time-boxed to finish and push by **12:45 IST**.

**Before H4 (real IBM hardware):** create a free IBM Quantum account at quantum.cloud.ibm.com. Add your API key as an environment variable named `QISKIT_IBM_TOKEN` in the cloud environment's settings (the gear next to "Default"). If you use a newer IBM Cloud account, also add `QISKIT_IBM_INSTANCE` (your instance CRN). Don't paste the key into the chat.

```
We are extending our published poster (PTB-XL MI vs NORM, 8-qubit re-uploading VQC). RESULTS.md has the
classical-simulation results: angle scale pi/6 fixes the re-uploading collapse (k=32 test AUC 0.951 vs
0.791 at pi). Now we test whether this survives REAL HARDWARE. Work stages H1-H5 in order.
HARD DEADLINE: everything committed and pushed by 12:45 IST today (check `TZ=Asia/Kolkata date` at
the start of every stage). If a stage would overrun, cut its size (fewer samples/shots), never skip
the commit. Commit + push after EVERY stage with a one-line finding. Append each stage to a new file
HARDWARE.md (what ran, table, 2-3 sentence reading). Do NOT retrain anything: use the trained params
already stored in results/stage2_*.jsonl ("params" field: 72 ansatz + 8 weights + 1 bias, linear
readout on <Z_j>). Rebuild that run's angles with scale_cv.prep()/load() for the same split, fold,
selector, k and scale. Models to use everywhere ("the 6 models"): fold 0, selector l1, 1000 steps,
k in {8, 32} x scale in {pi, pi/2, pi/6}. Test set = fold-0 test rows.

H1 - Qiskit circuit + equivalence check (~20 min)
- pip install qiskit qiskit-aer qiskit-ibm-runtime. Write qiskit_vqc.py: build_circuit(angles(32,),
  theta(72,)) that reproduces Vqc_fast8.forward EXACTLY: per layer L: Ry(angle+theta) on each qubit,
  Rz(theta) per qubit, CNOT ring i->(i+1)%8; final Ry layer; measure all. Vqc_fast8 is big-endian
  (qubit 0 = most significant bit), Qiskit is little-endian: handle the ordering and say how.
- Test: statevector from Qiskit vs Vqc_fast8.forward for 20 random inputs, max|diff| < 1e-10, and the
  6 models' noiseless predictions must reproduce their logged test AUC to 1e-6. Stop and report if not.

H2 - Finite-shot robustness (~30 min, own simulator, no Qiskit needed)
- For each of the 6 models: take exact probabilities from Vqc_fast8, sample counts with numpy
  multinomial at shots in {100, 300, 1000, 4000, 10000}, estimate <Z_j>, apply readout, compute test
  AUC. 10 repeats per shot count, report mean +- SD.
- Key question: at small angles (pi/6), the class differences in <Z> are smaller. Does pi/6 lose more
  AUC than pi/2 under shot noise? Report the shots needed for pi/6 to stay within 0.01 of noiseless.
- Plot results/figs/hw_shots.png (AUC vs shots, log-x, one line per model).

H3 - Compile to real devices + noisy simulation (~45 min)
- Transpile the k=32 pi/6 circuit (optimization_level=3, 5 seeds, keep best) for (a) IBM Heron via
  qiskit_ibm_runtime.fake_provider (FakeFez or FakeTorino; FakeSherbrooke/Eagle as second), (b) an
  IonQ-like all-to-all target (basis rz, ry, rx, rxx; no coupling map). Table: 1q gates, 2q gates,
  depth, 2q depth, estimated success (product of gate fidelities from the fake backend's properties).
  Note: the CNOT ring needs SWAPs on heavy-hex; report how many were added.
- Noisy simulation: AerSimulator.from_backend(FakeFez) on 300 stratified test ECGs for the 6 models,
  2000 shots each, transpiled circuits. Also apply measurement-error mitigation (qiskit or a simple
  per-qubit confusion-matrix inverse) and report AUC raw vs mitigated vs noiseless on the SAME 300.
- Plot results/figs/hw_noisy.png (grouped bars: noiseless / noisy / mitigated per model).

H4 - REAL IBM hardware (~45 min, only if QISKIT_IBM_TOKEN is set; otherwise write the exact script
  hw_run_ibm.py + instructions and skip)
- QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"],
  instance=os.environ.get("QISKIT_IBM_INSTANCE")); pick least_busy operational backend with >= 8 qubits.
  Open plan = ~10 QPU minutes per month: budget it. Plan: 2 models only (k=32 pi/6 and k=32 pi),
  100 stratified test ECGs each (same 100), 1000 shots, ONE SamplerV2 job per model in batch mode
  (each PUB = one ECG circuit), transpiled with the backend's preset pass manager, dynamical
  decoupling + twirling on if available. Save job IDs, backend name, calibration timestamp, raw counts
  to results/hw/ before any analysis.
- Report AUC on those 100 ECGs: hardware raw, hardware mitigated, Aer noisy (same backend's noise
  model), noiseless. Also the correlation between hardware and noiseless P(MI) per ECG.
- If the network blocks IBM (connection/proxy error), stop H4 after one retry, say so plainly, and
  leave hw_run_ibm.py ready for Deepak to run from his laptop.

H5 - Training-free encoding score (~30 min, only if time remains before 12:15 IST)
- For each scale in {pi/8, pi/6, pi/4, pi/3, pi/2, 2pi/3, pi} and k in {8, 16, 32}: on fold-0 train,
  encode with the ENCODING ONLY (ansatz theta = 0, so Ry(angle) layers + CNOT ring), compute the
  class-mean density matrices rho_MI, rho_NORM from 1000 samples per class, and report trace distance
  and fidelity between them. Correlate (Spearman) with the trained test AUC from results/*.jsonl.
  Plot results/figs/encoding_score.png. A strong correlation = a cheap way to choose the encoding
  before training.

Finish (by 12:45 IST): HARDWARE.md headline paragraph (does the pi/6 advantage survive shots, noise,
real hardware? exact numbers), the gate-count table, figures listed, limitations (100-300 ECGs, one
device, one calibration day, fold 0 only). Commit + push; print the commit hash and a 10-line summary
I can paste into the lab notebook.
```
