# Running the IBM hardware job from Google Colab

Qiskit is blocked by WDAC on the lab PC; use Colab (Python 3.10–3.11, Linux, Qiskit allowed).
Everything the script needs is in `hw_bundle.zip` (~2 MB). No API keys are stored in the bundle.

---

## Prerequisites

1. **IBM Quantum account** — free tier gives 10 QPU minutes.
2. **API token** — copy it from https://quantum.ibm.com → "Account" → "API token".
3. **Colab Secrets** — in the Colab left sidebar click the key icon → add a secret named
   `QISKIT_IBM_TOKEN` with your token as the value. Enable "Notebook access" for it.
   (Alternative: set it in the cell that calls `os.environ` — never paste it in plain text.)

---

## Cell 1 — Upload bundle

```python
# Run this cell to upload hw_bundle.zip from your local machine.
from google.colab import files
uploaded = files.upload()   # pick hw_bundle.zip from cloud_repo/
```

---

## Cell 2 — Extract and install

```python
import zipfile, os

with zipfile.ZipFile("hw_bundle.zip") as zf:
    zf.extractall("hw")          # extracts to ./hw/

os.chdir("hw")                   # all subsequent cells run from here
print("Working dir:", os.getcwd())
print("Files:", os.listdir("."))
```

```python
# Install Qiskit stack (torch and numpy already pre-installed on Colab; do NOT reinstall torch).
# Takes ~2 min on a fresh Colab runtime.
%pip install -q -r requirements-colab.txt
```

---

## Cell 3 — Set IBM token

```python
import os
from google.colab import userdata   # reads from Colab Secrets (key icon in left sidebar)

os.environ["QISKIT_IBM_TOKEN"] = userdata.get("QISKIT_IBM_TOKEN")
# Optional — only needed if your account uses a specific CRN instance:
# os.environ["QISKIT_IBM_INSTANCE"] = "ibm-q/open/main"

# BW_CACHE defaults to "cache/" (the folder inside the bundle), so no override needed.
print("Token set:", bool(os.environ.get("QISKIT_IBM_TOKEN")))
```

---

## Cell 4 — Local dry run (FakeFez simulator, no account needed)

Transpiles both circuits and prints estimated QPU time. Takes ~1–2 min.

```python
# --local: runs on FakeFez (built-in fake backend), no network, no token required.
# Use a small n to keep it fast; the real run uses --n 100.
!python hw_run_ibm_brickwall.py --local --n 4 --shots 200 --models bw8,vqc32
```

Expected output: AUC numbers close to the noiseless scores, backend `fake_fez`, no errors.

---

## Cell 5 — Submit to real hardware

**Run once only** — job ids are saved to `results/hw/ibm_bw_hw/meta.json` before the script
waits for results. If Colab disconnects after submission, use Cell 6 to retrieve.

```python
# --yes skips the interactive prompt. Remove --yes to get a confirmation step first.
!python hw_run_ibm_brickwall.py --submit --n 100 --shots 1000 --models bw8,vqc32 --yes
```

The script prints estimated QPU time (≤ 8 min limit) and refuses to submit if it exceeds it.
After the job finishes it writes counts and analysis to `results/hw/ibm_bw_hw/`.

---

## Cell 6 — Retrieve (if Colab disconnected after submit)

If the runtime was lost after the job was submitted, re-run Cells 1–3 to restore the
environment, then run this cell. It re-reads `results/hw/ibm_bw_hw/meta.json` for job IDs.

```python
!python hw_run_ibm_brickwall.py --retrieve
```

---

## Cell 7 — Analyze only (re-run after retrieval)

```python
!python hw_run_ibm_brickwall.py --analyze
```

Output: `results/hw/ibm_bw_hw/analysis.json` with `auc_noiseless`, `auc_raw`,
`auc_postselected` (bw8), `kept_fraction_mean` (bw8), `auc_raw` (vqc32).

---

## Cell 8 — Download results

```python
from google.colab import files
import shutil, os

shutil.make_archive("hw_results", "zip", "results/hw")
files.download("hw_results.zip")
```

Unzip into `cloud_repo/results/hw/` on the lab PC and commit.

---

## Cell 9 — FakeMarrakesh noisy simulation (matched comparison with H6)

Run this cell to simulate the **same 200 ECGs** (from the real hardware run) using `FakeMarrakesh`
(or `FakeFez` if `FakeMarrakesh` is not available), enabling a direct comparison between real
device and simulation for the same circuit and inputs.

**Purpose:** generate `results/hw/ibm_bw_fakehw/` with the same format as `ibm_bw_hw/`, so that
`hw_analyze_extended.py --tag fakehw` can be run on it. This isolates whether the real-hardware
AUC degradation is explained by the IBM noise model, or whether real-device noise (crosstalk,
drift, leakage) adds more.

**Do NOT run Aer on the lab PC (Qiskit DLL blocked by WDAC).**

```python
# Cell 9 — FakeMarrakesh noisy simulation (matched to H6 real hardware run)
import json, os
import numpy as np
from pathlib import Path

# -- Load existing H6 real hardware ECG inputs and circuit info --
hw_dir = Path("results/hw/ibm_bw_hw")
os.makedirs("results/hw/ibm_bw_fakehw", exist_ok=True)
out_dir = Path("results/hw/ibm_bw_fakehw")

import shutil
# Copy ECG inputs (labels, noiseless scores) from the real hardware run
for model_name in ("bw8", "vqc32"):
    shutil.copy(hw_dir / f"ecgs_{model_name}.npz", out_dir / f"ecgs_{model_name}.npz")
    shutil.copy(hw_dir / f"info_{model_name}.json", out_dir / f"info_{model_name}.json")

# -- Try FakeMarrakesh; fall back to FakeFez --
try:
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    fake_be = FakeMarrakesh()
    backend_name = "fake_marrakesh"
    print("Using FakeMarrakesh")
except ImportError:
    from qiskit_ibm_runtime.fake_provider import FakeFez
    fake_be = FakeFez()
    backend_name = "fake_fez_fallback"
    print("FakeMarrakesh not found; using FakeFez as fallback")

from qiskit_aer import AerSimulator
sim = AerSimulator.from_backend(fake_be)

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2
pm = generate_preset_pass_manager(optimization_level=3, backend=fake_be, seed_transpiler=0)

SHOTS = 1000  # match the real hardware run

import hw_bw as H, qiskit_vqc as Q
cache = Path(os.environ.get("BW_CACHE", "cache"))

for model_name in ("bw8", "vqc32"):
    info = json.load(open(out_dir / f"info_{model_name}.json"))
    e = np.load(out_dir / f"ecgs_{model_name}.npz")
    # Reconstruct parameter bindings from the saved info and ECG positions
    if model_name == "bw8":
        n, k = info["n"], info["k"]
        qc, th, wv = H.build(n, 4, k)  # columns=4 from the trained model (see info)
        # Load the actual trained params from cache (same as the real run)
        import sys; sys.path.insert(0, ".")
        import bw_torch as T
        rows = [json.loads(l) for l in open("results/bw/p3_stageC.jsonl")]
        r = [r for r in rows if r["n"] == n and r["fold"] == 0 and r.get("rank") == 0][0]
        d = np.load(cache / "l1_f0.npz"); Z = d["Z"].astype(np.float64)[:, :n]
        pos = e["pos"]
        theta = -r["scale"] * np.tanh(Z[d["te"]][pos])
        w_flat = np.asarray(r["params"]["w"]).ravel()
        tc = pm.run(qc)
        vals = {p: np.asarray(theta[:, int(p.name[2:])] if p.name.startswith("x[") else
                              np.full(len(pos), w_flat[int(p.name[2:-1])]), dtype=float)
                for p in tc.parameters}
        # Simpler: rebuild vals dict
        vals = {}
        for p in tc.parameters:
            if p.name.startswith("x["):
                j = int(p.name[2:-1])
                vals[p] = theta[:, j].astype(float)
            elif p.name.startswith("w["):
                i = int(p.name[2:-1])
                vals[p] = np.full(len(pos), float(w_flat[i]))
        pub = (tc, vals)
    else:  # vqc32
        row = Q.load_models()[(32, "pi/6")]; th_, w_, b_ = Q.split_params(row)
        qc, x = Q.build_template(th_, 32)
        hw_data = np.load(cache / "hw_inputs.npz")
        pos = e["pos"]
        A = hw_data["A_vqc_te"][pos]
        tc = pm.run(qc)
        vals = {p: A[:, int(p.name[2:-1])].astype(float) for p in tc.parameters}
        pub = (tc, vals)

    # Run SamplerV2 on the Aer simulator
    sampler = SamplerV2(mode=sim)
    job = sampler.run([pub], shots=SHOTS)
    result = job.result()
    data = result[0].data
    ba = getattr(data, list(data.keys())[0])
    counts = [ba[i].get_counts() for i in range(ba.shape[0])] if ba.shape else [ba.get_counts()]
    json.dump(counts, open(out_dir / f"counts_{model_name}.json", "w"))
    print(f"Saved {model_name} counts ({len(counts)} ECGs)")

# Save meta
import datetime
meta_hw = json.load(open(hw_dir / "meta.json"))
meta = dict(backend=backend_name, local=True,
            submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            shots=SHOTS, n_ecg=len(np.load(out_dir / "ecgs_bw8.npz")["y"]),
            info=meta_hw["info"])
json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)
print("Done. Run hw_analyze_extended.py --tag fakehw to get extended_analysis.json.")
```

After running: `python hw_run_ibm_brickwall.py --analyze --tag fakehw` and
`python hw_analyze_extended.py --tag fakehw`, then download and copy to
`cloud_repo/results/hw/ibm_bw_fakehw/` on the lab PC for comparison with H6.

---

## Notes

- **BW_CACHE**: defaults to `cache/` (inside the bundle). To use the lab PC cache directly
  set `os.environ["BW_CACHE"] = "C:/bwcache"` before running (not needed on Colab).
- **bw12**: excluded from the default `--models bw8,vqc32`; it needs ~1 × 10⁶ shots per ECG
  for a meaningful AUC (p6_shots.json) and is not feasible on the free 10 QPU-minute tier.
- **Lazy imports not in bundle**: `qiskit_vqc.py:model_data()` imports `scale_cv`;
  `qiskit_vqc.py:pennylane_qnode()` imports `pennylane`; `Vqc_fast8.py:verify()` imports
  `Vqc_angle8`/`Vqc_hybrid8`. None of these functions are called by the hardware script,
  so the missing modules do not cause errors during `--local`, `--submit`, `--retrieve`,
  or `--analyze`.
- **Reconnect tip**: if the Colab runtime disconnects mid-run, the job is already on IBM's
  queue; use Cell 6 (`--retrieve`) after reconnecting.
