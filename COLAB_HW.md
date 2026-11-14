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
