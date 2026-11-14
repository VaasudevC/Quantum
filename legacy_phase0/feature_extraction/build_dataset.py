"""
Build a consolidated MI + NORM dataset from PTB-XL.

Outputs (in PTB_XL/):
  dataset_signals.npy   — float32 array, shape (N, 5000, 12)  [10 s @ 500 Hz, 12 leads, raw mV]
  dataset_metadata.csv  — one row per record: ecg_id, patient_id, age, sex,
                          label_class, label_subclass, mi_scp_codes, filename_hr

Sampling frequency: 500 Hz  |  Duration: 10 s  |  Leads: 12
At 500 Hz: J+60 ms = 30 samples, J+80 ms = 40 samples — sufficient for
ST-slope, ST-area and curvature feature extraction.
"""

import ast
import os
import wfdb
import numpy as np
import pandas as pd
from collections import Counter

BASE_PATH = r"<project_root>\PTB_XL"
DB_CSV    = os.path.join(BASE_PATH, "ptbxl_database.csv")
SCP_CSV   = os.path.join(BASE_PATH, "scp_statements.csv")

FS      = 500    # 500 Hz recordings  (records500/)
SAMPLES = 5000   # 10 seconds @ 500 Hz

# ── Load & label metadata (same logic as ComparitiveAnalysisMIvsControl.py) ──
print("Labelling records …")
df  = pd.read_csv(DB_CSV, index_col="ecg_id")
scp = pd.read_csv(SCP_CSV, index_col=0)

df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
diag    = scp[scp["diagnostic"] == 1.0]
mi_scp  = diag[diag["diagnostic_class"] == "MI"]

def superclasses(codes):
    return {
        diag.loc[c, "diagnostic_class"]
        for c, lik in codes.items()
        if c in diag.index and lik > 0 and pd.notna(diag.loc[c, "diagnostic_class"])
    }

def mi_specific_codes(codes):
    return [c for c, lik in codes.items() if c in mi_scp.index and lik > 0]

df["superclasses"] = df["scp_codes"].apply(superclasses)
mi_df   = df[df["scp_codes"].apply(lambda c: bool(mi_specific_codes(c)))].copy()
# Use "NORM" in s (not strict equality) to match official PTB-XL count of 9,514
# Then drop the 1 ambiguous record that also carries an MI code
norm_df = df[df["superclasses"].apply(lambda s: "NORM" in s)].copy()
mi_ids  = set(df[df["scp_codes"].apply(lambda c: bool(mi_specific_codes(c)))].index)
norm_df = norm_df[~norm_df.index.isin(mi_ids)]

mi_df["mi_scp_codes"]   = mi_df["scp_codes"].apply(mi_specific_codes)
mi_df["label_subclass"] = mi_df["mi_scp_codes"].apply(
    lambda codes: "+".join(sorted(set(mi_scp.loc[c, "diagnostic_subclass"] for c in codes)))
)
mi_df["label_class"]   = "MI"
norm_df["label_class"] = "NORM"
norm_df["label_subclass"]  = "NORM"
norm_df["mi_scp_codes"]    = [[] for _ in range(len(norm_df))]

combined = pd.concat([
    mi_df[["patient_id","age","sex","label_class","label_subclass","mi_scp_codes","filename_hr"]],
    norm_df[["patient_id","age","sex","label_class","label_subclass","mi_scp_codes","filename_hr"]],
])

print(f"  MI   : {len(mi_df):,}")
print(f"  NORM : {len(norm_df):,}")
print(f"  Total: {len(combined):,}")

# ── Load signals ──────────────────────────────────────────────────────────────
print("\nLoading ECG signals (this takes a few minutes) …")

signals  = np.zeros((len(combined), SAMPLES, 12), dtype=np.float32)
kept_idx = []

for i, (ecg_id, row) in enumerate(combined.iterrows()):
    path = os.path.join(BASE_PATH, row["filename_hr"])
    try:
        rec = wfdb.rdrecord(path)
        sig = rec.p_signal[:SAMPLES, :]          # (5000, 12) — raw mV, physical units
        signals[i] = sig.astype(np.float32)
        kept_idx.append(ecg_id)
    except Exception:
        pass   # file not downloaded yet — skip

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(combined)} loaded …")

# Trim to only successfully loaded records
signals  = signals[:len(kept_idx)]
metadata = combined.loc[kept_idx].copy()
metadata["mi_scp_codes"] = metadata["mi_scp_codes"].apply(str)

# ── Save ──────────────────────────────────────────────────────────────────────
sig_path  = os.path.join(BASE_PATH, "dataset_signals.npy")
meta_path = os.path.join(BASE_PATH, "dataset_metadata.csv")

np.save(sig_path, signals)
metadata.to_csv(meta_path)

print(f"\nSaved {len(kept_idx):,} records.")
print(f"  Signals  : {sig_path}  shape={signals.shape}  {signals.nbytes/1e6:.1f} MB")
print(f"  Metadata : {meta_path}")
print()
print("Label distribution:")
for lbl, grp in metadata.groupby(["label_class","label_subclass"]):
    print(f"  {lbl[0]:<5} {lbl[1]:<15} {len(grp):>5}")
