"""Extract the PTB-XL feature table columns for (a) PTB Diagnostic ECG DB records (external validation) and
(b) a sanity subset of PTB-XL records (must reproduce the stored table rows).

REUSES ../extract_st_features.process_record and ../extract_tr_features.process_record_tr (imported, not copied).
PTB: 12 standard leads in PTB-XL order (I, II, III, aVR, aVL, aVF, V1-V6) from the 15-lead 1000 Hz records, resampled 1000 -> 500 Hz
(scipy.signal.resample_poly 1/2), first 10 s (5000 samples), mV.  Output has EXACTLY the columns of data/st_features_plus_TR.csv.gz
(ST columns + n_beats_detected + TR columns, same names/order).

  ../../.venv/Scripts/python.exe ptb_external/extract_ptb_features.py --mode xl-check --n 50
  ../../.venv/Scripts/python.exe ptb_external/extract_ptb_features.py --mode ptb
"""
import argparse, os, sys, json, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
os.environ["OMP_NUM_THREADS"] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, wfdb
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]                    # LimrickQuantumWork
sys.path.insert(0, str(ROOT))
import extract_st_features as EST
import extract_tr_features as ETR

# RECOVERED VERSION: the stored PTB-XL table's ST columns were built with the 0.5-40 Hz monitoring band-pass in place of the
# 0.05 Hz 'diagnostic_filter' that extract_st_features.py has used since its 2026-07-31 edit (verified: 0 of 153 columns differ,
# max abs diff 2.8e-14, on 8 records; see xl_check.json for 50).  We reproduce the table by patching the filter here (the original script is untouched).
EST.diagnostic_filter = lambda sig, fs, lo=0.05, hi=40.0: EST.bandpass(sig, fs, 0.5, 40.0)

TABLE = ROOT / "cloud_repo" / "data" / "st_features_plus_TR.csv.gz"
PTB = ROOT / "ptb-diagnostic-ecg-database-1.0.0"
XL = ROOT / "PTB_XL"
STD = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
META = ["ecg_id", "label_class", "label_subclass", "age", "sex"]


def table_cols():
    return list(pd.read_csv(TABLE, nrows=1).columns)


def feats_from_signal(sig, fs=500):
    st = EST.process_record("x", sig, fs)
    tr = ETR.process_record_tr(sig, fs)
    if st is None:
        return None
    row = dict(st)
    if tr is not None:
        row.update(tr)
    return row


def ptb_signal(rec_path):
    rec = wfdb.rdrecord(str(rec_path))
    names = [s.lower() for s in rec.sig_name]
    cols = [names.index(n) for n in STD]
    x = rec.p_signal[:, cols].astype(np.float64)              # mV
    assert rec.fs == 1000, rec.fs
    y = resample_poly(x, 1, 2, axis=0)                        # 500 Hz
    return y[:5000]


def header_info(rec_path):
    txt = open(str(rec_path) + ".hea", encoding="latin-1").read()
    reason = re.search(r"# Reason for admission:\s*(.*)", txt).group(1).strip()
    age = re.search(r"# age:\s*(\S+)", txt); sex = re.search(r"# sex:\s*(\S+)", txt)
    return reason, age.group(1) if age else None, sex.group(1) if sex else None


def do_ptb(rel):
    p = PTB / rel
    reason, age, sex = header_info(p)
    if reason not in ("Myocardial infarction", "Healthy control"):
        return dict(record=rel, reason=reason, status="dropped_other_diagnosis")
    try:
        sig = ptb_signal(p)
        if len(sig) < 5000:
            return dict(record=rel, reason=reason, status=f"short_{len(sig)}")
        f = feats_from_signal(sig)
    except Exception as e:
        return dict(record=rel, reason=reason, status=f"error_{type(e).__name__}")
    if f is None:
        return dict(record=rel, reason=reason, status="fewer_than_3_beats")
    return dict(record=rel, reason=reason, status="ok", patient=rel.split("/")[0], age=age, sex=sex, feats=f)


def do_xl(args):
    ecg_id, fn = args
    rec = wfdb.rdrecord(str(XL / fn))
    sig = rec.p_signal[:5000, :].astype(np.float64)
    return ecg_id, feats_from_signal(sig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=3); a = ap.parse_args()
    cols = table_cols(); fcols = [c for c in cols if c not in META]
    out_dir = Path(__file__).resolve().parent
    if a.mode == "xl-check":
        tab = pd.read_csv(TABLE).set_index("ecg_id")
        meta = pd.read_csv(XL / "ptbxl_database.csv", index_col="ecg_id")
        ids = np.random.default_rng(0).choice(tab.index.to_numpy(), a.n, replace=False)
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(do_xl, [(int(i), meta.loc[i, "filename_hr"]) for i in ids]))
        rows = {i: f for i, f in res if f is not None}
        new = pd.DataFrame.from_dict(rows, orient="index")[fcols]
        old = tab.loc[new.index, fcols]
        diff = (new.astype(float) - old.astype(float)).abs()
        rep = dict(n_requested=a.n, n_reproduced=len(new), max_abs_diff_overall=float(np.nanmax(diff.to_numpy())),
                   n_cols_diff_gt_1e9=int((diff.max() > 1e-9).sum()),
                   worst_cols=diff.max().sort_values(ascending=False).head(10).round(8).to_dict(),
                   n_nan_mismatch=int((new.isna() != old.isna()).to_numpy().sum()))
        json.dump(rep, open(out_dir / "xl_check.json", "w"), indent=1); print(json.dumps(rep, indent=1))
    elif a.mode == "ptb":
        recs = [l.strip() for l in open(PTB / "RECORDS") if l.strip()]
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(do_ptb, recs, chunksize=4))
        log = pd.DataFrame([{k: v for k, v in r.items() if k != "feats"} for r in res])
        log.to_csv(out_dir / "ptb_extraction_log.csv", index=False)
        ok = [r for r in res if r["status"] == "ok"]
        df = pd.DataFrame([r["feats"] for r in ok])
        for c in fcols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[fcols]
        df.insert(0, "record", [r["record"] for r in ok]); df.insert(1, "patient", [r["patient"] for r in ok])
        df.insert(2, "label", [1 if r["reason"] == "Myocardial infarction" else 0 for r in ok])
        df.to_csv(out_dir / "ptb_features.csv", index=False)
        print(log["status"].value_counts().to_string())
        print(df.groupby("label").agg(records=("record", "count"), patients=("patient", "nunique")).to_string())


if __name__ == "__main__":
    main()
