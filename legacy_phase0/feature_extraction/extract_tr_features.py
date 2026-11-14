"""
T-wave, R-wave, S-wave, Q-wave Feature Extraction — PTB-XL MI + NORM Cohort
=============================================================================
Extends st_features.csv with QRS-complex and T-wave amplitude features.
Addresses two gaps identified in ECG_Lead_Selection_Textbook_Literature_Week9.docx:
  1. Posterior MI — detected via dominant R in V1 (R/S ratio > 1), not ST change.
  2. Hyperacute T-waves — earliest infarct sign, before ST elevation develops.

Safety: this script does NOT modify PTB_XL/st_features.csv.
  - Creates PTB_XL/st_features_backup_<date>.csv as a one-time safety copy.
  - Joins new features with the existing CSV on ecg_id.
  - Writes PTB_XL/st_features_plus_TR.csv — strict superset of st_features.csv.

Filtering (same policy as extract_st_features.py):
  R-peak detection     : bandpass 0.5–40 Hz (monitoring grade)
  QRS amplitudes (R,S,Q): bandpass 0.5–40 Hz (monitoring grade — QRS deflections
                           are large; 0.5 Hz HPF distortion is negligible vs. QRS
                           amplitude)
  T-wave amplitude     : bandpass 0.05–40 Hz (diagnostic grade — AHA/ESC standard,
                           Kligfield 2006; slow T-wave shape is distorted by 0.5 Hz HPF)

New features per lead (12 leads × 8 = 96 columns):
  {LEAD}_r_amp        — R-wave peak relative to PR-segment baseline (mV)
  {LEAD}_s_amp        — S-wave nadir relative to PR-segment baseline (mV; ≤0 = normal S)
  {LEAD}_rs_ratio     — r_amp / |s_amp|; posterior MI criterion: R/S > 1 in V1
                         (normally R < S in V1; dominant R = posterior wall reciprocal)
  {LEAD}_q_depth      — Q-wave nadir relative to PR baseline (mV; negative = Q present)
  {LEAD}_q_duration   — Q-wave duration (ms); pathological: ≥30 ms (Thygesen 2018)
  {LEAD}_t_amp        — T-wave peak relative to PR baseline (mV; signed)
  {LEAD}_t_inverted   — 1 if T polarity opposes dominant QRS (R-wave) polarity
  {LEAD}_t_hyperacute — 1 if t_amp ≥ 0.4 mV and positive (Hampton hyperacute-T pattern)

References
----------
Hampton J.R. (2013) "The ECG Made Easy", 8th ed. Chapters 3 (T-waves) and 4 (Q-waves).
Thygesen et al. (2018) "Fourth Universal Definition of MI" JACC 72(18).
  Q-wave threshold: ≥0.03 s duration, amplitude ≥ 1/4 of R-wave.
Kligfield et al. (2006) "Recommendations for the standardisation and
  interpretation of the ECG: Part I" Circulation 115.
"""

import os
import shutil
import warnings
from datetime import date

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sp_signal

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = r"<project_root>\PTB_XL"
META_CSV = os.path.join(BASE, "dataset_metadata.csv")
IN_CSV   = os.path.join(BASE, "st_features.csv")
OUT_CSV  = os.path.join(BASE, "st_features_plus_TR.csv")

# ── Constants ──────────────────────────────────────────────────────────────────
FS         = 500     # Hz
SAMPLES    = 5000    # 10 s at 500 Hz
ST_SAMPLES = 40      # reused for J-point boundary guard

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
IDX = {name: i for i, name in enumerate(LEAD_NAMES)}

# T-wave search window (from J-point)
T_START_S = 0.100    # J + 100 ms — well past QRS end
T_MAX_S   = 0.360    # J + 360 ms — absolute upper bound
# Hyperacute T threshold (Hampton; will appear in validation summary for tuning)
T_HYPER_MV = 0.40

# QRS search windows (relative to Lead-II R-peak)
RS_HALF_WIN_S = 0.040   # R ± 40 ms for per-lead R-peak search
S_SEARCH_END_S = 0.080  # R + 80 ms for S-wave minimum search
Q_SEARCH_LO_S  = 0.060  # R - 60 ms start of Q-wave window
Q_SEARCH_HI_S  = 0.010  # R - 10 ms end   of Q-wave window

# Minimum absolute amplitude to consider a deflection "meaningful" for polarity
POLARITY_MIN_MV = 0.05


# ── Signal helpers (identical logic to extract_st_features.py) ────────────────

def bandpass(sig_1d, fs, lo=0.5, hi=40.0):
    """Butterworth bandpass; lo=0.5 Hz for monitoring-grade (R-peak / QRS)."""
    nyq = fs / 2.0
    b, a = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return sp_signal.filtfilt(b, a, sig_1d)


def diagnostic_filter(sig_1d, fs, lo=0.05, hi=40.0):
    """
    Diagnostic-grade bandpass for T-wave amplitude.
    HPF at 0.05 Hz — AHA/ESC standard (Kligfield 2006).
    """
    nyq = fs / 2.0
    b, a = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return sp_signal.filtfilt(b, a, sig_1d)


def detect_rpeaks(lead_ii, fs):
    """
    Pan-Tompkins-inspired R-peak detector (scipy only).
    Identical implementation to extract_st_features.py.
    """
    nyq = fs / 2.0
    b, a = sp_signal.butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
    filtered   = sp_signal.filtfilt(b, a, lead_ii)
    diff_sig   = np.diff(filtered, prepend=filtered[0])
    squared    = diff_sig ** 2
    win        = max(1, int(0.150 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    min_dist   = int(0.300 * fs)
    height     = 0.05 * integrated.max() if integrated.max() > 0 else None
    peaks, _   = sp_signal.find_peaks(integrated, distance=min_dist, height=height)
    if len(peaks) == 0:
        return np.array([], dtype=int)
    hw = int(0.050 * fs)
    refined = []
    for p in peaks:
        lo_r = max(0, p - hw)
        hi_r = min(len(lead_ii), p + hw + 1)
        refined.append(lo_r + int(np.argmax(np.abs(lead_ii[lo_r:hi_r]))))
    return np.unique(refined).astype(int)


def detect_jpoint(sig_lead, r_peak, fs):
    """
    Estimate J-point (QRS end) for one beat.
    Identical to extract_st_features.py: S-nadir + 20 ms.
    """
    s_lo = r_peak + int(0.020 * fs)
    s_hi = min(r_peak + int(0.110 * fs), len(sig_lead) - 1)
    if s_lo >= s_hi:
        return min(r_peak + int(0.060 * fs), len(sig_lead) - ST_SAMPLES - 1)
    s_idx = s_lo + int(np.argmin(sig_lead[s_lo:s_hi]))
    j     = s_idx + int(0.020 * fs)
    return max(0, min(j, len(sig_lead) - ST_SAMPLES - 1))


def pr_baseline(sig_lead, r_peak, fs):
    """
    Mean voltage of the PR segment [R-200 ms, R-40 ms] as isoelectric reference.
    Identical to extract_st_features.py: pr_baseline_value().
    """
    lo = max(0, r_peak - int(0.200 * fs))
    hi = max(0, r_peak - int(0.040 * fs))
    if lo >= hi:
        return 0.0
    return float(np.mean(sig_lead[lo:hi]))


# ── T-wave window sizing ───────────────────────────────────────────────────────

def t_end_from_j(median_rr_s):
    """
    Bazett-corrected T-wave window end, measured from J-point (seconds).

    QT ≈ 0.42 × √RR_s (Bazett). T-wave ends at roughly QT minus QRS duration
    (~80 ms between R and J). Clamped to [T_START_S + 50 ms, T_MAX_S].
    """
    rr_s  = max(median_rr_s, 0.30)   # floor at 200 BPM to avoid degenerate windows
    qt_s  = 0.42 * np.sqrt(rr_s)
    t_end = qt_s - 0.080             # subtract ~QRS duration already between R and J
    return float(np.clip(t_end, T_START_S + 0.050, T_MAX_S))


# ── Per-beat feature extractors ────────────────────────────────────────────────

def beat_qrs(sig_mon, r_peak, bl, fs):
    """
    Extract R, S, Q features for one beat from the monitoring-filtered signal.

    Args:
      sig_mon — 0.5 Hz HPF bandpass signal for this lead
      r_peak  — Lead-II R-peak index (may be slightly offset on other leads)
      bl      — PR-segment baseline from the same filtered signal (mV)
      fs      — sampling frequency

    Returns dict of QRS features, or None if windows fall outside signal bounds.
    """
    N = len(sig_mon)

    # R-wave: maximum within ±40 ms of the Lead-II R-peak index
    # The ±40 ms search absorbs the typical lead-to-lead R-peak timing offset
    hw   = int(RS_HALF_WIN_S * fs)
    r_lo = max(0, r_peak - hw)
    r_hi = min(N, r_peak + hw + 1)
    if r_lo >= r_hi:
        return None
    r_idx = r_lo + int(np.argmax(sig_mon[r_lo:r_hi]))
    r_amp = float(sig_mon[r_idx]) - bl

    # S-wave: minimum immediately after the per-lead R-peak, within QRS terminal window
    s_lo  = r_idx + 2
    s_hi  = min(N, r_idx + int(S_SEARCH_END_S * fs))
    s_amp = 0.0
    if s_lo < s_hi:
        s_idx = s_lo + int(np.argmin(sig_mon[s_lo:s_hi]))
        s_amp = float(sig_mon[s_idx]) - bl

    # R/S ratio: posterior MI criterion (R/S > 1 in V1)
    rs_ratio = r_amp / max(abs(s_amp), 1e-6)

    # Q-wave: minimum in the pre-QRS window [R-60 ms, R-10 ms]
    q_lo    = max(0, r_peak - int(Q_SEARCH_LO_S * fs))
    q_hi    = max(0, r_peak - int(Q_SEARCH_HI_S * fs))
    q_depth = 0.0
    q_dur   = 0.0
    if q_lo < q_hi:
        window  = sig_mon[q_lo:q_hi].astype(np.float64) - bl
        q_depth = float(np.min(window))
        # Duration = contiguous samples below baseline in Q window (ms)
        q_dur   = float(np.sum(window < 0)) * 1000.0 / fs

    return {
        "r_amp":      r_amp,
        "s_amp":      s_amp,
        "rs_ratio":   rs_ratio,
        "q_depth":    q_depth,
        "q_duration": q_dur,
    }


def beat_t_amp(sig_diag, j_point, t_end_samp, bl, fs):
    """
    Extract signed T-wave peak amplitude for one beat from the diagnostic-filtered
    signal.  Peak is the maximum absolute deviation from baseline in the T-window,
    with sign preserved.

    Args:
      sig_diag    — 0.05 Hz HPF bandpass signal for this lead
      j_point     — J-point sample index for this beat
      t_end_samp  — T-window end in samples from J-point
      bl          — PR-segment baseline from the same filtered signal (mV)

    Returns float t_amp (mV).
    """
    N       = len(sig_diag)
    t_start = j_point + int(T_START_S * fs)
    t_end   = min(j_point + t_end_samp, N - 1)
    if t_start >= t_end:
        return 0.0
    window  = sig_diag[t_start:t_end].astype(np.float64) - bl
    peak_i  = int(np.argmax(np.abs(window)))
    return float(window[peak_i])


# ── Per-lead beat averaging ────────────────────────────────────────────────────

QRS_KEYS = ["r_amp", "s_amp", "rs_ratio", "q_depth", "q_duration"]
TR_FEAT_NAMES = QRS_KEYS + ["t_amp", "t_inverted", "t_hyperacute"]


def _zero_row():
    return {k: 0.0 for k in TR_FEAT_NAMES}


def average_tr(sig_mon, sig_diag, rpeaks, fs):
    """
    Compute beat-averaged T/R/Q features for one lead.

    Uses:
      sig_mon  — monitoring-filtered (0.5 Hz HPF) for R, S, Q amplitude
      sig_diag — diagnostic-filtered (0.05 Hz HPF) for T-wave amplitude
      rpeaks   — R-peak indices from Lead-II detection
    """
    if len(rpeaks) == 0:
        return _zero_row()

    # Bazett T-window: based on median RR interval
    rr_arr      = np.diff(rpeaks)
    median_rr_s = float(np.median(rr_arr)) / fs if len(rr_arr) > 0 else 0.80
    t_end_samp  = int(t_end_from_j(median_rr_s) * fs)

    qrs_list = []
    t_amps   = []

    for r in rpeaks:
        bl_mon  = pr_baseline(sig_mon,  r, fs)
        bl_diag = pr_baseline(sig_diag, r, fs)

        qrs = beat_qrs(sig_mon, r, bl_mon, fs)
        if qrs is not None:
            qrs_list.append(qrs)

        j = detect_jpoint(sig_diag, r, fs)
        t_amps.append(beat_t_amp(sig_diag, j, t_end_samp, bl_diag, fs))

    avg_qrs = (
        {k: float(np.mean([b[k] for b in qrs_list])) for k in QRS_KEYS}
        if qrs_list else {k: 0.0 for k in QRS_KEYS}
    )
    avg_t = float(np.mean(t_amps)) if t_amps else 0.0

    # Derived binary flags computed from beat-averaged values
    r_amp = avg_qrs["r_amp"]
    t_inverted = int(
        abs(r_amp) >= POLARITY_MIN_MV
        and abs(avg_t) >= POLARITY_MIN_MV
        and np.sign(r_amp) != np.sign(avg_t)
    )
    t_hyperacute = int(avg_t >= T_HYPER_MV)

    return {**avg_qrs, "t_amp": avg_t, "t_inverted": t_inverted, "t_hyperacute": t_hyperacute}


# ── Record-level processor ─────────────────────────────────────────────────────

def process_record_tr(sig_12lead, fs):
    """
    Run the T/R/Q pipeline for one patient record.

    Returns flat dict {"{LEAD}_{feat}": value} for all 12 leads,
    or None if fewer than 3 beats are detected.
    """
    lead_ii = sig_12lead[:, IDX["II"]].astype(np.float64)
    try:
        lead_ii_filt = bandpass(lead_ii, fs, lo=0.5, hi=40.0)
    except Exception:
        return None

    rpeaks = detect_rpeaks(lead_ii_filt, fs)
    if len(rpeaks) < 3:
        return None

    row = {}
    for lead_name in LEAD_NAMES:
        li  = IDX[lead_name]
        raw = sig_12lead[:, li].astype(np.float64)

        try:
            sig_mon = bandpass(raw, fs, lo=0.5, hi=40.0)
        except Exception:
            sig_mon = raw
        try:
            sig_diag = diagnostic_filter(raw, fs, lo=0.05, hi=40.0)
        except Exception:
            sig_diag = raw

        feats = average_tr(sig_mon, sig_diag, rpeaks, fs)
        for fname, val in feats.items():
            row[f"{lead_name}_{fname}"] = val

    return row


def load_signal(filename_hr):
    path = os.path.join(BASE, filename_hr)
    rec  = wfdb.rdrecord(path)
    return rec.p_signal[:SAMPLES, :].astype(np.float64)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("T/R/Q Feature Extraction — PTB-XL")
    print(f"  Sampling frequency    : {FS} Hz")
    print(f"  T-wave search start   : J + {int(T_START_S * 1000)} ms")
    print(f"  T-wave max window end : J + {int(T_MAX_S * 1000)} ms (Bazett-corrected)")
    print(f"  Hyperacute threshold  : {T_HYPER_MV} mV")
    print(f"  Q pathological        : duration ≥ 30 ms (Thygesen 2018)")
    print(f"  Posterior MI crit.    : V1 R/S ratio > 1.0")
    print("=" * 60)

    # ── Safety backup ─────────────────────────────────────────────────────────
    bak = IN_CSV.replace(".csv", f"_backup_{date.today()}.csv")
    if not os.path.exists(bak):
        shutil.copy2(IN_CSV, bak)
        print(f"\nBacked up existing CSV to:\n  {bak}")
    else:
        print(f"\nBackup already exists (skipping):\n  {bak}")

    # ── Load inputs ───────────────────────────────────────────────────────────
    st_df = pd.read_csv(IN_CSV, index_col="ecg_id")
    meta  = pd.read_csv(META_CSV, index_col="ecg_id")
    # Process only records that already exist in st_features.csv
    meta  = meta.loc[meta.index.isin(st_df.index)]
    n_total = len(meta)
    print(f"\nProcessing {n_total:,} records …\n")

    tr_records = []
    skipped    = 0

    for i, (ecg_id, row_meta) in enumerate(meta.iterrows()):
        try:
            sig = load_signal(row_meta["filename_hr"])
        except Exception:
            skipped += 1
            continue

        feats = process_record_tr(sig, FS)
        if feats is None:
            skipped += 1
            continue

        feats["ecg_id"] = ecg_id
        tr_records.append(feats)

        if (i + 1) % 500 == 0:
            pct = 100.0 * (i + 1) / n_total
            print(f"  {i+1:>6}/{n_total}  ({pct:.1f}%)  skipped so far: {skipped}")

    print(f"\nDone. Extracted T/R/Q for {len(tr_records):,} records  |  Skipped {skipped}")

    tr_df = pd.DataFrame(tr_records).set_index("ecg_id")

    # ── Join: new TR columns appended to the right of all existing ST columns ──
    out_df = st_df.join(tr_df, how="left")

    tr_lead_cols = [f"{ln}_{fn}" for ln in LEAD_NAMES for fn in TR_FEAT_NAMES]
    existing_cols = list(st_df.columns)
    new_cols      = [c for c in tr_lead_cols if c in out_df.columns]
    out_df        = out_df[existing_cols + new_cols]

    out_df.to_csv(OUT_CSV)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Shape: {out_df.shape}  ({out_df.shape[1]} total columns)")
    print(f"  ST columns (original) : {len(existing_cols)}")
    print(f"  New T/R/Q columns     : {len(new_cols)}")

    # ── Validation summary ─────────────────────────────────────────────────────
    print("\n--- Mean T/R/Q features by class ---")
    check_cols = [
        "V1_r_amp",   "V2_r_amp",
        "V1_rs_ratio","V2_rs_ratio",
        "V1_t_amp",   "V2_t_amp",   "V3_t_amp",
        "II_q_duration", "III_q_duration", "aVF_q_duration",
    ]
    valid_check = [c for c in check_cols if c in out_df.columns]
    if valid_check:
        summary = out_df.groupby("label_class")[valid_check].mean().round(3)
        print(summary.to_string())

    print("\n--- V1/V2 R/S ratio > 1 (posterior MI indicator) ---")
    for lead in ["V1", "V2"]:
        col = f"{lead}_rs_ratio"
        if col in out_df.columns:
            for lbl, grp in out_df.groupby("label_class"):
                n_pos = int((grp[col] > 1.0).sum())
                print(f"  {lbl:4} {lead}: R/S > 1  →  {n_pos:>5}/{len(grp)}  ({100*n_pos/len(grp):.1f}%)")

    print("\n--- Hyperacute T rate by class (V1–V4) ---")
    for lbl, grp in out_df.groupby("label_class"):
        parts = []
        for lead in ["V1", "V2", "V3", "V4"]:
            col = f"{lead}_t_hyperacute"
            if col in grp.columns:
                n = int(grp[col].sum())
                parts.append(f"{lead}:{n}/{len(grp)}({100*n/len(grp):.1f}%)")
        print(f"  {lbl:4}  " + "  ".join(parts))

    print("\n--- T-wave inversion rate by class (key leads) ---")
    for lbl, grp in out_df.groupby("label_class"):
        parts = []
        for lead in ["V1", "V2", "V3", "III", "aVF"]:
            col = f"{lead}_t_inverted"
            if col in grp.columns:
                n = int(grp[col].sum())
                parts.append(f"{lead}:{n}/{len(grp)}({100*n/len(grp):.1f}%)")
        print(f"  {lbl:4}  " + "  ".join(parts))

    print("\n--- Q-wave duration (III, aVF, V3 — inferior/anterior localization) ---")
    for lbl, grp in out_df.groupby("label_class"):
        parts = []
        for lead in ["II", "III", "aVF", "V2", "V3"]:
            col = f"{lead}_q_duration"
            if col in grp.columns:
                mean_ms = grp[col].mean()
                n_path  = int((grp[col] >= 30.0).sum())
                parts.append(f"{lead}:{mean_ms:.1f}ms(≥30ms:{n_path})")
        print(f"  {lbl:4}  " + "  ".join(parts))
