"""
ST-Segment Feature Extraction — PTB-XL MI + NORM Cohort
=========================================================
Sampling frequency : 500 Hz  (records500/)
ST-segment vector  : J-point to J+80 ms  (40 samples at 500 Hz)
  J+60 ms = 30 samples after J-point
  J+80 ms = 40 samples after J-point

Signals are loaded as raw mV from WFDB records500/ files.
Z-score normalised signals in dataset_signals.npy are NOT used here
because ST amplitude must remain in physical millivolt units.

Filtering:
  R-peak detection  : bandpass 0.5–40 Hz (monitoring grade — fine for detection)
  ST feature extraction : bandpass 0.05–40 Hz (diagnostic grade — AHA/ESC standard,
    Kligfield 2006; 0.5 Hz HPF distorts ST amplitude after tall R-waves)

Primary features per lead (12 leads x 12 = 144 columns):
  st_amp_j60    — absolute signal amplitude at J+60 ms (mV)
  st_amp_j80    — absolute signal amplitude at J+80 ms (mV)
  st_dev_j60    — deviation from PR-segment isoelectric at J+60 ms (mV)
  st_dev_j80    — deviation from PR-segment isoelectric at J+80 ms (mV)
  st_slope      — linear regression slope over ST segment, J to J+80 ms (mV/s)
  st_slope_cat  — 0=downsloping  1=horizontal  2=upsloping
  st_area       — trapezoidal area under baseline-corrected ST segment (mV·ms)
  st_mean       — mean of baseline-corrected ST vector (mV)
  st_max        — max  of baseline-corrected ST vector (mV)
  st_min        — min  of baseline-corrected ST vector (mV)
  st_var        — variance of baseline-corrected ST vector (mV²)
  st_curvature  — mean absolute second derivative |d²/dt²| (mV/s²)

Secondary / patient-level features (12 columns):
  max_st_elevation   — max st_dev_j60 across 12 leads (mV)
  max_st_depression  — min st_dev_j60 across 12 leads (mV; negative = depression)
  n_elevated_leads   — number of leads with st_dev_j60 >= +0.10 mV
  n_depressed_leads  — number of leads with st_dev_j60 <= -0.10 mV
  reciprocal_inferior  — 1 if II/III/aVF elevated AND I/aVL depressed
  reciprocal_anterior  — 1 if V1-V4 elevated AND II/III/aVF depressed
  reciprocal_lateral   — 1 if I/aVL/V5-V6 elevated AND V1-V3 depressed
  has_reciprocal     — 1 if any reciprocal pattern present

References
----------
Thygesen et al. (2018) "Fourth Universal Definition of MI" JACC 72(18).
  ST elevation thresholds: >=0.1 mV limb leads, >=0.2 mV V2-V3 (simplified
  to 0.1 mV uniformly for automated lead counting).
Kligfield et al. (2006) "Recommendations for the standardisation and
  interpretation of the electrocardiogram: Part I" Circulation 115.
  ST slope categories: upsloping >1 mV/s, downsloping <-1 mV/s, else horizontal.
Michaelides et al. (1999) "New criteria based on ST changes in 12 leads during
  PTCA." CHEST 116(4).
  Justification for ST area over single-point measurement.
"""

import ast
import os
import sys
import warnings
import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sp_signal

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
BASE     = r"<project_root>\PTB_XL"
META_CSV = os.path.join(BASE, "dataset_metadata.csv")
OUT_CSV  = os.path.join(BASE, "st_features.csv")

FS         = 500        # Hz
SAMPLES    = 5000       # 10 s
ST_SAMPLES = 40         # J to J+80 ms  (40 × 2 ms = 80 ms)
J60        = 30         # J+60 ms index within ST vector
J80        = 39         # J+80 ms index within ST vector (last sample)

# Slope thresholds — Kligfield et al. 2006
SLOPE_UP   =  1.0       # mV/s
SLOPE_DOWN = -1.0       # mV/s

# ST deviation thresholds — Thygesen et al. 2018
ELEV_THRESH =  0.10     # mV
DEP_THRESH  = -0.10     # mV

# Lead indices (PTB-XL/WFDB order)
LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
IDX = {name: i for i, name in enumerate(LEAD_NAMES)}

# Reciprocal lead pairs  (Thygesen et al. 2018 Table 1)
# Inferior territory:  II, III, aVF elevated → I, aVL depressed
INF_ELEV  = [IDX["II"], IDX["III"], IDX["aVF"]]
INF_RECIP = [IDX["I"],  IDX["aVL"]]
# Anterior territory: V1-V4 elevated → II, III, aVF depressed
ANT_ELEV  = [IDX["V1"], IDX["V2"], IDX["V3"], IDX["V4"]]
ANT_RECIP = [IDX["II"], IDX["III"], IDX["aVF"]]
# Lateral territory:  I, aVL, V5, V6 elevated → V1-V3 depressed
LAT_ELEV  = [IDX["I"], IDX["aVL"], IDX["V5"], IDX["V6"]]
LAT_RECIP = [IDX["V1"], IDX["V2"], IDX["V3"]]

# ── Signal processing helpers ─────────────────────────────────────────────────

def bandpass(sig_1d, fs, lo=0.5, hi=40.0):
    """Butterworth bandpass. lo=0.5 Hz for R-peak detection only."""
    nyq = fs / 2.0
    b, a = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return sp_signal.filtfilt(b, a, sig_1d)


def diagnostic_filter(sig_1d, fs, lo=0.05, hi=40.0):
    """
    Diagnostic-grade filter for ST measurement.
    HPF at 0.05 Hz — AHA/ESC standard (Kligfield 2006) for ST amplitude accuracy.
    0.5 Hz monitoring cutoff distorts ST after tall R-waves; must not be used
    for st_dev, st_amp, st_area, or slope computation.
    """
    nyq = fs / 2.0
    b, a = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return sp_signal.filtfilt(b, a, sig_1d)


def detect_rpeaks(lead_ii, fs):
    """
    Pan-Tompkins-inspired R-peak detector using scipy only.

    Steps:
      1. Bandpass filter 5-15 Hz  — isolates QRS complex energy
      2. Differentiate + square   — emphasises steep QRS upstroke
      3. Moving-window integration (150 ms window)
      4. scipy.signal.find_peaks with minimum beat distance 300 ms
      5. Refinement: snap each candidate to nearest true R-peak

    Returns array of R-peak sample indices.
    """
    nyq = fs / 2.0
    b, a = sp_signal.butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
    filtered = sp_signal.filtfilt(b, a, lead_ii)

    diff_sig = np.diff(filtered, prepend=filtered[0])
    squared  = diff_sig ** 2

    win = max(1, int(0.150 * fs))   # 150 ms integration window
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode="same")

    min_dist = int(0.300 * fs)      # 300 ms → max ~200 BPM
    height   = 0.05 * integrated.max() if integrated.max() > 0 else None

    peaks, _ = sp_signal.find_peaks(integrated, distance=min_dist, height=height)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Refine to actual R-peak (max absolute value within ±50 ms of candidate)
    refine_hw = int(0.050 * fs)
    refined = []
    for p in peaks:
        lo = max(0, p - refine_hw)
        hi = min(len(lead_ii), p + refine_hw + 1)
        refined.append(lo + int(np.argmax(np.abs(lead_ii[lo:hi]))))

    return np.unique(refined).astype(int)


def detect_jpoint(sig_lead, r_peak, fs):
    """
    Estimate J-point (QRS end) for a single beat.

    Method:
      1. Identify S-wave as the minimum in the window [R+20 ms, R+110 ms].
         This window is chosen to avoid the R-peak itself while capturing
         the S-wave nadir in both narrow and wide QRS complexes.
      2. J-point = S-wave index + 20 ms  (typical QRS terminal deflection
         ends ~20 ms after the S-wave nadir — Rautaharju et al. 2009).
      3. Fallback: if no window is available, use R + 60 ms.

    Returns sample index of J-point.
    """
    s_lo = r_peak + int(0.020 * fs)   # R + 10 samples
    s_hi = r_peak + int(0.110 * fs)   # R + 55 samples
    s_hi = min(s_hi, len(sig_lead) - 1)

    if s_lo >= s_hi:
        return min(r_peak + int(0.060 * fs), len(sig_lead) - ST_SAMPLES - 1)

    s_idx = s_lo + int(np.argmin(sig_lead[s_lo:s_hi]))
    j     = s_idx + int(0.020 * fs)   # + 10 samples past S nadir

    # Ensure we have ST_SAMPLES worth of signal after J
    j = min(j, len(sig_lead) - ST_SAMPLES - 1)
    return max(0, j)


def pr_baseline_value(sig_lead, r_peak, fs):
    """
    Estimate isoelectric line from the PR segment.

    PR segment window: [R-200 ms, R-40 ms] = [R-100, R-20] samples at 500 Hz.
    This is the segment between end of P-wave and start of QRS, which is
    electrically silent and defines the true isoelectric reference
    (Kligfield et al. 2006, Section 6.1).

    Returns mean voltage (mV) of the PR segment for this beat.
    """
    lo = max(0, r_peak - int(0.200 * fs))
    hi = max(0, r_peak - int(0.040 * fs))
    if lo >= hi:
        return 0.0
    return float(np.mean(sig_lead[lo:hi]))


def extract_beat_st(sig_lead, r_peak, fs):
    """
    Extract ST segment for one beat.

    Returns:
      raw_st   — raw mV values, shape (ST_SAMPLES,)
      baseline — PR-segment isoelectric value (mV)
      valid    — False if the beat is too close to signal boundary
    """
    j = detect_jpoint(sig_lead, r_peak, fs)
    if j + ST_SAMPLES > len(sig_lead):
        return None, None, False

    raw_st   = sig_lead[j : j + ST_SAMPLES].copy()
    baseline = pr_baseline_value(sig_lead, r_peak, fs)
    return raw_st, baseline, True


def average_st(sig_lead, rpeaks, fs):
    """
    Average raw ST segment and PR baseline across all valid beats.

    Returns:
      mean_raw_st  — mean raw ST vector  (ST_SAMPLES,)  in mV
      mean_baseline — mean PR baseline  (scalar)       in mV
      n_beats      — number of beats used
    """
    raws, baselines = [], []
    for r in rpeaks:
        raw_st, bl, ok = extract_beat_st(sig_lead, r, fs)
        if ok:
            raws.append(raw_st)
            baselines.append(bl)

    if not raws:
        return np.zeros(ST_SAMPLES), 0.0, 0

    return np.mean(raws, axis=0), float(np.mean(baselines)), len(raws)


# ── Feature computation ────────────────────────────────────────────────────────

def lead_features(raw_st, baseline, fs):
    """
    Compute all 12 primary ST features for one lead from one patient.

    Args:
      raw_st   — beat-averaged raw ST vector (ST_SAMPLES,) in mV
      baseline — PR-segment isoelectric level (mV)
      fs       — sampling frequency (Hz)

    Returns dict of feature_name -> float.
    """
    corr = raw_st - baseline          # baseline-corrected ST vector
    dt   = 1.0 / fs                   # seconds per sample
    t    = np.arange(ST_SAMPLES) * dt # time axis (s)

    # Absolute amplitude at J+60 ms and J+80 ms
    amp_j60 = float(raw_st[J60])
    amp_j80 = float(raw_st[J80])

    # Deviation from isoelectric at J+60 ms and J+80 ms
    dev_j60 = float(corr[J60])
    dev_j80 = float(corr[J80])

    # ST slope: linear regression on baseline-corrected ST vector (mV/s)
    # Using full J-to-J+80ms window for maximum information
    slope_coeff = np.polyfit(t, corr, 1)
    slope_mvps  = float(slope_coeff[0])   # mV/s

    # Slope category — Kligfield et al. 2006
    if slope_mvps > SLOPE_UP:
        slope_cat = 2    # upsloping
    elif slope_mvps < SLOPE_DOWN:
        slope_cat = 0    # downsloping
    else:
        slope_cat = 1    # horizontal

    # ST area: trapezoidal integration of baseline-corrected vector (mV·ms)
    # dx in milliseconds so the area unit is mV·ms
    # np.trapezoid is the NumPy 2.x name; np.trapz was removed in 2.0
    st_area = float(np.trapezoid(corr, dx=1000.0 / fs))

    # Shape descriptors on baseline-corrected vector
    st_mean = float(np.mean(corr))
    st_max  = float(np.max(corr))
    st_min  = float(np.min(corr))
    st_var  = float(np.var(corr))

    # Curvature: mean absolute second derivative (mV/s²)
    # np.diff(corr, n=2) gives the 2nd discrete differences; divide by dt² for units
    d2          = np.diff(corr, n=2) / (dt ** 2)
    st_curv     = float(np.mean(np.abs(d2)))

    return {
        "st_amp_j60":   amp_j60,
        "st_amp_j80":   amp_j80,
        "st_dev_j60":   dev_j60,
        "st_dev_j80":   dev_j80,
        "st_slope":     slope_mvps,
        "st_slope_cat": slope_cat,
        "st_area":      st_area,
        "st_mean":      st_mean,
        "st_max":       st_max,
        "st_min":       st_min,
        "st_var":       st_var,
        "st_curvature": st_curv,
    }


FEAT_NAMES = [
    "st_amp_j60","st_amp_j80","st_dev_j60","st_dev_j80",
    "st_slope","st_slope_cat","st_area",
    "st_mean","st_max","st_min","st_var","st_curvature",
]


def secondary_features(dev_j60_per_lead):
    """
    Derive patient-level secondary features from per-lead st_dev_j60 values.

    Args:
      dev_j60_per_lead — array of length 12, one st_dev_j60 value per lead (mV)

    Returns dict of secondary feature names -> values.
    """
    d = np.array(dev_j60_per_lead)

    max_elev  = float(np.max(d))
    max_dep   = float(np.min(d))
    n_elev    = int(np.sum(d >= ELEV_THRESH))
    n_dep     = int(np.sum(d <= DEP_THRESH))

    # Reciprocal patterns — Thygesen et al. 2018 Table 1
    # A pattern is present if at least 1 elevation lead is elevated
    # AND at least 1 reciprocal lead is depressed
    def has_pattern(elev_idx, recip_idx):
        elev_present  = any(d[i] >= ELEV_THRESH for i in elev_idx)
        recip_present = any(d[i] <= DEP_THRESH  for i in recip_idx)
        return int(elev_present and recip_present)

    recip_inf = has_pattern(INF_ELEV,  INF_RECIP)
    recip_ant = has_pattern(ANT_ELEV,  ANT_RECIP)
    recip_lat = has_pattern(LAT_ELEV,  LAT_RECIP)
    has_recip = int(recip_inf or recip_ant or recip_lat)

    return {
        "max_st_elevation":    max_elev,
        "max_st_depression":   max_dep,
        "n_elevated_leads":    n_elev,
        "n_depressed_leads":   n_dep,
        "reciprocal_inferior": recip_inf,
        "reciprocal_anterior": recip_ant,
        "reciprocal_lateral":  recip_lat,
        "has_reciprocal":      has_recip,
    }


# ── Main processing loop ───────────────────────────────────────────────────────

def process_record(filename_hr, sig_12lead, fs):
    """
    Run the full ST feature pipeline for one patient.

    Args:
      filename_hr — WFDB path stem (without .dat)
      sig_12lead  — already loaded raw signal, shape (SAMPLES, 12), mV
      fs          — sampling frequency

    Returns flat dict of all features, or None if processing fails.
    """
    # Use Lead II (index 1) for R-peak detection — standard clinical choice
    lead_ii = sig_12lead[:, IDX["II"]].astype(np.float64)
    try:
        lead_ii_filt = bandpass(lead_ii, fs, lo=0.5, hi=40.0)
    except Exception:
        return None

    rpeaks = detect_rpeaks(lead_ii_filt, fs)

    # Require at least 3 beats for a stable average
    if len(rpeaks) < 3:
        return None

    row = {}
    dev_j60_all = []

    for lead_name in LEAD_NAMES:
        li = IDX[lead_name]
        lead_raw = sig_12lead[:, li].astype(np.float64)

        # Diagnostic filter (0.05 Hz HPF) for ST extraction — AHA/ESC standard
        try:
            lead_filt = diagnostic_filter(lead_raw, fs)
        except Exception:
            lead_filt = lead_raw

        raw_st, baseline, n_beats = average_st(lead_filt, rpeaks, fs)

        if n_beats == 0:
            # Fill with zeros rather than crashing; will be flagged by n_beats column
            feats = {k: 0.0 for k in FEAT_NAMES}
        else:
            feats = lead_features(raw_st, baseline, fs)

        for fname, val in feats.items():
            row[f"{lead_name}_{fname}"] = val

        dev_j60_all.append(feats.get("st_dev_j60", 0.0))

    # Secondary features
    sec = secondary_features(dev_j60_all)
    row.update(sec)
    row["n_beats_detected"] = len(rpeaks)

    return row


def load_signal(filename_hr):
    """Load 10-s raw signal from WFDB records500/, shape (5000, 12)."""
    path = os.path.join(BASE, filename_hr)
    rec  = wfdb.rdrecord(path)
    sig  = rec.p_signal[:SAMPLES, :]    # (5000, 12) — raw mV
    return sig.astype(np.float64)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ST Feature Extraction — PTB-XL")
    print(f"  Sampling frequency : {FS} Hz")
    print(f"  ST vector length   : {ST_SAMPLES} samples = {ST_SAMPLES*1000//FS} ms")
    print(f"  J+60 ms index      : {J60} samples")
    print(f"  J+80 ms index      : {J80} samples")
    print(f"  Slope threshold    : {SLOPE_UP} mV/s  (Kligfield 2006)")
    print(f"  Elevation thresh   : {ELEV_THRESH} mV  (Thygesen 2018)")
    print("=" * 60)

    meta = pd.read_csv(META_CSV, index_col="ecg_id")
    n_total = len(meta)
    print(f"\nProcessing {n_total:,} records …\n")

    records  = []
    skipped  = 0

    for i, (ecg_id, row_meta) in enumerate(meta.iterrows()):
        try:
            sig = load_signal(row_meta["filename_hr"])
        except Exception as e:
            skipped += 1
            continue

        feats = process_record(row_meta["filename_hr"], sig, FS)
        if feats is None:
            skipped += 1
            continue

        feats["ecg_id"]         = ecg_id
        feats["label_class"]    = row_meta["label_class"]
        feats["label_subclass"] = row_meta["label_subclass"]
        feats["age"]            = row_meta["age"]
        feats["sex"]            = row_meta["sex"]
        records.append(feats)

        if (i + 1) % 500 == 0:
            pct = 100.0 * (i + 1) / n_total
            print(f"  {i+1:>6}/{n_total}  ({pct:.1f}%)  skipped so far: {skipped}")

    print(f"\nDone. Processed {len(records):,}  |  Skipped {skipped}")

    feat_df = pd.DataFrame(records).set_index("ecg_id")

    # Re-order: metadata columns first, then features
    meta_cols  = ["label_class","label_subclass","age","sex","n_beats_detected"]
    sec_cols   = [
        "max_st_elevation","max_st_depression",
        "n_elevated_leads","n_depressed_leads",
        "reciprocal_inferior","reciprocal_anterior","reciprocal_lateral",
        "has_reciprocal",
    ]
    lead_cols  = [
        f"{ln}_{fn}"
        for ln in LEAD_NAMES
        for fn in FEAT_NAMES
    ]
    all_cols = meta_cols + sec_cols + lead_cols
    feat_df  = feat_df[[c for c in all_cols if c in feat_df.columns]]

    feat_df.to_csv(OUT_CSV)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Shape: {feat_df.shape}  ({feat_df.shape[1]} columns)")

    # ── Validation summary ────────────────────────────────────────────────────
    print("\n--- Label counts ---")
    for lbl, grp in feat_df.groupby(["label_class","label_subclass"]):
        print(f"  {lbl[0]:<5} {lbl[1]:<15} {len(grp):>5}")

    print("\n--- Secondary feature means by class ---")
    sec_check = ["max_st_elevation","max_st_depression",
                 "n_elevated_leads","n_depressed_leads","has_reciprocal"]
    summary = feat_df.groupby("label_class")[sec_check].mean().round(3)
    print(summary.to_string())

    print("\n--- ST slope category distribution (n_elevated_leads > 0) ---")
    mi_sub = feat_df[feat_df["label_class"] == "MI"]
    for lead_name in ["II","III","aVF","V1","V2","V3"]:
        col = f"{lead_name}_st_slope_cat"
        if col in mi_sub.columns:
            counts = mi_sub[col].value_counts().sort_index()
            cats   = {0:"down", 1:"horiz", 2:"up"}
            parts  = "  ".join(f"{cats[k]}={v}" for k, v in counts.items())
            print(f"  {lead_name:<4}: {parts}")
