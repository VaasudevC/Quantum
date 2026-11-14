# T-wave / R-wave feature extraction — spec for Claude Code

Paste this into Claude Code in the project folder. It extends `extract_st_features.py`
without touching the existing ST pipeline or overwriting the shared dataset.

## 0. Safety first — don't touch the shared file

Do not modify or overwrite `PTB_XL/st_features.csv` in place. Other people's
experiments read that file. Steps:

1. `cp PTB_XL/st_features.csv PTB_XL/st_features_backup_<date>.csv` (just in case).
2. Write all new output to a **new** file: `PTB_XL/st_features_plus_TR.csv`.
3. The new file should contain everything already in `st_features.csv` (join on
   `ecg_id`) plus the new T-wave and R-wave columns below — so it's a strict
   superset, not a replacement.

## 1. What's missing today

`extract_st_features.py` only measures the ST segment (deviation, slope, area,
curvature at J+60/J+80ms). It has no R-wave or T-wave amplitude features. Two
gaps this creates, both flagged in `ECG_Lead_Selection_Textbook_Literature_Week9.docx`:

- Posterior MI shows up as a dominant R wave in V1 (mirror-image reciprocal
  pattern), not as an ST change — we currently can't detect it at all.
- Hyperacute T waves are often the earliest visible sign of an infarct, before
  ST elevation develops — we're not capturing them either.

## 2. New features to add, per lead (12 leads × N features)

Reuse the existing R-peak detection (Lead II, 0.5 Hz HPF bandpass — unchanged,
detection doesn't need amplitude accuracy) and the existing per-beat averaging
approach already used for ST features.

**R-wave features** (use the 0.5 Hz monitoring-grade filter, same as R-peak
detection — QRS spike amplitude doesn't need the 0.05 Hz diagnostic filter):

- `{LEAD}_r_amp` — R-wave peak amplitude relative to the PR-segment baseline
  (mV). Search window: R_peak_index ± 40 ms on that lead's own signal (R-peaks
  can be offset slightly lead-to-lead vs. the Lead II reference).
- `{LEAD}_s_amp` — S-wave amplitude: local minimum immediately after the R
  peak, within the same QRS window.
- `{LEAD}_rs_ratio` — `r_amp / abs(s_amp)`, guarded against divide-by-zero.
  This is the actual clinical criterion for posterior MI: R/S ratio > 1 in V1
  (normally R/S < 1 in V1). Most useful on V1 and V2, but compute for all 12
  leads for consistency / future use.

**T-wave features** (use the 0.05 Hz diagnostic filter, same as ST — T-wave
amplitude is also distorted by the 0.5 Hz cutoff):

- `{LEAD}_t_amp` — T-wave amplitude relative to baseline (mV). Search window:
  J-point + 100 ms to J-point + 400 ms (find the peak absolute deviation in
  that window). This window should scale with heart rate if possible (use a
  Bazett-style correction: window ∝ RR interval) since QT/T-wave timing
  shortens at faster heart rates — a fixed window is an acceptable first pass
  but will be less accurate for tachycardic records.
- `{LEAD}_t_inverted` — 1 if the T wave's sign is opposite to that lead's
  dominant QRS polarity (i.e., R-wave positive but T negative, or vice versa),
  else 0.
- `{LEAD}_t_hyperacute` — 1 if `t_amp` exceeds a threshold (start with 0.4 mV,
  tune against the labelled MI/NORM split) and `t_amp` is positive — flags the
  early hyperacute-T pattern described in Hampton.

Optional, cheap to add at the same time since the R-peak/baseline machinery is
already there: `{LEAD}_q_depth` and `{LEAD}_q_duration` (Q-wave features —
closes the other gap noted in the lead-selection doc, the Chapter 4
Q-wave-based localization criteria, which are currently also unmeasured).

## 3. Output

New file: `PTB_XL/st_features_plus_TR.csv` — same row structure as
`st_features.csv` (one row per `ecg_id`), with the new columns appended.
Naming should make it obvious what's new — `plus_TR` (or similar) rather than
overwriting the original name, per the "separate dataset with an appropriate
name" instruction.

## 4. Suggested validation once built

Re-run the same Mann-Whitney U separation-score check used for the ST features
(see `check_leads.py` in the project) on `t_amp`, `rs_ratio`, and
`t_hyperacute` for V1 specifically, to see whether they add separation beyond
what `st_dev_j60` already gives on V1 alone.
