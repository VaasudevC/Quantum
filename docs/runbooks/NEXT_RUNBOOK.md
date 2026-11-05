# Next stages (N1–N4): prompt for the remaining cloud credit

Paste the code block into the same cloud session. It's time-boxed to push by **12:40 IST**.

```
New stages N1-N4 to improve the model, ordered by value. HARD DEADLINE: committed + pushed by
12:40 IST (check `TZ=Asia/Kolkata date` before each stage; shrink, never skip the commit).
Protocol unchanged unless stated: patient-grouped 5-fold CV, L1 selector (add random_state=0 for
new runs and say so), 8 qubits x 4 layers, k=32, scale pi/6, 1000 steps, choices on VAL AUC only.
Write results to NEXT.md (table + 2-3 sentence reading per stage); commit + push after each stage.
Compare every row against: VQC linear-Z pi/6 (0.951), LR (0.945), degree-2 LR (0.958).

N1 - ZZ readout (~25 min). Vqc_fast8 already supports readout "zz" (8 <Z_j> + 28 <Z_iZ_j>, all
from the same Z-basis shots, so no extra hardware cost). Add a --readout {z,zz} flag to scale_cv.py
(linear readout on 36 observables = 72+37 params) and run k=32, pi/6, 5 folds. Question: do pairwise
correlations close the gap to degree-2 LR? Also report the 180-shot AUC (multinomial sampling as in
H2) for zz vs z, since ZZ estimates are noisier.

N2 - Trainable input scaling (~40 min). Replace the fixed scale with a learned per-feature scale
s_f: angle_f = s_f * tanh(z_f), s_f initialised at pi/6, trained jointly with Adam. Gradient:
dL/ds_f = sum_b dL/dangle_{b,f} * tanh(z_{b,f}); the encoding Ry on qubit i, layer L is merged with
theta[base+i], so per-sample dL/dangle = that Ry gradient computed PER SAMPLE (extend backward()
with a per-sample variant; verify by finite differences on 5 params before running). Run k=32,
z readout, 5 folds; report AUC, and the learned scales (mean, range, which features go big/small).
If time is short, run folds 0-2 only and say so.

N3 - Better training-free encoding score (~20 min). H5's trace distance between class-mean states
was weak. Compute kernel-target alignment instead: K_ij = |<phi(x_i)|phi(x_j)>|^2 with the
encoding-only circuit (theta=0), 1000 stratified fold-0 train samples, KTA = <K, yy^T>_F /
(||K||_F * n) with y in {-1,+1}. Do it for the 7 scales x k in {8,16,32}; Spearman vs trained
test AUC from results/*.jsonl. Plot results/figs/kta_vs_auc.png. Report whether KTA beats trace
distance as a predictor.

N4 - Clinical metrics for the paper (~15 min, no training). From stored params (stage2 models and
N1/N2 best), report per model: sensitivity at 90% and 95% specificity, Brier score, calibration
slope, and AUC by MI subtype (label_subclass in the feature table, e.g. IMI vs AMI vs others, MI
vs all NORM). Table in NEXT.md.

Finish: NEXT.md headline (best model, does it beat degree-2 LR, best encoding score), commit +
push, print hash and a 10-line summary for the lab notebook.
```
