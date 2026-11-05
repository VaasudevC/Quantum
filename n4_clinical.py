"""N4: clinical metrics from stored params (no training).

Per model, on each fold's test rows, then mean ± SD over the 5 folds:
  sensitivity at 90% / 95% specificity (max TPR with FPR <= 0.10 / 0.05 on the test ROC),
  Brier score, calibration slope (logistic regression of y on logit(p); 1 = ideal, < 1 = over-confident),
  AUC by MI subtype vs all NORM test ECGs: AMI only, IMI only, AMI+IMI, other (LMI / PMI combinations).
Models: Stage 2 VQCs (k in {8,16,32} x {pi, pi/2, pi/6}, z readout, 1000 steps), N1 (zz) and N2 (trained scale),
plus classical references on the same k=32 features: LR (as lr_auc) and degree-2 LR (C chosen on validation AUC).
Note: the VQCs are trained with class-weighted BCE, i.e. calibrated to a 50/50 prior, not to the 36% MI prevalence.

  python -u n4_clinical.py   -> results/n4_clinical.json, results/n4_clinical.md
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import scale_cv as S
import Vqc_fast8 as V

X, y, _, _, feats = S.load()
sub = pd.read_csv(S._find("st_features_plus_TR.csv"), usecols=["label_subclass"])["label_subclass"].to_numpy()
GROUPS = {"AMI": sub == "AMI", "IMI": sub == "IMI", "AMI+IMI": sub == "AMI+IMI",
          "other MI": (y == 1) & ~np.isin(sub, ["AMI", "IMI", "AMI+IMI"])}


def metrics(yt, p, idx):
    fpr, tpr, _ = roc_curve(yt, p)
    lg = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lg[:, None], yt).coef_[0, 0]
    out = dict(auc=roc_auc_score(yt, p), sens_at_spec90=float(tpr[fpr <= 0.10].max()),
               sens_at_spec95=float(tpr[fpr <= 0.05].max()), brier=brier_score_loss(yt, p), cal_slope=float(slope))
    norm = yt == 0
    for g, mask in GROUPS.items():
        m = norm | mask[idx]
        out[f"auc_{g}"] = roc_auc_score(yt[m], p[m])
    return out


def vqc_rows(files, **flt):
    rows = []
    for f in files:
        if not Path(f).exists():
            continue
        for l in open(f):
            r = json.loads(l)
            if all(r.get(k, d) == v for k, (v, d) in flt.items()):
                rows.append(r)
    return rows


def vqc_proba(r):
    Z, tr, va, te, _ = S.prep(r["split"], r["fold"], r["selector"])
    sel = [feats.index(n) for n in r["features"]]
    T = np.zeros((len(y), V.N_FEATURES)); T[:, :r["k"]] = np.tanh(Z[:, sel])
    m = S.Linear(r["k"], eval(r["scale"], {"pi": np.pi}), r.get("readout", "z"), r.get("train_scale", False))
    return te, m.proba(T[te], np.asarray(r["params"])), Z[:, sel], tr, va


res = {}
base = dict(selector=("l1", "l1"), steps=(1000, 300), shuffle_later=(False, False), seed=(0, 0))
models = {}
for k in (8, 16, 32):
    for sc in ("pi", "pi/2", "pi/6"):
        rs = [r for r in vqc_rows(["results/stage2_steps1000.jsonl"], k=(k, None), scale=(sc, None), **base)]
        if k == 32 and sc == "pi/6":   # fold 4 rerun with the deterministic selector, if available
            f4 = vqc_rows(["results/n0_baseline_fold4.jsonl"])
            if f4:
                rs = [r for r in rs if r["fold"] != 4] + f4
        models[f"VQC z, k={k}, {sc}"] = rs
models["VQC zz (N1), k=32, pi/6"] = vqc_rows(["results/n1_zz.jsonl"])
models["VQC trained scale (N2), k=32"] = vqc_rows(["results/n2_trainscale.jsonl"])

for name, rs in models.items():
    if len(rs) < 5:
        print("skip (incomplete)", name, len(rs)); continue
    per = []; refs = {"LR": [], "degree-2 LR": []}
    for r in sorted(rs, key=lambda r: r["fold"]):
        te, p, Zs, tr, va = vqc_proba(r)
        per.append(metrics(y[te], p, te))
        if name == "VQC z, k=32, pi/6":                     # classical references on the same 32 features
            lr = LogisticRegression(max_iter=3000).fit(Zs[tr], y[tr])
            refs["LR"].append(metrics(y[te], lr.predict_proba(Zs[te])[:, 1], te))
            P = PolynomialFeatures(2, include_bias=False).fit_transform(Zs); P = StandardScaler().fit(P[tr]).transform(P)
            best = max((roc_auc_score(y[va], LogisticRegression(C=C, max_iter=5000).fit(P[tr], y[tr]).decision_function(P[va])), C)
                       for C in (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0))
            m2 = LogisticRegression(C=best[1], max_iter=5000).fit(P[tr], y[tr])
            refs["degree-2 LR"].append(metrics(y[te], m2.predict_proba(P[te])[:, 1], te))
    res[name] = pd.DataFrame(per).agg(["mean", "std"]).to_dict()
    for rn, v in refs.items():
        if v:
            res[f"{rn} (same k=32 features)"] = pd.DataFrame(v).agg(["mean", "std"]).to_dict()
    print(name, {k: round(v["mean"], 4) for k, v in res[name].items()}, flush=True)

json.dump(res, open("results/n4_clinical.json", "w"), indent=1)
cols = [("auc", "AUC"), ("sens_at_spec90", "Sens @ 90% spec"), ("sens_at_spec95", "Sens @ 95% spec"), ("brier", "Brier"),
        ("cal_slope", "Cal. slope"), ("auc_AMI", "AUC AMI"), ("auc_IMI", "AUC IMI"), ("auc_AMI+IMI", "AUC AMI+IMI"), ("auc_other MI", "AUC other MI")]
md = "| model | " + " | ".join(c[1] for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
for name, v in res.items():
    md += f"| {name} | " + " | ".join(f"{v[c]['mean']:.3f} ± {v[c]['std']:.3f}" for c, _ in cols) + " |\n"
n_te = {g: int(m.sum()) for g, m in GROUPS.items()}
md += f"\nSubtype sizes (all folds): {n_te}; each subtype AUC is that subtype vs all NORM ECGs of the test fold.\n"
Path("results/n4_clinical.md").write_text(md); print(md)
