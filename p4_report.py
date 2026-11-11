"""P4 report: external AUC / sens@90%spec with patient-cluster bootstrap CIs, one-record-per-patient variant, advantage rule (CI only).
  python p4_report.py  -> results/ext/p4_report.json, results/ext/p4_report.md   (numpy only)"""
import json, sys
from pathlib import Path
import numpy as np
import analysis_utils as A

def rows(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []

def sens90(y, s):
    thr = np.quantile(s[y == 0], 0.90); return float((s[y == 1] > thr).mean())

def boot_stat(y, g, s, fn, R=1000, seed=0):
    rng = np.random.default_rng(seed); u, inv = np.unique(g, return_inverse=True); idx = [np.flatnonzero(inv == i) for i in range(len(u))]
    v = []
    for _ in range(R):
        pick = rng.integers(0, len(u), len(u)); ii = np.concatenate([idx[p] for p in pick])
        if y[ii].min() == y[ii].max(): continue
        v.append(fn(y[ii], s[ii]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def main():
    d = np.load("C:/bwcache/full.npz", allow_pickle=True); te = d["te"]; y = d["y"][te]; g = d["g"][te]
    R = rows("results/ext/scores.jsonl"); latest = {}
    for r in R: latest[(r["model"], r["k"])] = r
    out = {}; rng = np.random.default_rng(0)
    # one record per patient: random pick, 200 draws; report mean AUC over draws and the cluster-bootstrap CI of the first draw
    u = np.unique(g); draws = [np.array([rng.choice(np.flatnonzero(g == p)) for p in u]) for _ in range(200)]
    for (m, k), r in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        s = np.array(r["scores"])
        a = A.auc(y, s)
        one = [A.auc(y[ii], s[ii]) for ii in draws]
        out[f"{m}|{k}"] = dict(model=m, k=k, family=r["family"], val_auc=r["val_auc"], auc=a, auc_ci=boot_stat(y, g, s, A.auc), sens90=sens90(y, s),
                               sens90_ci=boot_stat(y, g, s, sens90), auc_one_per_patient_mean=float(np.mean(one)), auc_one_per_patient_sd=float(np.std(one)))
    # advantage rule per feature count: quantum vs best classical at the same k
    cmp = {}
    for kq, qs in ((8, ["vqc_pi6", "bw8", "bw8_s1"]), (12, ["bw12"]), (32, ["vqc_pi6", "vqc_pi6_trained"])):
        cl = {m: out[f"{m}|{kq}"] for m in ("lr", "lr2", "rf", "xgb", "mlp") if f"{m}|{kq}" in out}
        if not cl: continue
        bestc = max(cl, key=lambda m: cl[m]["auc"])
        for q in qs:
            key = f"{q}|{kq}"
            if key not in out: continue
            sa, sb = np.array(latest[(q, kq)]["scores"]), np.array(latest[(bestc, kq)]["scores"])
            bs = A.boot_auc_diff([dict(y=y, g=g, sa=sa, sb=sb)], 2000, seed=3)
            beats_all = all(out[key]["auc"] > v["auc"] for v in cl.values())
            cmp[key] = dict(best_classical=bestc, diff=out[key]["auc"] - cl[bestc]["auc"], diff_ci=bs["diff_ci"], beats_every_classical=bool(beats_all),
                            advantage_rule_ci_met=bool(beats_all and bs["diff_ci"][0] > 0))
    json.dump(dict(models=out, comparisons=cmp, n_records=int(len(y)), n_patients=int(len(u)), n_mi=int(y.sum())), open("results/ext/p4_report.json", "w"), indent=1)
    L = ["| model | k | PTB-XL val AUC | external AUC [95% CI] | one-record-per-patient AUC (mean of 200 draws) | sens @ 90% spec [95% CI] |", "|---|---|---|---|---|---|"]
    for key, v in out.items():
        L.append(f"| {v['model']} | {v['k']} | {v['val_auc']:.4f} | {v['auc']:.4f} [{v['auc_ci'][0]:.3f}, {v['auc_ci'][1]:.3f}] | {v['auc_one_per_patient_mean']:.4f} | {v['sens90']:.3f} [{v['sens90_ci'][0]:.3f}, {v['sens90_ci'][1]:.3f}] |")
    L += ["", "| quantum model | k | best classical at same k | AUC diff [95% CI] | beats every classical | advantage rule (CI only) |", "|---|---|---|---|---|---|"]
    for key, v in cmp.items():
        q, k = key.split("|")
        L.append(f"| {q} | {k} | {v['best_classical']} | {v['diff']:+.4f} [{v['diff_ci'][0]:+.4f}, {v['diff_ci'][1]:+.4f}] | {v['beats_every_classical']} | {v['advantage_rule_ci_met']} |")
    open("results/ext/p4_report.md", "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()
