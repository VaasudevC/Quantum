"""P3 analysis: brick-wall finals vs every tuned classical baseline (P1) with the advantage rule.
Rule: quantum model OUTPERFORMS on a feature set iff (i) mean 5-fold test AUC > every classical baseline (LR, LR2, RF, XGB, MLP),
(ii) wins in >= 4 of 5 folds against the best classical one, (iii) paired patient-cluster bootstrap 95% CI (2000 resamples) of the AUC difference
vs the best classical excludes 0.   -> results/bw/p3_analysis.json, results/bw/p3_analysis.md
"""
import json, sys
from pathlib import Path
import numpy as np
import analysis_utils as A

CACHE = "C:/bwcache"
def fold_info(fold):
    d = np.load(f"{CACHE}/l1_f{fold}.npz"); te = d["te"]; return d["y"][te], d["g"][te]

def rows(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []

def main(tag="stageC", out="p3_analysis"):
    C = rows(f"results/bw/p3_{tag}.jsonl"); P1 = rows("results/p1_classical.jsonl")
    vq = json.load(open("results/bw/vqc_scores.json")) if Path("results/bw/vqc_scores.json").exists() else {}
    info = [fold_info(f) for f in range(5)]
    res = {}
    for n in (8, 12):
        cls = {}
        for m in ("lr", "lr2", "rf", "xgb", "mlp"):
            rs = {r["fold"]: r for r in P1 if r["selector"] == "l1" and r["k"] == n and r["model"] == m}
            if len(rs) == 5: cls[m] = [np.array(rs[f]["test_scores"]) for f in range(5)]
        q = {}
        for rank in (0, 1):
            rs = {r["fold"]: r for r in C if r["n"] == n and r.get("rank") == rank}
            if len(rs) == 5:
                q[f"bw{n}_rank{rank}"] = ([np.array(rs[f]["test_scores"]) for f in range(5)], {kk: rs[0][kk] for kk in ("columns", "k", "scale", "head", "lr", "steps")})
        if n == 8 and "vqc_pi6_k8" in vq: q["angle_vqc_k8"] = ([np.array(s) for s in vq["vqc_pi6_k8"]], {})
        aucs = lambda sc: [A.auc(info[f][0], sc[f]) for f in range(5)]
        cl_auc = {m: aucs(s) for m, s in cls.items()}
        best_cl = max(cl_auc, key=lambda m: np.mean(cl_auc[m]))
        res[str(n)] = dict(classical={m: dict(mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)), folds=a) for m, a in cl_auc.items()}, best_classical=best_cl, quantum={})
        for name, (sc, meta) in q.items():
            a = aucs(sc); wins = int(sum(a[f] > cl_auc[best_cl][f] for f in range(5)))
            folds = [dict(y=info[f][0], g=info[f][1], sa=sc[f], sb=cls[best_cl][f]) for f in range(5)]
            bs = A.boot_auc_diff(folds, 2000, seed=1)
            beat_all = all(np.mean(a) > np.mean(v) for v in cl_auc.values())
            res[str(n)]["quantum"][name] = dict(meta=meta, mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)), folds=a, wins_vs_best_classical=wins,
                diff_vs_best_classical=float(np.mean(a) - np.mean(cl_auc[best_cl])), diff_ci=bs["diff_ci"], beats_every_classical_mean=bool(beat_all),
                advantage_rule_met=bool(beat_all and wins >= 4 and bs["diff_ci"][0] > 0))
    json.dump(res, open(f"results/bw/{out}.json", "w"), indent=1)
    L = ["| features | model | mean test AUC ± SD | wins vs best classical (folds) | AUC diff vs best classical [95% cluster-bootstrap CI] | beats every classical (mean) | advantage rule met |", "|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        for m, v in r["classical"].items():
            L.append(f"| {n} | {m} (classical) | {v['mean']:.4f} ± {v['sd']:.4f} | – | – | – | – |")
        for m, v in r["quantum"].items():
            L.append(f"| {n} | **{m}** {v['meta']} | {v['mean']:.4f} ± {v['sd']:.4f} | {v['wins_vs_best_classical']}/5 | {v['diff_vs_best_classical']:+.4f} [{v['diff_ci'][0]:+.4f}, {v['diff_ci'][1]:+.4f}] vs {r['best_classical']} | {v['beats_every_classical_mean']} | {v['advantage_rule_met']} |")
    open(f"results/bw/{out}.md", "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main(*(sys.argv[1:3]))
