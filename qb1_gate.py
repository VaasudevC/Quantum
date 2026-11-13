"""QB1 gate + table + registry rows (pre-registered rule in QB.md)."""
import json, re, collections
import numpy as np
R = [json.loads(l) for l in open("results/qb1_screen.jsonl")]
key = lambda r: (r["map"], re.sub(r",pairs=\d+", "", r["params"]), r["scale"])
G = collections.defaultdict(list)
for r in R: G[key(r) + (r["N"], r["k"])].append(r)
gate = {}
for (mp, pa, sc, N, k), rs in G.items():
    if N != 500 or k != 8: continue
    ok = 0
    for r in rs:
        f = r["fid"]; th = r["sqrtN"] / 2
        if all(f[f"g_lam{l}"] >= th for l in (1e-3, 1e-2, 1e-1)) and f["off_mean"] > 0.05 and f["calign"] >= 0.9 * r["rbf"]["calign"]: ok += 1
    gate[(mp, pa, sc)] = ok
passed = sorted([s for s, ok in gate.items() if ok >= 2], key=lambda s: -np.mean([r["fid"]["calign"] for r in G[s + (500, 8)]]))[:4]
order = ["pi/8", "pi/6", "pi/4", "pi/3", "pi/2"]
L = ["| map | params | s | N | g(RBF‖Q) λ=1e-3 / 1e-2 / 1e-1 (mean of 3 folds) | √N/2 | mean off-diag | centered alignment Q / RBF | CV AUC Q / RBF | proj. kernel CV AUC | gate folds (N=500,k=8) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
reg = []
for (mp, pa, sc) in sorted({s for s in gate}, key=lambda s: (s[0], s[1], order.index(s[2]))):
    for N in (100, 500):
        rs = G[(mp, pa, sc, N, 8)]
        m = lambda f: float(np.mean([f(r) for r in rs]))
        gs = [m(lambda r, l=l: r["fid"][f"g_lam{l}"]) for l in (1e-3, 1e-2, 1e-1)]
        pj = [r["proj"]["cv_auc"] for r in rs if r["proj"]]
        L.append(f"| {mp} | {pa or '–'} | {sc} | {N} | {gs[0]:.1f} / {gs[1]:.1f} / {gs[2]:.1f} | {np.sqrt(N) / 2:.1f} | {m(lambda r: r['fid']['off_mean']):.3f} | {m(lambda r: r['fid']['calign']):.3f} / {m(lambda r: r['rbf']['calign']):.3f} | "
                 f"{m(lambda r: r['fid']['cv_auc']):.3f} / {m(lambda r: r['rbf']['cv_auc']):.3f} | {np.mean(pj):.3f} | {gate[(mp, pa, sc)] if N == 500 else ''} |" if pj else
                 f"| {mp} | {pa or '–'} | {sc} | {N} | {gs[0]:.1f} / {gs[1]:.1f} / {gs[2]:.1f} | {np.sqrt(N) / 2:.1f} | {m(lambda r: r['fid']['off_mean']):.3f} | {m(lambda r: r['fid']['calign']):.3f} / {m(lambda r: r['rbf']['calign']):.3f} | "
                 f"{m(lambda r: r['fid']['cv_auc']):.3f} / {m(lambda r: r['rbf']['cv_auc']):.3f} | degenerate | {gate[(mp, pa, sc)] if N == 500 else ''} |")
        reg.append(f"QB1,{mp}_{pa.replace(',', ';')}_{sc}_N{N},{mp},{sc},{pa.replace(',', ';')},{N},8,0-2,g1e-2={gs[1]:.2f};offmean={m(lambda r: r['fid']['off_mean']):.3f};calign={m(lambda r: r['fid']['calign']):.3f};cvauc={m(lambda r: r['fid']['cv_auc']):.3f},"
                   f"{'yes' if (mp, pa, sc) in passed else 'no'}{'' if N == 500 else ' (gate at N=500)'},gate folds {gate[(mp, pa, sc)]}/3")
open("results/qb1_table.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
open("results/qb_registry.csv", "a").write("\n".join(reg) + "\n")
json.dump(dict(passed=[list(p) for p in passed], gate_folds={"|".join(k): v for k, v in gate.items()}), open("results/qb1_gate.json", "w"), indent=1)
print("passed:", passed)
print("settings with >=1 gate fold:", {k: v for k, v in gate.items() if v >= 1})
# which criterion fails most
fails = collections.Counter()
for (mp, pa, sc, N, k), rs in G.items():
    if N == 500 and k == 8:
        for r in rs:
            f = r["fid"]
            fails["g"] += not all(f[f"g_lam{l}"] >= r["sqrtN"] / 2 for l in (1e-3, 1e-2, 1e-1)); fails["offdiag"] += not f["off_mean"] > 0.05; fails["align"] += not f["calign"] >= 0.9 * r["rbf"]["calign"]
print("criterion failures (of", sum(1 for (a, b, c, N, k) in G if N == 500 and k == 8) * 3, "fold-settings):", dict(fails))
