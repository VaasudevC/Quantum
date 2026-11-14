"""Paper-ready tables and figures -> results/paper/.  Every output names the commit it was built from (git HEAD at build time; the inputs are committed files).
  C:/bwvenv/Scripts/python.exe make_paper.py [K4]     (K4 = also write the k=4 rows of Table 3 if results/sd_analysis_k4_main.json exists)"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

OUT = Path("results/paper"); OUT.mkdir(parents=True, exist_ok=True)
HEAD = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
STAMP = f"Built from commit {HEAD} by make_paper.py."
rows = lambda p: [json.loads(l) for l in open(p)] if Path(p).exists() else []
SC = {"pi/8": np.pi / 8, "pi/6": np.pi / 6, "pi/4": np.pi / 4, "pi/3": np.pi / 3, "pi/2": np.pi / 2, "2*pi/3": 2 * np.pi / 3, "pi": np.pi}
ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"


def table1():
    s1 = rows("results/stage1_sweep.jsonl"); s2 = rows("results/stage2_steps1000.jsonl"); A = rows("results/bw/p3_stageA2.jsonl")
    L = ["# Table 1 — Encoding study: angle scale s in angle = s·tanh(z)", "", STAMP, "",
         "**(a) Angle VQC (8 qubits × 4 layers, re-uploading), 5-fold test AUC (mean ± SD), 1000 steps** — `results/stage2_steps1000.jsonl`", "",
         "| scale s | k = 8 | k = 16 | k = 32 |", "|---|---|---|---|"]
    for sc in ("pi", "pi/2", "pi/6"):
        L.append(f"| {sc} | " + " | ".join(ms([r["test_auc"] for r in s2 if r["k"] == k and r["scale"] == sc]) for k in (8, 16, 32)) + " |")
    L.append("| LR on the same features | " + " | ".join(ms([r["lr_auc"] for r in s2 if r["k"] == k and r["scale"] == "pi/6"]) for k in (8, 16, 32)) + " |")
    L += ["", "**(b) Scale sweep at k = 32 (600 steps), 5-fold test AUC** — `results/stage1_sweep.jsonl`", "", "| s | test AUC | val AUC |", "|---|---|---|"]
    for sc in ("pi/8", "pi/6", "pi/4", "pi/3", "pi/2", "2*pi/3", "pi"):
        r = [x for x in s1 if x["scale"] == sc]; L.append(f"| {sc} | {ms([x['test_auc'] for x in r])} | {np.mean([x['val_auc'] for x in r]):.3f} |")
    L += ["", "**(c) Replication in the magic-state brick-wall QNN (our reimplementation): mean validation AUC, folds 0–1, phase φ = s·tanh(z)** — `results/bw/p3_stageA2.jsonl`", "",
          "| configuration | s = π/6 | π/4 | π/2 | π |", "|---|---|---|---|---|"]
    for n, k, c in ((8, 4, 4), (8, 2, 4), (12, 6, 4), (12, 4, 2)):
        vals = []
        for s in (np.pi / 6, np.pi / 4, np.pi / 2, np.pi):
            v = [r["val_auc"] for r in A if r["n"] == n and r["k"] == k and r["columns"] == c and abs(r["scale"] - s) < 1e-9]
            vals.append(f"{np.mean(v):.3f}" if v else "–")
        L.append(f"| n={n}, particles k={k}, {c} columns | " + " | ".join(vals) + " |")
    (OUT / "table1_encoding.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def table2():
    P1 = rows("results/p1_classical.jsonl"); n4 = json.load(open("results/n4_clinical.json")); bw = json.load(open("results/bw/p3_analysis.json"))
    names = dict(lr="LR", lr2="degree-2 LR", rf="Random forest", xgb="XGBoost", mlp="MLP")
    L = ["# Table 2 — Full-data benchmark by feature count (5-fold patient-grouped CV, test AUC mean ± SD)", "", STAMP, "",
         "Sources: `results/p1_classical.jsonl` (classical, hyper-parameters by validation AUC), `results/n4_clinical.json` (angle VQC π/6), `results/bw/p3_analysis.json` (brick-wall finals).", "",
         "| model | k = 8 | k = 12 | k = 16 | k = 32 | all 249 |", "|---|---|---|---|---|---|"]
    for m in ("lr", "lr2", "rf", "xgb", "mlp"):
        cells = []
        for k in (8, 12, 16, 32, 249):
            v = [r["test_auc"] for r in P1 if r["model"] == m and r["selector"] == "l1" and r["k"] == k]
            cells.append(ms(v) if len(v) == 5 else "–")
        L.append(f"| {names[m]} | " + " | ".join(cells) + " |")
    vq = lambda k: (lambda a: f"{a['mean']:.3f} ± {a['std']:.3f}")(n4[f"VQC z, k={k}, pi/6"]["auc"]) if f"VQC z, k={k}, pi/6" in n4 else "–"
    L.append(f"| Angle VQC, s = π/6 | {vq(8)} | – | {vq(16)} | {vq(32)} | – |")
    t = n4["VQC trained scale (N2), k=32"]["auc"]; L.append(f"| Angle VQC, trained scales | – | – | – | {t['mean']:.3f} ± {t['std']:.3f} | – |")
    b8 = bw["8"]["quantum"]["bw8_rank0"]; b12 = bw["12"]["quantum"]["bw12_rank0"]
    L.append(f"| Brick-wall QNN (n qubits = k) | {b8['mean']:.3f} ± {b8['sd']:.3f} | {b12['mean']:.3f} ± {b12['sd']:.3f} | – | – | – |")
    L += ["", "No quantum model beats every tuned classical model at any feature count (advantage rule, `BRICKWALL.md`)."]
    (OUT / "table2_benchmark.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def table3():
    L = ["# Table 3 — Pre-registered small-data benchmark: best quantum − best classical by training size N", "", STAMP, "",
         "Paired over 5 folds × 40 subsets (N = 500: **reduced sample**, 10 subsets); 95% hierarchical-bootstrap CIs (folds, then subsets); one-sided superiority with Holm; non-inferiority margin −0.02. "
         "Source: `results/sd_analysis_k8_main.json` (and `_k4_main.json`). C7 TabPFN not run (licence login required); Q4 brick-wall only at N = 100 (torch blocked at other sizes).", ""]
    for K in (8, 4):
        p = Path(f"results/sd_analysis_k{K}_main.json")
        if not p.exists():
            L += [f"k = {K}: not available (incomplete)", ""]; continue
        r = json.load(open(p)); S = lambda m: {"Q1_vqc": "Q1 angle VQC", "Q2_vqc_trained_scales": "Q2 VQC trained scales", "Q3_qkernel_svm": "Q3 quantum-kernel SVM", "Q4_brickwall8": "Q4 brick-wall",
                                             "C1_lr": "C1 LR", "C2_svm_rbf": "C2 SVM-RBF", "C3_gpc": "C3 GP", "C4_knn": "C4 kNN", "C5_rf": "C5 random forest", "C6_xgb": "C6 XGBoost"}[m]
        L += [f"**k = {K}** (Holm family: {len(r['pairs'])} comparisons)", "", "| N | best quantum | best classical | mean d [95% CI] | quantum wins | superiority | non-inferior | label |", "|---|---|---|---|---|---|---|---|"]
        for b in r["best_vs_best"]:
            lab = "reduced sample" if b["N"] == 500 else ""
            if not b["superiority"] and not b["noninferior_ci"]: lab = (lab + "; " if lab else "") + "inconclusive"
            L.append(f"| {b['N']} | {S(b['best_q'])} | {S(b['best_c'])} | {b['mean_d']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] | {b['win_frac']:.2f} | {'yes' if b['superiority'] else 'no'} | {'yes' if b['noninferior_ci'] else 'no'} | {lab} |")
        L.append("")
    (OUT / "table3_smalldata.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def figures():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
    # Figure 1: scale sweep (k=32, 600 steps) + k-dependence at pi, pi/2, pi/6
    s1 = rows("results/stage1_sweep.jsonl"); s2 = rows("results/stage2_steps1000.jsonl")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    order = ["pi/8", "pi/6", "pi/4", "pi/3", "pi/2", "2*pi/3", "pi"]; x = [SC[s] for s in order]
    m = [np.mean([r["test_auc"] for r in s1 if r["scale"] == s]) for s in order]; sd = [np.std([r["test_auc"] for r in s1 if r["scale"] == s], ddof=1) for s in order]
    ax[0].errorbar(x, m, yerr=sd, marker="o", capsize=3, color="#cc3311", label="angle VQC, k = 32")
    ax[0].axhline(np.mean([r["lr_auc"] for r in s1 if r["scale"] == "pi/6"]), color="#888888", ls="--", label="LR, same 32 features")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["π/8", "π/6", "π/4", "π/3", "π/2", "2π/3", "π"]); ax[0].set_xlabel("angle scale s  (angle = s·tanh z)"); ax[0].set_ylabel("test AUC (5 folds, mean ± SD)")
    ax[0].set_title("(a) scale sweep, k = 32"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for sc, c in (("pi", "#882255"), ("pi/2", "#ee7733"), ("pi/6", "#cc3311")):
        ks = (8, 16, 32); mm = [np.mean([r["test_auc"] for r in s2 if r["k"] == k and r["scale"] == sc]) for k in ks]; ss = [np.std([r["test_auc"] for r in s2 if r["k"] == k and r["scale"] == sc], ddof=1) for k in ks]
        ax[1].errorbar(ks, mm, yerr=ss, marker="o", capsize=3, color=c, label=f"s = {sc.replace('pi', 'π')}")
    ax[1].set_xticks((8, 16, 32)); ax[1].set_xlabel("re-uploaded features k"); ax[1].set_title("(b) AUC vs k"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.text(0.01, 0.005, STAMP, fontsize=6, color="#666666"); plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(OUT / "fig1_scale_sweep.png", dpi=200); plt.close()
    # Figure 2: learning curves (SD, k=8)
    r = json.load(open("results/sd_analysis_k8_main.json")); col = {"Q1_vqc": "#cc3311", "Q2_vqc_trained_scales": "#ee7733", "Q3_qkernel_svm": "#aa3377", "Q4_brickwall8": "#882255", "C1_lr": "#888888",
         "C2_svm_rbf": "#228833", "C3_gpc": "#009988", "C4_knn": "#bbbb33", "C5_rf": "#4477aa", "C6_xgb": "#66ccee"}
    lab = {"Q1_vqc": "Q1 angle VQC", "Q2_vqc_trained_scales": "Q2 VQC, trained scales", "Q3_qkernel_svm": "Q3 quantum-kernel SVM", "Q4_brickwall8": "Q4 brick-wall (N=100 only)", "C1_lr": "C1 LR",
           "C2_svm_rbf": "C2 SVM-RBF", "C3_gpc": "C3 GP", "C4_knn": "C4 kNN", "C5_rf": "C5 random forest", "C6_xgb": "C6 XGBoost"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for mname in lab:
        xs = [N for N in (25, 50, 100, 200, 500) if f"{mname}|{N}" in r["mean_auc"]]
        if not xs: continue
        y = [r["mean_auc"][f"{mname}|{N}"]["mean"] for N in xs]; lo = [r["mean_auc"][f"{mname}|{N}"]["ci"][0] for N in xs]; hi = [r["mean_auc"][f"{mname}|{N}"]["ci"][1] for N in xs]
        q = mname.startswith("Q")
        ax[0].plot(xs, y, marker="o" if len(xs) > 1 else "s", ms=4, lw=2 if q else 1.1, ls="-" if q else "--", color=col[mname], label=lab[mname])
        ax[0].fill_between(xs, lo, hi, color=col[mname], alpha=0.12)
    ax[0].set_xscale("log"); ax[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax[0].set_xticks((25, 50, 100, 200, 500)); ax[0].set_xticklabels(["25", "50", "100", "200", "500*"])
    ax[0].set_xlabel("training ECGs N (one per patient; *reduced sample)"); ax[0].set_ylabel("test AUC (mean, 95% CI)"); ax[0].set_title("(a) learning curves, k = 8"); ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)
    bb = r["best_vs_best"]; xs = [b["N"] for b in bb]
    ax[1].errorbar(xs, [b["mean_d"] for b in bb], yerr=[[b["mean_d"] - b["ci"][0] for b in bb], [b["ci"][1] - b["mean_d"] for b in bb]], marker="o", capsize=3, color="#cc3311")
    ax[1].axhline(0, color="k", lw=1); ax[1].axhline(-0.02, color="k", lw=1, ls="--"); ax[1].set_xscale("log"); ax[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{x}*" if x == 500 else str(x) for x in xs])
    ax[1].set_xlabel("training ECGs N"); ax[1].set_ylabel("best quantum − best classical"); ax[1].set_title("(b) paired difference (dashed: non-inferiority margin)"); ax[1].grid(alpha=.3)
    fig.text(0.01, 0.005, STAMP + " C7 TabPFN not run; Q4 only at N = 100.", fontsize=6, color="#666666"); plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(OUT / "fig2_learning_curves.png", dpi=200); plt.close()
    # Figure 3: geometric difference vs sqrt(N)
    g = json.load(open("results/dx_x2_geometry.json")); fig, ax = plt.subplots(figsize=(6, 4.4))
    for lam, mk in ((1e-3, "o"), (1e-2, "s"), (1e-1, "^"), (1e-4, "x")):
        for N in (100, 500):
            vals = [v[f"g_rbf_lam{lam}"] for v in g.values() if v["N"] == N]
            ax.scatter([N] * len(vals), vals, marker=mk, color="#cc3311" if lam != 1e-4 else "#888888", label=f"g(K_RBF‖K_Q), λ = {lam:g}" if N == 100 else None, zorder=3)
    Ns = np.array([80, 600]); ax.plot(Ns, np.sqrt(Ns), "k-", label="√N")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_xticks((100, 500)); ax.set_xticklabels(["100", "500"]); ax.set_xlabel("N (ECGs, discovery folds 0–2)"); ax.set_ylabel("geometric difference g")
    ax.set_title("π/6 angle kernel vs tuned RBF kernel (Huang et al. 2021)"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    fig.text(0.01, 0.005, STAMP + " Exploratory; λ = 1e-4 (grey) = nearly unregularised RBF Gram.", fontsize=6, color="#666666"); plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(OUT / "fig3_geometric_difference.png", dpi=200); plt.close()


def table4():
    if not Path("results/qb1_table.md").exists(): return
    t = open("results/qb1_table.md", encoding="utf-8").read(); gate = json.load(open("results/qb1_gate.json"))
    L = ["# Table 4 — Training-free screen of quantum feature maps (QB1, pre-registered in QB.md; discovery folds 0–2, k = 8)", "", STAMP, "",
         "F0 product map, F1 re-uploading (L loads, CNOT ring), F2 IQP/ZZ (ring), F3 IQP with clinically paired RZZ, F4 Heisenberg-ring evolution (T Trotter steps), F5 brick-wall encoding state. "
         "g = geometric difference to a CV-tuned RBF kernel (Huang et al. 2021). Gate (N = 500, k = 8, ≥ 2 of 3 folds): g ≥ √N/2 at λ ∈ {1e-3, 1e-2, 1e-1}, mean off-diagonal > 0.05, centered alignment ≥ 90 % of RBF.",
         f"**Maps passing the gate: {len(gate['passed'])}.** Source `results/qb1_screen.jsonl`.", "", t]
    (OUT / "table4_feature_maps.md").write_text(chr(10).join(L) + chr(10), encoding="utf-8")


def fig4():
    if not Path("results/qb1_screen.jsonl").exists(): return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
    R = rows("results/qb1_screen.jsonl"); R = [r for r in R if r["k"] == 8]
    mk = {"F0": "o", "F1": "s", "F2": "^", "F3": "v", "F4": "D", "F5": "P"}; cs = {"pi/8": "#4477aa", "pi/6": "#228833", "pi/4": "#ccbb44", "pi/3": "#ee6677", "pi/2": "#aa3377"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for r in R:
        jit = np.exp(np.random.default_rng(hash((r["map"], r["params"], r["scale"], r["fold"])) % 2**32).normal(0, 0.04))
        ax[0].scatter(r["N"] * jit, r["fid"]["g_lam0.01"], marker=mk[r["map"]], color=cs[r["scale"]], s=14, alpha=0.6)
        if r["N"] == 500:
            ax[1].scatter(r["fid"]["calign"] / r["rbf"]["calign"], r["fid"]["g_lam0.01"], marker=mk[r["map"]], color=cs[r["scale"]], s=16, alpha=0.7)
    Ns = np.array([80, 620]); ax[0].plot(Ns, np.sqrt(Ns), "k-", label="√N"); ax[0].plot(Ns, np.sqrt(Ns) / 2, "k--", label="√N / 2 (gate)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax[0].set_xticks((100, 500)); ax[0].set_xticklabels(["100", "500"])
    ax[0].set_xlabel("N (discovery ECGs)"); ax[0].set_ylabel("g(K_RBF‖K_Q), λ = 1e-2"); ax[0].set_title("(a) all maps × scales × folds"); ax[0].grid(alpha=.3, which="both")
    ax[1].axhline(np.sqrt(500) / 2, color="k", ls="--"); ax[1].axvline(0.9, color="k", ls=":"); ax[1].set_yscale("log")
    ax[1].set_xlabel("centered alignment, quantum / RBF (N = 500)"); ax[1].set_ylabel("g, λ = 1e-2"); ax[1].set_title("(b) expressivity vs label alignment (gate: upper right)"); ax[1].grid(alpha=.3, which="both")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk[m], ls="", color="k", label=m) for m in mk] + [Line2D([], [], marker="o", ls="", color=cs[s], label=s.replace("pi", "π")) for s in cs]
    ax[0].legend(handles=h + ax[0].get_legend_handles_labels()[0], fontsize=7, ncol=2)
    fig.text(0.01, 0.005, STAMP + " Pre-registered QB1 screen; no map passed the gate.", fontsize=6, color="#666666"); plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(OUT / "fig4_g_all_maps.png", dpi=200); plt.close()


def table5():
    p = Path("results/hw/ibm_bw_hw/extended_analysis.json")
    if not p.exists():
        return
    r = json.load(open(p))
    def fmt(key, mr):
        v = mr.get(key)
        ci = mr.get(key + "_ci")
        if v is None: return "–"
        if ci: return f"{v:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        return f"{v:.4f}"
    L = ["# Table 5 — Real hardware results: ibm_marrakesh (Heron r2), 200 ECGs, 1000 shots, 2026-11-14", "", STAMP, "",
         "Source: `results/hw/ibm_bw_hw/extended_analysis.json`.",
         "Bootstrap CIs: B=2000, seed=0, percentile method. Noiseless AUC from exact simulation.",
         "2.2 RU consumed (ibm_marrakesh billing). bw8: 50% of shots violated k=4 particle number; postselection left AUC unchanged.", "",
         "## AUC with 95% bootstrap CIs and paired difference (hardware − noiseless)", "",
         "| model | circuit (CZ, depth) | AUC noiseless [95% CI] | AUC raw [95% CI] | AUC postselected [95% CI] | kept fraction | paired Δ [95% CI] |",
         "|---|---|---|---|---|---|---|"]
    circuits = {"bw8": "136 CZ, depth 254", "vqc32": "80 CZ, depth 208"}
    for mname in ("bw8", "vqc32"):
        if mname not in r: continue
        mr = r[mname]
        kf = f"{mr['kept_fraction_mean']:.3f} ± {mr['kept_fraction_std']:.3f}" if "kept_fraction_mean" in mr else "–"
        d = mr.get("diff_raw_minus_noiseless", float("nan"))
        ci = mr.get("diff_raw_minus_noiseless_ci", [float("nan"), float("nan")])
        paired = f"{d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        L.append(f"| {mname} | {circuits[mname]} | {fmt('auc_noiseless', mr)} | {fmt('auc_raw', mr)} | {fmt('auc_postselected', mr)} | {kf} | {paired} |")
    L += ["", "## Per-ECG agreement (hardware vs noiseless logit)", "",
          "| model | Pearson r | Spearman ρ | MAE logit | contraction factor |",
          "|---|---|---|---|---|"]
    for mname in ("bw8", "vqc32"):
        if mname not in r: continue
        mr = r[mname]
        L.append(f"| {mname} | {mr.get('pearson_r_raw', float('nan')):.3f} | {mr.get('spearman_rho_raw', float('nan')):.3f} | {mr.get('mae_logit_raw', float('nan')):.3f} | {mr.get('contraction_factor_raw', float('nan')):.3f} |")
    L += ["", "## Calibration (at hardware-optimal balanced-accuracy threshold)", "",
          "| model | bal. acc | sens | spec | acc |",
          "|---|---|---|---|---|"]
    for mname in ("bw8", "vqc32"):
        if mname not in r: continue
        cal = r[mname].get("calibration", {}).get("hw_raw_at_hw_thr", {})
        if cal:
            L.append(f"| {mname} hw raw | {cal.get('bal_acc', float('nan')):.3f} | {cal.get('sens', float('nan')):.3f} | {cal.get('spec', float('nan')):.3f} | {cal.get('acc', float('nan')):.3f} |")
        cal_nl = r[mname].get("calibration", {}).get("noiseless", {})
        if cal_nl:
            L.append(f"| {mname} noiseless | {cal_nl.get('bal_acc', float('nan')):.3f} | {cal_nl.get('sens', float('nan')):.3f} | {cal_nl.get('spec', float('nan')):.3f} | {cal_nl.get('acc', float('nan')):.3f} |")
    L += ["", "Note: thresholds tuned on the same test ECGs (optimistic; not held-out validation). The 200-ECG test set has ~±0.01 AUC sampling error.",
          "bw8 paired Δ excludes 0 (CI [+0.003, +0.046]): hardware ranked ECGs better than noiseless simulation; likely a noise-induced shift in the decision function, not a reliable improvement (one run, one fold)."]
    (OUT / "table5_hardware.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    table1(); table2(); table3(); figures(); table4(); fig4(); table5()
    (OUT / "README.md").write_text(f"# results/paper\n\n{STAMP}\n\n* `table1_encoding.md` — encoding study (angle scale; brick-wall replication)\n* `table2_benchmark.md` — full-data benchmark by feature count\n"
                                   f"* `table3_smalldata.md` — pre-registered small-data differences with CIs by N\n* `fig1_scale_sweep.png` — scale sweep and AUC vs k\n* `fig2_learning_curves.png` — SD learning curves and best-vs-best difference\n"
                                   f"* `fig3_geometric_difference.png` — g(K_RBF‖K_Q) vs √N (exploratory)\n* `table5_hardware.md` — real hardware results (ibm_marrakesh, 2026-11-14)\n"
                                   f"\nRebuild: `../.venv/Scripts/python.exe make_paper.py`\n", encoding="utf-8")
    print("ok", HEAD)
