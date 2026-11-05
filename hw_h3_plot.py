"""H3 figure + table: noiseless / shots-only / noisy raw / noisy mitigated AUC per model (FakeFez noise model).

  python hw_h3_plot.py   -> results/figs/hw_noisy.png, results/hw/h3_noisy.md
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = Path("results/hw")
rows = [json.loads(l) for l in open(R / "h3_noisy_line8.jsonl")]
cal = json.load(open(R / "h3_calibration.json")) if (R / "h3_calibration.json").exists() else {}
best = [json.loads(l) for l in open(R / "h3_noisy_best12.jsonl")] if (R / "h3_noisy_best12.jsonl").exists() else []
lab = lambda r: f"k={r['k']}\n{r['scale'].replace('pi', 'π')}"
order = sorted(rows, key=lambda r: (r["k"], {"pi": 0, "pi/2": 1, "pi/6": 2}[r["scale"]]))

fig, ax = plt.subplots(figsize=(9, 3.8)); x = np.arange(len(order)); w = 0.2
for i, (c, nm) in enumerate([("noiseless", "noiseless"), ("shots_only", "2000 shots, ideal device"),
                             ("noisy_raw", "FakeFez noise, raw"), ("noisy_mitigated", "FakeFez noise, readout-mitigated")]):
    ax.bar(x + (i - 1.5) * w, [r[c] for r in order], w, label=nm)
ax.set_xticks(x); ax.set_xticklabels([lab(r) for r in order]); ax.set_ylim(0.8, 0.98)
ax.set_ylabel(f"test AUC ({order[0]['n']} stratified fold-0 ECGs)"); ax.legend(fontsize=8, ncol=2, loc="upper left")
ax.set_title("IBM Heron (FakeFez) noise model, 8-qubit line compile")
fig.tight_layout(); fig.savefig("results/figs/hw_noisy.png", dpi=150)

md = ("| model | 2q gates (CZ) | noiseless | 2000 shots, ideal | noisy raw | noisy mitigated | corr P(MI) raw vs noiseless | "
      "⟨Z⟩ shrink | bal. acc noiseless → noisy → rescaled | Brier noiseless → noisy → rescaled |\n|---|---|---|---|---|---|---|---|---|---|\n")
for r in order:
    c = cal.get(f"k{r['k']}_{r['scale']}", {})
    ba = f"{c['z_exact']['bacc']:.3f} → {c['z_mit']['bacc']:.3f} → {c['z_mit_rescaled']['bacc']:.3f}" if c else "–"
    br = f"{c['z_exact']['brier']:.3f} → {c['z_mit']['brier']:.3f} → {c['z_mit_rescaled']['brier']:.3f}" if c else "–"
    md += (f"| k={r['k']} {r['scale']} | {r['n2q']} | {r['noiseless']:.4f} | {r['shots_only']:.4f} | {r['noisy_raw']:.4f} | "
           f"{r['noisy_mitigated']:.4f} | {r['corr_P_raw']:.3f} | {r['z_shrink_raw']:.2f} | {ba} | {br} |\n")
if best:
    md += ("\nBest-compile check (H3a compile: 12 active qubits, fewest CZ), same noise model, "
           f"{best[0]['n']} ECGs (the H4 set), {best[0]['shots']} shots:\n\n"
           "| model | 2q gates | noiseless | noisy raw | noisy mitigated | corr P(MI) raw vs noiseless | ⟨Z⟩ shrink |\n|---|---|---|---|---|---|---|\n")
    for b in best:
        md += (f"| k={b['k']} {b['scale']} | {b['n2q']} | {b['noiseless']:.4f} | {b['noisy_raw']:.4f} | {b['noisy_mitigated']:.4f} | "
               f"{b['corr_P_raw']:.3f} | {b['z_shrink_raw']:.2f} |\n")
(R / "h3_noisy.md").write_text(md); print(md)
