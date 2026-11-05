"""Summarise one or more scale_cv.py result files.

  python summarize.py results/*.jsonl            -> mean/SD test AUC per config, + Markdown table
  python summarize.py results/*.jsonl --by scale -> pivot over one column
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--md", default="")
a = ap.parse_args()
rows = [json.loads(l) for f in a.files for l in open(f)]
r = pd.DataFrame(rows).drop(columns=["features", "params"], errors="ignore")
defaults = {"selector": "l1", "split": "grouped5", "steps": 300, "seed": 0, "shuffle_later": False,
            "readout": "z", "train_scale": False}
for c, v in defaults.items():   # older pilot rows lack these fields
    r[c] = r[c].fillna(v) if c in r else v
r["steps"] = r["steps"].astype(int)
r["scale_val"] = [eval(s, {"pi": np.pi}) for s in r["scale"]]
# fraction of runs whose best (val-AUC) checkpoint is the final training step
r["at_last"] = (r["best_step"] == r["steps"]).astype(float)
grp = ["split", "selector", "readout", "train_scale", "shuffle_later", "steps", "k", "scale"]
s = (r.groupby(grp).agg(n=("test_auc", "size"), auc=("test_auc", "mean"), sd=("test_auc", "std"),
                        lr=("lr_auc", "mean"), last_step_frac=("at_last", "mean"),
                        scale_val=("scale_val", "first"))
       .reset_index().sort_values(grp[:-1] + ["scale_val"]))
s["gap_to_LR"] = s["auc"] - s["lr"]
pd.set_option("display.width", 200)
print(s.drop(columns="scale_val").round(4).to_string(index=False))
md = "| split | selector | readout | trained scale | shuffle | steps | k | scale | n | VQC AUC | LR AUC | gap |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
for _, x in s.iterrows():
    md += (f"| {x.split} | {x.selector} | {x.readout} | {int(x.train_scale)} | {int(x.shuffle_later)} | {x.steps} | {x.k} | {x.scale} | {x.n} | "
           f"{x.auc:.3f} ± {0 if np.isnan(x.sd) else x.sd:.3f} | {x.lr:.3f} | {x.gap_to_LR:+.3f} |\n")
if a.md:
    Path(a.md).write_text(md); print(f"\nMarkdown -> {a.md}")
