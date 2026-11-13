"""Phase 5 (Exploratory, added 2026-11-14): 5-fold QB3 summary table.
Reads results/qb3.jsonl (must have all 5 folds) and writes
results/qb3_5fold_table.md.
  ..\\.venv\\Scripts\\python.exe p5_qb3_extended.py"""
import json
from pathlib import Path

OUT = Path("results/qb3_5fold_table.md")


def fmt(v):
    return f"{v:.4f}"


def main():
    rows = {}
    for l in open("results/qb3.jsonl"):
        d = json.loads(l)
        rows[d["fold"]] = d

    if len(rows) < 5:
        missing = [f for f in range(5) if f not in rows]
        print(f"WARNING: only {len(rows)} folds present; missing: {missing}")

    folds = sorted(rows)

    # --- Table 1: test AUC per fold for each model (z-scored inputs) ---
    model_keys_z = ["LR", "FM_rank1", "FM_rank2", "MLP_32-2-1", "MLP_32-3-1", "L1_deg2_LR_le81"]
    header = "| fold | VQC_pi6 | VQC_tr | " + " | ".join(model_keys_z) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(model_keys_z))) + "|"

    table1_rows = [header, sep]
    for f in folds:
        d = rows[f]
        tag = " *(exp)*" if f >= 3 else ""
        vals = [fmt(d["vqc_pi6"]["test"]), fmt(d["vqc_trained"]["test"])]
        vals += [fmt(d["classical_z"][m]["test"]) for m in model_keys_z]
        table1_rows.append(f"| {f}{tag} | " + " | ".join(vals) + " |")

    # --- Table 2: surrogate R² (deg-2 vs linear) ---
    hdr2 = "| fold | VQC_pi6 R²_deg2 | R²_lin | eff_rank | diag_share | VQC_tr R²_deg2 | R²_lin | eff_rank | diag_share |"
    sep2 = "|" + "|".join(["---"] * 9) + "|"
    table2_rows = [hdr2, sep2]
    for f in folds:
        d = rows[f]
        s1 = d["vqc_pi6"]["surrogate"]
        s2 = d["vqc_trained"]["surrogate"]
        tag = " *(exp)*" if f >= 3 else ""
        row = (f"| {f}{tag} | {fmt(s1['r2_deg2'])} | {fmt(s1['r2_linear'])} | "
               f"{s1['eff_rank']:.1f} | {fmt(s1['diag_share'])} | "
               f"{fmt(s2['r2_deg2'])} | {fmt(s2['r2_linear'])} | "
               f"{s2['eff_rank']:.1f} | {fmt(s2['diag_share'])} |")
        table2_rows.append(row)

    # --- Bounded twin table ---
    hdr3 = "| fold | VQC_pi6 | VQC_tr | " + " | ".join(model_keys_z) + " |"
    sep3 = "|" + "|".join(["---"] * (2 + len(model_keys_z))) + "|"
    table3_rows = [hdr3 + " *(bounded inputs)*", sep3]
    for f in folds:
        d = rows[f]
        tag = " *(exp)*" if f >= 3 else ""
        vals = [fmt(d["vqc_pi6"]["test"]), fmt(d["vqc_trained"]["test"])]
        vals += [fmt(d["classical_bounded_twin"][m]["test"]) for m in model_keys_z]
        table3_rows.append(f"| {f}{tag} | " + " | ".join(vals) + " |")

    lines = [
        "# QB3 nonlinearity — 5-fold descriptive table (exploratory: folds 3–4)",
        "",
        "> Folds 0–2 are pre-registered (QB.md). Folds 3–4 are exploratory (logged 2026-11-14 BEFORE running).",
        "> No benefit claim is made from this extension; it is purely descriptive.",
        "",
        "## Test AUC (z-scored inputs)",
        "",
    ] + table1_rows + [
        "",
        "## Degree-2 surrogate fit",
        "",
    ] + table2_rows + [
        "",
        "## Test AUC (bounded inputs: (π/6)·tanh(z))",
        "",
    ] + table3_rows + [""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written -> {OUT}")
    for line in table1_rows:
        print(line)


if __name__ == "__main__":
    main()
