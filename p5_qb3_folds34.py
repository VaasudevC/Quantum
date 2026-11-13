"""Phase 5 (Exploratory, added 2026-11-14): run QB3 nonlinearity analysis on folds 3 and 4.
Logged in QB.md under 'Exploratory (not pre-registered) — added 2026-11-14' BEFORE running.
Reuses all functions from qb3_nonlinearity.py; only the fold-guard assertion is bypassed.
  ..\\.venv\\Scripts\\python.exe p5_qb3_folds34.py FOLD   (FOLD in {3, 4})
  -> appends one line to results/qb3.jsonl"""
import json, sys, time
from pathlib import Path
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

# Import all helpers from the pre-registered QB3 script (do not modify qb3_nonlinearity.py)
import importlib.util
spec = importlib.util.spec_from_file_location("qb3_nonlinearity", "qb3_nonlinearity.py")
qb3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qb3)

import numpy as np
import scale_cv as S

OUT = Path("results/qb3.jsonl")


def main():
    fold = int(sys.argv[1])
    assert fold in (3, 4), "This script is for exploratory folds 3 and 4 only"

    # Skip if already done
    for l in OUT.open():
        d = json.loads(l)
        if d.get("fold") == fold:
            print(f"fold {fold} already in {OUT}, skipping")
            return

    X, y, _, _, feats = S.load()
    Z, tr, va, te, _ = S.prep("grouped5", fold, "l1")
    ra, rb = qb3.vqc_rows(fold)
    sel = [feats.index(n) for n in ra["features"]]
    selb = [feats.index(n) for n in rb["features"]]

    t0 = time.time()
    rec = dict(fold=fold)
    la, Aa = qb3.vqc_logit(ra, Z, sel)
    lb, Ab = qb3.vqc_logit(rb, Z, selb)
    rec["vqc_pi6"] = dict(test=qb3.auc(y[te], la[te]), params=81,
                           surrogate=qb3.surrogate(la, Aa, tr, te))
    rec["vqc_trained"] = dict(test=qb3.auc(y[te], lb[te]), params=113,
                               surrogate=qb3.surrogate(lb, Ab, tr, te))
    Zs = Z[:, sel]
    rec["features"] = ra["features"]
    rec["classical_z"] = qb3.models(Zs, y, tr, va, te)
    rec["classical_bounded_twin"] = qb3.models((np.pi / 6) * np.tanh(Zs), y, tr, va, te)
    rec["secs"] = round(time.time() - t0, 1)

    with open(OUT, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    print(fold, "done", rec["secs"],
          {k: round(v["test"], 4) for k, v in rec["classical_z"].items()},
          round(rec["vqc_pi6"]["test"], 4),
          round(rec["vqc_trained"]["test"], 4), flush=True)


if __name__ == "__main__":
    main()
