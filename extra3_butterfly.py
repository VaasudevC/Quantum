"""EXTRA 3: unitary butterfly (paper Sec. III.C; strides 1,2,4 at n=8, all-to-all, depth 2 log n) vs the brick-wall at n=8.
Stage 1 (folds 0-1, validation AUC): k in {2,4} x layers in {3 (one butterfly), 6 (two)} x scale in {pi/6, pi/4}; stage 2: best config, 5 folds.
-> results/bw/extra3_butterfly.jsonl"""
import json, sys, numpy as np
import p3_brickwall as P
def main():
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4; out = "results/bw/extra3_butterfly.jsonl"
    s1 = [P.base(n=8, k=k, columns=L, scale=float(sc), fold=f, arch="butterfly", save_scores=True) for f in (0, 1) for k in (2, 4) for L in (3, 6) for sc in (np.pi / 6, np.pi / 4)]
    P.runner(s1, out, jobs)
    rows = [r for r in P.load_rows(out) if r["fold"] in (0, 1)]
    gm = P.group_mean_val(rows); best = dict(max(gm.items(), key=lambda kv: kv[1])[0]); print("best butterfly config", best, flush=True)
    s2 = [dict(best, fold=f, save_scores=True) for f in range(5)]
    for c in s2: c["arch"] = "butterfly"
    P.runner(s2, out, jobs)


if __name__ == "__main__":
    main()
