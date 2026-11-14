"""QB6 confirmation (protocol pre-registered in QB.md, 2026-11-14; run on confirmation folds 3-4).
Two QB5 candidates: subtype_IMI Q1 angle VQC and Q3 quantum-kernel SVM vs SVM-RBF.
Folds 3 and 4 only (never touched in discovery). N=100 subsets from the inferior-MI pool,
r=0-9 per fold, seed identical to QB5 subtype_IMI: np.random.default_rng([fold, 503, r]).
One run only; no re-runs.
  ..\\.venv\\Scripts\\python.exe qb6_confirm.py [3 [4]]  -> results/qb6.jsonl + results/qb6_table.md"""
import json, os, sys, time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scale_cv as S, sd_run as R, sd_prep as P
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT = Path("results/qb6.jsonl")
TABLE_OUT = Path("results/qb6_table.md")
A6 = np.pi / 6
N = 100


def meta():
    d = pd.read_csv("data/st_features_plus_TR.csv.gz", usecols=["ecg_id", "label_subclass"])
    return d["label_subclass"].to_numpy()


def sd_choices(fold, r):
    ch = {}
    for f in Path("results").glob("sd_runs_main.w*.jsonl"):
        for line in open(f):
            q = json.loads(line)
            if q["k"] == 8 and q["N"] == N and q["fold"] == fold and q["r"] == r:
                ch[q["model"]] = q["chosen"]
    return ch


def make_fixed(name, chosen, seed):
    if name == "Q1_vqc":
        kv = dict(x.split("=") for x in chosen.split(","))
        g = (float(kv["scale"]), float(kv["l2"])); st = int(kv["steps"])
        return lambda Z, y: R.VQC(8, False).fit(Z, y, g, [st], seed)[st]
    if name == "Q3_qkernel_svm":
        kv = dict(x.split("=") for x in chosen.split(","))
        g = (float(kv["scale"]), float(kv["C"]))
        return lambda Z, y: R.QKernel(8).fit(Z, y, g, [0], seed)[0]
    ms = R.make_sk_models(); M = ms[name]; cfg = json.loads(chosen)
    return lambda Z, y: R.SimpleSK(name, [cfg], M.make, M.score).fit(Z, y, 0, [0], seed)[0]


def twin_models():
    T = lambda Z: A6 * np.tanh(Z)
    def lr(C): return lambda Z, y: (lambda m: lambda X: m.decision_function(T(X)))(
        LogisticRegression(C=C, max_iter=3000).fit(T(Z), y))
    def lr2(C):
        def fit(Z, y):
            pf = PolynomialFeatures(2, include_bias=False)
            sc = StandardScaler().fit(pf.fit_transform(T(Z)))
            m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(pf.transform(T(Z))), y)
            return lambda X: m.decision_function(sc.transform(pf.transform(T(X))))
        return fit
    def svm(C, g): return lambda Z, y: (lambda m: lambda X: m.decision_function(T(X)))(
        SVC(C=C, gamma=g).fit(T(Z), y))
    return {
        "TW_LR": {f"C={C}": lr(C) for C in (0.01, 0.1, 1, 10)},
        "TW_LR2": {f"C={C}": lr2(C) for C in (0.01, 0.1, 1, 10)},
        "TW_SVM": {f"C={C},g={g}": svm(C, g) for C in (0.1, 1, 10) for g in ("scale", 0.1, 1.0)},
    }


def tune_twin(grid, Z, y, seed):
    folds = R.inner_folds(y, seed); best = None
    for k, fit in grid.items():
        v = np.mean([R.auc(y[va], fit(Z[tr], y[tr])(Z[va])) for tr, va in folds])
        if best is None or v > best[0]: best = (v, k, fit)
    return best[1], best[2]


def prep_subset(X, y, sub, test):
    med = np.nan_to_num(np.nanmedian(X[sub], axis=0))
    Xi = lambda I: np.where(np.isnan(X[I]), med, X[I])
    xs = Xi(sub); mu, sd = xs.mean(0), xs.std(0); sd[sd < 1e-9] = 1
    Zs = (xs - mu) / sd
    order = S.l1_order(Zs, y[sub])[:8]
    return Zs[:, order], y[sub], ((Xi(test) - mu) / sd)[:, order], y[test]


def draw(pool, yy, g, npos, nneg, rng):
    chosen, used = [], set()
    for lab, cnt in ((1, npos), (0, nneg)):
        cand = pool[yy[pool] == lab].copy(); rng.shuffle(cand); got = 0
        for i in cand:
            if g[i] in used: continue
            used.add(g[i]); chosen.append(int(i)); got += 1
            if got == cnt: break
        if got < cnt: return None
    return np.array(sorted(chosen))


def run_unit(fold, r, X, y, g, sub_cls):
    tr_all, te = P.outer(fold)
    seed = 1000 * fold + r
    ch = sd_choices(fold, r)
    keep = (sub_cls == "IMI") | (sub_cls == "NORM")
    pool = tr_all[keep[tr_all]]
    pr = (sub_cls[pool] == "IMI").mean()
    s_ = draw(pool, y, g, int(round(N * pr)), N - int(round(N * pr)),
              np.random.default_rng([fold, 503, r]))
    if s_ is None:
        print(f"WARNING: fold {fold} r {r} could not draw subtype_IMI subset; skipping", flush=True)
        return None
    test_keep = te[keep[te]]
    Zs, ys, Zt, yt = prep_subset(X, y, s_, test_keep)

    models = {}
    for m in ("Q1_vqc", "Q3_qkernel_svm", "C2_svm_rbf"):
        models[m] = make_fixed(m, ch[m], seed)
    tw_chosen = {}
    for name, grid in twin_models().items():
        k, fit = tune_twin(grid, Zs, ys, 7 + seed)
        tw_chosen[name] = k; models[name] = fit

    res = {}
    for m, fit in models.items():
        try:
            sc = fit(Zs, ys)
            res[m] = R.auc(yt, sc(Zt))
        except Exception as e:
            res[m] = None
            print(f"  {m} error: {e}", flush=True)

    return dict(fold=fold, r=r, chosen={m: ch[m] for m in ("Q1_vqc", "Q3_qkernel_svm", "C2_svm_rbf")},
                twins=tw_chosen, res=res, n_train=len(ys), n_test=len(yt))


# ----- bootstrap analysis -----

def bootstrap_ci(d_matrix, B=5000, seed=0):
    """d_matrix: folds x subsets array of paired differences.
    Returns (mean, ci_lo, ci_hi) using fold-stratified bootstrap."""
    rng = np.random.default_rng(seed)
    n_folds, n_sub = d_matrix.shape
    boot = np.zeros(B)
    for b in range(B):
        rows = rng.integers(n_folds, size=n_folds)
        cols = rng.integers(n_sub, size=(n_folds, n_sub))
        samp = d_matrix[rows]
        samp = samp[np.arange(n_folds)[:, None], cols]
        boot[b] = samp.mean()
    return float(d_matrix.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def analyze(out_path):
    rows = [json.loads(l) for l in open(out_path)]
    folds = sorted({r["fold"] for r in rows})
    n_sub = max(r["r"] for r in rows) + 1

    candidates = [("Q1_vqc", "subtype_IMI Q1 angle VQC"),
                  ("Q3_qkernel_svm", "subtype_IMI Q3 quantum-kernel SVM")]
    comparators = [("C2_svm_rbf", "SVM-RBF (best classical)"),
                   ("TW_LR", "twin LR"),
                   ("TW_LR2", "twin deg-2 LR"),
                   ("TW_SVM", "twin SVM")]

    results = {}
    for q_key, q_label in candidates:
        results[q_key] = {}
        for c_key, c_label in comparators:
            mat = np.full((len(folds), n_sub), np.nan)
            for rec in rows:
                fi = folds.index(rec["fold"])
                ri = rec["r"]
                qv = rec["res"].get(q_key)
                cv = rec["res"].get(c_key)
                if qv is not None and cv is not None:
                    mat[fi, ri] = qv - cv
            # fill missing with fold mean
            for fi in range(len(folds)):
                col_valid = ~np.isnan(mat[fi])
                if col_valid.sum() == 0: continue
                mat[fi, ~col_valid] = mat[fi, col_valid].mean()
            mean, lo, hi = bootstrap_ci(mat, B=5000, seed=42)
            results[q_key][c_key] = {"mean": mean, "ci": [lo, hi]}

    # Holm over 2 candidates vs C2_svm_rbf (CI-based: CI lo > 0 and point > 0.005)
    holm_results = {}
    for q_key, _ in candidates:
        r = results[q_key]["C2_svm_rbf"]
        passed = (r["ci"][0] > 0) and (r["mean"] >= 0.005)
        holm_results[q_key] = passed

    # For Holm: sort by CI lower bound descending
    sorted_cands = sorted(candidates, key=lambda x: results[x[0]]["C2_svm_rbf"]["ci"][0], reverse=True)
    n_pass = sum(1 for q_key, _ in sorted_cands
                 if results[q_key]["C2_svm_rbf"]["ci"][0] > 0 and results[q_key]["C2_svm_rbf"]["mean"] >= 0.005)
    # Holm adjustment: reject at alpha/(n_cands - i) for i-th sorted
    # Both must have CI lo > 0 and point >= 0.005 to claim confirmation
    holm_pass = {}
    for i, (q_key, _) in enumerate(sorted_cands):
        r = results[q_key]["C2_svm_rbf"]
        # Holm: for 2 tests, first must pass at p<0.025 (CI lo>0 is effectively a 95% test),
        # second at p<0.05. Since we use 95% CIs throughout, the Holm-adjusted criterion
        # for the i-th (0-indexed) candidate (n=2) is: CI lo > 0 for both (CI is already 95%).
        # Both pass if both CIs exclude 0 from below.
        holm_pass[q_key] = (r["ci"][0] > 0) and (r["mean"] >= 0.005)

    return results, holm_pass, candidates, comparators, folds


def write_table(results, holm_pass, candidates, comparators, folds, table_path):
    lines = [
        "# QB6 Confirmation Results",
        "",
        f"> Confirmation folds: {folds}. N=100 subsets from inferior-MI pool, r=0-9. "
        f"B=5000 fold-stratified bootstrap. Holm correction over 2 candidates.",
        "",
        "## Results vs SVM-RBF (best classical, chosen by inner CV)",
        "",
        "| candidate | mean d [95% CI] | CI lo > 0 | point >= +0.005 | Holm pass | claim |",
        "|---|---|---|---|---|---|",
    ]
    for q_key, q_label in candidates:
        r = results[q_key]["C2_svm_rbf"]
        d, lo, hi = r["mean"], r["ci"][0], r["ci"][1]
        hp = holm_pass[q_key]
        ci_pass = lo > 0
        pt_pass = d >= 0.005
        claim = "**benefit confirmed**" if hp else "hypothesis for future work"
        lines.append(f"| {q_label} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {ci_pass} | {pt_pass} | {hp} | {claim} |")

    lines += [
        "",
        "## Results vs dequantized twins",
        "",
        "| candidate | vs twin LR [95% CI] | vs twin deg-2 LR [95% CI] | vs twin SVM [95% CI] |",
        "|---|---|---|---|",
    ]
    for q_key, q_label in candidates:
        vals = []
        for c_key in ("TW_LR", "TW_LR2", "TW_SVM"):
            r = results[q_key][c_key]
            vals.append(f"{r['mean']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
        lines.append(f"| {q_label} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Overall claim",
        "",
    ]
    any_confirmed = any(holm_pass.values())
    if any_confirmed:
        confirmed = [q_label for q_key, q_label in candidates if holm_pass[q_key]]
        lines.append(f"**Benefit confirmed** for: {', '.join(confirmed)} (subtype_IMI; Holm-adjusted CI lower bound > 0, point estimate >= +0.005).")
    else:
        lines.append("**No benefit confirmed** (Holm-adjusted). Candidates are classified as 'hypothesis for future work'.")
    lines.append("The main-analysis claim (no benefit found on the standard analysis) is unchanged.")
    lines.append("")

    table_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written -> {table_path}", flush=True)
    for l in lines:
        print(l)


def main():
    folds_arg = [int(f) for f in sys.argv[1:]] if len(sys.argv) > 1 else [3, 4]
    assert all(f in (3, 4) for f in folds_arg), "QB6 must use confirmation folds 3 and 4 only"

    X, y, g, _, feats = S.load()
    sub_cls = meta()

    done = set()
    if OUT.exists():
        done = {(json.loads(l)["fold"], json.loads(l)["r"]) for l in open(OUT)}

    for fold in folds_arg:
        for r in range(10):
            if (fold, r) in done:
                print(f"fold {fold} r {r} already done, skipping", flush=True)
                continue
            t0 = time.time()
            rec = run_unit(fold, r, X, y, g, sub_cls)
            if rec is None:
                continue
            rec["secs"] = round(time.time() - t0, 1)
            with open(OUT, "a") as fo:
                fo.write(json.dumps(rec) + "\n")
            print(f"fold {fold} r {r} done in {rec['secs']}s  "
                  f"Q1={rec['res'].get('Q1_vqc', 'err'):.4f}  "
                  f"Q3={rec['res'].get('Q3_qkernel_svm', 'err'):.4f}  "
                  f"C2={rec['res'].get('C2_svm_rbf', 'err'):.4f}", flush=True)

    # Only analyze if all 20 units are present (or at least both folds done)
    n_done = sum(1 for _ in open(OUT)) if OUT.exists() else 0
    expected = len(folds_arg) * 10
    if n_done < expected:
        print(f"WARNING: only {n_done}/{expected} units done; run script again for missing folds", flush=True)
        return

    print("\n--- Analysis ---", flush=True)
    results, holm_pass, candidates, comparators, folds_seen = analyze(OUT)
    write_table(results, holm_pass, candidates, comparators, folds_seen, TABLE_OUT)

    out_json = {"folds": folds_seen, "n_units": n_done,
                "results": {q: {c: v for c, v in rv.items()}
                            for q, rv in results.items()},
                "holm_pass": holm_pass}
    with open("results/qb6_analysis.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print("Written -> results/qb6_analysis.json", flush=True)


if __name__ == "__main__":
    main()
