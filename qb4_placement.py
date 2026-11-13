"""QB4 (pre-registered in QB.md): feature placement at k = 32, s = pi/6, DISCOVERY folds 0-2.
Placement = permutation of the 32 L1-selected features over the (layer, qubit) slots (feature at position p -> layer p // 8, qubit p % 8), as in Vqc_fast8.
  (a) L1 order (must reproduce results/stage2_steps1000.jsonl exactly); (b) 10 random; (c) clinically paired (F3 pairs share a qubit in consecutive layers);
  (d) highest centered alignment among 200 random.
Training-free score: centered kernel-target alignment of the encoding-only re-uploading kernel (theta = 0) on 500 class-stratified training ECGs.
Training: exactly the stage-2 protocol (S.train: 1000 steps, batch 1000, Adam lr 0.01, best-validation checkpoint, seed = fold).
  ..\\.venv\\Scripts\\python.exe qb4_placement.py screen FOLD      -> results/qb4_screen.jsonl
  ..\\.venv\\Scripts\\python.exe qb4_placement.py train FOLD NAME  -> results/qb4_train.jsonl"""
import json, os, sys, time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
import scale_cv as S, Vqc_fast8 as V
from qb1_screen import clinical_pairs, centered_alignment

SC = np.pi / 6


def base(fold):
    X, y, _, _, feats = S.load(); Z, tr, va, te, _ = S.prep("grouped5", fold, "l1")
    row = [json.loads(l) for l in open("results/stage2_steps1000.jsonl")]
    row = [r for r in row if r["fold"] == fold and r["k"] == 32 and r["scale"] == "pi/6" and r["selector"] == "l1"][0]
    sel = [feats.index(n) for n in row["features"]]
    return Z[:, sel], y, tr, va, te, row


def paired_placement(names, rng):
    """greedy: put the two features of each clinical pair on the same qubit in consecutive layers; remaining features fill free slots in L1 order."""
    pairs = clinical_pairs(names); slots = {}; used = set(); free = [(l, q) for q in range(8) for l in range(4)]
    for (i, j) in pairs:
        if i in used or j in used: continue
        for q in range(8):
            for l in range(3):
                if (l, q) in free and (l + 1, q) in free:
                    slots[i] = (l, q); slots[j] = (l + 1, q); free.remove((l, q)); free.remove((l + 1, q)); used |= {i, j}; break
            if i in used: break
    rest = [i for i in range(32) if i not in used]
    free.sort()
    for i, s in zip(rest, free): slots[i] = s
    perm = [None] * 32                                    # perm[position] = feature index
    for i, (l, q) in slots.items(): perm[8 * l + q] = i
    return perm, len([p for p in pairs if p[0] in used and p[1] in used])


def enc_kernel(Zs, perm):
    T = np.tanh(Zs[:, perm]); A = SC * T
    psi = V.forward(A, np.zeros(V.N_ANSATZ))
    return np.abs(np.conj(psi) @ psi.T) ** 2


def placements(fold, names):
    rng = np.random.default_rng(500 + fold)
    P = {"a_L1": list(range(32))}
    for i in range(10): P[f"b_rand{i}"] = list(rng.permutation(32))
    P["c_paired"], npairs = paired_placement(names, rng)
    for i in range(200): P[f"d_cand{i}"] = list(rng.permutation(32))
    return P, npairs


def screen(fold):
    Zs, y, tr, va, te, row = base(fold); rng = np.random.default_rng(900 + fold)
    npos = int(round(500 * y[tr].mean())); idx = np.concatenate([rng.choice(tr[y[tr] == 1], npos, replace=False), rng.choice(tr[y[tr] == 0], 500 - npos, replace=False)])
    P, npairs = placements(fold, row["features"]); out = []
    for name, perm in P.items():
        out.append(dict(fold=fold, name=name, perm=[int(x) for x in perm], calign=centered_alignment(enc_kernel(Zs[idx], perm), y[idx])))
    d = max([o for o in out if o["name"].startswith("d_cand")], key=lambda o: o["calign"])
    out.append(dict(fold=fold, name="d_best_of_200", perm=d["perm"], calign=d["calign"], source=d["name"]))
    with open("results/qb4_screen.jsonl", "a") as fo:
        for o in out: fo.write(json.dumps(o) + "\n")
    print(fold, "screen done; clinical pairs placed:", npairs, flush=True)


def train(fold, name):
    S_ = [json.loads(l) for l in open("results/qb4_screen.jsonl")]
    perm = [o for o in S_ if o["fold"] == fold and o["name"] == name][0]["perm"]
    Zs, y, tr, va, te, row = base(fold)
    T = np.zeros((len(Zs), V.N_FEATURES)); T[:, :32] = np.tanh(Zs[:, perm])
    m = S.Linear(32, SC, "z", False); t0 = time.time()
    mm, (vauc, p, bstep) = S.train(m, T[tr], y[tr], T[va], y[va], 1000, 1000, 0.01, 0 + fold)
    rec = dict(fold=fold, name=name, perm=perm, val_auc=float(vauc), test_auc=float(roc_auc_score(y[te], mm.proba(T[te], p))), best_step=bstep, secs=round(time.time() - t0, 1),
               logged_L1_test_auc=row["test_auc"] if name == "a_L1" else None)
    with open("results/qb4_train.jsonl", "a") as fo: fo.write(json.dumps(rec) + "\n")
    print(fold, name, rec["test_auc"], rec["secs"], flush=True)


if __name__ == "__main__":
    mode, fold = sys.argv[1], int(sys.argv[2]); assert fold in (0, 1, 2), "discovery folds only"
    screen(fold) if mode == "screen" else train(fold, sys.argv[3])
