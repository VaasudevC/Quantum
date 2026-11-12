import sys, time, numpy as np
import sd_run as R
N = int(sys.argv[1]); fam = sys.argv[2]
d = np.load(R.CACHE / f"f0_N{N}_r0.npz"); Zs, ys, Zt, yt = d["Zsub"].astype(float), d["ysub"], d["Ztest"].astype(float), d["yte"]
if fam == "main":
    import Vqc_fast8 as V
    rng = np.random.default_rng(0); A = rng.uniform(-1, 1, (4, 8)); ang = np.zeros((4, 32)); ang[:, :8] = A
    psi = V.forward(ang, np.zeros(V.N_ANSATZ)); print("Q3 kernel check max diff", np.abs(np.abs(np.conj(psi) @ psi.T) ** 2 - R.QKernel.kern(A, A)).max(), flush=True)
for M in R.models_for(fam, 8):
    t = time.time(); rec = R.run_model(M, Zs, ys, Zt, yt, 0, 7)
    print(M.name, round(rec["test_auc"], 4), "train", round(rec["train_auc"], 3), rec["chosen"][:50], "secs", round(time.time() - t, 1), flush=True)
