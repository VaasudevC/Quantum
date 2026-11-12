"""smoke test: one fold x given N x few subsets for each model of a family; prints timings; checks the Q3 closed-form kernel against Vqc_fast8."""
import sys, time, json
import numpy as np
import sd_run as R
fam, N, nsub = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); k = int(sys.argv[4]) if len(sys.argv) > 4 else 8
if fam == "main":
    import Vqc_fast8 as V
    rng = np.random.default_rng(0); A = rng.uniform(-1, 1, (4, 8)); th = np.zeros(V.N_ANSATZ)
    ang = np.zeros((4, 32)); ang[:, :8] = A
    psi = V.forward(ang, th)
    Kq = np.abs(np.conj(psi[:, None, :]) * psi[None, :, :]).sum(-1) if False else np.abs((np.conj(psi) @ psi.T)) ** 2
    print("Q3 kernel check max diff", np.abs(Kq - R.QKernel.kern(A, A)).max())
for r in range(nsub):
    for rec in R.unit((fam, k, 0, N, r, set(), 0)):
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in rec.items() if kk in ("model", "test_auc", "train_auc", "chosen", "secs", "error", "n_params")}, flush=True)
