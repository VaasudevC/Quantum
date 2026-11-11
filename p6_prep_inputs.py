"""(.venv) hardware-test inputs without sklearn in the qiskit venv: fold-0 test labels and the angle-VQC (k=32, pi/6) angles for fold-0 test ECGs."""
import numpy as np, qiskit_vqc as Q
row = Q.load_models()[(32, "pi/6")]
A, y, tr, va, te = Q.model_data(row)
np.savez("C:/bwcache/hw_inputs.npz", te=te, y_te=y[te], A_vqc_te=A[te][:, :32])
print(len(te), y[te].mean())
