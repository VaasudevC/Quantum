import os, sys, json, platform
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np, scale_cv as S, Vqc_fast8 as V
from sklearn.metrics import roc_auc_score
for m in ["torch","pennylane","xgboost","qiskit","qiskit_aer","qiskit_ibm_runtime","wfdb","sklearn","scipy"]:
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e: print(m,"MISSING",e)
print("python",platform.python_version(),"numpy",np.__version__,"cores",os.cpu_count())
row=[json.loads(l) for l in open("results/stage2_steps1000.jsonl")]
r=[r for r in row if r["fold"]==0 and r["k"]==32 and r["scale"]=="pi/6" and r["selector"]=="l1"][0]
Z,tr,va,te,order=S.prep("grouped5",0,"l1"); X,y,_,_,feats=S.load()
sel=list(order[:32]); assert [feats[j] for j in sel]==r["features"], "feature order mismatch"
T=np.zeros((len(Z),V.N_FEATURES)); T[:,:32]=np.tanh(Z[:,sel])
m=S.Linear(32,np.pi/6,r.get("readout","z"),r.get("train_scale",False))
auc=roc_auc_score(y[te],m.proba(T[te],np.array(r["params"])))
print("recomputed",auc,"logged",r["test_auc"],"diff",abs(auc-r["test_auc"]))
