import sys, numpy as np, pandas as pd, wfdb
sys.path.insert(0, "cloud_repo/ptb_external"); import extract_ptb_features as X
md=pd.read_csv(X.XL/"dataset_metadata.csv", index_col="ecg_id"); sigs=np.load(X.XL/"dataset_signals.npy", mmap_mode="r")
print(sigs.shape, len(md))
i=md.index[5]; print(i, md.loc[i,"filename_hr"])
w=wfdb.rdrecord(str(X.XL/md.loc[i,"filename_hr"])).p_signal[:5000]
print(np.abs(np.asarray(sigs[5])-w).max())
