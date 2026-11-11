import sys, numpy as np, pandas as pd
sys.path.insert(0, "cloud_repo/ptb_external"); import extract_ptb_features as X
tab = pd.read_csv(X.TABLE).set_index("ecg_id"); md=pd.read_csv(X.XL/"dataset_metadata.csv", index_col="ecg_id")
sigs=np.load(X.XL/"dataset_signals.npy", mmap_mode="r"); pos={e:i for i,e in enumerate(md.index)}
cols=[c for c in X.table_cols() if c not in X.META]
ids = np.random.default_rng(0).choice(tab.index.to_numpy(), 6, replace=False)
rows={}
for i in ids:
    rows[i]=X.feats_from_signal(np.asarray(sigs[pos[i]],dtype=np.float32).astype(np.float64))
new=pd.DataFrame.from_dict(rows,orient="index")[cols]; old=tab.loc[new.index,cols]
print("max diff", (new-old).abs().max().max(), "cols>1e-9:", int(((new-old).abs().max()>1e-9).sum()))
