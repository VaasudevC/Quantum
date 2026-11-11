import sys, numpy as np, pandas as pd
sys.path.insert(0, "ptb_external"); import extract_ptb_features as X, wfdb
tab = pd.read_csv(X.TABLE).set_index("ecg_id"); meta = pd.read_csv(X.XL/"ptbxl_database.csv", index_col="ecg_id")
cols=[c for c in X.table_cols() if c not in X.META and c in X.EST.__dict__.get('x',[]) or True]
st_cols=[c for c in X.table_cols() if c not in X.META][:  8+ 12*0]
ids = np.random.default_rng(0).choice(tab.index.to_numpy(), 6, replace=False)
for variant in ("f64","f32"):
    rows={}
    for i in ids:
        sig=wfdb.rdrecord(str(X.XL/meta.loc[i,"filename_hr"])).p_signal[:5000]
        if variant=="f32": sig=sig.astype(np.float32).astype(np.float64)
        rows[i]=X.EST.process_record("x",sig,500)
    new=pd.DataFrame.from_dict(rows,orient="index"); c=[k for k in new.columns if k in tab.columns]
    d=(new[c]-tab.loc[new.index,c]).abs().max()
    print(variant,"cols>1e-9:",int((d>1e-9).sum()),"of",len(c),"max",d.max())
