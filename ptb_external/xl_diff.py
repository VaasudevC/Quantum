import sys, numpy as np, pandas as pd
sys.path.insert(0, "ptb_external"); import extract_ptb_features as X
tab = pd.read_csv(X.TABLE).set_index("ecg_id"); meta = pd.read_csv(X.XL/"ptbxl_database.csv", index_col="ecg_id")
cols=[c for c in X.table_cols() if c not in X.META]
ids = np.random.default_rng(0).choice(tab.index.to_numpy(), 6, replace=False)
rows={}
for i in ids:
    _, f = X.do_xl((int(i), meta.loc[i,"filename_hr"])); rows[i]=f
new=pd.DataFrame.from_dict(rows,orient="index")[cols]; old=tab.loc[new.index,cols]
d=(new-old).abs(); 
suf=lambda c: c.split("_",1)[1] if "_" in c and c.split("_")[0] in ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"] else c
g=d.max().groupby(suf).max(); print(g.round(6).to_string())
print(new.iloc[0][["II_st_curvature","II_st_amp_j60","II_r_amp","n_beats_detected"]], old.iloc[0][["II_st_curvature","II_st_amp_j60","II_r_amp","n_beats_detected"]])
