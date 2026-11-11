import sys, numpy as np, pandas as pd, wfdb
from scipy import signal as sps
sys.path.insert(0, "cloud_repo/ptb_external"); import extract_ptb_features as X
EST=X.EST
tab = pd.read_csv(X.TABLE).set_index("ecg_id"); meta = pd.read_csv(X.XL/"ptbxl_database.csv", index_col="ecg_id")
ids = np.random.default_rng(0).choice(tab.index.to_numpy(), 8, replace=False)
sigs={i:wfdb.rdrecord(str(X.XL/meta.loc[i,"filename_hr"])).p_signal[:5000] for i in ids}
orig=EST.diagnostic_filter
def run(label):
    rows={i:EST.process_record("x",s,500) for i,s in sigs.items()}
    new=pd.DataFrame.from_dict(rows,orient="index"); c=[k for k in new.columns if k in tab.columns]
    d=(new[c]-tab.loc[new.index,c]).abs().max()
    print(f"{label:28s} cols>1e-9: {int((d>1e-9).sum()):3d}/{len(c)} max {d.max():.4g}  med-of-colmax {d.median():.3g}", flush=True)
run("current")
EST.diagnostic_filter=lambda s,fs,lo=0.05,hi=40.0: s; run("no filter")
EST.diagnostic_filter=lambda s,fs,lo=0.05,hi=40.0: EST.bandpass(s,fs,0.5,40.0); run("bandpass 0.5-40")
for lo,hi,o in ((0.05,150,4),(0.05,40,2),(0.5,150,4),(0.05,100,4),(0.67,40,4)):
    def f(s,fs,lo=lo,hi=hi,o=o):
        nyq=fs/2; sos=sps.butter(o,[lo/nyq,hi/nyq],btype="band",output="sos"); return sps.sosfiltfilt(sos,s)
    EST.diagnostic_filter=f; run(f"butter {o} {lo}-{hi}")
def f(s,fs,lo=0.05,hi=40.0):
    nyq=fs/2; b,a=sps.butter(4,[lo/nyq,hi/nyq],btype="band"); return sps.filtfilt(b,a,s)
EST.diagnostic_filter=f; run("filtfilt ba 4 0.05-40")
