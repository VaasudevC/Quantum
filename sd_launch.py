"""Launch up to W sd_run workers (dynamic unit claiming).  A worker is started or restarted only while free RAM >= 3.5 GB; workers stopped by the
RAM guard (or slow-start exit 99) are restarted later; the launcher ends when a worker exits with code 0 (nothing left) and none are running.
  python sd_launch.py PYTHON FAMILY K SIZES W [extra sd_run args...]"""
import ctypes, os, subprocess, sys, time
py, fam, k, sizes, W = sys.argv[1:6]; extra = sys.argv[6:]; W = int(W)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]
def free_gb():
    m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); return m.ap / 2**30
def start(i):
    return subprocess.Popen([py, "-u", "sd_run.py", "--family", fam, "--k", k, "--sizes", sizes, "--workers", str(W), "--worker-id", str(i)] + extra, env=env,
                            stdout=open(f"results/sd_{fam}_k{k}_w{i}.log", "a"), stderr=subprocess.STDOUT)
procs = {}; slot_done = set(); restarts = 0
while True:
    for i, p in list(procs.items()):
        rc = p.poll()
        if rc is None: continue
        del procs[i]
        if rc == 0: slot_done.add(i); print("worker", i, "exited 0", flush=True)
        else: restarts += 1
    if len(slot_done) == W and not procs:
        print("all done; restarts", restarts, flush=True); break
    for i in range(W):
        if i not in procs and i not in slot_done and free_gb() >= 3.5:
            procs[i] = start(i); time.sleep(20)
    time.sleep(15)
