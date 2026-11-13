"""RAM-gated scheduler for resumable QB jobs: starts a job only while free RAM >= 3.5 GB and fewer than MAXP jobs run; stops the newest job if free RAM < 3 GB
(it is relaunched later; jobs are resumable).  Ends when every job has exited with code 0.
  python qb_sched.py MAXP "cmd1" "cmd2" ..."""
import ctypes, datetime, os, shlex, subprocess, sys, time
MAXP = int(sys.argv[1]); cmds = sys.argv[2:]
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]
def free_gb():
    m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); return m.ap / 2**30
def log(s): open("results/qb_sched.log", "a").write(f"{datetime.datetime.now():%F %T} {s}\n")
running = {}; done = set()
while len(done) < len(cmds):
    for i, (p, t) in list(running.items()):
        rc = p.poll()
        if rc is None: continue
        del running[i]
        if rc == 0: done.add(i); log(f"done: {cmds[i]}")
        else: log(f"exited rc={rc}, will relaunch: {cmds[i]}")
    f = free_gb()
    if f < 3.0 and running:
        i = max(running, key=lambda j: running[j][1]); subprocess.run(["taskkill", "/T", "/F", "/PID", str(running[i][0].pid)], capture_output=True); log(f"free {f:.2f} GB < 3: stopped {cmds[i]}"); time.sleep(30); continue
    for i, c in enumerate(cmds):
        if i in done or i in running or len(running) >= MAXP or free_gb() < 3.5: continue
        running[i] = (subprocess.Popen(shlex.split(c, posix=False), env=env, stdout=open(f"results/qb_sched_job{i}.log", "a"), stderr=subprocess.STDOUT), time.time()); log(f"start: {c}"); time.sleep(30)
    time.sleep(30)
log("all jobs done")
