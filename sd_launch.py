"""Launch W independent sd_run workers (each takes every W-th unit) and restart any that start in the slow state (exit code 99).
  python sd_launch.py PYTHON FAMILY K SIZES W [extra sd_run args...]"""
import subprocess, sys, time, os
py, fam, k, sizes, W = sys.argv[1:6]; extra = sys.argv[6:]; W = int(W)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
def start(i):
    return subprocess.Popen([py, "-u", "sd_run.py", "--family", fam, "--k", k, "--sizes", sizes, "--workers", str(W), "--worker-id", str(i)] + extra, env=env,
                            stdout=open(f"results/sd_{fam}_k{k}_w{i}.log", "a"), stderr=subprocess.STDOUT)
procs = {i: start(i) for i in range(W)}; tries = {i: 0 for i in range(W)}; finished = set()
while len(finished) < W:
    time.sleep(5)
    for i, p in list(procs.items()):
        if i in finished: continue
        rc = p.poll()
        if rc is None: continue
        if rc == 99 and tries[i] < 200:
            tries[i] += 1; procs[i] = start(i)
        else:
            finished.add(i); print("worker", i, "finished rc", rc, "restarts", tries[i], flush=True)
