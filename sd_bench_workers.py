"""Throughput benchmark: W concurrent workers (W in 1, 2, 4 by default) each run the SAME SD unit (family main or torch, k=8, N=50, fold 0, r=39);
nothing is written to the SD result files.  Prints units/hour for each W so the launcher can use the best worker count.
  ..\\.venv\\Scripts\\python.exe sd_bench_workers.py main 1,2,4          (main family: Q1-Q3, C1-C6)
  ..\\.venv\\Scripts\\python.exe sd_bench_workers.py torch 1,2,4 C:/bwvenv/Scripts/python.exe
"""
import os, subprocess, sys, time

fam = sys.argv[1] if len(sys.argv) > 1 else "main"
Ws = [int(w) for w in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4").split(",")]
py = sys.argv[3] if len(sys.argv) > 3 else sys.executable
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
child = ("import time, sd_run as R; t=time.time(); R.unit(('%s', 8, 0, 50, 39, set(), 0)); print(round(time.time()-t, 1))" % fam)
for W in Ws:
    t0 = time.time()
    ps = [subprocess.Popen([py, "-c", child], env=env, stdout=subprocess.PIPE, text=True) for _ in range(W)]
    secs = [float(p.communicate()[0].strip().splitlines()[-1]) for p in ps]
    wall = time.time() - t0
    print(f"W={W}: per-unit seconds {secs}, wall {wall:.0f}s -> {3600 * W / wall:.1f} units/hour", flush=True)
