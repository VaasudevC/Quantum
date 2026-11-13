"""Log one line to OVERNIGHT_LOG.md every 30 minutes: time, units done per job, free RAM."""
import ctypes, datetime, json, os, time
from pathlib import Path

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

def free_gb():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30

def count_units(pattern, keys=('fold','N','r')):
    units = set()
    for f in Path('results').glob(pattern):
        for l in open(f):
            try:
                d = json.loads(l)
                units.add(tuple(d.get(k) for k in keys))
            except: pass
    return len(units)

def count_qb5():
    units = set()
    try:
        for l in open('results/qb5.jsonl'):
            d = json.loads(l)
            units.add((d['fold'], d['r']))
    except: pass
    return len(units)

def count_scale_cv_seed3():
    done = 0
    try:
        for l in open('results/stage3_selectors.jsonl'):
            d = json.loads(l)
            if d.get('seed') == 3:
                done += 1
    except: pass
    return done

INTERVAL = 1800  # 30 minutes

start = time.time()
while True:
    time.sleep(INTERVAL)
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    k4 = count_units('sd_runs_main_k4.w*.jsonl')
    qb5 = count_qb5()
    seed3 = count_scale_cv_seed3()
    ram = free_gb()
    line = (f"- {t} progress: k4 units {k4}/850, QB5 {qb5}/30, seed3 runs {seed3}/5, "
            f"free RAM {ram:.2f} GB, elapsed {(time.time()-start)/3600:.1f}h\n")
    with open('OVERNIGHT_LOG.md', 'a') as f:
        f.write(line)
    print(line.strip(), flush=True)
