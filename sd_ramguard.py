"""Every 60 s: if free physical RAM < 3 GB, stop ONE sd_run worker (the most recently started) and log it; its unfinished units are picked up by a later relaunch."""
import ctypes, subprocess, time, datetime, json

class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

def free_gb():
    m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); return m.ullAvailPhys / 2**30

while True:
    f = free_gb()
    if f < 3.0:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -match 'sd_run.py' } | Sort CreationDate | Select -Last 1 -ExpandProperty ProcessId"],
            capture_output=True, text=True).stdout.strip()
        if out:
            subprocess.run(["taskkill", "/F", "/PID", out], capture_output=True)
            open("results/sd_ramguard.log", "a").write(f"{datetime.datetime.now():%F %T} free {f:.2f} GB < 3 -> stopped sd_run pid {out}\n")
    time.sleep(60)
