"""Opt the current Windows process out of EcoQoS / power throttling (hybrid-CPU laptops can park background workers on slow cores)."""
import ctypes, sys

def boot():
    if sys.platform != "win32":
        return
    try:
        k = ctypes.windll.kernel32
        class PT(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
        st = PT(1, 1, 0)                      # PROCESS_POWER_THROTTLING_EXECUTION_SPEED controlled, state OFF
        k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))   # ProcessPowerThrottling = 4
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00008000)   # ABOVE_NORMAL
    except Exception:
        pass
