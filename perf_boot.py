"""Opt the current Windows process and its calling thread out of EcoQoS / power throttling."""
import ctypes, sys


def boot():
    if sys.platform != "win32":
        return
    try:
        k = ctypes.windll.kernel32
        class ST(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
        st = ST(1, 1, 0)
        k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))
        k.SetThreadInformation(k.GetCurrentThread(), 3, ctypes.byref(st), ctypes.sizeof(st))
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00008000)
    except Exception:
        pass
