"""Low-overhead process metrics and optional generation timing."""
import math
import sys
import time
from pathlib import Path


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def peak_process_ram():
    """OS high-water mark of this process since startup; excludes child processes."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        size = counters.PeakWorkingSetSize
    else:
        import resource
        size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            size *= 1024
    return {"peak_process_ram_gib": size / 2**30,
            "ram_scope": "OS process lifetime peak RSS/working set; excludes child processes"}


def directory_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file() and not p.is_symlink())


class TokenTimingStreamer:
    """Generate's first put is the prompt; subsequent puts are generated tokens.

    Enabling a streamer synchronizes tokens to the host and can slow generation.
    This measures token availability, not first rendered word or network latency.
    """
    def __init__(self, started=None, clock=time.perf_counter):
        self.clock = clock
        self.started = clock() if started is None else started
        self.prompt_seen = False
        self.first = None
        self.last = None
        self.steps = 0

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        now = self.clock()
        if self.first is None:
            self.first = now
        self.last = now
        self.steps += 1

    def end(self):
        pass

    def metrics(self):
        return {"batch_ttft_seconds": self.first - self.started if self.first is not None else None,
                "subsequent_token_seconds": self.last - self.first if self.steps > 1 else None}


def measured_trainer_class(base):
    class MeasuredTrainer(base):
        def __init__(self, *args, **kwargs):
            self.training_token_counts = {"nonpadding": 0, "answer": 0, "padded": 0}
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, *args, **kwargs):
            counts = inputs.pop("_token_counts", None)
            if model.training and counts is not None:
                for key, value in zip(("nonpadding", "answer", "padded"), counts):
                    self.training_token_counts[key] += value
            return super().compute_loss(model, inputs, *args, **kwargs)
    return MeasuredTrainer
