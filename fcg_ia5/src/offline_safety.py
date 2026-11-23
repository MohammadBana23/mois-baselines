"""Local, offline-only input, resource and checkpoint controls."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys

from .prompt_contract import FEATURES, features


def raw_query(row):
    """Project a raw row onto the frozen label-free query schema."""
    return features({key: row[key] for key in FEATURES})


def raw_support(row):
    prompt = raw_query(row)
    label = {"0": 0, "1": 1}.get(str(row["y"]))
    sex, raw_race = prompt["sex"], prompt["race"]
    race = "White" if raw_race == "White" else "Non-white"
    if (label is None or sex not in {"Male", "Female"}
            or raw_race not in {"White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Non-white"}):
        raise ValueError("Invalid Support group or label")
    if ("race_binary" in row and row["race_binary"] != race
            or "group" in row and row["group"] != f"{race} {sex}"):
        raise ValueError("Support group metadata mismatch")
    return prompt, (sex, race), label


@contextmanager
def checkpoint_lock(path):
    """Lock a persistent sidecar. The OS releases the lock on process exit."""
    lock_path = Path(str(Path(path).resolve()) + ".lock")
    with lock_path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Checkpoint already has a writer") from exc
        yield


def sample_current_process():
    """Sample available physical RAM and peak RSS of this offline process."""
    if sys.platform == "linux":
        import resource

        available = next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        return int(available) * 1024, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong)
                for name in ("total", "available", "page_total", "page_available",
                             "virtual_total", "virtual_available", "extended")
            ]

        class Counters(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged",
                    "pagefile", "peak_pagefile",
                )
            ]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        counters = Counters()
        counters.length = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError("Physical RAM sampling failed")
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            wintypes.HANDLE(process()), ctypes.byref(counters), counters.length
        ):
            raise OSError("Peak process RAM sampling failed")
        return memory.available, counters.peak
    raise RuntimeError("Resource sampling unsupported on this platform")
