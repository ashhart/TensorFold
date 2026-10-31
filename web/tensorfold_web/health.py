"""Health findings for Monitor: each one a title, the evidence behind it and what to try, from pure rules.

The rules read what TensorFold Web already has (the engine's figures, this machine's memory) plus a few platform signals
read without administrator rights: macOS's memory pressure level and thermal state, NVIDIA's clock-event reasons.
Counters are judged against a baseline taken at the first reading, so a count from before this TensorFold Web started is not
reported as news.
"""

from __future__ import annotations

import contextlib
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass

from . import power

GIB = 1024**3
SIGNAL_TTL_S = 10.0                  # platform signals are read at most this often
# macOS's kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical
MAC_PRESSURE = {1: "normal", 2: "warning", 4: "critical"}


@dataclass
class Finding:
    id: str
    level: str                       # "bad", "warn" or "info"
    title: str
    evidence: str
    hint: str

    def to_dict(self) -> dict:
        return asdict(self)


class Signals:
    """Platform signals, cached for a few seconds: the page polls more often than they change."""

    def __init__(self) -> None:
        self._at = 0.0
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._thermal = None
        self._nvidia = power.NvidiaSmi() if shutil.which("nvidia-smi") else None
        if platform.system() == "Darwin":
            with contextlib.suppress(OSError, AttributeError, TypeError):
                self._thermal = power.MacThermal()         # the same reader the Bench uses

    def read(self) -> dict:
        with self._lock:
            if time.monotonic() - self._at < SIGNAL_TTL_S:
                return self._cache
            out: dict = {}
            if platform.system() == "Darwin":
                with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
                    level = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                                               capture_output=True, text=True, timeout=3, check=False).stdout.strip())
                    out["mac_pressure"] = MAC_PRESSURE.get(level, str(level))
                if self._thermal is not None:
                    with contextlib.suppress(OSError, ValueError):
                        out["thermal"] = self._thermal.state()
            if self._nvidia is not None:
                r = self._nvidia.read()
                if r is not None and r.throttled is not None:
                    out["gpu_slowdown"] = list(r.reasons)
            self._cache, self._at = out, time.monotonic()
            return out


class Checker:
    """Turns one reading of the engine and the machine into findings; keeps the counters' baselines."""

    def __init__(self) -> None:
        self._baseline: dict[str, tuple[float, float]] = {}      # counter -> (value at first sight, when)

    def _rose(self, key: str, value: float | None) -> tuple[float, float] | None:
        """How much a counter rose since it was first seen, and since when; a counter going back starts over."""

        if value is None:
            return None
        first = self._baseline.get(key)
        if first is None or value < first[0]:
            self._baseline[key] = (value, time.time())
            return None
        return (value - first[0], first[1]) if value > first[0] else None

    def reset(self) -> None:
        self._baseline.clear()

    def check(self, info: dict, signals: dict) -> list[Finding]:
        out: list[Finding] = []
        # Settings' memory budget reaches only the Python line; the native engine sizes its own memory
        budget = "give the engine less memory in Settings, " if info.get("engine_kind") == "python" else ""
        mm = info.get("machine_memory")
        if mm and mm.get("headroom") in ("low", "tight"):
            free, total = mm["available"] / GIB, mm["total"] / GIB
            low = mm["headroom"] == "low"
            out.append(Finding(
                "memory-headroom", "bad" if low else "warn",
                "Memory is nearly full" if low else "Little memory is left",
                f"{free:.1f} GB free of {total:.0f} GB" + (f"; the engine holds {mm['engine'] / GIB:.1f} GB"
                                                          if mm.get("engine") else ""),
                f"Close apps you do not need, {budget}or serve a shorter context or a smaller model"
                if low else "Long prompts or other apps may push the machine into swapping; watch the free memory."))
        pressure = signals.get("mac_pressure")
        if pressure in ("warning", "critical"):
            out.append(Finding(
                "mac-pressure", "bad" if pressure == "critical" else "warn", f"macOS reports {pressure} memory pressure",
                f"kern.memorystatus_vm_pressure_level is {pressure}",
                "macOS compresses memory and may end apps to make room; a smaller model or context leaves it room."))
        st = info.get("status")
        if st == "failed":
            err = info.get("error") or "no exit code"
            if err in ("exit code -9", "exit code 137"):           # SIGKILL: on a Mac, jetsam ending it for memory
                out.append(Finding(
                    "engine-killed", "bad", "The system stopped the engine",
                    f"{err}: killed from outside, with no error of its own",
                    f"On a Mac this is the system freeing memory (jetsam): {budget}serve a shorter context or a "
                    "smaller model, or close other apps." if platform.system() == "Darwin" else
                    f"Usually the kernel freeing memory (the OOM killer): {budget}serve a shorter context or a "
                    "smaller model."))
            else:
                out.append(Finding(
                    "engine-failed", "bad", "The engine stopped with an error", err,
                    "The engine log says why; out of memory shows there as an allocation error or a killed process."))
        rose = self._rose("preemptions", info.get("preemptions"))
        if rose:
            n, since = rose
            out.append(Finding(
                "preemptions", "warn", "Requests were set aside to free memory",
                f"{n:.0f} since {time.strftime('%H:%M', time.localtime(since))}",
                "The KV cache filled under this load: fewer requests at once or shorter prompts avoid it."))
        kv = info.get("kv_fill")
        if kv is not None and kv >= 0.9 and st == "running":
            out.append(Finding("kv-full", "warn", "The KV cache is nearly full", f"{kv * 100:.0f}% held",
                               "New requests may wait or be set aside; fewer at once, or a shorter context."))
        if (info.get("waiting") or 0) > 0:
            out.append(Finding("waiting", "info", "Requests are waiting for a turn",
                               f"{info['waiting']} waiting, {info.get('connections') or 0} running",
                               "The engine serves a set number at once; the others queue. This is normal under load."))
        thermal = signals.get("thermal")
        if thermal in ("serious", "critical"):
            out.append(Finding("thermal", "warn", "The Mac is slowing itself down to cool",
                               f"thermal state {thermal}",
                               "Speeds measured now understate the machine; give it air, or wait before a Bench run."))
        slow = [r for r in signals.get("gpu_slowdown") or [] if "thermal" in r or "slowdown" in r]
        if slow:
            out.append(Finding("gpu-slowdown", "warn", "The GPU is slowing itself down", ", ".join(slow),
                               "Speeds measured now understate the machine; check its airflow and temperature."))
        su = info.get("startup") or {}
        if su.get("prompt_cache_gib") == 0 and st == "running":
            used = f"; it found {su['in_use_gib']:.1f} GiB in use once loaded" if su.get("in_use_gib") else ""
            out.append(Finding(
                "prompt-cache-none", "warn", "The prompt cache got no memory",
                f"the engine kept 0 GiB for prompt states{used}",
                "Each turn then reads the whole conversation again (on a Mac mini M6, 5.8 s instead of 0.2 s for an "
                "8.5k-token chat). The engine sizes it from the memory in use when it loads: restart it when memory is "
                "free, or set --prompt-cache-gib in Settings' extra flags."))
        if st == "unauthorized":
            out.append(Finding("key", "bad", "The engine refuses TensorFold Web's key", info.get("error") or "",
                               "Enter the engine's key in Settings, or restart it to take TensorFold Web's."))
        return out
