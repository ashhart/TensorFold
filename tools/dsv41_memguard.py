"""Memory guard of a dev run (tools/dsv41_serial_run.py) on GB10, where the GPU's allocations are the host's RAM and
running out has rebooted a node (the host watchdog): before anything is loaded, refuse to start when another job holds
the memory, and cap torch's caching allocator so an allocation past the cap raises ``torch.OutOfMemoryError`` in this
process instead of exhausting the node.

The cap is ``MemAvailable`` at start less ``reserve`` (what must stay free; tools/dsv41_memwatch.sh kills the run
below it) and ``slack`` (what this process allocates outside torch's allocator: the CUDA context, NCCL buffers,
~40 MB of driver memory a decode graph, pinned staging). The cgroup of the container does not see GPU allocations on
GB10 (measured: tf-dev's memory.current 6.9 GB while its rank held 108.9 GB of GPU memory), so ``docker update
--memory`` cannot bound them; this cap and the watcher do.

    TF_MEM_CAP_GIB       auto (default), 0 / off (no cap), or a cap in GiB
    TF_MEM_RESERVE_GIB   3     GiB left available (the watcher's floor too)
    TF_MEM_SLACK_GIB     3     GiB for allocations outside torch's allocator
    TF_MEM_MIN_START_GIB 100   refuse to start below this MemAvailable (a rank's weights are ~99 GiB); 0: never

No torch at import: the planning runs in host-only tests.
"""

from __future__ import annotations

import os

GIB = 1 << 30
DEFAULTS = {"cap": "auto", "reserve": 3.0, "slack": 3.0, "min_start": 100.0}


def meminfo(path: str = "/proc/meminfo") -> dict[str, int]:
    """/proc/meminfo in bytes (inside a container: the host's, there is no lxcfs on the dev boxes)."""

    out = {}
    with open(path) as f:
        for ln in f:
            name, _, rest = ln.partition(":")
            parts = rest.split()
            if parts:
                out[name.strip()] = int(parts[0]) * (1024 if parts[1:] == ["kB"] else 1)
    return out


def setting(name: str, cli, env=None) -> str:
    """A knob: the command line's value, else TF_MEM_<NAME>_GIB, else the default."""

    env = os.environ if env is None else env
    if cli is not None:
        return str(cli)
    v = env.get(f"TF_MEM_{name.upper()}_GIB", "").strip()
    return v or str(DEFAULTS[name])


def plan(avail: int, *, cap: str = "auto", reserve: float = 3.0, slack: float = 3.0,
         min_start: float = 100.0) -> tuple[int | None, str]:
    """(the allocator cap in bytes or None, a line saying why) for ``avail`` bytes available at start; MemoryError
    when the run should not start."""

    if min_start and avail < min_start * GIB:
        raise MemoryError(f"MemAvailable {avail / GIB:.1f} GiB < {min_start:g} GiB at start: another job holds the "
                          "memory (TF_MEM_MIN_START_GIB=0 starts anyway)")
    c = cap.strip().lower()
    if c in ("0", "off", "no", "none"):
        return None, f"no allocator cap (TF_MEM_CAP_GIB={cap}); MemAvailable {avail / GIB:.1f} GiB"
    if c in ("", "auto"):
        b = int(avail - (reserve + slack) * GIB)
        if b <= 0:
            raise MemoryError(f"MemAvailable {avail / GIB:.1f} GiB leaves nothing above reserve {reserve:g} + slack "
                              f"{slack:g} GiB")
        return b, (f"allocator cap {b / GIB:.1f} GiB = MemAvailable {avail / GIB:.1f} - reserve {reserve:g} - slack "
                   f"{slack:g} GiB")
    b = int(float(c) * GIB)
    if b <= 0:
        raise ValueError(f"TF_MEM_CAP_GIB={cap}")
    note = " (above what is available)" if b > avail - reserve * GIB else ""
    return b, f"allocator cap {b / GIB:.1f} GiB (TF_MEM_CAP_GIB){note}; MemAvailable {avail / GIB:.1f} GiB"


def apply(cap: int, device: int = 0) -> float:
    """Cap torch's caching allocator on ``device`` at ``cap`` bytes; returns the fraction of the device's total."""

    import torch

    total = torch.cuda.get_device_properties(device).total_memory
    frac = min(1.0, cap / total)
    torch.cuda.set_per_process_memory_fraction(frac, device)
    return frac


def guard(rank: int, *, cap=None, reserve=None, slack=None, min_start=None, log=print, path: str = "/proc/meminfo"):
    """Plan from this node's MemAvailable and apply; returns the cap (bytes) or None."""

    b, why = plan(meminfo(path)["MemAvailable"], cap=setting("cap", cap),
                  reserve=float(setting("reserve", reserve)), slack=float(setting("slack", slack)),
                  min_start=float(setting("min_start", min_start)))
    if b is not None:
        frac = apply(b)
        why += f" (fraction {frac:.3f})"
    log(f"[rank {rank}] memory guard: {why}", flush=True)
    return b
