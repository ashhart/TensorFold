"""Keep the loaded weights resident: MLX wires its live buffers up to a limit and leaves every later buffer out."""

from __future__ import annotations

import subprocess
from typing import Any


def system_ceiling(mx: Any) -> int:
    """The most one process may wire: the kernel's iogpu.wired_limit_mb when set, else Metal's recommended working set."""

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    recommended = int(info.get("max_recommended_working_set_size", 0) or 0)
    try:
        out = subprocess.run(["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        mb = int(out or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        mb = 0
    return min(recommended, mb * 1024**2) if mb > 0 else recommended


def wire_resident(mx: Any, budget_bytes: int) -> int:
    """Wire what MLX holds now (the weights) within the budget and the system ceiling; the bytes wired, or 0."""

    mx.synchronize()
    mx.clear_cache()
    limit = min(int(mx.get_active_memory()), int(budget_bytes), system_ceiling(mx))
    if limit <= 0:
        return 0
    try:
        mx.set_wired_limit(limit)
    except (ValueError, RuntimeError):         # a limit MLX refuses leaves prompts as they were: unwired
        return 0
    return limit


def unwire(mx: Any) -> None:
    """Release the wired weights, so nothing stays wired past the server."""

    mx.synchronize()
    mx.set_wired_limit(0)


__all__ = ["system_ceiling", "unwire", "wire_resident"]
