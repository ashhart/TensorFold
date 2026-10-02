"""The Apple GPU's generation, read once from the Metal device whatever the default device is."""

from __future__ import annotations

import functools
import re

import mlx.core as mx

# applegpu_g17 (M5) and later have Metal 4 tensor units; M1 is g13, M3 g15
TENSOR_UNITS = 17


@functools.cache
def generation() -> int:
    """The N of the Metal GPU's ``applegpu_gN``; 0 without Metal or for a GPU that names none (``air64_v27``, the
    paravirtual GPU of a macOS VM). The CPU's ``arm64`` is never read: a CPU default device asks the GPU."""

    if not mx.metal.is_available():
        return 0
    info = mx.device_info(mx.gpu) if hasattr(mx, "device_info") else mx.metal.device_info()
    found = re.match(r"applegpu_g(\d+)", str(info.get("architecture", "")))
    return int(found.group(1)) if found else 0


def tensor_units() -> bool:
    """Whether this GPU has the M5 generation's tensor units."""

    return generation() >= TENSOR_UNITS


__all__ = ["TENSOR_UNITS", "generation", "tensor_units"]
