"""The routed experts' EXL3 widths from the safetensors headers: a width per tensor, as the grouped kernels take it."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

EXPERT = re.compile(r"\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj$")


def widths(model_dir: str | Path) -> Counter:
    """Weights at each width (bits -> summed K * N); ValueError naming the first EXL3 group the loader does not read."""

    from tensorfold.cuda.exl3 import format as fmt

    ckpt = fmt.scan(model_dir, read_markers=False)
    if ckpt.bad:
        prefix, why = sorted(ckpt.bad.items())[0]
        raise ValueError(f"{prefix}: {why}")
    out: Counter = Counter()
    for prefix, g in sorted(ckpt.groups.items()):
        if not EXPERT.search(prefix):
            raise ValueError(f"{prefix} is EXL3: the reader takes EXL3 routed experts only, BF16 elsewhere")
        if (g.codebook, g.in_scales, g.out_scales, g.bias) != ("mcg", "suh", "svh", False):
            raise ValueError(f"{prefix}: {g.codebook} codebook, {g.in_scales}/{g.out_scales} scales"
                             f"{' and a bias' if g.bias else ''}; the reader takes mcg with suh/svh and no bias")
        out[g.bits] += g.k * g.n
    return out


def describe(found: Counter) -> str:
    """``"2-bit 50.0%, 3-bit 33.3%, 4-bit 16.7%"`` of the experts' weights."""

    total = sum(found.values()) or 1
    return ", ".join(f"{b:g}-bit {100 * n / total:.1f}%" for b, n in sorted(found.items()))
