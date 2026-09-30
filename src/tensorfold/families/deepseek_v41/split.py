"""How each DeepSeek-V4.1 tensor splits over two ranks, and what a rank keeps resident (header-only, no torch).

Rules (ARCH.md §10, vLLM's TP=2 layout, with experts split along the expert width like the shared expert):
  * ``cols``: output columns halved (EXL3: tile columns + svh; plain: rows of [out, in])
  * ``rows``: input rows halved (EXL3: tile rows + suh), partial sums added across ranks
  * ``part``: whole tensors owned by one rank (wo_a slices 0-3 / 4-7)
  * ``rep``: replicated
  * ``skip``: not loaded for text serving (vision tower, aligner, image markers)
"""

from __future__ import annotations

import re

RULES: tuple[tuple[str, str], ...] = (
    (r"^(vision|aligner|image_)", "skip"),
    (r"\.attn\.wq_b\.", "cols"),
    (r"\.attn\.wo_a\.slice\.\d+\.", "part"),
    (r"\.attn\.wo_b\.", "rows"),
    (r"\.ffn\.(experts\.\d+|shared_experts)\.w[13]\.", "cols"),
    (r"\.ffn\.(experts\.\d+|shared_experts)\.w2\.", "rows"),
    (r"^head\.", "cols"),
    (r"\.attn\.attn_sink$", "cols"),
)


def rule(name: str) -> str:
    for pattern, kind in RULES:
        if re.search(pattern, name):
            return kind
    return "rep"


def owner_of_slice(name: str, groups: int = 8, world: int = 2) -> int | None:
    m = re.search(r"\.wo_a\.slice\.(\d+)\.", name)
    return None if m is None else int(m.group(1)) * world // groups


def rank_bytes(name: str, nbytes: int, rank: int, world: int = 2) -> int:
    """Bytes of tensor ``name`` resident on ``rank``."""

    kind = rule(name)
    if kind == "skip":
        return 0
    if kind == "part":
        return nbytes if owner_of_slice(name) == rank else 0
    if kind in ("cols", "rows"):
        # halved parts: the trellis and the halved scale; the other scale stays whole (small; counted halved)
        return nbytes // world
    return nbytes


def budget(tensors: dict[str, tuple[str, list[int], int]], rank: int, world: int = 2) -> dict[str, int]:
    """Resident bytes of one rank by category, from {name: (dtype, shape, nbytes)}."""

    out: dict[str, int] = {}
    for name, (_, _, nbytes) in tensors.items():
        b = rank_bytes(name, nbytes, rank, world)
        if not b:
            continue
        if name.startswith("mtp."):
            cat = "mtp"
        elif ".ffn.experts." in name:
            cat = "routed experts"
        elif ".ffn.shared_experts." in name:
            cat = "shared experts"
        elif ".attn." in name:
            cat = "attention"
        elif ".engram." in name:
            cat = "engram wkv"
        elif name.startswith(("embed", "head", "norm")):
            cat = name.split(".")[0]
        else:
            cat = "hc/norm/router"
        out[cat] = out.get(cat, 0) + b
    return out
