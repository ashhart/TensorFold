"""How Flash Next splits widths over ranks: whole blocks a rank, the lower ranks one more where they do not divide."""

from __future__ import annotations

UNIT = 64                 # an NVFP4 expert width share is whole 64-input groups


def share(width: int, rank: int, world: int, unit: int = UNIT) -> tuple[int, int]:
    """A rank's [lo, hi) of ``width``: equal parts where whole ``unit`` blocks divide, else the lower ranks one more."""

    if world == 1:
        return 0, width
    if width % unit:
        raise ValueError(f"width {width} is not whole {unit}-wide blocks")
    blocks = width // unit
    counts = [blocks // world + (r < blocks % world) for r in range(world)]
    if not counts[-1]:
        raise ValueError(f"width {width} leaves rank {world - 1} of {world} no {unit}-wide block")
    lo = sum(counts[:rank]) * unit
    return lo, lo + counts[rank] * unit
