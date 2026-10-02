"""Gated delta rule for trees and replays: each runs the serial step, so a row's bits never depend on its window."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import VOLTA, load

    here = Path(__file__).parent
    return load(name="tensorfold_gdn_v2", sources=[str(here / "gdn.cpp"), str(here / "gdn.cu"),
                                                   str(here / "gdn_prefill.cu")],
                need=VOLTA, extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def _order_slots(parents: Sequence[int], order: list[int]) -> tuple[list[int], int]:
    """Plan entries (node, source, destination) for one visiting order, and the slots it needs."""

    pos = {node: i for i, node in enumerate(order)}
    children: dict[int, list[int]] = {}
    for node, parent in enumerate(parents):
        if parent >= 0:
            children.setdefault(parent, []).append(node)
    slot_of: dict[int, int] = {}
    free: list[int] = []
    used = 0
    entries: list[int] = []
    for i, node in enumerate(order):
        parent = parents[node]
        if parent < 0:
            source = -1
        elif i > 0 and order[i - 1] == parent:
            source = -2
        else:
            source = slot_of[parent]
        if parent >= 0 and parent in slot_of and pos[max(children[parent], key=pos.__getitem__)] == i:
            free.append(slot_of.pop(parent))        # its last child reads it now; the slot may be reused
        dest = -1
        if any(pos[c] != i + 1 for c in children.get(node, ())):
            if free:
                free.sort()
                dest = free.pop(0)
            else:
                dest, used = used, used + 1
            slot_of[node] = dest
        entries += [node, source, dest]
    return entries, used


def schedule(parents: Sequence[int]) -> tuple[list[int], int]:
    """(node, source, dest) entries and slot count for one tree, depth-first or level order, whichever needs fewer."""

    parents = [int(p) for p in parents]
    if not parents or parents[0] != -1 or any(not -1 <= p < i or (p == -1) != (i == 0) for i, p in enumerate(parents)):
        raise ValueError("a tree needs one root at row zero and parents before their children")
    children: dict[int, list[int]] = {}
    depth = [0] * len(parents)
    for node, parent in enumerate(parents[1:], start=1):
        children.setdefault(parent, []).append(node)
        depth[node] = depth[parent] + 1
    dfs, stack = [], [0]
    while stack:
        node = stack.pop()
        dfs.append(node)
        stack.extend(reversed(children.get(node, ())))
    level = sorted(range(len(parents)), key=lambda n: (depth[n], n))
    best = min((_order_slots(parents, o) for o in (dfs, level)), key=lambda e: e[1])
    if best[1] > 32:
        raise ValueError("this tree needs more than 32 live states")
    return best


@dataclass
class Plan:
    """A window's GDN schedule on the device: (W, 3) entries in stream order, stream row ranges, slot count."""

    entries: torch.Tensor
    starts: torch.Tensor
    slots: int
    max_rows: int


def plan_host(streams: Sequence[Sequence[int]]) -> tuple[list[int], list[int], int, int]:
    """Host half of ``plan`` for callers packing their own copy: entries in window rows, starts, slots and max rows."""

    entries, starts, slots = [], [0], 0
    for parents in streams:
        flat, need = schedule(parents)
        base = starts[-1]
        entries += [x + base if j % 3 == 0 else x for j, x in enumerate(flat)]
        starts.append(base + len(parents))
        slots = max(slots, need)
    return entries, starts, slots, max(b - a for a, b in zip(starts, starts[1:]))


def to_device(values: Sequence[int], dtype: torch.dtype, device) -> torch.Tensor:
    """One pinned, non-blocking host-to-device copy (the caching host allocator keeps the buffer until it lands)."""

    return torch.tensor(values, dtype=dtype).pin_memory().to(device, non_blocking=True)


def plan(streams: Sequence[Sequence[int]], device) -> Plan:
    entries, starts, slots, rows = plan_host(streams)
    dev = to_device(entries + starts, torch.int32, device)
    w = starts[-1]
    return Plan(dev[:3 * w].view(w, 3), dev[3 * w:], slots, rows)


def tree(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, p: Plan,
         state: torch.Tensor | None = None, table: torch.Tensor | None = None,
         pending: Sequence[torch.Tensor] | None = None, final: torch.Tensor | None = None) -> torch.Tensor:
    """Outputs (W, Hv, Dv) bf16; ``pending`` steps committed states in place first, so no state may be shared."""

    if (state is None) == (table is None):
        raise ValueError("pass one stream's state or a table of every stream's")
    return _ext().tree(q, k, v, g, beta, state, table, None if table is None else p.starts, p.entries, p.slots,
                       p.max_rows, None if pending is None else list(pending), final)


def pointers(tensors: Sequence[torch.Tensor]) -> list[int]:
    """Device addresses for a pointer table (the tensors must outlive the launch that reads them)."""

    if any(not t.is_cuda or not t.is_contiguous() for t in tensors):
        raise ValueError("pointer tables take contiguous CUDA tensors")
    return [t.data_ptr() for t in tensors]


def replay_table(k: Sequence[torch.Tensor], v: Sequence[torch.Tensor], g: Sequence[torch.Tensor],
                 beta: Sequence[torch.Tensor], states: Sequence[Sequence[torch.Tensor]]) -> list[int]:
    """Host pointers for ``replay``: k, v, g, beta of each layer, then states[stream][layer]."""

    layers = len(k)
    if not (len(v) == len(g) == len(beta) == layers) or any(len(s) != layers for s in states):
        raise ValueError("one k, v, g, beta per layer and one state per stream and layer")
    shape = states[0][0].shape
    if any(t.shape != shape or t.dtype != torch.float32 for s in states for t in s):
        raise ValueError("every state is one fp32 (Hv, Dv, 128) tensor")
    table = []
    for layer in range(layers):
        table += pointers([k[layer], v[layer], g[layer], beta[layer]])
    for s in states:
        table += pointers(s)
    return table


def replay(table: torch.Tensor, layers: int, streams: int, rows: torch.Tensor, counts: torch.Tensor,
           k0: torch.Tensor, v0: torch.Tensor, *, in_place: bool = False) -> torch.Tensor | None:
    """States after the accepted rows; ``in_place`` overwrites the table's states, which nothing else may hold."""

    out = _ext().replay(table, layers, streams, rows, counts, k0.shape[1], v0.shape[1], v0.shape[2],
                        k0.dtype == torch.float32, in_place)
    return None if in_place else out


def chain(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor,
          final: torch.Tensor) -> torch.Tensor:
    """Prefill chain from ``state`` (read only) to ``final``: chunk-invariant bits, not the verify kernel's."""

    return _ext().prefill(q, k, v, g, beta, state, final)
