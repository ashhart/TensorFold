"""Kept prompt prefixes on one rank: committed rings and tails, cache rows saved aside when overwritten."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

ALIGN = 256             # bytes between two tensors laid out in one space


@dataclass
class Snapshot:
    """The committed state at ``len(ids)``: rings and tails copied, cache rows still live unless ``rows`` holds them."""

    ids: list[int]
    rings: torch.Tensor
    tail: torch.Tensor
    tail_valid: torch.Tensor
    rows: list[torch.Tensor] | None = None    # State.row_views(len(ids)) copied out, in that order
    nbytes: int = 0                           # space the saved rows take


def _span(tensors: Sequence[torch.Tensor]) -> tuple[list[int], int]:
    """Each tensor's byte offset when laid out one after another at ``ALIGN`` boundaries, and the total."""

    offsets, end = [], 0
    for t in tensors:
        offsets.append(end)
        end += -(-t.numel() * t.element_size() // ALIGN) * ALIGN
    return offsets, end


def _copy(tensors: Sequence[torch.Tensor], space: torch.Tensor | None) -> list[torch.Tensor]:
    """Copies of ``tensors``, laid out in ``space`` when given."""

    if space is None:
        return [t.clone() for t in tensors]
    offsets, end = _span(tensors)
    if space.dtype != torch.uint8 or space.dim() != 1 or space.numel() < end or space.data_ptr() % ALIGN:
        raise ValueError(f"a snapshot copy needs {end} bytes of {ALIGN}-aligned uint8 space, not {space.dtype} "
                         f"{tuple(space.shape)}")
    out = []
    for t, at in zip(tensors, offsets):
        view = space[at:at + t.numel() * t.element_size()].view(t.dtype).view(t.shape)
        out.append(view.copy_(t))
    return out


def state_bytes(e) -> int:
    """The space ``take`` lays its copies out in."""

    st = e.st
    return _span([st.rings, st.tail, st.tail_valid])[1]


@torch.no_grad()
def take(e, ids: Sequence[int], space: torch.Tensor | None = None) -> Snapshot:
    """The committed state of ``ids`` (``st.pos == len(ids)``), its rings and tails copied into ``space``."""

    st = e.st
    if st.pos != len(ids):
        raise ValueError(f"a snapshot of {len(ids)} ids at {st.pos} committed positions")
    rings, tail, valid = _copy([st.rings, st.tail, st.tail_valid], space)
    return Snapshot(list(ids), rings, tail, valid)


@torch.no_grad()
def restore(e, snap: Snapshot) -> None:
    """The live state back at ``snap``: rings, tails, position and token tail; cache rows must be live."""

    st = e.st
    st.rings.copy_(snap.rings)
    st.tail.copy_(snap.tail)
    st.tail_valid.copy_(snap.tail_valid)
    back = e.w.cfg.engram_max_ngram_size - 1
    st.history = list(snap.ids[-back:]) if back > 0 else []
    st.set_pos(len(snap.ids))


def row_bytes(e, snap: Snapshot) -> int:
    """The space ``save_rows`` needs for this snapshot."""

    return _span(e.st.row_views(len(snap.ids)))[1]


@torch.no_grad()
def save_rows(e, snap: Snapshot, space: torch.Tensor | None = None) -> None:
    """Copy the snapshot's position-addressed rows from the live caches into ``space`` before an overwrite."""

    snap.rows = _copy(e.st.row_views(len(snap.ids)), space)
    snap.nbytes = row_bytes(e, snap)


@torch.no_grad()
def load_rows(e, snap: Snapshot) -> None:
    """Put a snapshot's saved rows back into the live caches."""

    views = e.st.row_views(len(snap.ids))
    if snap.rows is None or len(views) != len(snap.rows):
        raise ValueError("this snapshot holds no saved rows for these caches")
    for dst, src in zip(views, snap.rows):
        dst.copy_(src)


def snapshot_bytes(snap: Snapshot) -> int:
    """A kept snapshot's space: rings, tails and any saved rows."""

    return _span([snap.rings, snap.tail, snap.tail_valid])[1] + (snap.nbytes if snap.rows is not None else 0)
