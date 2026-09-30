"""A host-RAM tier under a CUDA prefix cache: prompt-end states it evicts, copied back instead of prefilled again."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

CHUNK = 16 << 20                 # host memory in chunks of this many bytes (a power of two: the pinned pool keeps it)
ALIGN = 256                      # each tensor's bytes start on this boundary inside an entry or a segment
GIB = 1024**3


@dataclass(frozen=True)
class Slot:
    """A spilled tensor's index among its entry's tensors."""

    index: int


@dataclass(eq=False)
class Segment:
    """Attention rows [start, end) of every layer's K and V in host chunks, computed for ``ids`` (a prompt's first
    ``end`` ids). Row p depends only on ids[:p + 1]: an entry whose ids agree with these through p shares row p."""

    ids: list[int]
    start: int
    end: int
    layout: tuple                # per layer None, or each attention buffer's dtype and row shape
    offsets: list[int]           # each buffer's rows start at this byte of the chunks
    chunks: list[torch.Tensor]
    refs: int = 0                # entries holding rows of it; none: cached for a later spill to chain onto
    done: Any = None             # a CUDA event: its copies have landed
    used: int = 0                # the tier's clock when last written or read


@dataclass(eq=False)
class Spilled:
    """An entry in host memory: its ids, the state and drafter snapshot with ``Slot``s for tensors, their bytes, and
    the segment pieces (segment, first row, end row) that hold its attention rows [0, pos) in order."""

    ids: list[int]
    state: tuple                 # the committed state's class and attributes (``kv`` comes back from ``pieces``)
    snap: Any
    specs: list[tuple]           # each other tensor's dtype, shape and byte offset
    chunks: list[torch.Tensor]
    layout: tuple
    kv: list                     # per layer None, or the cache's type and each buffer's row count
    pieces: list[tuple[Segment, int, int]]
    done: Any = None             # a CUDA event: the spill's copies have landed


def _pack(obj: Any, sources: list) -> Any:
    """``obj`` with each tensor a ``Slot`` into ``sources``."""

    if isinstance(obj, torch.Tensor):
        sources.append(obj)
        return Slot(len(sources) - 1)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_pack(x, sources) for x in obj)
    return obj


def _unpack(obj: Any, tensors: list[torch.Tensor]) -> Any:
    if isinstance(obj, Slot):
        return tensors[obj.index]
    if isinstance(obj, (list, tuple)):
        return type(obj)(_unpack(x, tensors) for x in obj)
    return obj


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def _aligned(nbytes: int) -> int:
    return -(-nbytes // ALIGN) * ALIGN


def _widths(layout: tuple) -> list[int]:
    """Bytes per row of each attention buffer, layer by layer, K before V."""

    return [math.prod(shape) * dtype.itemsize for pair in layout if pair is not None for dtype, shape in pair]


def _offsets(widths: list[int], rows: int) -> tuple[list[int], int]:
    """Each buffer's byte offset in a segment of ``rows`` rows, and the segment's bytes."""

    offsets, size = [], 0
    for width in widths:
        offsets.append(size)
        size += _aligned(rows * width)
    return offsets, size


def _fits(into: list, layout: tuple, pos: int) -> bool:
    """Restore buffers aligned with a saved ``kv``: None where it has none, else contiguous buffers of its dtypes and
    row shapes holding at least ``pos`` rows."""

    return len(into) == len(layout) and all(
        got is None if pair is None else got is not None and len(got) == len(pair) and all(
            t.dtype == dtype and tuple(t.shape[1:]) == shape and t.shape[0] >= pos and t[:pos].is_contiguous()
            for t, (dtype, shape) in zip(got, pair)) for got, pair in zip(into, layout))


def _pinned(jobs: list[tuple]) -> bool:
    return all(c.is_pinned() for chunks in {id(job[1]): job[1] for job in jobs}.values() for c in chunks)


def _reach(pieces: list) -> int:
    return pieces[-1][2] if pieces else 0


def _common(a: Sequence[int], b: Sequence[int], n: int) -> int:
    """``a`` and ``b``'s common prefix length, at most ``n``: slices compared at C speed, doubling then halving."""

    n, lo, step = min(n, len(a), len(b)), 0, 64
    while lo < n:                                       # a[:lo] == b[:lo]
        hi = min(n, lo + step)
        if a[lo:hi] != b[lo:hi]:
            break
        lo, step = hi, 2 * step
    else:
        return n
    while hi - lo > 1:                                  # they differ below hi
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if a[lo:mid] == b[lo:mid] else (lo, mid)
    return lo


def _empty_host_cache() -> None:
    """Pinned blocks no tensor holds, back to the driver (``torch.accelerator.empty_host_cache``, or older builds'
    ``torch._C._host_emptyCache``; without either the caching allocator keeps them for reuse)."""

    empty = (getattr(getattr(torch, "accelerator", None), "empty_host_cache", None)
             or getattr(torch._C, "_host_emptyCache", None))
    if empty is not None:
        empty()


class HostTier:
    """Evicted prompt-end states in at most ``budget`` bytes of host chunks, the oldest out first, taken back whole.

    Attention rows below ``pos`` live in ``Segment``s shared by prefix: a spill copies only the rows no cached segment
    holds for its ids, into one new segment, and a taken entry's segments stay cached for its next turn's spill to
    chain onto. Every other tensor, the drafter snapshot's too, is copied byte for byte per entry. Room comes from
    unreferenced segments, least recently used first, then whole entries, oldest first. Chunks are pinned until the
    host refuses more.
    """

    def __init__(self, budget: int, device: Any = "cuda", *, chunk: int = CHUNK, pin: bool | None = None) -> None:
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.chunk = int(chunk)
        self.capacity = max(0, int(budget)) // self.chunk      # chunks the budget holds
        self.pin = self.cuda if pin is None else bool(pin)
        self.entries: list[Spilled] = []                        # newest last
        self.segments: list[Segment] = []
        self.free: list[torch.Tensor] = []                      # chunks of entries and segments gone, reused first
        self.held = 0                                           # chunks allocated, in use or free
        self.clock = 0                                          # puts and takes so far: segments' last use
        self.to_host = self.from_host = 0                       # bytes copied each way so far
        self.side = None                                        # the copy stream
        self.flying: list = []                                  # spills' events since the last ``fence``
        self.reading = None                                     # the last restore's event: its chunks are free after it

    @property
    def used(self) -> int:
        return (self.held - len(self.free)) * self.chunk

    def stats(self) -> dict:
        """For server logs: entries, segments, bytes in use, and bytes copied to host and back so far."""

        return {"entries": len(self.entries), "segments": len(self.segments), "used": self.used,
                "to_host": self.to_host, "from_host": self.from_host}

    def longest(self, prompt: Sequence[int], than: int = 0) -> Spilled | None:
        """The longest entry the prompt strictly extends (one prompt token is left to prefill), past ``than`` tokens."""

        best = None
        for entry in self.entries:
            n = len(entry.ids)
            if than < n < len(prompt) and (best is None or n > len(best.ids)) and list(prompt[:n]) == entry.ids:
                best = entry
        return best

    def put(self, ids: Sequence[int], state: Any, snap: Any) -> bool:
        """Copy an entry the GPU cache let go into host chunks: attention rows past what cached segments hold for its
        ids into one new segment, the rest whole; room as the class says. False: the entry alone is too big."""

        ids, pos, kv = list(ids), state.pos, state.kv
        sources: list[torch.Tensor] = []
        skeleton = (type(state), {k: None if k == "kv" else _pack(v, sources) for k, v in vars(state).items()})
        snap = _pack(snap, sources)
        specs, size = [], 0
        for t in sources:
            specs.append((t.dtype, tuple(t.shape), size))
            size += _aligned(t.numel() * t.element_size())
        layout = tuple(None if pair is None else tuple((t.dtype, tuple(t.shape[1:])) for t in pair) for pair in kv)
        widths = _widths(layout)
        own = -(-size // self.chunk)
        self._remove([e for e in self.entries if e.ids == ids])
        if own + self._span(widths, pos) > self.capacity:
            return False
        pieces = self._cover(ids, layout, pos)
        while self.capacity - self.held + len(self.free) < own + self._span(widths, pos - _reach(pieces)):
            idle = [s for s in self.segments if not s.refs]
            if idle:
                self._drop(min(idle, key=lambda s: s.used))
            elif self.entries:
                self._remove(self.entries[:1])
            elif pieces:                                # the cover's segments crowd the entry out: its rows copied anew
                for seg, _, _ in pieces:
                    seg.refs -= 1
                pieces = []
            else:
                return False
        chunks = [self._chunk() for _ in range(own)]
        jobs = [(_flat(t), chunks, at) for t, (_, _, at) in zip(sources, specs)]
        start, fresh = _reach(pieces), None
        if start < pos and widths:
            blocks = [self._chunk() for _ in range(self._span(widths, pos - start))]
            fresh = Segment(ids[:pos], start, pos, layout, _offsets(widths, pos - start)[0], blocks, refs=1)
            self.segments.append(fresh)
            pieces.append((fresh, start, pos))
            buffers = [t for pair in kv if pair is not None for t in pair]
            jobs += [(_flat(t[start:pos]), blocks, at) for t, at in zip(buffers, fresh.offsets)]
        self.clock += 1
        for seg, _, _ in pieces:
            seg.used = self.clock
        rows = [None if pair is None else (type(pair), tuple(t.shape[0] for t in pair)) for pair in kv]
        entry = Spilled(ids, skeleton, snap, specs, chunks, layout, rows, pieces)
        entry.done = self._spill(jobs)
        if fresh is not None:
            fresh.done = entry.done
        self.entries.append(entry)
        return True

    def take(self, prompt: Sequence[int], than: int = 0, into: list | None = None) -> tuple[list[int], Any, Any] | None:
        """``longest``'s entry as (ids, state, snapshot) on the device; it leaves the tier, its segments stay cached.
        Attention rows below ``pos`` go into ``into`` (aligned with ``kv``: None, or contiguous (K, V) buffers of at
        least ``pos`` rows, which the state then holds, rows past ``pos`` untouched; ``fence`` first if a spill in
        flight reads them), else into buffers of the old row count; every other tensor into tensors of its own."""

        entry = self.longest(prompt, than)
        if entry is None:
            return None
        cls, attrs = entry.state
        pos = attrs["pos"]
        if into is None:
            kv = [None if meta is None else meta[0](torch.empty((n, *shape), dtype=dtype, device=self.device)
                                                    for n, (dtype, shape) in zip(meta[1], pair))
                  for meta, pair in zip(entry.kv, entry.layout)]
        elif not _fits(into, entry.layout, pos):
            raise ValueError(f"the restore buffers must match the saved attention caches, {pos} rows or more each")
        else:
            kv = list(into)
        tensors = [torch.empty(shape, dtype=dtype, device=self.device) for dtype, shape, _ in entry.specs]
        jobs = [(_flat(t), entry.chunks, at) for t, (_, _, at) in zip(tensors, entry.specs)]
        buffers, widths = [t for pair in kv if pair is not None for t in pair], _widths(entry.layout)
        for seg, a, b in entry.pieces:
            jobs += [(_flat(t[a:b]), seg.chunks, at + (a - seg.start) * width)
                     for t, at, width in zip(buffers, seg.offsets, widths)]
        current = torch.cuda.current_stream(self.device) if self.cuda else None
        if current is not None:
            for event in [entry.done] + [seg.done for seg, _, _ in entry.pieces]:
                current.wait_event(event)
        self._copy(jobs, out=False)
        if current is not None:
            self.reading = torch.cuda.Event()
            self.reading.record(current)
            if not _pinned(jobs):
                self.reading.synchronize()              # pageable chunks: read before any spill may reuse them
        self.clock += 1
        for seg, _, _ in entry.pieces:
            seg.used = self.clock
        self._remove([entry])
        state = object.__new__(cls)
        state.__dict__.update({k: kv if k == "kv" else _unpack(v, tensors) for k, v in attrs.items()})
        return entry.ids, state, _unpack(entry.snap, tensors)

    def fence(self) -> None:
        """Later work on the current stream waits for spills in flight: a resumed prefill may overwrite their rows."""

        if self.cuda and self.flying:
            current = torch.cuda.current_stream(self.device)
            for event in self.flying:
                current.wait_event(event)
        self.flying = []

    def _cover(self, ids: list[int], layout: tuple, pos: int) -> list[tuple[Segment, int, int]]:
        """Cached segments holding the entry's rows from row 0 on, each step the one reaching furthest (its end, the
        ids it agrees on, ``pos``); each gains a reference. Rows past the last one's reach are left to copy."""

        pieces, covered, reach = [], 0, {}
        while True:
            best, end = None, covered
            for seg in self.segments:
                if seg.start <= covered < min(seg.end, pos) and seg.layout == layout:
                    if id(seg) not in reach:
                        reach[id(seg)] = _common(seg.ids, ids, min(seg.end, pos))
                    if reach[id(seg)] > end:
                        best, end = seg, reach[id(seg)]
            if best is None:
                return pieces
            best.refs += 1
            pieces.append((best, covered, end))
            covered = end

    def _spill(self, jobs: list[tuple]) -> Any:
        """Device to host, on the side stream after the work that wrote the sources; the allocator keeps their memory
        until the copies land (``record_stream``), so dropping the GPU entry frees nothing early. Returns the event."""

        if not self.cuda:
            self._copy(jobs, out=True)
            return None
        if self.side is None:
            self.side = torch.cuda.Stream(self.device)
        self.side.wait_stream(torch.cuda.current_stream(self.device))
        if self.reading is not None:                    # a restore may still read chunks this spill reuses
            self.side.wait_event(self.reading)
        with torch.cuda.stream(self.side):
            for flat, _, _ in jobs:
                flat.record_stream(self.side)
            self._copy(jobs, out=True)
            done = torch.cuda.Event()
            done.record(self.side)
        if not _pinned(jobs):
            done.synchronize()                          # pageable chunks: written before this returns
        self.flying = [e for e in self.flying if not e.query()] + [done]
        return done

    def _copy(self, jobs: list[tuple], out: bool) -> None:
        """Each job's (device bytes, chunks, byte offset in them): to the chunks (``out``) or back, split only where a
        chunk ends."""

        for flat, chunks, offset in jobs:
            for k, at, start, n in self._pieces(offset, flat.numel()):
                if out:
                    chunks[k][at:at + n].copy_(flat[start:start + n], non_blocking=True)
                else:
                    flat[start:start + n].copy_(chunks[k][at:at + n], non_blocking=True)
        moved = sum(flat.numel() for flat, _, _ in jobs)
        if out:
            self.to_host += moved
        else:
            self.from_host += moved

    def _span(self, widths: list[int], rows: int) -> int:
        """Chunks a segment of ``rows`` rows takes."""

        return -(-_offsets(widths, rows)[1] // self.chunk)

    def _pieces(self, offset: int, nbytes: int):
        """(chunk, start in it, start in the tensor, bytes) spans holding bytes [offset, offset + nbytes) of chunks."""

        done = 0
        while done < nbytes:
            k, at = divmod(offset + done, self.chunk)
            n = min(self.chunk - at, nbytes - done)
            yield k, at, done, n
            done += n

    def _remove(self, gone: list[Spilled]) -> None:
        """Entries leave: their chunks are free, their segments lose a reference (and stay cached)."""

        for entry in gone:
            self.entries = [e for e in self.entries if e is not entry]
            self.free += entry.chunks
            for seg, _, _ in entry.pieces:
                seg.refs -= 1

    def _drop(self, seg: Segment) -> None:
        self.segments = [s for s in self.segments if s is not seg]
        self.free += seg.chunks

    def reserve(self, slab: int = GIB) -> int:
        """Pin the whole budget now, in slabs of power-of-two sizes (the pinned allocator rounds a size up to one)
        carved into chunks: pinning 16 MiB at a time ran at 1.5 GiB/s on an R9700 host, 1 GiB at a time at 3.6, and a
        request should not wait for either. When the host refuses a slab, at least ``slab`` bytes of the last ones go
        back to it: the decode path pins small staging buffers of its own every round. Returns the bytes pinned; the
        rest stays for ``_chunk`` (pageable)."""

        if not self.pin:
            return 0
        start, slabs = self.held, []                            # chunks of each slab pinned, in order
        while self.pin and self.held < self.capacity:
            size = min(slab, 1 << ((self.capacity - self.held) * self.chunk).bit_length() - 1)
            if size < self.chunk:
                break
            try:
                block = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            except RuntimeError as exc:
                self._refused(exc, self._give_back(slabs, slab))
                break
            pieces = [c for c in block.split(self.chunk) if c.numel() == self.chunk]
            self.free += pieces
            self.held += len(pieces)
            slabs.append(len(pieces))
            del block, pieces                                   # the free chunks alone hold a slab
        return (self.held - start) * self.chunk

    def _give_back(self, slabs: list[int], least: int) -> int:
        """Unpin the last slabs ``reserve`` pinned, at least ``least`` bytes of them while any are left: their chunks
        leave ``free`` and ``held`` (``_chunk`` makes them pageable), and the host cache returns their memory to the
        driver. Returns the bytes given back."""

        gone = 0
        while slabs and gone < least:
            n = slabs.pop()
            del self.free[-n:]
            self.held -= n
            gone += n * self.chunk
        if gone:
            _empty_host_cache()
        return gone

    def _chunk(self) -> torch.Tensor:
        """A free chunk, else a new one: pinned until the host first refuses, pageable from then on."""

        if self.free:
            return self.free.pop()
        if self.pin:
            try:
                chunk = self._alloc(True)
            except RuntimeError as exc:
                self._refused(exc)
            else:
                self.held += 1
                return chunk
        self.held += 1
        return self._alloc(False)

    def _refused(self, exc: RuntimeError, back: int = 0) -> None:
        self.pin = False
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        print(f"[tensorfold] RAM tier: the host pinned {(self.held * self.chunk + back) / GIB:.1f} GiB, then refused "
              f"more ({reason})" + (f"; {back / GIB:.1f} GiB of it goes back for other pinned buffers" if back else "")
              + "; the rest is pageable, its copies synchronous (`ulimit -l` may be the cap)", flush=True)

    def _alloc(self, pinned: bool) -> torch.Tensor:
        return torch.empty(self.chunk, dtype=torch.uint8, pin_memory=pinned)
