"""GLM-5.3-Flash's per-token state as one shared pool: arenas of rows every stream draws from, and the extents that cut
them up.

An ``Arena`` holds planes of device rows (per DSA layer its latents; per indexed layer its index keys, gates and
pooled keys; the MTP layer's), each ``rows // div + pad`` rows: latents div 1, pooled keys (a pool of 4 tokens) div 4
with 2 rows of pad (the token selection reads up to 2 pools past a window). What a plane holds and in which element
type (bf16, FP8 with scales, a ring) is the cache module's business (``forward.Caches``); the arena only views,
copies and places rows.

An extent [base, base + size), both multiples of ``ALIGN``, views every plane at [base // div, (base + size) // div +
pad): the single-stream code (prefill, decode windows, snapshots) runs on those views exactly as on tensors of its
own. The pad rows of one extent's pooled keys are the next extent's first rows: they are only ever read (and masked),
never written (``sparse._pool_keys`` writes complete pools, all inside the extent).

``Pool`` is the host-side bookkeeping of the extents (no device work): lowest-base first fit, growth in place,
placement for relocations. Both ranks apply the same operations in the same order (rank 0 decides, rank 1 applies
what it is sent), so their pools stay equal."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

ALIGN = 2048          # extents start and end on multiples of this many tokens (the pooled keys at base / 4)


def align_up(n: int, align: int = ALIGN) -> int:
    return -(-int(n) // align) * align


class Plane:
    """One cache tensor over the arena's rows: row r of token t at t // div (``div`` tokens a row), ``pad`` rows
    past the last token's."""

    def __init__(self, tensor, div: int = 1, pad: int = 0) -> None:
        self.tensor, self.div, self.pad = tensor, int(div), int(pad)

    def rows_for(self, tokens: int) -> int:
        return tokens // self.div

    def view(self, base: int, size: int):
        return self.tensor[base // self.div:(base + size) // self.div + self.pad]


class Arena:
    """Planes over ``rows`` tokens (a multiple of ``ALIGN`` when shared by several extents)."""

    def __init__(self, rows: int, planes: list[Plane]) -> None:
        self.rows = int(rows)
        self.planes = planes
        for p in planes:
            want = self.rows // p.div + p.pad
            if p.tensor.shape[0] != want:
                raise ValueError(f"a plane of {p.tensor.shape[0]} rows over {self.rows} tokens (div {p.div}, pad "
                                 f"{p.pad}): want {want}")

    def view(self, i: int, base: int, size: int):
        if base < 0 or base + size > self.rows:
            raise ValueError(f"extent [{base}, {base + size}) outside the arena's {self.rows} tokens")
        return self.planes[i].view(base, size)

    def nbytes(self) -> int:
        return sum(p.tensor.numel() * p.tensor.element_size() for p in self.planes)

    def token_bytes(self) -> float:
        """Device bytes a token of every plane takes (the pad aside)."""
        return sum(p.tensor[0].numel() * p.tensor.element_size() / p.div for p in self.planes)

    def copy(self, src: int, dst: int, n: int) -> None:
        """Tokens [src, src + n) of every plane to [dst, dst + n): a whole last row of a plane whose rows hold
        several tokens (``div``) goes too (its tokens past n are the source's, which nothing reads). Overlapping
        ranges copy in pieces that never read a row already written."""

        if n <= 0 or src == dst:
            return
        if min(src, dst) < 0 or max(src, dst) + n > self.rows:
            raise ValueError(f"a copy of {n} tokens from {src} to {dst} reaches outside the arena's {self.rows}")
        for p in self.planes:
            a, b = src // p.div, dst // p.div
            m = -(-n // p.div)
            m = min(m, p.tensor.shape[0] - max(a, b))
            _move(p.tensor, a, b, m)


def _move(t, a: int, b: int, m: int) -> None:
    """t[b:b + m] = t[a:a + m] (rows), safe when they overlap."""

    gap = abs(a - b)
    if gap >= m:
        t[b:b + m].copy_(t[a:a + m])
        return
    step = gap
    starts = range(0, m, step) if b < a else reversed(range(0, m, step))
    for s in starts:                       # down: front to back; up: back to front
        k = min(step, m - s)
        t[b + s:b + s + k].copy_(t[a + s:a + s + k])


@dataclass(eq=False)
class Extent:
    """Pool rows [base, base + size). ``owner``: the live stream writing into it (its sid), or None; ``kept``: the
    kept prompts whose rows it holds (each a prefix of the rows)."""

    base: int
    size: int
    eid: int = 0
    owner: int | None = None
    kept: list = field(default_factory=list)

    @property
    def end(self) -> int:
        return self.base + self.size

    def free(self) -> bool:
        return self.owner is None and not self.kept


class Pool:
    """Extents over ``rows`` pool rows, host side only. Placement is lowest base first; every call is deterministic
    in the calls before it, so two ranks making the same calls hold the same extents."""

    def __init__(self, rows: int, align: int = ALIGN) -> None:
        if rows <= 0 or rows % align:
            raise ValueError(f"a pool of {rows} rows: a positive multiple of {align}")
        self.rows, self.align = int(rows), int(align)
        self.extents: list[Extent] = []           # by base
        self.next_id = 0

    # -- queries -------------------------------------------------------------------------------------------------
    def gaps(self, ignore: Iterable[Extent] = ()) -> list[tuple[int, int]]:
        """Free (base, size) ranges by base, the extents in ``ignore`` counted as free."""

        skip = set(id(x) for x in ignore)
        out, at = [], 0
        for x in self.extents:
            if id(x) in skip:
                continue
            if x.base > at:
                out.append((at, x.base - at))
            at = max(at, x.end)
        if at < self.rows:
            out.append((at, self.rows - at))
        return _merge(out)

    def free_rows(self) -> int:
        return self.rows - sum(x.size for x in self.extents)

    def place(self, size: int, ignore: Iterable[Extent] = ()) -> int | None:
        """The lowest base with ``size`` free rows (``ignore``'s rows free), or None."""

        size = align_up(size, self.align)
        for base, n in self.gaps(ignore):
            if n >= size:
                return base
        return None

    def room_after(self, x: Extent) -> int:
        """Free rows right after x (before the next extent or the pool's end)."""

        nxt = min((y.base for y in self.extents if y.base >= x.end and y is not x), default=self.rows)
        return nxt - x.end

    def get(self, eid: int) -> Extent:
        for x in self.extents:
            if x.eid == eid:
                return x
        raise KeyError(f"no extent {eid}")

    # -- changes -------------------------------------------------------------------------------------------------
    def add(self, base: int, size: int) -> Extent:
        """A new extent at exactly [base, base + size) (rank 0's placement, applied on both ranks)."""

        size = align_up(size, self.align)
        if base % self.align or base < 0 or base + size > self.rows:
            raise ValueError(f"extent [{base}, {base + size}) is not an aligned range of the {self.rows}-row pool")
        if any(x.base < base + size and base < x.end for x in self.extents):
            raise ValueError(f"extent [{base}, {base + size}) overlaps another")
        x = Extent(base, size, self.next_id)
        self.next_id += 1
        self.extents.append(x)
        self.extents.sort(key=lambda y: y.base)
        return x

    def alloc(self, size: int) -> Extent | None:
        base = self.place(size)
        return None if base is None else self.add(base, size)

    def resize(self, x: Extent, size: int) -> None:
        """Grow in place (the rows after it free) or shrink (from its end)."""

        size = align_up(size, self.align)
        if size > x.size and size - x.size > self.room_after(x):
            raise ValueError(f"extent {x.eid} cannot grow to {size} rows in place")
        if size <= 0:
            raise ValueError("an extent keeps at least one aligned block")
        x.size = size

    def move(self, x: Extent, base: int, size: int | None = None) -> int:
        """Place x at ``base`` (with ``size`` rows): the bookkeeping only (the caller copies the rows). Returns the old
        base."""

        size = x.size if size is None else align_up(size, self.align)
        if base % self.align or base < 0 or base + size > self.rows:
            raise ValueError(f"extent [{base}, {base + size}) is not an aligned range of the pool")
        if any(y is not x and y.base < base + size and base < y.end for y in self.extents):
            raise ValueError(f"extent {x.eid} cannot move to [{base}, {base + size}): taken")
        old = x.base
        x.base, x.size = base, size
        self.extents.sort(key=lambda y: y.base)
        return old

    def remove(self, x: Extent) -> None:
        self.extents = [y for y in self.extents if y is not x]


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for base, n in ranges:
        if out and out[-1][0] + out[-1][1] == base:
            out[-1] = (out[-1][0], out[-1][1] + n)
        elif n > 0:
            out.append((base, n))
    return out
