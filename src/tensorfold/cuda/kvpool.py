"""Fixed pages of cache rows shared by every lane: a lane maps its logical pages to physical ones through its table."""
# Allocation is lowest-free-page first, so two ranks making the same calls in the same order hold the same pages.

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .memory_gate import NoRoom


@dataclass(frozen=True)
class Plane:
    """One cache a page holds rows of: ``per_tokens`` tokens share a row, written only once all of them are known."""

    name: str
    row_bytes: int
    per_tokens: int = 1


def pages_for(tokens: int, page_tokens: int) -> int:
    return -(-max(int(tokens), 0) // page_tokens)


class PagePool:
    """Every plane's rows in one buffer each, ``pages`` pages plus a null page that unmapped table entries name."""

    def __init__(self, planes: Sequence[Plane], pages: int, page_tokens: int, device: Any = None) -> None:
        if pages < 1 or page_tokens < 1:
            raise ValueError("a page pool needs at least one page of at least one token")
        for p in planes:
            if page_tokens % p.per_tokens:
                raise ValueError(f"plane {p.name}: {p.per_tokens} tokens a row don't divide a {page_tokens}-token page")
        self.planes = {p.name: p for p in planes}
        self.pages, self.page_tokens, self.device = int(pages), int(page_tokens), device
        self.null = self.pages  # all zeros: a stray read of an unmapped page reads nothing
        self.heap = list(range(self.pages))
        self.refs: dict[int, int] = {}  # holders of a page past its first (lanes and cache entries)
        self.tables: list[PageTable] = []
        self.buffers = {p.name: self._zeros((self.pages + 1) * self.rows_per_page(p.name), p.row_bytes) for p in planes}

    def _zeros(self, rows: int, width: int, dtype: str = "uint8"):
        if self.device is None:
            return np.zeros((rows, width), dtype=dtype)
        import torch

        return torch.zeros((rows, width), dtype=getattr(torch, dtype), device=self.device)

    @property
    def free(self) -> int:
        return len(self.heap)

    def outstanding(self, exclude: PageTable | None = None) -> int:
        """Pages admitted lanes reserved and have not mapped yet."""

        return sum(t.outstanding for t in self.tables if t is not exclude)

    def available(self) -> int:
        """Free pages no admitted lane has reserved: what a new admission may count on."""

        return self.free - self.outstanding()

    def rows_per_page(self, name: str) -> int:
        return self.page_tokens // self.planes[name].per_tokens

    def view(self, name: str):
        """The plane's buffer [(pages + 1) x rows a page, row bytes] uint8; the last page is the null page."""

        return self.buffers[name]

    def need(self, tokens: int) -> int:
        return pages_for(tokens, self.page_tokens)

    def page_bytes(self) -> int:
        return sum(p.row_bytes * self.rows_per_page(n) for n, p in self.planes.items())

    def table(self, tokens: int) -> PageTable:
        """A lane's table for up to ``tokens`` positions; its device table keeps its address for the pool's life."""

        t = PageTable(self, tokens)
        self.tables.append(t)
        return t

    def take(self, k: int) -> list[int]:
        if k > len(self.heap):
            raise NoRoom(f"the page pool has {len(self.heap)} free pages of {self.pages}, {k} asked")
        return [heapq.heappop(self.heap) for _ in range(k)]

    def hold(self, pages: Sequence[int]) -> None:
        """One more holder of each page (a cache entry keeping a lane's prompt pages)."""

        for p in pages:
            self.refs[int(p)] = self.refs.get(int(p), 0) + 1

    def drop(self, pages: Sequence[int]) -> int:
        """One holder fewer of each page; a page nobody holds goes back to the free list. Returns pages freed."""

        freed = 0
        for p in map(int, pages):
            n = self.refs.get(p, 0)
            if n > 1:
                self.refs[p] = n - 1
            elif n == 1:
                del self.refs[p]
            else:
                heapq.heappush(self.heap, p)
                freed += 1
        return freed

    def holders(self, page: int) -> int:
        return 1 + self.refs.get(int(page), 0)

    def copy_page(self, src: int, dst: int) -> None:
        for name, buf in self.buffers.items():
            n = self.rows_per_page(name)
            buf[dst * n : (dst + 1) * n] = buf[src * n : (src + 1) * n]

    def zero_pages(self, pages: Sequence[int]) -> None:
        for name, buf in self.buffers.items():
            n = self.rows_per_page(name)
            for p in pages:
                buf[int(p) * n : (int(p) + 1) * n] = 0

    def read_pages(self, pages: Sequence[int]) -> dict[str, np.ndarray]:
        """Each plane's rows of ``pages`` in order, on the host."""

        out = {}
        for name, buf in self.buffers.items():
            n = self.rows_per_page(name)
            parts = [buf[p * n : (p + 1) * n] for p in pages]
            if self.device is None:
                out[name] = np.concatenate(parts) if parts else buf[:0].copy()
            else:
                import torch

                out[name] = (torch.cat(parts) if parts else buf[:0]).cpu().numpy()
        return out

    def write_pages(self, pages: Sequence[int], data: dict[str, np.ndarray]) -> None:
        """``read_pages``' arrays back into ``pages`` (mapped by the caller)."""

        for name, buf in self.buffers.items():
            n = self.rows_per_page(name)
            rows = data[name]
            if len(rows) != n * len(pages):
                raise ValueError(f"plane {name}: {len(rows)} rows for {len(pages)} pages of {n}")
            if self.device is not None:
                import torch

                rows = torch.from_numpy(np.ascontiguousarray(rows)).to(buf.device)
            for i, p in enumerate(pages):
                buf[p * n : (p + 1) * n] = rows[i * n : (i + 1) * n]

    def null_clean(self) -> bool:
        """Nothing wrote the null page."""

        for name, buf in self.buffers.items():
            n = self.rows_per_page(name)
            if bool((buf[self.null * n : (self.null + 1) * n] != 0).any()):
                return False
        return True

    def describe(self) -> str:
        return f"{self.free} of {self.pages} pages free, {self.outstanding()} reserved"


class PageTable:
    """One lane's pages: ``pages[i]`` holds positions [i x page, (i + 1) x page); ``device`` is the kernels' copy."""

    def __init__(self, pool: PagePool, tokens: int) -> None:
        self.pool, self.tokens = pool, int(tokens)
        self.most = pages_for(tokens, pool.page_tokens)
        self.pages: list[int] = []
        self.quota = 0  # pages the lane's admission reserved
        self.device = pool._zeros(1, self.most, "int32")[0]
        self.device[:] = pool.null

    @property
    def outstanding(self) -> int:
        return max(0, self.quota - len(self.pages))

    def _write(self, at: int, values: Sequence[int]) -> None:
        if values:
            self.device[at : at + len(values)] = (
                np.asarray(values, dtype=np.int32) if self.pool.device is None else _tensor(values, self.device)
            )

    def reserve(self, pages: int) -> None:
        """Hold ``pages`` for this lane: later admissions count them as taken."""

        self.quota = int(pages)

    def ensure(self, end: int) -> None:
        """Map every page holding a position below ``end``; past the quota only pages nobody reserved (else NoRoom)."""

        need = pages_for(end, self.pool.page_tokens)
        have = len(self.pages)
        if need <= have:
            return
        if need > self.most:
            raise ValueError(f"position {end} is past the lane's {self.tokens} tokens")
        extra = need - max(self.quota, have)
        if extra > 0 and extra > self.pool.free - self.pool.outstanding(exclude=self):
            raise NoRoom(f"a lane needs {need} pages; its admission reserved {self.quota}")
        new = self.pool.take(need - have)
        self.pages.extend(new)
        self._write(have, new)

    def truncate(self, keep: int) -> int:
        """Let go of the pages past position ``keep``; returns the pages freed."""

        k = pages_for(keep, self.pool.page_tokens)
        if k >= len(self.pages):
            return 0
        gone = self.pages[k:]
        del self.pages[k:]
        self._write(k, [self.pool.null] * len(gone))
        return self.pool.drop(gone)

    def release(self) -> int:
        self.quota = 0
        return self.truncate(0)

    def adopt(self, pages: Sequence[int], tokens: int) -> None:
        """Map a cache entry's rows below ``tokens``: its full pages shared, its partial last page copied."""

        if self.pages:
            raise ValueError("a lane adopts an entry's pages only when it maps none")
        full = int(tokens) // self.pool.page_tokens
        shared = [int(p) for p in pages[:full]]
        self.pool.hold(shared)
        self.pages.extend(shared)
        if int(tokens) % self.pool.page_tokens:  # the lane writes into this page: its own copy
            new = self.pool.take(1)[0]
            self.pool.copy_page(int(pages[full]), new)
            self.pages.append(new)
        self._write(0, self.pages)

    def share(self, tokens: int) -> list[int]:
        """The pages holding positions below ``tokens``, each with one more holder (a cache entry's)."""

        pages = self.pages[: pages_for(tokens, self.pool.page_tokens)]
        if len(pages) < pages_for(tokens, self.pool.page_tokens):
            raise ValueError(f"the lane maps {len(self.pages)} pages, {tokens} tokens need more")
        self.pool.hold(pages)
        return list(pages)


def _tensor(values: Sequence[int], like):
    import torch

    return torch.tensor(list(values), dtype=torch.int32).to(like.device)


__all__ = ["PagePool", "PageTable", "Plane", "pages_for"]
