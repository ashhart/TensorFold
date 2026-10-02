"""Multi-stream decode windows: several streams' rows back to back ([s1: rows][s2: rows]...), each stream's per-token
DSA state in its own extent of shared arenas (latents [P, 512], index keys and gates [P, 128], pooled keys
[P / 4 + 2, 128]; extents aligned to EXTENT tokens). ``SegRows`` holds the per-row and per-segment device tables the
segmented kernels read (latent.seg_*, sparse.seg_*); ``SelectScratch`` the token selection's buffers. Static
buffers, so one CUDA graph per window size covers every mix of positions and context lengths."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .sparse import POOL, SPARSE_FROM, TOKENS, split_scratch

MAX_SEGS = 4          # streams a window holds
EXTENT = 2048         # a stream's extent starts at a multiple of this many tokens (its pooled keys at base / 4)


class SegRows:
    """Device tables of a window of up to ``rows`` rows in up to ``max_segs`` segments, one int32 tensor:

    per row: ``pos`` (the row's position in its stream), ``base`` (its stream's first arena row), ``seg`` (segment
    index), ``sparse`` (1: past the dense limit, attends to its selected tokens; 0: dense over keys 0 .. pos);
    per segment: ``seg_start`` (first row), ``seg_rows`` (0: unused), ``seg_pbase`` (pooled-key base, base / 4),
    ``seg_pools`` (pools its last row sees when it has sparse rows, else 0: what the selection scores)."""

    def __init__(self, rows: int, device, max_segs: int = MAX_SEGS) -> None:
        self.rows, self.max_segs = rows, max_segs
        n = 4 * rows + 4 * max_segs
        pin = torch.cuda.is_available() and torch.device(device).type == "cuda"
        self.host = torch.zeros((n,), dtype=torch.int32, pin_memory=pin)
        self.dev = torch.zeros((n,), dtype=torch.int32, device=device)
        self.pos, self.base, self.seg, self.sparse = (self.dev[i * rows:(i + 1) * rows] for i in range(4))
        o = 4 * rows
        self.seg_start, self.seg_rows, self.seg_pbase, self.seg_pools = (
            self.dev[o + i * max_segs:o + (i + 1) * max_segs] for i in range(4))
        self.staged = torch.cuda.Event() if pin else None
        self.R = 0
        self.segments: list[tuple[int, int, int]] = []

    def fill(self, segments: Sequence[tuple[int, int, int]], *, sparse_from: int | None = SPARSE_FROM):
        """The host tables for ``segments`` [(base, pos, rows)] in window order (numpy int32, the device layout).
        ``sparse_from``: the first position that attends sparsely (None: no index, every row dense)."""

        if not 1 <= len(segments) <= self.max_segs:
            raise ValueError(f"a window of {len(segments)} segments, at most {self.max_segs}")
        R = sum(n for _, _, n in segments)
        if R > self.rows:
            raise ValueError(f"a window of {R} rows, the tables hold {self.rows}")
        rows, S = self.rows, self.max_segs
        t = torch.zeros((4 * rows + 4 * S,), dtype=torch.int32).numpy()
        r = 0
        for s, (base, pos, n) in enumerate(segments):
            if n < 1 or pos < 0 or base < 0 or base % EXTENT:
                raise ValueError(f"segment {s}: base {base} (a multiple of {EXTENT}), position {pos}, {n} rows")
            for i in range(n):
                q = pos + i
                t[r + i] = q
                t[rows + r + i] = base
                t[2 * rows + r + i] = s
                t[3 * rows + r + i] = int(sparse_from is not None and q >= sparse_from)
            last = pos + n - 1
            t[4 * rows + s] = r
            t[4 * rows + S + s] = n
            t[4 * rows + 2 * S + s] = base // POOL
            t[4 * rows + 3 * S + s] = (last + 1) // POOL if sparse_from is not None and last >= sparse_from else 0
            r += n
        return t, R

    def set(self, segments: Sequence[tuple[int, int, int]], *, sparse_from: int | None = SPARSE_FROM) -> int:
        """Stage ``segments`` [(base, pos, rows)] into the device tables (a pinned copy, ordered on the current
        stream like forward.stage's token ids); returns the window's rows."""

        t, R = self.fill(segments, sparse_from=sparse_from)
        if self.staged is not None:
            self.staged.synchronize()          # the previous copy has left the pinned buffer
        self.host.numpy()[:] = t
        self.dev.copy_(self.host, non_blocking=True)
        if self.staged is not None:
            self.staged.record()
        self.R, self.segments = R, list(segments)
        return R

    def any_sparse(self) -> bool:
        """Host view of the last ``set``: does any row attend sparsely."""
        return bool(self.host[3 * self.rows:3 * self.rows + self.R].any())


class SelectScratch:
    """The segmented selection's buffers: fp32 scores [rows, pools] (a stream's pools at most: its largest context
    / 4), tokens [rows, 2051] int32, counts [rows] int32, the split selection's scratch."""

    def __init__(self, rows: int, max_context: int, device) -> None:
        self.rows = rows
        pools = max(1, -(-max_context // POOL))
        self.scores = torch.empty((rows, pools), dtype=torch.float32, device=device)
        self.tokens = torch.full((rows, TOKENS), -1, dtype=torch.int32, device=device)
        self.counts = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.split = split_scratch(rows, pools, device)
