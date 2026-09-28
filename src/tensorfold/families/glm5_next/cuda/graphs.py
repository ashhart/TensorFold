"""CUDA graphs preserve eager bits by replaying static buffers, device positions, and collectives, with empty attention chunks skipped."""

from __future__ import annotations

import torch

from . import latent, prof
from .forward import compute
from .mtp import mtp_compute


def pool_buckets(np_max: int) -> list[int]:
    """Every value ``sparse.pool_bucket`` can return for a sparse step (past TOPK_POOLS complete pools)."""
    out, b = [], 1024
    while b < np_max:
        out.append(b)
        b *= 2
    return out + [np_max]


class Graphs:
    def __init__(self, e, main_rows=(1, 2, 3, 4), mtp_rows=(1, 2, 3, 4)) -> None:
        self.pool = torch.cuda.graph_pool_handle()
        self.main: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mtp: dict[int, torch.cuda.CUDAGraph] = {}
        self.sparse: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}      # (rows, parity, pool bucket)
        self.sparse_mtp: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}       # (rows, pool bucket)
        w, st = e.w, e.st
        prof.active = False
        saved = list(st.cur)
        parities = (0, 1) if st.cur else (0,)
        with torch.no_grad():
            for R in main_rows:
                for parity in parities:
                    st.cur = [parity] * len(st.cur)
                    for _ in range(2):
                        compute(w, st, e.buf, R)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        compute(w, st, e.buf, R)
                    self.main[(R, parity)] = g
            st.cur = saved
            if w.mtp is not None:
                e.mbuf.zero_first = False
                for n in mtp_rows:
                    for _ in range(2):
                        mtp_compute(w, st, e.mbuf, n)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        mtp_compute(w, st, e.mbuf, n)
                    self.mtp[n] = g
            if st.index is not None and latent.ENABLED:
                self._capture_sparse(e, main_rows, mtp_rows, parities)
        torch.cuda.synchronize()

    def _capture_sparse(self, e, main_rows, mtp_rows, parities) -> None:
        w, st = e.w, e.st
        np_max = st.index[0][2].shape[0] - 2
        saved = list(st.cur)
        for bucket in pool_buckets(np_max):
            for R in main_rows:
                for parity in parities:
                    st.cur = [parity] * len(st.cur)
                    for _ in range(2):
                        compute(w, st, e.buf, R, sparse_np=bucket)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        compute(w, st, e.buf, R, sparse_np=bucket)
                    self.sparse[(R, parity, bucket)] = g
            if w.mtp is not None:
                for n in mtp_rows:
                    for _ in range(2):
                        mtp_compute(w, st, e.mbuf, n, sparse_np=bucket)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        mtp_compute(w, st, e.mbuf, n, sparse_np=bucket)
                    self.sparse_mtp[(n, bucket)] = g
        st.cur = saved
