"""CUDA graphs for GLM-5.3-Flash decode steps: one per (window rows, KDA state parity) for the model, one per row
count for the MTP head. A captured step replays the eager kernels with their arguments (static buffers,
positions read on the device), so its bits equal the eager step's; attention visits every chunk the cache
can hold (empty chunks are skipped by the merge). Collectives (NCCL on the current stream) are captured too.
"""

from __future__ import annotations

import torch

from .forward import compute
from .mtp import mtp_compute


class Graphs:
    def __init__(self, e, main_rows=(1, 2, 3, 4), mtp_rows=(1, 2, 3, 4)) -> None:
        self.pool = torch.cuda.graph_pool_handle()
        self.main: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mtp: dict[int, torch.cuda.CUDAGraph] = {}
        w, st = e.w, e.st
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
        torch.cuda.synchronize()
