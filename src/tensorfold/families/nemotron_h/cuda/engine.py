"""Nemotron-H's verify forward on CUDA: a window row gets the serial step's bits; ``meta`` makes graphs reusable."""

from __future__ import annotations

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.kernels import prefill_attention
from tensorfold.families.qwen3_5.cuda import glue as base

from . import attention as A, glue as G, mamba as M, sampler as S
from .weights import Attention, MoE, Weights

ROWS = 16            # the window limit (and the row tile of the row-parallel kernels)
PREFILL_ROWS = 2048  # rows of a prompt chunk


def sampling_mode(sampling) -> tuple:
    """The part of a Sampling compiled into the graphs: greedy, or keyed with its top_k, top-p cut and min-p cut."""

    if sampling is None or sampling.temperature <= 0:
        return ("greedy",)
    return ("keyed", int(sampling.top_k), 0.0 < float(sampling.top_p) < 1.0, float(sampling.min_p) > 0.0)


class Engine:
    STATE = ("k_cache", "v_cache", "ssm", "conv_base", "raw", "xc", "dt")

    def __init__(self, w: Weights, *, max_len: int = 8192, max_rows: int = ROWS, graphs: bool = True,
                 prefill_rows: int = PREFILL_ROWS):
        if max_rows > ROWS:
            raise ValueError(f"windows hold at most {ROWS} rows")
        if max_len % A.CHUNK:
            raise ValueError(f"max_len must be a multiple of {A.CHUNK}")
        self.w = w
        c = self.c = w.config
        dev = self.device = w.norm_f.device
        self.max_len, self.max_rows, self.max_chunks = max_len, max_rows, max_len // A.CHUNK
        self.ns = c.slots
        nm, na = c.pattern.count("M"), c.pattern.count("*")
        self.meta = torch.zeros(4, dtype=torch.int32, device=dev)
        self.ids = torch.zeros(max_rows, dtype=torch.int32, device=dev)
        self.k_cache = torch.zeros((na, max_len, c.kv_heads, c.head_dim), dtype=torch.bfloat16, device=dev)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.ssm = torch.zeros((nm, c.m_heads, c.m_head_dim, c.m_state), dtype=torch.float32, device=dev)
        self.conv_base = torch.zeros((nm, c.conv_kernel - 1, c.conv_dim), dtype=torch.bfloat16, device=dev)
        self.raw = torch.zeros((nm, 2, max_rows, c.conv_dim), dtype=torch.bfloat16, device=dev)
        self.xc = torch.zeros_like(self.raw)
        self.dt = torch.zeros((nm, 2, max_rows, c.m_heads), dtype=torch.float32, device=dev)
        self.pick = torch.zeros((max_rows, self.ns), dtype=torch.int32, device=dev)
        self.wts = torch.zeros((max_rows, self.ns), dtype=torch.float32, device=dev)
        self.plan = grouped.Plan(max_rows, self.ns, c.experts + 2, dev)
        self.act = torch.empty((max_rows * self.ns * c.moe_width,), dtype=torch.bfloat16, device=dev)
        self.hidden = torch.zeros((max_rows, c.hidden), dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros((max_rows, c.vocab), dtype=torch.bfloat16, device=dev)
        self.sampled = torch.zeros(max_rows, dtype=torch.int32, device=dev)
        self.params = S.Params(dev)
        self.mode: tuple = ("greedy",)
        self._host_meta = torch.zeros(4, dtype=torch.int32).pin_memory()
        self._host_ids = torch.zeros(max_rows, dtype=torch.int32).pin_memory()
        self._host_sampled = torch.zeros(max_rows, dtype=torch.int32).pin_memory()
        self._sampled_ready = torch.cuda.Event()
        self._copied = torch.cuda.Event()       # the last copy from the pinned buffers has run
        self._copied.record()
        self.pos = self.parity = self.prev_keep = 0
        self.bias, self.masked = None, False    # a reply grammar's -inf per disallowed column, rows of the next window
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.pool = None
        self.use_graphs = graphs
        p = self.prefill_rows = prefill_rows
        self.p_ids = torch.zeros(p, dtype=torch.int32, device=dev)
        self.p_hidden = torch.zeros((p, c.hidden), dtype=torch.bfloat16, device=dev)
        self.p_xc = torch.zeros((p, c.conv_dim), dtype=torch.bfloat16, device=dev)
        self.p_pick = torch.zeros((p, self.ns), dtype=torch.int32, device=dev)
        self.p_wts = torch.zeros((p, self.ns), dtype=torch.float32, device=dev)
        self.p_plan = grouped.Plan(p, self.ns, c.experts + 2, dev, prefill=True)
        self.p_act = torch.empty((p * self.ns * c.moe_width,), dtype=torch.bfloat16, device=dev)
        self.p_y = torch.empty((p * self.ns, c.hidden), dtype=torch.bfloat16, device=dev)
        self.p_meta = torch.zeros(4, dtype=torch.int32, device=dev)
        self.p_sampled = torch.zeros(1, dtype=torch.int32, device=dev)
        self._host_p = torch.zeros(1, dtype=torch.int32).pin_memory()

    # -- state ------------------------------------------------------------------------------------
    def reset(self) -> None:
        for name in self.STATE:
            getattr(self, name).zero_()
        self.pos = self.parity = self.prev_keep = 0

    def snapshot(self) -> dict:
        snap = {name: getattr(self, name).clone() for name in self.STATE}
        snap["host"] = (self.pos, self.parity, self.prev_keep)
        return snap

    def restore(self, snap: dict) -> None:
        for name in self.STATE:
            getattr(self, name).copy_(snap[name])
        self.pos, self.parity, self.prev_keep = snap["host"]

    # -- the forward ------------------------------------------------------------------------------
    def norm(self, x, delta, weight):
        eps = self.c.eps
        if delta is None:
            return base.add_rmsnorm(x, None, weight, eps)
        if delta[0] == "moe":
            return G.add_moe_norm(x, delta[1], delta[2], weight, eps, self.c.top_k)
        return base.add_rmsnorm(x, delta[1], weight, eps)

    def mamba(self, m, normed, xs, rows: int, j: int):
        c = self.c
        proj = G.dense(normed, m.in_proj, xs)
        M.conv(proj, self.conv_base[j], self.raw[j], self.xc[j], m.conv_w, m.conv_b, self.meta, rows, xd=c.xd)
        y = M.scan(proj, self.xc[j], self.dt[j], self.ssm[j], m.a, m.d, m.dt_bias, self.meta, rows,
                   heads=c.m_heads, head_dim=c.m_head_dim, groups=c.m_groups, state_dim=c.m_state, lo=c.dt_min,
                   hi=c.dt_max)
        g, gxs = M.group_rmsnorm(y, m.gnorm, c.eps, c.m_groups)
        return G.dense(g, m.out_proj, gxs)

    def attention(self, a: Attention, normed, xs, rows: int, k_cache, v_cache, meta, cfg=None):
        c = cfg or self.c
        qkv = G.dense(normed, a.qkv, xs)
        A.kv_write(qkv, k_cache, v_cache, meta, rows, q_dim=c.heads * c.head_dim)
        out, oxs = A.attention(qkv, k_cache, v_cache, meta, rows, heads=c.heads, kv_heads=c.kv_heads,
                               head_dim=c.head_dim, max_chunks=self.max_chunks)
        return G.dense(out, a.o, oxs)

    def moe(self, moe: MoE, normed, rows: int):
        """Each row's experts (routed, then the shared halves): ("moe", fp32 outputs by pair, weights)."""

        c, ex = self.c, moe.experts
        G.route(normed, moe.router, moe.bias, self.pick[:rows], self.wts[:rows], top_k=c.top_k, scaling=c.scaling,
                norm=c.norm_topk)
        grouped.route(self.pick[:rows], self.plan)
        y = torch.empty((rows * self.ns, c.hidden), dtype=torch.float32, device=normed.device)
        act = self.act[:self.max_rows * self.ns * ex.width].view(-1, ex.width)      # a rank holds part of the width
        grouped.gate_up(normed, ex, self.plan, act, rows)
        grouped.down(act, ex, self.plan, y, rows)
        return ("moe", y, self.wts[:rows])

    def _forward(self, rows: int) -> None:
        w, c = self.w, self.c
        x = base.embed(self.ids[:rows], w.embed.weight, w.embed.scales, w.embed.biases, c.hidden)
        delta = None
        mj = aj = 0
        for blk in w.blocks:
            x, normed, xs = self.norm(x, delta, blk.norm)
            if blk.kind == "M":
                delta = ("dense", self.mamba(blk.mamba, normed, xs, rows, mj))
                mj += 1
            elif blk.kind == "*":
                delta = ("dense", self.attention(blk.attn, normed, xs, rows, self.k_cache[aj], self.v_cache[aj],
                                                 self.meta))
                aj += 1
            else:
                delta = self.moe(blk.moe, normed, rows)
        _, normed, xs = self.norm(x, delta, w.norm_f)
        self.hidden[:rows].copy_(normed)
        self.logits[:rows].copy_(G.dense(normed, w.head, xs))
        if self.masked:
            self.logits[:rows].add_(self.bias[:rows])
        # every row samples its position (pos + row + 1) on the GPU; the tokens go to pinned host memory
        S.sample(self.logits[:rows], self.meta, self.params, self.sampled[:rows])
        self._host_sampled[:rows].copy_(self.sampled[:rows], non_blocking=True)

    # -- prompt chunks --------------------------------------------------------------------------
    def _meta_at(self, pos: int) -> torch.Tensor:
        """``p_meta`` with position ``pos``: the kernels read it as a window's committed length."""

        self.p_meta.fill_(pos)
        return self.p_meta

    def mamba_rows(self, m, normed, xs, rows: int, j: int):
        c = self.c
        proj = G.prefill_dense(normed, m.in_proj)
        M.conv_rows(proj, self.conv_base[j], self.p_xc[:rows], m.conv_w, m.conv_b, rows, xd=c.xd)
        y = M.scan_rows(proj, self.p_xc[:rows], self.ssm[j], m.a, m.d, m.dt_bias, rows, heads=c.m_heads,
                        head_dim=c.m_head_dim, groups=c.m_groups, state_dim=c.m_state, lo=c.dt_min, hi=c.dt_max)
        g, gxs = M.group_rmsnorm(y, m.gnorm, c.eps, c.m_groups)
        return self._dense_delta(g, m.out_proj, gxs)

    def _dense_delta(self, x, q, xs):
        return ("dense", G.prefill_dense(x, q))

    def attention_rows(self, a: Attention, normed, xs, rows: int, k_cache, v_cache, cfg=None):
        """A chunk's rows at positions pos .. pos + rows - 1: keys written, then the shared prefill attention."""

        c = cfg or self.c
        qkv = G.prefill_dense(normed, a.qkv)
        qd = c.heads * c.head_dim
        A.kv_write(qkv, k_cache, v_cache, self._meta_at(self.pos), rows, q_dim=qd)
        q = qkv[:, :qd].contiguous().view(rows, c.heads, c.head_dim)
        out = prefill_attention.attention(q, k_cache, v_cache, self.pos, scale=c.head_dim ** -0.5)
        return self._dense_delta(out.view(rows, qd), a.o, None)

    def moe_rows(self, moe: MoE, normed, rows: int):
        """Each row's experts in the prefill form: ("moe", bf16 outputs by pair, weights)."""

        c, ex = self.c, moe.experts
        G.route(normed, moe.router, moe.bias, self.p_pick[:rows], self.p_wts[:rows], top_k=c.top_k,
                scaling=c.scaling, norm=c.norm_topk)
        grouped.route(self.p_pick[:rows], self.p_plan)
        act = self.p_act[:rows * self.ns * ex.width].view(-1, ex.width)
        y = self.p_y[:rows * self.ns]
        grouped.gate_up(normed, ex, self.p_plan, act, rows)
        grouped.down(act, ex, self.p_plan, y, rows)
        return ("moe", y, self.p_wts[:rows])

    def prefill_moe(self, moe: MoE, normed, rows: int):
        return self.moe_rows(moe, normed, rows)

    def sample_last(self, normed, xs) -> None:
        """The chunk's last row (its position pos - 1 once committed): sample the next token into ``p_sampled``."""

        logits = G.prefill_dense(normed, self.w.head)
        S.sample(logits + self.bias[:1] if self.masked else logits, self._meta_at(self.pos - 1), self.params,
                 self.p_sampled)

    @torch.no_grad()
    def prefill_chunk(self, tokens) -> None:
        """Commit a prompt chunk (every row kept) and sample the next token from its last row."""

        rows = len(tokens)
        if not 1 <= rows <= self.prefill_rows or self.pos + rows > self.max_len:
            raise ValueError("a prompt chunk must fit the chunk buffers and the KV cache")
        if self.prev_keep:
            raise RuntimeError("a prompt chunk runs from a committed state (no window rows left to replay)")
        w, c = self.w, self.c
        self.p_ids[:rows].copy_(torch.as_tensor(tokens, dtype=torch.int32), non_blocking=False)
        x = base.embed(self.p_ids[:rows], w.embed.weight, w.embed.scales, w.embed.biases, c.hidden)
        delta = None
        mj = aj = 0
        for blk in w.blocks:
            x, normed, xs = self.norm(x, delta, blk.norm)
            if blk.kind == "M":
                delta = self.mamba_rows(blk.mamba, normed, xs, rows, mj)
                mj += 1
            elif blk.kind == "*":
                delta = self.attention_rows(blk.attn, normed, xs, rows, self.k_cache[aj], self.v_cache[aj])
                aj += 1
            else:
                delta = self.prefill_moe(blk.moe, normed, rows)
        _, normed, xs = self.norm(x, delta, w.norm_f)
        self.p_hidden[:rows].copy_(normed)
        self.pos += rows
        self.sample_last(normed[rows - 1:rows], xs[rows - 1:rows])
        self._host_p.copy_(self.p_sampled, non_blocking=True)
        self._sampled_ready.record()

    def prefill_token(self) -> int:
        """The token sampled from the last prompt chunk's last row."""

        self._sampled_ready.synchronize()
        return int(self._host_p[0])

    def set_sampling(self, sampling) -> None:
        """The request's sampling (seed, temperature, top_p live on the device; the mode picks the graphs)."""

        self.params.set(sampling)
        self.mode = sampling_mode(sampling)

    def capture(self, rows_list=None, sampling=None) -> None:
        """One CUDA graph per row count for ``sampling``'s mode, captured on a fresh state (warm-ups, then reset)."""

        self.set_sampling(sampling)
        rows_list = list(rows_list or range(1, self.max_rows + 1))
        self.meta.zero_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for rows in rows_list:
                self._forward(rows)
                self._forward(rows)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        for rows in rows_list:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._forward(rows)
            self.graphs[(rows, self.mode)] = g
        torch.cuda.synchronize()
        self.reset()

    def forward(self, tokens, *, rows: int | None = None) -> torch.Tensor:
        """Logits (R, V) for R tokens at pos ..; one token with ``rows``: the MTP head already wrote the drafts."""

        rows = len(tokens) if rows is None else rows
        if not 1 <= rows <= self.max_rows:
            raise ValueError(f"a window holds 1..{self.max_rows} rows")
        if self.pos + rows > self.max_len:
            raise ValueError("the KV cache is full")
        n = len(tokens)
        self._copied.synchronize()              # the previous window's copies have read the pinned buffers
        self._host_ids[:n] = torch.as_tensor(tokens, dtype=torch.int32)
        self._host_meta[0], self._host_meta[1], self._host_meta[2] = self.pos, self.parity, self.prev_keep
        self.ids[:n].copy_(self._host_ids[:n], non_blocking=True)
        self.meta.copy_(self._host_meta, non_blocking=True)
        self._copied.record()
        g = self.graphs.get((rows, self.mode)) if self.use_graphs and not self.masked else None
        if g is not None:
            g.replay()
        else:
            self._forward(rows)
        self._sampled_ready.record()
        self.parity ^= 1
        self._rows = rows
        return self.logits[:rows]

    def mask(self, constraint, window) -> None:
        """The next window's (or prompt's) sampling masked by a reply's grammar; ``None`` lifts it."""

        if constraint is None or not window.rows:        # nothing masked: the plain graphs
            self.masked = False
            return
        width, offset = self.head_width
        if self.bias is None:
            self.bias = torch.zeros((self.max_rows, width), dtype=torch.bfloat16, device=self.device)
        self.bias.zero_()
        allowed = constraint.allowed(window, width, self.device, offset)
        index = torch.tensor(window.rows, device=self.device)
        self.bias[index] = torch.zeros_like(allowed, dtype=torch.bfloat16).masked_fill_(~allowed, float("-inf"))
        self.masked = True

    @property
    def head_width(self) -> tuple[int, int]:
        """(columns, first token id) of the head this engine samples from."""

        return self.c.vocab, 0

    def tokens(self) -> list[int]:
        """The last window's sampled tokens (row r: the token at position pos + r + 1)."""

        self._sampled_ready.synchronize()
        return self._host_sampled[:self._rows].tolist()

    def commit(self, keep: int) -> None:
        """Keep the first ``keep`` rows of the last window."""

        if not 1 <= keep <= self._rows:
            raise ValueError("keep must be between 1 and the window's rows")
        self.pos += keep
        self.prev_keep = keep
