"""Kolibri 1's forward over streams' serial chains of rows: prompt chunks and decode rows, a cache slot a stream."""
# Every kernel is row-invariant, so a row's bits never depend on its chunk or the streams beside it.

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import triton
import triton.language as tl

from tensorfold.cuda import moe as shared

from . import attention, glue, moe
from .weights import Weights

PROMPT_CHUNK = 8192            # prompt rows a forward: every chunk reads all experts once, so wider is cheaper


@triton.jit
def _rms(X, W, OUT, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr, BF16: tl.constexpr):
    """Program r: row r times rsqrt(mean of squares + eps) times the weight, in fp32 (bf16 out when BF16)."""

    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    ok = c < N
    x = tl.load(X + r.to(tl.int64) * N + c, mask=ok, other=0.0).to(tl.float32)
    w = tl.load(W + c, mask=ok, other=0.0)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / N + EPS) * w
    tl.store(OUT + r.to(tl.int64) * N + c, y.to(tl.bfloat16) if BF16 else y, mask=ok)


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float, *, bf16: bool) -> torch.Tensor:
    """RMSNorm over the last dimension, a program a row (its bits never depend on the other rows)."""

    n = x.shape[-1]
    x = x.contiguous()
    out = torch.empty(x.shape, dtype=torch.bfloat16 if bf16 else torch.float32, device=x.device)
    _rms[(x.numel() // n,)](x, weight, out, N=n, EPS=float(eps), BLOCK=triton.next_power_of_2(n), BF16=bf16,
                            num_warps=4 if n > 256 else 1)
    return out


@dataclass
class Chain:
    """A stream's rows in one forward: its cache slot, first position, tokens."""

    slot: int
    p0: int
    tokens: Sequence[int]


class Model:
    """The weights and ``slots`` streams' caches: full layers keep ``context`` positions, sliding ones a ring."""

    def __init__(self, w: Weights, context: int, slots: int = 1, device: str = "cuda") -> None:
        cfg = w.config
        self.w, self.cfg, self.context, self.slots = w, cfg, int(context), int(slots)
        self.ring = attention.ring_size(PROMPT_CHUNK, cfg.window)
        kv = (cfg.kv_heads, cfg.head_dim)
        self.k, self.v = [], []
        for full in cfg.full:
            shape = (self.slots, self.context if full else self.ring, *kv)
            self.k.append(torch.zeros(shape, dtype=torch.bfloat16, device=device))
            self.v.append(torch.zeros(shape, dtype=torch.bfloat16, device=device))
        steps = torch.arange(0, cfg.head_dim, 2, dtype=torch.float64, device=device) / cfg.head_dim
        ang = torch.arange(self.context, dtype=torch.float64, device=device)[:, None] / cfg.rope_theta ** steps
        self.cos = torch.cat([ang.cos(), ang.cos()], -1).float()          # [context, D] (neox halves)
        self.sin = torch.cat([ang.sin(), ang.sin()], -1).float()
        self.qk_norms = [torch.stack([L.q_norm, L.k_norm]).contiguous() for L in w.layers]
        self.scale = cfg.head_dim ** -0.5

    @staticmethod
    def slot_bytes(cfg, context: int) -> int:
        """Cache bytes a stream takes at ``context`` positions."""

        row = 2 * cfg.kv_heads * cfg.head_dim * 2
        full = sum(cfg.full)
        return row * (full * context + (len(cfg.full) - full) * attention.ring_size(PROMPT_CHUNK, cfg.window))

    @torch.no_grad()
    def forward(self, chains: Sequence[Chain], *, prompt: bool, rows: Sequence[int] | None = None) -> torch.Tensor:
        """fp32 logits of ``rows`` (default: each chain's last); ``prompt``: one chain on the prompt kernels."""

        cfg, w = self.cfg, self.w
        if prompt and len(chains) != 1:
            raise ValueError("a prompt forward takes one stream's chunk")
        dev = w.embed.device
        ids = torch.tensor([t for c in chains for t in c.tokens], dtype=torch.int64)
        pos_l = [c.p0 + i for c in chains for i in range(len(c.tokens))]
        slot_l = [c.slot for c in chains for _ in c.tokens]
        for c in chains:
            if len(c.tokens) > self.ring - cfg.window:             # its first row's window must survive its writes
                raise ValueError(f"a chain of {len(c.tokens)} rows overruns the {self.ring}-key sliding ring")
            if c.p0 + len(c.tokens) > self.context:
                raise ValueError(f"positions through {c.p0 + len(c.tokens)} exceed the {self.context}-token cache")
        n = len(pos_l)
        ids = ids.to(dev, non_blocking=True)
        pos = torch.tensor(pos_l, dtype=torch.int64).to(dev, non_blocking=True)
        pos32, slot32 = pos.to(torch.int32), torch.tensor(slot_l, dtype=torch.int32).to(dev, non_blocking=True)
        groups = [(len(c.tokens), c.p0) for c in chains]
        res = w.embed[ids].float()
        h, hk, d, eps = cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.eps
        x = rms(res, w.layers[0].input_norm, eps, bf16=True)
        for i, L in enumerate(w.layers):
            q, k, v = glue.qkv(L.qkv(x), self.qk_norms[i], self.cos, self.sin, pos32, slot32, self.k[i], self.v[i],
                               heads=h, kv_heads=hk, eps=eps, rope=not cfg.full[i])
            if not cfg.full[i]:
                a = attention.sliding(q, self.k[i], self.v[i], pos32, slot32, window=cfg.window, scale=self.scale)
            elif prompt:
                c = chains[0]
                a = attention.full_prompt(q, self.k[i][c.slot], self.v[i][c.slot], c.p0, scale=self.scale)
            else:
                caches = [(self.k[i][c.slot], self.v[i][c.slot]) for c in chains]
                a = attention.full_rows(q, k, v, caches, groups, scale=self.scale)
            x = glue.add_rms(L.o(a.view(n, h * d)), res, L.post_attn_norm, L.pre_moe_norm, eps)
            m = moe.run(x, L.router, L.bias, L.experts, cfg.top_k, prefill=prompt)
            after = w.layers[i + 1].input_norm if i + 1 < len(w.layers) else w.norm
            x = glue.add_rms(m, res, L.post_moe_norm, after, eps)
        if rows is None:
            ends, at = [], 0
            for c in chains:
                at += len(c.tokens)
                ends.append(at - 1)
            rows = ends
        x = x[torch.tensor(list(rows), device=dev)]
        return shared.router(x, w.head)

    def prefill(self, prompt: Sequence[int], start: int = 0, slot: int = 0, chunk: int | None = None) -> torch.Tensor:
        """The prompt from ``start`` in chunks; the last row's logits [1, V]."""

        chunk = chunk or self.ring - self.cfg.window
        logits = None
        for a in range(start, len(prompt), chunk):
            b = min(len(prompt), a + chunk)
            logits = self.forward([Chain(slot, a, prompt[a:b])], prompt=True)
        return logits

    def step(self, tokens: Sequence[int], positions: Sequence[int], slots: Sequence[int]) -> torch.Tensor:
        """One decode row a stream: logits [streams, V]."""

        return self.forward([Chain(s, p, [t]) for t, p, s in zip(tokens, positions, slots)], prompt=False)
