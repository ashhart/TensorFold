"""The serial two-rank DeepSeek-V4.1 forward: split EXL3 weights, position-addressed caches, rank-order reductions.

One request, R rows a call (a prompt chunk of up to 128 rows, or one decode row). The arithmetic follows the
single-GPU reference (``reference.py``, checked against vLLM); tensor parallelism splits attention heads, output
groups, expert widths and the vocabulary head, and every partial sum crosses ranks as fp32 added in rank order.
Contexts stay within the short-context regime (every compressed entry visible, no indexer) for now.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from tensorfold.cuda.exl3 import experts as ex3

from .. import engram as E
from ..reference import hc_post, inv_freq, rms, rope
from .weights import HCW, LayerW, Weights

BF, F32 = torch.bfloat16, torch.float32
MAX_ROWS = 128
SHORT_CONTEXT = 512          # beyond this the indexer's top-512 selection drops entries (not implemented yet)


class Comm:
    """Rank-order sums and gathers over NCCL; a single process (world 1) passes tensors through."""

    def __init__(self, nccl=None) -> None:
        self.nccl = nccl
        self.world = nccl.world if nccl is not None else 1

    def sum(self, partial: torch.Tensor) -> torch.Tensor:
        if self.world == 1:
            return partial
        send = partial.contiguous().float()
        recv = torch.empty((self.world, *send.shape), dtype=F32, device=send.device)
        self.nccl.all_gather(send.view(-1), recv.view(-1))
        total = recv[0].clone()
        for r in range(1, self.world):
            total += recv[r]
        return total

    def gather_last(self, part: torch.Tensor) -> torch.Tensor:
        """Concatenate each rank's slice of the last axis in rank order."""

        if self.world == 1:
            return part
        send = part.contiguous()
        recv = torch.empty((self.world, *send.shape), dtype=send.dtype, device=send.device)
        self.nccl.all_gather(send.view(-1), recv.view(-1))
        return torch.cat(list(recv), dim=-1)


@dataclass
class Caches:
    """Position-addressed state of one request (rows beyond the committed length are overwritten on reuse)."""

    cap: int
    swa: list[torch.Tensor]                     # per layer bf16 [cap, 512], RoPE'd window keys (= values)
    comp: dict[int, torch.Tensor]               # per kv source fp32 [cap // ratio + 1, 512], RoPE'd entries
    raw: dict[int, torch.Tensor]                # per ratio-2 source fp32 [cap, 1024]: projected kv | gate
    ids: list[int] = field(default_factory=list)


class SerialEngine:
    def __init__(self, w: Weights, comm: Comm, engram_dir: str, tokenizer_json: str, *, cap: int = 4096,
                 device: str = "cuda") -> None:
        self.w, self.c, self.comm, self.dev = w, w.cfg, comm, torch.device(device)
        c = self.c
        self.freqs = {r: inv_freq(c, r, self.dev) for r in set(c.layer_ratios)}
        self.layout = E.Layout.from_config(c)
        self.tmap = E.token_map(tokenizer_json, c.engram_compressed_vocab_size)
        self.tables = E.Tables(engram_dir, c.engram_layer_ids)
        self.scratch = [ex3.Scratch(layer.moe.experts, MAX_ROWS, c.num_experts_per_tok) for layer in w.layers]
        self.cap = cap
        self.reset()

    def reset(self) -> None:
        c, cap = self.c, self.cap
        self.state = Caches(
            cap,
            [torch.zeros((cap, c.head_dim), dtype=BF, device=self.dev) for _ in self.w.layers],
            {s: torch.zeros((cap // c.layer_ratios[s] + 1, c.head_dim), dtype=F32, device=self.dev)
             for s in c.kv_source_layer_ids},
            {s: torch.zeros((cap, 2 * c.head_dim), dtype=F32, device=self.dev)
             for s in c.kv_source_layer_ids if c.layer_ratios[s] == 2},
        )

    # -- one forward over new rows ------------------------------------------------------------------------
    def forward(self, tokens: list[int]) -> torch.Tensor:
        """Logits fp32 [R, vocab] of the new rows; positions continue the committed ones."""

        c, st = self.c, self.state
        R = len(tokens)
        p0 = len(st.ids)
        if not 0 < R <= MAX_ROWS:
            raise ValueError(f"1..{MAX_ROWS} rows a call, got {R}")
        if p0 + R > min(st.cap, SHORT_CONTEXT):
            raise ValueError(f"context {p0 + R} beyond {min(st.cap, SHORT_CONTEXT)} (indexer not implemented yet)")
        st.ids.extend(tokens)
        pos = torch.arange(p0, p0 + R, device=self.dev)
        hashes = E.hashes(np.array(st.ids), self.tmap, self.layout, c.engram_pad_token_id)[p0:]

        ids = torch.tensor(tokens, device=self.dev)
        e = self.w.embed[ids]
        X = e[:, None, :].expand(R, c.hc_mult, c.hidden_size).contiguous()
        pre = torch.zeros((R, c.hc_mult), dtype=F32, device=self.dev)
        pre[:, 0] = 1.0
        f = post = comb = None
        for layer in self.w.layers:
            L = layer.index
            if L > 0:
                X = hc_post(f, X, post, comb)
            if layer.engram is not None:
                ell = c.engram_layer_ids.index(L)
                X = self.engram(layer, X, self.tables.rows(ell, hashes[:, ell, :]).to(self.dev))
            post, comb, x, pre_a = self.hc(layer.hc_attn, X, pre)
            a = self.attention(layer, x, pos)
            X = hc_post(a, X, post, comb)
            post, comb, x, pre = self.hc(layer.hc_ffn, X, pre_a)
            f = self.moe(layer, x, R)
        X = hc_post(f, X, post, comb)
        h = (pre[:, :, None] * X.float()).sum(1).to(BF)
        h = rms(h, self.w.norm, c.rms_norm_eps).to(BF)
        return self.comm.gather_last(self.w.head(h, out_dtype=F32))

    # -- pieces ----------------------------------------------------------------------------------------------
    def hc(self, w: HCW, X: torch.Tensor, pre_in: torch.Tensor):
        c = self.c
        T, S, D = X.shape
        eps = c.hc_eps
        xf = X.float().reshape(T, S * D)
        mix = (xf @ w.fn.T) * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + c.rms_norm_eps)
        pre = torch.sigmoid(mix[:, 0:S] * w.scale[0] + w.base[0:S]) + eps
        post = 2 * torch.sigmoid(mix[:, S:2 * S] * w.scale[1] + w.base[S:2 * S])
        comb = mix[:, 2 * S:].view(T, S, S) * w.scale[2] + w.base[2 * S:].view(S, S)
        comb = torch.softmax(comb, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(c.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        x_in = (pre_in[:, :, None] * X.float()).sum(1)
        x_in = rms(x_in.to(BF), w.norm, c.rms_norm_eps).to(BF)
        return post, comb, x_in, pre

    def attention(self, layer: LayerW, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        c, st, a = self.c, self.state, layer.attn
        L, R = layer.index, x.shape[0]
        Dh = c.head_dim
        freqs = self.freqs[a.ratio]
        eps = c.rms_norm_eps
        qr = rms(a.wq_a(x), a.q_norm, eps).to(BF)
        kv = rms(a.wkv(x), a.kv_norm, eps).to(BF)
        H = a.wq_b.n // Dh
        q = rope(a.wq_b(qr).view(R, H, Dh), pos, freqs)                 # fp32, this rank's heads
        kv = rope(kv, pos, freqs).to(BF)
        p0, p1 = int(pos[0]), int(pos[-1]) + 1
        st.swa[L][p0:p1] = kv
        if a.compressor is not None:
            self.compress(layer, x, pos, freqs)

        lo = max(0, p0 - (c.sliding_window - 1))
        win = st.swa[L][lo:p1].float()
        t = pos[:, None]
        s_idx = torch.arange(lo, p1, device=self.dev)[None, :]
        mask = (s_idx <= t) & (s_idx >= t - (c.sliding_window - 1))
        keys = win
        if a.ratio > 0:
            src = max(s for s in c.kv_source_layer_ids if s <= L)
            n = p1 // a.ratio
            if n:
                vis = torch.arange(n, device=self.dev)[None, :] < ((t + 1) // a.ratio)
                keys = torch.cat([st.comp[src][:n], win])
                mask = torch.cat([vis, mask], dim=1)
        scores = torch.einsum("thd,sd->ths", q, keys) * Dh ** -0.5
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        full = torch.cat([scores, a.sink.view(1, H, 1).expand(R, H, 1)], dim=-1)
        o = torch.einsum("ths,sd->thd", torch.softmax(full, dim=-1)[..., :-1], keys)
        o = rope(o, pos, freqs, inverse=True).to(BF)
        groups = len(a.wo_a)
        o = o.view(R, groups, (H // groups) * Dh)
        z = torch.cat([wo(o[:, g].contiguous()) for g, wo in enumerate(a.wo_a)], dim=1)
        return self.comm.sum(a.wo_b(z, out_dtype=F32)).to(BF)

    def compress(self, layer: LayerW, x: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor) -> None:
        c, st, a = self.c, self.state, layer.attn
        L, r = layer.index, a.ratio
        cw = a.compressor
        kv = cw.wkv(x, out_dtype=F32)
        if r == 1:
            latent = rms(kv, cw.norm, c.rms_norm_eps).to(BF)
            st.comp[L][int(pos[0]):int(pos[-1]) + 1] = rope(latent, pos, freqs).to(BF).float()
            return
        gate = cw.wgate(x, out_dtype=F32)
        raw = st.raw[L]
        raw[int(pos[0]):int(pos[-1]) + 1] = torch.cat([kv, gate], dim=1)
        closing = [int(p) for p in pos.tolist() if (p + 1) % 2 == 0]
        if not closing:
            return
        ends = torch.tensor(closing, device=self.dev)
        pair = torch.stack([raw[ends - 1], raw[ends]], dim=1)            # [G, 2, 1024]
        wts = torch.softmax(pair[..., c.head_dim:], dim=1)
        latent = rms((wts * pair[..., :c.head_dim]).sum(1), cw.norm, c.rms_norm_eps).to(BF)
        start = (ends // 2) * 2
        st.comp[L][ends // 2] = rope(latent, start, freqs).to(BF).float()

    def moe(self, layer: LayerW, x: torch.Tensor, R: int) -> torch.Tensor:
        c, m = self.c, layer.moe
        limit = c.swiglu_limit
        logits = x.float() @ m.gate.float().T
        sc = torch.sqrt(torch.nn.functional.softplus(logits))
        pick = torch.topk(sc + m.bias, c.num_experts_per_tok, dim=-1).indices
        w = sc.gather(1, pick)
        w = (w / w.sum(-1, keepdim=True) * c.routed_scaling_factor).float().contiguous()
        routed = ex3.routed(x.contiguous(), pick.int().contiguous(), w, m.experts, self.scratch[layer.index], None,
                            R, limit=limit)
        g = m.shared[0](x, out_dtype=F32)
        u = m.shared[1](x, out_dtype=F32)
        act = (torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)).to(BF)
        shared = m.shared[2](act, out_dtype=F32)
        return self.comm.sum(routed + shared).to(BF)

    def engram(self, layer: LayerW, X: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        c, g = self.c, layer.engram
        R, S, D = X.shape
        kv = g.wkv(rows.to(BF).reshape(R, -1))
        h = X.float()
        key = kv[:, :S * D].view(R, S, D).float()
        val = kv[:, S * D:].float()
        eps = c.rms_norm_eps
        dot = (h * g.q * g.k * key).sum(-1)
        dot = dot * torch.rsqrt(h.pow(2).mean(-1) + eps) * torch.rsqrt(key.pow(2).mean(-1) + eps) / math.sqrt(D)
        gate = torch.sigmoid(torch.sign(dot) * torch.sqrt(dot.abs().clamp(min=1e-6)))
        return (h + gate[:, :, None] * val[:, None, :]).to(BF)

    # -- requests ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: list[int], max_tokens: int, *, chunk: int = MAX_ROWS, on_token=None) -> dict:
        """Greedy decode after a chunked prefill; returns tokens and timings."""

        self.reset()
        t0 = time.perf_counter()
        logits = None
        for i in range(0, len(prompt), chunk):
            logits = self.forward(prompt[i:i + chunk])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = []
        nxt = int(logits[-1].argmax())
        for _ in range(max_tokens):
            out.append(nxt)
            if on_token:
                on_token(nxt)
            if nxt == self.c.eos_token_id:
                break
            logits = self.forward([nxt])
            nxt = int(logits[-1].argmax())
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return {"tokens": out, "prefill_s": t1 - t0, "decode_s": t2 - t1,
                "prefill_tps": len(prompt) / (t1 - t0), "decode_tps": len(out) / max(t2 - t1, 1e-9)}
