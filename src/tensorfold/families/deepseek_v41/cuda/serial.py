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
from ..reference import inv_freq, rms
from . import hc as hcf
from . import kernels as K
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
        self.hcbuf = hcf.HCBuffers(MAX_ROWS, c.hidden_size, device=self.dev)
        self.tables_rope = {r: K.rope_tables(f, cap) for r, f in self.freqs.items()}
        self.attnbuf = K.AttnBuffers(MAX_ROWS, w.layers[0].attn.wq_b.n // c.head_dim, c.head_dim,
                                     cap + 1 + c.sliding_window, device=self.dev)
        self.graph = None
        self.cap = cap
        self.reset()

    def reset(self) -> None:
        """Forget the request; caches are zeroed in place (a captured graph holds their addresses)."""

        if getattr(self, "state", None) is not None:
            for t in [*self.state.swa, *self.state.comp.values(), *self.state.raw.values()]:
                t.zero_()
            self.state.ids.clear()
            return
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

        st = self.state
        R = len(tokens)
        p0 = len(st.ids)
        if not 0 < R <= MAX_ROWS:
            raise ValueError(f"1..{MAX_ROWS} rows a call, got {R}")
        if p0 + R > min(st.cap, SHORT_CONTEXT):
            raise ValueError(f"context {p0 + R} beyond {min(st.cap, SHORT_CONTEXT)} (indexer not implemented yet)")
        rows = self.engram_rows(tokens)
        if R == 1 and self.graph is not None:
            self.g_tok.fill_(tokens[0])
            self.g_pos.fill_(p0)
            self.g_rows.copy_(rows)
            self.graph.replay()
            return self.g_logits
        return self.core(torch.tensor(tokens, device=self.dev), torch.arange(p0, p0 + R, device=self.dev), rows,
                         static=False)

    def engram_rows(self, tokens: list[int]) -> torch.Tensor:
        """Commit the tokens and read their Engram rows: fp32 [layers, R, 24, 256] on the device."""

        c, st = self.c, self.state
        p0 = len(st.ids)
        st.ids.extend(tokens)
        start = max(0, p0 - (c.engram_max_ngram_size - 1))        # the n-gram history of the first new row
        hashes = E.hashes(np.array(st.ids[start:]), self.tmap, self.layout, c.engram_pad_token_id)[p0 - start:]
        got = self.tables.raw([(ell, hashes[:, ell, :]) for ell in range(len(c.engram_layer_ids))])
        R = len(tokens)
        w = torch.from_numpy(np.stack([g[0] for g in got])).to(self.dev).view(len(got), R, -1, c.engram_head_dim)
        sc = torch.from_numpy(np.stack([g[1] for g in got])).to(self.dev).view(len(got), R, w.shape[2], -1)
        return E.dequant(w, sc)

    def core(self, ids: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, *, static: bool) -> torch.Tensor:
        """The device-only forward (capturable when ``static``: fixed shapes, positions read on the device)."""

        c = self.c
        R = ids.shape[0]
        e = self.w.embed[ids]
        X = e[:, None, :].expand(R, c.hc_mult, c.hidden_size).contiguous()
        pre = torch.zeros((R, c.hc_mult), dtype=F32, device=self.dev)
        pre[:, 0] = 1.0
        f = post = comb = None
        for layer in self.w.layers:
            L = layer.index
            if L > 0:
                X = hcf.post(f, X, post, comb)
            if layer.engram is not None:
                X = self.engram(layer, X, rows[c.engram_layer_ids.index(L)])
            post, comb, x, pre_a = self.hc(layer.hc_attn, X, pre)
            a = self.attention(layer, x, pos, static)
            X = hcf.post(a, X, post, comb)
            post, comb, x, pre = self.hc(layer.hc_ffn, X, pre_a)
            f = self.moe(layer, x, R)
        X = hcf.post(f, X, post, comb)
        h = (pre[:, :, None] * X.float()).sum(1).to(BF)
        h = rms(h, self.w.norm, c.rms_norm_eps).to(BF)
        return self.comm.gather_last(self.w.head(h, out_dtype=F32))

    # -- decode graph -----------------------------------------------------------------------------------------
    def capture(self) -> None:
        """Capture the one-row decode step (both ranks must call this together)."""

        c = self.c
        self.graph = None
        self.g_tok = torch.zeros((1,), dtype=torch.long, device=self.dev)
        self.g_pos = torch.zeros((1,), dtype=torch.long, device=self.dev)
        self.g_rows = torch.zeros((len(c.engram_layer_ids), 1, 3 * c.engram_n_heads, c.engram_head_dim),
                                  dtype=F32, device=self.dev)
        saved = [t.clone() for t in self.state.swa], {k: v.clone() for k, v in self.state.comp.items()}, \
            {k: v.clone() for k, v in self.state.raw.items()}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.core(self.g_tok, self.g_pos, self.g_rows, static=True)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.g_logits = self.core(self.g_tok, self.g_pos, self.g_rows, static=True)
        torch.cuda.synchronize()
        for dst, src in zip(self.state.swa, saved[0]):         # the warm-up wrote position 0: put it back
            dst.copy_(src)
        for k, v in saved[1].items():
            self.state.comp[k].copy_(v)
        for k, v in saved[2].items():
            self.state.raw[k].copy_(v)
        self.graph = graph

    # -- pieces ----------------------------------------------------------------------------------------------
    def hc(self, w: HCW, X: torch.Tensor, pre_in: torch.Tensor):
        c = self.c
        return hcf.pre(X, w.fn, w.base, w.scale, pre_in, w.norm, self.hcbuf, c.rms_norm_eps, c.hc_eps,
                       c.hc_sinkhorn_iters)

    def attention(self, layer: LayerW, x: torch.Tensor, pos: torch.Tensor, static: bool) -> torch.Tensor:
        c, st, a = self.c, self.state, layer.attn
        L, R = layer.index, x.shape[0]
        Dh, W = c.head_dim, c.sliding_window
        cos, sin = self.tables_rope[a.ratio]
        eps = c.rms_norm_eps
        qr = K.rmsnorm(a.wq_a(x), a.q_norm, eps)
        kv = K.rmsnorm(a.wkv(x), a.kv_norm, eps)
        H = a.wq_b.n // Dh
        q = K.rope(a.wq_b(qr).view(R, H, Dh), pos, cos, sin)          # bf16, this rank's heads
        st.swa[L].index_copy_(0, pos, K.rope(kv, pos, cos, sin))
        if a.compressor is not None:
            self.compress(layer, x, pos, static)
        comp, n = None, 0
        if a.ratio > 0:
            comp = st.comp[max(s for s in c.kv_source_layer_ids if s <= L)]
            n = comp.shape[0] if static else (int(pos[-1]) + 1) // a.ratio
        o = K.mqa(q, comp, n, a.ratio, st.swa[L], pos, a.sink, W, self.attnbuf, Dh ** -0.5)
        o = K.rope(o, pos, cos, sin, inverse=True, out_dtype=BF)
        groups = len(a.wo_a)
        o = o.view(R, groups, (H // groups) * Dh)
        z = torch.cat([wo(o[:, g].contiguous()) for g, wo in enumerate(a.wo_a)], dim=1)
        return self.comm.sum(a.wo_b(z, out_dtype=F32)).to(BF)

    def compress(self, layer: LayerW, x: torch.Tensor, pos: torch.Tensor, static: bool) -> None:
        c, st, a = self.c, self.state, layer.attn
        L, r = layer.index, a.ratio
        cos, sin = self.tables_rope[r]
        cw = a.compressor
        kv = cw.wkv(x, out_dtype=F32)
        if r == 1:
            latent = K.rmsnorm(kv, cw.norm, c.rms_norm_eps)
            st.comp[L].index_copy_(0, pos, K.rope(latent, pos, cos, sin).float())
            return
        gate = cw.wgate(x, out_dtype=F32)
        raw = st.raw[L]
        raw.index_copy_(0, pos, torch.cat([kv, gate], dim=1))
        if static:                                                      # write the group only when pos closes it
            ends = pos
        else:
            closing = [int(p) for p in pos.tolist() if (p + 1) % 2 == 0]
            if not closing:
                return
            ends = torch.tensor(closing, device=self.dev)
        pair = torch.stack([raw[(ends - 1).clamp(min=0)], raw[ends]], dim=1)   # [G, 2, 1024]
        wts = torch.softmax(pair[..., c.head_dim:], dim=1)
        latent = K.rmsnorm((wts * pair[..., :c.head_dim]).sum(1), cw.norm, c.rms_norm_eps)
        row = K.rope(latent, (ends // 2) * 2, cos, sin).float()
        if static:
            closes = ((ends + 1) % 2 == 0)[:, None]
            row = torch.where(closes, row, st.comp[L][ends // 2])
        st.comp[L].index_copy_(0, ends // 2, row)

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
