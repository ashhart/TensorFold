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
from ..reference import inv_freq
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

    def partials(self, partial: torch.Tensor) -> torch.Tensor:
        """Every rank's fp32 partial, stacked in rank order [world, ...] (the consumer adds them in order)."""

        send = partial.contiguous().float()
        if self.world == 1:
            return send[None]
        recv = torch.empty((self.world, *send.shape), dtype=F32, device=send.device)
        self.nccl.all_gather(send.view(-1), recv.view(-1))
        return recv

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
        self.split = c.engram_layer_ids[1]
        self.tables_rope = {r: K.rope_tables(f, cap) for r, f in self.freqs.items()}
        self.attnbuf = K.AttnBuffers(MAX_ROWS, w.layers[0].attn.wq_b.n // c.head_dim, c.head_dim,
                                     cap + 1 + c.sliding_window, device=self.dev)
        self.graph = None
        self.graphs: dict[int, dict] = {}
        self.drafter = None
        self.taps: list[torch.Tensor] = []
        self.cap = cap
        self.reset()

    def enable_dspark(self, tokens: int = 3) -> None:
        from .dspark import DSpark

        self.drafter = DSpark(self, tokens)

    def reset(self) -> None:
        """Forget the request; caches are zeroed in place (a captured graph holds their addresses)."""

        if getattr(self, "state", None) is not None:
            for t in [*self.state.swa, *self.state.comp.values(), *self.state.raw.values()]:
                t.zero_()
            self.state.ids.clear()
            if self.drafter is not None:
                self.drafter.reset()
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
        if R in self.graphs:
            self.step_rows(tokens)
            return self.graphs[R]["logits"]
        rows = self.engram_rows(tokens)
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
        """The device-only forward over all layers (eager prompt chunks)."""

        carry = self.part_a(ids, pos, rows[0], static=static)
        return self.part_b(carry, pos, rows[1], static=static)

    def part_a(self, ids: torch.Tensor, pos: torch.Tensor, rows1: torch.Tensor, *, static: bool) -> tuple:
        """Embedding and the layers before the second Engram layer."""

        return self.part_a1(self.part_a0(ids, pos, static=static), pos, rows1, static=static)

    def part_a0(self, ids: torch.Tensor, pos: torch.Tensor, *, static: bool) -> tuple:
        """Embedding and the layers before the first Engram layer (no table rows needed)."""

        c = self.c
        R = ids.shape[0]
        X = self.w.embed[ids][:, None, :].expand(R, c.hc_mult, c.hidden_size).contiguous()
        pre = torch.zeros((R, c.hc_mult), dtype=F32, device=self.dev)
        pre[:, 0] = 1.0
        return self.layers((X, pre, None, None, None), pos, {}, 0, c.engram_layer_ids[0], static)

    def part_a1(self, carry: tuple, pos: torch.Tensor, rows1: torch.Tensor, *, static: bool) -> tuple:
        c = self.c
        return self.layers(carry, pos, {c.engram_layer_ids[0]: rows1}, c.engram_layer_ids[0], self.split, static)

    def part_b(self, carry: tuple, pos: torch.Tensor, rows14: torch.Tensor, *, static: bool) -> torch.Tensor:
        """The remaining layers and the vocabulary head."""

        c = self.c
        self.taps = []
        X, pre, f, post, comb = self.layers(carry, pos, {c.engram_layer_ids[1]: rows14}, self.split,
                                            len(self.w.layers), static)
        X = hcf.post(f, X, post, comb)
        if self.drafter is not None:
            self.drafter.context(self.taps, pos)
        h = (pre[:, :, None] * X.float()).sum(1).to(BF)
        h = K.rmsnorm(h, self.w.norm, c.rms_norm_eps)
        return self.comm.gather_last(self.w.head(h, out_dtype=F32))

    def _tap(self, X: torch.Tensor) -> torch.Tensor:
        from .dspark import VARIANT

        if "tap0" in VARIANT:
            return X[:, 0].contiguous()
        if "tapsum" in VARIANT:
            return X.float().sum(1).to(BF)
        return X.float().mean(1).to(BF)

    def layers(self, carry: tuple, pos: torch.Tensor, rows: dict, first: int, last: int, static: bool) -> tuple:
        X, pre, f, post, comb = carry
        for layer in self.w.layers[first:last]:
            if f is not None:
                X = hcf.post(f, X, post, comb)
                if self.drafter is not None and layer.index in self.c.dspark_target_layer_ids:
                    self.taps.append(self._tap(X))                    # V4.1 taps the entry stream of layer L
            if layer.engram is not None:
                X = self.engram(layer, X, rows[layer.index])
            post, comb, x, pre_a = self.hc(layer.hc_attn, X, pre)
            a = self.attention(layer, x, pos, static)
            X = hcf.post(a, X, post, comb)
            post, comb, x, pre = self.hc(layer.hc_ffn, X, pre_a)
            f = self.moe(layer, x, x.shape[0])
        return X, pre, f, post, comb

    # -- decode graphs ----------------------------------------------------------------------------------------
    def capture(self, rows: int = 1) -> None:
        """Capture an R-row decode step as three graphs (embedding + layer 0 / layers 1-13 / the rest), so the Engram
        rows of each table are read while the graph before it runs. Both ranks must capture together."""

        c = self.c
        n_rows = 3 * c.engram_n_heads
        row_bytes = c.engram_head_dim + c.engram_head_dim // 32
        g = {"tok": torch.zeros((rows,), dtype=torch.long, device=self.dev),
             "pos": torch.zeros((rows,), dtype=torch.long, device=self.dev),
             "raw": [torch.zeros((rows, n_rows, row_bytes), dtype=torch.uint8, device=self.dev) for _ in range(2)],
             "h_raw": torch.zeros((2, rows * n_rows, row_bytes), dtype=torch.uint8).pin_memory(),
             "h_next": torch.zeros((rows,), dtype=torch.long).pin_memory()}
        saved = self._save_caches()
        hd = c.engram_head_dim

        def table(k):
            raw = g["raw"][k]
            return E.dequant(raw[..., :hd], raw[..., hd:])

        def run_a0():
            return self.part_a0(g["tok"], g["pos"], static=True)

        def run_a1(carry):
            return self.part_a1(carry, g["pos"], table(0), static=True)

        def run_b(carry):
            logits = self.part_b(carry, g["pos"], table(1), static=True)
            return logits, logits.argmax(-1)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run_b(run_a1(run_a0()))
        torch.cuda.current_stream().wait_stream(side)
        g["a0"], g["a1"], g["b"] = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(g["a0"]):
            g["carry0"] = run_a0()
        with torch.cuda.graph(g["a1"], pool=g["a0"].pool()):
            g["carry"] = run_a1(g["carry0"])
        with torch.cuda.graph(g["b"], pool=g["a0"].pool()):
            g["logits"], g["next"] = run_b(g["carry"])
        torch.cuda.synchronize()
        self._restore_caches(saved)
        self.graphs[rows] = g
        self.graph = True

    def _save_caches(self):
        st = self.state
        extra = [t.clone() for t in self.drafter.swa] if self.drafter is not None else []
        return ([t.clone() for t in st.swa], {k: v.clone() for k, v in st.comp.items()},
                {k: v.clone() for k, v in st.raw.items()}, extra)

    def _restore_caches(self, saved) -> None:
        st = self.state
        for dst, src in zip(st.swa, saved[0]):
            dst.copy_(src)
        for k, v in saved[1].items():
            st.comp[k].copy_(v)
        for k, v in saved[2].items():
            st.raw[k].copy_(v)
        if self.drafter is not None:
            for dst, src in zip(self.drafter.swa, saved[3]):
                dst.copy_(src)

    def step(self, token: int) -> int:
        return self.step_rows([token])[0]

    def step_rows(self, tokens: list[int]) -> list[int]:
        """R rows through the captured graphs; returns the target's argmax at each row."""

        c, st = self.c, self.state
        R = len(tokens)
        g = self.graphs[R]
        p0 = len(st.ids)
        if p0 + R > min(st.cap, SHORT_CONTEXT):
            raise ValueError(f"context {p0 + R} beyond {min(st.cap, SHORT_CONTEXT)} (indexer not implemented yet)")
        st.ids.extend(tokens)
        start = max(0, p0 - (c.engram_max_ngram_size - 1))
        g["tok"].copy_(torch.tensor(tokens), non_blocking=True)
        g["pos"].copy_(torch.arange(p0, p0 + R), non_blocking=True)
        g["a0"].replay()                                                # layer 0 needs no table rows
        h = E.hashes(np.array(st.ids[start:]), self.tmap, self.layout, c.engram_pad_token_id)[-R:]   # [R, 2, 24]
        self.tables.gather(h[:, 0, :].reshape(1, -1), out=g["h_raw"][:1], layers=[0])     # overlaps layer 0
        g["raw"][0].copy_(g["h_raw"][0].view_as(g["raw"][0]), non_blocking=True)
        g["a1"].replay()
        self.tables.gather(h[:, 1, :].reshape(1, -1), out=g["h_raw"][1:], layers=[1])     # overlaps layers 1-13
        g["raw"][1].copy_(g["h_raw"][1].view_as(g["raw"][1]), non_blocking=True)
        g["b"].replay()
        g["h_next"].copy_(g["next"], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return g["h_next"].tolist()

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
        return self.comm.partials(a.wo_b(z, out_dtype=F32))

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
        slot = ends // 2
        if static:                                                      # rows that close no group write the spare slot
            slot = torch.where((ends + 1) % 2 == 0, slot, st.comp[L].shape[0] - 1)
        st.comp[L].index_copy_(0, slot, row)

    def moe(self, layer: LayerW, x: torch.Tensor, R: int, top_k: int | None = None, scratch=None) -> torch.Tensor:
        c, m = self.c, layer.moe
        limit = c.swiglu_limit
        # fp16 inputs (bf16 -> fp16 is exact for normed rows), fp32 accumulation and output: no TF32, half the bytes
        logits = torch.mm(x.half(), m.gate.T, out_dtype=F32)
        pick, w = K.route(logits, m.bias, top_k or c.num_experts_per_tok, c.routed_scaling_factor)
        scratch = scratch if scratch is not None else self.scratch[layer.index]
        routed = ex3.routed(x.contiguous(), pick, w, m.experts, scratch, None, R, limit=limit)
        g = m.shared[0](x, out_dtype=F32)
        u = m.shared[1](x, out_dtype=F32)
        act = (torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)).to(BF)
        shared = m.shared[2](act, out_dtype=F32)
        return self.comm.partials(routed + shared)

    def engram(self, layer: LayerW, X: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        c, g = self.c, layer.engram
        R, S, D = X.shape
        kv = self.comm.gather_last(g.wkv(rows.to(BF).reshape(R, -1)))    # each rank projects half the columns
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
        rounds = accepted = 0
        eos = self.c.eos_token_id
        dsp = self.drafter if self.drafter is not None and self.drafter.graph is not None else None
        while len(out) < max_tokens:
            if dsp is None:
                out.append(nxt)
                if on_token:
                    on_token(nxt)
                if nxt == eos:
                    break
                nxt = self.step(nxt) if self.graph is not None else int(self.forward([nxt])[-1].argmax())
                continue
            P = len(self.state.ids)
            drafts = dsp.propose(nxt, P)
            target = self.step_rows([nxt, *drafts])
            m = 0
            while m < len(drafts) and drafts[m] == target[m]:
                m += 1
            del self.state.ids[P + 1 + m:]                         # rejected rows: overwritten by later positions
            rounds += 1
            accepted += m
            emitted = [nxt, *drafts[:m]]
            nxt = target[m]
            for tok in emitted:
                out.append(tok)
                if on_token:
                    on_token(tok)
                if tok == eos or len(out) >= max_tokens:
                    break
            if out[-1] == eos:
                break
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        res = {"tokens": out, "prefill_s": t1 - t0, "decode_s": t2 - t1,
               "prefill_tps": len(prompt) / (t1 - t0), "decode_tps": len(out) / max(t2 - t1, 1e-9)}
        if dsp is not None:
            res.update(rounds=rounds, accepted_per_round=accepted / max(rounds, 1),
                       tokens_per_round=len(out) / max(rounds, 1))
        return res
