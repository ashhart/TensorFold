"""DSpark drafting for the DeepSeek-V4.1 engine: three draft blocks draft N tokens a round in one pass.

Context: after every target forward, the stream means entering layers 37, 38 and 39 (``dspark_target_layer_ids``;
vLLM's eagle3_utils: V4.1 reads its target layers' inputs, V4 the outputs) are projected, ``main_x = main_norm(main_proj(taps))``, and each draft block stores ``kv_norm(wkv(main_x))``
(plain RoPE, theta 1e4) in its own window cache at those positions.

A round (``sample_from_anchor``): queries [anchor, noise x (N - 1)] at positions P .. P + N - 1 run through the
blocks; each query attends to the context window before P and to every block row (non-causal); the collapsed,
normed hidden goes through the target's head, and a sequential Markov bias from the previous token picks each draft
greedily. The target then verifies [anchor, d_0 .. d_{N-1}] and keeps the longest agreeing prefix plus its own
next token. Every cache is position-addressed, so a rejected row is simply overwritten later.
"""

from __future__ import annotations

import torch

from tensorfold.cuda.exl3 import experts as ex3

from . import hc as hcf
from . import kernels as K
from .weights import LayerW

BF, F32 = torch.bfloat16, torch.float32
VARIANT = set(filter(None, __import__("os").environ.get("DSPARK_VARIANT", "").split(",")))


class DSpark:
    def __init__(self, eng, tokens: int = 3) -> None:
        c = eng.c
        self.eng, self.c, self.dw = eng, c, eng.w.draft
        if self.dw is None:
            raise ValueError("the checkpoint's DSpark blocks were not loaded")
        if not 1 <= tokens <= c.dspark_block_size:
            raise ValueError(f"DSpark drafts 1..{c.dspark_block_size} tokens a round, got {tokens}")
        self.N = tokens
        self.dev = eng.dev
        from .serial import DRING, RING

        self.ring = DRING
        # every stream slot's decode context rings side by side (the engine's slots; ``swa`` is the current slot's),
        # and the prompt chunks' staging rings (the engine copies a slot's window in and out around them)
        S = eng.slots
        self.swa_big = [torch.zeros((S * DRING, c.head_dim), dtype=BF, device=self.dev) for _ in self.dw.layers]
        self.stage = [torch.zeros((RING, c.head_dim), dtype=BF, device=self.dev) for _ in self.dw.layers]
        self.stage_ring = RING
        self.slot = eng.slot
        self.g_base = torch.zeros((1,), dtype=torch.long, device=self.dev)   # the drafting stream's first ring row
        from .serial import decode_slots

        self.scratch = [ex3.Scratch(b.moe.experts, 8, decode_slots(b.moe, c.dspark_num_experts_per_tok))
                        for b in self.dw.layers]
        self.noise = torch.full((tokens - 1,), c.dspark_noise_token_id, dtype=torch.long, device=self.dev)
        self.taps: list[torch.Tensor] = []
        self.graph = None

    @property
    def swa(self) -> list[torch.Tensor]:
        """The current slot's context rings (views)."""

        R = self.ring
        return [t[self.slot * R:(self.slot + 1) * R] for t in self.swa_big]

    def reset(self) -> None:
        for t in self.swa:
            t.zero_()

    # -- context ----------------------------------------------------------------------------------------------
    def context(self, taps: list[torch.Tensor], pos: torch.Tensor, off=0, static: bool = True) -> None:
        """Store the draft blocks' context keys for target rows at ``pos`` (taps: stream means [R, D] each); ``off``:
        each row's stream's first ring row (a tensor for decode rows of several streams, else the slot's)."""

        c, dw = self.c, self.dw
        cos, sin = self.eng.tables_rope[0]
        main_x = K.rmsnorm(dw.main_proj(torch.cat(taps, dim=1)), dw.main_norm, c.rms_norm_eps)
        for j, block in enumerate(dw.layers):
            a = block.attn
            kv = K.rmsnorm(a.wkv(main_x), a.kv_norm, c.rms_norm_eps)
            if static:
                self.swa_big[j].index_copy_(0, off + pos % self.ring, K.rope(kv, pos, cos, sin))
            else:                                                   # a prompt chunk: the staging rings
                self.stage[j].index_copy_(0, pos % self.stage_ring, K.rope(kv, pos, cos, sin))

    # -- one drafting pass ------------------------------------------------------------------------------------
    def draft(self, anchor: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """N greedy drafts (long [N], on the device) after ``anchor`` at position ``P`` (both [1] device tensors)."""

        c, eng, dw, N = self.c, self.eng, self.dw, self.N
        ids = torch.cat([anchor, self.noise])
        pos = P + torch.arange(N, device=self.dev)
        X = eng.w.embed[ids][:, None, :].expand(N, c.hc_mult, c.hidden_size).contiguous()
        pre = torch.zeros((N, c.hc_mult), dtype=F32, device=self.dev)
        pre[:, 0] = 1.0
        f = post = comb = None
        for j, block in enumerate(dw.layers):
            if f is not None:
                X = hcf.post(f, X, post, comb)
            post, comb, x, pre_a = eng.hc(block.hc_attn, X, pre)
            X = hcf.post(self.attention(j, block, x, pos, P), X, post, comb)
            post, comb, x, pre = eng.hc(block.hc_ffn, X, pre_a)
            f = eng.moe(block, x, N, top_k=c.dspark_num_experts_per_tok, scratch=self.scratch[j])
        X = hcf.post(f, X, post, comb)
        h = K.rmsnorm((pre[:, :, None] * X.float()).sum(1).to(BF), dw.norm, c.rms_norm_eps)
        base = eng.comm.gather_last(eng.w.head(h, out_dtype=F32))            # [N, V]
        prev, out = anchor, []
        for j in range(N):                                                    # sequential Markov stage
            if "nomarkov" in VARIANT:
                prev = base[j:j + 1].argmax(-1)
            else:
                bias = torch.mm(dw.markov_embed[prev].half(), dw.markov_head.T, out_dtype=F32)
                prev = (base[j:j + 1] + bias).argmax(-1)
            out.append(prev)
        return torch.cat(out)

    def attention(self, j: int, block: LayerW, x: torch.Tensor, pos: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Window-only attention: query i sees context keys P + i - 127 .. P - 1 and every block row."""

        c, eng, a = self.c, self.eng, block.attn
        R, Dh, W = x.shape[0], c.head_dim, c.sliding_window
        cos, sin = eng.tables_rope[0]
        qr = K.rmsnorm(a.wq_a(x), a.q_norm, c.rms_norm_eps)
        kv = K.rmsnorm(a.wkv(x), a.kv_norm, c.rms_norm_eps)
        H = a.wq_b.n // Dh
        q = K.rope(a.wq_b(qr).view(R, H, Dh), pos, cos, sin).float()
        base = self.g_base                                                  # the drafting stream's ring
        self.swa_big[j].index_copy_(0, base + pos % self.ring, K.rope(kv, pos, cos, sin))
        idx = P - (W - 1) + torch.arange(W - 1 + R, device=self.dev)           # P - 127 .. P + N - 1
        keys = self.swa_big[j][base + idx.clamp(min=0) % self.ring].float()
        mask = (idx[None, :] >= (pos[:, None] - (W - 1))) & (idx[None, :] >= 0)
        if "causal" in VARIANT:
            mask = mask & (idx[None, :] <= pos[:, None])
        if "fixedctx" in VARIANT:                                           # context P - 128 .. P - 1 for every query
            mask = (idx[None, :] >= P - W) & (idx[None, :] >= 0) & torch.ones_like(mask)
        s = torch.einsum("thd,sd->ths", q, keys) * Dh ** -0.5
        s = s.masked_fill(~mask[:, None, :], float("-inf"))
        full = torch.cat([s, a.sink.view(1, H, 1).expand(R, H, 1)], dim=-1)
        o = torch.einsum("ths,sd->thd", torch.softmax(full, dim=-1)[..., :-1], keys)
        o = K.rope(o, pos, cos, sin, inverse=True, out_dtype=BF)
        groups = len(a.wo_a)
        o = o.view(R, groups, (H // groups) * Dh)
        z = torch.cat([wo(o[:, g].contiguous()) for g, wo in enumerate(a.wo_a)], dim=1)
        return eng.comm.partials(a.wo_b(z, out_dtype=F32))

    # -- graph --------------------------------------------------------------------------------------------------
    def capture(self) -> None:
        self.g_anchor = torch.zeros((1,), dtype=torch.long, device=self.dev)
        self.g_P = torch.zeros((1,), dtype=torch.long, device=self.dev)
        self.h_drafts = torch.zeros((self.N,), dtype=torch.long).pin_memory()
        saved = [t.clone() for t in self.swa_big]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.draft(self.g_anchor, self.g_P)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.g_drafts = self.draft(self.g_anchor, self.g_P)
        torch.cuda.synchronize()
        for dst, src in zip(self.swa_big, saved):
            dst.copy_(src)

    def propose(self, anchor: int, P: int) -> list[int]:
        self.g_anchor.fill_(anchor)
        self.g_P.fill_(P)
        self.g_base.fill_(self.slot * self.ring)
        self.graph.replay()
        self.h_drafts.copy_(self.g_drafts, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self.h_drafts.tolist()

    # -- several streams in one pass (concurrent rounds) ---------------------------------------------------------
    def draft_multi(self, anchor: torch.Tensor, P: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """N greedy drafts for each of M streams (long [M, N]): anchors, positions and ring bases [M], one pass."""

        c, eng, dw, N = self.c, self.eng, self.dw, self.N
        M = anchor.shape[0]
        ids = torch.cat([anchor[:, None], self.noise[None, :].expand(M, N - 1)], dim=1).reshape(-1)
        pos = (P[:, None] + torch.arange(N, device=self.dev)[None, :]).reshape(-1)
        rbase = base[:, None].expand(M, N).reshape(-1)
        X = eng.w.embed[ids][:, None, :].expand(M * N, c.hc_mult, c.hidden_size).contiguous()
        pre = torch.zeros((M * N, c.hc_mult), dtype=F32, device=self.dev)
        pre[:, 0] = 1.0
        f = post = comb = None
        for j, block in enumerate(dw.layers):
            if f is not None:
                X = hcf.post(f, X, post, comb)
            post, comb, x, pre_a = eng.hc(block.hc_attn, X, pre)
            X = hcf.post(self.attention_multi(j, block, x, pos, P, base, rbase, M), X, post, comb)
            post, comb, x, pre = eng.hc(block.hc_ffn, X, pre_a)
            f = eng.moe(block, x, M * N, top_k=c.dspark_num_experts_per_tok, scratch=self.scratch_multi[j])
        X = hcf.post(f, X, post, comb)
        h = K.rmsnorm((pre[:, :, None] * X.float()).sum(1).to(BF), dw.norm, c.rms_norm_eps)
        logits = eng.comm.gather_last(eng.w.head(h, out_dtype=F32)).view(M, N, -1)
        prev, out = anchor, []
        for j in range(N):                                                    # sequential Markov stage, batched
            bias = torch.mm(dw.markov_embed[prev].half(), dw.markov_head.T, out_dtype=F32)
            prev = (logits[:, j] + bias).argmax(-1)
            out.append(prev)
        return torch.stack(out, dim=1)

    def attention_multi(self, j: int, block, x: torch.Tensor, pos: torch.Tensor, P: torch.Tensor,
                        base: torch.Tensor, rbase: torch.Tensor, M: int) -> torch.Tensor:
        """``attention`` for M streams' block rows: each stream's rows see its own context ring's window."""

        c, eng, a = self.c, self.eng, block.attn
        R, Dh, W, N = x.shape[0], c.head_dim, c.sliding_window, self.N
        cos, sin = eng.tables_rope[0]
        qr = K.rmsnorm(a.wq_a(x), a.q_norm, c.rms_norm_eps)
        kv = K.rmsnorm(a.wkv(x), a.kv_norm, c.rms_norm_eps)
        H = a.wq_b.n // Dh
        q = K.rope(a.wq_b(qr).view(R, H, Dh), pos, cos, sin).float().view(M, N, H, Dh)
        self.swa_big[j].index_copy_(0, rbase + pos % self.ring, K.rope(kv, pos, cos, sin))
        idx = P[:, None] - (W - 1) + torch.arange(W - 1 + N, device=self.dev)[None, :]      # [M, keys]
        keys = self.swa_big[j][base[:, None] + idx.clamp(min=0) % self.ring].float()        # [M, keys, D]
        pr = pos.view(M, N)
        mask = (idx[:, None, :] >= (pr[:, :, None] - (W - 1))) & (idx[:, None, :] >= 0)    # [M, N, keys]
        sc = torch.einsum("mthd,msd->mths", q, keys) * Dh ** -0.5
        sc = sc.masked_fill(~mask[:, :, None, :], float("-inf"))
        full = torch.cat([sc, a.sink.view(1, 1, H, 1).expand(M, N, H, 1)], dim=-1)
        o = torch.einsum("mths,msd->mthd", torch.softmax(full, dim=-1)[..., :-1], keys).reshape(R, H, Dh)
        o = K.rope(o, pos, cos, sin, inverse=True, out_dtype=BF)
        groups = len(a.wo_a)
        o = o.view(R, groups, (H // groups) * Dh)
        z = torch.cat([wo(o[:, g].contiguous()) for g, wo in enumerate(a.wo_a)], dim=1)
        return eng.comm.partials(a.wo_b(z, out_dtype=F32))

    def capture_multi(self, streams: int) -> None:
        """Draft graphs for 1..``streams`` streams at once (each stream's N block rows: M * N <= the decode rows)."""

        c = self.c
        from .serial import decode_slots

        self.scratch_multi = [ex3.Scratch(b.moe.experts, streams * self.N, decode_slots(b.moe, c.dspark_num_experts_per_tok))
                              for b in self.dw.layers]
        self.multi_graphs = {}
        saved = [t.clone() for t in self.swa_big]
        for M in range(1, streams + 1):
            g = {"anchor": torch.zeros((M,), dtype=torch.long, device=self.dev),
                 "P": torch.full((M,), 200, dtype=torch.long, device=self.dev),
                 "base": torch.zeros((M,), dtype=torch.long, device=self.dev),
                 "h": torch.zeros((M, self.N), dtype=torch.long).pin_memory()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.draft_multi(g["anchor"], g["P"], g["base"])
            torch.cuda.current_stream().wait_stream(side)
            g["graph"] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g["graph"]):
                g["out"] = self.draft_multi(g["anchor"], g["P"], g["base"])
            self.multi_graphs[M] = g
        torch.cuda.synchronize()
        for dst, src in zip(self.swa_big, saved):
            dst.copy_(src)

    def propose_multi(self, items: list[tuple[int, int, int]]) -> list[list[int]]:
        """Drafts for several streams in one pass: ``items`` = (slot, anchor, position) each."""

        g = self.multi_graphs[len(items)]
        g["anchor"].copy_(torch.tensor([a for _, a, _ in items]), non_blocking=True)
        g["P"].copy_(torch.tensor([p for _, _, p in items]), non_blocking=True)
        g["base"].copy_(torch.tensor([s * self.ring for s, _, _ in items]), non_blocking=True)
        g["graph"].replay()
        g["h"].copy_(g["out"], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return g["h"].tolist()


class DraftPolicy:
    """How many of the N drafts to verify each round: the k (0 = no drafting) with the most expected tokens per
    millisecond, from running acceptance estimates per draft position and round costs per k.

    Both ranks must choose the same k (their collectives must match): each rank keeps its own estimates (costs from
    its clock, starting from the agreed capture-time replays), and the engine shares rank 0's choice every round."""

    def __init__(self, n: int, cost: list[float], alpha: float = 0.15, refresh: int = 16, prior: float = 0.7) -> None:
        self.n, self.alpha, self.refresh = n, alpha, refresh
        self.accept = [prior] * n                     # P(draft j accepted | drafts before it accepted)
        self.cost = list(cost)
        self.rounds = 0

    def expected_tokens(self, k: int) -> float:
        total, run = 1.0, 1.0
        for j in range(k):
            run *= self.accept[j]
            total += run
        return total

    def choose(self) -> int:
        self.rounds += 1
        if self.rounds % self.refresh == 0:            # keep the later positions' estimates fresh
            return self.n
        return max(range(self.n + 1), key=lambda k: self.expected_tokens(k) / self.cost[k])

    def update(self, k: int, accepted: int, ms: float) -> None:
        a = self.alpha
        self.cost[k] = (1 - a) * self.cost[k] + a * ms
        for j in range(min(k, accepted + 1)):          # positions after the first rejection are unobserved
            self.accept[j] = (1 - a) * self.accept[j] + a * (1.0 if j < accepted else 0.0)
