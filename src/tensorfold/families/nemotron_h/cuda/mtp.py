"""Nemotron-H's MTP head on CUDA: cheap drafts of the token after next; the target verifies every one."""

from __future__ import annotations

import torch

from tensorfold.families.qwen3_5.cuda import glue as base
from tensorfold.families.qwen3_5.cuda.qmm_fast import tile, untile
from tensorfold.families.qwen3_5.cuda.weights import QLinear

from . import attention as A, glue as G, sampler as S
from .engine import Engine

MAX_CHAIN = 8


def head_rows(head: QLinear, ids: torch.Tensor) -> QLinear:
    """The vocabulary head's rows for these token ids, as a tiled head of its own."""

    full = untile(head)
    return tile(QLinear(full.weight.index_select(0, ids).contiguous(), full.scales.index_select(0, ids).contiguous(),
                        full.biases.index_select(0, ids).contiguous()))


class MTPHead:
    """``draft_ids``: draft among these ids (a multiple of 64); ``split``: two ranks, vocabulary by halves."""

    def __init__(self, engine: Engine, *, draft_ids=None, split: bool = False):
        if engine.w.mtp is None:
            raise ValueError("the checkpoint has no MTP head (mtp-4bit.safetensors)")
        self.e = engine
        self.split = bool(split) and hasattr(engine, "gather")
        self.id_map = None
        self.local_ids = None
        ids = None if draft_ids is None else torch.as_tensor(list(draft_ids), dtype=torch.int64, device=engine.device)
        if self.split:
            from .tp import WORLD, vocab_rows

            self.m = engine.w.extra["mtp_local"]
            c = self.cfg = engine.c
            if ids is not None:                                # each rank scores its half of the id list
                if ids.numel() % (64 * WORLD):
                    raise ValueError("split draft ids must hold a multiple of 128 ids")
                half = ids.numel() // WORLD
                self.local_ids = ids[engine.rank * half:(engine.rank + 1) * half].contiguous()
                self.offset = 0
                self.head = head_rows(engine.w.head, self.local_ids)
            else:
                half = engine.w.config.vocab // WORLD
                self.offset = engine.rank * half
                self.head = vocab_rows(engine.w.head, self.offset, self.offset + half)
        else:
            self.m = engine.w.mtp
            c = self.cfg = engine.w.extra.get("full_config", engine.c)      # replicated on every rank
            self.head = engine.w.head
            if ids is not None:
                if ids.numel() % 64:
                    raise ValueError("draft ids must hold a multiple of 64 ids")
                self.head = head_rows(engine.w.head, ids)
                self.id_map = ids
        dev = engine.device
        # meta[0]: head positions of an absorb; meta[1 + j]: position of chain step j + 2 (pos + keep + j)
        self.meta = torch.zeros(MAX_CHAIN + 2, dtype=torch.int32, device=dev)
        self.k_cache = torch.zeros((engine.max_len, c.kv_heads, c.head_dim), dtype=torch.bfloat16, device=dev)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.hin = torch.zeros((engine.max_rows, c.hidden), dtype=torch.bfloat16, device=dev)
        self.tok = torch.zeros(engine.max_rows, dtype=torch.int32, device=dev)
        self._host_meta = torch.zeros(MAX_CHAIN + 2, dtype=torch.int32).pin_memory()
        self._host_drafts = torch.zeros(MAX_CHAIN, dtype=torch.int32).pin_memory()
        self.probs = torch.zeros(MAX_CHAIN, dtype=torch.float32, device=dev)
        self._host_probs = torch.zeros(MAX_CHAIN, dtype=torch.float32).pin_memory()
        self._copied = torch.cuda.Event()
        self._copied.record()
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.pos = 0
        self._count = 0

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.pos = 0

    def snapshot(self) -> dict:
        return {"k": self.k_cache.clone(), "v": self.v_cache.clone(), "pos": self.pos}

    def restore(self, snap: dict) -> None:
        self.k_cache.copy_(snap["k"])
        self.v_cache.copy_(snap["v"])
        self.pos = snap["pos"]

    # -- the head's forward ---------------------------------------------------------------------
    def _forward(self, rows: int, tail: bool, meta: torch.Tensor):
        """(out, candidates) of the last row when ``tail``: its logits, or the ranks' merged top lists when split."""

        e, m, w, c = self.e, self.m, self.e.w, self.cfg
        emb = base.embed(self.tok[:rows], w.embed.weight, w.embed.scales, w.embed.biases, c.hidden)
        cat, cxs = G.concat_norms(emb, self.hin[:rows], m.enorm, m.hnorm, c.eps)
        x = G.dense(cat, m.eh_proj, cxs)
        _, normed, xs = base.add_rmsnorm(x, None, m.attn_norm, c.eps)
        if self.split:
            parts = e.attention_tp(m.attn, normed, xs, rows, self.k_cache, self.v_cache, meta)[1]
            if not tail:
                return None
            last = torch.stack([parts[rows - 1], parts[2 * rows - 1]])
            h, normed, xs = e.norm(x[rows - 1:rows].contiguous(), ("ranks", last), m.moe_norm)
            _, out, oxs = e.norm(h, e.moe_tp(m.moe, normed, 1), m.final_norm)
            logits = G.dense(out, self.head, oxs)
            vals, ids = torch.topk(logits.float(), 28, dim=-1)
            ids = self.local_ids[ids] if self.local_ids is not None else ids + self.offset
            both = e.gather(torch.cat([vals.view(torch.int32), ids.view(torch.int32)], dim=1)).view(2, 1, 84)
            v = both[:, :, :28].contiguous().view(torch.float32)
            i = both[:, :, 28:].contiguous().view(torch.int64)
            return out, (torch.cat([v[0], v[1]], dim=1), torch.cat([i[0], i[1]], dim=1))
        delta = e.attention(m.attn, normed, xs, rows, self.k_cache, self.v_cache, meta, cfg=self.cfg)
        if not tail:
            return None
        h, normed, xs = base.add_rmsnorm(x[rows - 1:rows].contiguous(), delta[rows - 1:rows].contiguous(), m.moe_norm,
                                         c.eps)
        _, y, wts = e.moe(m.moe, normed, 1)
        _, out, oxs = G.add_moe_norm(h, y, wts, m.final_norm, c.eps, c.top_k)
        return out, G.dense(out, self.head, oxs)

    def _sample(self, cand, meta: torch.Tensor, out: torch.Tensor, offset: int, prob: torch.Tensor | None):
        if self.split:
            return S.sample_candidates(cand[0], cand[1], meta, self.e.params, out, offset=offset, prob=prob)
        return S.sample(cand, meta, self.e.params, out, offset=offset, prob=prob, id_map=self.id_map)

    def step(self, hidden: torch.Tensor, tokens, *, commit: bool, tail: bool = True, offset: int = 0):
        """Eager head rows at pos + offset; ``commit`` makes them context; (out, logits) of the last row if ``tail``."""

        rows = hidden.shape[0]
        self.hin[:rows].copy_(hidden)
        self.tok[:rows].copy_(torch.as_tensor(tokens, dtype=torch.int32), non_blocking=False)
        self.meta[0] = self.pos + offset
        result = self._forward(rows, tail, self.meta)
        if commit:
            self.pos += rows
        return result

    @torch.no_grad()
    def absorb_rows(self, hidden: torch.Tensor, tokens) -> None:
        """A prompt chunk's rows into the head's cache: only the keys and values (no row reads the output)."""

        e, m, w, c = self.e, self.m, self.e.w, self.cfg
        rows = hidden.shape[0]
        tok = torch.as_tensor(list(tokens), dtype=torch.int32).to(e.device)
        emb = base.embed(tok, w.embed.weight, w.embed.scales, w.embed.biases, c.hidden)
        cat, _ = G.concat_norms(emb, hidden, m.enorm, m.hnorm, c.eps)
        x = G.prefill_dense(cat, m.eh_proj)
        _, normed, _ = base.add_rmsnorm(x, None, m.attn_norm, c.eps)
        qkv = G.prefill_dense(normed, m.attn.qkv)
        A.kv_write(qkv, self.k_cache, self.v_cache, e._meta_at(self.pos), rows, q_dim=c.heads * c.head_dim)
        self.pos += rows

    # -- a round's work: absorb the kept rows, then draft (graph-captured) -----------------------
    def _round(self, keep: int, count: int) -> None:
        e = self.e
        self.hin[:keep].copy_(e.hidden[:keep])
        self.tok[:keep].copy_(e.sampled[:keep])
        if count == 0:
            self._forward(keep, False, self.meta)
            return
        out, cand = self._forward(keep, True, self.meta)
        self._sample(cand, self.meta[1:], e.ids[1:2], 0, self.probs[0:1])
        for j in range(2, count + 1):
            self.hin[:1].copy_(out)
            self.tok[:1].copy_(e.ids[j - 1:j])
            out, cand = self._forward(1, True, self.meta[j - 1:])
            self._sample(cand, self.meta[1:], e.ids[j:j + 1], j - 1, self.probs[j - 1:j])
        self._host_drafts[:count].copy_(e.ids[1:1 + count], non_blocking=True)
        self._host_probs[:count].copy_(self.probs[:count], non_blocking=True)

    def capture(self, keeps, counts) -> None:
        """Graphs for every (kept rows, drafts) pair, in the engine's current sampling mode."""

        e = self.e
        pairs = [(k, n) for k in keeps for n in counts]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for k, n in pairs:
                self._round(k, n)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        if e.pool is None:
            e.pool = torch.cuda.graph_pool_handle()
        for k, n in pairs:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=e.pool):
                self._round(k, n)
            self.graphs[(k, n, e.mode)] = g
        torch.cuda.synchronize()
        self.reset()

    def round(self, keep: int, count: int) -> None:
        """Absorb the first ``keep`` rows of the last window and queue ``count`` drafts into ``engine.ids[1:]``."""

        self._copied.synchronize()
        base_pos = self.pos
        self._host_meta[0] = base_pos
        for j in range(MAX_CHAIN + 1):
            self._host_meta[1 + j] = base_pos + keep + j
        self.meta.copy_(self._host_meta, non_blocking=True)
        self._copied.record()
        g = self.graphs.get((keep, count, self.e.mode)) if self.e.use_graphs else None
        if g is not None:
            g.replay()
        else:
            self._round(keep, count)
        self.pos += keep
        self._count = count

    def drafts(self) -> list[int]:
        """The drafts of the last ``round`` (valid once the engine's next window has been read)."""

        return self._host_drafts[:self._count].tolist()

    def confidences(self) -> list[float]:
        """Each draft's share of the head's top-k probability (read with ``drafts``)."""

        return self._host_probs[:self._count].tolist()
