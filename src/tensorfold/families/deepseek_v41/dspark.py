"""DeepSeek-V4.1's DSpark drafter: the 3 native MTP stages draft a 5-row block in one pass from the taps.

Per the official ``inference/model.py`` (DSparkBlock / DSparkAttention / DSparkMarkovHead /
DSparkConfidenceHead): stage 0's ``main_norm(main_proj(main_hidden))`` feeds every stage's key ring
(``main_hidden`` is the concatenated taps at the target layers); the block ``[seed, noise x 4]``
runs through the standard Single-Pass block code with window-only attention where every draft row
attends the ring rows up to its own position plus the draft rows before it (a causal chain in the
engine's window semantics); the last stage's head adds the Markov bias of the token before each
draw and exposes per-position confidence. Exact arithmetic: every row runs its own one-row bits,
so a drafted block equals the same rows verified one at a time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v41.attention import Attention, sparse_attention
from tensorfold.families.deepseek_v41.caches import LayerCache
from tensorfold.families.deepseek_v41.compressor import norm_rope
from tensorfold.families.deepseek_v41.dense import dense
from tensorfold.families.deepseek_v41.model import Block, DeepSeekV41, hc_pre
from tensorfold.families.deepseek_v41.quant import quantize_cache
from tensorfold.families.deepseek_v41.rmsrows import rms_rows
from tensorfold.families.glm5_next.linear import Q, _rows, project


class DSparkAttention(Attention):
    """A stage's window-only attention: absorbed target rows write keys, draft rows attend them.

    Every draft row's keys are the SAME set: the stage's ring rows up to the last verified position
    (the last ``window`` of them — ``get_dspark_topk_idxs``'s ``arange(min(win, start_pos + 1))``)
    plus ALL block rows — the official block is **bidirectional** within itself (the noise rows are
    placeholders of one parallel draft, not a chain; RT ``test_bidirectional_draft_attention_with_sink``
    pins this). One row one ``sparse_attention`` call, so the block pass is row-exact by construction.
    Draft rows never enter the ring (the official writes only the absorbed ``main_kv``).
    """

    def absorb(self, main_x: mx.array, cache: LayerCache) -> None:
        """Context rows (main_proj of the taps) as this stage's keys at positions cache.offset .. .

        Row-exact projections: a batched absorb must write the same keys as the same rows
        absorbed one at a time (CPU quantized matmuls are not row-invariant across call shapes).
        """
        start, rows = cache.offset, int(main_x.shape[0])
        wkv = _rows(self.x_proj, self.q_rank, self.x_proj.outs)
        kv = norm_rope(project(main_x, wkv, rows_exact=True), mx.arange(start, start + rows), self.inv_freq,
                       weight=self.kv_norm, eps=self.eps)
        kv = quantize_cache(kv, 8, 32)                 # the reference's FP8 e8m0 g32 window rounding
        cache.write_keys(kv, start)
        cache.offset = start + rows

    def __call__(self, x: mx.array, caches: list[Any], lengths: tuple[int, ...], decode: bool,
                 positions: mx.array | None = None) -> mx.array:
        """x [R, D] (attn-normed), the whole block one stream: rows at positions cache.offset .. ."""
        cache = caches[0]
        rows = int(x.shape[0])
        segments = mx.arange(cache.offset, cache.offset + rows)
        xq = dense(x, self.x_proj, False)
        qr = rms_rows(xq[:, :self.q_rank], self.q_norm, self.eps)
        kv = norm_rope(xq[:, self.q_rank:], segments, self.inv_freq, weight=self.kv_norm, eps=self.eps)
        kv = quantize_cache(kv, 8, 32)
        q = dense(qr, self.wq_b, False).reshape(rows, self.heads, self.dim)
        q = norm_rope(q, segments, self.inv_freq, eps=self.eps)
        # the ring's last window (the just-absorbed row included) then the whole block, every row the same
        ring = cache.window_keys(cache.offset - 1)
        keys = mx.concatenate([ring, kv]) if int(ring.shape[0]) else kv
        outs = [sparse_attention(q[i], keys, self.sink)[None] for i in range(rows)]
        o = mx.concatenate(outs)                       # [R, H, dim]
        o = norm_rope(o, segments, self.inv_freq, norm=False, inverse=True).reshape(rows, self.groups, -1)
        u = mx.concatenate([project(o[:, g], self.wo_a[g], rows_exact=False) for g in range(self.groups)],
                           axis=-1)
        return dense(u, self.wo_b, False)


class DSpark:
    """main_proj over the taps feeds each stage's ring; a block of [seed, noise...] drafts ``size`` rows."""

    def __init__(self, blocks: list[Block], main_proj: Q, main_norm: mx.array, norm: mx.array,
                 markov_embed: Q, markov_head: Q, confidence: mx.array, *, size: int, noise: int,
                 taps: tuple[int, ...], window: int, eps: float) -> None:
        self.blocks = blocks
        self.main_proj, self.main_norm = main_proj, main_norm
        self.norm = norm
        self.markov_embed, self.markov_head = markov_embed, markov_head
        self.confidence_proj = confidence.astype(mx.float32)  # bf16 in the checkpoint, fp32 math (the reference)
        self.size, self.noise, self.taps = int(size), int(noise), tuple(taps)
        self.window, self.eps = window, eps

    def make_cache(self) -> list[LayerCache]:
        """One key ring a stage, all fresh at offset 0."""
        return [LayerCache(0, self.window) for _ in self.blocks]

    def absorb(self, taps: mx.array, caches: list[LayerCache]) -> None:
        """The target's verified rows (their taps [n, len(taps) * D]) as ring context at each stage's offset."""
        main_x = rms_rows(project(taps, self.main_proj, rows_exact=True), self.main_norm, self.eps, True)
        for block, cache in zip(self.blocks, caches):
            block.attn.absorb(main_x, cache)

    def _block_rows(self, model: DeepSeekV41, token: mx.array, caches: list[LayerCache]) -> tuple[mx.array, mx.array]:
        """The block pass over [seed, noise x (size-1)] -> (collapsed head input, its pre mix)."""
        cfg = model.args
        ids = mx.concatenate([token.reshape(1).astype(mx.uint32),
                              mx.full((self.size - 1,), self.noise, dtype=mx.uint32)])
        h = model.embed_tokens(ids)
        x = mx.contiguous(mx.broadcast_to(h[:, None, :], (self.size, cfg.hc_mult, h.shape[-1])))
        pre = mx.zeros((self.size, cfg.hc_mult), dtype=mx.float32)
        pre[:, 0] = 1.0                                 # make_identity_pre_mix: [1, 0, 0, 0]
        for block, cache in zip(self.blocks, caches):
            x, pre = block(x, ids, [cache], (self.size,), True, pre=pre)
        return hc_pre(x, pre), pre

    def logits(self, model: DeepSeekV41, token: mx.array, caches: list[LayerCache]) -> mx.array:
        """Logits [size, V] of the block after ``token``: row j predicts the token j + 1 after the context."""
        hidden = self._block_rows(model, token, caches)[0]
        normed = rms_rows(hidden, self.norm, self.eps)
        return model.head(normed[None]).reshape(self.size, -1).astype(mx.float32)

    def draw(self, logits: mx.array, token: mx.array, count: int, draw: Any) -> mx.array:
        """``count`` drafts one after another: each row's logits plus the Markov bias of the token before it."""
        drafts, prev = [], token.reshape(1).astype(mx.uint32)
        for j in range(count):
            bias = self._markov_bias(prev)
            prev = draw(logits[j:j + 1] + bias, j)
            drafts.append(prev.reshape(1).astype(mx.uint32))
        return mx.concatenate(drafts)

    def _markov_bias(self, prev: mx.array) -> mx.array:
        return self.markov_head(self._markov_embed(prev).astype(mx.float32))

    def confidence(self, hidden: mx.array, logits: mx.array, token: mx.array, count: int) -> mx.array:
        """The official confidence head: proj([hidden ‖ markov embeds]) [1, D + rank] -> [count] fp32.

        The embeds are those of the tokens the Markov iteration saw (the seed then each greedy
        draw), exactly ``DSparkConfidenceHead.forward``'s inputs; the official repo computes but
        never consumes these (no threshold or stop rule ships in the reference).
        """
        embeds = [self._markov_embed(token.reshape(1).astype(mx.uint32))]
        prev = token.reshape(1).astype(mx.uint32)
        for j in range(count - 1):
            prev = mx.argmax(logits[j] + self._markov_bias(prev), -1).reshape(1).astype(mx.uint32)
            embeds.append(self._markov_embed(prev))
        flat = mx.concatenate([hidden[:count].astype(mx.float32), mx.concatenate(embeds).astype(mx.float32)],
                              axis=-1)
        return (flat @ self.confidence_proj.T).reshape(-1)

    def _markov_embed(self, token: mx.array) -> mx.array:
        e = self.markov_embed
        if not isinstance(e, Q):                             # the converter leaves small tables BF16
            return e.weight[token]
        return mx.dequantize(e.weight[token], e.scales[token], e.biases[token], group_size=e.group, bits=e.bits)

    def arrays(self) -> list[mx.array]:
        from tensorfold.families.deepseek_v41.weights import block_arrays

        out = [*self.main_proj.arrays(), self.main_norm, self.norm, *self.markov_embed.arrays(),
               *self.markov_head.arrays(), self.confidence_proj]
        return out + [a for b in self.blocks for a in block_arrays(b)]


def load(model: DeepSeekV41, path: Path, config: dict[str, Any]) -> DSpark:
    """The drafter from a checkpoint holding native ``mtp.0/1/2`` stages (same folder or file as the backbone).

    ``config`` is the checkpoint's config.json (or its ``text_config``): the dspark_* fields name the
    block size, noise id, and the stages' own expert count and top-k, which the backbone's Config does
    not carry.
    """
    from tensorfold.families.deepseek_v41.weights import Weights, load_attention, load_moe

    cfg = model.args
    t = dict(config.get("text_config") or config)
    experts = int(t.get("dspark_n_routed_experts") or t.get("dspark_num_experts") or cfg.n_routed_experts)
    topk = int(t.get("dspark_num_experts_per_tok") or t.get("dspark_experts_per_token")
               or cfg.num_experts_per_tok)
    path = Path(path)
    w = Weights(path) if path.is_dir() else Weights.file(path)
    count = len([k for k in w.where if k.startswith("mtp.") and k.endswith(".attn_norm.weight")])
    if not count:
        raise ValueError(f"{path} holds no native DSpark stages: no mtp.<i>.attn_norm.weight entries")
    blocks = []
    for i in range(count):
        p = f"mtp.{i}"
        base = load_attention(w, cfg, cfg.num_hidden_layers + 1 + i, p)
        attn = DSparkAttention.__new__(DSparkAttention)
        attn.__dict__.update(base.__dict__)
        blocks.append(Block(attn, load_moe(w, cfg, cfg.num_hidden_layers + 1 + i, p, experts, topk),
                            w.get(f"{p}.attn_norm.weight"), w.get(f"{p}.ffn_norm.weight"),
                            w.hc(f"{p}.hc_attn", cfg), w.hc(f"{p}.hc_ffn", cfg), cfg.rms_norm_eps))
    last = f"mtp.{count - 1}"
    drafter = DSpark(blocks, w.q("mtp.0.main_proj"), w.get("mtp.0.main_norm.weight"),
                     w.get(f"{last}.norm.weight"), w.q(f"{last}.markov_head.embed"),
                     w.q(f"{last}.markov_head.head"), w.get(f"{last}.confidence_head.proj.weight"),
                     size=int(t.get("dspark_block_size") or 5), noise=int(t.get("dspark_noise_token_id") or 0),
                     taps=tuple(cfg.dspark_target_layer_ids), window=cfg.sliding_window, eps=cfg.rms_norm_eps)
    mx.eval(*drafter.arrays())
    return drafter
