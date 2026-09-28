"""DeepSeek's DSpark drafter: three MoE blocks draft a 5-token block in one pass from the target's layer taps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v4.attention import Attention
from tensorfold.families.deepseek_v4.caches import LayerCache
from tensorfold.families.deepseek_v4.compressor import norm_rope
from tensorfold.families.deepseek_v4.model import Block, DeepSeekV4, HeadHC
from tensorfold.families.glm5_next.linear import Q, _rows, project


class DSparkAttention(Attention):
    """Draft rows attend the last 128 context rows and each other (no mask); context rows only write keys."""

    def absorb(self, main_x: mx.array, cache: LayerCache) -> None:
        """Context rows (main_proj of the taps) at positions cache.offset .. into the key ring."""

        start, rows = cache.offset, int(main_x.shape[0])
        wkv = _rows(self.x_proj, self.q_rank, self.x_proj.outs)
        kv = norm_rope(project(main_x, wkv, rows_exact=False), mx.arange(start, start + rows), self.inv_freq,
                       weight=self.kv_norm, eps=self.eps)
        cache.write_keys(kv, start)
        cache.offset = start + rows

    def __call__(self, x: mx.array, caches: list[LayerCache], lengths: tuple[int, ...], decode: bool,
                 positions: mx.array | None = None) -> mx.array:
        cache, rows = caches[0], int(x.shape[0])
        start = cache.offset
        segments = mx.arange(start, start + rows)
        xq = project(x, self.x_proj, rows_exact=False)
        qr = mx.fast.rms_norm(xq[:, :self.q_rank], self.q_norm, self.eps)
        kv = norm_rope(xq[:, self.q_rank:], segments, self.inv_freq, weight=self.kv_norm, eps=self.eps)
        q = project(qr, self.wq_b, rows_exact=False).reshape(rows, self.heads, self.dim)
        q = norm_rope(q, segments, self.inv_freq, eps=self.eps)
        lo = max(0, start - self.window)
        keys = mx.concatenate([cache.ring_rows(cache.keys, lo, start), kv]) if start > lo else kv
        o = norm_rope(self._attend(q, keys), segments, self.inv_freq, norm=False, inverse=True).reshape(
            rows, self.groups, -1)
        u = mx.concatenate([project(o[:, g], self.wo_a[g], rows_exact=False) for g in range(self.groups)], axis=-1)
        return project(u, self.wo_b, rows_exact=False)


class DSpark:
    """main_proj over the taps feeds each block's ring; a block of [token, noise...] gives ``size`` draft logits."""

    def __init__(self, blocks: list[Block], main_proj: Q, main_norm: mx.array, head_hc: HeadHC, norm: mx.array,
                 markov_in: mx.array, markov_out: mx.array, *, size: int, noise: int, taps: tuple[int, ...],
                 window: int, eps: float) -> None:
        self.blocks = blocks
        self.main_proj, self.main_norm = main_proj, main_norm
        self.head_hc, self.norm = head_hc, norm
        self.markov_in, self.markov_out = markov_in, markov_out
        self.size, self.noise, self.taps = int(size), int(noise), tuple(taps)
        self.window, self.eps = window, eps

    def make_cache(self) -> list[LayerCache]:
        return [LayerCache(0, self.window) for _ in self.blocks]

    def absorb(self, taps: mx.array, caches: list[LayerCache]) -> None:
        """The target's rows (their taps [n, 3D]) as context at the caches' next positions."""

        main_x = mx.fast.rms_norm(project(taps, self.main_proj, rows_exact=False), self.main_norm, self.eps)
        for block, cache in zip(self.blocks, caches):
            block.attn.absorb(main_x, cache)

    def logits(self, model: DeepSeekV4, token: mx.array, caches: list[LayerCache]) -> mx.array:
        """Logits [size, V] of the block after ``token``: row j predicts the token j + 2 places after the context."""

        ids = mx.concatenate([token.reshape(1).astype(mx.uint32),
                              mx.full((self.size - 1,), self.noise, dtype=mx.uint32)])
        h = model.embed_tokens(ids)
        x = mx.contiguous(mx.broadcast_to(h[:, None, :], (self.size, 4, h.shape[-1])))
        for block, cache in zip(self.blocks, caches):
            x = block(x, ids, [cache], (self.size,), False)
        return model.head(mx.fast.rms_norm(self.head_hc(x, False), self.norm, self.eps)[None])[0]

    def draw(self, logits: mx.array, token: mx.array, count: int, draw: Any) -> mx.array:
        """``count`` drafts one after another, each row's logits plus the Markov bias of the token before it."""

        drafts, prev = [], token.reshape(1).astype(mx.uint32)
        for j in range(count):
            bias = self.markov_in[prev] @ self.markov_out.T
            prev = draw(logits[j:j + 1].astype(mx.float32) + bias.astype(mx.float32), j)
            drafts.append(prev.reshape(1).astype(mx.uint32))
        return mx.concatenate(drafts)

    def arrays(self) -> list[mx.array]:
        from tensorfold.families.deepseek_v4.weights import block_arrays

        out = [*self.main_proj.arrays(), self.main_norm, self.head_hc.fn, self.head_hc.base, self.head_hc.scale,
               self.norm, self.markov_in, self.markov_out]
        return out + [a for b in self.blocks for a in block_arrays(b)]


def load(model: DeepSeekV4, path: Path, config: dict[str, Any]) -> DSpark:
    """The drafter from a converted file (``convert.convert_dspark``) and the DSpark config's block fields."""

    from tensorfold.families.deepseek_v4.weights import Weights, load_attention, load_moe

    cfg = model.args
    w = Weights.file(Path(path))
    count = len([k for k in w.where if k.endswith(".attn_norm.weight")])
    blocks = []
    for i in range(count):
        p = f"dspark.{i}"
        base = load_attention(w, p, cfg, cfg.num_hidden_layers + 1 + i)
        attn = DSparkAttention.__new__(DSparkAttention)
        attn.__dict__.update(base.__dict__)
        blocks.append(Block(attn, load_moe(w, p, cfg, cfg.num_hidden_layers + 1 + i), w.get(f"{p}.attn_norm.weight"),
                            w.get(f"{p}.ffn_norm.weight"), w.hc(f"{p}.attn_hc", cfg), w.hc(f"{p}.ffn_hc", cfg),
                            cfg.rms_norm_eps))
    last = f"dspark.{count - 1}"
    head = HeadHC(w.get(f"{last}.hc_head.fn"), w.get(f"{last}.hc_head.base"), w.get(f"{last}.hc_head.scale"),
                  cfg.rms_norm_eps, cfg.hc_eps)
    drafter = DSpark(blocks, w.q("dspark.0.main_proj"), w.get("dspark.0.main_norm.weight"), head,
                     w.get(f"{last}.norm.weight"), w.get(f"{last}.markov_head.markov_w1.weight"),
                     w.get(f"{last}.markov_head.markov_w2.weight"), size=int(config["dspark_block_size"]),
                     noise=int(config["dspark_noise_token_id"]), taps=tuple(config["dspark_target_layer_ids"]),
                     window=cfg.sliding_window, eps=cfg.rms_norm_eps)
    mx.eval(*drafter.arrays())
    return drafter
