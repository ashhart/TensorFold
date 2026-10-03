"""DeepSeek-V4.1's own DSpark drafter (``mtp.0-2``), after oMLX's ``patches/deepseek_v41/dspark.py`` (MIT)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v41 import kernels as KV
from tensorfold.families.deepseek_v41.caches import LayerCache
from tensorfold.families.deepseek_v41.model import Attention, Block, DeepSeekV41, hc_pre, rms, rope
from tensorfold.families.deepseek_v41.quant import Linear, fp8


class DSparkAttention:
    """A draft block's attention: context rows only write keys; draft rows attend those and each other."""

    def __init__(self, base: Attention) -> None:
        self.__dict__.update(base.__dict__)

    def arrays(self) -> list[mx.array]:
        return Attention.arrays(self)        # type: ignore[arg-type]

    def _kv(self, x: mx.array, positions: mx.array) -> mx.array:
        return fp8(rope(rms(self.wkv(x), self.kv_norm, self.eps), positions, self.inv_freq))

    def absorb(self, main_x: mx.array, cache: LayerCache) -> None:
        """Context rows (main_proj of the taps) at positions cache.offset .. into the key ring."""

        start, rows = cache.offset, int(main_x.shape[0])
        cache.write_keys(self._kv(main_x, mx.arange(start, start + rows)), start)
        cache.offset = start + rows

    def __call__(self, x: mx.array, cache: LayerCache) -> mx.array:
        rows, start = int(x.shape[0]), cache.offset
        positions = mx.arange(start, start + rows)
        q = rope(self.wq_b(rms(self.wq_a(x), self.q_norm, self.eps)).reshape(rows, self.heads, self.dim), positions,
                 self.inv_freq)
        draft = self._kv(x, positions).astype(mx.float32)
        lo = max(0, start - self.window)
        kv = mx.concatenate([cache.key_rows(lo, start), draft]) if start > lo else draft
        out = mx.fast.scaled_dot_product_attention(q.astype(mx.float32).transpose(1, 0, 2)[None], kv[None, None],
                                                   kv[None, None], scale=self.scale, sinks=self.sink)
        out = out[0].transpose(1, 0, 2).astype(q.dtype)
        out = rope(out, positions, self.inv_freq, inverse=True)
        u = mx.einsum("lgd,grd->lgr", out.reshape(rows, self.groups, -1), self.wo_a).reshape(rows, -1)
        return self.wo_b(u)


class DSparkBlock:
    """A draft block: Block's single-pass mHC with separate hc_pre and RMSNorm (as oMLX's DSparkBlock)."""

    def __init__(self, block: Block) -> None:
        self.block = block
        self.attn = DSparkAttention(block.attn)

    def arrays(self) -> list[mx.array]:
        return self.block.arrays()

    def __call__(self, h: mx.array, pre: mx.array, cache: LayerCache) -> tuple[mx.array, mx.array]:
        b = self.block
        ap, ao, ac = b.attn_hc(h)
        h = KV.hc_post(self.attn(rms(hc_pre(h, pre), b.attn_norm, b.eps), cache), h, ao, ac)
        fp, fo, fc = b.ffn_hc(h)
        return KV.hc_post(b.moe(rms(hc_pre(h, ap), b.ffn_norm, b.eps)), h, fo, fc), fp


class DSpark:
    """main_proj over the taps feeds each block's ring; a block of [token, noise...] gives ``size`` draft logits."""

    def __init__(self, blocks: list[DSparkBlock], main_proj: Linear, main_norm: mx.array, norm: mx.array,
                 markov_embed: mx.array, markov_head: mx.array, *, size: int, noise: int, taps: tuple[int, ...],
                 window: int, eps: float, hc: int) -> None:
        self.blocks = blocks
        self.main_proj, self.main_norm, self.norm = main_proj, main_norm, norm
        self.markov_embed, self.markov_head = markov_embed, markov_head
        self.size, self.noise, self.taps = int(size), int(noise), tuple(taps)
        self.window, self.eps, self.hc = int(window), float(eps), int(hc)

    def make_cache(self) -> list[LayerCache]:
        return [LayerCache(0, self.window) for _ in self.blocks]

    def absorb(self, taps: mx.array, caches: list[LayerCache]) -> None:
        """The target's rows (their taps [n, 3D]) as context at the caches' next positions."""

        main_x = rms(self.main_proj(taps), self.main_norm, self.eps)
        for block, cache in zip(self.blocks, caches):
            block.attn.absorb(main_x, cache)

    def logits(self, model: DeepSeekV41, token: mx.array, caches: list[LayerCache]) -> mx.array:
        """Logits [size, V] of the block after ``token``: row j predicts the token j + 2 places after the context."""

        ids = mx.concatenate([token.reshape(1).astype(mx.uint32), mx.full((self.size - 1,), self.noise, mx.uint32)])
        h = model.embed_tokens(ids)
        x = mx.contiguous(mx.broadcast_to(h[:, None, :], (self.size, self.hc, h.shape[-1])))
        pre = mx.broadcast_to((mx.arange(self.hc) == 0).astype(mx.float32), (self.size, self.hc))
        for block, cache in zip(self.blocks, caches):
            x, pre = block(x, pre, cache)
        normed = rms(hc_pre(x, pre), self.norm, self.eps)
        return KV.head_logits(normed, model.lm_head, False)

    def draw(self, logits: mx.array, token: mx.array, count: int, draw: Any) -> mx.array:
        """``count`` drafts one after another, each row's logits plus the Markov bias of the token before it."""

        drafts, prev = [], token.reshape(1).astype(mx.uint32)
        for j in range(count):
            bias = KV.head_logits(self.markov_embed[prev], self.markov_head, False)
            prev = draw(logits[j:j + 1].astype(mx.float32) + bias, j)
            drafts.append(prev.reshape(1).astype(mx.uint32))
        return mx.concatenate(drafts)

    def arrays(self) -> list[mx.array]:
        out = [*self.main_proj.arrays(), self.main_norm, self.norm, self.markov_embed, self.markov_head]
        return out + [a for b in self.blocks for a in b.arrays()]


def load(model: DeepSeekV41, w: Any) -> DSpark:
    """The drafter from the checkpoint's ``language_model.mtp.*`` tensors (``w``: the backbone's ``Weights``)."""

    from dataclasses import replace

    from tensorfold.families.deepseek_v41.weights import PREFIX, load_attention, load_moe

    cfg = model.args
    if cfg.dspark_block_size <= 0 or not cfg.dspark_target_layer_ids or cfg.num_nextn_predict_layers <= 0:
        raise ValueError("deepseek_v41: this checkpoint's config has no complete DSpark drafter")
    draft = replace(cfg, n_routed_experts=cfg.dspark_n_routed_experts,
                    num_experts_per_tok=cfg.dspark_num_experts_per_tok)
    blocks = []
    for s in range(cfg.num_nextn_predict_layers):
        p = f"{PREFIX}.mtp.{s}"
        layer = cfg.num_hidden_layers + s
        if cfg.ratio(layer):
            raise ValueError("deepseek_v41: DSpark stages attend a window only")
        block = Block(layer, load_attention(w, p, draft, layer), load_moe(w, p, draft),
                      w.get(f"{p}.attn_norm.weight"), w.get(f"{p}.ffn_norm.weight"), w.hc(p, "attn", draft),
                      w.hc(p, "ffn", draft), cfg.rms_norm_eps)
        blocks.append(DSparkBlock(block))
    first, last = f"{PREFIX}.mtp.0", f"{PREFIX}.mtp.{cfg.num_nextn_predict_layers - 1}"
    drafter = DSpark(blocks, w.lin(f"{first}.main_proj"), w.get(f"{first}.main_norm.weight"),
                     w.get(f"{last}.norm.weight"), w.get(f"{last}.markov_head.embed.weight"),
                     w.dense(f"{last}.markov_head.head"), size=cfg.dspark_block_size,
                     noise=cfg.dspark_noise_token_id, taps=tuple(cfg.dspark_target_layer_ids),
                     window=cfg.sliding_window, eps=cfg.rms_norm_eps, hc=cfg.hc_mult)
    mx.eval(*drafter.arrays())
    return drafter


def has_drafter(raw: dict[str, Any], where: dict[str, str]) -> bool:
    """The checkpoint kept its DSpark stages (oMLX's ``preserve_mtp``)."""

    return any(k.startswith("language_model.mtp.0.main_proj") for k in where)
