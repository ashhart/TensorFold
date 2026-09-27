"""GLM-5.3-Flash's MTP layer as draft head: eh_proj over the next token and the normed row, then MLA and MoE."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next.caches import MLACache
from tensorfold.families.glm5_next.linear import project
from tensorfold.families.glm5_next.weights import load_layer
from tensorfold.families.glm5_next.model import GLM5


class GLMMTP:
    def __init__(self, layer: Any, eh_proj: Any, enorm: mx.array, hnorm: mx.array, norm: mx.array, eps: float) -> None:
        self.layer = layer
        self.eh_proj = eh_proj
        self.enorm, self.hnorm, self.norm = enorm, hnorm, norm
        self.eps = eps

    def make_cache(self) -> MLACache:
        return MLACache()

    def __call__(self, model: GLM5, h: mx.array, tokens: mx.array, caches: list[MLACache], lengths: tuple[int, ...],
                 decode: bool) -> mx.array:
        """Rows h [n, D] with their next tokens, ``lengths`` rows for each stream's head cache: output rows [n, D]."""

        e = mx.fast.rms_norm(model.embed_tokens(tokens), self.enorm, self.eps)
        hh = mx.fast.rms_norm(h, self.hnorm, self.eps)
        x = project(mx.concatenate([e, hh], axis=-1), self.eh_proj, rows_exact=decode)
        return self.layer(x, caches, lengths, decode)

    def logits(self, model: GLM5, out: mx.array) -> mx.array:
        return model.head(mx.fast.rms_norm(out, self.norm, self.eps))


def load(model: GLM5) -> GLMMTP:
    """The head from the model's checkpoint (its router fp32; an unquantized ``eh_proj`` read as ``Dense``)."""

    from tensorfold.families.glm5_next.weights import _materialize

    w = model.weights
    cfg = model.args
    i = cfg.num_hidden_layers
    layer = load_layer(w, i, cfg, plain=True)
    head = GLMMTP(layer, w.linear(f"layers.{i}.eh_proj"), w.get(f"layers.{i}.enorm.weight"),
                  w.get(f"layers.{i}.hnorm.weight"), w.get(f"layers.{i}.shared_head.norm.weight"), cfg.rms_norm_eps)
    _materialize(head.eh_proj, head.enorm, head.hnorm, head.norm)
    return head
