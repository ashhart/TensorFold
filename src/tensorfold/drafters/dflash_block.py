"""DFlash without DFlash2's selector: a block through the draft layers, fixed-shape parts compiled, argmax a position."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .dflash_attention import _dflash_attend


def _parts(drafter: Any) -> list[Any]:
    """Each layer's norm before attention, and its residual, norm and MLP after it, compiled once."""

    parts = getattr(drafter, "_block_parts", None)
    if parts is None:
        def make(layer: Any) -> tuple[Any, Any]:
            def post(residual: mx.array, attended: mx.array) -> mx.array:
                x = residual + attended
                return x + layer.mlp(layer.post_attention_layernorm(x))

            return mx.compile(layer.input_layernorm), mx.compile(post)

        parts = drafter._block_parts = [make(layer) for layer in drafter.model.layers]
    return parts


def block_chain(drafter: Any, inputs: mx.array, context: mx.array, cache: list[Any]) -> mx.array:
    """The block after ``inputs``' anchor: each position's most likely token [1, block - 1], unread."""

    model = drafter.model
    h = model.embed_tokens(inputs) * model.embed_scale
    h_ctx = model.hidden_norm(model.fc(context))
    masks: dict = {}                                 # one mask a layer kind (``_dflash_attend``)
    for (pre, post), layer, item in zip(_parts(drafter), model.layers, cache):
        h = post(h, _dflash_attend(layer.self_attn, pre(h), h_ctx, model.rope, item, masks))
    return mx.argmax(model.compute_logits(model.norm(h[:, 1:])), axis=-1)


__all__ = ["block_chain"]
