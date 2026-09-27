"""GLM-5.3-Flash's MTP head (``layers.<num_hidden_layers>`` in the checkpoint, DeepSeek-V3 style): drafts the
token after next.

At position i it reads the backbone's raw hidden h_i (streams collapsed, before the final norm) and the
embedding of token i+1:

    x = eh_proj([enorm(embed(t_{i+1})), hnorm(h_i)])
    x = x + sparse attention(input_layernorm(x))        over the head's own cache (MLA + indexer)
    x = x + MoE(post_attention_layernorm(x))            one stream: this layer has no hyper-connections
    logits = lm_head(shared_head.norm(x))               the backbone's head

Its output x feeds the next draft in place of h (drafts chain), as oMLX's GLM-5.3 runtime does.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next.model import GLM5, MLACache, Q, load_layer, project


class GLMMTP:
    def __init__(self, layer: Any, eh_proj: Q, enorm: mx.array, hnorm: mx.array, norm: mx.array, eps: float) -> None:
        self.layer = layer
        self.eh_proj = eh_proj
        self.enorm, self.hnorm, self.norm = enorm, hnorm, norm
        self.eps = eps

    def make_cache(self) -> MLACache:
        return MLACache()

    def __call__(self, model: GLM5, h: mx.array, tokens: mx.array, cache: MLACache, decode: bool) -> mx.array:
        """Rows h [n, D] (raw hidden) with their next tokens [n]: the head's output rows [n, D] (pre-norm)."""

        e = mx.fast.rms_norm(model.embed_tokens(tokens), self.enorm, self.eps)
        hh = mx.fast.rms_norm(h, self.hnorm, self.eps)
        x = project(mx.concatenate([e, hh], axis=-1), self.eh_proj, rows_exact=decode)
        return self.layer(x, cache, decode)

    def logits(self, model: GLM5, out: mx.array) -> mx.array:
        return model.head(mx.fast.rms_norm(out, self.norm, self.eps))


def has_mtp(model_dir: Any) -> bool:
    """Whether the checkpoint kept the nextn layer (``layers.<num_hidden_layers>.eh_proj``)."""

    import json
    from pathlib import Path

    config = json.loads((Path(model_dir) / "config.json").read_text())
    text = config.get("text_config") or config
    n = int(text.get("num_hidden_layers", 0))
    if int(text.get("num_nextn_predict_layers", 0)) < 1:
        return False
    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    names = json.loads(index.read_text())["weight_map"]
    return any(name.endswith(f"layers.{n}.eh_proj.weight") for name in names)


def load(model: GLM5) -> GLMMTP:
    """The head from the checkpoint the model was loaded from (4-bit like the backbone, its router fp32)."""

    from tensorfold.families.glm5_next.model import _materialize

    w = model.weights
    cfg = model.args
    i = cfg.num_hidden_layers
    layer = load_layer(w, i, cfg, plain=True)
    head = GLMMTP(layer, w.q(f"layers.{i}.eh_proj"), w.get(f"layers.{i}.enorm.weight"),
                  w.get(f"layers.{i}.hnorm.weight"), w.get(f"layers.{i}.shared_head.norm.weight"), cfg.rms_norm_eps)
    _materialize(head.eh_proj, head.enorm, head.hnorm, head.norm)
    return head
