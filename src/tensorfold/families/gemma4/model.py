"""Gemma 4: mlx_lm's model with the backbone and the vocabulary head apart.

Sliding-window layers keep mlx_lm's ``RotatingKVCache``, full-attention layers its ``KVCache``; the last
``num_kv_shared_layers`` layers read an earlier layer's keys and values and have no cache of their own. The
engine only feeds tokens forward and copies whole caches at prompt checkpoints, so neither needs trimming.

``TF_GEMMA4_PIPELINE=1``: the decode step takes its token as an unread GPU array (``gpu_tokens``), so the serial
engine queues the next step before reading the token, and full-attention layers alternate their decode writes
between two buffers (``AlternatingKVCache``). Same forward, same greedy tokens; off by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class Gemma4:
    """``hidden(inputs, cache)`` is mlx_lm's ``Gemma4TextModel`` (ends in the final norm); ``head`` is the tied
    embedding as a linear plus the final logit soft-cap, as in mlx_lm's ``gemma4_text.Model.__call__``."""

    def __init__(self, model: Any) -> None:
        import mlx.core as mx

        # private arrays (the full-attention layers' ProportionalRoPE ``_freqs``) are not parameters, so mlx_lm's
        # load leaves them lazy on this thread's stream; the engine thread would fail with "There is no
        # Stream(gpu, N) in current thread". Materialize them here.
        mx.eval([v for _, module in model.named_modules() for v in module.values() if isinstance(v, mx.array)])
        self.model = model
        self.text = getattr(model, "language_model", model)      # gemma4.Model wraps gemma4_text.Model
        self.backbone = self.text.model
        self.args = self.text.args
        self.softcap = self.text.final_logit_softcapping
        self.gpu_tokens = os.environ.get("TF_GEMMA4_PIPELINE", "0") != "0"

    @property
    def layers(self) -> list[Any]:
        return self.text.layers

    def make_cache(self) -> list[Any]:
        caches = self.model.make_cache()
        if not self.gpu_tokens:
            return caches
        from mlx_lm.models.cache import KVCache

        from tensorfold.engine.alternating_kv import AlternatingKVCache

        return [AlternatingKVCache() if type(c) is KVCache else c for c in caches]

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        if not self.gpu_tokens:
            return cache
        from mlx_lm.models.cache import KVCache

        from tensorfold.engine.alternating_kv import AlternatingKVCache

        for i, item in enumerate(cache):
            if type(item) is KVCache:
                adopted = AlternatingKVCache()
                adopted.keys, adopted.values, adopted.offset = item.keys, item.values, item.offset
                cache[i] = adopted
        return cache

    def hidden(self, inputs: Any, cache: list[Any] | None = None) -> Any:
        return self.backbone(inputs, cache=cache)

    def head(self, hidden: Any) -> Any:
        # mlx_lm's compiled soft-cap: a hand-written tanh rounds differently from the compiled kernel
        from mlx_lm.models.gemma4_text import logit_softcap

        if self.text.tie_word_embeddings:
            out = self.backbone.embed_tokens.as_linear(hidden)
        else:
            out = self.text.lm_head(hidden)
        if self.softcap is not None:
            out = logit_softcap(self.softcap, out)
        return out

    def __call__(self, inputs: Any, cache: list[Any] | None = None) -> Any:
        return self.head(self.hidden(inputs, cache))


def load(model_dir: Path) -> tuple[Any, Any]:
    from mlx_lm import load as mlx_load

    loaded = mlx_load(str(model_dir))
    return Gemma4(loaded[0]), loaded[1]
