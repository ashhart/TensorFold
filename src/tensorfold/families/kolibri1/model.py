"""Kolibri 1 on the lane engine: mlx_lm's forward (the vendored ``kolibri1``) for prompts and decode rows alike."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.gemma4.cache import LinearKVCache, RingKVCache


def register() -> None:
    """The vendored architecture as ``mlx_lm.models.kolibri1``, the module mlx_lm's loader imports by model type."""

    from tensorfold.families.kolibri1 import mlx_kolibri1

    sys.modules["mlx_lm.models.kolibri1"] = mlx_kolibri1


class Kolibri1:
    """mlx_lm's Kolibri 1 with the backbone and the head apart; one row a step until a row-exact decode lands."""

    lane_family = True
    # ``hidden`` takes an unread GPU token: one-token rounds run one step ahead
    gpu_tokens = True
    mtp = None
    drafts = 0
    batch_rows = 1
    max_streams = 1

    def __init__(self, model: Any) -> None:
        self.model = model
        self.backbone = model.model
        self.args = model.args
        # mlx_lm's batched kernels give a window's rows other bits than one-row steps: one row a step, alone
        self.exact_width, self.window_costs = 1, {}
        self.shared_costs: dict[int, float] = {}
        self.multi_row_exact = False

    # -- the engine's model interface --------------------------------------------------------------------------------
    @property
    def layers(self) -> list[Any]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        """A ring for each sliding-window layer, a growing cache for each full-attention one."""

        window = int(self.args.sliding_window)
        return [LinearKVCache() if layer.is_full_attention else RingKVCache(window) for layer in self.backbone.layers]

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        from tensorfold.families.gemma4.cache import adopt

        return adopt(cache)

    def prefill(self, inputs: Any, cache: list[Any]) -> mx.array:
        """A prompt chunk [1, L] through mlx_lm's forward into the stream's caches."""

        return self.backbone(inputs, cache=cache)

    def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> mx.array:
        """Hidden rows [1, R, D] of R consecutive tokens (an array, or an unread GPU token), advancing the caches."""

        if parents is not None and list(parents) != list(range(-1, len(parents) - 1)):
            raise NotImplementedError("Kolibri 1 verifies draft chains, not trees")
        return self.backbone(self._tokens(inputs)[None], cache=cache)

    def head(self, hidden: mx.array) -> mx.array:
        return self.model.lm_head(hidden)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))

    def keep_rows(self, cache: list[Any], rows: int, keep: Any) -> None:
        """After an R-row call: the caches keep its first ``keep`` rows (a count, or a chain's path)."""

        if isinstance(keep, int):
            kept = keep
        else:
            path = [int(r) for r in keep]
            if path != list(range(len(path))):
                raise NotImplementedError("Kolibri 1 keeps a prefix of a window's rows")
            kept = len(path)
        drop = int(rows) - kept
        if drop:
            for c in cache:
                c.trim(drop)

    def keep_rows_streams(self, caches_: list[list[Any]], lengths: Any, keeps: Any) -> None:
        for cache, rows, keep in zip(caches_, lengths, keeps):
            self.keep_rows(cache, rows, keep)

    @staticmethod
    def _tokens(inputs: Any) -> mx.array:
        if isinstance(inputs, mx.array):
            return inputs.reshape(-1).astype(mx.uint32)
        return mx.array([int(t) for t in inputs], dtype=mx.uint32).reshape(-1)


def load(model_dir: Path) -> tuple[Kolibri1, Any]:
    from mlx_lm import load as mlx_load

    register()
    model, tokenizer = mlx_load(str(model_dir))
    return Kolibri1(model), tokenizer
