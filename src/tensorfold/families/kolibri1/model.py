"""Kolibri 1 on the lane engine: mlx_lm's forward (the vendored ``kolibri1``) for prompts, row-exact kernels for every
decode row."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.gemma4.cache import LinearKVCache, RingKVCache
from tensorfold.families.gemma4.model import Gemma4


def register() -> None:
    """The vendored architecture as ``mlx_lm.models.kolibri1``, the module mlx_lm's loader imports by model type."""

    from tensorfold.families.kolibri1.vendor import kolibri1

    sys.modules["mlx_lm.models.kolibri1"] = kolibri1


class Kolibri1:
    """mlx_lm's Kolibri 1 with the backbone and the head apart, decoded through ``RowDecode`` (4-bit checkpoints) or
    mlx_lm's forward one row a step (``backend`` None)."""

    lane_family = True
    # ``hidden`` takes an unread GPU token: one-token rounds run one step ahead
    gpu_tokens = True
    mtp = None
    drafts = 0
    speculate_early = False
    # the widest verify window checked at load
    fused_rows = 16
    # a shared forward's rows and streams (``hidden_rows``): the bf16 router's matmul is row-exact to 16 rows here
    batch_rows = 16
    max_streams = 8

    # Gemma 4's load-time checks: they read only the lane interface below
    _check_tokens = Gemma4._check_tokens
    _base = Gemma4._base
    check_windows = Gemma4.check_windows
    check_streams = Gemma4.check_streams
    time_shared_rows = Gemma4.time_shared_rows
    _kept = staticmethod(Gemma4._kept)
    _chain_only = staticmethod(Gemma4._chain_only)
    _tokens = staticmethod(Gemma4._tokens)

    def __init__(self, model: Any, *, backend: str | None = "rows", check: bool = True, tokenizer: Any = None) -> None:
        self.model = model
        self.backbone = model.model
        self.args = model.args
        self.decode = None
        if backend is not None:
            from tensorfold.kernels.kolibri.v1.decode import RowDecode

            self.decode = RowDecode(model, backend)
        # mlx_lm's batched kernels give a window's rows other bits than one-row steps: without RowDecode, one row
        self.exact_width, self.window_costs = 1, {}
        self.shared_costs: dict[int, float] = {}
        if check and self.decode is not None:
            self.exact_width, self.window_costs = self.check_windows(tokenizer)
        self.multi_row_exact = self.exact_width >= 2
        if not self.multi_row_exact:
            self.max_streams = 1
        elif check and not self.check_streams(tokenizer):
            self.max_streams = 1
            print("[kolibri1] a forward over several streams' rows does not reproduce each stream's own call here: one "
                  "stream a forward", flush=True)
        if check and self.multi_row_exact and self.max_streams > 1:
            self.shared_costs = self.time_shared_rows(tokenizer)
        if check and self.decode is not None:
            timing = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.window_costs.items()))
            shared = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.shared_costs.items()))
            print(f"[kolibri1] {self.decode.backend} matmul; windows of up to {self.exact_width} rows reproduce one-row "
                  f"steps here (ms by rows {timing}; shared forwards {shared or 'none'})", flush=True)

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
        """A prompt chunk [1, L] through mlx_lm's forward (MLX's batched kernels) into the stream's caches."""

        return self.backbone(inputs, cache=cache)

    def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> mx.array:
        """Hidden rows [1, R, D] of R consecutive tokens (an array, or an unread GPU token), advancing the caches."""

        self._chain_only(parents)
        tokens = self._tokens(inputs)
        if self.decode is None:
            return self.backbone(tokens[None], cache=cache)
        return self.decode(tokens, [(cache, int(tokens.shape[0]), cache[0].offset)])[None]

    def hidden_rows(self, windows: list[Any], caches_: list[list[Any]], parents: Any = None) -> mx.array:
        """Every stream's window in one forward [1, N, D], stream i's rows advancing only ``caches_[i]``."""

        for rows in parents or ():
            self._chain_only(rows)
        parts = [self._tokens(w) for w in windows]
        tokens = mx.concatenate(parts) if len(parts) > 1 else parts[0]
        return self.decode(tokens, [(c, int(p.shape[0]), c[0].offset) for c, p in zip(caches_, parts)])[None]

    def head(self, hidden: mx.array) -> mx.array:
        if self.decode is None:
            return self.model.lm_head(hidden)
        return self.decode.logits(hidden)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))

    def keep_rows(self, cache: list[Any], rows: int, keep: Any) -> None:
        """After an R-row call: the caches keep its first ``keep`` rows (a count, or a chain's path)."""

        drop = int(rows) - self._kept(keep)
        if drop:
            for c in cache:
                c.trim(drop)

    def keep_rows_streams(self, caches_: list[list[Any]], lengths: Any, keeps: Any) -> None:
        for cache, rows, keep in zip(caches_, lengths, keeps):
            self.keep_rows(cache, rows, keep)


def load(model_dir: Path, *, backend: str | None = "rows", check: bool = True) -> tuple[Kolibri1, Any]:
    from mlx_lm import load as mlx_load

    from tensorfold.families.gemma4.model import realize

    register()
    model, tokenizer = mlx_load(str(model_dir))
    realize(model)
    return Kolibri1(model, backend=backend, check=check, tokenizer=tokenizer), tokenizer
