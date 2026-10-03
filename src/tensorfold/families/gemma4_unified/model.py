"""Gemma 4 unified (12B) on the lane engine: mlx_lm's forward for prompts, row-exact dense kernels for decode rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.gemma4 import cache as caches
from tensorfold.families.gemma4.model import Gemma4, realize
from tensorfold.kernels.gemma.dense.v1.decode import DenseDecode


class Gemma4Unified(Gemma4):
    """mlx_lm's Gemma 4 dense text model decoded through ``DenseDecode``; drafts from its MTP assistant, if given."""

    # the assistant reads only the kept row's final hidden state: earlier prompt chunks skip MLX's last layer
    draft_reads_hidden = False

    def __init__(self, model: Any, *, backend: str | None = None, head_backend: str | None = None, check: bool = True,
                 tokenizer: Any = None, drafter: Any = None, drafts: int = 0) -> None:
        realize(model)
        self.model = model
        self.text = getattr(model, "language_model", model)
        self.backbone = self.text.model
        self.args = self.text.args
        self.decode = DenseDecode(self.text, backend or "rows", head_backend)
        self.head_drafts = None
        self.mtp = None
        self.drafts = 0
        self._last: dict[int, tuple[int, int]] = {}
        self._hidden: mx.array | None = None                 # the last forward's final-normed rows [1, N, D]
        if drafter is not None:
            drafter.bind(self)
            self.mtp, self.head_drafts = drafter, drafter
            self.drafts = int(drafts or drafter.default_drafts)
        self.exact_width, self.window_costs = (1, {})
        self.shared_costs: dict[int, float] = {}
        if check:
            self.exact_width, self.window_costs = self.check_windows(tokenizer)
        self.multi_row_exact = self.exact_width >= 2
        if check and self.multi_row_exact and not self.check_streams(tokenizer):
            self.max_streams = 1
            print("[gemma4_unified] a forward over several streams' rows does not reproduce each stream's own call "
                  "here: one stream a forward", flush=True)
        if check and self.multi_row_exact and self.max_streams > 1:
            self.shared_costs = self.time_shared_rows(tokenizer)
        if check:
            timing = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.window_costs.items()))
            shared = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.shared_costs.items()))
            print(f"[gemma4_unified] {self.decode.backend} matmul; windows of up to {self.exact_width} rows reproduce "
                  f"one-row steps here (ms by rows {timing}; shared forwards {shared or 'none'})", flush=True)

    # -- forwards: as Gemma 4's, keeping the rows for the assistant ------------------------------------------------
    def prefill(self, inputs: Any, cache: list[Any]) -> mx.array:
        out = super().prefill(inputs, cache)
        self._hidden = out
        return out

    def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> mx.array:
        out = super().hidden(inputs, cache, parents)
        self._hidden = out
        return out

    def hidden_rows(self, windows: list[Any], caches_: list[list[Any]], parents: Any = None) -> mx.array:
        out = super().hidden_rows(windows, caches_, parents)
        self._hidden = out
        return out

    # -- draft-head protocol: one MTP chain a round, read after the verify (late speculation) ---------------------
    def absorb_draft_context(self, hidden: mx.array, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        """Nothing to absorb: the assistant reads the target's caches and one row's hidden state when it drafts."""

    def speculate(self, cache: list[Any], tokens: Any, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False, rows: Any = None) -> mx.array:
        """Remember the last kept row of the last forward (its hidden state, position) and the token after it."""

        follow = [int(t) for t in (tokens.reshape(-1).tolist() if isinstance(tokens, mx.array) else tokens)]
        kept = [int(r) for r in rows] if rows is not None else list(range(start, start + len(follow)))
        first_position, first_row = self._last.get(id(cache), (int(position), 0))
        row = kept[-1]
        slot = cache[-1]
        slot.hidden = self._hidden[:, row:row + 1, :]
        slot.position = int(first_position) + row - int(first_row)
        slot.token = follow[-1]
        return mx.array([0], dtype=mx.uint32)

    def settle(self, cache: list[Any], keep: int, first: Any, position: int, sampling: Any, count: int) -> Any:
        if count <= 0:
            return []
        slot = cache[-1]
        return self.head_drafts.draft(self._layers(cache), slot.hidden, slot.token, slot.position, int(position),
                                      sampling, int(count))

    def draft_streams(self, caches_: list[list[Any]], follows: list[list[int]], rows: list[list[int]],
                      positions: list[int], samplings: list[Any], depths: list[int]) -> list[Any]:
        """Every stream of a shared round: its kept row read, then one batched chain for those that draft."""

        for cache, follow, kept, sampling in zip(caches_, follows, rows, samplings):
            self.speculate(cache, follow, 0, sampling, rows=kept)
        live = [i for i, d in enumerate(depths) if int(d) > 0]
        out: list[Any] = [[] for _ in caches_]
        if live:
            drafted = self.head_drafts.draft_many(
                [self._layers(caches_[i]) for i in live], [caches_[i][-1] for i in live],
                [int(positions[i]) for i in live], [samplings[i] for i in live], [int(depths[i]) for i in live])
            for i, d in zip(live, drafted):
                out[i] = d
        return out

    def unspeculate(self, cache: list[Any]) -> None:
        pass

    def make_cache(self) -> list[Any]:
        made = caches.make_cache(self.text)
        return made + [self.head_drafts.slot()] if self.head_drafts is not None else made

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        cache = caches.adopt(cache)
        if self.head_drafts is not None and len(cache) == len(self.text.layers):
            cache.append(self.head_drafts.slot())
        return cache


def load_target(model_dir: Path) -> tuple[Any, Any]:
    """mlx_lm's Gemma 4 (``gemma4``: the language model, vision/audio inputs dropped) on the unified checkpoint."""

    from mlx_lm.models import gemma4
    from mlx_lm.utils import load_model, load_tokenizer

    class Unified(gemma4.Model):
        """gemma4's sanitize, which also drops the unified checkpoint's patch embedder (``vision_embedder``)."""

        def sanitize(self, weights: dict) -> dict:
            return super().sanitize({k: v for k, v in weights.items()
                                     if not k.removeprefix("model.").startswith("vision_embedder")})

    # the unified text model is gemma4_text's dense decoder: the oQ and mlx-vlm layouts both sanitize to it
    model, config = load_model(Path(model_dir), get_model_classes=lambda config: (Unified, gemma4.ModelArgs))
    tokenizer = load_tokenizer(Path(model_dir), eos_token_ids=config.get("eos_token_id", None))
    return model, tokenizer


def load(model_dir: Path, *, backend: str | None = None, check: bool = True, drafter: str = "",
         drafts: int = 0) -> tuple[Gemma4Unified, Any]:
    model, tokenizer = load_target(Path(model_dir))
    realize(model)
    draft = None
    if drafter:
        from tensorfold.families.gemma4_unified.assistant import Assistant

        draft = Assistant.load(Path(drafter))
        print(f"[tensorfold] drafter {drafter} (Gemma 4 MTP assistant, {draft.describe()})", flush=True)
    return Gemma4Unified(model, backend=backend, check=check, tokenizer=tokenizer, drafter=draft,
                         drafts=drafts), tokenizer


__all__ = ["Gemma4Unified", "load", "load_target"]
