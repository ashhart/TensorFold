"""The lane-protocol runtime for DeepSeek-V4.1: GLMFlash's engine integration over the V4.1 backbone.

``DSparkFlash`` adds the native 3-stage MTP drafter: the stages' key rings ride behind the backbone's
cache list, ``speculate``/``settle``/``draft_streams`` absorb a round's kept taps and draft a block
(one pass, every row its own bits — a drafted round equals the same rows verified one at a time),
and ``keep_rows`` rolls the rings back with the pools and the engram history (a trim moves offsets
only; the rings' slots are pure functions of their positions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.deepseek_v41.config import PREFILL_QUERIES
from tensorfold.families.glm5_next.runtime import GLMFlash


class DeepSeekV41Flash(GLMFlash):
    """The V4.1 backbone behind the lane protocol; prefill runs chunk by chunk through ``hidden``."""

    tag = "deepseek_v41"
    hidden_pass = None                  # V4.1's backbone has no prompt pass: the engine feeds a chunk a forward

    def __init__(self, model: Any, head: Any = None, *, drafts: int = 0, check: bool = True) -> None:
        self.token_map = getattr(model, "token_map", None)
        super().__init__(model, head, drafts=drafts, check=check)
        # the backbone's own cache list ends with the engram history: keep it, slice only what this
        # runtime appended (the MTP cache), so the engine's engram layer sees the stream's history
        self.layer_count = len(model.layers) + 1

    @property
    def prefill_workspace_per_token(self) -> int:
        """Prefill bytes a position: a query block's fp32 indexer scores over the pool."""
        return PREFILL_QUERIES * self.args.index_n_heads * 4 // 2

    def keep_rows(self, cache: list[Any], rows: int, keep: Any) -> None:
        """Keep the first ``keep`` of ``rows`` rows: layer rings, pool groups, engram history (prefix keeps only)."""
        self.model.keep_rows(cache, rows, self._kept(keep))

    def absorb_kept(self, cache: list[Any], tokens: mx.array, position: int, start: int = 0,
                    rows: Any = None) -> None:
        """A plain round: the head read the kept rows (the engine calls this when it drafts none)."""

    def draft_rows(self) -> mx.array:
        return self.model.last_streams

    def blank_draft_rows(self) -> mx.array:
        return mx.zeros((1, self.args.hc_mult, self.args.hidden_size), dtype=mx.bfloat16)


class DSparkFlash(DeepSeekV41Flash):
    """The V4.1 backbone with its native DSpark stages drafting after a round is read."""

    speculate_early = False             # the head reads the round's taps, so it drafts after the read
    dspark: Any = None

    def __init__(self, model: Any, drafter: Any, *, check: bool = True) -> None:
        model.tap_layers = drafter.taps
        super().__init__(model, None, drafts=drafter.size, check=check)
        self.dspark = drafter if self.multi_row_exact else None
        self.mtp = self.dspark                      # what the engine reads to draft with a head
        self.mtp_step_ms = self._time_pass() / drafter.size if self.dspark is not None else 0.0

    def make_cache(self) -> list[Any]:
        caches = self.model.make_cache()
        return caches + (self.dspark.make_cache() if self.dspark is not None else [])

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        if self.dspark is not None and len(cache) == self.layer_count:
            cache.extend(self.dspark.make_cache())
        return cache

    def draft_rows(self) -> mx.array:
        return self.model.last_taps                 # type: ignore[attr-defined]  # set when tap_layers is non-empty

    def blank_draft_rows(self) -> mx.array:
        return mx.zeros((1, len(self.dspark.taps) * int(self.args.hidden_size)), dtype=mx.bfloat16)

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        """A prompt chunk's rows into the rings (only the last window's worth is read; the rest moves the offset)."""

        count = int(np.asarray(next_tokens).size) if not isinstance(next_tokens, mx.array) else int(next_tokens.size)
        rings = cache[self.layer_count:]
        skip = max(0, count - self.dspark.window)
        for ring in rings:
            ring.offset += skip
        self.dspark.absorb(self.model.last_taps[skip:count], rings)   # type: ignore[attr-defined]

    def _drafts(self, rings: list[Any], token: mx.array, sampling: Any, key: int, count: int) -> mx.array:
        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        logits = self.dspark.logits(self.model, token, rings)
        return self.dspark.draw(logits, token, count, lambda row, j: gpu_sample(row, sampling, [key + j]))

    def speculate(self, cache: list[Any], tokens: mx.array, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False, rows: Any = None) -> mx.array:
        """Absorb rows ``start`` .. (or ``rows``) of the last forward, then draft a block after the last token."""

        tokens = tokens.reshape(-1).astype(mx.uint32)
        count = int(tokens.shape[0])
        index = mx.array([int(r) for r in rows], dtype=mx.int32) if rows is not None else None
        taps = mx.take(self._rows, index, axis=0) if index is not None else self._rows[start:start + count]
        rings = cache[self.layer_count:]
        self.dspark.absorb(taps, rings)
        drafts = self._drafts(rings, tokens[-1:], sampling, position + 1 + count, self.dspark.size)
        self._specs[id(rings[0])] = (drafts, count)
        return drafts[:1]

    def settle(self, cache: list[Any], keep: int, first: Any, position: int, sampling: Any, count: int) -> Any:
        """Trim the rings to the accepted prefix, then hand back ``count`` of the drafted block."""
        rings = cache[self.layer_count:]
        drafts, absorbed = self._specs.pop(id(rings[0]))
        if absorbed > keep:
            for ring in rings:
                ring.trim(absorbed - keep)
        if count <= 0:
            return []
        out = drafts[:count]
        mx.async_eval(out)
        return out

    def unspeculate(self, cache: list[Any]) -> None:
        """Undo ``speculate`` entirely (the round's rows are absorbed another way)."""
        rings = cache[self.layer_count:]
        spec = self._specs.pop(id(rings[0]), None)
        if spec is not None:
            for ring in rings:
                ring.trim(spec[1])

    def draft_streams(self, caches: list[list[Any]], follows: list[list[int]], rows: list[list[int]],
                      positions: list[int], samplings: list[Any], depths: list[int]) -> list[Any]:
        """Each stream's rings take its kept rows of the shared round, then its own pass drafts ``depths[i]``."""

        out: list[Any] = []
        for cache, follow, kept, key, sampling, depth in zip(caches, follows, rows, positions, samplings, depths):
            rings = cache[self.layer_count:]
            self.dspark.absorb(mx.take(self._rows, mx.array([int(r) for r in kept], dtype=mx.int32), axis=0),
                               rings)
            if depth <= 0:
                out.append([])
                continue
            token = mx.array([int(follow[-1])], dtype=mx.uint32)
            out.append(self._drafts(rings, token, sampling, key, min(int(depth), self.dspark.size)))
        mx.async_eval(*[d for d in out if isinstance(d, mx.array)])
        return out

    def _time_pass(self) -> float:
        """One draft pass after one absorbed row, in ms (fastest of 5): the depth rule's cost of a block."""

        import time

        rings = self.dspark.make_cache()
        taps = mx.zeros((1, len(self.dspark.taps) * int(self.args.hidden_size)), dtype=mx.bfloat16)
        best = float("inf")
        for i in range(6):
            started = time.perf_counter()
            self.dspark.absorb(taps, rings)
            mx.eval(self._drafts(rings, mx.array([3000 + i], dtype=mx.uint32), None, 100 + i, self.dspark.size))
            if i:
                best = min(best, (time.perf_counter() - started) * 1e3)
        return round(best, 3)


def load(model_dir: Path, *, mtp_drafts: int | None = None, check: bool = True,
         **_: Any) -> tuple[DeepSeekV41Flash, Any]:
    """The runtime and tokenizer; the native DSpark stages draft when the checkpoint holds them.

    ``mtp_drafts`` (default 3, 0: none) caps the drafted rows a round takes from a block.
    """
    import json

    from tensorfold.families.tokenizer import load_tokenizer
    from tensorfold.families.deepseek_v41 import dspark as dspark_module
    from tensorfold.families.deepseek_v41 import weights
    from tensorfold.families.deepseek_v41.prompts import DeepSeekTokenizer

    model = weights.load_backbone(Path(model_dir))
    tokenizer = DeepSeekTokenizer(load_tokenizer(Path(model_dir), eos_token_ids=model.args.eos_token_id or None))
    drafts = 3 if mtp_drafts is None else int(mtp_drafts)
    config = json.loads((Path(model_dir) / "config.json").read_text())
    has_stages = model.has_draft_weights
    if drafts > 0 and has_stages:
        drafter = dspark_module.load(model, Path(model_dir), config)
        runtime: DeepSeekV41Flash = DSparkFlash(model, drafter, check=check)
        kind = f"DSpark, {len(drafter.blocks)} stages, step {runtime.mtp_step_ms} ms"
    else:
        runtime = DeepSeekV41Flash(model, None, drafts=0, check=check)
        kind = "no draft head"
    print(f"[deepseek_v41] exact window {runtime.exact_width} rows, forward ms by width {runtime.window_costs}, "
          f"{kind}", flush=True)
    return runtime, tokenizer
