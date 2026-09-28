"""DeepSeek-V4-Flash as the lane engine's family rounds drive it: GLM-5.3's runtime with V4's draft heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.deepseek_v4.config import PREFILL_QUERIES
from tensorfold.families.deepseek_v4.mtp import MTPCache
from tensorfold.families.glm5_next.runtime import GLMFlash

MTP_FILE = "mtp.safetensors"          # the converted MTP layer in a drafter folder (convert.convert_mtp)
DSPARK_FILE = "dspark.safetensors"    # the converted DSpark blocks (convert.convert_dspark), beside its config.json


class DeepSeekFlash(GLMFlash):
    """The backbone behind the lane protocol; the MTP head reads the 4 streams of the rows it drafts from."""

    tag = "deepseek_v4"

    def new_mtp_cache(self) -> Any:
        return MTPCache(self.args.sliding_window)

    def draft_rows(self) -> mx.array:
        return self.model.last_streams

    def blank_draft_rows(self) -> mx.array:
        return mx.zeros((1, self.args.hc_mult, self.args.hidden_size), dtype=mx.bfloat16)

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        """A prompt chunk's rows into the head's cache: the chunk's streams and the tokens after them."""

        tokens = next_tokens if isinstance(next_tokens, mx.array) else mx.array(np.asarray(next_tokens).reshape(-1))
        tokens = tokens.reshape(-1).astype(mx.uint32)
        self._absorb(self.model.last_streams[: int(tokens.shape[0])], tokens, cache[-1])

    @property
    def prefill_workspace_per_token(self) -> int:
        """Prefill bytes a position of context: a query block's fp32 and bf16 indexer scores over the ratio-4 pool."""

        return PREFILL_QUERIES * self.args.index_n_heads * 6 // 4


class DSparkFlash(DeepSeekFlash):
    """DSpark drafts after a round is read: the kept rows' taps go into its rings, one pass drafts a block."""

    speculate_early = False

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
        return self.model.last_taps

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        """A prompt chunk's rows into the rings (only the last window's worth is read; the rest moves the offset)."""

        count = int(np.asarray(next_tokens).size) if not isinstance(next_tokens, mx.array) else int(next_tokens.size)
        rings = cache[self.layer_count:]
        skip = max(0, count - self.dspark.window)
        for ring in rings:
            ring.offset += skip
        self.dspark.absorb(self.model.last_taps[skip:count], rings)

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
            self.dspark.absorb(mx.take(self._rows, mx.array([int(r) for r in kept], dtype=mx.int32), axis=0), rings)
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


def materialize(root: Any) -> int:
    """Evaluate every array the runtime can reach, here: a lazy one first read on the server's thread fails there."""

    found: list[mx.array] = []
    seen: set[int] = set()
    todo = [root]
    while todo:
        obj = todo.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, mx.array):
            found.append(obj)
        elif isinstance(obj, (list, tuple)):
            todo.extend(obj)
        elif isinstance(obj, dict):
            todo.extend(obj.values())
        elif type(obj).__module__.startswith("tensorfold") and hasattr(obj, "__dict__"):
            todo.extend(vars(obj).values())
    for i in range(0, len(found), 256):
        mx.eval(*found[i:i + 256])
    return len(found)


def load(model_dir: Path, *, drafter: str = "", mtp_drafts: int | None = None, check: bool = True,
         **_: Any) -> tuple[DeepSeekFlash, Any]:
    """The runtime and tokenizer; ``drafter`` holds DSpark (preferred) or the MTP layer (default 3 drafts, 0: none)."""

    from mlx_lm.utils import load_tokenizer

    from tensorfold.families.deepseek_v4 import dspark as dspark_module
    from tensorfold.families.deepseek_v4 import mtp as mtp_module
    from tensorfold.families.deepseek_v4.prompts import DeepSeekTokenizer
    from tensorfold.families.deepseek_v4.weights import load_backbone

    model = load_backbone(Path(model_dir))
    tokenizer = DeepSeekTokenizer(load_tokenizer(Path(model_dir), eos_token_ids=model.args.eos_token_id or None))
    drafts = 3 if mtp_drafts is None else int(mtp_drafts)
    folder = Path(drafter) if drafter else None
    if drafts > 0 and folder is not None and (folder / DSPARK_FILE).is_file():
        config = json.loads((folder / "config.json").read_text())
        runtime: DeepSeekFlash = DSparkFlash(model, dspark_module.load(model, folder / DSPARK_FILE, config),
                                             check=check)
        kind = "DSpark"
    else:
        source = folder / MTP_FILE if folder is not None else None
        head = mtp_module.load(model, source) if drafts > 0 and source is not None and source.is_file() else None
        if drafts > 0 and head is None:
            print(f"[deepseek_v4] no draft head ({DSPARK_FILE} or {MTP_FILE} in --drafter): decoding without drafts",
                  flush=True)
        runtime, kind = DeepSeekFlash(model, head, drafts=drafts, check=check), "MTP"
    materialize(runtime)
    print(f"[deepseek_v4] exact window {runtime.exact_width} rows, forward ms by width {runtime.window_costs}, "
          f"{kind} step {runtime.mtp_step_ms} ms, drafts up to {runtime.drafts if runtime.mtp else 0}", flush=True)
    return runtime, tokenizer
