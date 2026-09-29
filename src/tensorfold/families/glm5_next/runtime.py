"""GLM-5.3-Flash as the lane engine's family rounds drive it: backbone, MTP draft head, load-time checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.glm5_next.caches import MLACache
from tensorfold.families.glm5_next.config import DECODE_ROWS
from tensorfold.families.glm5_next.mla import PREFILL_QUERIES
from tensorfold.families.glm5_next.model import GLM5


class MTPCache(MLACache):
    """The MTP head's cache; ``drafted``: its last entries that are chained drafts, trimmed before the next absorb."""

    drafted = 0


class GLMFlash:
    """GLM-5.3-Flash with the backbone and head apart, the row-exact decode path, and MTP drafting."""

    lane_family = True
    # the decode path's widest window (``model.DECODE_ROWS``: wider calls take the prefill path)
    fused_rows = DECODE_ROWS
    # ``hidden`` takes an unread GPU token, so one-token rounds run one step ahead
    gpu_tokens = True
    # the head drafts from every verify row before the round is read: a host round trip less than drafting after it
    speculate_early = True
    # streams in one shared forward (each takes a row at least); its rows are ``batch_rows``, the widest exact window
    max_streams = DECODE_ROWS
    # False when a forward over several streams' rows misses a stream's own bits: the engine then shares no round
    streams_exact = True
    tag = "glm5"                 # log prefix

    def __init__(self, model: GLM5, head: Any | None = None, *, drafts: int = 1, check: bool = True) -> None:
        self.model = model
        self.args = model.args
        self.layer_count = len(model.layers)
        self.mtp = None
        self.drafts = int(drafts)
        self.check_report: dict[int, bool] = {}
        self.exact_width, self.window_costs = self.check_windows() if check else (1, {})
        self.multi_row_exact = self.exact_width >= 2
        self.batch_rows = self.exact_width
        if check and not self.multi_row_exact:
            print(f"[{self.tag}] a multi-row forward does not reproduce serial steps on this MLX/GPU "
                  f"({self.check_report}): no drafts", flush=True)
        elif check and not self.check_streams():
            self.max_streams = 1
            self.streams_exact = False
            print(f"[{self.tag}] a forward over several streams' rows does not reproduce each stream's own call here: "
                  "one stream a forward", flush=True)
        self._rows: mx.array | None = None
        self._specs: dict[int, tuple[mx.array, int]] = {}     # head cache id -> (speculate's output rows, rows)
        self.mtp_step_ms = 0.0
        if self.multi_row_exact and head is not None and self.drafts > 0:
            self.mtp = head
            self.mtp_step_ms = self._time_mtp_step()

    # -- the engine's model interface ---------------------------------------------------
    @property
    def layers(self) -> list[Any]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        caches = self.model.make_cache()
        if self.mtp is not None:
            caches.append(self.new_mtp_cache())                # last: the model's layers never reach it
        return caches

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        """A stored or copied cache without an MTP entry gets an empty one (drafts then see less context)."""

        if self.mtp is not None and len(cache) == self.layer_count:
            cache.append(self.new_mtp_cache())
        return cache

    def new_mtp_cache(self) -> Any:
        return MTPCache()

    def draft_rows(self) -> mx.array:
        """What the draft head reads of the last forward: its final-normed rows."""

        return self.model.last_normed

    def blank_draft_rows(self) -> mx.array:
        return mx.zeros((1, int(self.args.hidden_size)), dtype=mx.bfloat16)

    def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> mx.array:
        """Hidden states [1, R, D] of R tokens (maybe unread on the GPU): up to ``fused_rows`` decode, else prefill."""

        self._chain_only(parents)
        tokens = inputs if isinstance(inputs, mx.array) else mx.array(np.asarray(inputs, dtype=np.int64))
        out = self.model.hidden(tokens, cache[: self.layer_count])
        self._rows = self.draft_rows()
        return out

    def hidden_rows(self, windows: list[Any], caches: list[list[Any]], parents: Any = None) -> mx.array:
        """Every stream's window in one forward, rows stream by stream, each with its stream's own bits: [1, N, D]."""

        for rows in parents or ():
            self._chain_only(rows)
        parts = [w.reshape(-1).astype(mx.uint32) if isinstance(w, mx.array)
                 else mx.array([int(t) for t in w], dtype=mx.uint32) for w in windows]
        lengths = tuple(int(p.shape[0]) for p in parts)
        out = self.model.hidden_rows(mx.concatenate(parts) if len(parts) > 1 else parts[0],
                                     [c[: self.layer_count] for c in caches], lengths)
        self._rows = self.draft_rows()
        return out

    def head(self, hidden: mx.array) -> mx.array:
        return self.model.head(hidden)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))

    def keep_rows(self, cache: list[Any], rows: int, keep: Any) -> None:
        self.model.keep_rows(cache, rows, self._kept(keep))

    def keep_rows_streams(self, caches: list[list[Any]], lengths: Any, keeps: Any) -> None:
        """After ``hidden_rows``: stream i keeps the first ``keeps[i]`` of its ``lengths[i]`` rows."""

        self.model.keep_rows_streams(caches, lengths, [self._kept(k) for k in keeps])

    @staticmethod
    def _chain_only(parents: Any) -> None:
        """GLM's head drafts chains: a window whose rows are not a chain is a caller's error."""

        if parents is not None and list(parents) != list(range(-1, len(parents) - 1)):
            raise NotImplementedError("GLM-5.3-Flash verifies draft chains, not trees")

    @staticmethod
    def _kept(keep: Any) -> int:
        """A kept-row count from a count or a path from the root (a chain's path is a prefix)."""

        if isinstance(keep, int):
            return keep
        path = [int(r) for r in keep]
        if path != list(range(len(path))):
            raise NotImplementedError("GLM-5.3-Flash keeps a prefix of a window's rows")
        return len(path)

    @property
    def prefill_workspace_per_token(self) -> int:
        """Prefill bytes a position of context: two live copies of a query chunk's per-head indexer scores (bf16)."""

        a = self.args
        return PREFILL_QUERIES * (2 * a.index_n_heads * 2 + 10) // a.index_kpool

    def resolve_prefill_identity(self) -> None:
        """Build and check the sorted expert kernels at startup, not at the first prompt (the CLI calls this)."""

        from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

        if prefill_mm.fast_prefill():
            prefill_mm.tiles()

    # -- drafting ---------------------------------------------------------------------
    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        """Prompt rows into the head's cache: final-normed rows ``hidden`` [1, n, D] and the tokens after them."""

        tokens = next_tokens if isinstance(next_tokens, mx.array) else mx.array(np.asarray(next_tokens).reshape(-1))
        tokens = tokens.reshape(-1).astype(mx.uint32)
        self._absorb(hidden.reshape(-1, hidden.shape[-1])[: int(tokens.shape[0])], tokens, cache[-1])

    def _absorb(self, rows: mx.array, tokens: mx.array, mtp_cache: MTPCache) -> mx.array:
        """Rows (final-normed hidden [n, D], the tokens that follow them [n]) into the head; its output rows [n, D]."""

        self._trim_chained(mtp_cache)
        count = int(tokens.shape[0])
        return self.mtp(self.model, rows, tokens, [mtp_cache], (count,), count <= self.fused_rows)

    @staticmethod
    def _trim_chained(mtp_cache: MTPCache) -> None:
        if mtp_cache.drafted:
            mtp_cache.trim(mtp_cache.drafted)
            mtp_cache.drafted = 0

    def _draft_draw(self, out: mx.array, sampling: Any, positions: Any) -> mx.array:
        """Drafts (uint32 [n], lazy) from the head's output rows [n, D]: the target's keyed rule at ``positions``."""

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        return gpu_sample(self.mtp.logits(self.model, out), sampling, positions)

    def speculate(self, cache: list[Any], tokens: mx.array, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False, rows: Any = None) -> mx.array:
        """The head absorbs rows ``start`` .. of the last forward and draws each one's first draft before the read."""

        if rows is not None:
            start = int(rows[0])
            if list(rows) != list(range(start, start + len(rows))):
                raise NotImplementedError("GLM-5.3-Flash's head reads consecutive rows")
        mtp_cache = cache[-1]
        tokens = tokens.reshape(-1).astype(mx.uint32)
        count = int(tokens.shape[0])
        total = int(self._rows.shape[0])
        start = start + total if start < 0 else start
        out = self._absorb(self._rows[start:start + count], tokens, mtp_cache)
        self._specs[id(mtp_cache)] = (out, count)
        if last_only:                      # the last row's draft only (every row still enters the head's cache)
            return self._draft_draw(out[-1:], sampling, [position + 1 + count])
        return self._draft_draw(out, sampling, [position + 2 + r for r in range(count)])

    def settle(self, cache: list[Any], keep: int, first: Any, position: int, sampling: Any, count: int) -> Any:
        """Keep the head's first ``keep`` rows, then ``count`` drafts: ``first`` and chained ones, drawn on the GPU."""

        mtp_cache = cache[-1]
        out, rows = self._specs.pop(id(mtp_cache))
        if rows > keep:
            mtp_cache.trim(rows - keep)
        if count <= 0:
            return []
        lazy = isinstance(first, mx.array)
        head = first.reshape(1).astype(mx.uint32) if lazy else mx.array([int(first)], dtype=mx.uint32)
        if count == 1:
            return head if lazy else [int(first)]
        chain, last = [head], out[keep - 1:keep]
        for j in range(1, count):
            last = self.mtp(self.model, last, chain[-1], [mtp_cache], (1,), True)
            mtp_cache.drafted += 1
            chain.append(self._draft_draw(last, sampling, [position + j]))
        drafts = mx.concatenate(chain)
        mx.async_eval(drafts)
        return drafts

    def unspeculate(self, cache: list[Any]) -> None:
        """Undo ``speculate`` entirely (the round's rows are absorbed another way)."""

        spec = self._specs.pop(id(cache[-1]), None)
        if spec is not None:
            cache[-1].trim(spec[1])

    def draft_streams(self, caches: list[list[Any]], follows: list[list[int]], rows: list[list[int]],
                      positions: list[int], samplings: list[Any], depths: list[int]) -> list[Any]:
        """Each stream's head after a shared round, one forward a depth: absorb its kept rows, draft ``depths[i]``."""

        from tensorfold.engine.gpu_sampling import sample_rows

        mtp = [c[-1] for c in caches]
        for m in mtp:
            self._trim_chained(m)
        index = mx.array([int(r) for kept in rows for r in kept], dtype=mx.int32)
        tokens = mx.array([int(t) for f in follows for t in f], dtype=mx.uint32)
        lengths = tuple(len(f) for f in follows)
        out = self.mtp(self.model, mx.take(self._rows, index, axis=0), tokens, mtp, lengths,
                       sum(lengths) <= self.fused_rows)
        chains: list[list[mx.array]] = [[] for _ in rows]
        live = [i for i, depth in enumerate(depths) if depth > 0]
        if live:
            lasts = mx.array([sum(lengths[:i + 1]) - 1 for i in live], dtype=mx.int32)
            state = mx.take(out, lasts, axis=0)
            level = sample_rows(self.mtp.logits(self.model, state), [samplings[i] for i in live],
                                [positions[i] for i in live])
            for k, i in enumerate(live):
                chains[i].append(level[k:k + 1])
        step = 1
        while live:
            going = [k for k, i in enumerate(live) if depths[i] > step]
            if not going:
                break
            pick = mx.array(going, dtype=mx.int32)
            live = [live[k] for k in going]
            state = self.mtp(self.model, mx.take(state, pick, axis=0), mx.take(level, pick, axis=0),
                             [mtp[i] for i in live], (1,) * len(live), True)
            for i in live:
                mtp[i].drafted += 1
            level = sample_rows(self.mtp.logits(self.model, state), [samplings[i] for i in live],
                                [positions[i] + step for i in live])
            for k, i in enumerate(live):
                chains[i].append(level[k:k + 1])
            step += 1
        drafts = [mx.concatenate(chain) if chain else [] for chain in chains]
        mx.async_eval(*[d for d in drafts if isinstance(d, mx.array)])
        return drafts

    def _time_mtp_step(self) -> float:
        """One chained draft step as ``settle`` takes it, in ms (fastest of 6): the depth rule's cost a draft."""

        import time

        vocab = int(self.args.vocab_size)
        cache = self.new_mtp_cache()
        rows = self.blank_draft_rows()
        out = self.mtp(self.model, rows, mx.array([3001 % vocab], dtype=mx.uint32), [cache], (1,), True)
        mx.eval(out)
        best = float("inf")
        for i in range(6):
            started = time.perf_counter()
            out = self.mtp(self.model, out, mx.array([(3002 + i) % vocab], dtype=mx.uint32), [cache], (1,), True)
            self._draft_draw(out, None, [100 + i]).item()
            best = min(best, (time.perf_counter() - started) * 1e3)
        return round(best, 3)

    # -- load-time check ------------------------------------------------------------------
    def check_windows(self, widest: int | None = None) -> tuple[int, dict[int, float]]:
        """The widest window whose every row gets one-row logits bit for bit, and each exact width's forward ms."""

        import time

        from tensorfold.engine.lane_engine import LaneEngine

        copy = LaneEngine.copy_single_cache
        widest = int(widest or self.fused_rows)
        vocab = int(self.args.vocab_size)
        prompt = mx.array([[((37 * i + 11) % 50_000 + 1000) % vocab for i in range(48)]], dtype=mx.uint32)
        window = [(3001 + 17 * r) % vocab for r in range(widest)]
        base = self.model.make_cache()
        mx.eval(self.model.hidden(prompt, base))
        one = copy(base)
        serial = []
        for token in window:
            logits = self.model.head(self.model.hidden(mx.array([[token]], dtype=mx.uint32), one))
            mx.eval(logits)
            serial.append(logits[0, -1])
        exact = 1
        self.check_report = {}
        for width in range(2, widest + 1):
            logits = self.model.head(self.model.hidden(mx.array([window[:width]], dtype=mx.uint32), copy(base)))
            mx.eval(logits)
            same = all(bool(mx.array_equal(logits[0, i], serial[i]).item()) for i in range(width))
            self.check_report[width] = same
            if not same:
                break
            exact = width
        costs: dict[int, float] = {}
        for width in range(1, exact + 1):
            best = float("inf")
            for _ in range(3):
                cache = copy(base)
                started = time.perf_counter()
                mx.eval(self.model.head(self.model.hidden(mx.array([window[:width]], dtype=mx.uint32), cache)))
                best = min(best, (time.perf_counter() - started) * 1e3)
            costs[width] = round(best, 3)
        return exact, costs

    def check_streams(self) -> bool:
        """Streams of different lengths in one forward against their own calls, bit for bit, before and after a keep."""

        from tensorfold.engine.lane_engine import LaneEngine

        copy = LaneEngine.copy_single_cache
        lengths, keeps = ([3, 1, 4], [2, 1, 1]) if self.exact_width >= 8 else ([2, 1], [1, 1])
        vocab = int(self.args.vocab_size)
        base = self.model.make_cache()
        prompt = [((37 * i + 11) % 50_000 + 1000) % vocab for i in range(48)]
        mx.eval(self.model.hidden(mx.array([prompt], dtype=mx.uint32), base))

        def streams() -> list[list[Any]]:
            out = []
            for b in range(len(lengths)):
                cache = copy(base)
                for extra in range(b):                      # different lengths: b more tokens each
                    mx.eval(self.model.hidden(mx.array([[(5001 + 13 * b + extra) % vocab]], dtype=mx.uint32), cache))
                out.append(cache)
            return out

        windows = [[(3001 + 17 * (3 * b + r)) % vocab for r in range(n)] for b, n in enumerate(lengths)]
        follows = [[(7001 + 11 * b) % vocab] for b in range(len(lengths))]
        alone, together = streams(), streams()
        for step, wins in enumerate((windows, follows)):
            single = []
            for b, cache in enumerate(alone):
                logits = self.model.head(self.model.hidden(mx.array([wins[b]], dtype=mx.uint32), cache))
                mx.eval(logits)
                single.append(logits[0])
                if step == 0 and keeps[b] < len(wins[b]):
                    self.model.keep_rows(cache, len(wins[b]), keeps[b])
            flat = mx.array([t for w in wins for t in w], dtype=mx.uint32)
            joint = self.model.head(self.model.hidden_rows(flat, together, [len(w) for w in wins]))[0]
            mx.eval(joint)
            at = 0
            for b, w in enumerate(wins):
                if not bool(mx.array_equal(joint[at:at + len(w)], single[b]).item()):
                    return False
                at += len(w)
            if step == 0:
                self.model.keep_rows_streams(together, [len(w) for w in wins], keeps)
        return True


def load(model_dir: Path, *, drafts: int | None = None, check: bool = True,
         ssd_experts: float | None = None) -> tuple[GLMFlash, Any]:
    """The runtime and tokenizer; ``drafts`` (default 3, 0: none) caps the MTP drafts a round."""

    from tensorfold.families.glm5_next import has_mtp
    from tensorfold.families.glm5_next import mtp as mtp_module
    from tensorfold.families.glm5_next import weights as glm

    model, tokenizer = glm.load(Path(model_dir), ssd_experts=ssd_experts)
    drafts = 3 if drafts is None else int(drafts)
    head = mtp_module.load(model) if drafts > 0 and has_mtp(model_dir) else None
    model.weights = None                                                 # the checkpoint's shard index is done
    runtime = GLMFlash(model, head, drafts=drafts, check=check)
    print(f"[glm5] exact window {runtime.exact_width} rows, forward ms by width {runtime.window_costs}, "
          f"MTP step {runtime.mtp_step_ms} ms, drafts up to {runtime.drafts if runtime.mtp else 0}", flush=True)
    return runtime, tokenizer
