"""Qwen3.6 MoE on the Mac row decoder, drafting chains with the checkpoint's own MTP layer in the lane rounds."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence

from tensorfold.families.qwen3_5.family import Qwen35Family

DRAFTS = 4               # most MTP drafts a round; the depth rule picks fewer, or none, from measured costs


class Qwen36Family(Qwen35Family):
    """The dense family's verify rounds; drafts come from the MTP layer reading each kept row's final normed state."""

    speculate_early = False      # the kept rows are absorbed after the round is read
    draft_reads_hidden = True    # every prompt chunk's rows feed the MTP layer
    plain_guard = True           # the depth rule may pick plain rounds where drafts cost more than they land
    # the head's acceptance at depth 1, 2, ... given the ones before it, until a stream has its own
    draft_prior = (0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)

    def __init__(self, model: Any, *, drafter: Any = None, mtp_path: Path | None = None, drafts: int = DRAFTS,
                 **options: Any) -> None:
        if drafter is not None:
            raise ValueError("Qwen3.6's MTP family drafts with its own layer, not a draft model")
        super().__init__(model, drafter=None, **options)
        self._last_hidden: Any = None
        self.draft_probabilities = None          # chains are sized by DraftDepth, not by per-node chances
        self.mtp: Any = None
        self.drafts = 0
        self.mtp_step_ms = 0.0
        self._draft_ids = self._draft_head = None
        if mtp_path is None or int(drafts) <= 0 or self.exact_width < 2:
            return
        from tensorfold.families.qwen3_5_moe import mtp

        args = getattr(model, "language_model", model).args
        self._width = int(args.hidden_size)
        try:
            self.mtp = mtp.load(Path(mtp_path), args)
        except ValueError as exc:                 # another model's layer: its shapes or names differ
            raise SystemExit(f"[tensorfold] {mtp_path} is not this checkpoint's MTP layer: {exc}") from None
        self.drafts = int(drafts)
        if os.environ.get("TF_QWEN36_DRAFT_VOCAB", "1") != "0":      # 0 scores the full vocabulary
            import mlx.core as mx

            from tensorfold.families.qwen4_exp.draft_head import cut_head, draft_ids

            ids = draft_ids()
            if int(ids.max()) < int(self.lm_head["weight"].shape[0]):    # the Qwen 3.5-3.8 tokenizer's ids
                self._draft_ids, self._draft_head = mx.array(ids), cut_head(self.lm_head, ids)
        self.mtp_step_ms = self._time_mtp_step()
        vocab = "every token" if self._draft_ids is None else f"{int(self._draft_ids.shape[0]):,} draft ids"
        print(f"[tensorfold] MTP drafts from {Path(mtp_path).name}: up to {self.drafts} a round over {vocab}, a "
              f"chained draft {self.mtp_step_ms:.2f} ms; measured round costs pick each round's depth, plain included",
              flush=True)

    # -- caches -----------------------------------------------------------------------------------------------------
    @staticmethod
    def _layers(cache: list[Any]) -> list[Any]:
        from tensorfold.families.qwen3_5_moe.mtp import MTPCache

        return cache[:-1] if cache and isinstance(cache[-1], MTPCache) else cache

    def make_cache(self) -> list[Any]:
        caches = list(self.inner.make_cache())
        if self.mtp is not None:
            caches.append(self.mtp.make_cache())
        return caches

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        """A stored prefix with this model's head cache: a missing one starts empty (drafts see less context)."""

        from tensorfold.families.qwen3_5_moe.mtp import MTPCache

        held = bool(cache) and isinstance(cache[-1], MTPCache)
        if self.mtp is not None and not held:
            cache.append(self.mtp.make_cache())
        elif self.mtp is None and held:
            cache.pop()
        return cache

    def release_rounds(self) -> None:
        self._last_hidden = None

    # -- the target's forwards, their final normed rows kept for the head -------------------------------------------
    def hidden(self, inputs: Any, cache: list[Any], parents: Sequence[int] | None = None) -> Any:
        self._last_hidden = super().hidden(inputs, cache, parents)
        return self._last_hidden

    def prefill(self, inputs: Any, cache: list[Any]) -> Any:
        self._last_hidden = super().prefill(inputs, cache)
        return self._last_hidden

    def hidden_rows(self, windows: Sequence[Any], caches: Sequence[list[Any]],
                    parents: Sequence[Sequence[int]] | None = None) -> Any:
        self._last_hidden = super().hidden_rows(windows, caches, parents)
        return self._last_hidden

    # -- the MTP head -------------------------------------------------------------------------------------------------
    def _embed(self, tokens: Any) -> Any:
        return self.core.embed_tokens(tokens.reshape(1, -1))

    def _draw(self, state: Any, samplings: Sequence[Any], positions: Sequence[int]) -> Any:
        """Drafts [S] (uint32, lazy) from head states [1, S, D] with the target's keyed rule, at their positions."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample_rows

        head = self._draft_head if self._draft_head is not None else self.lm_head
        return sample_rows(head(state), list(samplings), [int(p) for p in positions],
                           ids=self._draft_ids).astype(mx.uint32)

    def _absorb(self, mcache: Any, hidden: Any, tokens: Any, first: int, outputs: str) -> Any:
        """Write rows (row t at ``first`` + i, token t + 1) into the head's buffers; outputs "all", "last" or "none"."""

        import mlx.core as mx

        from tensorfold.families.qwen3_5_moe.mtp import absorb_attention

        rows = int(hidden.shape[1])
        x = self.mtp.inputs(hidden, self._embed(tokens))
        queries, keys, values, gate = self.mtp.qkv(x, mx.arange(first, first + rows, dtype=mx.int32))
        if outputs == "none":
            mcache.side = None
            mcache.update_and_fetch(keys, values)
            return None
        if outputs == "last":
            queries, x, gate = queries[:, :, -1:], x[:, -1:], gate[:, -1:]
        attended = absorb_attention(self.mtp.layers[0].self_attn, mcache, queries, keys, values)
        return self.mtp.finish(x, attended, gate)

    def _chain(self, mcaches: Sequence[Any], states: Any, tokens: Any, positions: Sequence[int]) -> Any:
        """A chained head row a stream (state [1, S, D], draft at ``positions`` + 1), kept apart: outputs [1, S, D]."""

        import mlx.core as mx

        from tensorfold.families.qwen3_5_moe.mtp import chain_attention

        attn = self.mtp.layers[0].self_attn
        x = self.mtp.inputs(states, self._embed(tokens))
        queries, keys, values, gate = self.mtp.qkv(x, mx.array([int(p) for p in positions], dtype=mx.int32))
        attended = [chain_attention(attn, c, queries[:, :, i:i + 1], keys[:, :, i:i + 1], values[:, :, i:i + 1])
                    for i, c in enumerate(mcaches)]
        return self.mtp.finish(x, attended[0] if len(attended) == 1 else mx.concatenate(attended, axis=1), gate)

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        """Prompt rows into the head's attention context, drafting nothing from them."""

        self._absorb(cache[-1], hidden, next_tokens, self._last[id(cache)][2] + int(start), "none")

    def absorb_kept(self, cache: list[Any], tokens: Any, position: int, start: int = 0,
                    rows: Sequence[int] | None = None) -> None:
        """A plain round's kept rows (from ``start``, at ``position`` on) into the head's context, drafting nothing."""

        import mlx.core as mx

        if rows is not None:
            start = int(rows[0])
        tokens = tokens.reshape(-1)
        count = int(tokens.shape[0])
        self._absorb(cache[-1], self._last_hidden[:, start:start + count], tokens, int(position), "none")
        mx.async_eval(*_buffers(cache[-1]))

    def speculate(self, cache: list[Any], tokens: Any, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False, rows: Sequence[int] | None = None) -> Any:
        """Absorb rows from ``start`` (at ``position`` on) with their next tokens; lazy first drafts after them."""

        if rows is not None:
            start = int(rows[0])
            if list(rows) != list(range(start, start + len(rows))):
                raise NotImplementedError("Qwen3.6's head reads consecutive rows")
        mcache = cache[-1]
        tokens = tokens.reshape(-1)
        count = int(tokens.shape[0])
        out = self._absorb(mcache, self._last_hidden[:, start:start + count], tokens, int(position),
                           "last" if last_only else "all")
        mcache.speculation = (out, count, last_only)
        drafted = [position + 1 + count] if last_only else [position + 2 + i for i in range(count)]
        return self._draw(out, [sampling] * len(drafted), drafted)

    def settle(self, cache: list[Any], keep: int, first: Any, position: int, sampling: Any, count: int) -> Any:
        """Keep ``keep`` speculated rows; ``first`` and the chain after it at ``position`` on (lazy uint32), or []."""

        import mlx.core as mx

        mcache = cache[-1]
        out, rows, last_only = mcache.speculation
        mcache.speculation = None
        if rows > keep:
            mcache.trim(rows - keep)
        if count <= 0:
            mx.async_eval(*_buffers(mcache))      # the absorbed keys alone: a plain round builds no lazy history
            return []
        drafts = [first.reshape(1).astype(mx.uint32) if isinstance(first, mx.array)
                  else mx.array([int(first)], dtype=mx.uint32)]
        mx.async_eval(drafts[0])                   # each level queued as built: the GPU works while the next is built
        state = out[:, -1:] if last_only else out[:, keep - 1:keep]
        for j in range(1, count):
            state = self._chain([mcache], state, drafts[-1], [position + j - 2])
            drafts.append(self._draw(state, [sampling], [position + j]))
            mx.async_eval(drafts[-1])
        return mx.concatenate(drafts) if len(drafts) > 1 else drafts[0]

    def unspeculate(self, cache: list[Any]) -> None:
        mcache = cache[-1]
        if mcache.speculation is not None:
            mcache.trim(mcache.speculation[1])
            mcache.speculation = None

    def draft_streams(self, caches: Sequence[list[Any]], follows: Sequence[Sequence[int]],
                      rows: Sequence[Sequence[int]], positions: Sequence[int], samplings: Sequence[Any],
                      depths: Sequence[int]) -> list[Any]:
        """Every stream's kept rows absorbed in one head pass, then the chains level by level: lazy uint32 drafts."""

        import mlx.core as mx

        from tensorfold.families.qwen3_5_moe.mtp import absorb_attention

        mcaches = [c[-1] for c in caches]
        lengths = [len(f) for f in follows]
        for r in rows:
            if list(r) != list(range(int(r[0]), int(r[0]) + len(r))):
                raise NotImplementedError("Qwen3.6's head reads consecutive rows")
        hidden = mx.concatenate([self._last_hidden[:, int(r[0]):int(r[0]) + n] for r, n in zip(rows, lengths)], axis=1)
        x = self.mtp.inputs(hidden, self._embed(mx.array([int(t) for f in follows for t in f], dtype=mx.uint32)))
        firsts = [int(p) - 1 - n for p, n in zip(positions, lengths)]       # each stream's first kept row's position
        at = mx.array([p + i for p, n in zip(firsts, lengths) for i in range(n)], dtype=mx.int32)
        queries, keys, values, gate = self.mtp.qkv(x, at)
        attn, ends, attended, begin = self.mtp.layers[0].self_attn, [], [], 0
        for mcache, n in zip(mcaches, lengths):
            ends.append(begin + n - 1)
            attended.append(absorb_attention(attn, mcache, queries[:, :, begin + n - 1:begin + n],
                                             keys[:, :, begin:begin + n], values[:, :, begin:begin + n]))
            begin += n
        last = mx.array(ends, dtype=mx.int32)
        state = self.mtp.finish(x[:, last], mx.concatenate(attended, axis=1), gate[:, last])
        found: dict[int, list[Any]] = {}
        active = [i for i, d in enumerate(depths) if d > 0]
        level = None
        for j in range(max(depths, default=0)):
            if j:
                still = [k for k, i in enumerate(active) if depths[i] > j]
                if not still:
                    break
                pick = mx.array(still, dtype=mx.int32)
                active = [active[k] for k in still]
                state = self._chain([mcaches[i] for i in active], state[:, pick], level[pick],
                                    [positions[i] + j - 2 for i in active])
            elif len(active) < len(follows):
                state = state[:, mx.array(active, dtype=mx.int32)]
            level = self._draw(state, [samplings[i] for i in active], [positions[i] + j for i in active])
            mx.async_eval(level)
            for k, i in enumerate(active):
                found.setdefault(i, []).append(level[k:k + 1])
        result = [mx.concatenate(found[i]) if i in found else [] for i in range(len(follows))]
        mx.async_eval(*[r for r in result if isinstance(r, mx.array)],
                      *[a for c, d in zip(mcaches, depths) if d <= 0 for a in _buffers(c)])
        return result

    def _time_mtp_step(self) -> float:
        """The shortest measured chained draft step (head row, draft head, sampling) in ms."""

        import mlx.core as mx

        cache = self.mtp.make_cache()
        hidden = mx.zeros((1, 8, self._width), dtype=mx.bfloat16)
        mx.eval(self._absorb(cache, hidden, mx.arange(1000, 1008, dtype=mx.uint32), 0, "last"))
        state = mx.zeros((1, 1, self._width), dtype=mx.bfloat16)
        token = mx.array([1000], dtype=mx.uint32)
        best = float("inf")
        for step in range(6):
            started = time.perf_counter()
            out = self._chain([cache], state, token, [8 + step])
            mx.eval(self._draw(out, [None], [10 + step]))
            best = min(best, (time.perf_counter() - started) * 1e3)
        return round(best, 3)


def _buffers(mcache: Any) -> list[Any]:
    return [a for a in (mcache.keys, mcache.values) if a is not None]


__all__ = ["DRAFTS", "Qwen36Family"]
