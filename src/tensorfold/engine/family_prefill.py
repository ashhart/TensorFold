"""The family rounds' prefill: prompts on the engine's grid, conversation checkpoints on it, the first token and
the draft head's first drafts."""

from __future__ import annotations

import time
from typing import Any, Sequence

from tensorfold.engine.family_common import cache_arrays, drop_spares


class FamilyPrefill:
    """Prefill for ``FamilyRounds``."""

    def _family_feed(self, tokens: Sequence[int], cache: list[Any], following: int | None = None) -> Any:
        """Absorb ``tokens``; the last position's hidden state [1, 1, D]. The draft head's cache takes every
        position whose next token is known: the next prompt token, and ``following`` after the last one."""

        import mlx.core as mx

        last = None
        # ``tokens`` start on the grid: every chunk is a fresh prefill's chunk
        step = self._align() or max(1, int(self.prefill_step))
        feed = getattr(self.model, "prefill", None) or self.model.hidden
        self._fed_rows = 0
        for begin in range(0, len(tokens), step):
            chunk = [int(t) for t in tokens[begin:begin + step]]
            if self.prefill_guard is not None:
                self.prefill_guard.before_chunk(cache, len(chunk))
            hidden = feed(mx.array([chunk], dtype=mx.uint32), cache)
            self._fed_rows = len(chunk)
            self.prefill_chunks += 1
            last = hidden[:, -1:, :]
            if getattr(self.model, "mtp", None) is not None:
                nxt = [int(t) for t in tokens[begin + 1:begin + step + 1]]
                if len(nxt) < len(chunk) and following is not None:
                    nxt.append(int(following))
                if nxt:
                    self.model.absorb_draft_context(hidden[:, :len(nxt)], mx.array(nxt, dtype=mx.uint32), cache,
                                                    start=0)
            mx.eval(last, *cache_arrays(cache))
            if self.prefill_guard is not None:
                self.prefill_guard.after_chunk(cache, len(chunk))
        return last

    def _family_start(self, cache: list[Any] | None, cached_tokens: int, length: int) -> tuple[list[Any], int]:
        """The working cache and the position its prefill starts at: a stored state off the grid is not used
        (it cannot resume exactly), so the prompt is prefilled from 0."""

        align = self._align()
        if cache is None or (align and int(cached_tokens) % align):
            return self.model.make_cache(), 0
        start = int(cached_tokens)
        if not 0 <= start < length:
            raise ValueError("cached_tokens must leave a suffix to prefill")
        adopt = getattr(self.model, "adopt_cache", None)
        return (adopt(cache) if adopt is not None else cache), start

    def _family_prefill(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                        checkpoints_at: Sequence[int]) -> list[Any]:
        if not stream.prompt_ids:
            raise ValueError(f"{stream.stream_id}: empty prompt")
        work, start = self._family_start(cache, cached_tokens, len(stream.prompt_ids))
        cached_tokens = start
        stream.history_checkpoints = []
        for boundary in sorted({self.on_grid(int(b)) for b in checkpoints_at}):
            if not start < boundary < len(stream.prompt_ids):
                continue
            self._family_feed(stream.prompt_ids[start:boundary], work, following=stream.prompt_ids[boundary])
            if self.prefill_guard is None or self.prefill_guard.allow_checkpoint(work):
                stream.history_checkpoints.append((list(stream.prompt_ids[:boundary]),
                                                   drop_spares(self.copy_single_cache(work))))
            start = boundary
        hidden = self._family_feed(stream.prompt_ids[start:], work)
        first = self._family_first(stream, work, hidden, cached_tokens, self._fed_rows - 1)
        self._family_commit_first(stream, int(first.item()) if hasattr(first, "item") else int(first))
        return work

    def _family_first(self, stream: Any, work: list[Any], hidden: Any, cached_tokens: int, row: int) -> Any:
        """Draw or force the first token before the draft head or the next forward reads it."""

        import mlx.core as mx

        prompt_len = len(stream.prompt_ids)
        stream.emitted = []
        stream.pending = []
        stream.cache_len = prompt_len
        stream.cached_tokens = int(cached_tokens)
        stream.started_at = time.perf_counter()
        token = self._draw(self.model.head(hidden), stream.sampling, [prompt_len])
        forced = self._forced_next(stream)
        if forced is not None:
            token = mx.array([forced], dtype=mx.uint32)
        if self.family_mtp and stream.drafts:
            # the head reads the prompt's last position and the first token, and drafts the one after it
            firsts = self.model.speculate(work, token, prompt_len - 1, stream.sampling, start=row)
            self._next[stream.stream_id] = self.model.settle(work, 1, firsts.reshape(-1)[:1], prompt_len + 1,
                                                             stream.sampling, self._depth(stream))
        elif self.pipelined:
            self._queue_next(stream, work, token)
        return token

    @staticmethod
    def _family_commit_first(stream: Any, first: int) -> None:
        stream.commit([first])
        stream.pending = [first]

    def _queue_next(self, stream: Any, cache: list[Any], token: Any) -> None:
        """Feed ``token`` (a GPU array, not read yet) and queue the draw of the one after it."""

        import mlx.core as mx

        hidden = self.model.hidden(token.reshape(1, 1), cache)
        stream.cache_len += 1
        nxt = self._draw(self.model.head(hidden), stream.sampling, [stream.cache_len])
        mx.async_eval(nxt)
        self._inflight[stream.stream_id] = nxt

    def _family_prefill_prefix(self, prompt_ids: Sequence[int], *, cache: list[Any] | None,
                               cached_tokens: int) -> list[Any]:
        if not prompt_ids:
            raise ValueError("empty prefix")
        work, start = self._family_start(cache, cached_tokens, len(prompt_ids))
        self._family_feed(list(prompt_ids[start:]), work)
        return drop_spares(work)

    def _family_add_stream(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                           checkpoints_at: Sequence[int]) -> None:
        work = self._family_prefill(stream, cache=cache, cached_tokens=cached_tokens, checkpoints_at=checkpoints_at)
        self.streams.append(stream)
        if stream.finished:
            stream.finished_at = time.perf_counter()
            self._release_stream_state(stream.stream_id)
            return
        self._live.append((stream, work))
