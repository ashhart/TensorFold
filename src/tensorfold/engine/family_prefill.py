"""The family rounds' prefill: prompt chunks, checkpoints at chunk starts, the first token and first drafts."""

from __future__ import annotations

import time
from typing import Any, Sequence

from tensorfold.engine.family_common import cache_arrays, drop_spares


class FamilyPrefill:
    """Prefill for ``FamilyRounds``."""

    _prefill_at: int | None = None                     # the prompt position the working cache holds whole

    def _family_feed(self, tokens: Sequence[int], cache: list[Any], chunks: Sequence[tuple[int, int]]) -> Any:
        """Absorb the ``chunks`` ([begin, end) of ``tokens``); the last hidden state [1, 1, D], draft head fed too."""

        import mlx.core as mx

        last = None
        feed = getattr(self.model, "prefill", None) or self.model.hidden
        self._fed_rows = 0
        chunks = list(chunks)
        ahead = getattr(self.model, "prefetch_prompt", None)
        if ahead is not None and chunks:
            ahead(tokens, *chunks[0])
        for n, (begin, end) in enumerate(chunks):
            if ahead is not None and n + 1 < len(chunks):
                ahead(tokens, *chunks[n + 1])                 # its host reads run while this chunk computes
            chunk = [int(t) for t in tokens[begin:end]]
            if self.prefill_guard is not None:
                self.prefill_guard.before_chunk(cache, len(chunk))
            self._prefill_at = None                    # a chunk in flight: the cache holds no prompt prefix whole
            hidden = feed(mx.array([chunk], dtype=mx.uint32), cache)
            self._fed_rows = len(chunk)
            self.prefill_chunks += 1
            last = hidden[:, -1:, :]
            drafting = getattr(self.model, "mtp", None) is not None
            if drafting:
                nxt = [int(t) for t in tokens[begin + 1:end + 1]]
                if nxt:
                    self.model.absorb_draft_context(hidden[:, :len(nxt)], mx.array(nxt, dtype=mx.uint32), cache,
                                                    start=0)
            # an earlier chunk is read only through its caches: MLX then skips its last layer's attention and MLP
            mx.eval(*((last,) if drafting or n + 1 == len(chunks) else ()), *cache_arrays(cache))
            self._prefill_at = end
            if self.prefill_guard is not None:
                self.prefill_guard.after_chunk(cache, len(chunk))
        return last

    def _family_start(self, cache: list[Any] | None, cached_tokens: int, chunks: Any) -> tuple[list[Any], int]:
        """The working cache and where its prefill starts: a stored state only at one of the prompt's chunk starts."""

        if cache is None or int(cached_tokens) not in chunks:
            return self.model.make_cache(), 0
        adopt = getattr(self.model, "adopt_cache", None)
        return (adopt(cache) if adopt is not None else cache), int(cached_tokens)

    def _family_prefill(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                        checkpoints_at: Sequence[int]) -> list[Any]:
        prompt = stream.prompt_ids
        if not prompt:
            raise ValueError(f"{stream.stream_id}: empty prompt")
        if stream.multimodal is not None:
            if cache is not None or cached_tokens or checkpoints_at:
                raise ValueError("image requests cannot resume token-only prompt caches")
            work = self.model.make_vision_cache()
            if self.prefill_guard is not None:
                self.prefill_guard.before_chunk(work, len(prompt))
            self._prefill_at = None
            hidden = self.model.vision_prefill(stream.multimodal, work)
            self._fed_rows = len(prompt)
            self.prefill_chunks += 1
            self._prefill_at = len(prompt)
            first = self._family_first(stream, work, hidden[:, -1:, :], 0, len(prompt) - 1)
            self._family_commit_first(
                stream, int(first.item()) if hasattr(first, "item") else int(first)
            )
            if self.prefill_guard is not None:
                self.prefill_guard.after_chunk(work, len(prompt))
            discard = getattr(stream.multimodal, "discard_buffers", None)
            if callable(discard):
                discard()
            return work
        chunks = self.prompt_chunks(prompt)
        work, start = self._family_start(cache, cached_tokens, chunks)
        cached_tokens = self._prefill_at = start
        stream.history_checkpoints = []
        try:
            for boundary in sorted({chunks.floor(int(b)) for b in checkpoints_at}):
                if not start < boundary < len(prompt):
                    continue
                self._family_feed(prompt, work, chunks.between(start, boundary))
                if self.prefill_guard is None or self.prefill_guard.allow_checkpoint(work):
                    stream.history_checkpoints.append((list(prompt[:boundary]),
                                                       drop_spares(self.copy_single_cache(work))))
                start = boundary
            hidden = self._family_feed(prompt, work, chunks.between(start, len(prompt)))
        except BaseException:
            at = self._prefill_at                      # stopped between chunks: keep the progress, a taken prefix too
            kept = [len(tokens) for tokens, _ in stream.history_checkpoints]
            if at is not None and at in chunks and at not in kept:
                stream.history_checkpoints.append((list(prompt[:at]), drop_spares(self.copy_single_cache(work))))
            raise
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
        forced = self._forced_next(stream, token)
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
        chunks = self.prompt_chunks(prompt_ids)
        work, start = self._family_start(cache, cached_tokens, chunks)
        self._family_feed(prompt_ids, work, chunks.between(start, len(prompt_ids)))
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

    def _family_add_vision_streams(self, streams: Sequence[Any]) -> None:
        """Run one multimodal prefill and split its rows into independently decoded streams."""

        if len(streams) < 2:
            raise ValueError("a vision batch requires at least two streams")
        if any(not stream.prompt_ids or stream.multimodal is None for stream in streams):
            raise ValueError("every vision batch row requires a non-empty multimodal prompt")
        prompts = [stream.multimodal for stream in streams]
        batch_cache = self.model.make_vision_batch_cache(prompts)
        rows = sum(len(stream.prompt_ids) for stream in streams)
        if self.prefill_guard is not None:
            self.prefill_guard.before_chunk(batch_cache, rows)
        self._prefill_at = None
        hidden, rope_deltas = self.model.vision_prefill_batch(prompts, batch_cache)
        self.prefill_chunks += len(streams)
        if self.prefill_guard is not None:
            self.prefill_guard.after_chunk(batch_cache, rows, instances=len(streams))
        active = [
            index
            for index, stream in enumerate(streams)
            if stream.cancellation is None or not stream.cancellation.cancelled
        ]
        works = self.model.split_vision_batch(prompts, batch_cache, rope_deltas, active)
        staged: list[tuple[Any, list[Any]]] = []
        for index in active:
            stream, work = streams[index], works[index]
            self._fed_rows = len(stream.prompt_ids)
            first = self._family_first(stream, work, hidden[index : index + 1, -1:, :], 0, len(stream.prompt_ids) - 1)
            self._family_commit_first(stream, int(first.item()) if hasattr(first, "item") else int(first))
            staged.append((stream, work))
        for prompt in prompts:
            discard = getattr(prompt, "discard_buffers", None)
            if callable(discard):
                discard()
        self._prefill_at = max(len(stream.prompt_ids) for stream in streams)
        for stream, work in staged:
            self.streams.append(stream)
            if stream.finished:
                stream.finished_at = time.perf_counter()
                self._release_stream_state(stream.stream_id)
            else:
                self._live.append((stream, work))
