"""The family rounds' prefill: prompt chunks, checkpoints at chunk starts, the first token and first drafts."""

from __future__ import annotations

from functools import partial
import time
from typing import Any, Iterator, Sequence

from tensorfold.engine.family_common import cache_arrays, drop_spares
from tensorfold.engine.prefill_plan import PromptChunks


def drain(steps: Iterator[Any]) -> Any:
    """Run a generator of prefill steps to its end and return its value."""

    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


class FamilyPrefill:
    """Prefill for ``FamilyRounds``."""

    prefill_tokens = 0          # prompt tokens fed, for the server's live line

    _prefill_at: int | None = None                     # the prompt position the working cache holds whole

    def _family_feed(self, tokens: Sequence[int], cache: list[Any], chunks: Sequence[tuple[int, int]],
                     prompt_data: Any = None) -> Any:
        """Absorb the ``chunks`` ([begin, end) of ``tokens``); the last hidden state [1, 1, D], draft head fed too."""

        return drain(self._family_feed_steps(tokens, cache, chunks, prompt_data))

    def _family_feed_steps(self, tokens: Sequence[int], cache: list[Any], chunks: Sequence[tuple[int, int]],
                           prompt_data: Any = None) -> Iterator[None]:
        """``_family_feed`` as steps: it yields between chunks, each chunk's forward, draft absorb and eval whole."""

        import mlx.core as mx

        last = None
        feed = getattr(self.model, "prefill", None) or self.model.hidden
        self._fed_rows = 0
        chunks = list(chunks)
        ahead = getattr(self.model, "prefetch_prompt", None)
        if ahead is not None and chunks:
            ahead(tokens, *chunks[0])
        for n, (begin, end) in enumerate(chunks):
            if n:
                yield                                         # between chunks: the scheduler may run decode rounds
            if ahead is not None and n + 1 < len(chunks):
                ahead(tokens, *chunks[n + 1])                 # its host reads run while this chunk computes
            chunk = [int(t) for t in tokens[begin:end]]
            if self.prefill_guard is not None:
                self.prefill_guard.before_chunk(cache, len(chunk))
            self._prefill_at = None                    # a chunk in flight: the cache holds no prompt prefix whole
            inputs = mx.array([chunk], dtype=mx.uint32)
            hidden = (self.model.prefill_vision(inputs, cache, prompt_data, begin, end)
                      if prompt_data is not None else feed(inputs, cache))
            self._fed_rows = len(chunk)
            self.prefill_chunks += 1
            self.prefill_tokens += len(chunk)
            last = hidden[:, -1:, :]
            drafting = getattr(self.model, "mtp", None) is not None
            if drafting:
                nxt = [int(t) for t in tokens[begin + 1:end + 1]]
                if nxt:
                    self.model.absorb_draft_context(hidden[:, :len(nxt)], mx.array(nxt, dtype=mx.uint32), cache,
                                                    start=0)
            # an earlier chunk is read only through its caches (and a taps head's taps): MLX skips its last layer
            reads_last = n + 1 == len(chunks) or (drafting and getattr(self.model, "draft_reads_hidden", True))
            mx.eval(*((last,) if reads_last else ()), *cache_arrays(cache))
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
        return drain(self._family_prefill_steps(stream, cache=cache, cached_tokens=cached_tokens,
                                                checkpoints_at=checkpoints_at))

    def _family_prefill_steps(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                              checkpoints_at: Sequence[int]) -> Iterator[None]:
        """The prompt's prefill, yielding only between its chunks; the last chunk and the first token go together."""

        prompt = stream.prompt_ids
        if not prompt:
            raise ValueError(f"{stream.stream_id}: empty prompt")
        prepared = getattr(stream, "prompt_data", None)
        # an image prompt never resumes, so it needs no cut at message starts: its chunks are the step grid alone
        chunks = self.prompt_chunks(prompt) if prepared is None else PromptChunks(
            None, len(prompt), step=getattr(self.prefill_plan, "step", None) or self.prefill_step)
        work, start = self._family_start(cache, cached_tokens, chunks)
        if prepared is not None:
            if start or cache is not None:
                raise ValueError("image prompts require a fresh cache")
            prepared = self.model.encode_vision(prepared, work)
            checkpoints_at = ()
        cached_tokens = self._prefill_at = start
        stream.history_checkpoints = []
        try:
            fed = False
            for boundary in sorted({chunks.floor(int(b)) for b in checkpoints_at}):
                if not start < boundary < len(prompt):
                    continue
                if fed:
                    yield
                yield from self._family_feed_steps(prompt, work, chunks.between(start, boundary))
                fed = True
                if self.prefill_guard is None or self.prefill_guard.allow_checkpoint(work):
                    stream.history_checkpoints.append((list(prompt[:boundary]),
                                                       drop_spares(self.copy_single_cache(work))))
                start = boundary
            if fed:
                yield
            hidden = yield from self._family_feed_steps(prompt, work, chunks.between(start, len(prompt)), prepared)
        except BaseException:
            at = self._prefill_at                      # stopped between chunks: keep the progress, a taken prefix too
            kept = [len(tokens) for tokens, _ in stream.history_checkpoints]
            if prepared is None and at is not None and at in chunks and at not in kept:
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
        logits = self.model.head(hidden)
        if stream.constraint is not None:                # the first token under the reply's grammar
            logits = stream.constraint.mask(logits)
        token = self._draw(logits, stream.sampling, [prompt_len])
        forced = self._forced_next(stream, token)
        if forced is not None:
            token = mx.array([forced], dtype=mx.uint32)
        if self.family_mtp and stream.drafts:
            # the head reads the prompt's last position and the first token, and drafts the one after it
            firsts = self.model.speculate(work, token, prompt_len - 1, stream.sampling, start=row)
            # the first drafts settle at the next round, so the first token goes out without the draft forward
            self._next[stream.stream_id] = partial(self.model.settle, work, 1, firsts.reshape(-1)[:1],
                                                   prompt_len + 1, stream.sampling, self._depth(stream))
        elif self.pipelined and stream.constraint is None:      # a grammar reads each token before the next
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
                           checkpoints_at: Sequence[int]) -> Iterator[None]:
        """Prefill a stream a chunk a step; after the last chunk it takes part in the rounds."""

        work = yield from self._family_prefill_steps(stream, cache=cache, cached_tokens=cached_tokens,
                                                     checkpoints_at=checkpoints_at)
        self.streams.append(stream)
        if stream.finished:
            stream.finished_at = time.perf_counter()
            self._release_stream_state(stream.stream_id)
            return
        self._live.append((stream, work))
