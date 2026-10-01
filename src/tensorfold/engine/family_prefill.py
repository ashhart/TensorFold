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
    prefill_pass = 8            # plan chunks one forward may take while a prompt fills alone (1: a chunk a forward)
    pass_cache = 16 * 1024**3   # MLX's cache of freed buffers during a pass, where the memory budget has room for it

    def _family_feed(self, tokens: Sequence[int], cache: list[Any], chunks: Sequence[tuple[int, int]],
                     prompt_data: Any = None) -> Any:
        """Absorb the ``chunks`` ([begin, end) of ``tokens``); the last hidden state [1, 1, D], draft head fed too."""

        return drain(self._family_feed_steps(tokens, cache, chunks, prompt_data))

    def _family_feed_steps(self, tokens: Sequence[int], cache: list[Any], chunks: Sequence[tuple[int, int]],
                           prompt_data: Any = None, wide: bool = False,
                           widths: list[int] | None = None, raised: list[bool] | None = None,
                           whole: list[Any] | None = None) -> Iterator[None]:
        """``_family_feed`` as steps: yields between forwards; chunks, absorbs and eval stay whole."""

        import mlx.core as mx

        last = None
        feed = getattr(self.model, "prefill", None) or self.model.hidden
        passes = wide and prompt_data is None and getattr(self.model, "prompt_pass", True)
        together = getattr(self.model, "hidden_pass", None) if passes else None
        reach = max(1, int(self.prefill_pass)) if together is not None else 1
        self._fed_rows = 0
        chunks = list(chunks)
        ahead = getattr(self.model, "prefetch_prompt", None)
        if ahead is not None and chunks:
            ahead(tokens, chunks[0][0], chunks[min(reach, len(chunks)) - 1][1])
        n = 0
        while n < len(chunks):
            if n:
                yield                                         # between forwards: the scheduler may run decode rounds
            width = self._pass_width(chunks, n, cache) if together is not None else 1
            span = chunks[n:n + width]
            begin, end = span[0][0], span[-1][1]
            n += width
            if widths is not None:
                widths.append(width)
            if ahead is not None and n < len(chunks):
                ahead(tokens, chunks[n][0], chunks[min(n + reach, len(chunks)) - 1][1])   # read while this computes
            rows = [int(t) for t in tokens[begin:end]]
            if self.prefill_guard is not None:
                self.prefill_guard.before_chunk(cache, len(rows))
            if whole is not None:
                whole[0] = None                        # a forward in flight: the cache holds no prompt prefix whole
            inputs = mx.array([rows], dtype=mx.uint32)
            sizes = [b - a for a, b in span]
            kept = self._raise_pass_cache(cache, sizes) if width > 1 else None
            if raised is not None:
                raised.append(kept is not None)
            try:
                if prompt_data is not None:
                    hidden = self.model.prefill_vision(inputs, cache, prompt_data, begin, end)
                elif width > 1:
                    hidden = together(inputs, cache, sizes)
                else:
                    hidden = feed(inputs, cache)
                self._fed_rows = len(rows)
                self.prefill_chunks += width
                self.prefill_tokens += len(rows)
                last = hidden[:, -1:, :]
                drafting = getattr(self.model, "mtp", None) is not None
                if drafting:
                    for a, b in span:                         # chunk by chunk, as one chunk a forward feeds the head
                        nxt = [int(t) for t in tokens[a + 1:b + 1]]
                        if nxt:
                            at = a - begin
                            self.model.absorb_draft_context(hidden[:, at:at + len(nxt)],
                                                            mx.array(nxt, dtype=mx.uint32), cache, start=at)
                # an earlier chunk is read only through its caches (and a taps head's taps): MLX skips its last layer
                reads_last = n == len(chunks) or (drafting and getattr(self.model, "draft_reads_hidden", True))
                mx.eval(*((last,) if reads_last else ()), *cache_arrays(cache))
            finally:
                if kept is not None:
                    mx.set_cache_limit(kept)
            if whole is not None:
                whole[0] = end
            if self.prefill_guard is not None:
                self.prefill_guard.after_chunk(cache, len(rows))
        return last

    def _raise_pass_cache(self, cache: list[Any], sizes: list[int]) -> int | None:
        """The pass-cache limit raised for one pass where the budget has room, or None when unchanged."""

        import mlx.core as mx

        old = int(mx.set_cache_limit(int(self.pass_cache)))
        guard = self.prefill_guard
        if old >= self.pass_cache or (guard is not None and not guard.pass_room(cache, sizes, self.pass_cache - old)):
            mx.set_cache_limit(old)
            return None
        return old

    def _pass_width(self, chunks: list[tuple[int, int]], n: int, cache: list[Any]) -> int:
        """How many chunks a forward takes: several only while no stream waits, and as many as fit."""

        guard = self.prefill_guard
        if guard is None and getattr(self, "active_count", 0):      # in process: live streams' rounds wait on it
            return 1
        small = int(getattr(self.model, "fused_rows", 16))
        sizes: list[int] = []
        for a, b in chunks[n:n + max(1, int(self.prefill_pass))]:
            if b - a <= small:
                break
            sizes.append(b - a)
        if len(sizes) < 2:
            return 1
        return max(1, min(len(sizes), guard.pass_width(cache, sizes))) if guard is not None else len(sizes)

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
        cached_tokens = start
        whole: list[Any] = [start]         # this prompt's own progress: other prompts' forwards run between its own
        stream.history_checkpoints = []
        stream.prefill_widths = []
        stream.prefill_raised = []
        try:
            fed = False
            for boundary in sorted({chunks.floor(int(b)) for b in checkpoints_at}):
                if not start < boundary < len(prompt):
                    continue
                if fed:
                    yield
                yield from self._family_feed_steps(prompt, work, chunks.between(start, boundary), wide=True,
                                                   widths=stream.prefill_widths, raised=stream.prefill_raised,
                                                   whole=whole)
                fed = True
                if self.prefill_guard is None or self.prefill_guard.allow_checkpoint(work):
                    stream.history_checkpoints.append((list(prompt[:boundary]),
                                                       drop_spares(self.copy_single_cache(work))))
                else:
                    self.prefill_guard.refuse(boundary, work)        # logged where it refuses (issue #155)
                start = boundary
            if fed:
                yield
            hidden = yield from self._family_feed_steps(prompt, work, chunks.between(start, len(prompt)), prepared,
                                                        wide=True, widths=stream.prefill_widths,
                                                        raised=stream.prefill_raised, whole=whole)
        except BaseException:
            at = whole[0]                              # stopped between chunks: keep the progress, a taken prefix too
            kept = [len(tokens) for tokens, _ in stream.history_checkpoints]
            if prepared is None and at is not None and at in chunks and at not in kept:
                stream.history_checkpoints.append((list(prompt[:at]), drop_spares(self.copy_single_cache(work))))
            raise
        if getattr(stream, "label_ids", ()):                 # a decision: the last row, then no round
            self._family_score(stream, hidden, cached_tokens)
            return work
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

    def _family_score(self, stream: Any, hidden: Any, cached_tokens: int) -> None:
        """Read the decision's last row, draw nothing, and keep the stream out of the rounds."""

        prompt_len = len(stream.prompt_ids)
        stream.emitted = []
        stream.pending = []
        stream.cache_len = prompt_len
        stream.cached_tokens = int(cached_tokens)
        stream.started_at = time.perf_counter()
        stream.scored = self._label_logits(hidden, stream.label_ids)
        stream.finished = True
        stream.finish_reason = "decision"

    def _label_logits(self, hidden: Any, label_ids: Sequence[int]) -> tuple[list[float], float]:
        """Last-row logits of ``label_ids`` and the full-vocabulary logsumexp."""

        import math

        import mlx.core as mx

        logits = self.model.head(hidden)
        row = logits.reshape(-1, logits.shape[-1])[-1].astype(mx.float32)
        picked = row[mx.array([int(token) for token in label_ids], dtype=mx.int32)]
        peak = mx.max(row)
        logsumexp = peak + mx.log(mx.sum(mx.exp(row - peak)))
        mx.eval(picked, logsumexp)
        values = [float(item) for item in picked.tolist()]
        total = float(logsumexp.item())
        if not math.isfinite(total) or any(not math.isfinite(value) for value in values):
            raise ValueError("label scoring produced a non-finite logit")
        return values, total

    def score_labels(self, prompt_ids: Sequence[int], label_ids: Sequence[int]) -> tuple[list[float], float]:
        """Last-position logits of ``label_ids`` and the full-vocabulary logsumexp. No token is sampled."""

        prompt = [int(token) for token in prompt_ids]
        labels = [int(token) for token in label_ids]
        if not prompt:
            raise ValueError("empty prompt")
        if not labels:
            raise ValueError("empty labels")
        chunks = self.prompt_chunks(prompt)
        work, start = self._family_start(None, 0, chunks)
        try:
            hidden = self._family_feed(prompt, work, chunks.between(start, len(prompt)))
            return self._label_logits(hidden, labels)
        finally:
            del work

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
