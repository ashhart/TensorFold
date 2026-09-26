"""Serial exact decoding for model families the lane engine does not cover yet.

``SerialEngine`` implements the part of ``LaneEngine``'s interface that the lane
server's scheduler and chat app use (``add_stream``, ``step``, cache copies for
prompt checkpoints), so every family in ``tensorfold.families`` is served by
the same OpenAI endpoint with the same prompt rendering, tool-call parsing and
prefix reuse. Each round feeds one token and samples the next with
``exact_sampling`` at its absolute position: this is the reference output a
faster path for the family has to reproduce bit for bit.

A family model provides ``make_cache()``, ``hidden(inputs, cache)`` (the last
hidden states, [1, L, D]) and ``head(hidden)`` (logits). A model that sets
``gpu_tokens`` (its decode step takes the token as a GPU array) is decoded one
step ahead: tokens are drawn on the GPU (``gpu_sampling``, the same rule in
fp32), and each round queues the next forward on the new token before reading
it, so reading tokens, streaming them and building the next graph overlap the
GPU. The tokens are the ones a synchronous loop with the same sampler draws.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from tensorfold.engine.alternating_kv import drop_spares
from tensorfold.engine.lane_engine import LaneEngine, LaneStream, RoundStats


def mx_eval_all(*arrays: Any) -> None:
    import mlx.core as mx

    mx.eval(*arrays)


def cache_arrays(cache: list[Any]) -> list[Any]:
    arrays: list[Any] = []
    for item in cache:
        if getattr(item, "keys", 0) is None:       # a KV cache nothing was written to yet (the MTP head's)
            continue
        state = item.state
        if isinstance(state, (list, tuple)):
            arrays.extend(a for a in state if a is not None and hasattr(a, "shape"))
        elif state is not None and hasattr(state, "shape"):
            arrays.append(state)
    return arrays


class SerialEngine:
    """One token per forward for each live stream, streams taken in turn."""

    # Prompt tokens per prefill forward; only the last position goes through the head.
    prefill_step = 2048
    copy_single_cache = staticmethod(LaneEngine.copy_single_cache)
    cache_nbytes = staticmethod(LaneEngine.cache_nbytes)

    def __init__(self, model: Any, *, retain_finished_caches: bool = False, **_: Any) -> None:
        self.model = model
        self.retain_finished_caches = bool(retain_finished_caches)
        self.streams: list[LaneStream] = []
        self.finished_caches: dict[str, tuple[list[int], list[Any]]] = {}
        self.round_stats: list[RoundStats] = []
        self._live: list[tuple[LaneStream, list[Any]]] = []
        self.pipelined = bool(getattr(model, "gpu_tokens", False))
        self._inflight: dict[str, Any] = {}      # stream id -> next token (GPU array), already queued
        # Copy windows: when the stream's proposer finds the context continuing an earlier span, the engine lets
        # the queued step land and verifies the copied tokens in windows of up to 1 + max_copy rows (exact when
        # a multi-row forward gives each row one-row bits, which the model checks at load), then goes back to
        # pipelined steps. With an MTP head (TF_MTP_ROUNDS=0 turns it off): every round a two-row window with the
        # head's draft instead.
        import os

        self.windows = self.pipelined and bool(getattr(model, "multi_row_exact", False))
        self.drafting = (self.windows and getattr(model, "mtp", None) is not None
                         and os.environ.get("TF_MTP_ROUNDS", "1") != "0")
        # a model decoded synchronously (host tokens) that drafts with its own head (model.draft): every round
        # verifies the pending token and its drafts in one forward and keeps them up to the first mismatch
        self.sync_drafts = (not self.pipelined and bool(getattr(model, "multi_row_exact", False))
                            and getattr(model, "mtp", None) is not None and callable(getattr(model, "draft", None)))
        self._mode: dict[str, str] = {}          # stream id -> "pipe" | "drain" | "verify"
        self._last_rows = 1                       # rows of the last synchronous round
        self._accept_rate: dict[str, float] = {}   # stream id -> recent share of drafts accepted (sync drafts)
        self._pos_rate: dict[str, list[float]] = {}  # stream id -> per draft position, recent acceptance given
        #                                                the drafts before it landed (models with depth_for)
        self._draft: dict[str, Any] = {}          # stream id -> drafted token for position cache_len + 1 (GPU array)
        self.drafted = 0
        self.accepted = 0
        self.round_times: list[tuple[float, float, float]] = []   # (build, wait for the GPU, after) seconds
        # copied drafts per round at most (a verify window of 1 + max_copy rows)
        self.max_copy = 7
        # matching tokens behind a proposal before leaving pipelined steps for copy windows: coincidental short
        # matches in fresh code (indentation, "self.") failed 56 of 70 copied tokens and cost 5% (2026-09-25)
        self.enter_match = 8

    @property
    def active_count(self) -> int:
        return sum(1 for stream, _ in self._live if not stream.finished)

    def reset(self) -> None:
        self._live = []
        self._inflight = {}
        self._draft = {}
        self._mode = {}

    # -- prefill -----------------------------------------------------------
    def _feed(self, tokens: Sequence[int], cache: list[Any], following: int | None = None) -> Any:
        """Absorb ``tokens`` into ``cache``; the last position's hidden state [1, 1, D].

        With an MTP head, its cache absorbs every position whose next token is known: the next prompt
        token, and ``following`` after the last one (the first token after this segment, if known).
        """

        import mlx.core as mx

        last = None
        step = max(1, int(self.prefill_step))
        for begin in range(0, len(tokens), step):
            chunk = [int(t) for t in tokens[begin:begin + step]]
            hidden = self.model.hidden(mx.array([chunk], dtype=mx.uint32), cache)
            last = hidden[:, -1:, :]
            if self.drafting or self.sync_drafts:
                nxt = [int(t) for t in tokens[begin + 1:begin + step + 1]]
                if len(nxt) < len(chunk) and following is not None:
                    nxt.append(int(following))
                if nxt:
                    self.model.absorb_draft_context(hidden[:, :len(nxt)], mx.array(nxt, dtype=mx.uint32), cache)
            mx.eval(last, *cache_arrays(cache))
        return last

    def _sample(self, stream: LaneStream, logits: Any, position: int) -> int:
        return self._draw(stream, logits.reshape(1, -1), [position])[0]

    def _draw(self, stream: LaneStream, logits: Any, positions: Sequence[int]) -> list[int]:
        """Tokens for rows of logits [R, V] at ``positions``: ``gpu_sampling`` when the model asks for it
        (one kernel, only the ids come back), else ``exact_sampling`` (the same keyed rule, finished on the host).
        Serial steps and drafted rounds draw the same way, so they emit the same tokens."""

        import mlx.core as mx

        if stream.sampling is None:
            return [int(t) for t in mx.argmax(logits, axis=-1).tolist()]
        if getattr(self.model, "gpu_sampling", False):
            from tensorfold.engine.gpu_sampling import sample as gpu_sample

            return [int(t) for t in gpu_sample(logits, stream.sampling, positions).tolist()]
        from tensorfold.engine.exact_sampling import sample_rows

        return [int(t) for t in sample_rows(logits, positions, stream.sampling)]

    def prefill(
        self,
        stream: LaneStream,
        *,
        cache: list[Any] | None = None,
        cached_tokens: int = 0,
        checkpoints_at: Sequence[int] = (),
    ) -> list[Any]:
        """As ``LaneEngine.prefill``: absorb the prompt, sample the first token, keep checkpoints."""

        if not stream.prompt_ids:
            raise ValueError(f"{stream.stream_id}: empty prompt")
        if cache is None:
            work, start = self.model.make_cache(), 0
        else:
            work, start = cache, int(cached_tokens)
            adopt = getattr(self.model, "adopt_cache", None)
            if adopt is not None:
                work = adopt(work)                  # e.g. a stored block's KV layers as the model's own class
        if not 0 <= start < len(stream.prompt_ids):
            raise ValueError(f"{stream.stream_id}: cached_tokens must leave a suffix to prefill")
        stream.history_checkpoints = []
        for boundary in sorted({int(b) for b in checkpoints_at}):
            if not start < boundary < len(stream.prompt_ids):
                continue
            self._feed(stream.prompt_ids[start:boundary], work, following=stream.prompt_ids[boundary])
            stream.history_checkpoints.append((list(stream.prompt_ids[:boundary]),
                                               drop_spares(self.copy_single_cache(work))))
            start = boundary
        hidden = self._feed(stream.prompt_ids[start:], work)
        stream.emitted = []
        stream.pending = []
        stream.cache_len = len(stream.prompt_ids)
        stream.cached_tokens = int(cached_tokens)
        stream.started_at = time.perf_counter()
        if self.drafting and stream.drafts:
            from tensorfold.engine.gpu_sampling import sample as gpu_sample

            position = len(stream.prompt_ids)
            token = gpu_sample(self.model.head(hidden), stream.sampling, [position])
            draft = gpu_sample(self.model.draft_logits(hidden, token, work), stream.sampling, [position + 1])
            mx_eval_all(token, draft)
            first = int(token.item())
            self._draft[stream.stream_id] = draft
        elif self.pipelined:
            from tensorfold.engine.gpu_sampling import sample as gpu_sample

            token = gpu_sample(self.model.head(hidden), stream.sampling, [len(stream.prompt_ids)])
            self._queue_next(stream, work, token)
            first = int(token.item())
        else:
            first = self._sample(stream, self.model.head(hidden), len(stream.prompt_ids))
            if self.sync_drafts and stream.drafts:
                self._draft[stream.stream_id] = self.model.draft(
                    work, self.model.last_streams[-1:], [first], len(stream.prompt_ids) + 1, stream.sampling, 1)
        stream.commit([first])
        stream.pending = [first]
        return work

    def _queue_next(self, stream: LaneStream, cache: list[Any], token: Any) -> None:
        """Feed ``token`` (a GPU array, not read yet) and queue the draw of the one after it."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        hidden = self.model.hidden(token.reshape(1, 1), cache)
        stream.cache_len += 1
        nxt = gpu_sample(self.model.head(hidden), stream.sampling, [stream.cache_len])
        mx.async_eval(nxt)
        self._inflight[stream.stream_id] = nxt

    def prefill_prefix(
        self, prompt_ids: Sequence[int], *, cache: list[Any] | None = None, cached_tokens: int = 0
    ) -> list[Any]:
        if not prompt_ids:
            raise ValueError("empty prefix")
        work = self.model.make_cache() if cache is None else cache
        start = int(cached_tokens) if cache is not None else 0
        if not 0 <= start < len(prompt_ids):
            raise ValueError("cached_tokens must leave a suffix to prefill")
        self._feed(list(prompt_ids[start:]), work)
        return drop_spares(work)

    def add_stream(
        self,
        stream: LaneStream,
        *,
        cache: list[Any] | None = None,
        cached_tokens: int = 0,
        checkpoints_at: Sequence[int] = (),
    ) -> None:
        work = self.prefill(stream, cache=cache, cached_tokens=cached_tokens, checkpoints_at=checkpoints_at)
        self.streams.append(stream)
        if stream.finished:
            stream.finished_at = time.perf_counter()
            self._inflight.pop(stream.stream_id, None)
            self._draft.pop(stream.stream_id, None)
            return
        self._live.append((stream, work))

    def add_forked(self, stream: LaneStream, *, cache: list[Any], cached_tokens: int) -> None:
        self.add_stream(stream, cache=cache, cached_tokens=cached_tokens)

    def fork_shared(self, prefix_cache: list[Any]) -> list[Any]:
        return self.copy_single_cache(prefix_cache)

    # -- decode ------------------------------------------------------------
    def step(self) -> dict[str, list[int]]:
        import mlx.core as mx

        landed: dict[str, list[int]] = {}
        for stream, cache in self._live:
            if stream.finished:
                self._inflight.pop(stream.stream_id, None)
                continue
            started = time.perf_counter()
            if self.sync_drafts:
                got = self._sync_draft_round(stream, cache)
                stream.rounds += 1
                landed[stream.stream_id] = got
                ms = (time.perf_counter() - started) * 1e3
                rows = self._last_rows
                self.round_stats.append(RoundStats(
                    streams=1, width=rows, rows=rows, ragged=False, rollbacks=0,
                    committed=len(got), forward_ms=ms, finalize_ms=0.0, rollback_ms=0.0, total_ms=ms,
                    started_at=started))
                if stream.finished:
                    stream.finished_at = time.perf_counter()
                    self._draft.pop(stream.stream_id, None)
                    if self.retain_finished_caches and stream.retain:
                        self.finished_caches[stream.stream_id] = (stream.context[: stream.cache_len], drop_spares(cache))
                continue
            if self.drafting and stream.drafts:
                got = self._draft_round(stream, cache)
                stream.rounds += 1
                landed[stream.stream_id] = got
                ms = (time.perf_counter() - started) * 1e3
                self.round_stats.append(RoundStats(
                    streams=1, width=2, rows=2, ragged=False, rollbacks=0, committed=len(got),
                    forward_ms=ms, finalize_ms=0.0, rollback_ms=0.0, total_ms=ms, started_at=started))
                if stream.finished:
                    stream.finished_at = time.perf_counter()
                    self._draft.pop(stream.stream_id, None)
                    if self.retain_finished_caches and stream.retain:
                        self.finished_caches[stream.stream_id] = (stream.context[: stream.cache_len], drop_spares(cache))
                continue
            if self.pipelined:
                mode = self._mode.get(stream.stream_id, "pipe")
                if mode in ("verify", "exit"):
                    proposal = self._proposal(stream) if mode == "verify" else []
                    if proposal:
                        got = self._copy_round(stream, cache, proposal)
                        stream.rounds += 1
                        landed[stream.stream_id] = got
                        ms = (time.perf_counter() - started) * 1e3
                        self.round_stats.append(RoundStats(
                            streams=1, width=1 + len(proposal), rows=1 + len(proposal), ragged=False,
                            rollbacks=0, committed=len(got), forward_ms=ms, finalize_ms=0.0, rollback_ms=0.0,
                            total_ms=ms, started_at=started))
                        if stream.finished:
                            stream.finished_at = time.perf_counter()
                            if self.retain_finished_caches and stream.retain:
                                self.finished_caches[stream.stream_id] = (stream.context[: stream.cache_len], drop_spares(cache))
                        continue
                    # nothing to copy: back to pipelined steps (this step lands next round)
                    self._mode[stream.stream_id] = "pipe"
                    self._queue_next(stream, cache, mx.array([stream.pending[-1]], dtype=mx.uint32))
                    landed[stream.stream_id] = []
                    continue
                current = self._inflight.pop(stream.stream_id)
                forced = self._forced_next(stream)
                if forced is not None:
                    current = mx.array([forced], dtype=mx.uint32)   # the thinking budget's token, not the sample
                if mode == "drain":
                    token = int(current.item())                 # the last queued step: no new one
                    self._mode[stream.stream_id] = "verify"
                else:
                    self._queue_next(stream, cache, current)    # the GPU starts the next step first
                    token = int(current.item())
            else:
                hidden = self.model.hidden(mx.array([[stream.pending[-1]]], dtype=mx.uint32), cache)
                token = self._sample(stream, self.model.head(hidden), stream.cache_len + 1)
                forced = self._forced_next(stream)
                token = token if forced is None else forced
                stream.cache_len += 1
            stream.rounds += 1
            got = stream.commit([token])
            stream.pending = [token]
            landed[stream.stream_id] = got
            if (self.windows and self._mode.get(stream.stream_id, "pipe") == "pipe" and not stream.finished
                    and self._proposal(stream, self.enter_match)):
                self._mode[stream.stream_id] = "drain"          # a copy window is ahead: land the queued step
            ms = (time.perf_counter() - started) * 1e3
            self.round_stats.append(RoundStats(
                streams=1, width=1, rows=1, ragged=False, rollbacks=0, committed=len(got),
                forward_ms=ms, finalize_ms=0.0, rollback_ms=0.0, total_ms=ms, started_at=started))
            if stream.finished:
                stream.finished_at = time.perf_counter()
                self._inflight.pop(stream.stream_id, None)
                if self.retain_finished_caches and stream.retain:
                    # synchronous: the cache holds all but the last token; pipelined: the last one too
                    self.finished_caches[stream.stream_id] = (stream.context[: stream.cache_len], drop_spares(cache))
        self._live = [(s, c) for s, c in self._live if not s.finished]
        return landed

    @staticmethod
    def _forced_next(stream: LaneStream) -> int | None:
        """The token the thinking budget writes at the stream's next position (by count, whatever the model
        samples there; the rest of the close follows), or None."""

        if stream.force:
            return int(stream.force.pop(0))
        if stream.think_cut([-1]) == 0:
            return stream.start_close()
        return None

    def _proposal(self, stream: LaneStream, min_match: int = 0) -> list[int]:
        """The proposer's copied continuation (2+ tokens), backed by at least ``min_match`` matching tokens."""

        if stream.proposer is None or self.max_copy <= 0 or stream.force:
            return []
        copied = [int(t) for t in stream.proposer.propose(stream.context, self.max_copy)]
        if len(copied) < 2 or getattr(stream.proposer, "last_match", 0) < min_match:
            return []
        return copied

    def _copy_round(self, stream: LaneStream, cache: list[Any], proposal: list[int]) -> list[int]:
        """Verify a copied continuation: the pending token and ``proposal`` in one forward, kept up to the first
        token that differs from what serial decoding samples there."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        position = stream.cache_len
        inputs = mx.array([stream.pending[-1], *proposal], dtype=mx.uint32)
        rows = int(inputs.shape[0])
        hidden = self.model.hidden(inputs.reshape(1, rows), cache)
        tokens = gpu_sample(self.model.head(hidden)[0], stream.sampling, [position + 1 + r for r in range(rows)])
        sampled = [int(t) for t in tokens.tolist()]
        keep = 1
        for i, draft in enumerate(proposal):
            if sampled[i] != draft:
                break
            keep += 1
        self.drafted += len(proposal)
        self.accepted += keep - 1
        stream.drafted += len(proposal)
        stream.accepted += keep - 1
        stream.proposer.observe(len(proposal), keep - 1)
        cut = stream.think_cut(sampled[:keep])
        if cut is not None:
            keep = cut + 1
            sampled = [*sampled[:cut], stream.start_close()]
        if keep < rows:
            self.model.keep_rows(cache, rows, keep)
        stream.cache_len += keep
        got = stream.commit(sampled[:keep])
        stream.pending = [sampled[keep - 1]]
        if keep == 1 or cut is not None:
            self._mode[stream.stream_id] = "exit"               # back to pipelined steps
        return got

    def _draft_round(self, stream: LaneStream, cache: list[Any]) -> list[int]:
        """One verify round: the pending token and its drafts in one forward of 1 + k rows.

        Drafts are a copied continuation from the context when the proposer has one backed by
        ``enter_match`` tokens (up to ``max_copy``), else the MTP head's one draft. Row i (position p + i)
        samples position p + i + 1; drafts are kept up to the first that differs from the sampled token, and
        the rest of the rows are dropped from every cache. Then the MTP head reads the kept rows (its cache)
        and drafts from the last one: queued, not read, so the GPU runs it while the next round is built, and
        the next round takes the draft as a GPU array (its value is read once that round's tokens are in).
        """

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        t0 = time.perf_counter()
        position = stream.cache_len
        mtp_draft = self._draft.pop(stream.stream_id, None)
        forced, stream.force = list(stream.force), []
        copied = [] if forced else self._proposal(stream, self.enter_match)
        if forced:                      # the thinking budget's close: read, committed as it is
            inputs = mx.array([stream.pending[-1], *forced], dtype=mx.uint32)
        elif copied:
            inputs = mx.array([stream.pending[-1], *copied], dtype=mx.uint32)
        elif mtp_draft is not None:
            inputs = mx.concatenate([mx.array([stream.pending[-1]], dtype=mx.uint32), mtp_draft.astype(mx.uint32)])
        else:
            inputs = mx.array([stream.pending[-1]], dtype=mx.uint32)
        rows = int(inputs.shape[0])
        hidden = self.model.hidden(inputs.reshape(1, rows), cache)
        tokens = gpu_sample(self.model.head(hidden)[0], stream.sampling,
                            [position + 1 + r for r in range(rows)])
        t1 = time.perf_counter()
        sampled = [int(t) for t in tokens.tolist()]
        t2 = time.perf_counter()
        keep = 1
        if forced:
            keep = rows
            sampled = [*forced, sampled[-1]]
        else:
            proposed = copied if copied else ([int(mtp_draft.item())] if rows > 1 and mtp_draft is not None else [])
            for i, draft in enumerate(proposed):
                if sampled[i] != draft:
                    break
                keep += 1
            if proposed:
                self.drafted += len(proposed)
                self.accepted += keep - 1
                stream.drafted += len(proposed)
                stream.accepted += keep - 1
                if copied and stream.proposer is not None:
                    stream.proposer.observe(len(copied), keep - 1)
        cut = stream.think_cut(sampled[:keep])
        if cut is not None:
            keep = cut + 1
            sampled = [*sampled[:cut], stream.start_close()]
        if keep < rows:
            self.model.keep_rows(cache, rows, keep)
        stream.cache_len += keep
        got = stream.commit(sampled[:keep])
        stream.pending = [sampled[keep - 1]]
        if not stream.finished:
            # the head reads the kept rows with the tokens that follow them (a forced close included)
            nxt = mx.array(sampled[:keep], dtype=mx.uint32)
            logits = self.model.draft_logits(hidden[:, :keep], nxt, cache, last_only=True)
            draft = gpu_sample(logits[0], stream.sampling, [position + 1 + keep])
            mx.async_eval(draft)
            self._draft[stream.stream_id] = draft
        self.round_times.append((t1 - t0, t2 - t1, time.perf_counter() - t2))
        return got

    def _sync_draft_round(self, stream: LaneStream, cache: list[Any]) -> list[int]:
        """The pending token and the model's drafts in one forward; kept up to the first draft that differs from
        the token sampled there (the same sampler, keyed by position, as serial decoding); then the model drafts
        again from the kept rows."""

        import mlx.core as mx

        position = stream.cache_len
        drafts = list(self._draft.pop(stream.stream_id, None) or []) if stream.drafts else []
        forced, stream.force = list(stream.force), []
        # a copied continuation of the context, backed by enter_match tokens, beats the model's own drafts (a
        # file edit or rewrite: up to 1 + max_copy rows a round)
        copied = [] if forced else self._proposal(stream, self.enter_match)
        if forced:
            drafts = forced                      # the thinking budget's close: read, committed as they are
        elif copied:
            drafts = copied
        inputs = [stream.pending[-1], *drafts]
        rows = len(inputs)
        self._last_rows = rows
        hidden = self.model.hidden(mx.array([inputs], dtype=mx.uint32), cache)
        logits = self.model.head(hidden)
        logits = logits.reshape(logits.shape[1:])       # [R, V] as a view (MLX's [0] is a gather that copies)
        positions = [position + 1 + r for r in range(rows)]
        # the MTP head's first draft for every row, queued behind the verify before anything is read (the rows
        # past the kept ones are dropped afterwards); forced rows take the plain path
        speculate = (stream.drafts and not forced and getattr(self.model, "gpu_sampling", False)
                     and callable(getattr(self.model, "speculate", None)))
        firsts: list[int] = []
        if speculate:
            from tensorfold.engine.gpu_sampling import sample as gpu_sample

            tokens = gpu_sample(logits, stream.sampling, positions)
            both = mx.concatenate([tokens, self.model.speculate(cache, tokens, position, stream.sampling)]).tolist()
            sampled, firsts = [int(t) for t in both[:rows]], [int(t) for t in both[rows:]]
        else:
            sampled = self._draw(stream, logits, positions)
        keep = 1
        if forced:
            keep = rows
            sampled = [*forced, sampled[-1]]
        else:
            for i, draft in enumerate(drafts):
                if sampled[i] != draft:
                    break
                keep += 1
            self.drafted += len(drafts)
            self.accepted += keep - 1
            stream.drafted += len(drafts)
            stream.accepted += keep - 1
            if copied and stream.proposer is not None:
                stream.proposer.observe(len(copied), keep - 1)
        cut = stream.think_cut(sampled[:keep])
        if cut is not None:
            keep = cut + 1
            sampled = [*sampled[:cut], stream.start_close()]
            drafts = []                          # no acceptance sample this round
            if speculate:
                self.model.unspeculate(cache)    # the last kept row's next token is the forced one
                speculate = False
        if keep < rows:
            self.model.keep_rows(cache, rows, keep)
        streams = self.model.last_streams[:keep]
        stream.cache_len += keep
        got = stream.commit(sampled[:keep])
        stream.pending = [sampled[keep - 1]]
        if not stream.finished and stream.drafts:
            depth = self._depth(stream, [] if forced else drafts, keep)
            if stream.force or self._proposal(stream, self.enter_match):
                depth = 0          # the next round reads forced tokens or a copied continuation: absorb only
            if speculate:
                self._draft[stream.stream_id] = self.model.settle(cache, keep, firsts[keep - 1],
                                                                  stream.cache_len + 1, stream.sampling, depth)
            else:
                self._draft[stream.stream_id] = self.model.draft(cache, streams, sampled[:keep],
                                                                 stream.cache_len + 1, stream.sampling, depth)
        elif speculate:
            self.model.settle(cache, keep, firsts[keep - 1], stream.cache_len + 1, stream.sampling, 0)
        return got

    def _depth(self, stream: LaneStream, drafts: list[int], keep: int) -> int:
        """Drafts for the next round, from the stream's recent acceptance (an average over ~8 rounds): each
        verified row costs about a third of a one-row forward, so a deeper chain pays only while most of its
        drafts land (Flash Next on an M3 Ultra: ~87% for code, ~73% for prose, 2026-09-25)."""

        most = int(getattr(self.model, "drafts", 1))
        depth_for = getattr(self.model, "depth_for", None)
        if callable(depth_for):
            # the model picks the depth from, for each draft position j, the chance that draft j lands given that
            # drafts 0 .. j-1 landed (a running average over ~8 rounds that checked position j): a chained draft
            # lands less often than the first. Checked this round: the accepted drafts plus the first rejected one.
            rates = self._pos_rate.setdefault(stream.stream_id, [])
            for j in range(min(len(drafts), keep)):
                while len(rates) <= j:
                    rates.append(rates[-1] * 0.8 if rates else 0.8)
                rates[j] = 0.875 * rates[j] + 0.125 * (1.0 if j < keep - 1 else 0.0)
            return max(1, min(most, int(depth_for(list(rates) or [0.8]))))   # unmeasured positions: the model's prior
        rate = self._accept_rate.get(stream.stream_id, 0.8)
        if drafts:
            rate = 0.875 * rate + 0.125 * ((keep - 1) / len(drafts))
        self._accept_rate[stream.stream_id] = rate
        return max(1, min(most, 1 if rate < 0.8 else 2 if rate < 0.9 else 3))

    def summary(self) -> dict[str, Any]:
        return {"engine": "serial", "rounds": len(self.round_stats), "streams": len(self.streams),
                "drafted": self.drafted, "accepted": self.accepted}
