"""Model families on the lane engine: one stream's verify rounds over the family's own forward and caches.

A family model (``lane_family = True``: Nemotron-H, Flash Next) brings its forward instead of mlx_lm's
batch-cache protocol:

    make_cache()                     one cache object per layer, the draft head's last
    prefill_rows                     optional: absorb prompts in chunks of this many rows (the row-exact decode
                                     path), so a prompt resumed from a stored prefix equals a fresh prefill
    hidden(inputs, cache)            hidden states [1, R, D] of R consecutive tokens [1, R], advancing the caches
    head(hidden)                     logits [1, R, V]
    keep_rows(cache, rows, keep)     after an R-row call, keep its first ``keep`` rows: KV trimmed, recurrent
                                     state as it was after row keep - 1
    exact_width                      the widest window whose every row gets a one-row forward's bits here
                                     (checked at load; 1: no drafts)
    window_costs                     {rows: ms} of a forward, timed at load for every exact width
    gpu_tokens                       ``hidden`` takes unread GPU token arrays: one-token rounds run one step ahead
    adopt_cache(cache)               optional: a stored or copied cache in the model's own classes

and with a draft head (``mtp`` set):

    absorb_draft_context(hidden, next_tokens, cache)       prompt positions into the head's cache
    speculate(cache, tokens, position, sampling, start=0, last_only=False)
                                     the head absorbs rows start .. start + n - 1 of the last ``hidden`` call
                                     (row start + i followed by tokens[i], n = len(tokens)) and draws each one's
                                     first draft, for positions position + 2 + i (``position``: row start's);
                                     ``last_only``: only the last row's
    speculate_early                  True: ``speculate`` every row right behind the verify, before anything is
                                     read (a host round trip less); False: after the read, only the kept rows,
                                     so the head's GPU work overlaps the host building the next round
    settle(cache, keep, first, position, sampling, count)  after ``speculate``: the head keeps its first
                                     ``keep`` rows; returns ``count`` drafts for positions position, ...: the
                                     kept row's ``first`` and count - 1 chained on the head's own output (a
                                     host list, or a lazy GPU array the next round feeds unread)
    unspeculate(cache)               forget the speculated rows
    drafts                           the most drafts a round
    mtp_step_ms                      optional: one chained draft step, timed at load

A round verifies the pending token and its drafts in one forward, samples every row with the keyed rule on the
GPU, keeps drafts up to the first that differs from the target's own sample, and rolls every cache back to
exactly the kept rows. Every committed token is the target's sample at its position, so drafted output is
byte-identical to one-token rounds, which run through the same kernels. Drafts come from, in order: the
thinking budget's forced close, a copied continuation of the context or a tool call's known structure (backed
by ``enter_match`` matching tokens), then the draft head's chain. Its depth each round is the one with the most
expected tokens a millisecond at the stream's recent acceptance at each depth and the measured costs.

A stream with ``drafts`` off decodes one token a round: the serial reference. With ``gpu_tokens`` those rounds
run one step ahead (the next forward is queued on the unread token), as before, and copied continuations are
verified in windows between them.
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

# TF_FAMILY_PROFILE=1: every 200 rounds, the mean host time of each phase of a round (build, wait for the GPU,
# after the read, the head's drafts), printed to the log
_PROFILE = os.environ.get("TF_FAMILY_PROFILE", "") == "1"


def drop_spares(cache: list[Any]) -> list[Any]:
    """``alternating_kv.drop_spares`` (imported when used: this module loads without MLX)."""

    from tensorfold.engine.alternating_kv import drop_spares as drop

    return drop(cache)


def cache_arrays(cache: list[Any]) -> list[Any]:
    """Every array a cache list holds (a KV cache nothing was written to yet has none)."""

    arrays: list[Any] = []
    for item in cache:
        if getattr(item, "keys", 0) is None:
            continue
        state = item.state
        if isinstance(state, (list, tuple)):
            arrays.extend(a for a in state if a is not None and hasattr(a, "shape"))
        elif state is not None and hasattr(state, "shape"):
            arrays.append(state)
    return arrays


class FamilyRounds:
    """The lane engine's rounds for model families; ``LaneEngine`` mixes it in and uses it when the model says
    ``lane_family``."""

    # prompt tokens per prefill forward (only the last position goes through the head)
    family_prefill_step = 2048
    # matching tokens behind a copied continuation before a round takes it: coincidental short matches in fresh
    # code (indentation, "self.") failed 56 of 70 copied tokens and cost 5% (Flash Next, 2026-09-25)
    enter_match = 8
    # per-depth acceptance: the prior, the weight of the newest round, and rounds between probes one deeper
    depth_prior = (0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    depth_rate = 0.15
    depth_probe_every = 8
    # a round's wall time by depth: measured rounds replace the load-time estimate at this rate
    cost_rate = 0.2

    def _family_setup(self) -> None:
        model = self.model
        self._live: list[tuple[Any, list[Any]]] = []
        self._inflight: dict[str, Any] = {}          # stream id -> next token (GPU array), already queued
        self._next: dict[str, Any] = {}              # stream id -> the head's drafts for the next round
        self._mode: dict[str, str] = {}              # stream id -> "pipe" | "drain" | "verify" | "exit"
        self._depth_state: dict[str, dict[str, Any]] = {}
        self.drafted = 0
        self.accepted = 0
        self.pipelined = bool(getattr(model, "gpu_tokens", False))
        self.family_width = max(1, min(int(getattr(model, "exact_width", 1) or 1), int(self.max_rows)))
        self.max_copy = self.family_width - 1
        self.family_mtp = (getattr(model, "mtp", None) is not None and self.family_width >= 2
                           and callable(getattr(model, "speculate", None)))
        self.speculate_early = bool(getattr(model, "speculate_early", True))
        prior = getattr(model, "draft_prior", None)
        if prior:
            self.depth_prior = tuple(float(p) for p in prior)
        self.most_drafts = max(0, min(int(getattr(model, "drafts", 1) or 0), self.family_width - 1,
                                      int(self.max_draft)))
        costs = getattr(model, "window_costs", None) or {}
        self.family_costs = {int(w): float(ms) for w, ms in costs.items() if 1 <= int(w) <= self.family_width}
        self.mtp_step_ms = float(getattr(model, "mtp_step_ms", 0.0) or 0.0)
        self._round_ms: dict[int, float] = {}        # measured wall time of a round with d head drafts

    # -- prefill -----------------------------------------------------------------------------------------------
    def _family_feed(self, tokens: Sequence[int], cache: list[Any], following: int | None = None) -> Any:
        """Absorb ``tokens``; the last position's hidden state [1, 1, D]. The draft head's cache takes every
        position whose next token is known: the next prompt token, and ``following`` after the last one."""

        import mlx.core as mx

        last = None
        # a model whose prompt must be absorbed through its row-exact decode kernels (``prefill_rows``: their
        # widest window) gets the same cache wherever a prefill starts, so a resumed prompt equals a fresh one
        step = max(1, int(getattr(self.model, "prefill_rows", 0) or self.family_prefill_step))
        self._fed_rows = 0
        for begin in range(0, len(tokens), step):
            chunk = [int(t) for t in tokens[begin:begin + step]]
            hidden = self.model.hidden(mx.array([chunk], dtype=mx.uint32), cache)
            self._fed_rows = len(chunk)
            last = hidden[:, -1:, :]
            if getattr(self.model, "mtp", None) is not None:
                nxt = [int(t) for t in tokens[begin + 1:begin + step + 1]]
                if len(nxt) < len(chunk) and following is not None:
                    nxt.append(int(following))
                if nxt:
                    self.model.absorb_draft_context(hidden[:, :len(nxt)], mx.array(nxt, dtype=mx.uint32), cache)
            mx.eval(last, *cache_arrays(cache))
        return last

    def _family_prefill(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                        checkpoints_at: Sequence[int]) -> list[Any]:
        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        if not stream.prompt_ids:
            raise ValueError(f"{stream.stream_id}: empty prompt")
        if cache is None:
            work, start = self.model.make_cache(), 0
        else:
            work, start = cache, int(cached_tokens)
            adopt = getattr(self.model, "adopt_cache", None)
            if adopt is not None:
                work = adopt(work)
        if not 0 <= start < len(stream.prompt_ids):
            raise ValueError(f"{stream.stream_id}: cached_tokens must leave a suffix to prefill")
        stream.history_checkpoints = []
        for boundary in sorted({int(b) for b in checkpoints_at}):
            if not start < boundary < len(stream.prompt_ids):
                continue
            self._family_feed(stream.prompt_ids[start:boundary], work, following=stream.prompt_ids[boundary])
            stream.history_checkpoints.append((list(stream.prompt_ids[:boundary]),
                                               drop_spares(self.copy_single_cache(work))))
            start = boundary
        hidden = self._family_feed(stream.prompt_ids[start:], work)
        prompt_len = len(stream.prompt_ids)
        stream.emitted = []
        stream.pending = []
        stream.cache_len = prompt_len
        stream.cached_tokens = int(cached_tokens)
        stream.started_at = time.perf_counter()
        token = gpu_sample(self.model.head(hidden), stream.sampling, [prompt_len])
        if self.family_mtp and stream.drafts:
            import mlx.core as mx

            # the head reads the prompt's last position and the first token, and drafts the one after it
            firsts = self.model.speculate(work, token, prompt_len - 1, stream.sampling, start=self._fed_rows - 1)
            first, first_draft = [int(t) for t in mx.concatenate([token, firsts.astype(token.dtype)]).tolist()]
            depth = self._depth(stream)
            self._next[stream.stream_id] = self.model.settle(work, 1, first_draft, prompt_len + 1, stream.sampling,
                                                             depth)
        elif self.pipelined:
            self._queue_next(stream, work, token)
            first = int(token.item())
        else:
            first = int(token.item())
        stream.commit([first])
        stream.pending = [first]
        return work

    def _queue_next(self, stream: Any, cache: list[Any], token: Any) -> None:
        """Feed ``token`` (a GPU array, not read yet) and queue the draw of the one after it."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        hidden = self.model.hidden(token.reshape(1, 1), cache)
        stream.cache_len += 1
        nxt = gpu_sample(self.model.head(hidden), stream.sampling, [stream.cache_len])
        mx.async_eval(nxt)
        self._inflight[stream.stream_id] = nxt

    def _family_prefill_prefix(self, prompt_ids: Sequence[int], *, cache: list[Any] | None,
                               cached_tokens: int) -> list[Any]:
        if not prompt_ids:
            raise ValueError("empty prefix")
        work = self.model.make_cache() if cache is None else cache
        start = int(cached_tokens) if cache is not None else 0
        if not 0 <= start < len(prompt_ids):
            raise ValueError("cached_tokens must leave a suffix to prefill")
        self._family_feed(list(prompt_ids[start:]), work)
        return drop_spares(work)

    def _family_add_stream(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                           checkpoints_at: Sequence[int]) -> None:
        work = self._family_prefill(stream, cache=cache, cached_tokens=cached_tokens, checkpoints_at=checkpoints_at)
        self.streams.append(stream)
        if stream.finished:
            stream.finished_at = time.perf_counter()
            self._inflight.pop(stream.stream_id, None)
            self._next.pop(stream.stream_id, None)
            return
        self._live.append((stream, work))

    # -- rounds ------------------------------------------------------------------------------------------------
    def _family_step(self) -> dict[str, list[int]]:
        landed: dict[str, list[int]] = {}
        for stream, cache in self._live:
            if stream.finished:
                self._inflight.pop(stream.stream_id, None)
                self._next.pop(stream.stream_id, None)
                continue
            started = time.perf_counter()
            if (self.family_mtp and stream.drafts) or not self.pipelined:
                got, rows, keep = self._family_round(stream, cache)
            else:
                got, rows, keep = self._pipelined_round(stream, cache)
            landed[stream.stream_id] = got
            stream.min_rows = rows if not stream.min_rows else min(stream.min_rows, rows)
            ms = (time.perf_counter() - started) * 1e3
            from tensorfold.engine.lane_engine import RoundStats

            self.round_stats.append(RoundStats(
                streams=1, width=rows, rows=rows, ragged=False, rollbacks=int(0 < keep < rows), committed=len(got),
                forward_ms=ms, finalize_ms=0.0, rollback_ms=0.0, total_ms=ms, started_at=started))
            if stream.finished:
                stream.finished_at = time.perf_counter()
                self._inflight.pop(stream.stream_id, None)
                self._next.pop(stream.stream_id, None)
                if self.retain_finished_caches and stream.retain:
                    # one-token rounds leave the last token absorbed when they ran ahead; ``cache_len`` counts it
                    self.finished_caches[stream.stream_id] = (stream.context[: stream.cache_len], drop_spares(cache))
        self._live = [(s, c) for s, c in self._live if not s.finished]
        return landed

    @staticmethod
    def _forced_next(stream: Any) -> int | None:
        """The token the thinking budget writes at the stream's next position, or None."""

        if stream.force:
            return int(stream.force.pop(0))
        if stream.think_cut([-1]) == 0:
            return stream.start_close()
        return None

    def _copy_proposal(self, stream: Any, min_match: int | None = None) -> list[int]:
        """The proposer's continuation (2+ tokens): a copied span backed by ``enter_match`` matching tokens, or
        a tool call's known structure."""

        if stream.proposer is None or self.max_copy <= 0 or stream.force:
            return []
        try:
            copied = [int(t) for t in stream.proposer.propose(stream.context, min(self.max_copy,
                                                                                 stream.draft_room - 1))]
        except Exception:  # noqa: BLE001 - a proposer must never break a stream
            return []
        need = self.enter_match if min_match is None else min_match
        if len(copied) < 2 or int(getattr(stream.proposer, "last_match", 0) or 0) < need:
            return []
        return copied

    def _pipelined_round(self, stream: Any, cache: list[Any]) -> tuple[list[int], int, int]:
        """One token, the next forward queued first; a copied continuation ahead switches to verify windows."""

        import mlx.core as mx

        mode = self._mode.get(stream.stream_id, "pipe")
        if mode in ("verify", "exit"):
            proposal = self._copy_proposal(stream) if mode == "verify" and self.family_width >= 2 else []
            if proposal:
                got, rows, keep = self._family_round(stream, cache, copied=proposal)
                if keep == 1 or stream.force:
                    self._mode[stream.stream_id] = "exit"
                return got, rows, keep
            # nothing to copy: back to steps queued ahead (this one lands next round)
            self._mode[stream.stream_id] = "pipe"
            self._queue_next(stream, cache, mx.array([stream.pending[-1]], dtype=mx.uint32))
            return [], 1, 1
        current = self._inflight.pop(stream.stream_id)
        forced = self._forced_next(stream)
        if forced is not None:
            current = mx.array([forced], dtype=mx.uint32)   # the thinking budget's token, not the sample
        if mode == "drain":
            token = int(current.item())                     # the last queued step: no new one
            self._mode[stream.stream_id] = "verify"
        else:
            self._queue_next(stream, cache, current)        # the GPU starts the next step first
            token = int(current.item())
        stream.rounds += 1
        got = stream.commit([token])
        stream.pending = [token]
        if (self.family_width >= 2 and self._mode.get(stream.stream_id, "pipe") == "pipe" and not stream.finished
                and stream.drafts and self._copy_proposal(stream)):
            self._mode[stream.stream_id] = "drain"          # a copy window is ahead: land the queued step
        return got, 1, 1

    def _family_round(self, stream: Any, cache: list[Any], copied: list[int] | None = None
                      ) -> tuple[list[int], int, int]:
        """The pending token and its drafts in one forward, kept up to the first draft that differs from the
        token sampled there; returns (committed tokens, rows, rows kept)."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        model = self.model
        started = time.perf_counter()
        position = stream.cache_len
        queued = self._next.pop(stream.stream_id, None)
        forced, stream.force = list(stream.force), []
        if copied is None:
            copied = [] if forced or not stream.drafts else self._copy_proposal(stream)
        kind, drafts = "none", []
        if forced:
            kind, drafts = "forced", forced
        elif copied:
            kind, drafts = "copy", copied
        elif stream.drafts and queued is not None:
            drafts = queued
            count = int(drafts.shape[0]) if isinstance(drafts, mx.array) else len(drafts)
            kind = "head" if count else "none"
        pending = mx.array([stream.pending[-1]], dtype=mx.uint32)
        if isinstance(drafts, mx.array):
            inputs = mx.concatenate([pending, drafts.astype(mx.uint32)])
        else:
            inputs = mx.array([stream.pending[-1], *[int(t) for t in drafts]], dtype=mx.uint32)
        rows = int(inputs.shape[0])
        hidden = model.hidden(inputs.reshape(1, rows), cache)
        logits = model.head(hidden)
        logits = logits.reshape(logits.shape[1:])            # [R, V] as a view (MLX's [0] is a gather)
        tokens = gpu_sample(logits, stream.sampling, [position + 1 + r for r in range(rows)])
        speculate = self.family_mtp and stream.drafts and kind != "forced" and self.speculate_early
        parts = [tokens]
        if speculate:
            # the head's first draft for every row, queued behind the verify before anything is read
            parts.append(model.speculate(cache, tokens, position, stream.sampling).astype(tokens.dtype))
        if isinstance(drafts, mx.array):
            parts.append(drafts.astype(tokens.dtype))
        built = time.perf_counter()
        values = [int(t) for t in (mx.concatenate(parts) if len(parts) > 1 else tokens).tolist()]
        read = time.perf_counter()
        sampled = values[:rows]
        firsts = values[rows:2 * rows] if speculate else []
        proposed = values[-(rows - 1):] if isinstance(drafts, mx.array) and rows > 1 else [int(t) for t in drafts]
        keep = 1
        if kind == "forced":
            keep = rows
            sampled = [*forced, sampled[-1]]
        elif proposed:
            for i, draft in enumerate(proposed):
                if sampled[i] != draft:
                    break
                keep += 1
            self.drafted += len(proposed)
            self.accepted += keep - 1
            stream.drafted += len(proposed)
            stream.accepted += keep - 1
            if kind == "copy":
                observe = getattr(stream.proposer, "observe", None)
                if callable(observe):
                    observe(len(proposed), keep - 1)
            elif kind == "head":
                self._observe_depth(stream, len(proposed), keep - 1)
        cut = stream.think_cut(sampled[:keep])
        if cut is not None:
            keep = cut + 1
            sampled = [*sampled[:cut], stream.start_close()]
        stream.rounds += 1
        got = stream.commit(sampled[:keep])
        if stream.finished and len(got) < keep:
            # an end token or the length limit inside the kept drafts: the cache keeps only rows whose tokens
            # landed (row r holds the token after r - 1 committed ones), so a retained cache matches its tokens
            keep = len(got) + 1
        if keep < rows:
            model.keep_rows(cache, rows, keep)
        stream.cache_len += keep
        stream.pending = [sampled[keep - 1]]
        if self.family_mtp and stream.drafts:
            depth = 0 if stream.finished else self._depth(stream)
            if stream.force:
                depth = 0              # the next round verifies the thinking budget's forced tokens
            elif depth and self._copy_proposal(stream):
                depth = 1              # a copied continuation is ahead: one head draft, in case it is gone
            if speculate and cut is None:
                self._next[stream.stream_id] = model.settle(cache, keep, firsts[keep - 1], stream.cache_len + 1,
                                                            stream.sampling, depth)
            else:
                # the head reads the kept rows now, with the tokens that follow them (row r is followed by
                # sampled[r]): late speculation, a forced round, or the budget replaced the kept row's next token
                if speculate:
                    model.unspeculate(cache)
                follow = mx.array(sampled[:keep], dtype=mx.uint32)
                heads = model.speculate(cache, follow, position, stream.sampling, last_only=True)
                self._next[stream.stream_id] = model.settle(cache, keep, heads[-1], stream.cache_len + 1,
                                                            stream.sampling, depth)
            if kind == "head":
                self._observe_cost(len(proposed), (time.perf_counter() - started) * 1e3)
        if _PROFILE:
            self._profile(rows, built - started, read - built, time.perf_counter() - read)
        return got, rows, keep

    def _profile(self, rows: int, build: float, wait: float, after: float) -> None:
        acc = self.__dict__.setdefault("_prof", [0, 0.0, 0.0, 0.0, 0.0])
        acc[0] += 1
        acc[1] += rows
        acc[2] += build * 1e3
        acc[3] += wait * 1e3
        acc[4] += after * 1e3
        if acc[0] >= 200:
            n = acc[0]
            print(f"[lanes] family rounds: {acc[1] / n:.2f} rows, build {acc[2] / n:.2f} ms, wait for the GPU "
                  f"{acc[3] / n:.2f}, after the read {acc[4] / n:.2f} (mean of {n})", flush=True)
            self._prof = [0, 0.0, 0.0, 0.0, 0.0]

    # -- the draft head's depth ----------------------------------------------------------------------------------
    def _depth_rates(self, stream: Any) -> list[float]:
        state = self._depth_state.get(stream.stream_id)
        if state is None:
            state = {"p": [float(p) for p in self.depth_prior[:max(1, self.most_drafts)]], "rounds": 0}
            self._depth_state[stream.stream_id] = state
        return state["p"]

    def _observe_depth(self, stream: Any, proposed: int, accepted: int) -> None:
        """Draft j was tried when drafts 1 .. j - 1 were kept; its estimate moves toward whether it was kept."""

        rates = self._depth_rates(stream)
        for j in range(min(proposed, len(rates))):
            if accepted < j:
                break
            rates[j] += self.depth_rate * ((1.0 if accepted > j else 0.0) - rates[j])

    def _observe_cost(self, drafts: int, ms: float) -> None:
        if drafts <= 0:
            return
        before = self._round_ms.get(drafts)
        self._round_ms[drafts] = ms if before is None else before + self.cost_rate * (ms - before)

    def _round_cost(self, drafts: int) -> float | None:
        """A round's wall time with ``drafts`` head drafts: measured, else the load-time forward cost plus the
        head's steps."""

        if drafts in self._round_ms:
            return self._round_ms[drafts]
        forward = self.family_costs.get(drafts + 1)
        if forward is None:
            return None
        return forward + self.mtp_step_ms * drafts

    def _depth(self, stream: Any) -> int:
        """Head drafts for the next round (1 .. most): the most expected tokens a millisecond at the stream's
        per-depth acceptance, one deeper every ``depth_probe_every`` rounds so the deeper estimate stays current.
        Without measured costs: 1 below 80% first-draft acceptance, 2 below 90%, else 3."""

        # every drafted round verifies at least one draft (a window of 2+ rows), also when the length limit or
        # the thinking budget leaves room for one token: ``commit`` and the budget's cut drop what does not land
        most = min(self.most_drafts, max(1, stream.draft_room - 1))
        if most <= 0:
            return 0
        rates = self._depth_rates(stream)
        if not self.family_costs:
            rate = rates[0]
            return max(1, min(most, 1 if rate < 0.8 else 2 if rate < 0.9 else 3))
        best, best_rate = 1, -1.0
        expected = run = 1.0
        for d in range(1, most + 1):
            cost = self._round_cost(d)
            if cost is None:
                break
            run *= rates[d - 1] if d - 1 < len(rates) else rates[-1]
            expected += run
            if expected / cost > best_rate:
                best, best_rate = d, expected / cost
        state = self._depth_state[stream.stream_id]
        state["rounds"] += 1
        if best < most and state["rounds"] % self.depth_probe_every == 0:
            best += 1
        return best

    # -- engine surface ------------------------------------------------------------------------------------------
    def _family_reset(self) -> None:
        self._live = []
        self._inflight = {}
        self._next = {}
        self._mode = {}

    def _family_summary(self) -> dict[str, Any]:
        return {"engine": "lanes", "family": True, "rounds": len(self.round_stats), "streams": len(self.streams),
                "drafted": self.drafted, "accepted": self.accepted, "exact_width": self.family_width}
