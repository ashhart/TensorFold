"""Verify each draft against the target sample and roll caches back to accepted rows so shared rounds match serial decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterator, Sequence

from tensorfold.engine.family_prefill import drain
from tensorfold.engine.lane_family import FamilyRounds
from tensorfold.engine.prefill_plan import PrefillPlan, PromptChunks


class SuffixLookupProposer:
    """Propose continuations only with enough matching context, limiting width by evidence so rejected drafts cost rows without changing output."""

    name = "suffix-lookup"

    def __init__(
        self,
        *,
        ngram: int = 3,
        min_match: int = 6,
        max_extension: int = 64,
        silence_rounds: int = 16,
        window: int = 4,
    ) -> None:
        if ngram < 1:
            raise ValueError("ngram must be positive")
        if min_match < ngram:
            raise ValueError("min_match must be at least ngram")
        self.ngram = int(ngram)
        self.min_match = int(min_match)
        self.max_extension = int(max_extension)
        # Pause after repeated rejections because short n-gram matches can be coincidental.
        self.silence_rounds = int(silence_rounds)
        self.window = max(1, int(window))
        self._recent: list[int] = []
        self._silent_for = 0
        self._index: dict[tuple[int, ...], list[int]] = {}
        self._indexed = 0
        self.proposals = 0
        self.proposed_tokens = 0
        self.accepted_tokens = 0
        self.silenced_rounds = 0

    def _extend_index(self, context: Sequence[int]) -> None:
        n = self.ngram
        start = max(self._indexed, n - 1)
        for position in range(start, len(context)):
            key = tuple(int(t) for t in context[position - n + 1 : position + 1])
            self._index.setdefault(key, []).append(position)
        self._indexed = len(context)

    def _match_length(self, context: Sequence[int], end: int) -> int:
        """Tokens matching backwards from ``end`` (exclusive) vs the context tail."""

        length = 0
        limit = min(self.max_extension, end)
        while length < limit and context[end - 1 - length] == context[len(context) - 1 - length]:
            length += 1
            if len(context) - 1 - length < 0:
                break
        return length

    def propose(self, context: Sequence[int], max_draft: int) -> list[int]:
        if max_draft <= 0 or len(context) < self.ngram + 1:
            return []
        if self._silent_for > 0:
            self._silent_for -= 1
            self.silenced_rounds += 1
            return []
        if self._indexed > len(context) or (
            self._indexed and tuple(context[self._indexed - self.ngram : self._indexed])
            != tuple(self._last_key)
        ):
            # Context changed underneath the index (new request): rebuild.
            self._index = {}
            self._indexed = 0
        self._extend_index(context)
        self._last_key = tuple(int(t) for t in context[self._indexed - self.ngram : self._indexed])
        key = tuple(int(t) for t in context[-self.ngram :])
        positions = self._index.get(key)
        if not positions:
            return []
        best_end = -1
        best_len = 0
        # Most recent first; longest evidence wins, ties to the most recent.
        for position in reversed(positions):
            end = position + 1
            if end >= len(context):
                continue
            length = self._match_length(context, end)
            if length > best_len:
                best_len = length
                best_end = end
                if length >= self.max_extension:
                    break
        self.last_confident = False
        self.last_match = best_len
        if best_end < 0 or best_len < self.min_match:
            return []
        self.last_confident = best_len >= self.confident_match
        proposal = [int(t) for t in context[best_end : best_end + max_draft]]
        if proposal:
            self.proposals += 1
            self.proposed_tokens += len(proposal)
        return proposal

    _last_key: tuple[int, ...] = ()
    # A proposal backed by this many matching tokens may use a wide window.
    confident_match = 24
    last_confident = False
    last_match = 0          # matching tokens behind the last proposal

    def observe(self, proposed: int, accepted: int) -> None:
        self.judged_tokens += int(proposed)
        self.accepted_tokens += int(accepted)
        if proposed > 0:
            self._recent.append(int(accepted))
            del self._recent[: -self.window]
            if len(self._recent) >= self.window and max(self._recent) == 0:
                self._silent_for = self.silence_rounds
                self._recent = []

    judged_tokens: int = 0

    def telemetry(self) -> dict[str, Any]:
        return {
            "proposals": self.proposals,
            "proposed_tokens": self.proposed_tokens,
            "judged_tokens": self.judged_tokens,
            "accepted_tokens": self.accepted_tokens,
            "silenced_rounds": self.silenced_rounds,
        }

@dataclass
class LaneStream:
    """One exact stream: its prompt, its commits, and what its cache holds."""

    stream_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    eos_ids: frozenset[int] = frozenset()
    proposer: Any = None
    prompt_data: Any = None
    emitted: list[int] = field(default_factory=list)
    pending: list[int] = field(default_factory=list)
    cache_len: int = 0
    finished: bool = False
    finish_reason: str = ""
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    min_rows: int = 0           # the narrowest verify window of the stream's rounds so far (0: none yet)
    cached_tokens: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    # Disable retention for lane fragments that will not resume to skip cache-row extraction.
    retain: bool = True
    # Capture (tokens, single-row cache copy) at prefill boundaries the next turn can match.
    history_checkpoints: list[tuple[list[int], list[Any]]] = field(default_factory=list)
    # plan chunks each prefill forward took (1s: a chunk a forward; more: a prompt pass)
    prefill_widths: list[int] = field(default_factory=list)
    # the forwards whose freed buffers MLX kept in the raised pass cache (the budget had room for it)
    prefill_raised: list[bool] = field(default_factory=list)
    # Key sampling by each row's logits and position so drafts verify identically; None means greedy.
    sampling: Any = None
    # False: one token a round, no drafts of any kind (the serial reference drafted output is checked against)
    drafts: bool = True
    # The budget replaces its reply token with think_close; remaining close tokens wait in force.
    think_budget: int = 0
    think_close: tuple[int, ...] = ()
    think_end: int = -1
    think_open: bool = False
    force: list[int] = field(default_factory=list)
    stop_check: Callable[[list[int]], bool] | None = None
    # a request that must call a tool: its answer opens a call to an offered tool (call_gate.CallGate)
    call_gate: Any = None
    # response_format's grammar (engine.grammar.Constraint): follows every committed token, masks each drawn row
    constraint: Any = None
    error: Any = None           # why the stream ended with finish_reason "error" (its grammar failed)

    @property
    def context(self) -> list[int]:
        return [*self.prompt_ids, *self.emitted]

    @property
    def budget_left(self) -> int:
        return int(self.max_new_tokens) - len(self.emitted)

    def _budget_active(self) -> bool:
        return self.think_open and self.think_budget > 0 and bool(self.think_close)

    def think_cut(self, tokens: Sequence[int]) -> int | None:
        """Return the index the thinking budget replaces with ``think_close[0]``, or None if the model already closed the think block."""

        if not self._budget_active():
            return None
        for i, token in enumerate(tokens):
            if len(self.emitted) + i + 1 >= self.think_budget:
                return i
            if int(token) == self.think_end:
                return None
        return None

    def start_close(self) -> int:
        """Begin the thinking budget's close: returns its first token; the rest wait in ``force``."""

        self.think_open = False
        close = self.think_close
        if self.constraint is not None and self.think_end in close:    # a grammar takes the reply from </think> on
            close = close[:close.index(self.think_end) + 1]
        self.force = list(close[1:])
        return int(close[0])

    @property
    def draft_room(self) -> int:
        """Tokens a round may commit before the length limit or the thinking budget's cut."""

        room = self.budget_left
        if self._budget_active():
            room = min(room, self.think_budget - len(self.emitted))
        return room

    def fail(self, error: BaseException) -> None:
        """End the stream with ``error`` (its request answers with it); the round and the other streams go on."""

        self.finished, self.finish_reason, self.error = True, "error", error

    def commit(self, tokens: Sequence[int]) -> list[int]:
        """Append committed tokens until the stream finishes; return what landed."""

        landed: list[int] = []
        for token in tokens:
            if self.finished:
                break
            value = int(token)
            if self.constraint is not None:
                try:
                    self.constraint.advance([value])
                except Exception as exc:    # noqa: BLE001  the grammar failed: this reply ends, other streams go on
                    self.fail(exc)
                    break
            self.emitted.append(value)
            landed.append(value)
            if value == self.think_end:
                self.think_open = False
            if self.call_gate is not None:
                self.call_gate.observe(value)
            if value in self.eos_ids or (self.stop_check is not None and self.stop_check(self.emitted)):
                self.finished = True
                self.finish_reason = "stop"
            elif len(self.emitted) >= int(self.max_new_tokens):
                self.finished = True
                self.finish_reason = "length"
        return landed


def sanitize_tree(tokens: Sequence[int], parents: Sequence[int], budget: int) -> tuple[list[int], list[int]]:
    """Truncate to the node budget and discard orphaned subtrees or nodes whose parents do not precede them."""

    kept: dict[int, int] = {}
    out_t: list[int] = []
    out_p: list[int] = []
    for i, (t, q) in enumerate(zip(list(tokens)[:budget], list(parents)[:budget])):
        q = int(q)
        if q >= 0 and q not in kept:
            continue
        kept[i] = len(out_t)
        out_t.append(int(t))
        out_p.append(-1 if q < 0 else kept[q])
    return out_t, out_p


@dataclass
class RoundStats:
    streams: int
    width: int
    rows: int
    ragged: bool
    rollbacks: int
    committed: int
    forward_ms: float
    finalize_ms: float
    rollback_ms: float
    total_ms: float
    draft_ms: float = 0.0      # proposer time inside the round (precise rounds)
    post_ms: float = 0.0       # predictions to host, verification, proposer bookkeeping
    started_at: float = 0.0    # perf_counter at the round's start (gaps between rounds)


class LaneEngine(FamilyRounds):
    """Exact streams of a family model, verified in shared rounds."""

    # where prompt chunks start (None: anywhere, in ``prefill_step`` chunks, decoded states kept)
    prefill_plan: Any = PrefillPlan(2048)
    prefill_step = 2048
    # prompt chunks fed so far (every prefill path): the server's stall check counts them as progress
    prefill_chunks = 0

    def __init__(self, model: Any, *, max_rows: int = 128, max_draft: int = 32,
                 retain_finished_caches: bool = False, prefill_plan: Any = None,
                 prefill_pass: int | None = None, pass_cache: int | None = None) -> None:
        if not getattr(model, "lane_family", False):
            raise TypeError(f"{type(model).__name__} is not a lane-engine family (engine.lane_family)")
        if max_rows < 1 or max_draft < 0:
            raise ValueError("max_rows >= 1 and max_draft >= 0 required")
        self.model = model
        self.max_rows = int(max_rows)
        self.max_draft = int(max_draft)
        # a finished stream's cache is handed to the caller for the next turn (never under a prefill plan)
        self.retain_finished_caches = bool(retain_finished_caches)
        if prefill_plan is not None:
            self.prefill_plan = prefill_plan
        if prefill_pass is not None:
            self.prefill_pass = max(1, int(prefill_pass))
        if pass_cache is not None:
            self.pass_cache = max(0, int(pass_cache))
        self.finished_caches: dict[str, tuple[list[int], list[Any]]] = {}
        self.streams: list[LaneStream] = []
        self.round_stats: list[RoundStats] = []
        self.family = True
        self.prefill_guard: Any = None       # the server's cancellation and memory checks between prompt chunks
        self._family_setup()

    def prefill_prefix(self, prompt_ids: Sequence[int], *, cache: list[Any] | None = None,
                       cached_tokens: int = 0) -> list[Any]:
        """Prefill a caller-owned cache without generating, resuming at ``cached_tokens`` when given."""

        return self._family_prefill_prefix(prompt_ids, cache=cache, cached_tokens=cached_tokens)

    def prompt_chunks(self, prompt_ids: Sequence[int]) -> PromptChunks:
        """Where this prompt's prefill chunks start: the positions a stored state resumes it from."""

        if self.prefill_plan is None:
            return PromptChunks(None, len(prompt_ids), step=int(self.prefill_step))
        return self.prefill_plan.chunks(prompt_ids)

    def _keeps_decoded(self, stream: LaneStream) -> bool:
        """Whether a finished stream's decoded state is kept for the next turn (never under a plan: decoded rows)."""

        return self.retain_finished_caches and stream.retain and self.prefill_plan is None

    @property
    def active_count(self) -> int:
        """Streams holding or about to hold a row."""

        return sum(1 for s, _ in self._live if not s.finished)

    def reset(self) -> None:
        """Drop every live stream's rows after a failed round."""

        self._family_reset()
        self.streams.clear()

    def release_rounds(self) -> None:
        """With no stream live, let the family drop what its last forward keeps for rolling rows back."""

        release = getattr(self.model, "release_rounds", None)
        if release is not None and not self.active_count:
            release()

    def discard_stream(self, stream: LaneStream) -> None:
        """Release a cancelled stream between rounds: no retained cache, no pending draws or drafts."""

        stream.finished, stream.finish_reason, stream.finished_at = True, "cancelled", time.perf_counter()
        self._live[:] = [(s, cache) for s, cache in self._live if s is not stream]
        self._release_stream_state(stream.stream_id)
        self.finished_caches.pop(stream.stream_id, None)
        if stream in self.streams:
            self.streams.remove(stream)

    def add_stream(self, stream: LaneStream, *, cache: list[Any] | None = None, cached_tokens: int = 0,
                   checkpoints_at: Sequence[int] = ()) -> None:
        """Prefill a stream (from ``cache`` at ``cached_tokens`` when given); it takes part from the next round."""

        drain(self.begin_stream(stream, cache=cache, cached_tokens=cached_tokens, checkpoints_at=checkpoints_at))

    def begin_stream(self, stream: LaneStream, *, cache: list[Any] | None = None, cached_tokens: int = 0,
                     checkpoints_at: Sequence[int] = ()) -> Iterator[None]:
        """``add_stream`` a prompt chunk a step (each ``next`` feeds one); rounds may run between the steps."""

        return self._family_add_stream(stream, cache=cache, cached_tokens=cached_tokens, checkpoints_at=checkpoints_at)

    @staticmethod
    def cache_nbytes(cache: list[Any]) -> int:
        """Bytes held by a cache list's arrays (KV timelines plus GDN state)."""

        total = 0
        for item in cache:
            state = getattr(item, "state", None)
            values = state if isinstance(state, (list, tuple)) else [state]
            for value in values:
                nbytes = getattr(value, "nbytes", None)
                if isinstance(nbytes, int):
                    total += nbytes
        return total

    @staticmethod
    def copy_single_cache(cache: list[Any]) -> list[Any]:
        """Share whole arrays because MLX never writes shared buffers; copy views so retained caches release unused base rows."""

        import copy as _copy

        import mlx.core as mx

        out: list[Any] = []
        copies: list[Any] = []

        def own(value: Any) -> Any:
            if isinstance(value, mx.array):
                value = mx.contiguous(value)       # a whole array's buffer is shared, a view's elements copied
                copies.append(value)
            return value

        for item in cache:
            materialize = getattr(item, "materialize", None)
            if materialize is not None:
                materialize()                      # a lazily held state becomes arrays before it is copied
            clone = _copy.copy(item)
            for key, value in vars(item).items():
                if isinstance(value, mx.array):
                    setattr(clone, key, own(value))
                elif isinstance(value, list):
                    setattr(clone, key, [own(v) for v in value])
            out.append(clone)
        if copies:
            mx.async_eval(*copies)
        return out

    def step(self) -> dict[str, list[int]]:
        """One round for every live stream. Returns newly committed tokens per stream."""

        return self._family_step()

    def run(self, on_tokens: Callable[[str, list[int]], None] | None = None) -> None:
        while self.active_count:
            landed = self.step()
            if on_tokens is not None:
                for stream_id, tokens in landed.items():
                    if tokens:
                        on_tokens(stream_id, tokens)

    def summary(self) -> dict[str, Any]:
        return self._family_summary()
