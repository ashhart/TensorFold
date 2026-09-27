"""The lane engine: every model decodes as a lane-engine family (``engine.lane_family``).

A round verifies each live stream's pending token and its drafts in one forward, every row with the bits a one-row
step gives it, keeps the drafts up to the first that differs from the target's own sample and rolls the caches
back to exactly the kept rows; concurrent streams share each round's forward. Every committed token is the model's
own sample at its position, so drafted output is byte-identical to one-token-a-round decoding, which runs through
the same kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Sequence

from tensorfold.engine.lane_family import FamilyRounds


class SuffixLookupProposer:
    """Copy-span proposer: continue the longest earlier occurrence of the suffix.

    Width is earned by evidence (7ee74d3): a proposal is made only when the
    current suffix matches an earlier span of at least ``min_match`` tokens,
    and its length never exceeds the evidence-scaled budget. Wrong proposals
    cost rows, never bytes.
    """

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


# --------------------------------------------------------------------------
# stream state and pure bookkeeping
# --------------------------------------------------------------------------


@dataclass
class LaneStream:
    """One exact stream: its prompt, its commits, and what its cache holds."""

    stream_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    eos_ids: frozenset[int] = frozenset()
    proposer: Any = None
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
    # False for lane fragments nobody resumes: the engine then skips the
    # row extraction it would otherwise do for ``retain_finished_caches``.
    retain: bool = True
    # (tokens, single-row cache copy) pairs captured mid-prefill at boundaries
    # the next turn of the conversation can still match.
    history_checkpoints: list[tuple[list[int], list[Any]]] = field(default_factory=list)
    # ``exact_sampling.Sampling`` (None = greedy): the token at each position is a fixed
    # function of that row's logits and the position, so drafts verify the same way.
    sampling: Any = None
    # False: one token a round, no drafts of any kind (the serial reference drafted output is checked against)
    drafts: bool = True
    # The budget replaces its reply token with think_close; remaining close tokens wait in force.
    think_budget: int = 0
    think_close: tuple[int, ...] = ()
    think_end: int = -1
    think_open: bool = False
    force: list[int] = field(default_factory=list)

    @property
    def context(self) -> list[int]:
        return [*self.prompt_ids, *self.emitted]

    @property
    def budget_left(self) -> int:
        return int(self.max_new_tokens) - len(self.emitted)

    def _budget_active(self) -> bool:
        return self.think_open and self.think_budget > 0 and bool(self.think_close)

    def think_cut(self, tokens: Sequence[int]) -> int | None:
        """The index in ``tokens`` (about to be committed) that the thinking budget replaces by ``think_close[0]``,
        or None: the budget-th reply token, unless the model closed the think block before it."""

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
        self.force = list(self.think_close[1:])
        return int(self.think_close[0])

    @property
    def draft_room(self) -> int:
        """Tokens a round may commit before the length limit or the thinking budget's cut."""

        room = self.budget_left
        if self._budget_active():
            room = min(room, self.think_budget - len(self.emitted))
        return room

    def commit(self, tokens: Sequence[int]) -> list[int]:
        """Append committed tokens until the stream finishes; return what landed."""

        landed: list[int] = []
        for token in tokens:
            if self.finished:
                break
            value = int(token)
            self.emitted.append(value)
            landed.append(value)
            if value == self.think_end:
                self.think_open = False
            if value in self.eos_ids:
                self.finished = True
                self.finish_reason = "stop"
            elif len(self.emitted) >= int(self.max_new_tokens):
                self.finished = True
                self.finish_reason = "length"
        return landed


def sanitize_tree(tokens: Sequence[int], parents: Sequence[int], budget: int) -> tuple[list[int], list[int]]:
    """Truncate a proposal to ``budget`` nodes and drop any node whose parent is missing or comes after it.

    Proposers emit parents before children, so this normally just truncates; a malformed proposal loses the
    orphaned subtrees instead of raising inside the round.
    """

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

    # prompt tokens a prefill forward takes: chunks and checkpoints sit on this grid from position 0 and decoded
    # states are not kept, so a resumed prompt gets a fresh prefill's chunks and bits (0: no grid)
    prefill_step = 2048
    prefill_align = 2048
    # prompt chunks fed so far (every prefill path): the server's stall check counts them as progress
    prefill_chunks = 0

    def __init__(self, model: Any, *, max_rows: int = 128, max_draft: int = 32,
                 retain_finished_caches: bool = False) -> None:
        if not getattr(model, "lane_family", False):
            raise TypeError(f"{type(model).__name__} is not a lane-engine family (engine.lane_family)")
        if max_rows < 1 or max_draft < 0:
            raise ValueError("max_rows >= 1 and max_draft >= 0 required")
        self.model = model
        self.max_rows = int(max_rows)
        self.max_draft = int(max_draft)
        # a finished stream's cache is handed to the caller for the next turn (never on a prefill grid)
        self.retain_finished_caches = bool(retain_finished_caches)
        self.finished_caches: dict[str, tuple[list[int], list[Any]]] = {}
        self.streams: list[LaneStream] = []
        self.round_stats: list[RoundStats] = []
        self.family = True
        self.prefill_guard: Any = None       # the server's cancellation and memory checks between prompt chunks
        self._family_setup()

    # -- prefill and membership -----------------------------------------------------------------------------------
    def prefill_prefix(self, prompt_ids: Sequence[int], *, cache: list[Any] | None = None,
                       cached_tokens: int = 0) -> list[Any]:
        """Absorb ``prompt_ids`` into a new cache (from ``cache`` at ``cached_tokens`` when given) without
        generating; the caller owns the cache."""

        return self._family_prefill_prefix(prompt_ids, cache=cache, cached_tokens=cached_tokens)

    def _align(self) -> int:
        return int(self.prefill_align) if self.prefill_align else 0

    @property
    def prefill_grid(self) -> int:
        """Tokens between the grid points prompts are prefilled and checkpointed on (0: anywhere)."""

        return self._align()

    def on_grid(self, position: int) -> int:
        """The checkpoint a prefill takes for ``position``: the grid point at or before it (itself without a grid)."""

        align = self._align()
        return (int(position) // align) * align if align else int(position)

    def _keeps_decoded(self, stream: LaneStream) -> bool:
        """Whether a finished stream's decoded state is kept for the next turn (never on a prefill grid: its rows
        were decoded, not prefilled)."""

        return self.retain_finished_caches and stream.retain and not self._align()

    @property
    def active_count(self) -> int:
        """Streams holding or about to hold a row."""

        return sum(1 for s, _ in self._live if not s.finished)

    def reset(self) -> None:
        """Drop every live stream's rows after a failed round."""

        self._family_reset()
        self.streams.clear()

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

        self._family_add_stream(stream, cache=cache, cached_tokens=cached_tokens, checkpoints_at=checkpoints_at)

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
        """A batch-1 cache list to retain: whole arrays shared (MLX never writes a shared buffer), a view (a state
        sliced from a call's per-row states) copied out, so it does not keep its whole base buffer alive."""

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

    # -- rounds ---------------------------------------------------------------------------------------------------
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
