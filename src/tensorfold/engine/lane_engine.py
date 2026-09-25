"""The lane engine: exact streams verified in wide passes (Qwen3.8 dense and other mlx_lm models).

A decode step of a large dense model streams all its weights once; a pass over many rows costs only a few
times one row (Qwen3.8-27B, 8-bit, M5 Max: 55.8 ms for 1 row, 148 ms for 64, 206 ms for 128). The rows of one
pass can hold the near future of one stream (drafts), alternative futures (a draft tree), or rows of other
streams. Every stream keeps its own window (pending refeed + drafts) and all windows share one right-padded
forward.

The protocol is mlx_lm's own batch-cache protocol:
  1. each stream is prefilled alone on a batch-1 cache (the path serial decoding takes) and merged into the
     batch (BatchKVCache.merge, ArraysCache.merge);
  2. each round every stream offers window_i = pending_i + drafts_i; rows are right-padded to the widest;
     caches get prepare(lengths, right_padding) before the forward and finalize() after it;
  3. each row is verified against its own sampled outputs (exact_sampling: the token at a position is a fixed
     function of its logits, the seed and the position);
  4. a fully accepted row keeps its advanced cache; a partly accepted row is rolled back (KV: the rejected
     tail becomes padding; recurrent state: restored from the pre-round arrays, or with one stream rolled back
     to exactly the kept prefix) and its accepted tokens become pending;
  5. finished rows leave through filter(), new rows join through extend().

Every committed token is the model's own sample for its row. With the lane kernels (``kernels.lane_qmm``,
``kernels.lane_attention``; GPUs with tensor units) a row of any window gets the bits a one-row step gives it,
so drafted output is byte-identical to one-token-a-round decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any, Callable, Sequence


# --------------------------------------------------------------------------
# proposers
# --------------------------------------------------------------------------


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
        # Prose repeats short n-grams by coincidence; a run of rejected
        # proposals means the text is novel, so stop offering for a while
        # (measured: junk spans halved a single stream to 7 tok/s).
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
    full_rounds: int = 0
    partial_rounds: int = 0
    refeed_tokens: int = 0
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
    # Thinking budget (0: none): while the reply is inside the prompt's open think block, its ``think_budget``-th
    # token is ``think_close[0]`` whatever the model samples there, and the rest of ``think_close``
    # ("\n</think>\n\n") follows unsampled. ``think_end``: the "</think>" id, which closes the block when the
    # model writes it first. The cut depends only on the token count, so drafted and serial decoding cut at the
    # same place. The serial engine feeds the close itself (``force``, ``start_close``); for the lane engine
    # ``commit`` puts it in place of a round's sampled token, one token a round (drafts stop short of it:
    # ``draft_room``).
    think_budget: int = 0
    think_close: tuple[int, ...] = ()
    think_end: int = -1
    think_open: bool = False
    force: list[int] = field(default_factory=list)
    forced_in_commit: list[int] = field(default_factory=list)
    _substituted: int | None = None

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
        """Tokens a round may commit: the length limit, and the thinking budget's forced tokens, which drafts
        never reach (only a round's own sample lands there, and ``commit`` replaces it)."""

        room = self.budget_left
        if self.forced_in_commit:
            return min(room, 1)
        if self._budget_active():
            room = min(room, self.think_budget - len(self.emitted))
        return room

    def last_bonus(self, bonus: int) -> int:
        """The token the last commit put where the round sampled ``bonus`` (the thinking budget's), else bonus."""

        return int(bonus) if self._substituted is None else self._substituted

    def commit(self, tokens: Sequence[int]) -> list[int]:
        """Append committed tokens until the stream finishes; return what landed."""

        landed: list[int] = []
        self._substituted = None
        for token in tokens:
            if self.finished:
                break
            value = int(token)
            if self.forced_in_commit or (self._budget_active() and len(self.emitted) + 1 >= self.think_budget):
                # the thinking budget's close in place of the sample (lane engine): nothing after it lands
                if not self.forced_in_commit:
                    self.think_open = False
                    self.forced_in_commit = list(self.think_close)
                value = self.forced_in_commit.pop(0)
                self._substituted = value
                self.emitted.append(value)
                landed.append(value)
                if len(self.emitted) >= int(self.max_new_tokens):
                    self.finished = True
                    self.finish_reason = "length"
                break
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


def plan_windows(
    pendings: Sequence[Sequence[int]],
    drafts: Sequence[Sequence[int]],
    *,
    max_rows: int,
) -> tuple[list[list[int]], int]:
    """Trim drafts so ``B * Wmax`` stays inside the row budget.

    Pending tokens are committed truth awaiting absorption and are never
    trimmed; only drafts are. Returns the windows and their padded width.
    """

    if len(pendings) != len(drafts):
        raise ValueError("pendings and drafts must align")
    if not pendings:
        return [], 0
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    per_row = max(1, int(max_rows) // len(pendings))
    windows: list[list[int]] = []
    for pending, draft in zip(pendings, drafts):
        if not pending:
            raise ValueError("every active stream must hold at least one pending token")
        room = max(0, per_row - len(pending))
        windows.append([int(t) for t in pending] + [int(t) for t in draft[:room]])
    return windows, max(len(w) for w in windows)


def verify_window(window: Sequence[int], n_pending: int, preds: Sequence[int]) -> tuple[int, int]:
    """Accepted draft count and the bonus token for one row.

    ``preds[j]`` is the target argmax after consuming ``window[:j+1]``. The
    drafts start at ``window[n_pending]``; the prediction that judges draft
    ``j`` sits at ``n_pending - 1 + j``. The bonus is the target's own token
    at the first mismatch (or after the last accepted draft).
    """

    if n_pending < 1 or n_pending > len(window):
        raise ValueError("n_pending must lie in [1, len(window)]")
    if len(preds) < len(window):
        raise ValueError("preds must cover the whole window")
    accepted = 0
    for j, draft in enumerate(window[n_pending:]):
        if int(preds[n_pending - 1 + j]) != int(draft):
            break
        accepted += 1
    return accepted, int(preds[n_pending - 1 + accepted])


def apply_round(
    stream: LaneStream,
    window: Sequence[int],
    preds: Sequence[int],
) -> tuple[list[int], int]:
    """Commit one row's outcome into its stream. Returns (landed, rollback).

    ``rollback`` is the number of window positions the row's cache must give
    back: 0 when every draft was accepted (the cache absorbed the window),
    otherwise the whole window, whose accepted part becomes pending.
    """

    n_pending = len(stream.pending)
    drafts = list(window[n_pending:])
    accepted, bonus = verify_window(window, n_pending, preds)
    accepted_drafts = drafts[:accepted]
    stream.rounds += 1
    stream.drafted += len(drafts)
    stream.accepted += accepted
    observe = getattr(stream.proposer, "observe", None)
    if callable(observe) and drafts:
        observe(len(drafts), accepted)
    landed = stream.commit([*accepted_drafts, bonus])
    if accepted == len(drafts):
        stream.full_rounds += 1
        stream.cache_len += len(window)
        stream.pending = [stream.last_bonus(bonus)]
        return landed, 0
    stream.partial_rounds += 1
    stream.pending = [*stream.pending, *accepted_drafts, stream.last_bonus(bonus)]
    stream.refeed_tokens += len(stream.pending)
    return landed, len(window)


def apply_round_precise(
    stream: LaneStream,
    window: Sequence[int],
    preds: Sequence[int],
) -> tuple[list[int], int]:
    """Commit one row's outcome for a cache rolled back to exactly its kept prefix.

    Returns (landed, keep): ``keep`` is how many window positions stay in the cache,
    the pending tokens plus the accepted drafts that landed (none past an EOS or the
    token limit). The bonus token becomes the only pending token; nothing is re-fed.
    """

    n_pending = len(stream.pending)
    drafts = list(window[n_pending:])
    accepted, bonus = verify_window(window, n_pending, preds)
    stream.rounds += 1
    stream.drafted += len(drafts)
    stream.accepted += accepted
    observe = getattr(stream.proposer, "observe", None)
    if callable(observe) and drafts:
        observe(len(drafts), accepted)
    landed = stream.commit([*drafts[:accepted], bonus])
    keep = n_pending + min(accepted, len(landed))
    if accepted == len(drafts):
        stream.full_rounds += 1
    else:
        stream.partial_rounds += 1
    stream.cache_len += keep
    stream.pending = [stream.last_bonus(bonus)]
    return landed, keep


# --------------------------------------------------------------------------
# the engine (MLX only inside methods, like the rest of the package)
# --------------------------------------------------------------------------


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


# TF_ROUND_LOG=path appends one line per tree round: start width accepted kind forward_ms draft_ms [ab]
_ROUND_LOG = os.environ.get("TF_ROUND_LOG", "")
# TF_KERNEL_AB=N: flip lane_qmm.AB_FLAG every N rounds (bit-identical kernel variants only), logged as the last field
_KERNEL_AB = int(os.environ.get("TF_KERNEL_AB", "0") or 0)
_PROFILE = os.environ.get("TF_ROUND_PROFILE", "") == "1"   # diagnostic syncs: commit and prefill timed apart
# TF_SHADOW_STATS=1 (measurement only, outputs unchanged): how often the target's own samples on the
# drafted path past the first rejected token (thrown away today) name the tokens that come after
_SHADOW = os.environ.get("TF_SHADOW_STATS", "") == "1"


class LaneEngine:
    """Drive N exact streams through shared wide verify rounds."""

    # Rows a draft window may use without strong evidence (see ``_drafts_for``).
    cheap_window = 4
    # Hard cap on a window's rows so verification stays bit-identical to serial
    # decoding (0 = no cap). Set by a family (``families.qwen3_5.LANE_SETTINGS``).
    exact_window = 0
    # Layers per slice of the forward handed to the GPU while the next slice is built
    # (0 = build the whole graph, then evaluate). Same graph, same bits.
    pipeline_layers = 4
    # Prompt tokens per prefill forward (mlx_lm's default step).
    prefill_step = 2048
    # Prefill through the lane decoder (``lane_tree.tree_forward`` chains of ``lane_prefill``
    # tokens) instead of MLX's chunked kernels: every prompt row then gets the bits serial
    # decoding would give it, whatever the chunking or cached prefix (0 = MLX's prefill).
    lane_prefill = 0
    # Draft-tree rounds for one stream (``lane_tree``): up to this many drafted nodes per
    # round, verified in one forward (0 = chains only).
    tree_nodes = 0
    # A tree round's budget for a chain proposal (a verbatim copy, a tool call's known
    # structure) when above ``tree_nodes``: 32 rows cost ~1.4x 16 and a long copy fills them.
    chain_nodes = 0
    # Pass one unpadded stream's attention a plain causal mask (``lane_attention``).
    simple_masks = False
    # One live stream: roll a partly accepted window back to exactly its kept prefix
    # (``gdn_capture``) instead of restoring the whole window and re-feeding it, and
    # report the kept rows to the proposer (DFlash2 reads their hidden states).
    precise_single = False

    PAD_TOKEN = 0

    def __init__(
        self,
        model: Any,
        *,
        max_rows: int = 128,
        max_draft: int = 32,
        pending_cap: int = 32,
        copy_recurrent_snapshots: bool = False,
        retain_finished_caches: bool = False,
        shrink_slack: int = 0,
        one_token: bool = False,
    ) -> None:
        if max_rows < 1 or max_draft < 0 or pending_cap < 2 or shrink_slack < 0:
            raise ValueError(
                "max_rows >= 1, max_draft >= 0, pending_cap >= 2, shrink_slack >= 0 required")
        self.model = model
        self.max_rows = int(max_rows)
        self.shadow_stats: dict[int, list[int]] = {}      # TF_SHADOW_STATS: offset -> [hits, seen]
        self.max_draft = int(max_draft)
        self.pending_cap = int(pending_cap)
        self.copy_recurrent_snapshots = bool(copy_recurrent_snapshots)
        # When set, a finishing row is rolled back to exactly its absorbed
        # prefix and its single-row cache is handed to the caller, so the next
        # turn of that conversation can prefill only its new suffix.
        self.retain_finished_caches = bool(retain_finished_caches)
        # Finished rows tolerated in the batch before it is filtered. Every
        # filter re-allocates each cache array at a new size (about 550 ms at
        # 64 rows on the M5 Max, 2026-09-19) while round cost is flat from 16
        # to 64 rows, so a fan-out caller lets rows idle and sheds them in
        # groups, or together with the next join. 0 keeps the old behaviour.
        self.shrink_slack = int(shrink_slack)
        self.proposer_errors = 0
        self.prefill_profile: list[tuple[int, float, float]] = []
        # Every forward is one token on each live lane. A wide suffix is walked
        # a token at a time instead of padded into one [lanes, width] block.
        self.one_token = bool(one_token)
        self.finished_caches: dict[str, tuple[list[int], list[Any]]] = {}
        self.streams: list[LaneStream] = []
        self._active: list[LaneStream] = []
        self._batch: list[Any] | None = None
        self.round_stats: list[RoundStats] = []
        self._stack: tuple[Any, Any, Callable[[Any], Any]] | None = None
        self._pending_join: list[tuple[LaneStream, list[Any]]] = []

    # -- model plumbing --------------------------------------------------
    def _resolve_stack(self) -> tuple[Any, Any, Callable[[Any], Any]]:
        if self._stack is None:
            language_model = getattr(self.model, "language_model", self.model)
            core = getattr(language_model, "model", None)
            if core is None:
                raise TypeError("model has no language-model core (.model)")
            args = getattr(language_model, "args", None)
            if args is not None and getattr(args, "tie_word_embeddings", False):
                head = core.embed_tokens.as_linear
            else:
                head = language_model.lm_head
            self._stack = (language_model, core, head)
        return self._stack

    def _layer_masks(self, hidden: Any, cache: list[Any]) -> tuple[Any, Any]:
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        _, core, _ = self._resolve_stack()
        fa_idx = getattr(core, "fa_idx", None)
        ssm_idx = getattr(core, "ssm_idx", None)
        fa_mask = create_attention_mask(hidden, cache[fa_idx]) if fa_idx is not None else None
        ssm_mask = create_ssm_mask(hidden, cache[ssm_idx]) if ssm_idx is not None else None
        return fa_mask, ssm_mask

    def _forward_rows(self, rows: list[list[int]], cache: list[Any], *,
                      simple_mask: bool = False, logits: bool = False) -> tuple[Any, Any]:
        """One [B, W] forward returning (preds [B, W] int, normed hidden).

        ``simple_mask``: one unpadded row, so attention gets MLX's plain causal mask
        (None for one token) instead of an explicit mask array; the lane attention
        only takes the plain form.
        """

        import mlx.core as mx

        _, core, head = self._resolve_stack()
        hidden = core.embed_tokens(mx.array(rows, dtype=mx.uint32))
        fa_mask, ssm_mask = self._layer_masks(hidden, cache)
        if simple_mask:
            fa_mask = "causal" if len(rows[0]) > 1 else None
        layers = list(core.layers)
        for index, (layer, item) in enumerate(zip(layers, cache)):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            hidden = layer(hidden, mask=mask, cache=item)
            if self.pipeline_layers and (index + 1) % self.pipeline_layers == 0 and index + 1 < len(layers):
                # start the GPU on the layers built so far while Python builds the rest
                # (building a 64-layer graph takes ~8 ms; the GPU idles through it otherwise)
                mx.async_eval(hidden)
        normed = core.norm(hidden)
        if logits:
            return head(normed), normed
        preds = mx.argmax(head(normed), axis=-1)
        return preds, normed

    def _predict_rows(self, streams: Sequence[LaneStream | None], logits: Any) -> list[list[int]]:
        """Each row's predictions: argmax, or its stream's exact sampling at absolute positions.

        Window row j of a stream holds the token at position cache_len + j, so its
        prediction is the token at position cache_len + j + 1 (as in serial decoding).
        """

        import mlx.core as mx

        from tensorfold.engine.exact_sampling import sample_rows

        greedy = self._to_rows(mx.argmax(logits, axis=-1))
        out: list[list[int]] = []
        for i, stream in enumerate(streams):
            if stream is None or stream.sampling is None:
                out.append(greedy[i])
                continue
            width = int(logits.shape[1])
            positions = [int(stream.cache_len) + j + 1 for j in range(width)]
            out.append(sample_rows(logits[i], positions, stream.sampling))
        return out

    @staticmethod
    def _is_kv(item: Any) -> bool:
        return hasattr(item, "keys") and hasattr(item, "values")

    def _cache_arrays(self, cache: list[Any]) -> list[Any]:
        arrays: list[Any] = []
        for item in cache:
            state = item.state
            if isinstance(state, (list, tuple)):
                arrays.extend(a for a in state if a is not None and hasattr(a, "shape"))
            elif state is not None and hasattr(state, "shape"):
                arrays.append(state)
        return arrays

    # -- prefill and membership -------------------------------------------
    def prefill(
        self,
        stream: LaneStream,
        *,
        cache: list[Any] | None = None,
        cached_tokens: int = 0,
        checkpoints_at: Sequence[int] = (),
    ) -> list[Any]:
        """Batch-1 prefill on the model's own cache type; returns the cache.

        The prompt (or, given a ``cache`` that already holds its first
        ``cached_tokens`` tokens, only the new suffix) is fed and the first
        token is the argmax at the last prompt position. At each boundary in
        ``checkpoints_at`` the prefill pauses and stores a copy of the cache
        in ``stream.history_checkpoints``: the generation prompt's tail
        (assistant header, empty think block) does not survive re-rendering
        as history, the tokens before it do. The caller owns ``cache``; it is
        consumed.
        """

        import mlx.core as mx

        if not stream.prompt_ids:
            raise ValueError(f"{stream.stream_id}: empty prompt")
        work: list[Any]
        if cache is None:
            work = self.model.make_cache()
            cached_tokens = 0
        else:
            work = cache
        start = int(cached_tokens)
        if not 0 <= start < len(stream.prompt_ids):
            raise ValueError(f"{stream.stream_id}: cached_tokens must leave a suffix to prefill")
        stream.history_checkpoints = []
        self._route_hidden(stream)
        # DFlash taps of every prefill segment, not just the last: a checkpoint boundary splits the
        # suffix, and the drafter's first blocks saw only the part after the last boundary (a fresh
        # agent session: the 5-token assistant header, not the user's message; 2026-09-23)
        taps_store = getattr(self.model, "_hidden_states", None)
        carried: list[Any] | None = None
        for boundary in sorted({int(b) for b in checkpoints_at}):
            if not start < boundary < len(stream.prompt_ids):
                continue
            head = stream.prompt_ids[start:boundary]
            self._prefill_tokens(head, work)
            if taps_store and all(t is not None for t in taps_store):
                carried = [t if carried is None else mx.concatenate([carried[i], t], axis=1)[:, -2048:]
                           for i, t in enumerate(taps_store)]
            stream.history_checkpoints.append(
                (list(stream.prompt_ids[:boundary]), self.copy_single_cache(work))
            )
            start = boundary
        suffix = stream.prompt_ids[start:]
        logits = self._prefill_tokens(suffix, work)
        self._route_hidden(None)
        if carried is not None and taps_store and all(t is not None for t in taps_store):
            for i, tap in enumerate(taps_store):
                taps_store[i] = mx.concatenate([carried[i], tap], axis=1)[:, -2048:]
        if stream.sampling is not None:
            from tensorfold.engine.exact_sampling import sample_rows

            first = sample_rows(logits[0, -1:, :], [len(stream.prompt_ids)], stream.sampling)[0]
        else:
            first = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        on_prefill = getattr(stream.proposer, "on_prefill", None)
        if callable(on_prefill):
            on_prefill(len(stream.prompt_ids))
        stream.emitted = []
        stream.pending = []
        stream.cache_len = len(stream.prompt_ids)
        stream.cached_tokens = int(cached_tokens)
        stream.started_at = time.perf_counter()
        stream.commit([first])
        stream.pending = [first]
        return work

    def prefill_prefix(
        self,
        prompt_ids: Sequence[int],
        *,
        cache: list[Any] | None = None,
        cached_tokens: int = 0,
    ) -> list[Any]:
        """Absorb ``prompt_ids`` into a batch-1 cache without generating.

        This is the fork point for K lanes that share a prompt prefix: one
        prefill here, ``copy_single_cache`` K times, then each lane prefills
        only its own short suffix. The caller owns the returned cache.
        """


        if not prompt_ids:
            raise ValueError("empty prefix")
        work: list[Any] = self.model.make_cache() if cache is None else cache
        start = int(cached_tokens) if cache is not None else 0
        if not 0 <= start < len(prompt_ids):
            raise ValueError("cached_tokens must leave a suffix to prefill")
        self._prefill_tokens(list(prompt_ids[start:]), work)
        return work

    @staticmethod
    def _route_hidden(stream: Any) -> None:
        """Point the forward's hidden-state feed at this stream's own proposer (one that drafts from the target's
        hidden states exposes a ``sink``), or at nothing. One global feed let a second request's proposer take it
        from a stream still decoding, whose proposer then drafted nothing for the rest of its request."""
        from tensorfold.kernels import lane_tree

        lane_tree.HIDDEN_SINK = getattr(getattr(stream, "proposer", None), "sink", None)

    def _prefill_tokens(self, tokens: Sequence[int], cache: list[Any]) -> Any:
        """Feed ``tokens`` into ``cache`` in chunks of ``prefill_step``; the last position's logits [1, 1, V].

        One forward over a long suffix materialises its attention scores whole when the
        fused kernel does not cover the head size (256 here): an uncached 87k-token
        prompt asked Metal for 237 GB (2026-09-23). Only the last position goes through
        the vocabulary head: 2,048 rows of it were ~5 TFLOP and 1 GB a chunk, unused.
        """

        import mlx.core as mx

        if self.lane_prefill:
            return self._lane_prefill_tokens(tokens, cache)
        _, core, head = self._resolve_stack()
        logits = None
        step = max(1, int(self.prefill_step))
        for begin in range(0, len(tokens), step):
            chunk = [int(t) for t in tokens[begin:begin + step]]
            hidden = core(mx.array([chunk], dtype=mx.uint32), cache=cache)
            last = hidden[:, -1:, :]
            logits = head(last) if begin + step >= len(tokens) else None
            mx.eval(*self._cache_arrays(cache), logits if logits is not None else last)
            if begin + step < len(tokens):
                mx.clear_cache()
        return logits

    def _lane_prefill_tokens(self, tokens: Sequence[int], cache: list[Any]) -> Any:
        """The prompt through the decode path: chains of ``lane_prefill`` rows, each committed.

        Row-exact kernels make each row's bits independent of the chain it rides in, so the
        cache ends up exactly as feeding the prompt one token at a time would leave it. DFlash
        taps are gathered across chains (the last 2,048 rows) for the drafter's first block.
        """

        import mlx.core as mx

        from tensorfold.kernels import lane_tree

        _, core, head = self._resolve_stack()
        width = max(1, int(self.lane_prefill))
        kv = next((item for item in cache if hasattr(item, "keys")), None)
        start = int(getattr(kv, "offset", 0) or 0) if kv is not None else 0
        taps_store = getattr(self.model, "_hidden_states", None)
        gathered: list[Any] | None = [None] * len(taps_store) if taps_store else None
        logits = None
        end = start + len(tokens)
        for begin in range(0, len(tokens), width):
            chunk = [int(t) for t in tokens[begin:begin + width]]
            n = len(chunk)
            last = begin + width >= len(tokens)
            self._reserve_keys(cache, end)           # once the first chunk has made the buffers
            t0 = time.perf_counter()
            logits, record = lane_tree.tree_forward(core, head, chunk, [-1] + list(range(n - 1)), cache, start,
                                                    pipeline_layers=self.pipeline_layers, last_only=True)
            if _PROFILE:
                mx.eval(*self._cache_arrays(cache), logits, *[a for entry in record for a in entry[1:] if isinstance(a, mx.array)])
                t1 = time.perf_counter()
            lane_tree.commit_tree(cache, record, list(range(n)), n, start)
            if _PROFILE:
                mx.eval(*self._cache_arrays(cache))
                t2 = time.perf_counter()
                self.prefill_profile.append((n, (t1 - t0) * 1e3, (t2 - t1) * 1e3))
            start += n
            if gathered is not None:
                for i, tap in enumerate(taps_store):
                    if tap is not None:
                        gathered[i] = tap if gathered[i] is None else mx.concatenate([gathered[i], tap], axis=1)[:, -2048:]
            mx.eval(*self._cache_arrays(cache), *(g for g in (gathered or []) if g is not None),
                    *([logits] if last else []))
        if gathered is not None:
            for i, tap in enumerate(gathered):
                taps_store[i] = tap
        if _PROFILE and self.prefill_profile:
            rows = sum(r for r, _, _ in self.prefill_profile)
            fwd = sum(f for _, f, _ in self.prefill_profile)
            com = sum(c for _, _, c in self.prefill_profile)
            print(f"[lanes] prefill profile: {rows} rows in {len(self.prefill_profile)} chains, forward {fwd:.0f} ms "
                  f"({fwd / len(self.prefill_profile):.1f} a chain), commit {com:.0f} ms ({com / len(self.prefill_profile):.1f} a chain)",
                  flush=True)
            self.prefill_profile = []
        mx.clear_cache()
        return logits

    def _reserve_keys(self, cache: list[Any], length: int) -> None:
        """Grow each attention cache once to hold ``length`` positions (it grew 256 at a time, copying)."""

        import mlx.core as mx

        step = 256
        cap = step * (-(-int(length) // step))
        for item in cache:
            keys = getattr(item, "keys", None)
            if keys is None or not hasattr(item, "offset") or isinstance(getattr(item, "offset"), mx.array):
                continue
            have = int(keys.shape[2])
            if have >= cap:
                continue
            used = int(item.offset)
            extra_k = mx.zeros((*keys.shape[:2], cap - used, keys.shape[3]), dtype=keys.dtype)
            extra_v = mx.zeros((*item.values.shape[:2], cap - used, item.values.shape[3]), dtype=item.values.dtype)
            item.keys = mx.concatenate([keys[..., :used, :], extra_k], axis=2)
            item.values = mx.concatenate([item.values[..., :used, :], extra_v], axis=2)

    @property
    def active_count(self) -> int:
        """Streams holding or about to hold a row."""

        return sum(1 for s in self._active if not s.finished) + len(self._pending_join)

    def _shed_finished(self, *, force: bool = False) -> None:
        """Filter finished rows out of the batch, in groups when slack allows.

        Covers rows the engine finished and rows a caller finished between
        rounds (text stops). Rows inside the slack stay as idle placeholders.
        """

        idle = sum(1 for s in self._active if s.finished)
        if not idle:
            return
        if not force and idle < len(self._active) and idle <= self.shrink_slack:
            return
        keep = [i for i, s in enumerate(self._active) if not s.finished]
        self._filter(keep)
        self._active = [self._active[i] for i in keep]
        self._release_buffers()

    def _release_buffers(self) -> None:
        """Drop MLX's cached buffers after the batch changed size.

        A resized batch allocates every cache array anew and MLX keeps the old
        sizes cached; over a ragged finish the process grew until a round took
        3 to 25 s (64 lanes, 2026-09-19). Overridable like the other seams.
        """

        try:
            import mlx.core as mx
        except ImportError:  # fake-target tests
            return
        mx.clear_cache()

    def reset(self) -> None:
        """Drop every row after a failed round; streams keep their state."""

        self._batch = None
        self._active = []
        self._pending_join = []

    def add_stream(
        self,
        stream: LaneStream,
        *,
        cache: list[Any] | None = None,
        cached_tokens: int = 0,
        checkpoints_at: Sequence[int] = (),
    ) -> None:
        """Prefill and queue a stream; it joins the batch at the next round."""

        cache = self.prefill(
            stream, cache=cache, cached_tokens=cached_tokens, checkpoints_at=checkpoints_at
        )
        self.streams.append(stream)
        if stream.finished:
            stream.finished_at = time.perf_counter()
            return
        self._pending_join.append((stream, cache))

    def fork_shared(self, prefix_cache: list[Any]) -> list[Any]:
        """A lane cache whose attention layers share ``prefix_cache`` by reference.

        For ``add_forked`` lanes of one fan-out: the prefix keys and values are
        stored once for the whole batch instead of once per lane (see
        ``engine.shared_prefix``). Recurrent state is still copied per lane.
        Lanes forked this way can only batch with lanes of the same prefix.
        """

        from tensorfold.engine.shared_prefix import fork_shared, install_shared_attention

        if not getattr(self, "_shared_attention_installed", False):
            install_shared_attention(self.model)
            self._shared_attention_installed = True
        return fork_shared(prefix_cache, lambda item: self.copy_single_cache([item])[0])

    def add_forked(self, stream: LaneStream, *, cache: list[Any], cached_tokens: int) -> None:
        """Queue a stream on a forked cache with no per-stream prefill.

        ``cache`` already holds the first ``cached_tokens`` prompt tokens. The
        rest of the prompt rides as pending tokens, so K lanes of one answer
        absorb their suffixes together in one batched forward instead of K
        back-to-back prefills. The engine owns ``cache`` afterwards.
        """

        cut = int(cached_tokens)
        if not 0 < cut < len(stream.prompt_ids):
            raise ValueError(f"{stream.stream_id}: cached_tokens must leave a suffix to absorb")
        stream.emitted = []
        stream.pending = [int(t) for t in stream.prompt_ids[cut:]]
        stream.cache_len = cut
        stream.cached_tokens = cut
        stream.history_checkpoints = []
        stream.started_at = time.perf_counter()
        self.streams.append(stream)
        self._pending_join.append((stream, cache))

    def _extract_row(self, cache: list[Any], row: int) -> list[Any]:
        return [item.extract(int(row)) for item in cache]

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
        """Deep copy a batch-1 cache list so a retained checkpoint survives reuse."""

        import copy as _copy

        import mlx.core as mx

        out: list[Any] = []
        for item in cache:
            clone = _copy.copy(item)
            for key, value in vars(item).items():
                if isinstance(value, mx.array):
                    setattr(clone, key, mx.array(value))
                elif isinstance(value, list):
                    setattr(
                        clone,
                        key,
                        [mx.array(v) if isinstance(v, mx.array) else v for v in value],
                    )
            out.append(clone)
        return out

    def _merge(self, caches: list[list[Any]]) -> list[Any]:
        merged: list[Any] = []
        for layers in zip(*caches):
            merge = getattr(layers[0], "merge", None)
            if not callable(merge):
                raise TypeError(f"{type(layers[0]).__name__} cannot batch with history")
            merged.append(merge(list(layers)))
        return merged

    def _absorb_joins(self) -> dict[str, list[int]]:
        """Bring queued streams into the batch. Returns tokens they committed."""

        landed: dict[str, list[int]] = {}
        if not self._pending_join:
            return landed
        joining = [s for s, _ in self._pending_join]
        batch = self._merge([c for _, c in self._pending_join])
        self._pending_join = []
        running = self._batch is not None and any(not s.finished for s in self._active)
        if (
            not self.one_token
            and running
            and any(len(s.pending) > 1 for s in joining)
        ):
            # A wide suffix would pad every running row to its width. Absorb
            # the joiners' suffixes in a forward of their own first.
            landed = self._absorb_suffixes(joining, batch)
        self._shed_finished(force=True)  # one resize for leavers and joiners
        if self._batch is None or not self._active:
            self._batch = batch
            self._active = []
        else:
            for current, extra in zip(self._batch, batch):
                current.extend(extra)
        self._active.extend(joining)
        self._release_buffers()
        return landed

    def _absorb_suffixes(self, joining: list[LaneStream], batch: list[Any]) -> dict[str, list[int]]:
        windows = [list(s.pending) for s in joining]
        width = max(len(w) for w in windows)
        lengths = [len(w) for w in windows]
        right_padding = [width - n for n in lengths]
        ragged = max(right_padding) > 0
        rows = [w + [self.PAD_TOKEN] * (width - len(w)) for w in windows]
        if ragged:
            self._prepare(batch, lengths, right_padding)
        preds, _ = self._forward_rows(rows, batch)
        self._eval(batch, preds)
        if ragged:
            self._finalize(batch)
            self._eval(batch)
        pred_rows = self._to_rows(preds)
        landed: dict[str, list[int]] = {}
        for i, stream in enumerate(joining):
            tokens, _ = apply_round(stream, windows[i], pred_rows[i])
            landed[stream.stream_id] = tokens
            if stream.finished:
                stream.finished_at = time.perf_counter()
        return landed

    def _filter(self, keep: list[int]) -> None:
        import mlx.core as mx

        if self._batch is None:
            return
        if not keep:
            self._batch = None
            return
        indices = mx.array(keep, dtype=mx.int32)
        for item in self._batch:
            item.filter(indices)

    # -- MLX seams, overridable by tests that drive a fake target ----------
    def _eval(self, cache: list[Any], *extra: Any) -> None:
        import mlx.core as mx

        arrays = [*self._cache_arrays(cache), *[e for e in extra if e is not None]]
        if arrays:
            mx.eval(*arrays)

    def _prepare(self, cache: list[Any], lengths: list[int], right_padding: list[int]) -> None:
        for item in cache:
            item.prepare(lengths=lengths, right_padding=right_padding)

    def _finalize(self, cache: list[Any]) -> None:
        for item in cache:
            item.finalize()

    @staticmethod
    def _to_rows(preds: Any) -> list[list[int]]:
        rows = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        return [[int(t) for t in row] for row in rows]

    # -- the round ----------------------------------------------------------
    def _drafts_for(self, stream: LaneStream) -> list[int]:
        if stream.proposer is None or self.max_draft <= 0:
            return []
        if len(stream.pending) >= self.pending_cap:
            return []
        budget = min(self.max_draft, stream.draft_room - 1)
        if budget <= 0:
            return []
        try:
            proposal = stream.proposer.propose(stream.context, budget)
        except Exception:  # noqa: BLE001 - a proposer must never break a stream
            return []
        # MLX's 4-bit matmul costs the same for 1 to 4 rows and 2-4x from 6 rows on
        # (M5 Max, 2026-09-23): a window of up to CHEAP_WINDOW rows is nearly free, a
        # wider one pays only when the proposer has strong evidence (a long copy, a
        # tool call's structure). Unguarded 32-token windows ran 17 tok/s against ~30
        # plain on HTML (139 of 2,165 drafted tokens accepted).
        if not getattr(stream.proposer, "last_confident", False):
            budget = min(budget, max(0, self.cheap_window - len(stream.pending)))
        if self.exact_window:
            # Byte-exact verification: MLX 0.31.2's 4-bit matmul gives each row of a call
            # of up to 9 rows the bits of a 1-row call, not from 10 rows (2026-09-23).
            budget = min(budget, max(0, self.exact_window - len(stream.pending)))
        return [int(t) for t in proposal][:budget]

    def _snapshot_recurrent(self, cache: list[Any]) -> list[list[Any] | None]:
        import mlx.core as mx

        snaps: list[list[Any] | None] = []
        for item in cache:
            if self._is_kv(item):
                snaps.append(None)
                continue
            arrays = list(item.state)
            if self.copy_recurrent_snapshots:
                arrays = [mx.array(a) if a is not None else None for a in arrays]
            snaps.append(arrays)
        return snaps

    def _rollback_rows(
        self,
        cache: list[Any],
        amounts: list[int],
        snaps: list[list[Any] | None],
    ) -> None:
        """Give back ``amounts[i]`` window positions on row i."""

        import mlx.core as mx

        if not any(amounts):
            return
        rows = [i for i, amount in enumerate(amounts) if amount > 0]
        padding = mx.array(amounts, dtype=mx.int32)
        uniform = len(set(amounts)) == 1
        for item, snap in zip(cache, snaps):
            if self._is_kv(item):
                if uniform and hasattr(item, "trim"):
                    # Every row gives back the same tail (always so for one stream): drop
                    # it, so the keys sit exactly where serial decoding puts them. Rolled
                    # into masked left padding instead, attention ran over a different
                    # key layout and flipped near-ties against the serial decode
                    # (2026-09-23).
                    item.trim(amounts[0])
                    continue
                # The rejected tail is right padding: finalize() rolls it into
                # masked left padding and steps the row's offset back.
                item._right_padding = padding
                item.finalize()
                continue
            if snap is None:
                continue
            live = list(item.state)
            for slot, (current, before) in enumerate(zip(live, snap)):
                if current is None or before is None:
                    continue
                if current is before:
                    continue
                for row in rows:
                    current[row : row + 1] = before[row : row + 1]
                live[slot] = current
            item.state = live

    def step(self) -> dict[str, list[int]]:
        """One shared verify round. Returns newly committed tokens per stream."""

        started = time.perf_counter()
        joined = self._absorb_joins()
        self._shed_finished()
        active = list(self._active)
        idle = [s.finished for s in active]  # placeholders inside the slack
        if not active or self._batch is None or all(idle):
            return joined
        cache = self._batch
        if self.precise_single and len(active) == 1 and not self.one_token:
            return self._step_precise(active[0], cache, joined, started)

        # One token per lane. Extra pending tokens are the rest of that lane's
        # prompt; they stay queued and are not output.
        prompt_rest: list[list[int]] = []
        shown: list[list[int]] = []
        for stream, gone in zip(active, idle):
            if gone:
                prompt_rest.append([])
                shown.append([self.PAD_TOKEN])
                continue
            if self.one_token and len(stream.pending) > 1:
                prompt_rest.append([int(t) for t in stream.pending[1:]])
                stream.pending = [int(stream.pending[0])]
            else:
                prompt_rest.append([])
            shown.append(list(stream.pending))
        drafts = [
            [] if gone or self.one_token else self._drafts_for(s)
            for s, gone in zip(active, idle)
        ]
        windows, width = plan_windows(shown, drafts, max_rows=self.max_rows)
        lengths = [len(w) for w in windows]
        right_padding = [width - n for n in lengths]
        ragged = max(right_padding) > 0
        rows = [w + [self.PAD_TOKEN] * (width - len(w)) for w in windows]

        snaps = self._snapshot_recurrent(cache)
        if ragged:
            self._prepare(cache, lengths, right_padding)
        forward_started = time.perf_counter()
        sampled = any(s.sampling is not None for s, gone in zip(active, idle) if not gone)
        preds, _ = (self._forward_rows(rows, cache, logits=True) if sampled
                    else self._forward_rows(rows, cache))
        self._eval(cache, preds)
        forward_ms = (time.perf_counter() - forward_started) * 1e3
        for stream, gone in zip(active, idle):
            # A shared round re-feeds accepted drafts: a proposer that reads the kept
            # rows' hidden states (DFlash2) cannot follow it and stops proposing.
            invalidate = getattr(stream.proposer, "invalidate", None)
            if not gone and callable(invalidate):
                invalidate()
        finalize_started = time.perf_counter()
        if ragged:
            self._finalize(cache)
            self._eval(cache)
        finalize_ms = (time.perf_counter() - finalize_started) * 1e3

        pred_rows = (self._predict_rows([None if gone else s for s, gone in zip(active, idle)], preds)
                     if sampled else self._to_rows(preds))
        landed: dict[str, list[int]] = dict(joined)
        amounts: list[int] = []
        committed = 0
        for i, stream in enumerate(active):
            if idle[i]:
                amounts.append(0)
                continue
            if prompt_rest[i]:
                stream.cache_len += len(windows[i])
                stream.rounds += 1
                stream.pending = prompt_rest[i]
                amounts.append(0)
                continue
            tokens, rollback = apply_round(stream, windows[i], pred_rows[i])
            landed[stream.stream_id] = tokens
            committed += len(tokens)
            if stream.finished:
                stream.finished_at = time.perf_counter()
                if not self.retain_finished_caches or not stream.retain:
                    amounts.append(0)  # the row is about to be dropped
                elif stream.cache_len > len(stream.context):
                    # The commit was cut short (token limit or EOS inside the
                    # accepted drafts) after the row absorbed the whole window:
                    # give the window back so the row holds only committed
                    # tokens, and account for it.
                    stream.cache_len -= len(windows[i])
                    amounts.append(len(windows[i]))
                else:
                    amounts.append(rollback)
            else:
                amounts.append(rollback)
        rollback_started = time.perf_counter()
        self._rollback_rows(cache, amounts, snaps)
        if any(amounts):
            self._eval(cache)
        rollback_ms = (time.perf_counter() - rollback_started) * 1e3

        if self.retain_finished_caches:
            for i, stream in enumerate(active):
                if stream.finished and not idle[i] and stream.retain:  # finished this round
                    absorbed = stream.context[: stream.cache_len]
                    self.finished_caches[stream.stream_id] = (
                        absorbed,
                        self._extract_row(cache, i),
                    )
        self._shed_finished()
        self.round_stats.append(
            RoundStats(
                streams=len(active) - sum(idle),
                width=width,
                rows=len(active) * width,
                ragged=ragged,
                rollbacks=sum(1 for a in amounts if a > 0),
                committed=committed,
                forward_ms=forward_ms,
                finalize_ms=finalize_ms,
                rollback_ms=rollback_ms,
                total_ms=(time.perf_counter() - started) * 1e3,
            )
        )
        return landed

    def _unpadded(self, cache: list[Any]) -> bool:
        """True when no attention row of the batch carries left padding."""

        import mlx.core as mx

        for item in cache:
            padding = getattr(item, "left_padding", None)
            if self._is_kv(item) and padding is not None:
                return int(mx.max(padding).item()) == 0 if padding.size else True
        return True

    def _step_precise(
        self,
        stream: LaneStream,
        cache: list[Any],
        joined: dict[str, list[int]],
        started: float,
    ) -> dict[str, list[int]]:
        """One stream's round, rolled back to exactly its kept prefix (``precise_single``)."""

        from tensorfold.kernels import gdn_capture

        if self.tree_nodes > 0 and len(stream.pending) == 1:
            # every round, drafted or not, through the tree forward: one decoder, one arithmetic
            return self._step_tree(stream, cache, joined, started)
        gdn_capture.install()
        draft_started = time.perf_counter()
        drafts = self._drafts_for(stream)
        draft_ms = (time.perf_counter() - draft_started) * 1e3
        windows, width = plan_windows([list(stream.pending)], [drafts], max_rows=self.max_rows)
        window = windows[0]
        simple = self.simple_masks and self._unpadded(cache)
        sampled = stream.sampling is not None
        gdn_capture.begin()
        try:
            forward_started = time.perf_counter()
            extra = {k: v for k, v in (("simple_mask", simple), ("logits", sampled)) if v}
            self._route_hidden(stream)
            preds, _ = self._forward_rows([window], cache, **extra)
            self._route_hidden(None)
            self._eval(cache, preds)
            forward_ms = (time.perf_counter() - forward_started) * 1e3
        finally:
            captured = gdn_capture.end()
        post_started = time.perf_counter()
        pred_row = self._predict_rows([stream], preds)[0] if sampled else self._to_rows(preds)[0]
        tokens, keep = apply_round_precise(stream, window, pred_row)
        on_round = getattr(stream.proposer, "on_round", None)
        if callable(on_round):
            on_round(0, keep)
        post_ms = (time.perf_counter() - post_started) * 1e3
        rollback_started = time.perf_counter()
        if keep < len(window):
            gdn_capture.rollback(cache, captured, len(window), keep)
            self._eval(cache)
        rollback_ms = (time.perf_counter() - rollback_started) * 1e3
        del captured
        landed: dict[str, list[int]] = dict(joined)
        landed[stream.stream_id] = tokens
        if stream.finished:
            stream.finished_at = time.perf_counter()
            if self.retain_finished_caches and stream.retain:
                self.finished_caches[stream.stream_id] = (
                    stream.context[: stream.cache_len],
                    self._extract_row(cache, 0),
                )
        self._shed_finished()
        self.round_stats.append(
            RoundStats(
                streams=1,
                width=width,
                rows=width,
                ragged=False,
                rollbacks=int(keep < len(window)),
                committed=len(tokens),
                forward_ms=forward_ms,
                finalize_ms=0.0,
                rollback_ms=rollback_ms,
                total_ms=(time.perf_counter() - started) * 1e3,
                draft_ms=draft_ms,
                post_ms=post_ms,
                started_at=started,
            )
        )
        return landed

    def _shadow_stats(self, stream: LaneStream, rows_parents: Sequence[int], preds: Sequence[int],
                      path: Sequence[int], start: int, depths: Sequence[int]) -> None:
        """Tally last rounds' shadows against what committed, and record this round's.

        A shadow is the target's sample at a row below the last accepted one (on the drafter's first-child
        chain from the rejected token): the target's own next token given a prefix that is wrong at the
        rejected position. Keyed noise makes a sample depend on the prefix only through the logits, so
        these may often still be right: a free, target-quality deep draft for the next round. Offset 1 is
        the position after the rejected one.
        """

        context = stream.context
        pending: dict[int, tuple[int, int]] = getattr(stream, "_shadows", {})
        for pos in [p for p in pending if p < len(context)]:
            offset, token = pending.pop(pos)
            tally = self.shadow_stats.setdefault(offset, [0, 0])
            tally[0] += int(int(context[pos]) == token)
            tally[1] += 1
        children: dict[int, list[int]] = {}
        for row, parent in enumerate(rows_parents):
            if parent >= 0:
                children.setdefault(parent, []).append(row)
        row = (children.get(path[-1]) or [None])[0]
        offset = 1
        while row is not None:
            pending[start + depths[row] + 1] = (offset, int(preds[row]))
            row = (children.get(row) or [None])[0]
            offset += 1
        stream._shadows = pending
        self._shadow_rounds = getattr(self, "_shadow_rounds", 0) + 1
        if self._shadow_rounds % 500 == 0:
            parts = [f"{k}: {100 * h / n:.0f}% of {n}" for k, (h, n) in sorted(self.shadow_stats.items())[:8] if n]
            print(f"[lanes] shadow hits by offset after {self._shadow_rounds} rounds: " + ", ".join(parts), flush=True)

    def _step_tree(
        self,
        stream: LaneStream,
        cache: list[Any],
        joined: dict[str, list[int]],
        started: float,
    ) -> dict[str, list[int]]:
        """One stream's round over a draft tree: one forward, the target's path kept exactly."""

        import mlx.core as mx

        from tensorfold.kernels import lane_tree

        draft_started = time.perf_counter()
        nodes = max(int(self.tree_nodes), int(self.chain_nodes))
        budget = min(nodes, stream.draft_room - 1, (int(self.exact_window) - 1) if self.exact_window else nodes)
        tokens: list[int] = []
        parents: list[int] = []
        propose_tree = getattr(stream.proposer, "propose_tree", None)
        if budget > 0 and callable(propose_tree):
            try:
                tokens, parents = propose_tree(stream.context, budget)
            except Exception as exc:  # noqa: BLE001 - a proposer must never break a stream
                tokens, parents = [], []
                self.proposer_errors += 1
                if self.proposer_errors <= 3:
                    import traceback

                    print(f"[lanes] proposer failed (round drafted nothing): {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
            tokens, parents = [int(t) for t in tokens][:budget], [int(q) for q in parents][:budget]
        draft_ms = (time.perf_counter() - draft_started) * 1e3
        window = [int(stream.pending[0]), *tokens]
        rows_parents = [-1] + [0 if q < 0 else q + 1 for q in parents]
        start = int(stream.cache_len)
        _, core, head = self._resolve_stack()
        if _KERNEL_AB:
            from tensorfold.kernels import lane_qmm

            lane_qmm.AB_FLAG[0] = (stream.rounds // _KERNEL_AB) % 2 == 1
        forward_started = time.perf_counter()
        # The sampling candidates are evaluated with the forward (no GPU round trip of their own) and the
        # forward's first layer goes to the GPU alone: 57.08 -> 56.47 ms a round, 700 rounds each, interleaved
        # in one process (2026-09-24)
        self._route_hidden(stream)
        logits, record = lane_tree.tree_forward(core, head, window, rows_parents, cache, start,
                                                pipeline_layers=self.pipeline_layers)
        lane_tree.HIDDEN_SINK = None
        top = None
        if stream.sampling is not None:
            from tensorfold.engine.exact_sampling import top_candidates

            top = top_candidates(logits[0], stream.sampling)
        self._eval(cache, logits, *(top or ()))
        forward_ms = (time.perf_counter() - forward_started) * 1e3
        post_started = time.perf_counter()
        depths, _ = lane_tree.tree_paths(rows_parents)
        if stream.sampling is not None:
            from tensorfold.engine.exact_sampling import sample_rows

            keep: dict[str, Any] | None = {} if callable(getattr(stream.proposer, "capture_target", None)) else None
            preds = sample_rows(logits[0], [start + d + 1 for d in depths], stream.sampling, keep=keep, top=top)
        else:
            keep = None
            preds = [int(t) for t in mx.argmax(logits[0], axis=-1).tolist()]
        path = lane_tree.accept_path(window, rows_parents, preds)
        if keep:
            # the target's candidates behind each committed token: distillation data for the drafter
            try:
                stream.proposer.capture_target([start + depths[r] + 1 for r in path],
                                               keep["cand"][path], keep["vals"][path])
            except Exception:  # noqa: BLE001 - capture must never break a stream
                pass
        accepted = len(path) - 1
        bonus = int(preds[path[-1]])
        if _ROUND_LOG:
            chain = rows_parents == list(range(-1, len(window) - 1))
            with open(_ROUND_LOG, "a") as handle:
                ab = ""
                if _KERNEL_AB:
                    from tensorfold.kernels import lane_qmm

                    ab = f" {int(lane_qmm.AB_FLAG[0])}"
                handle.write(f"{start} {len(window)} {accepted} {'chain' if chain else 'tree'} {forward_ms:.1f} {draft_ms:.1f}{ab}\n")
        stream.rounds += 1
        stream.drafted += len(tokens)
        stream.accepted += accepted
        observe = getattr(stream.proposer, "observe", None)
        if callable(observe) and tokens:
            observe(len(tokens), accepted)
        landed = stream.commit([window[r] for r in path[1:]] + [bonus])
        if _SHADOW:
            self._shadow_stats(stream, rows_parents, preds, path, start, depths)
        kept = path[: 1 + min(accepted, len(landed))]
        if tokens and accepted == max(depths):
            stream.full_rounds += 1
        elif tokens:
            stream.partial_rounds += 1
        post_ms = (time.perf_counter() - post_started) * 1e3
        rollback_started = time.perf_counter()
        lane_tree.commit_tree(cache, record, kept, len(window), start)
        # left lazy: the next forward (or a finished row's extraction) evaluates the commit. Handing it to
        # the GPU at once measured no faster (58.4 against 57.7 ms a round, 400 rounds each, 2026-09-24):
        # its GPU time is ~0.5 ms, less than the extra host sync costs
        if _PROFILE:                       # diagnostic: the commit's own GPU time (an extra sync a round)
            self._eval(cache)
        rollback_ms = (time.perf_counter() - rollback_started) * 1e3
        stream.cache_len += len(kept)
        stream.pending = [stream.last_bonus(bonus)]
        on_rows = getattr(stream.proposer, "on_rows", None)
        if callable(on_rows):
            on_rows(kept)
        del record
        landed_map: dict[str, list[int]] = dict(joined)
        landed_map[stream.stream_id] = landed
        if stream.finished:
            stream.finished_at = time.perf_counter()
            if self.retain_finished_caches and stream.retain:
                self.finished_caches[stream.stream_id] = (
                    stream.context[: stream.cache_len],
                    self._extract_row(cache, 0),
                )
        self._shed_finished()
        self.round_stats.append(
            RoundStats(
                streams=1,
                width=len(window),
                rows=len(window),
                ragged=False,
                rollbacks=int(len(kept) < len(window)),
                committed=len(landed),
                forward_ms=forward_ms,
                finalize_ms=0.0,
                rollback_ms=rollback_ms,
                total_ms=(time.perf_counter() - started) * 1e3,
                draft_ms=draft_ms,
                post_ms=post_ms,
                started_at=started,
            )
        )
        return landed_map

    def run(self, on_tokens: Callable[[str, list[int]], None] | None = None) -> None:
        while True:
            landed = self.step()
            if not landed and not self._pending_join and not any(
                not s.finished for s in self._active
            ):
                break
            if on_tokens is not None:
                for stream_id, tokens in landed.items():
                    if tokens:
                        on_tokens(stream_id, tokens)

    # -- reporting ------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        rounds = self.round_stats
        total_rows = sum(r.rows for r in rounds)
        total_committed = sum(r.committed for r in rounds)
        return {
            "rounds": len(rounds),
            "streams": len(self.streams),
            "committed_tokens": total_committed,
            "mean_rows_per_round": (total_rows / len(rounds)) if rounds else 0.0,
            "mean_width": (sum(r.width for r in rounds) / len(rounds)) if rounds else 0.0,
            "ragged_rounds": sum(1 for r in rounds if r.ragged),
            "rollback_rows": sum(r.rollbacks for r in rounds),
            "row_conversion": (total_committed / total_rows) if total_rows else 0.0,
            "forward_ms_total": sum(r.forward_ms for r in rounds),
            "finalize_ms_total": sum(r.finalize_ms for r in rounds),
            "rollback_ms_total": sum(r.rollback_ms for r in rounds),
            "round_ms_total": sum(r.total_ms for r in rounds),
            "max_rows": self.max_rows,
            "max_draft": self.max_draft,
            "pending_cap": self.pending_cap,
        }
