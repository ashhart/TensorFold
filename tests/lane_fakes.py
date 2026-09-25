"""A history-dependent fake target for lane engine and lane server tests.

The next token depends on the WHOLE absorbed history, so any rollback,
refeed, padding, or checkpoint mistake shows up as a byte divergence from
the fake's own serial decode.
"""

from __future__ import annotations

from typing import Any, Sequence

from tensorfold.engine.lane_engine import LaneEngine, LaneStream

VOCAB = 97


def fake_next(history: list[int]) -> int:
    return (sum(history) * 7 + len(history) * 3) % VOCAB


def fake_serial(prompt: list[int], max_new: int, eos: set[int]) -> list[int]:
    history = list(prompt)
    out: list[int] = []
    while len(out) < max_new:
        token = fake_next(history)
        out.append(token)
        history.append(token)
        if token in eos:
            break
    return out


class PatternProposer:
    """Proposes the true continuation for ``good`` tokens, then a wrong one."""

    def __init__(self, pattern: list[int]) -> None:
        self.pattern = pattern
        self.calls = 0
        self.observed: list[tuple[int, int]] = []

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        good = self.pattern[self.calls % len(self.pattern)]
        self.calls += 1
        history = list(context)
        out: list[int] = []
        for j in range(max_draft):
            token = fake_next(history)
            if j >= good:
                token = (token + 1) % VOCAB
            out.append(token)
            history.append(token)
        return out

    def observe(self, proposed: int, accepted: int) -> None:
        self.observed.append((proposed, accepted))


class FakeBatchItem:
    """Per-row absorbed history standing in for a cache layer."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = [list(r) for r in rows]
        self.prepared: list[tuple[list[int], list[int]]] = []
        self.lengths: list[int] | None = None
        self.finalized = 0

    @property
    def state(self) -> list[Any]:
        return []

    def prepare(self, *, lengths: list[int], right_padding: list[int]) -> None:
        # Real caches mask the padded tail; the fake must not absorb it.
        self.prepared.append((list(lengths), list(right_padding)))
        self.lengths = list(lengths)

    def finalize(self) -> None:
        self.finalized += 1
        self.lengths = None

    def extend(self, other: "FakeBatchItem") -> None:
        self.rows.extend(other.rows)

    def filter(self, keep: list[int]) -> None:
        self.rows = [self.rows[i] for i in keep]

    def extract(self, row: int) -> "FakeBatchItem":
        return FakeBatchItem([self.rows[row]])


class FakeEngine(LaneEngine):
    """LaneEngine over the fake target; MLX seams replaced with lists."""

    # The fake target has no row cost curve: windows are not capped, so the tests
    # keep exercising ragged windows and rollbacks.
    cheap_window = 1_000

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.filter_calls: list[list[int]] = []
        self.rollback_calls: list[list[int]] = []
        self.prefill_calls: list[tuple[str, int]] = []

    def prefill(
        self,
        stream: LaneStream,
        *,
        cache: list[Any] | None = None,
        cached_tokens: int = 0,
        checkpoints_at: Sequence[int] = (),
    ) -> list[Any]:
        if cache is None:
            history = list(stream.prompt_ids)
            cached_tokens = 0
        else:
            history = list(cache[0].rows[0]) + list(stream.prompt_ids[int(cached_tokens) :])
        if history != list(stream.prompt_ids):
            raise AssertionError("checkpoint was not a prefix of the prompt")
        self.prefill_calls.append((stream.stream_id, int(cached_tokens)))
        stream.history_checkpoints = []
        for boundary in sorted({int(b) for b in checkpoints_at}):
            if int(cached_tokens) < boundary < len(history):
                head = list(stream.prompt_ids[:boundary])
                stream.history_checkpoints.append((head, [FakeBatchItem([head])]))
        first = fake_next(history)
        stream.emitted = []
        stream.cache_len = len(stream.prompt_ids)
        stream.cached_tokens = int(cached_tokens)
        stream.commit([first])
        stream.pending = [first]
        return [FakeBatchItem([history])]

    def prefill_prefix(
        self,
        prompt_ids: Sequence[int],
        *,
        cache: list[Any] | None = None,
        cached_tokens: int = 0,
    ) -> list[Any]:
        if cache is None:
            history = list(prompt_ids)
        else:
            history = list(cache[0].rows[0]) + list(prompt_ids[int(cached_tokens) :])
        if history != list(prompt_ids):
            raise AssertionError("prefix checkpoint was not a prefix of the prompt")
        return [FakeBatchItem([history])]

    def _merge(self, caches: list[list[Any]]) -> list[Any]:
        return [FakeBatchItem([c[0].rows[0] for c in caches])]

    def _snapshot_recurrent(self, cache: list[Any]) -> list[Any]:
        return [[list(r) for r in cache[0].rows]]

    def _forward_rows(self, rows: list[list[int]], cache: list[Any]) -> tuple[Any, Any]:
        item = cache[0]
        preds: list[list[int]] = []
        for row_index, row in enumerate(rows):
            real = item.lengths[row_index] if item.lengths is not None else len(row)
            history = list(item.rows[row_index])
            out: list[int] = []
            for position, token in enumerate(row):
                if position < real:
                    history.append(token)
                    out.append(fake_next(history))
                else:
                    out.append(-1)  # masked pad position: garbage, never read
            preds.append(out)
            item.rows[row_index] = history
        return preds, None

    def _eval(self, cache: list[Any], *extra: Any) -> None:
        del cache, extra

    def _rollback_rows(self, cache: list[Any], amounts: list[int], snaps: list[Any]) -> None:
        self.rollback_calls.append(list(amounts))
        for row, amount in enumerate(amounts):
            if amount > 0:
                cache[0].rows[row] = list(snaps[0][row])

    def _filter(self, keep: list[int]) -> None:
        self.filter_calls.append(list(keep))
        if self._batch is not None:
            for item in self._batch:
                item.filter(keep)

    @staticmethod
    def copy_single_cache(cache: list[Any]) -> list[Any]:
        return [FakeBatchItem([list(r) for r in item.rows]) for item in cache]
