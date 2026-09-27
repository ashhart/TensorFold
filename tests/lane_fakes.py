"""A history-dependent fake target for lane engine and lane server tests.

The next token depends on the WHOLE absorbed history, so any rollback, refeed or checkpoint mistake shows up as a
byte divergence from the fake's own serial decode. ``FakeFamily`` serves it as a lane-engine family.
"""

from __future__ import annotations

import time
from typing import Any

from tensorfold.engine.lane_engine import LaneEngine

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
    """Proposes the true continuation for ``good`` tokens, then a wrong one; every proposal is confident (the family
    round verifies it as a copy)."""

    last_match = 1 << 30

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
    """Absorbed history standing in for a cache layer (one row)."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = [list(r) for r in rows]

    @property
    def state(self) -> list[Any]:
        return []


class FakeFamily:
    """The fake target as a family: a cache holds the absorbed history, each row's hidden state carries the token the
    target picks after it, and a rollback trims the history. ``delay``: seconds a forward takes."""

    lane_family = True
    streams_exact = True
    exact_width = 16
    mtp = None
    gpu_tokens = False

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = float(delay)

    def make_cache(self) -> list[Any]:
        return [FakeBatchItem([[]])]

    def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> Any:
        import mlx.core as mx
        import numpy as np

        if self.delay:
            time.sleep(self.delay)
        history = cache[0].rows[0]
        out = []
        for token in np.array(inputs).reshape(-1).tolist():
            history.append(int(token))
            out.append(fake_next(history))
        return mx.array(out, dtype=mx.float32).reshape(1, -1, 1)

    def hidden_rows(self, windows: list[Any], caches: list[list[Any]], parents: Any = None) -> Any:
        import mlx.core as mx

        return mx.concatenate([self.hidden(w, c) for w, c in zip(windows, caches)], axis=1)

    def head(self, hidden: Any) -> Any:
        import mlx.core as mx
        import numpy as np

        picks = np.array(hidden).reshape(-1).astype(np.int64)
        logits = np.zeros((1, len(picks), VOCAB), dtype=np.float32)
        logits[0, np.arange(len(picks)), picks] = 10.0
        return mx.array(logits)

    def keep_rows(self, cache: list[Any], rows: int, keep: Any) -> None:
        kept = keep if isinstance(keep, int) else len(keep)
        if kept < rows:
            del cache[0].rows[0][kept - rows:]

    def keep_rows_streams(self, caches: list[list[Any]], lengths: Any, keeps: Any) -> None:
        for cache, rows, keep in zip(caches, lengths, keeps):
            self.keep_rows(cache, int(rows), keep)


class FakeEngine(LaneEngine):
    """LaneEngine over ``FakeFamily`` without a prefill plan; records each prefill's resume and checks its prefix."""

    prefill_plan = None

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        super().__init__(model if model is not None else FakeFamily(), **kwargs)
        self.prefill_calls: list[tuple[str, int]] = []

    def _family_prefill(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                        checkpoints_at: Any) -> list[Any]:
        cached = int(cached_tokens) if cache is not None else 0
        if cache is not None and list(cache[0].rows[0]) != list(stream.prompt_ids[:cached]):
            raise AssertionError("checkpoint was not a prefix of the prompt")
        self.prefill_calls.append((stream.stream_id, cached))
        return super()._family_prefill(stream, cache=cache, cached_tokens=cached_tokens,
                                       checkpoints_at=checkpoints_at)

    @staticmethod
    def copy_single_cache(cache: list[Any]) -> list[Any]:
        return [FakeBatchItem([list(r) for r in item.rows]) for item in cache]
