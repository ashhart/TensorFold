"""What a family implements for the shared lane decoder; every call runs on every rank with the same arguments."""
# A row's bits depend only on its lane's committed tokens and position, never on the window or the other lanes.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Piece:
    """Prompt ids of ``lane`` at positions [start, start + len(ids)): prefilled, no logits."""

    lane: int
    start: int
    ids: tuple[int, ...]

    @property
    def end(self) -> int:
        return self.start + len(self.ids)


@dataclass(frozen=True)
class Rows:
    """A lane's verify window: ``tokens[0]`` its pending token at ``start``, then its drafts, a chain."""

    lane: int
    start: int
    tokens: tuple[int, ...]


@dataclass
class Candidates:
    """Each verify row's top ``count`` logits (``ids`` int64, ``values`` float32 [rows, count]), equal on all ranks."""

    ids: np.ndarray
    values: np.ndarray

    def rows(self, a: int, b: int) -> Candidates:
        return Candidates(self.ids[a:b], self.values[a:b])


@runtime_checkable
class LaneForward(Protocol):
    """A family's model over lanes. Optional: ``grid`` (pieces start on its multiples), ``replay_tail``, ``bind``."""

    # A compressed plane's row is written once all its tokens are known; an incomplete one lives in ``snapshot``.

    # Optional ``request_bytes(prompt_len, max_new)``: what a request allocates outside the page pool (admission)
    vocab: int

    def reset(self, lane: int) -> None:
        """A new request in ``lane``: its state at position 0 (its pages are the decoder's)."""

    def prefill(self, pieces: Sequence[Piece]) -> None:
        """One forward over several lanes' pieces; a row's bits do not depend on what shares the forward."""

    def finish_prompt(self, lane: int, end: int, tail: Sequence[int]) -> None:
        """With ``replay_tail`` > 0: the prompt's prefill ended at ``end``; rebuild decode state from its last ids."""

    def window(self, rows: Sequence[Rows], count: int, masks: Sequence[Any] | None = None) -> Candidates:
        """Every window in one forward, rows in order; ``masks[i]`` (a ``grammar.Window``) masks window i's rows."""

    def commit(self, lane: int, kept: int) -> None:
        """Keep the lane's last window's pending row and ``kept`` drafts; the rows past them are dead."""

    def snapshot(self, lane: int) -> Any:
        """The lane's own state at its position (pool pages excluded); what ``restore`` takes back."""

    def restore(self, lane: int, snap: Any) -> None:
        """Load ``snap`` into a reset lane whose pages below ``snap``'s position already hold its rows."""


REQUIRED = ("reset", "prefill", "finish_prompt", "window", "commit", "snapshot", "restore")


def check(forward: Any) -> None:
    """TypeError naming what ``forward`` lacks of ``LaneForward``."""

    missing = [m for m in REQUIRED if not callable(getattr(forward, m, None))]
    if not isinstance(getattr(forward, "vocab", None), int):
        missing.append("vocab")
    if missing:
        raise TypeError(f"not a LaneForward: missing {', '.join(missing)}")


def chain(n: int) -> list[int]:
    """The parents of a window of ``n`` rows that is one chain."""

    return list(range(-1, n - 1))


def split(cand: Candidates, rows: Sequence[Rows]) -> list[Candidates]:
    """A window's candidates cut into each lane's rows."""

    out, at = [], 0
    for r in rows:
        out.append(cand.rows(at, at + len(r.tokens)))
        at += len(r.tokens)
    return out


__all__ = ["REQUIRED", "Candidates", "LaneForward", "Piece", "Rows", "chain", "check", "split"]
