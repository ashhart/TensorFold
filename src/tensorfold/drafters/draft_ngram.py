"""The stream's own text as a prior over DFlash2's draft lattice: a session n-gram model.

Agent sessions repeat themselves: identifiers, paths, tool-call syntax and whole phrases come
back from the prompt and from what the model already wrote. ``SessionNGram`` counts the 1..4-grams
of everything in one stream's context (prompt + committed output) and gives P(next | last 3 tokens)
with interpolated backoff: starting from 1/V, each order whose context was seen c times mixes in
its relative frequency with weight c / (c + 2). The draft tree adds ``weight * log P`` to each
child's score before the log-softmax over its siblings (``DFlashProposer.ngram_weight``).

Replayed on 703 traced agent rounds (2026-09-23): weight 0.1 lands 4.615 tokens per pass against
4.494 without it (+2.7%, better in 10 of 14 sessions); 0.2 and above lose.

Cost. The first update indexes the prompt with numpy: per order the sorted distinct 64-bit
(context, next) keys and their counts (~2 ms for 30k tokens, run while the drafter's forward is
on the GPU). Later updates add only the new tokens, to small dicts. A context's next-token counts
are pulled out of the index the first time a tree node needs them and kept current after that, so
a round's lookups are mostly dict reads; only the 16 candidates of each expanded node are scored.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

_BITS = 18                       # token ids < 2**18 (Qwen3.8's vocabulary: 248,320)
_LOW = (1 << _BITS) - 1
_MULT = 0x9E3779B97F4A7C15       # a 3-token context hashes to 46 bits, so (context, next) fits 64
_M64 = (1 << 64) - 1


def key4(t3: int, t2: int, t1: int) -> int:
    """The 4-gram context (t3, t2, t1) as a 46-bit key (odd-multiplier hash, top bits)."""

    return ((((t3 << 36) | (t2 << 18) | t1) * _MULT) & _M64) >> _BITS


class SessionNGram:
    """Counts of one stream's 1..4-grams; ``rescorer`` turns them into a tree-score bonus.

    ``update(context)`` is called with the whole context every round; only the tokens past what
    it has seen are counted. A context that does not extend the previous one starts it over.
    """

    def __init__(self, vocab: int = 248320, *, prior: float = 2.0) -> None:
        self.vocab = int(vocab)
        self.prior = float(prior)        # an order's weight is c / (c + prior) for a context seen c times
        self.pulls = 0                   # contexts pulled out of the index (telemetry)
        self.reset()

    def reset(self) -> None:
        self.seen = 0                    # context tokens counted
        self.tail: list[int] = []        # the last (up to) 3 of them
        self.unigram: np.ndarray | None = None
        # order k in 2..4, from the first update: the sorted distinct (context << 18 | next) keys
        # and where each one's run starts among all the keys (so key i was seen bounds[i+1] - bounds[i] times) ...
        self._keys: list[np.ndarray | None] = [None] * 5
        self._bounds: list[np.ndarray | None] = [None] * 5
        # ... {context: [total, {next: count}]} once pulled out of the index (kept current) ...
        self._entries: list[dict[int, list[Any]]] = [{} for _ in range(5)]
        # ... and {context: {next: count}} counted after the index for contexts not pulled yet
        self._pending: list[dict[int, dict[int, int]]] = [{} for _ in range(5)]

    # -- counting -------------------------------------------------------------------------
    def update(self, context: Sequence[int]) -> None:
        """Count the n-grams ending in ``context``'s tokens past the ones already counted."""

        n = len(context)
        seen = self.seen
        if n < seen or (seen and [int(t) for t in context[seen - len(self.tail): seen]] != self.tail):
            self.reset()
            seen = 0
        if n == seen:
            return
        if self.unigram is None:
            self._build(context)
            return
        unigram = self.unigram
        add = self._add
        t3, t2, t1 = ([None] * 3 + self.tail)[-3:]
        for i in range(seen, n):
            t = int(context[i])
            if t < 0 or t >= self.vocab:
                self.reset()             # not a token id of this vocabulary: count nothing
                return
            unigram[t] += 1
            if t1 is not None:
                add(2, t1, t)
                if t2 is not None:
                    add(3, (t2 << _BITS) | t1, t)
                    if t3 is not None:
                        add(4, key4(t3, t2, t1), t)
            t3, t2, t1 = t2, t1, t
        self.seen = n
        self.tail = [t for t in (t3, t2, t1) if t is not None]

    def _build(self, context: Sequence[int]) -> None:
        tokens = np.asarray(context, dtype=np.int64).reshape(-1)
        n = int(tokens.size)
        if n == 0 or int(tokens.min()) < 0 or int(tokens.max()) >= min(self.vocab, _LOW + 1):
            return
        self.unigram = np.bincount(tokens, minlength=self.vocab)
        u = tokens.astype(np.uint64)
        b, b2 = np.uint64(_BITS), np.uint64(2 * _BITS)
        grams: dict[int, np.ndarray] = {}
        if n >= 2:
            grams[2] = (u[:-1] << b) | u[1:]
        if n >= 3:
            grams[3] = (u[:-2] << b2) | (u[1:-1] << b) | u[2:]
        if n >= 4:
            ctx = (u[:-3] << b2) | (u[1:-2] << b) | u[2:-1]
            grams[4] = (((ctx * np.uint64(_MULT)) >> b) << b) | u[3:]      # uint64 products wrap: the hash
        for order, keys in grams.items():
            keys = np.sort(keys)
            starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
            self._keys[order] = keys[starts]
            self._bounds[order] = np.append(starts, keys.size)
        self.seen = n
        self.tail = [int(t) for t in tokens[-3:].tolist()]

    def _add(self, order: int, context: int, token: int) -> None:
        entry = self._entries[order].get(context)
        if entry is not None:
            entry[0] += 1
            counts = entry[1]
        else:
            counts = self._pending[order].get(context)
            if counts is None:
                self._pending[order][context] = {token: 1}
                return
        counts[token] = counts.get(token, 0) + 1

    def entry(self, order: int, context: int) -> list[Any]:
        """[total, {next: count}] of an order-``order`` context (pulled from the index on first use)."""

        entry = self._entries[order].get(context)
        if entry is not None:
            return entry
        counts: dict[int, int] = {}
        total = 0
        keys = self._keys[order]
        if keys is not None:
            lo, hi = keys.searchsorted(np.array([context << _BITS, (context << _BITS) | _LOW],
                                                dtype=np.uint64)).tolist()
            if hi > lo:
                nexts = (keys[lo:hi] & np.uint64(_LOW)).tolist()
                at = self._bounds[order][lo:hi + 1].tolist()
                total = at[-1] - at[0]
                counts = {t: at[i + 1] - at[i] for i, t in enumerate(nexts)}
            self.pulls += 1
        pending = self._pending[order].pop(context, None)
        if pending:
            for t, c in pending.items():
                counts[t] = counts.get(t, 0) + c
                total += c
        entry = [total, counts]
        self._entries[order][context] = entry
        return entry

    # -- scoring --------------------------------------------------------------------------
    def logp(self, history: Sequence[int], token: int) -> float:
        """log P(token | history's last 3 tokens), one at a time (tests and offline checks)."""

        prior = self.rescorer(np.array([[int(token)]], dtype=np.int64), 1.0)
        if prior is None:
            return float(np.log(1.0 / self.vocab))
        hist = ([None] * 3 + [int(t) for t in history])[-3:]
        return float(prior(tuple(hist), 0)[0])

    def rescorer(self, cands: np.ndarray, weight: float) -> Callable[[tuple, int], np.ndarray] | None:
        """For one lattice ``cands`` [D, K]: (history (t3, t2, t1), depth) -> weight * log P(cands[depth]).

        None when there is nothing to add (weight 0 or no text counted yet).
        """

        n = self.seen
        if not weight or not n or self.unigram is None:
            return None
        weight = float(weight)
        prior = self.prior
        base = 1.0 / self.vocab
        lam1 = n / (n + prior)
        uni = self.unigram[cands] / n                      # every depth's unigram P, once a round
        lists = cands.tolist()
        index: list[dict[int, int] | None] = [None] * len(lists)
        get2, get3, get4 = self._entries[2].get, self._entries[3].get, self._entries[4].get
        entry = self.entry
        log = np.log

        def bonus(history: tuple, depth: int) -> np.ndarray:
            t3, t2, t1 = history
            parts = []
            if t1 is not None:
                e = get2(t1) or entry(2, t1)
                if e[0]:
                    parts.append(e)
                    # a context never seen as an n-gram of one order lower (its parent's counts, when
                    # already pulled) was never followed by anything: no lookup
                    up = get2(t2) if t2 is not None else None
                    if t2 is not None and (up is None or t1 in up[1]):
                        c3 = (t2 << _BITS) | t1
                        e = get3(c3) or entry(3, c3)
                        if e[0]:
                            parts.append(e)
                            up = get3((t3 << _BITS) | t2) if t3 is not None else None
                            if t3 is not None and (up is None or t1 in up[1]):
                                c4 = key4(t3, t2, t1)
                                e = get4(c4) or entry(4, c4)
                                if e[0]:
                                    parts.append(e)
            rest = 1.0                                     # the weight left for the lower orders
            add: dict[int, float] = {}
            if parts:
                row = lists[depth]
                where = index[depth]
                if where is None:
                    where = index[depth] = {t: j for j, t in enumerate(row)}
                for total, counts in reversed(parts):     # highest order first
                    lam = total / (total + prior)
                    scale = rest * lam / total
                    if len(counts) < len(row) and len(where) == len(row):
                        for t, c in counts.items():
                            j = where.get(t)
                            if j is not None:
                                add[j] = add.get(j, 0.0) + scale * c
                    else:
                        for j, t in enumerate(row):
                            c = counts.get(t)
                            if c:
                                add[j] = add.get(j, 0.0) + scale * c
                    rest *= 1.0 - lam
            p = uni[depth] * (rest * lam1) + rest * (1.0 - lam1) * base
            for j, v in add.items():
                p[j] += v
            return weight * log(p)

        return bonus


__all__ = ["SessionNGram", "key4"]
