"""DFlash2 as a family model's draft head: each stream's drafter state rides as the last entry of its cache list.

A stream's lattice reads the target's taps of the rows the stream kept, and its best-first tree is the next round's
drafts: (tokens, parents), parents indexing the drafts (-1: the pending row), most likely first, each with its chance
of landing (``draft_probabilities``) so the engine can split a shared round's rows across streams.
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import os
from typing import Any, Sequence

import mlx.core as mx

_numbers = itertools.count(1)


class DraftSlot:
    """A stream's DFlash2 proposer, its cache list's last entry; a copy keeps only prompt taps not read yet."""

    keys = None
    stored = False                                       # left out of snapshots on disk; adopt_cache adds a new one

    def __init__(self, drafter: Any, chains: bool = False) -> None:
        self.drafter = drafter
        self.chains = bool(chains)                       # one child a node: the tree search draws a chain
        self.proposer: Any = None
        self.anchor = 0
        self.kept: list[int] = []                        # the last round's committed tokens, the anchor last
        self.chances: list[float] | None = None          # the last drafts' chances of landing
        self.number = next(_numbers)

    def _unread(self) -> Any:
        """The proposer while it holds only a prompt's taps (a prefill's state), else None."""

        proposer = self.proposer
        if self.kept or not getattr(proposer, "ready", False) or getattr(proposer, "context", None) is None:
            return None
        return proposer

    @property
    def state(self) -> list[Any]:
        unread = self._unread()
        return [] if unread is None else [unread.context]

    def __copy__(self) -> "DraftSlot":
        slot = DraftSlot(self.drafter, self.chains)
        unread = self._unread()
        if unread is not None:
            proposer = slot.get(unread.sampling)
            proposer.context, proposer.ready = unread.context, True
            for mine, theirs in zip(proposer.cache, unread.cache):
                mine.offset = theirs.offset
        return slot

    def get(self, sampling: Any) -> Any:
        if self.proposer is None:
            proposer = self.drafter.proposer(copy=None, sampling=sampling)
            proposer.ngram_weight = 0.0                  # the prior and the traces read the whole token context
            proposer.trace_path = proposer.capture_dir = ""
            if self.chains:
                # DFlash2's own chain: its trained block, one child a node, unary + pairwise / T + the keyed noise
                proposer.tree_children, proposer.tree_edge, proposer.tree_noise = 1, 1.0, 1.0
                proposer.tree_block = int(self.drafter.block_size)
            self.proposer = proposer
        self.proposer.sampling = sampling
        return self.proposer


class _Context:
    """What a lattice reads of a stream's tokens: their count and the last one (the anchor)."""

    __slots__ = ("length", "last")

    def __init__(self, length: int, last: int) -> None:
        self.length, self.last = int(length), int(last)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self.last] if (index.start or 0) < 0 else []     # history for a prior, which this head leaves off
        if index == -1 or index == self.length - 1:
            return self.last
        raise IndexError("a draft context holds only its last token")


class DFlashHead:
    """The draft-head half of the family protocol over a ``DFlashDrafter``."""

    def __init__(self, drafter: Any, nodes: int = 15, calibration: dict[str, Any] | None = None,
                 chains: bool = False) -> None:
        self.drafter = drafter
        self.nodes = int(nodes)
        self.chains = bool(chains)
        self.calibration = dict(calibration or {})     # "greedy" / "sampled" -> drafters.calibration.Calibration
        self.log_path = os.environ.get("TF_DRAFT_LOG", "")

    def slot(self) -> DraftSlot:
        return DraftSlot(self.drafter, self.chains)

    def absorb(self, cache: list[Any], first: int, rows: int, row: int = 0) -> None:
        """The last forward's rows ``row`` .. ``row + rows`` - 1 (positions ``first`` ..) into the stream's drafter
        context."""

        taps = self.drafter.taps()
        if taps is None:
            return
        taps = taps[:, row:row + rows]
        proposer = cache[-1].get(getattr(cache[-1].proposer, "sampling", None))
        window = self.drafter.window
        if not proposer.ready:
            if window and rows > window:
                taps, first = taps[:, -window:], first + rows - window
            for item in proposer.cache:
                item.offset = first
            proposer.context, proposer.ready = mx.contiguous(taps), True
            return
        proposer.absorb(taps)
        held = int(proposer.context.shape[1])
        if window and held > window:
            proposer.context = proposer.context[:, held - window:]
            for item in proposer.cache:
                item.offset += held - window

    def read(self, cache: list[Any], rows: Sequence[int], follow: Sequence[int], sampling: Any) -> None:
        """The stream's kept ``rows`` of the last forward into its drafter context; ``follow``: the tokens committed
        after them, the new pending token last."""

        proposer = cache[-1].get(sampling)
        taps = self.drafter.taps()
        if taps is not None and rows:
            proposer.absorb(mx.take(taps, mx.array([int(r) for r in rows], dtype=mx.int32), axis=1))
        cache[-1].kept = [int(t) for t in follow]
        cache[-1].anchor = cache[-1].kept[-1]

    def tree(self, cache: list[Any], position: int, sampling: Any, nodes: int) -> Any:
        """The stream's next drafts after its anchor, for positions ``position`` ..: a tree of up to ``nodes``."""

        proposer = cache[-1].get(sampling)
        context = _Context(int(position), cache[-1].anchor)
        tree = proposer._finish_tree(context, proposer._start_tree(context, self.nodes))
        return self._drafts(cache[-1], tree, position, sampling, nodes)

    def probabilities(self, cache: list[Any]) -> list[float] | None:
        return cache[-1].chances

    def _drafts(self, slot: DraftSlot, tree: Any, position: int, sampling: Any, nodes: int) -> Any:
        """The tree most likely first (parents before children) as drafts; each node's chance to ``slot.chances``:
        the calibrated table's for its depth and path score, else the drafter's own (e^score)."""

        tokens, parents = [int(t) for t in tree[0]], [int(q) for q in tree[1]]
        scores = slot.proposer.last_scores if slot.proposer is not None else None
        if scores is None or len(scores) != len(tokens):
            slot.chances = None
            return as_drafts((tokens, parents), nodes)
        if self.log_path:
            record = {"stream": slot.number, "position": int(position), "greedy": sampling is None,
                      "kept": slot.kept, "tokens": tokens, "parents": parents, "scores": [round(x, 4) for x in scores]}
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")
        table = self.calibration.get("greedy" if sampling is None else "sampled")
        chances = table.probabilities(parents, scores) if table is not None else [math.exp(x) for x in scores]
        tokens, parents, chances = by_chance(tokens, parents, chances)
        slot.chances = chances[:nodes]
        return as_drafts((tokens, parents), nodes)

    def draft_streams(self, caches: Sequence[list[Any]], follows: Sequence[Sequence[int]], rows: Sequence[Sequence[int]],
                      positions: Sequence[int], samplings: Sequence[Any], depths: Sequence[int]) -> list[Any]:
        """``draft`` for every stream of a shared round, their lattices in one drafter forward."""

        from tensorfold.drafters.dflash_batch import start_trees

        items = []
        nodes = max(1, min(self.nodes, max(depths, default=self.nodes)))     # the widest budget sets every block
        for cache, follow, kept, position, sampling in zip(caches, follows, rows, positions, samplings):
            self.read(cache, kept, follow, sampling)
            items.append((cache[-1].proposer, _Context(int(position), int(follow[-1])), nodes))
        states = start_trees(self.drafter, items)     # one block length for all: one forward
        return [self._drafts(cache[-1], proposer._finish_tree(context, state), position, sampling, depth)
                for cache, (proposer, context, _), state, position, sampling, depth
                in zip(caches, items, states, positions, samplings, depths)]


def by_chance(tokens: Sequence[int], parents: Sequence[int], chances: Sequence[float]
              ) -> tuple[list[int], list[int], list[float]]:
    """A tree re-ordered most likely first, parents before children; a node's chance is at most its parent's."""

    capped: list[float] = []
    children: dict[int, list[int]] = {}
    for i, (q, p) in enumerate(zip(parents, chances)):
        capped.append(min(float(p), capped[q]) if q >= 0 else float(p))
        children.setdefault(q, []).append(i)
    heap = [(-capped[i], i) for i in children.get(-1, [])]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        _, i = heapq.heappop(heap)
        order.append(i)
        for c in children.get(i, []):
            heapq.heappush(heap, (-capped[c], c))
    place = {old: new for new, old in enumerate(order)}
    return ([int(tokens[i]) for i in order], [-1 if parents[i] < 0 else place[parents[i]] for i in order],
            [capped[i] for i in order])


def as_drafts(tree: tuple[list[int], list[int]], nodes: int) -> Any:
    """A best-first tree's first ``nodes`` pops as the family rounds take drafts: a list for a chain, else (tokens,
    parents)."""

    tokens, parents = [int(t) for t in tree[0][:nodes]], [int(q) for q in tree[1][:nodes]]
    return tokens if parents == list(range(-1, len(tokens) - 1)) else (tokens, parents)


__all__ = ["DFlashHead", "DraftSlot", "as_drafts", "by_chance"]
