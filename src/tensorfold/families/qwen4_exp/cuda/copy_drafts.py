"""Copy drafts for Flash Next on CUDA: when the reply's last tokens repeat earlier text, draft that text's continuation.

A stream that quotes, edits or refactors its prompt lands far more drafts a round this way than its MTP head does
(the head stops a chain at its confidence floor; a copied continuation is proposed whole, up to the round's depth),
and the verify step keeps replies byte-identical either way. ``CopyIndex`` is Nemotron's (``nemotron_h/cuda/decode.py``):
n-gram starts keep each round O(new). Both ranks hold the same tokens, so both propose the same chain.
``TENSORFOLD_COPY_DRAFTS=0`` turns copy drafts off."""

from __future__ import annotations

import os
from collections.abc import Sequence

COPY_MATCH = 8          # the context's last this many tokens must have appeared before, and at least this many follow


def enabled(default: bool = True) -> bool:
    value = os.environ.get("TENSORFOLD_COPY_DRAFTS")
    return default if value is None else value.strip().lower() not in ("0", "off", "false", "no")


class CopyIndex:
    """Longest continuation of an earlier copy of the last ``min_match`` tokens; n-gram starts keep a round O(new)."""

    def __init__(self, context: Sequence[int], min_match: int = COPY_MATCH):
        self.m = int(min_match)
        self.ctx: list[int] = []
        self.starts: dict[tuple, list[int]] = {}
        self.extend(context)

    def extend(self, tokens: Sequence[int]) -> None:
        for t in tokens:
            self.ctx.append(int(t))
            s = len(self.ctx) - self.m
            if s >= 0:
                self.starts.setdefault(tuple(self.ctx[s:]), []).append(s)

    def chain(self, max_nodes: int) -> list[int]:
        """Up to ``max_nodes`` tokens that followed the latest earlier copy of the context's tail, or [] (fewer than
        ``min_match`` would follow, or no copy)."""

        ctx, m = self.ctx, self.m
        if len(ctx) < 2 * m or max_nodes < m:
            return []
        best: list[int] = []
        for start in reversed(self.starts.get(tuple(ctx[-m:]), [])):
            if start > len(ctx) - m - 1:                 # the needle itself
                continue
            cont = ctx[start + m:start + m + max_nodes]
            if len(cont) > len(best):
                best = cont
                if len(best) == max_nodes:
                    break
        return best if len(best) >= m else []
