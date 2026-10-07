"""Copy drafts for Flash Next on CUDA: a repeated tail drafts its earlier continuation, the same chain on both ranks."""

from __future__ import annotations

import os
from collections.abc import Sequence

COPY_MATCH = 8          # the context's last this many tokens must have appeared before, and at least this many follow


def enabled(default: bool = True) -> bool:
    """TENSORFOLD_COPY_DRAFTS=0 (or off, false, no) turns copy drafts off."""
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
        """Up to ``max_nodes`` tokens after the tail's latest earlier copy; [] without one or under ``min_match``."""

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
