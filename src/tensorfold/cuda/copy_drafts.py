"""Copy (prompt-lookup) drafts: when the context's last tokens occurred before, propose the tokens that followed them.

Model-agnostic: a drafted round verifies ``[pending, *drafts]`` like any other drafts, so replies stay equal to
serial decoding; a copy proposal only replaces the drafter's for that round. Replies that quote, edit or repeat
their context (code edits, structured output, tool arguments) accept long copies; prose rarely matches, and a round
without a match costs one small numpy search.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

MATCH = 8              # the context's last this many tokens must have occurred before
WINDOW = 1 << 16       # the earlier occurrence is searched for in the context's last this many tokens


@dataclass(frozen=True)
class CopySettings:
    match: int
    most: int

    @classmethod
    def from_env(cls, most: int, env=None) -> CopySettings | None:
        """Copy drafts unless TF_COPY_DRAFTS=0; TF_COPY_MATCH (default 8) and TF_COPY_MAX (default ``most``, the
        longest window the decode graphs verify) tune them."""

        env = os.environ if env is None else env
        if env.get("TF_COPY_DRAFTS", "1") == "0" or most < 1:
            return None
        match = int(env.get("TF_COPY_MATCH") or MATCH)
        cap = int(env.get("TF_COPY_MAX") or most)
        if not 2 <= match <= 64 or cap < 1:
            raise ValueError(f"TF_COPY_MATCH 2..64 and TF_COPY_MAX >= 1, not {match} / {cap}")
        return cls(match, min(cap, most))


class CopyDrafts:
    """One sequence's context (prompt, committed reply tokens, then the pending token) and its copy proposals."""

    def __init__(self, context, settings: CopySettings) -> None:
        self.match, self.most = settings.match, settings.most
        self._buf = np.empty(max(1024, 2 * len(context) + 64), dtype=np.int64)
        self._n = 0
        self.extend(context)

    def __len__(self) -> int:
        return self._n

    def extend(self, tokens) -> None:
        tokens = np.asarray(list(tokens), dtype=np.int64)
        need = self._n + len(tokens)
        if need > len(self._buf):
            grown = np.empty(max(need, 2 * len(self._buf)), dtype=np.int64)
            grown[:self._n] = self._buf[:self._n]
            self._buf = grown
        self._buf[self._n:need] = tokens
        self._n = need

    def truncate(self, n: int) -> None:
        """Back to the first ``n`` tokens (a round's pending token is replaced by the next round's)."""

        self._n = min(self._n, n)

    def propose(self, room: int | None = None) -> list[int]:
        """Up to ``min(most, room)`` drafts: what followed the latest earlier occurrence of the last ``match`` tokens
        that has that many tokens after it, else the most that followed any occurrence; [] without one."""

        want = self.most if room is None else min(self.most, room)
        n, m = self._n, self.match
        if want < 1 or n <= m:
            return []
        lo = max(0, n - WINDOW)
        ctx = self._buf[lo:n]
        suffix = ctx[-m:]
        # starts s (relative to lo) with ctx[s:s+m] == suffix and s + m < len(ctx) (a token follows the match)
        last = len(ctx) - m                               # the suffix's own start: excluded
        cand = np.flatnonzero(ctx[m - 1:last + m - 1] == suffix[-1])      # by the last token
        if cand.size == 0:
            return []
        ok = np.ones(cand.size, dtype=bool)
        for j in range(m - 1):                            # then the rest, a column at a time
            ok &= ctx[cand + j] == suffix[j]
        starts = cand[ok]
        if starts.size == 0:
            return []
        after = len(ctx) - (starts + m)                   # tokens available after each match
        full = starts[after >= want]
        if full.size:
            s = int(full[-1])                             # the latest occurrence with room for every draft
            take = want
        else:
            i = int(np.argmax(after))
            s, take = int(starts[i]), int(after[i])
        return ctx[s + m:s + m + take].tolist()
