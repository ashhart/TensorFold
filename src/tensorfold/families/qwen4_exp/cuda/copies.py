"""Flash Next's copies from the context: a window grows from 16 rows while copies land whole, halving after a break."""

from __future__ import annotations

from typing import Sequence

from tensorfold.families.qwen3_5.cuda.decode import CopyIndex, next_copy_rows


class Copies:
    """One reply's copies beside MTP chains of ``chain`` rows: its context's index and its window, up to ``most``."""

    def __init__(self, chain: int, most: int) -> None:
        self.index, self.chain, self.most = CopyIndex(), chain, most
        self.rows = next_copy_rows(chain, False, chain, most)
        self.copied = False                          # whether this round's drafts are a copy

    def propose(self, context: Sequence[int], left: int) -> list[int]:
        """A repeat of the context for the window, at most ``left`` tokens (none: an MTP chain drafts instead)."""

        got = self.index.propose(context, min(self.rows - 1, left))
        self.copied = bool(got)
        return got

    def landed(self, verified: int, kept: int) -> None:
        """After a round: a copy's window doubles when it kept every verified row, else halves."""

        if self.copied:
            self.rows = next_copy_rows(self.rows, kept == verified, self.chain, self.most)


def fit(live: Sequence, rows: int, chain: int) -> None:
    """A round's ``rows``: ``chain`` for every stream, copies past it sharing the rest, oldest stream first."""

    spare = rows - chain * len(live)
    for s in sorted(live, key=lambda x: x.sid):
        take = min(max(0, len(s.drafts) + 1 - chain), spare)
        s.drafts = s.drafts[:chain - 1 + take]
        spare -= take
