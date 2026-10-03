"""A serial run's stop on every rank: rank 0's wish rides as one float32 word on the next verify sample's all-gather."""

from __future__ import annotations

import torch


class StopVote:
    """Wraps ``on_tokens``: a true result is this rank's wish to stop; ``stop`` is the decision every rank shares."""

    def __init__(self, on_tokens=None) -> None:
        self.fn = on_tokens
        self.mine = False
        self.agreed = False
        self.flags: torch.Tensor | None = None

    def __call__(self, tokens) -> bool:
        if self.fn is not None and self.fn(tokens):
            self.mine = True
        return self.mine

    def gather(self, gather):
        """``gather`` with this rank's wish as one word more on its first call, given back without that word."""

        first = [True]

        def voting(words: torch.Tensor) -> torch.Tensor:
            if not first[0]:
                return gather(words)
            first[0] = False
            flag = torch.full((1,), 1.0 if self.mine else 0.0, dtype=words.dtype, device=words.device)
            got = gather(torch.cat([words.reshape(-1), flag]))
            self.flags = got[:, -1]
            return got[:, :-1].contiguous()

        return voting

    @property
    def stop(self) -> bool:
        if self.flags is not None:
            self.agreed = self.agreed or bool((self.flags > 0).any())
            self.flags = None
        return self.agreed


def stopped(e) -> bool:
    """Whether every rank agreed to end this run; False outside a voting run."""

    vote = getattr(e, "vote", None)
    return vote is not None and vote.stop
