"""DFlash drafts for Gemma 4: the draft model reads the target's taps of committed rows and drafts one chain a round."""

from __future__ import annotations

from typing import Any

from tensorfold.families.qwen3_5.dflash_head import DFlashHead, _Context


class DFlashChains(DFlashHead):
    """The draft-head protocol over a ``DFlashDrafter`` without DFlash2's selector: its block's most likely tokens."""

    def __init__(self, drafter: Any) -> None:
        super().__init__(drafter, int(drafter.block_size) - 1, chains=True)

    def tree(self, cache: list[Any], position: int, sampling: Any, nodes: int) -> list[int]:
        """The stream's next ``nodes`` drafts after its anchor, a chain for positions ``position`` onwards."""

        proposer = cache[-1].get(sampling)
        return proposer.propose(_Context(int(position), cache[-1].anchor), int(nodes))


__all__ = ["DFlashChains"]
