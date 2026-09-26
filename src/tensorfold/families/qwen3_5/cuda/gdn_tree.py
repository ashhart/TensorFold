"""One warp per value row walks GDN trees and replays committed paths.

The CUDA kernel owns the arithmetic for both serial and tree steps. Parents must
precede children. A chain keeps one state slot; larger branching trees traverse
device-built depth-first preorder with 32 state slots for root depth 0..31.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    here = Path(__file__).parent
    return load(
        name="tensorfold_gdn_tree_v3",
        sources=[str(here / "gdn_tree.cpp"), str(here / "gdn_tree.cu")],
        extra_cuda_cflags=["-O3", "--fmad=false"],
        verbose=False,
    )


def tree(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
         beta: torch.Tensor, state: torch.Tensor, parents: torch.Tensor, *, chain: bool = False) -> torch.Tensor:
    """Return node outputs (W, Hv, Dv) from the untouched committed state.

    Parents are topologically ordered with a single root at row zero. For a
    branching tree with more than 32 nodes, the longest path has at most 32
    nodes; a 128-node chain uses ``chain=True`` instead.
    """

    return _extension().tree(q, k, v, g, beta, state, parents, chain)


def replay(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
           beta: torch.Tensor, state: torch.Tensor, rows: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Return the state after the accepted rows, in path order; count stays on device."""

    return _extension().replay(q, k, v, g, beta, state, rows, count)


def replay_many(q: list[torch.Tensor], k: list[torch.Tensor], v: list[torch.Tensor], g: list[torch.Tensor],
                beta: list[torch.Tensor], state: list[torch.Tensor], rows: torch.Tensor,
                count: torch.Tensor) -> torch.Tensor:
    """``replay`` for every layer in one launch: (layers, Hv, Dv, Dk) states, bit-identical to per-layer calls."""

    return _extension().replay_many(q, k, v, g, beta, state, rows, count)
