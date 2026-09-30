"""Four-rank collectives: the exact reduce-scatter (all-to-all of fp32 column blocks, rank-order sum of the owned
block, all-gather of the result) gives the same bits as TensorFold's all-gather of fp32 partials summed in rank order.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from threadcomm import run_ranks  # noqa: E402

WORLD = 4


def _partials(rows: int, width: int) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(1234)
    return [torch.randn(rows, width, generator=g).mul_(10.0 ** (r - 1)).cuda() for r in range(WORLD)]


def _gather_sum(rank, comm, parts):
    """Today's contract: every rank gathers all fp32 partials and adds them in rank order."""
    part = parts[rank]
    recv = torch.empty((WORLD, *part.shape), dtype=torch.float32, device="cuda")
    comm.all_gather(part.contiguous(), recv)
    out = recv[0].clone()
    for r in range(1, WORLD):
        out += recv[r]
    return out


def _reduce_scatter_gather(rank, comm, parts):
    """The exact reduce-scatter: each rank sums (in rank order) only its column block, then the blocks are gathered."""
    part = parts[rank]
    rows, width = part.shape
    block = width // WORLD
    send = part.view(rows, WORLD, block).permute(1, 0, 2).contiguous().view(WORLD, rows * block)
    recv = torch.empty_like(send)
    comm.all_to_all(send, recv)
    mine = recv[0].clone()
    for r in range(1, WORLD):
        mine += recv[r]
    gathered = torch.empty((WORLD, rows * block), dtype=torch.float32, device="cuda")
    comm.all_gather(mine, gathered)
    return gathered.view(WORLD, rows, block).permute(1, 0, 2).reshape(rows, width)


@pytest.mark.parametrize("rows", [1, 3, 16, 2048])
def test_exact_reduce_scatter_matches_rank_order_gather(rows):
    parts = _partials(rows, 6144)
    ref = run_ranks(lambda r, c: _gather_sum(r, c, parts), WORLD)
    rs = run_ranks(lambda r, c: _reduce_scatter_gather(r, c, parts), WORLD)
    for r in range(WORLD):
        assert torch.equal(ref[r], ref[0]), "ranks disagree under all-gather"
        assert torch.equal(rs[r], ref[0]), f"rank {r}: reduce-scatter bits differ from the rank-order all-gather sum"


def test_all_to_all_routes_rows():
    def body(rank, comm):
        send = torch.stack([torch.full((5,), 10.0 * rank + j, device="cuda") for j in range(WORLD)])
        recv = torch.empty_like(send)
        comm.all_to_all(send, recv)
        return recv.cpu()

    out = run_ranks(body, WORLD)
    for rank in range(WORLD):
        for j in range(WORLD):
            assert torch.all(out[rank][j] == 10.0 * j + rank)
