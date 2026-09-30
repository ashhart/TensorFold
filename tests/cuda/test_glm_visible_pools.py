"""Visible-only radix scans retain lower-index ties, including dense-limit rows."""
import pytest
import torch

from tensorfold.families.glm5_next.cuda import sparse


@pytest.mark.parametrize("pos", [0, 2048, 2051, 4095, 205000])
@pytest.mark.parametrize("ties", [False, True])
def test_visible_top_pools_matches_full_reference(pos, ties):
    rows, width = 6, 65536
    torch.manual_seed(127)
    scores = torch.zeros((rows, width), device="cuda") if ties else torch.rand((rows, width), device="cuda")
    index = torch.arange(width, device="cuda")
    visible = (pos + torch.arange(rows, device="cuda") + 1) // 4
    scores.masked_fill_(index[None, :] >= visible[:, None], float("-inf"))
    position = torch.tensor([pos], device="cuda", dtype=torch.int32)
    want = sparse._top_pools(scores, 512)
    got = sparse.top_pools(scores, 512, position)
    assert torch.equal(want, got)
    for _ in range(3):
        sparse.top_pools(scores, 512, position)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = sparse.top_pools(scores, 512, position)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(want, replayed)
