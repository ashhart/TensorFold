"""The CUDA target sampler is invariant to the other rows in a verify window."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.cuda.sampling import sample_rows


@pytest.mark.parametrize("sampling", [None, Sampling(seed=7382, top_k=20, top_p=0.95),
                                      Sampling(seed=7382, top_k=20, top_p=0.95, min_p=0.1)])
def test_rows_match_single_row(sampling):
    torch.manual_seed(841)
    logits = torch.randn((128, 4096), device="cuda", dtype=torch.bfloat16)
    logits[:, 2:6] = 4.0  # tied leaders exercise token-ID tie handling
    positions = list(range(312, 440))
    wide = sample_rows(logits, positions, sampling)
    alone = [sample_rows(logits[i:i + 1], positions[i:i + 1], sampling)[0]
             for i in range(128)]
    assert wide == alone


def test_streams_match_their_own_calls():
    from tensorfold.cuda.sampling import sample_streams

    torch.manual_seed(5)
    vocab = 248320
    # bf16 logits rounded to a coarse grid: many ties at every candidate boundary
    logits = (torch.randn((520, vocab), device="cuda") * 2).mul(4).round().div(4).to(torch.bfloat16)
    logits[:, 100:140] = 6.0
    samplings = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(99, 0.7, 20, 1.0, 0.2), Sampling(5, 1.0, 40, 0.9),
                 None, Sampling(8, 1.0, 20, 0.95, 0.05)] * 5
    sizes = [16, 12, 1, 16, 3, 7] * 5
    starts = [0]
    for n in sizes[:len(samplings)]:
        starts.append(starts[-1] + n)
    positions = [list(range(50 + a, 50 + b)) for a, b in zip(starts, starts[1:])]
    got = sample_streams(logits, starts, positions, samplings)
    for s, smp in enumerate(samplings):
        want = sample_rows(logits[starts[s]:starts[s + 1]], positions[s], smp)
        assert got[s] == want, s
