"""The CUDA target sampler is invariant to the other rows in a verify window."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen3_5.cuda.sampling import sample_rows


@pytest.mark.parametrize("sampling", [None, Sampling(seed=7382, top_k=20, top_p=0.95)])
def test_rows_match_single_row(sampling):
    torch.manual_seed(841)
    logits = torch.randn((128, 4096), device="cuda", dtype=torch.bfloat16)
    logits[:, 2:6] = 4.0  # tied leaders exercise token-ID tie handling
    positions = list(range(312, 440))
    wide = sample_rows(logits, positions, sampling)
    alone = [sample_rows(logits[i:i + 1], positions[i:i + 1], sampling)[0]
             for i in range(128)]
    assert wide == alone
