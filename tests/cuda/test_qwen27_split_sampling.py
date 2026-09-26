"""Sampling from both vocabulary halves' candidates draws the same token as sampling the whole vocabulary."""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen3_5.cuda.decode_tp import choose_merged, split_candidates
from tensorfold.families.qwen3_5.cuda.sampling import sample_rows


@pytest.mark.parametrize("sampling", [None, Sampling(11, 1.0, 20, 0.95), Sampling(5, 0.7, 20, 0.9)])
def test_split_candidates_match_whole_vocabulary(sampling):
    gen = torch.Generator(device="cuda").manual_seed(2)
    rows, vocab = 64, 4096
    logits = (torch.randn(rows, vocab, generator=gen, device="cuda") * 3).bfloat16()
    logits[::3, 100:140] = logits[::3, 100:140].max()          # ties inside one half
    logits[1::4, [7, vocab // 2 + 7]] = 30.0                    # a tie for the top across the halves
    positions = list(range(1000, 1000 + rows))
    whole = sample_rows(logits, positions, sampling)
    half = vocab // 2
    parts = [split_candidates(logits[:, r * half:(r + 1) * half], sampling, r * half) for r in (0, 1)]
    values = torch.cat([parts[0][0], parts[1][0]], dim=1).cpu().numpy()
    ids = torch.cat([parts[0][1], parts[1][1]], dim=1).cpu().numpy().astype(np.int64)
    assert choose_merged(values, ids, positions, sampling) == whole
