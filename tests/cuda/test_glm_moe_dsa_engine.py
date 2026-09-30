"""Full GLM-5.3 M2 engine protocol on real weights (first layers only), four thread ranks on one GPU: rank 0 serves a
request, ranks 1-3 follow it through the shared header and prompt, and the reply is reproducible. Needs TF_GLM53_CKPT."""

from __future__ import annotations

import os

import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")]

from threadcomm import run_ranks  # noqa: E402


def _serve(rank, comm, sampling):
    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    e = Glm53Engine(CKPT, rank=rank, master="", port=0, context=256, comm=comm, layers=4)
    if rank:
        e.follow(requests=1)
        return None
    got = []
    stats = e.generate([151331, 151333, 9707, 11, 1246, 525, 498, 30], 6, sampling, got.extend, stop_eos=False)
    return got, stats


@pytest.mark.parametrize("sampled", [False, True])
def test_four_ranks_serve_and_follow(sampled):
    from tensorfold.engine.exact_sampling import Sampling

    s = Sampling(1234, 0.8, 20, 0.95, 0.0) if sampled else None
    first = run_ranks(lambda r, c: _serve(r, c, s), 4)[0]
    torch.cuda.empty_cache()
    again = run_ranks(lambda r, c: _serve(r, c, s), 4)[0]
    assert len(first[0]) == 6
    assert first[0] == again[0], (first[0], again[0])


def _drafted_vs_serial(rank, comm, sampling):
    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    e = Glm53Engine(CKPT, rank=rank, master="", port=0, context=512, comm=comm, layers=4, mtp_drafts=2)
    prompt = [151331, 151333, 9707, 11, 1246, 525, 498, 30, 3555, 374, 279]
    if rank:
        e.follow(requests=2)
        return None
    serial, drafted = [], []
    e.generate(prompt, 24, sampling, serial.extend, draft=False, stop_eos=False)
    stats = e.generate(prompt, 24, sampling, drafted.extend, draft=True, stop_eos=False)
    return serial, drafted, stats


@pytest.mark.parametrize("sampled", [False, True])
def test_mtp_drafted_reply_equals_serial(sampled):
    """M4 exactness: MTP drafts verified in a window give exactly the serial reply (greedy and sampled)."""
    from tensorfold.engine.exact_sampling import Sampling

    s = Sampling(99, 0.9, 20, 0.95, 0.0) if sampled else None
    serial, drafted, stats = run_ranks(lambda r, c: _drafted_vs_serial(r, c, s), 4)[0]
    assert len(serial) == len(drafted) == 24
    assert drafted == serial, (serial, drafted, stats)
    assert stats["rounds"] < 23                           # some drafts were accepted (tokens per round > 1)
