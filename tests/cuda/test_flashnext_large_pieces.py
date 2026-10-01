"""Larger prompt pieces preserve committed state bytes and a resume cut inside the piece."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA state comparisons require an NVIDIA GPU", allow_module_level=True)

from test_flashnext_forward import _model
from test_flashnext_prompt_cache import _assert_same_state, _prompt, _same_bits, _same_snap
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=31, top_k=20, top_p=0.95)])
def test_2048_and_4096_pieces_keep_state_and_interior_resume_bits(mtp, sampling):
    w = _model()
    prompt, cut = _prompt(5003), 3001
    small = Engine(w, capacity=8192, max_rows=8, prefill_rows=2048)
    large = Engine(w, capacity=8192, max_rows=8, prefill_rows=4096)
    first = prefill(small, prompt, sampling, mtp=mtp, keep_at=cut)
    assert prefill(large, prompt, sampling, mtp=mtp, keep_at=cut) == first
    _assert_same_state(small.st, large.st)
    assert _same_snap(small.kept["state"], large.kept["state"])
    assert _same_bits(small.last_streams, large.last_streams)
    fresh = Engine(w, capacity=8192, max_rows=8, prefill_rows=2048)
    prefill(fresh, prompt[:cut], sampling, mtp=mtp)
    assert _same_snap(large.kept["state"], fresh.st.snapshot())
    if mtp:
        assert _same_bits(large.kept["tail"], fresh.last_streams)
    extended = prompt[:cut] + _prompt(111, seed=7)
    got = prefill(large, extended, sampling, mtp=mtp, resume=large.kept)
    want = prefill(fresh, extended, sampling, mtp=mtp)
    assert got == want
    _assert_same_state(large.st, fresh.st)
    assert serial_decode(large, got, 12, sampling, stop_eos=False).tokens == \
           serial_decode(fresh, want, 12, sampling, stop_eos=False).tokens
