"""A request admitted between real prompt passes runs first without changing any reply's solo bits."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA arrival integration needs an NVIDIA GPU", allow_module_level=True)

from test_flashnext_forward import _model
from test_flashnext_fork_lanes import fresh
from test_flashnext_prompt_cache import _prompt
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder


@pytest.mark.parametrize("dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=41, top_k=20, top_p=0.95)])
def test_a_short_arrival_joins_between_passes_and_every_reply_equals_fresh(dtype, sampling):
    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, stop_eos=False,
                       prefill_rows=128, kv_dtype=dtype)
    long = Stream(_prompt(603, seed=51), 14, sampling, stop_eos=False)
    dec.admit(long)
    dec.arrived = lambda: True
    dec.finish(dec.round())
    assert dec.fills[long.sid][2] == 128 and not long.out
    short = Stream(_prompt(73, seed=52), 14, sampling, stop_eos=False)
    dec.admit(short)
    dec.arrived = lambda: False
    dec.finish(dec.round())
    assert short.out and not long.out
    while dec.live():
        dec.finish(dec.round())
    for s in (long, short):
        assert s.out == fresh(w, s.prompt, sampling, dtype, 14)
    later = Stream(long.prompt + long.out + _prompt(7, seed=53), 12, sampling, stop_eos=False)
    dec.admit(later)
    assert later.cached == len(long.prompt) - 1
    while dec.live():
        dec.finish(dec.round())
    assert later.out == fresh(w, later.prompt, sampling, dtype, 12)
