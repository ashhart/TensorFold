"""Spare-slot forks keep cache bytes, original chains and each concurrent reply exact."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA fork integration needs an NVIDIA GPU", allow_module_level=True)

from test_flashnext_forward import _model
from test_flashnext_prompt_cache import _assert_same_state, _prompt, _same_bits, _same_snap
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder


def fresh(w, prompt, sampling, dtype, count):
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=128, kv_dtype=dtype)
    first = prefill(e, prompt, sampling)
    return serial_decode(e, first, count, sampling, stop_eos=False).tokens


@pytest.mark.parametrize("dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=41, top_k=20, top_p=0.95)])
@pytest.mark.parametrize("busy", [False, True])
def test_three_forks_preserve_the_source_and_equal_fresh_serial(dtype, sampling, busy):
    w = _model()
    w.cfg.index_budget = 64
    prefix = _prompt(603, seed=9)
    one = prefix + _prompt(15, seed=11)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, stop_eos=False,
                       prefill_rows=256, kv_dtype=dtype, points=lambda ids: [len(prefix)])
    original = Stream(one, 32, sampling, stop_eos=False)
    dec.admit(original)
    if busy:
        while not dec.streams:
            dec.finish(dec.round())
    else:
        while dec.live():
            dec.finish(dec.round())
    source = original.st
    before = source.clone()
    kept = next(k for k in dec.kept if k[0] == prefix)
    snap = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy() if hasattr(v, "copy") else v
            for k, v in kept[2].items()}
    tail = kept[3].clone()
    forks = [Stream(prefix + _prompt(11 + i, seed=20 + i), 14, sampling, stop_eos=False) for i in range(3)]
    for s in forks:
        dec.admit(s)
        assert s.cached == len(prefix) and s.st is not source
        assert _same_snap(s.st.snapshot(), {**snap, "mtp_len": snap["mtp_len"] + 1})
        for got, want in zip(s.st.pooled, source.pooled):
            assert _same_bits(got[:len(prefix) // s.st.ratio], want[:len(prefix) // s.st.ratio])
    _assert_same_state(source, before)
    assert _same_snap(kept[2], snap) and _same_bits(kept[3], tail)
    while dec.live():
        dec.finish(dec.round())
    for s in forks:
        assert s.out == fresh(w, s.prompt, sampling, dtype, 14)
    assert original.out == fresh(w, one, sampling, dtype, 32)
    if not busy:
        _assert_same_state(source, before)
    assert _same_snap(kept[2], snap) and _same_bits(kept[3], tail)
    later = Stream(one + original.out + _prompt(7, seed=31), 12, sampling, stop_eos=False)
    dec.admit(later)
    assert later.cached == len(one) - 1
    while dec.live():
        dec.finish(dec.round())
    assert later.out == fresh(w, later.prompt, sampling, dtype, 12)
