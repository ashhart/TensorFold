"""A retained large graph slot accepts a smaller live stream without changing its decoded tokens."""

import gc

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _model
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
def test_smaller_live_stream_moves_into_retained_graph_rows_and_matches_serial(kv_dtype):
    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, prefill_rows=64, kv_dtype=kv_dtype)
    target = dec.solo.st
    assert dec._grow(target, 300) and target.capacity == 1024
    first = Stream([5, 17, 99, 250], 1, draft=False, stop_eos=False)
    sampling = Sampling(seed=23, top_k=20, top_p=0.95)
    second = Stream([1023, 7, 64, 300, 11, 12], 24, sampling, stop_eos=False)
    dec.admit(first)
    dec.admit(second)
    assert first.st is target and second.st is not target and second.st.capacity == 256
    moved = []
    original = target.copy_from
    def copy(other):
        moved.append((target.capacity, other.capacity))
        original(other)
    target.copy_from = copy
    events = []
    while dec.live():
        dec.finish(dec.round())
        events.append((len(second.out), second.error, dec.solo.st is target, len(dec.streams),
                       [(len(ids), st is target) for ids, st, _, _ in dec.kept]))
    assert first.error is None and second.error is None, events
    assert (1024, 256) in moved and second.st is target, events
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, kv_dtype=kv_dtype)
    assert second.out == serial_decode(e, prefill(e, second.prompt, sampling), second.count, sampling).tokens
    del dec, e
    gc.collect()
