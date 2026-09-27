"""Concurrent Flash Next keeps every stream slot: replacing a kept prompt never loses the displaced state."""

import importlib
from types import SimpleNamespace

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)


def decoder(module, free, kept, keep=8):
    dec = module.MultiDecoder.__new__(module.MultiDecoder)
    dec.streams, dec.free, dec.kept, dec.keep = {}, list(free), list(kept), keep
    return dec


def test_the_same_prompt_twice_keeps_both_slots(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    old, fresh = object(), object()
    prompt = [1, 2, 3]
    dec = decoder(multi, [fresh], [(prompt, old, {}, None)])
    chosen, resume, cached = dec._slot_for(prompt, True)          # equal, not a strict prefix: a fresh prefill
    assert chosen is fresh and resume is None and cached == 0
    dec._remember(prompt, chosen, {}, None)
    dec.streams = {0: SimpleNamespace(st=chosen)}
    other, _, _ = dec._slot_for([9, 9], True)                      # the second slot is still there
    assert other is old


def test_a_displaced_state_shared_or_busy_stays_out_of_the_free_list(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    old, fresh, busy = object(), object(), object()
    dec = decoder(multi, [], [([1], old, {}, None), ([2], old, {}, None), ([3], busy, {}, None)], keep=8)
    dec.streams = {0: SimpleNamespace(st=busy)}
    dec._remember([1], fresh, {}, None)                            # old still backs [2]: not free
    assert old not in dec.free
    dec._remember([3], fresh, {}, None)                            # busy is a live stream's: not free
    assert busy not in dec.free
