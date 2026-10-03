"""Two-rank kept-prefix forks and scheduler yields retain fresh serial token bytes."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_tp_multi import PROMPTS, _serial, _two_rank_run, ranks, release_decoder_cycles
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling


@pytest.mark.parametrize("sampling", [None, Sampling(seed=29, top_k=20, top_p=0.95)])
def test_shared_message_prefix_copies_into_a_free_slot_on_both_ranks(ranks, sampling):
    prefix = (PROMPTS[0] * 64)[:256]
    original, fork = prefix + [7] * 60, prefix + [9] * 70
    expected = _serial(ranks, fork, sampling, 16, "int8")

    def drive(dec):
        def finish(prompt):
            s = Stream(prompt, 16, sampling, stop_eos=False)
            dec.admit(s)
            while dec.live():
                dec.finish(dec.round())
            return s
        first = finish(original)
        chain = [(ids, st) for ids, st, _, _ in dec.kept if st is first.st]
        second = finish(fork)
        assert second.cached == 256 and second.st is not first.st
        assert all(any(ids == kept and st is slot for kept, slot, _, _ in dec.kept) for ids, st in chain)
        assert second.out == expected
        return second.out

    assert _two_rank_run(ranks, "int8", drive, points=lambda ids: [256]) == expected


def test_arrival_yields_after_one_pass_on_both_ranks_then_resumes_exactly(ranks):
    prompt = (PROMPTS[0] * 80)[:310]
    expected = _serial(ranks, prompt, None, 12, "bf16")

    def drive(dec):
        dec.arrived = lambda: True
        s = Stream(prompt, 12, stop_eos=False)
        dec.admit(s)
        dec.finish(dec.round())
        assert dec.fills[s.sid][2] == 256 and not s.out
        dec.arrived = lambda: False
        while dec.live():
            dec.finish(dec.round())
        return s.out

    assert _two_rank_run(ranks, "bf16", drive, prefill_rows=256) == expected


def test_two_rank_graph_slot_keeps_moved_message_snapshots_for_resume_and_fork(ranks):
    from tensorfold.families.qwen4_exp.cuda.multi_solo import solo

    sampling = Sampling(seed=61, top_k=20, top_p=0.95)
    first = [7] * 310
    resumed, forked = first + [13], first[:256] + [15] * 40
    refs = [_serial(ranks, ids, sampling, 12, "int8") for ids in (resumed, forked)]

    def setup(dec):
        dec.solo = solo(dec.w, dec.slots[0], dec.capacity, dec.depth, dec.pbuf)
        dec.solo.graphs = None                    # fake collectives run real eager kernels; NCCL captures run on boxes
        dec.solo_on = True
        dec._state_changed = lambda st: None

    def drive(dec):
        target = dec.solo.st
        def run(ids):
            s = Stream(ids, 12, sampling, stop_eos=False)
            dec.admit(s)
            while dec.live():
                dec.finish(dec.round())
            assert s.error is None
            return s
        run(first)
        run([9] * 320)
        assert dec.solo.st is target
        kept = [(ids, st) for ids, st, _, _ in dec.kept if ids == first[:256] or ids == first[:-1]]
        assert len(kept) == 2 and all(st is not target for _, st in kept)
        long = run(resumed)
        fork = run(forked)
        assert long.cached == len(first) - 1 and fork.cached == 256
        assert [long.out, fork.out] == refs
        assert any(ids == resumed[:-1] for ids, _, _, _ in dec.kept)
        return long.out, fork.out

    assert _two_rank_run(ranks, "int8", drive, setup=setup, points=lambda ids: [256]) == tuple(refs)
