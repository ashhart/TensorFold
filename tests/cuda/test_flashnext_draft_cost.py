"""Flash Next's expected-time draft stop (``--mtp-cost``): a later draft is verified only while the product of the
head's probabilities of drafts 1..j repays the ms it adds to the round, priced on the engine's own measured round
costs. Drafts change speed only: drafted output equals serial output, greedy and sampled, at any cost."""

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.decode import cost_bars


def test_cost_bars_price_each_draft_by_the_ms_it_adds() -> None:
    verify, step = (30.0, 34.0, 38.5, 41.0), 1.5
    bars = cost_bars(0.1, 4, verify, step)
    want = [0.1 * (x + step) for x in (4.0, 4.5, 2.5, 2.5, 2.5)]      # past the table: its last step again
    assert [round(b, 9) for b in bars] == [round(w, 9) for w in want]
    assert cost_bars(0.0, 2, verify, step) == [0.0, 0.0, 0.0]


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@cuda
def test_measured_round_costs_are_positive_ordered_and_leave_the_state_empty() -> None:
    from test_flashnext_forward import _model

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, measure_round_costs, prefill

    w = _model()
    e = Engine(w, capacity=512, max_rows=8, prefill_rows=16)
    prompt = [5, 17, 99, 250, 7, 64]
    first = prefill(e, prompt, None)
    verify, step = measure_round_costs(e, 7, reps=2)
    assert len(verify) == 7 and all(v > 0 for v in verify) and step >= 0
    assert all(b >= a for a, b in zip(verify, verify[1:]))           # a wider window never prices below a narrower
    assert e.st.pos == 0 and e.st.mtp_len == 0                       # emptied, as warm leaves it
    assert prefill(e, prompt, None) == first


@cuda
@pytest.mark.parametrize("sampling_seed", [None, 1234])
def test_drafts_under_the_cost_stop_give_the_serial_tokens(sampling_seed) -> None:
    from test_flashnext_forward import _model

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    w = _model()
    sampling = None if sampling_seed is None else Sampling(seed=sampling_seed, temperature=0.7, top_k=20, top_p=0.8)
    prompt = [5, 17, 99, 250, 7, 64, 30, 11, 12, 13]
    e = Engine(w, capacity=512, max_rows=8, prefill_rows=16)
    e.round_costs = ((24.0, 28.0, 32.0, 35.0, 38.0, 42.0, 44.0), 1.2)  # any table: the stop moves speed only
    first = prefill(e, prompt, sampling)
    ref = serial_decode(e, first, 32, sampling).tokens
    for depth in (1, 3, 6):
        for confidence in (0.0, 0.7):
            for cost in (0.02, 0.2, 2.0):
                assert prefill(e, prompt, sampling) == first
                got = mtp_decode(e, first, 32, sampling, depth=depth, confidence=confidence, cost=cost)
                assert got.tokens == ref, (depth, confidence, cost)
                if cost >= 2.0:                     # an unpayable bar (> 1): every round verifies its first draft only
                    assert max(got.widths) <= 2
