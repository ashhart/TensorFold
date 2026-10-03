"""Flash Next's expected-time draft stop (``--mtp-cost``): a later draft is verified only while its chance of being
kept, the head's chain product calibrated by live rounds, repays the ms it adds to the round. Drafts change speed
only: drafted output equals serial output, greedy and sampled, at any price."""

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.draft_price import RATE, DraftPrice


def test_each_draft_is_priced_by_the_ms_it_adds() -> None:
    price = DraftPrice(0.1, (30.0, 34.0, 38.5, 41.0), 1.5, 5)
    assert [round(price.adds(j), 9) for j in range(5)] == [5.5, 6.0, 4.0, 4.0, 4.0]   # past the table: its last step
    assert price.pays(0, 0.6) and not price.pays(0, 0.5)                              # 0.1 x 5.5 ms = 0.55 tokens


def test_live_rounds_calibrate_the_chain_product_per_depth_and_mode() -> None:
    price = DraftPrice(0.05, (30.0, 34.0, 38.0), 1.0, 2)
    assert price.chance(1, 0.4) == 0.4                         # unseen: the head's product
    assert price.pays(1, 0.4)
    for _ in range(400):
        price.begin(False)
        price.products = [0.8, 0.4]
        price.observe(2, 1)                                    # depth 1 always kept, depth 2 never
    assert price.chance(0, 0.8) == pytest.approx(1.0, abs=1e-6)
    assert price.chance(1, 0.4) < 0.01 and not price.pays(1, 0.4)
    price.begin(True)
    assert price.chance(1, 0.4) == 0.4                         # sampled rounds keep their own counts


def test_unverified_drafts_teach_nothing() -> None:
    price = DraftPrice(0.05, (30.0, 34.0, 38.0), 1.0, 2)
    price.begin(False)
    price.products = [0.9, 0.5]
    price.observe(1, 0)                                        # a window cut the second draft before verifying it
    assert price.kept[False][1] is None and price.kept[False][0] == pytest.approx(0.9 * (1 - RATE))


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@cuda
def test_measured_round_costs_are_positive_ordered_and_leave_the_state_empty() -> None:
    from test_flashnext_forward import _model

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.draft_price import measure_round_costs

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
    first = prefill(e, prompt, sampling)
    ref = serial_decode(e, first, 32, sampling).tokens
    for depth in (1, 3, 6):
        for confidence in (0.0, 0.7):
            for cost in (0.0, 0.02, 0.2, 2.0):                       # any table: the stop moves speed only
                price = DraftPrice(cost, (24.0, 28.0, 32.0, 35.0, 38.0, 42.0, 44.0), 1.2, depth)
                for _ in range(2):                                   # the second reply runs on calibrated counts
                    assert prefill(e, prompt, sampling) == first
                    got = mtp_decode(e, first, 32, sampling, depth=depth, confidence=confidence, price=price)
                    assert got.tokens == ref, (depth, confidence, cost)
                    if cost >= 2.0:                 # an unpayable bar (> 1): every round verifies its first draft only
                        assert max(got.widths) <= 2
