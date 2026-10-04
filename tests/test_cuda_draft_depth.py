"""The draft depth rule: a draft is verified while its calibrated chance of landing pays for its row's cost."""

import pytest

from tensorfold.cuda.draft_depth import Costs, DepthRule


def _costs(rows=(10.0, 12.0, 14.0, 16.0, 17.0, 18.0), level=0.5):
    return Costs((0.0,) + tuple(rows), level)


def test_a_row_costs_what_it_adds_to_its_window():
    c = _costs()
    assert [c.row(k) for k in range(1, 5)] == [2.0, 2.0, 2.0, 1.0]
    assert Costs((0.0, 10.0, 9.0), 0.5).row(1) == 0.0          # timing noise never makes a row pay you


def test_measured_rows_never_get_dearer_with_more_rows():
    c = Costs.measured((0.0, 10.0, 12.5, 15.1, 16.3, 18.0), -0.1)
    assert c.verify == pytest.approx((0.0, 10.0, 12.55, 15.1, 16.55, 18.0)) and c.level == 0.0
    assert [round(c.row(k), 4) for k in range(1, 5)] == [2.55, 2.55, 1.45, 1.45]
    assert Costs.measured((0.0, 9.0, 11.0, 12.0), 0.5).verify == (0.0, 9.0, 11.0, 12.0)


def test_cost_rule_verifies_while_the_chance_pays_for_the_row():
    rule = DepthRule(_costs(), 4)
    rule.tokens, rule.ms = 2.0, 10.0                           # 0.2 tokens a ms
    assert rule.keep(1, 0.01)                                  # a drafted window always verifies its first draft
    assert rule.keep(2, 0.41) and not rule.keep(2, 0.39)       # 0.2 x 2 ms
    assert rule.keep(4, 0.21) and not rule.keep(4, 0.19)       # the fifth row adds 1 ms
    assert rule.more(1, 0.5) and not rule.more(1, 0.49)        # a certain next draft: 0.2 x (2 + 0.5)
    assert not rule.more(4, 1.0)                               # the chain is at its most


def test_power_calibrates_an_overconfident_head():
    rule = DepthRule(_costs(), 4, power=2.0)
    rule.tokens, rule.ms = 2.0, 10.0
    assert rule.keep(2, 0.64) and not rule.keep(2, 0.63)       # 0.64 ** 2 >= 0.4 > 0.63 ** 2


def test_floor_rule_keeps_the_first_draft_then_holds_the_floor():
    rule = DepthRule(_costs(), 4, floor=0.5)
    assert rule.keep(1, 0.01) and not rule.more(1, 0.49) and rule.more(1, 0.5)
    assert rule.keep(2, 0.5) and not rule.keep(2, 0.49)
    rule.done(5, 5, 4)                                         # the rate never moves a floor
    assert rule.keep(3, 0.5)


def test_rate_follows_modeled_rounds_not_clocks():
    a, b = DepthRule(_costs(), 4), DepthRule(_costs(), 4)
    for tokens, rows, levels in ((3, 3, 2), (1, 2, 1), (5, 5, 4), (1, 1, 1)):
        a.done(tokens, rows, levels)
        b.done(tokens, rows, levels)
    assert a.rate == b.rate                                    # two ranks given the same rounds agree exactly
    start = DepthRule(_costs(), 4)
    assert start.rate == pytest.approx(2.0 / 12.5)
    start.done(5, 5, 4)
    assert start.rate == pytest.approx((2.0 + (5 - 2.0) / 8) / (12.5 + (17.0 + 2.0 - 12.5) / 8))


@pytest.mark.parametrize("most, floor", [(0, None), (6, None), (2, 1.5), (2, -0.1)])
def test_bad_rules_are_refused(most, floor):
    with pytest.raises(ValueError):
        DepthRule(_costs(), most, floor=floor)
