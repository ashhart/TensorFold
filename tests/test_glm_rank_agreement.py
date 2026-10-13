"""GLM's startup agreement across every rank: settings must equal rank 0's, the kept-prompt budget is the least."""

from __future__ import annotations

import pytest

from tensorfold.families.glm5_next.cuda.engine import agree

SETTINGS = [1, 2560, 0, 0, 1, 2048, 1, 1]


def test_every_rank_agrees_and_keeps_the_smallest_budget():
    assert agree([SETTINGS + [mib] for mib in (100, 100, 10, 20)]) == 10
    assert agree([SETTINGS + [100], SETTINGS + [30]]) == 30


@pytest.mark.parametrize("odd", [1, 2, 3])
def test_a_mismatch_on_any_rank_refuses_and_names_it(odd):
    rows = [list(SETTINGS) + [100] for _ in range(4)]
    rows[odd][1] = 4096                                 # one rank was given another context
    with pytest.raises(RuntimeError, match=f"different settings.*rank {odd} "):
        agree(rows)
