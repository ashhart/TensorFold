"""Drafted rounds on Macs without tensor units: the row-exact matvec, the per-round draft count, and the DFlash2
proposer's running acceptance."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.drafters.dflash_drafter import DFlashProposer  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import row_qmv  # noqa: E402


@pytest.mark.parametrize("group_size", [32, 64])
def test_row_qmv_rows_keep_their_bits_at_any_row_count(group_size):
    linear = nn.Linear(1024, 256, bias=False)
    linear.set_dtype(mx.bfloat16)
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=group_size, bits=4)
    assert row_qmv.fits(quantized)
    x = mx.random.normal((row_qmv.MAX_ROWS, 1024), key=mx.random.key(7)).astype(mx.bfloat16)
    args = (quantized["weight"], quantized["scales"], quantized["biases"], group_size)
    full = row_qmv.qmv(x, *args)
    for rows in range(1, row_qmv.MAX_ROWS + 1):
        assert bool(mx.array_equal(row_qmv.qmv(x[:rows], *args), full[:rows]).item())
    reference = x.astype(mx.float32) @ mx.dequantize(*args[:3], group_size=group_size, bits=4).astype(mx.float32).T
    assert float(mx.abs(full.astype(mx.float32) - reference).max().item()) < 0.05 * float(mx.abs(reference).max().item())


def _stream(rate: float) -> SimpleNamespace:
    proposer = SimpleNamespace(continue_rate=lambda: rate, draft_ms=40.0, proposals=10)
    return SimpleNamespace(proposer=proposer, idle_rounds=0)


def test_paying_drafts_follow_the_acceptance_and_probe_when_idle():
    costs = {1: 27.0, 2: 35.0, 3: 43.0, 4: 54.0, 5: 64.0, 6: 77.0, 7: 89.0, 8: 102.0}   # M3 Ultra, 26 Sep
    engine = SimpleNamespace(window_costs=costs, probe_every=8)
    assert LaneEngine._paying_drafts(engine, _stream(0.9), 7) == 4
    assert LaneEngine._paying_drafts(engine, _stream(0.9), 2) == 2       # never past the budget
    low = _stream(0.2)
    picks = [LaneEngine._paying_drafts(engine, low, 7) for _ in range(16)]
    assert picks == [0] * 7 + [1] + [0] * 7 + [1]                       # one draft every 8 idle rounds


def test_dflash_proposer_tracks_recent_acceptance():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer._last_was_copy = False
    proposer.accepted_tokens = 0
    proposer.hits = proposer.misses = 0.0
    assert proposer.continue_rate() == pytest.approx(0.6)
    for _ in range(20):
        proposer.observe(4, 4)
    assert proposer.continue_rate() > 0.9
    for _ in range(40):
        proposer.observe(4, 0)
    assert proposer.continue_rate() < 0.3
