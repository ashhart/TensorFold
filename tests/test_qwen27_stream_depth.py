"""The 27B's stream planner: measured round overhead, the drafter's block by tree depth, the startup curve's steps."""

import importlib
from types import SimpleNamespace

import pytest

from tensorfold.cuda.streams import Stream
from tests.test_cuda_27b_ignore_eos import scripted_decoder
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch

CURVE = [(1, 14.0), (16, 16.3), (17, 23.5), (32, 23.9), (33, 33.0), (48, 34.0), (64, 38.8)]


class Drafter:
    """Records the block each pass drafts; a stream's tree is a chain as deep as ``depths`` says."""

    def __init__(self, depths):
        self.depths, self.blocks = depths, []

    def launch_blocks(self, snaps, pendings, max_nodes, block=None):
        self.blocks.append(block)
        return list(range(len(snaps)))

    def finish_tree(self, launched, length, max_nodes, sampling):
        d = self.depths[launched]
        return list(range(1, d + 1)), list(range(-1, d - 1)), [0.01 * (i + 1) for i in range(d)]


def decoder(multi, monkeypatch, depths, costs=None):
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.w, dec.max_rows, dec.costs, dec.overhead = None, 16, costs, (8.0, 1.5)
    dec.rank, dec.world, dec.split, dec.drafts, dec.device = 0, 1, False, True, None
    dec.draft = Drafter(depths)
    dec.streams = {sid: SimpleNamespace(snap=None, sampling=None, st=SimpleNamespace(pos=10), draft=True,
                                        constraint=None) for sid in range(len(depths))}
    monkeypatch.setattr(multi, "multi_tree_forward", lambda w, wins, **kw: (None, None, None, None))
    monkeypatch.setattr(multi, "sample_streams", lambda logits, starts, positions, samplings: [[]] * len(samplings))
    return dec


def plan(dec, modes=None):
    return [(sid, (modes or {}).get(sid, 1), 5, 20) for sid in dec.streams]   # 1: TREE


def test_one_stream_drafts_the_block_its_trees_keep_and_keeps_the_prior(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = decoder(multi, monkeypatch, [3], CURVE)
    dec.block, dec.spent = 6, {1: [1.0] * 8}
    dec._verify(plan(dec), {})
    assert dec.draft.blocks == [6] and dec.block == 5 and dec._overhead(1) == 9.5


def test_streams_draft_a_level_below_the_deepest_kept_node(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = decoder(multi, monkeypatch, [3, 5, 2])
    dec._verify(plan(dec), {})
    assert dec.draft.blocks == [16] and dec.block == 7          # depth 5 kept: 6 levels and the pending row
    dec.draft.depths = [6, 1, 1]                                 # a chain to the block's last level: four more
    dec._verify(plan(dec), {})
    assert dec.draft.blocks == [16, 7] and dec.block == 11
    dec.draft.depths = [1, 1, 1]
    dec._verify(plan(dec), {})
    assert dec.block == 4                                        # never below four rows
    dec._verify(plan(dec, {0: 2, 1: 2, 2: 2}), {})               # no tree window this round: the block stays
    assert dec.block == 4 and dec.draft.blocks == [16, 7, 11]


def test_other_gpus_draft_every_level(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    assert (12, 0) in multi.DEPTH_CHIPS and (12, 1) not in multi.DEPTH_CHIPS
    dec = decoder(multi, monkeypatch, [3, 2])
    dec.depth = False                                            # a GB10 or an unmeasured GPU
    dec._verify(plan(dec), {})
    dec._verify(plan(dec), {})
    assert dec.draft.blocks == [16, 16]


def test_the_block_never_drops_under_the_drafters_training_block(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = decoder(multi, monkeypatch, [3, 2])
    dec.draft.trained = 8                                        # DFlash2's block_size
    dec._verify(plan(dec), {})
    assert dec.block == 8                                        # depth 3 kept: 5 rows would do, 8 drafted
    dec.draft.depths = [7, 1]                                    # a chain to the block's last level: four more
    dec._verify(plan(dec), {})
    assert dec.draft.blocks == [16, 8] and dec.block == 12
    dec.draft.depths = [9, 1]
    dec._verify(plan(dec), {})
    assert dec.block == 11


def test_the_overhead_past_one_stream_is_the_median_of_the_last_rounds(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.costs, dec.overhead = CURVE, (8.0, 1.5)
    for i, ms in enumerate([33.0, 34.0, 60.0]):
        dec.last = (0.0, 8, 32)
        dec._timed(ms / 1e3)
    assert dec._overhead(8) == 20.0                                # three rounds: still the prior
    dec.last = (0.0, 8, 32)
    dec._timed(0.035)
    assert dec._overhead(8) == pytest.approx(35.0 - 23.9)          # the median of 9.1, 10.1, 11.1, 36.1
    for _ in range(multi.TIMED):
        dec.last = (0.0, 8, 32)
        dec._timed(0.030)
    assert dec._overhead(8) == pytest.approx(30.0 - 23.9) and len(dec.spent[8]) == multi.TIMED
    assert dec._overhead(4) == 14.0                                # unseen stream counts keep the prior


def test_rounds_time_only_while_streams_go_on(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = scripted_decoder(multi, list(range(10, 40)))          # no end token: each stream decodes its count
    dec.costs, dec.overhead = CURVE, (8.0, 1.5)
    streams = [Stream([1, 2], 13), Stream([3, 4], 13)]
    for s in streams:
        dec.admit(s)
    dec.finish([s for s in streams if s.done])
    timed = []
    while dec.live():
        dec.finish(dec.round())
        timed.append(dec.last and dec.last[1:])
    # one stream while the second prompt fills, then both (three drafts a window), none after the last round
    assert timed == [(1, 4), (2, 8), (2, 8), None]
    assert {n: len(v) for n, v in dec.spent.items()} == {1: 1, 2: 2}   # each round timed by the next one's start


def test_the_curve_times_both_sides_of_each_row_step(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    assert multi.calibration_rows(1) == [1, 2, 4, 8, 12, 16]
    rows = multi.calibration_rows(8)
    assert {16, 17, 32, 33, 64, 65, 128} <= set(rows) and rows[-1] == 128 and 129 not in rows
