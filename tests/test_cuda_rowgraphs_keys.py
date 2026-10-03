"""Row graphs without a GPU: buckets, padding, capture on first use, rank agreement, moved buffers."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.cuda.rowgraphs import BUCKETS, RowGraphs, bucket, pad


def test_buckets_cover_every_row_count_up_to_the_largest():
    assert [bucket(n) for n in (1, 8, 9, 13, 64)] == [1, 8, 12, 16, 64] and bucket(65) is None
    assert BUCKETS[-1] == 64


def test_padding_repeats_the_last_real_row_and_writes_nothing():
    t = np.array([[5, 6, 7], [10, 11, 12], [0, 1, 1]])
    out = pad(t, 5, write=2)
    assert out.tolist() == [[5, 6, 7, 7, 7], [10, 11, 12, 12, 12], [0, 1, 1, -1, -1]]
    with pytest.raises(ValueError):
        pad(t, 2)


class Graph:
    def __init__(self, fn, log):
        self.fn, self.log = fn, log

    def replay(self):
        self.log.append("replay")
        self.fn()


def graphs(agree=lambda v: v, floor=lambda: True, **kw):
    log = []
    g = RowGraphs(
        lambda r: log.append(("run", r)),
        agree=agree,
        floor=floor,
        capture=lambda fn: (log.append("capture"), Graph(fn, log))[1],
        **kw,
    )
    return g, log


def test_first_use_runs_eagerly_and_captures_then_replays():
    g, log = graphs()
    assert g.run(9) == "capture" and g.run(11, 0) == "replay" and g.run(9, ctx=1) == "capture"
    assert log == [("run", 12), "capture", "replay", ("run", 12), ("run", 12), "capture"]
    assert g.rows(9) == 12 and g.rows(5) == 5 and g.rows(70) == 70
    assert g.run(70) == "eager" and log[-1] == ("run", 70)


def test_a_rank_that_cannot_capture_keeps_every_rank_eager():
    g, log = graphs(agree=lambda v: [0])
    assert g.run(3) == "eager" and g.run(3) == "eager" and "capture" not in log
    low, _ = graphs(floor=lambda: False)
    assert low.run(3) == "eager"


def test_the_budget_and_the_count_cap_captures():
    g, _ = graphs(most=1)
    assert g.run(1) == "capture" and g.run(2) == "eager"
    spent, _ = graphs(budget_s=0.0)
    assert spent.run(1) == "eager"


def test_a_moved_buffer_drops_every_graph():
    class Buf:
        def __init__(self, at):
            self.at = at

        def data_ptr(self):
            return self.at

    g, _ = graphs()
    g.fingerprint(Buf(1), Buf(2))
    g.run(4)
    g.fingerprint(Buf(1), Buf(2))
    assert g.run(4) == "replay"
    g.fingerprint(Buf(1), Buf(3))
    assert g.graphs == {} and g.run(4) == "capture"


def test_warm_captures_context_zero_once():
    staged = []
    g, _ = graphs()
    assert g.warm([1, 2, 10, 11], staged.append) == 3 and staged == [1, 2, 12]
    assert g.warm([1], staged.append) == 0


def test_the_row_table_stages_columns_in_one_copy():
    pytest.importorskip("torch")
    from tensorfold.cuda.rowgraphs import RowTable

    t = RowTable(["ids", "pos", "write"], 8, device="cpu")
    at = t.dev.data_ptr()
    t.stage(pad(np.array([[3, 4], [10, 11], [0, 0]]), 4, write=2))
    assert t.col("ids", 4).tolist() == [3, 4, 4, 4] and t.col("write", 4).tolist() == [0, 0, -1, -1]
    assert t.dev.data_ptr() == at
    with pytest.raises(ValueError):
        t.stage(np.zeros((2, 4), dtype=np.int64))
