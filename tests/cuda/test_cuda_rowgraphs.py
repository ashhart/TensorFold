"""Row graphs on the GPU: a replay gives every real row the eager run's bits, and padding rows write nothing."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from tensorfold.cuda.rowgraphs import RowGraphs, RowTable, pad

VOCAB, WIDTH, LANES, POSITIONS = 50, 64, 4, 96
COLS = ["ids", "pos", "lane", "write"]


class Toy:
    """Row-independent arithmetic: a row reads its lane's state and writes its own position (padding: a spare lane)."""

    def __init__(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        self.emb = torch.randn((VOCAB, WIDTH), generator=g, device="cuda")
        self.kv = torch.zeros((LANES + 1, POSITIONS, WIDTH), device="cuda")
        self.out = torch.zeros((64, WIDTH), device="cuda")
        self.table = RowTable(COLS, 64)

    def run(self, rows: int) -> None:
        ids, pos, lane, write = (self.table.col(c, rows) for c in COLS)
        x = self.emb[ids] * (pos[:, None] + 1).float() + self.kv[lane].sum(dim=1)
        y = torch.tanh(x * 0.37)
        self.out[:rows] = y
        self.kv[torch.where(write >= 0, write, LANES), pos] = y

    def stage(self, windows, rows):
        cols = []
        for lane, start, ids in windows:
            for i, t in enumerate(ids):
                cols.append([t, start + i, lane, lane])
        t = np.array(cols, dtype=np.int64).T
        self.table.stage(pad(t, rows, write=3))
        return len(cols)


ROUNDS = [
    [(0, 0, [1, 2, 3])],
    [(0, 3, [4]), (1, 0, [5, 6, 7, 8])],
    [(2, 0, [9, 9]), (0, 4, [10, 11]), (1, 4, [12])],
    [(3, 0, list(range(13)))],
    [(0, 6, [3]), (1, 5, [2]), (2, 2, [1]), (3, 13, [4])],
]


def play(graphed: bool):
    toy = Toy()
    g = RowGraphs(toy.run) if graphed else None
    outs = []
    for _ in range(2):  # the second pass replays every key the first captured
        toy.kv.zero_()
        for windows in ROUNDS:
            n = sum(len(w[2]) for w in windows)
            rows = g.rows(n) if g is not None else n
            toy.stage(windows, rows)
            if g is not None:
                g.fingerprint(toy.emb, toy.kv, toy.out, toy.table.dev)
                g.run(n)
            else:
                toy.run(rows)
            outs.append(toy.out[:n].clone())
    torch.cuda.synchronize()
    return outs, toy.kv[:LANES].clone(), g


def test_replay_equals_eager_bit_for_bit():
    eager, kv_eager, _ = play(False)
    graphed, kv_graphed, g = play(True)
    assert g.replays >= len(ROUNDS) and g.graphs
    for a, b in zip(eager, graphed):
        assert torch.equal(a, b)
    assert torch.equal(kv_eager, kv_graphed)


def test_a_lane_alone_equals_the_lane_beside_others():
    toy = Toy()
    toy.stage([(1, 0, [5, 6, 7, 8])], 4)
    toy.run(4)
    alone = toy.out[:4].clone()
    toy.kv.zero_()
    toy.stage([(0, 0, [1, 2]), (1, 0, [5, 6, 7, 8])], 8)
    toy.run(8)
    assert torch.equal(toy.out[2:6], alone)
