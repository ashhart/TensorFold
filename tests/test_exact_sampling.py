"""Exact sampling: a fixed function of (logits, position, seed); an exact top-k/top-p draw."""

import numpy as np
import pytest

from tensorfold.engine.exact_sampling import Sampling, choose, seed_for, uniform


def test_same_inputs_same_token_and_position_matters():
    rng = np.random.default_rng(0)
    values = rng.normal(size=28).astype(np.float32)
    ids = rng.permutation(1000)[:28].astype(np.int64)
    s = Sampling(seed=123, temperature=1.0, top_k=20, top_p=0.95)
    assert choose(values, ids, 7, s) == choose(values.copy(), ids.copy(), 7, s)
    picks = {choose(values, ids, p, s) for p in range(200)}
    assert len(picks) > 3          # the draw moves with the position


def test_candidate_order_does_not_matter():
    rng = np.random.default_rng(1)
    values = rng.normal(size=28).astype(np.float32)
    ids = np.arange(28, dtype=np.int64) + 500
    s = Sampling(seed=9)
    perm = rng.permutation(28)
    assert choose(values, ids, 3, s) == choose(values[perm], ids[perm], 3, s)


def test_ties_break_by_token_id():
    values = np.array([5.0, 5.0, 5.0, 1.0], dtype=np.float32)
    ids = np.array([40, 10, 30, 20], dtype=np.int64)
    s = Sampling(seed=4, temperature=1.0, top_k=2, top_p=1.0)
    for p in range(50):
        assert choose(values, ids, p, s) in (10, 30)   # ids 10 and 30 hold the top-2 slots


def test_draw_matches_the_distribution():
    values = np.log(np.array([0.5, 0.3, 0.2], dtype=np.float64)).astype(np.float32)
    ids = np.array([1, 2, 3], dtype=np.int64)
    s = Sampling(seed=77, temperature=1.0, top_k=3, top_p=1.0)
    counts = np.bincount([choose(values, ids, p, s) for p in range(20000)], minlength=4)[1:]
    assert np.allclose(counts / counts.sum(), [0.5, 0.3, 0.2], atol=0.015)


def test_top_p_cuts_the_tail():
    values = np.log(np.array([0.6, 0.3, 0.1], dtype=np.float64)).astype(np.float32)
    ids = np.array([1, 2, 3], dtype=np.int64)
    s = Sampling(seed=5, temperature=1.0, top_k=3, top_p=0.85)
    assert {choose(values, ids, p, s) for p in range(3000)} == {1, 2}


def test_uniform_in_open_interval_and_seed_for_is_stable():
    u = uniform(1, 2, np.arange(10000, dtype=np.int64))
    assert (u > 0).all() and (u < 1).all()
    assert seed_for([1, 2, 3]) == seed_for([1, 2, 3]) != seed_for([1, 2, 4])


def test_sample_rows_matches_choose_on_the_gpu():
    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine.exact_sampling import sample_rows

    logits = (mx.random.normal((3, 5000)) * 3).astype(mx.bfloat16)
    s = Sampling(seed=11)
    rows = sample_rows(logits, [10, 11, 12], s)
    alone = [sample_rows(logits[i:i + 1], [10 + i], s)[0] for i in range(3)]
    assert rows == alone


def test_choose_rows_matches_choose_row_by_row_alone_or_batched():
    from tensorfold.engine.exact_sampling import choose_rows

    rng = np.random.default_rng(7)

    def bf16(a):
        b = a.astype(np.float32).view(np.uint32)
        return ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).view(np.float32)

    for trial in range(300):
        s = Sampling(seed=int(rng.integers(0, 2**63 - 1)), top_p=[0.95, 1.0, 0.5][trial % 3])
        rows = int(rng.choice([1, 3, 16]))
        values = bf16(rng.normal(0, rng.choice([0.5, 3.0, 12.0]), (rows, 28)) + 20)
        if trial % 5 == 0:
            values[:, 2] = values[:, 9]                     # value ties resolve by token id
        ids = np.stack([rng.choice(248320, 28, replace=False) for _ in range(rows)]).astype(np.int64)
        positions = rng.integers(0, 100000, rows)
        want = [choose(values[r], ids[r], int(positions[r]), s) for r in range(rows)]
        assert choose_rows(values, ids, positions, s) == want
        assert [choose_rows(values[r:r + 1], ids[r:r + 1], positions[r:r + 1], s)[0] for r in range(rows)] == want


def test_nucleus_without_top_k_matches_the_whole_vocabulary_draw():
    """top_k 0: the fast nucleus (GPU candidates + normalizer) draws what sorting every logit draws."""

    import mlx.core as mx

    from tensorfold.engine import exact_sampling as es

    rng = np.random.default_rng(5)
    drawn = 0
    for trial in range(60):
        row = (mx.array(rng.normal(size=(1, 50_000)).astype(np.float32)) * 8.0).astype(mx.bfloat16)
        s = es.Sampling(seed=int(rng.integers(1 << 62)), temperature=float(rng.choice([0.7, 1.0])), top_k=0,
                        top_p=0.95)
        fast = es._nucleus_rows(row, [trial], s)
        full = es.choose_rows(np.array(row.astype(mx.float32)), np.arange(50_000, dtype=np.int64)[None], [trial], s)
        if fast is not None:
            drawn += 1
            assert fast == full
    assert drawn > 40
    flat = mx.zeros((1, 50_000), dtype=mx.bfloat16)          # every token tied: the nucleus needs them all
    assert es._nucleus_rows(flat, [0], es.Sampling(seed=1, top_k=0, top_p=0.95)) is None
