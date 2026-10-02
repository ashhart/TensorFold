"""Flash Next's n-gram row ids hashed on the GPU equal NGramEmbedding.ids, EOS resets and 64-bit wraps included."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

from tensorfold.families.qwen4_exp.model import NGramEmbedding, _nth_prime_after, layer_multipliers  # noqa: E402
from tensorfold.kernels.qwen.flash_next.v1 import ngram  # noqa: E402

VOCAB, EOS = 248320, 248044


def _emb(multipliers=None, n=3, per=8):
    """NGramEmbedding's hashing attributes for Flash Next's shape (no tables)."""

    heads = (n - 1) * per
    sizes = [_nth_prime_after(20_000_000 - 1, heads + h + 1) for h in range(heads)]
    mults = layer_multipliers(VOCAB, n, 1, 1234) if multipliers is None else np.asarray(multipliers, np.int64)
    return SimpleNamespace(n=n, context=n - 1, per_ngram=per, heads=heads, eos=EOS, multipliers=mults,
                           head_sizes=np.array(sizes, np.int64),
                           head_offsets=np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64))


def _check(emb, history, window):
    want = NGramEmbedding.ids(emb, np.array([history], np.int64), np.array([window], np.int64))[0]
    got = ngram.NgramHash(emb)(mx.array([history], dtype=mx.uint32), mx.array([window], dtype=mx.uint32))
    assert np.array_equal(np.asarray(got), want.astype(np.uint32)), (history, window)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 9, 16])
def test_gpu_ids_equal_the_host_ids(rows):
    rng = np.random.default_rng(rows)
    emb = _emb()
    for _ in range(40):
        seq = rng.integers(0, VOCAB, size=2 + rows)
        seq[rng.random(2 + rows) < 0.2] = EOS                   # end ids anywhere: history, window, both
        _check(emb, [int(t) for t in seq[:2]], [int(t) for t in seq[2:]])
    _check(emb, [EOS, EOS], [int(t) for t in rng.integers(0, VOCAB, size=rows)])      # a fresh stream


def test_products_wrap_and_remainders_floor_as_numpy():
    # multipliers past 2^63 / vocab: products wrap in 64 bits, XORs can go negative, numpy's % floors
    emb = _emb(multipliers=[(1 << 62) + 1, (1 << 63) - 1, -(1 << 61) - 7])
    rng = np.random.default_rng(3)
    for _ in range(40):
        seq = [int(t) for t in rng.integers(VOCAB // 2, VOCAB, size=6)]
        _check(emb, seq[:2], seq[2:])


def test_histories_join_on_either_side():
    gpu = ngram.join_history(mx.array([[5, 6]], dtype=mx.uint32), mx.array([[7, 8, 9]], dtype=mx.uint32), 2)
    host = ngram.join_history(np.array([[5, 6]], np.int64), np.array([[7, 8, 9]], np.int64), 2)
    mixed = ngram.join_history(np.array([[5, 6]], np.int64), mx.array([[7]], dtype=mx.uint32), 2)
    assert np.asarray(gpu).tolist() == host.tolist() == [[8, 9]] and np.asarray(mixed).tolist() == [[6, 7]]
    assert ngram.host_ids(gpu).dtype == np.int64 and ngram.host_ids(None) is None
