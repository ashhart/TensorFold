"""Nemotron's MTP speculation belongs to its stream: streams prefilled back to back each settle their own."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")


class _Cache:
    def __init__(self) -> None:
        self.trimmed = 0

    def trim(self, rows: int) -> None:
        self.trimmed += rows


def _family():
    from tensorfold.families.nemotron_h.model import NemotronH

    nem = NemotronH.__new__(NemotronH)                      # the head's plumbing only, no checkpoint
    nem._last_hidden = mx.zeros((1, 4, 8))
    nem._draft_ids = None
    nem.model = SimpleNamespace(backbone=SimpleNamespace(embeddings=lambda t: mx.zeros((1, t.shape[-1], 8))))
    nem._head_step = lambda hidden, emb, mcache, tail: mx.zeros((1, hidden.shape[1], 8))
    nem._draft_logits = lambda out: mx.zeros((1, out.shape[1], 16))
    return nem


def test_two_streams_speculate_in_turn_and_settle_their_own_rows():
    nem = _family()
    a, b = [_Cache()], [_Cache()]
    nem.speculate(a, mx.array([1, 2, 3]), 10, None)        # a new stream's three rows
    nem.speculate(b, mx.array([4]), 20, None)              # the next stream's one, before a settles
    nem.settle(a, 1, 5, 10, None, 0)
    nem.settle(b, 1, 6, 20, None, 0)
    assert (a[-1].trimmed, b[-1].trimmed) == (2, 0)


def test_unspeculate_undoes_only_its_stream():
    nem = _family()
    a, b = [_Cache()], [_Cache()]
    nem.speculate(a, mx.array([1, 2]), 10, None)
    nem.speculate(b, mx.array([3, 4, 5]), 20, None)
    nem.unspeculate(a)
    nem.settle(b, 3, 6, 20, None, 0)
    assert (a[-1].trimmed, b[-1].trimmed) == (2, 0)
