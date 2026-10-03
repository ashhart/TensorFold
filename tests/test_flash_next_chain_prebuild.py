"""The chain's first step built before the read leaves settle's drafts and the head cache as settle alone does."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.qwen4_exp.runtime import FlashNext  # noqa: E402


class _Cache:
    """MTPCache's protocol for speculate and settle: rows go to the buffer, or beside it while chaining."""

    drafted = 0
    chaining = False
    side = None

    def __init__(self) -> None:
        self.offset = 40

    def trim(self, n: int, ratio: int = 4) -> None:
        held = 0 if self.side is None else len(self.side)
        self.side = None if n >= held else self.side[:held - n]
        self.offset -= n


def _flash() -> FlashNext:
    flash = FlashNext.__new__(FlashNext)
    flash.args = SimpleNamespace(indexer_compress_ratio=4)
    flash.queued_chains = True
    flash._specs, flash._prepared = {}, {}
    flash.mtp_fused = SimpleNamespace(eval_every=1)

    def step(tokens, streams, cache):
        """A step whose output depends on its token, streams and the cache it sees; rows land as the real one's."""
        tokens = tokens if isinstance(tokens, mx.array) else mx.array(tokens, dtype=mx.uint32)
        rows = int(streams.shape[0])
        mixed = streams * 3.0 + tokens.astype(mx.float32)[:, None] + float(cache.offset)
        if cache.chaining:
            cache.side = (cache.side or ()) + tuple(range(cache.offset, cache.offset + rows))
        cache.offset += rows
        return mixed[None], streams + 1.0

    flash._mtp_step = step
    flash._draft_draw = lambda mixed, sampling, positions: (
        (mx.sum(mixed.reshape(-1, mixed.shape[-1]), axis=-1) + positions[0]).astype(mx.uint32) % 997)
    return flash


def _round(prepared: bool, keep: int, count: int) -> tuple:
    flash, cache = _flash(), [_Cache()]
    rows, position = 4, 100
    flash._streams = mx.arange(rows * 3, dtype=mx.float32).reshape(rows, 3)
    tokens = mx.array([5, 6, 7, 8], dtype=mx.uint32)
    firsts = FlashNext.speculate(flash, cache, tokens, position, None)
    if prepared:
        flash.prepare_settle(cache, firsts, position, None)
    first = int(firsts[keep - 1].item())
    drafts = flash.settle(cache, keep, first, position + keep + 1, None, count)
    values = [int(t) for t in (drafts.tolist() if isinstance(drafts, mx.array) else drafts)]
    head = cache[0]
    return values, head.offset, head.drafted, head.side, head.chaining


@pytest.mark.parametrize("keep", [4, 3, 2, 1])
@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_settle_after_prepare_equals_settle_alone(keep, count):
    assert _round(True, keep, count) == _round(False, keep, count)


def test_the_prepared_step_is_the_one_queued():
    flash, cache = _flash(), [_Cache()]
    flash._streams = mx.arange(12, dtype=mx.float32).reshape(4, 3)
    firsts = FlashNext.speculate(flash, cache, mx.array([5, 6, 7, 8], dtype=mx.uint32), 100, None)
    flash.prepare_settle(cache, firsts, 100, None)
    assert sorted(flash._prepared[id(cache[0])]) == [3, 4] and cache[0].offset == 44 and cache[0].side is None
    assert flash.mtp_fused.eval_every == 1      # the steps were built with nothing queued, then the cadence restored
    calls = []
    real = flash._mtp_step
    flash._mtp_step = lambda *a: calls.append(1) or real(*a)
    flash.settle(cache, 4, int(firsts[3].item()), 105, None, 3)
    assert len(calls) == 1                      # the second step only: the first was built before the read


def test_a_one_draft_round_builds_nothing():
    flash, cache = _flash(), [_Cache()]
    flash._streams = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    firsts = FlashNext.speculate(flash, cache, mx.array([5, 6], dtype=mx.uint32), 100, None)
    flash.prepare_settle(cache, firsts, 100, None)
    assert id(cache[0]) not in flash._prepared and cache[0].offset == 42
