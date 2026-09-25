"""The drafter's reduced vocabulary head: the same logits as the full head on the kept token ids."""

import types

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.kernels.qwen.dense.v1 import lane_qmm  # noqa: E402
from tensorfold.drafters.dflash_drafter import DFlashDrafter  # noqa: E402


def _drafter_with_head(n: int, k: int):
    mx.random.seed(9)
    holder = nn.Sequential(nn.Linear(k, n, bias=False))
    holder.set_dtype(mx.bfloat16)
    nn.quantize(holder, group_size=64, bits=4)
    mx.eval(holder.parameters())
    head = holder.layers[0]
    config = types.SimpleNamespace(output_multiplier=1.0, final_logit_softcapping=None)
    model = types.SimpleNamespace(lm_head=head, config=config)
    model.compute_logits = lambda hidden: head(hidden) * config.output_multiplier
    drafter = object.__new__(DFlashDrafter)
    drafter.model = model
    drafter.draft_vocab = ((0, 1024), (3968, 4096))
    return drafter, holder


def test_draft_vocab_keeps_the_full_heads_logits():
    try:
        drafter, holder = _drafter_with_head(4096, 512)
        lane_qmm.install(holder, rows=lane_qmm.MAX_ROWS)
        hidden = (mx.random.normal((1, 16, 512)) * 0.5).astype(mx.bfloat16)
        full = drafter.model.compute_logits(hidden)
        sub, ids = drafter.candidate_logits(hidden)
        mx.eval(full, sub, ids)
    except RuntimeError as exc:  # no Metal 4 tensor ops on this machine
        lane_qmm.uninstall()
        pytest.skip(str(exc).splitlines()[0][:80])
    try:
        assert ids is not None and ids.shape == (1024 + 128,)
        assert ids.tolist() == list(range(1024)) + list(range(3968, 4096))
        kept = mx.take(full, ids, axis=-1).astype(mx.float32)
        assert sub.shape == kept.shape
        assert float(mx.max(mx.abs(sub.astype(mx.float32) - kept)).item()) <= 0.02 * float(mx.max(mx.abs(kept)).item())
    finally:
        lane_qmm.uninstall()


def test_draft_vocab_off_without_the_lane_head():
    drafter, _ = _drafter_with_head(4096, 512)                # not installed: the head is not tiled
    hidden = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)
    logits, ids = drafter.candidate_logits(hidden)
    assert ids is None and logits.shape == (1, 4, 4096)
