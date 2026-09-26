"""GLM-5.3-Flash family on a tiny random checkpoint (CPU unless noted): loading, the two forward paths, exact
multi-row decoding, rollback, MTP drafting through the serial engine."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from glm5_fakes import TEXT, write_checkpoint  # noqa: E402
from tensorfold.families.glm5_next import model as glm  # noqa: E402
from tensorfold.families.glm5_next import mtp as glm_mtp  # noqa: E402
from tensorfold.families.glm5_next.runtime import GLMFlash  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return write_checkpoint(tmp_path_factory.mktemp("glm5"))
    finally:
        mx.set_default_device(previous)


def backbone(checkpoint):
    return glm.load_backbone(checkpoint)


def tokens(n: int, seed: int = 1) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(6, TEXT["vocab_size"], size=n)]


def test_family_is_detected_and_checked(checkpoint):
    from tensorfold import families
    from tensorfold.families import glm5_next

    assert families.detect(checkpoint).module == "tensorfold.families.glm5_next"
    glm5_next.check(checkpoint)
    assert glm_mtp.has_mtp(checkpoint)


def test_loads_the_checkpoint_layout(checkpoint):
    model = backbone(checkpoint)
    kinds = ["kda" if layer.is_linear else "mla" for layer in model.layers]
    assert kinds == ["kda", "kda", "kda", "mla", "kda", "mla"]
    assert isinstance(model.layers[0].mlp, glm.DenseMLP) and isinstance(model.layers[1].mlp, glm.MoE)
    assert model.layers[1].mlp.gate.weight.shape[0] == TEXT["n_routed_experts"]
    mla = model.layers[3].attn
    # kv_b split per head in its stored layout: keys [H, nope, rank], values [H, v, rank]
    assert mla.wk.weight.shape[:2] == (2, 64) and mla.wv.weight.shape[:2] == (2, 64)


@pytest.mark.parametrize("length", [9, 40])   # 40 > index_topk (16): the sparse selection with pooled blocks
def test_prefill_path_agrees_with_decode_path(checkpoint, length):
    """The batched prefill path and the one-row decode path compute the same function (to rounding)."""

    model = backbone(checkpoint)
    ids = tokens(length)
    whole = model.make_cache()
    a = model.head(model.hidden(mx.array([ids]), whole))[0, -1]
    step = model.make_cache()
    for t in ids:
        b = model.head(model.hidden(mx.array([[t]]), step))[0, -1]
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert int(a.argmax()) == int(b.argmax())
    assert np.max(np.abs(a - b)) < 0.05 * np.max(np.abs(b)) + 0.05
    for c1, c2 in zip(whole, step):
        assert c1.offset == c2.offset == length


def test_sparse_attention_reads_a_subset_past_the_budget(checkpoint):
    model = backbone(checkpoint)
    mla = model.layers[3].attn
    cache = model.make_cache()
    model.hidden(mx.array([tokens(40)]), cache)
    c = cache[3]
    blocks = 40 // 4
    scores = mla.index_scores(mx.random.normal((1, 2, 64)).astype(mx.bfloat16),
                              mx.ones((1, 2), dtype=mx.bfloat16), c.pool[:blocks])
    ids = np.array(mla.selected(scores, 39))
    assert len(ids) == 4 * (TEXT["index_topk"] // 4)             # 4 blocks of 4, no tail at position 39
    assert len(set(ids.tolist())) == len(ids) and ids.max() < 40
    ids = np.array(mla.selected(scores[:, :9], 37))             # position 37: tail keys 36, 37
    assert ids[-2:].tolist() == [36, 37]


def test_decode_rows_give_one_row_bits(checkpoint):
    runtime = GLMFlash(backbone(checkpoint), check=True)
    assert runtime.multi_row_exact, runtime.check_report


def test_keep_rows_rolls_every_cache_back(checkpoint):
    from tensorfold.engine.lane_engine import LaneEngine

    model = backbone(checkpoint)
    base = model.make_cache()
    model.hidden(mx.array([tokens(30)]), base)
    window = [7, 9, 11, 13, 17]
    a, b = LaneEngine.copy_single_cache(base), LaneEngine.copy_single_cache(base)
    model.hidden(mx.array([window]), a)
    model.keep_rows(a, len(window), 2)
    for t in window[:2]:
        model.hidden(mx.array([[t]]), b)
    for c1, c2 in zip(a, b):
        assert c1.offset == c2.offset
        for x, y in zip(c1.state, c2.state):
            n = c1.offset if isinstance(c1, glm.MLACache) else None
            if n is not None and x.shape[0] >= n:
                x, y = x[:n], y[:n]
            assert bool(mx.array_equal(x, y).item())
    # and both continue alike
    la = model.head(model.hidden(mx.array([[21]]), a))
    lb = model.head(model.hidden(mx.array([[21]]), b))
    assert bool(mx.array_equal(la, lb).item())


def _run_engine(runtime, prompt, n):
    from tensorfold.engine.family_engine import SerialEngine
    from tensorfold.engine.lane_engine import LaneStream

    engine = SerialEngine(runtime)
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=n)
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    return engine, stream


def test_mtp_drafts_change_speed_only(checkpoint):
    model = backbone(checkpoint)
    head = glm_mtp.load(model)
    drafted = GLMFlash(model, head, drafts=3)
    serial = GLMFlash(model, None, drafts=0)
    assert drafted.mtp is not None and serial.mtp is None
    prompt = tokens(21, seed=4)
    engine_a, a = _run_engine(drafted, prompt, 24)
    engine_b, b = _run_engine(serial, prompt, 24)
    assert engine_a.sync_drafts and not engine_b.sync_drafts
    assert engine_a.drafted > 0
    assert a.emitted == b.emitted


def test_serial_engine_resumes_from_a_stored_cache(checkpoint, tmp_path):
    """A prefix snapshot written to disk and read back continues exactly like the in-memory cache."""

    from tensorfold.engine.family_engine import SerialEngine
    from tensorfold.engine.lane_engine import LaneStream
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    model = backbone(checkpoint)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=2)
    prefix = tokens(26, seed=5)
    engine = SerialEngine(runtime)
    cache = engine.prefill_prefix(prefix)
    path = save_snapshot(tmp_path, "glm-test", prefix, cache)
    got_tokens, stored = load_snapshot(path, "glm-test")
    assert got_tokens == prefix
    follow = [*prefix, 7, 8, 9]

    def run(c):
        e = SerialEngine(runtime)
        s = LaneStream(stream_id="x", prompt_ids=follow, max_new_tokens=8)
        e.add_stream(s, cache=SerialEngine.copy_single_cache(c), cached_tokens=len(prefix))
        while e.active_count:
            e.step()
        return s.emitted

    assert run(stored) == run(cache)


def test_qmv_rows_gives_mlx_one_row_bits():
    """On Metal: every row of a qmv_rows call equals MLX's one-row quantized matmul."""

    from tensorfold.kernels.glm.flash.v1 import kernels

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    mx.set_default_device(mx.gpu)
    assert kernels.metal()
    mx.random.seed(3)
    w = glm.Q(*mx.quantize((0.05 * mx.random.normal((256, 1024))).astype(mx.bfloat16), group_size=64, bits=4))
    for rows in (2, 3, 4, 8):
        x = mx.random.normal((rows, 1024)).astype(mx.bfloat16)
        many = kernels.qmv_rows(x, w)
        one = mx.concatenate([w(x[r:r + 1]) for r in range(rows)])
        assert bool(mx.array_equal(many, one).item()), rows


def test_on_metal_rows_are_exact_and_drafts_change_speed_only(checkpoint):
    """The same checks on the GPU, through the Metal kernels (hyper-connection split, gated delta)."""

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    mx.set_default_device(mx.gpu)
    model = backbone(checkpoint)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.multi_row_exact, runtime.check_report
    prompt = tokens(30, seed=6)
    engine_a, a = _run_engine(runtime, prompt, 20)
    _, b = _run_engine(GLMFlash(model, None, drafts=0), prompt, 20)
    assert engine_a.drafted > 0 and a.emitted == b.emitted


@pytest.mark.skipif(not __import__("os").environ.get("TF_GLM5_MODEL"), reason="set TF_GLM5_MODEL to the checkpoint")
def test_real_weights_first_layers_rows_are_exact():
    """The real checkpoint's first six layers (three KDA with dense MLPs, then sparse MLA and KDA with MoE) and its
    head: a 2/3/4/8-row window gives each row its one-row bits on this GPU, and MTP drafts change speed only."""

    import os

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    mx.set_default_device(mx.gpu)
    path = os.environ["TF_GLM5_MODEL"]
    model = glm.load_backbone(path, layers=6)
    assert ["kda" if layer.is_linear else "mla" for layer in model.layers] == ["kda"] * 3 + ["mla", "kda", "kda"]
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.multi_row_exact, runtime.check_report
    prompt = [int(t) for t in np.random.default_rng(7).integers(1000, 100_000, size=40)]
    engine_a, a = _run_engine(runtime, prompt, 16)
    _, b = _run_engine(GLMFlash(model, None, drafts=0), prompt, 16)
    assert engine_a.drafted > 0 and a.emitted == b.emitted
