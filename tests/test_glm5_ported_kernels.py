"""The kernels ported from mlx-vlm into GLM-5.3-Flash's decode path:

- the fused KDA step (mlx-vlm #2105): one launch per KDA layer, R rows in order, f_b / g_b folded in;
- indexed sparse attention (mlx-vlm #2245): chosen latent keys read from the cache by index;
- the sparse indexer's incremental pool (mlx-vlm #2107's decode path and its stale-pool fix): TensorFold's cache
  already pools once per completed block; a trim followed by new rows must never score a stale block.

On Metal: each kernel against its MLX-ops form (close), and against itself one row at a time (bit-identical). On
Linux the MLX-ops forms are checked for the same row rule through the whole tiny model."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from glm5_fakes import TEXT, write_checkpoint  # noqa: E402
from tensorfold.families.glm5_next import model as glm  # noqa: E402
from tensorfold.families.glm5_next.runtime import GLMFlash  # noqa: E402
from tensorfold.kernels.glm.flash.v1 import kda as KDA_K  # noqa: E402
from tensorfold.kernels.glm.flash.v1 import sparse_attention as SA  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return write_checkpoint(tmp_path_factory.mktemp("glm5p"))
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def gpu():
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _same(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def _tokens(n: int, seed: int = 3) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(6, TEXT["vocab_size"], size=n)]


def _kda_inputs(kda, rows: int, seed: int):
    mx.random.seed(seed)
    proj = (0.5 * mx.random.normal((rows, kda.in_proj.outs))).astype(mx.bfloat16)
    conv = (0.5 * mx.random.normal((kda.taps - 1, 3 * kda.width))).astype(mx.bfloat16)
    state = 0.1 * mx.random.normal((1, kda.heads, kda.dim, kda.dim))
    return proj, conv, state


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 8, 16])
def test_fused_kda_rows_are_one_row_steps(gpu, checkpoint, rows):
    """R rows in one launch == R one-row launches chained (y, state, conv window), and close to the MLX ops."""

    model = glm.load_backbone(checkpoint)
    kda = model.layers[0].attn
    assert KDA_K.fits(kda)
    proj, conv, state = _kda_inputs(kda, rows, seed=rows)
    y, st, cs = KDA_K.kda_rows(kda, proj, conv, state)
    ys, s1, c1 = [], state, conv
    for r in range(rows):
        yr, s1, c1 = KDA_K.kda_rows(kda, mx.contiguous(proj[r:r + 1]), c1, s1)
        ys.append(yr)
    assert _same(y, mx.concatenate(ys)) and _same(st, s1) and _same(cs, c1)
    yo, so, co = KDA_K.kda_rows_ops(kda, proj, conv, state)
    assert _same(cs, co)
    assert float(mx.abs(y.astype(mx.float32) - yo.astype(mx.float32)).max()) <= 0.02 * float(mx.abs(yo).max()) + 1e-3
    assert float(mx.abs(st - so).max()) <= 1e-3 * float(mx.abs(so).max()) + 1e-4


def test_fused_kda_keep_replays_a_prefix_exactly(gpu, checkpoint):
    from tensorfold.engine.lane_engine import LaneEngine

    model = glm.load_backbone(checkpoint)
    base = model.make_cache()
    model.hidden(mx.array([_tokens(30)]), base)
    window = [7, 9, 11, 13, 17, 19]
    for keep in (0, 1, 3, 6):
        a, b = LaneEngine.copy_single_cache(base), LaneEngine.copy_single_cache(base)
        model.hidden(mx.array([window]), a)
        model.keep_rows(a, len(window), keep)
        for t in window[:keep]:
            model.hidden(mx.array([[t]]), b)
        for c1, c2 in zip(a, b):
            if isinstance(c1, glm.KDACache):
                assert _same(c1.ssm, c2.ssm) and _same(c1.conv, c2.conv) and c1.offset == c2.offset
        assert _same(model.head(model.hidden(mx.array([[21]]), a)), model.head(model.hidden(mx.array([[21]]), b)))


@pytest.mark.parametrize("fused", [True, False])
def test_windows_stay_exact_with_and_without_the_fused_kda(checkpoint, monkeypatch, fused):
    monkeypatch.setattr(glm, "FUSED_KDA", fused)
    runtime = GLMFlash(glm.load_backbone(checkpoint), check=True)
    assert runtime.multi_row_exact, runtime.check_report


def test_indexed_attention_rows_are_independent_and_close(gpu):
    mx.random.seed(7)
    rows, heads, dim, cap, n = 5, 4, 128, 512, 300
    keys = mx.random.normal((cap, dim)).astype(mx.bfloat16)
    q = mx.random.normal((rows, heads, dim)).astype(mx.bfloat16)
    width = 67
    idx = np.full((rows, width), -1, dtype=np.int32)
    rng = np.random.default_rng(1)
    for r in range(rows):
        k = rng.choice(n, size=60 + r, replace=False)
        idx[r, :len(k)] = k
    idx = mx.array(idx)
    out = SA.indexed_attention(q, keys, idx, n, 0.0625)
    for r in range(rows):
        assert _same(out[r:r + 1], SA.indexed_attention(q[r:r + 1], keys, idx[r:r + 1], n, 0.0625))
    ref = SA.indexed_attention_ops(q, keys, idx, n, 0.0625)
    assert float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()) < 2e-2


@pytest.mark.parametrize("sparse", [True, False])
def test_long_context_windows_are_exact(checkpoint, monkeypatch, sparse):
    """Past index_topk (16 in the test checkpoint) every decode row reads its chosen blocks: windows of 2-8 rows
    give each row its one-row bits with the sparse kernel on (Metal) or its ops form (CPU), and without it."""

    from tensorfold.engine.lane_engine import LaneEngine

    monkeypatch.setattr(glm, "SPARSE_KERNEL", sparse)
    model = glm.load_backbone(checkpoint)
    base = model.make_cache()
    mx.eval(model.hidden(mx.array([_tokens(37, seed=5)]), base))
    for width in (2, 3, 5, 8):
        rows = [(101 + 13 * r) % TEXT["vocab_size"] for r in range(width)]
        one, many = LaneEngine.copy_single_cache(base), LaneEngine.copy_single_cache(base)
        serial = mx.concatenate([model.head(model.hidden(mx.array([[t]]), one)) for t in rows], axis=1)
        window = model.head(model.hidden(mx.array([rows]), many))
        assert _same(serial, window), width


def test_sparse_kernel_matches_the_gathered_attention(checkpoint, monkeypatch):
    """The indexed kernel reads the keys ``selected`` picks, in its order: one MLA decode row past the budget with the
    kernel and with gather + SDPA agree to bf16 rounding (logits of the random test model amplify rounding)."""

    model = glm.load_backbone(checkpoint)
    cache = model.make_cache()
    mx.eval(model.hidden(mx.array([_tokens(37, seed=5)]), cache))
    mla, c = model.layers[3].attn, cache[3]
    mx.random.seed(0)
    q = mx.random.normal((TEXT["num_attention_heads"], TEXT["qk_nope_head_dim"])).astype(mx.bfloat16)
    iq = mx.random.normal((TEXT["index_n_heads"], TEXT["index_head_dim"])).astype(mx.bfloat16)
    iw = mx.ones((TEXT["index_n_heads"],), dtype=mx.bfloat16)
    position = 36
    scores = mla.index_scores(iq[None], iw[None], c.pool[:(position + 1) // 4])
    chosen = np.array(mla.selected(scores, position)).tolist()
    ids = np.array(mla._sparse_indices(iq[None], iw[None], c, [position]))[0].tolist()
    assert ids[:len(chosen)] == chosen and all(v == -1 for v in ids[len(chosen):])
    outs = {}
    for sparse in (True, False):
        monkeypatch.setattr(glm, "SPARSE_KERNEL", sparse)
        outs[sparse] = mla._decode_row(q, iq, iw, c, position).astype(mx.float32)
    diff = float(mx.abs(outs[True] - outs[False]).max())
    assert diff <= 0.02 * float(mx.abs(outs[False]).max()) + 1e-3


def test_trim_never_scores_a_stale_pool_block(checkpoint):
    """mlx-vlm #2107's stale-pool bug (a trimmed cache reusing pooled blocks of the positions it dropped) cannot
    happen here: the pool of a block is rewritten when its last position is written again, and a query scores only
    blocks complete at its own position. A rolled-back cache that takes other tokens == a cache that never saw
    the dropped ones."""

    model = glm.load_backbone(checkpoint)
    prompt = _tokens(37, seed=9)
    a, b = model.make_cache(), model.make_cache()
    mx.eval(model.hidden(mx.array([prompt]), a), model.hidden(mx.array([prompt]), b))
    for t in (40, 41, 42, 43, 44, 45):                         # a: 6 tokens that will be dropped
        model.hidden(mx.array([[t]]), a)
    for c in a:
        if isinstance(c, glm.MLACache):
            c.trim(6)
    kda_b = [c for c in b if isinstance(c, glm.KDACache)]
    for i, c in enumerate(a):                                  # KDA states are not what this checks: copy them
        if isinstance(c, glm.KDACache):
            c.conv, c.ssm, c.offset = kda_b[0].conv, kda_b[0].ssm, kda_b[0].offset
            kda_b.pop(0)
    for t in (60, 61, 62, 63, 64, 65, 66):
        la = model.head(model.hidden(mx.array([[t]]), a))
        lb = model.head(model.hidden(mx.array([[t]]), b))
        assert _same(la, lb), t
    for c1, c2 in zip(a, b):
        if isinstance(c1, glm.MLACache):
            n = c1.offset // 4
            assert _same(c1.pool[:n], c2.pool[:n])
