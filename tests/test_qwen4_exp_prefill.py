"""Qwen3.8 Flash Next's fast prefill path (GPU): each piece against MLX or the reference forward on small layers.

The kernels read 4-bit, group-32 weights; the modules here are small but have the shapes the kernels need (256-dim
heads, hidden sizes that are multiples of 256). TF_FLASH_PREFILL=0 gives the reference forward to compare with.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.families.qwen4_exp import model as q4

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

TEXT = {
    "hidden_size": 256, "num_hidden_layers": 4, "vocab_size": 97, "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
    "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1]},
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "output_gate_type": "sigmoid",
    "num_experts": 16, "num_experts_per_tok": 4, "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64, "hc_count": 4, "hc_lowrank": 32,
    "indexer_n_heads": 2, "indexer_head_dim": 128, "indexer_budget": 16, "indexer_compress_ratio": 4,
    "ple_layer_ids": [1], "ple_embed_dim": 512, "ple_conv_kernel_size": 4, "ngram_size": 3,
    "heads_per_ngram": 2, "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8,
    "split_ngram_parts": 8, "eos_token_id": 5,
}


@pytest.fixture(autouse=True)
def _gpu(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    monkeypatch.setenv("TF_FLASH_PREFILL", "1")
    yield
    mx.set_default_device(previous)


def config(**overrides) -> q4.Config:
    return q4.Config.from_dict({"text_config": {**TEXT, **overrides}})


def q4bits(module, keep=()):
    """bf16 weights, then 4-bit group-32 layers (except those whose path ends with a name in ``keep``)."""

    module.set_dtype(mx.bfloat16)
    nn.quantize(module, group_size=32, bits=4,
                class_predicate=lambda path, m: hasattr(m, "to_quantized") and not path.endswith(keep))
    mx.eval(module.parameters())
    return module


def rel(a, b) -> float:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.sqrt(mx.sum((a - b) ** 2)) / mx.sqrt(mx.sum(b * b))).item())


# -- MLX's kernels with other tiles: same bits -------------------------------------------------------------------
@pytest.mark.parametrize(("m", "n", "k"), [(512, 640, 512), (512, 8192, 512), (96, 64, 512), (4096, 324, 1024)])
def test_matmul_is_mlx_bit_for_bit(m, n, k):
    """The tuned tile where MLX runs qmm (64 x 64 for wide outputs, else 64 x 32), MLX itself where it splits K
    ((96, 64, 512))."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

    mx.random.seed(0)
    x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
    w = mx.random.randint(0, 2**31, (n, k // 8), dtype=mx.uint32)
    s = (mx.random.normal((n, k // 32)) * 0.02).astype(mx.bfloat16)
    b = (mx.random.normal((n, k // 32)) * 0.02).astype(mx.bfloat16)
    ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=32, bits=4)
    assert bool(mx.array_equal(prefill_mm.matmul(x, w, s, b), ref).item())
    if not prefill_mm._mlx_splits_k(m, n, k):
        assert bool(mx.array_equal(prefill_mm.qmm(x, w, s, b), ref).item())


def test_linear_falls_back_off_the_fast_path(monkeypatch):
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

    layer = q4bits(nn.Linear(256, 64, bias=False))
    small = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)           # under 64 rows: layer(x)
    assert bool(mx.array_equal(prefill_mm.linear(layer, small), layer(small)).item())
    monkeypatch.setenv("TF_FLASH_PREFILL", "0")
    big = mx.random.normal((1, 96, 256)).astype(mx.bfloat16)
    assert not prefill_mm.active(96)
    assert bool(mx.array_equal(prefill_mm.linear(layer, big), layer(big)).item())


def test_tiles_pass_their_self_check_on_this_gpu():
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

    assert prefill_mm._self_check()
    assert prefill_mm.tiles() == (not prefill_mm._tensor_units())


@pytest.mark.parametrize("why", ["tensor units", "check fails", "kernel does not build"])
def test_mlx_matmuls_serve_when_the_tiles_step_aside(monkeypatch, why):
    """On an M5 (tensor units), or when the kernels built from this MLX's headers do not reproduce its bits or do
    not build, every matmul is MLX's own and the MoE still matches the reference."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

    def broken():
        raise RuntimeError("no such kernel")

    monkeypatch.setattr(prefill_mm, "_tiles", [])
    if why == "tensor units":
        monkeypatch.setattr(prefill_mm, "_tensor_units", lambda: True)
    else:
        monkeypatch.setattr(prefill_mm, "_self_check", (lambda: False) if why == "check fails" else broken)
    monkeypatch.setattr(prefill_mm, "qmm", None)                         # any use would raise
    monkeypatch.setattr(prefill_mm, "gather_sorted", None)
    mx.random.seed(12)
    layer = q4bits(nn.Linear(256, 8192, bias=False))
    x = mx.random.normal((1, 128, 256)).astype(mx.bfloat16)
    assert bool(mx.array_equal(prefill_mm.linear(layer, x), layer(x)).item())
    assert not prefill_mm.tiles()
    moe = q4bits(q4.SparseMoE(config()), keep=("gate",))
    x = (mx.random.normal((1, 128, 256)) * 0.5).astype(mx.bfloat16)
    fast = moe(x)
    monkeypatch.setenv("TF_FLASH_PREFILL", "0")
    assert rel(fast, moe(x)) < 1e-2


def test_gather_sorted_is_mlx_bit_for_bit():
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

    mx.random.seed(1)
    experts, k, n, m = 16, 512, 64, 400
    idx = mx.array(np.sort(np.random.default_rng(2).integers(0, experts, size=m)).astype(np.uint32))
    x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
    w = mx.random.randint(0, 2**31, (experts, n, k // 8), dtype=mx.uint32)
    s = (mx.random.normal((experts, n, k // 32)) * 0.02).astype(mx.bfloat16)
    b = (mx.random.normal((experts, n, k // 32)) * 0.02).astype(mx.bfloat16)
    ref = mx.gather_qmm(x[:, None, :], w, s, b, rhs_indices=idx, transpose=True, group_size=32, bits=4,
                        sorted_indices=True)[:, 0, :]
    assert bool(mx.array_equal(prefill_mm.gather_sorted(x, w, s, b, idx), ref).item())


# -- modules against the reference forward --------------------------------------------------------------------
def test_moe_matches_the_reference(monkeypatch):
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

    mx.random.seed(3)
    moe = q4bits(q4.SparseMoE(config()), keep=("gate",))                # the router stays bf16, as in the checkpoint
    x = (mx.random.normal((1, 128, 256)) * 0.5).astype(mx.bfloat16)
    assert prefill_mm.moe_applies(moe, x)
    fast = moe(x)
    monkeypatch.setenv("TF_FLASH_PREFILL", "0")
    assert not prefill_mm.moe_applies(moe, x)
    ref = moe(x)
    assert rel(fast, ref) < 1e-2, rel(fast, ref)


def test_fused_hyper_connection_matches_the_reference():
    from tensorfold.families.qwen4_exp.decode import _HC
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_hc

    mx.random.seed(4)
    cfg = config()
    hc = q4bits(q4.HyperConnection(cfg))
    hc.hc_norm.weight = 0.1 * mx.random.normal(hc.hc_norm.weight.shape)
    weights = _HC(hc, inject=True)                                      # the fused decode's stacked rows
    eps = mx.array([cfg.rms_norm_eps], dtype=mx.float32)
    h = mx.random.normal((96, 4 * 256)).astype(mx.bfloat16)
    ref_mixed, ref_inject = hc(h[None])
    _, mixed, inject = prefill_hc.hyper_connection(weights, h, None, streams=4, eps=eps)
    assert rel(mixed, ref_mixed[0]) < 1e-2 and rel(inject, ref_inject[0]) < 2e-2
    # with the previous block's write-back: the streams are the reference's bit for bit
    branch = mx.random.normal((96, 256)).astype(mx.bfloat16)
    gates = (2 * mx.sigmoid(mx.random.normal((96, 4)))).astype(mx.bfloat16)
    h_new, mixed2, _ = prefill_hc.hyper_connection(weights, h, (branch, gates), streams=4, eps=eps)
    written = q4._write_back(h[None], branch[None], gates[None])[0]
    assert bool(mx.array_equal(h_new, written).item())
    assert rel(mixed2, hc(written[None])[0][0]) < 1e-2


def _selection_mask(ids, counts, sparse, keys):
    """A boolean [1, 1, L, keys] mask from index_select's key lists (all keys up to its end for a dense row)."""

    m = np.zeros((len(counts), keys), dtype=bool)
    ids = np.array(ids) if ids is not None else None
    for r, (n, sp) in enumerate(zip(counts, sparse)):
        if sp:
            m[r, ids[r, :n]] = True
        else:
            m[r, :n] = True
    return mx.array(m)[None, None]


def _assert_top_blocks(ids, scores, complete, top):
    """Each sparse row's selected blocks (the first key of each, index_select's layout) are a top-``top`` of its
    complete blocks' scores [L, blocks] (ties either way)."""

    ids = np.array(ids)
    for r, c in enumerate(complete):
        if c <= top:
            continue
        chosen = ids[r, : 4 * top: 4] // 4
        valid = scores[r, :c]
        rest = np.setdiff1d(np.arange(c), chosen)
        assert len(set(chosen.tolist())) == top and chosen.max() < c, r
        assert valid[chosen].min() >= valid[rest].max() - 1e-4 * max(1.0, abs(valid[rest].max())), r


def _reference_block_scores(indexer, query, pooled, past):
    """Indexer.select's fp32 block scores [L, blocks] for queries [1, L, HI, DI] at offset ``past``."""

    q = indexer.q_layernorm(query).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, indexer.rotary_dim, traditional=False, base=indexer.base, scale=1.0, offset=past)
    scores = q.astype(mx.float32) @ pooled.astype(mx.float32)[:, None].transpose(0, 1, 3, 2)
    return np.array(mx.sum(mx.maximum(scores, 0), axis=1)[0] / np.sqrt(indexer.dims))


@pytest.mark.parametrize("gqa", [True, False])
def test_sparse_prefill_attention_matches_masked_attention(monkeypatch, gqa):
    """Past the indexer's budget (4 blocks here) a row reads only its selected blocks and tail. The reference applies
    its selection as a mask over every key: each row's selected blocks must be a top-k of the reference's scores, and
    with the same selection the outputs agree. The selections themselves differ only on ties: with 2 indexer heads many blocks score exactly 0 after the ReLU, index_select takes the
    lower block and argpartition either (``test_index_selection_is_a_top_k_of_the_reference_scores``). Two chunks: the
    second extends a cache whose pooled blocks the first made."""

    from tensorfold.kernels.qwen.flash_next.v1 import attention as K
    from tensorfold.kernels.qwen.flash_next.v1 import prefill as P

    mx.random.seed(5)
    attn = q4bits(q4.SparseAttention(config()))
    if not gqa:                                                         # the decode kernel attention_rows instead
        monkeypatch.setattr(P, "gqa_supported", lambda *a: False)
    chunks = [(mx.random.normal((1, n, 256)) * 0.5).astype(mx.bfloat16) for n in (80, 72)]
    selected = []
    original = K.index_select

    def recording(q, pooled, complete, ends, *, top):
        ids = original(q, pooled, complete, ends, top=top)
        selected.append((ids, complete, ends, top))
        return ids

    monkeypatch.setattr(K, "index_select", recording)
    cache = q4.AttentionCache()
    fast = [attn(x, cache) for x in chunks]
    mx.eval(fast)
    assert len(selected) == 2                                           # both chunks went through the sparse path

    def same_selection(query, raw, cache, past):
        ids, complete, ends, top = selected.pop(0)
        q4.Indexer.select(attn.indexer, query, raw, cache, past)       # keeps the pooled blocks up to date
        _assert_top_blocks(ids, _reference_block_scores(attn.indexer, query, cache.pooled, past), complete, top)
        ratio = attn.indexer.ratio
        sparse = [c > top for c in complete]
        counts = [ratio * top + e - ratio * c if sp else e for e, c, sp in zip(ends, complete, sparse)]
        return _selection_mask(ids, counts, sparse, past + len(ends))

    monkeypatch.setenv("TF_FLASH_PREFILL", "0")
    monkeypatch.setattr(attn.indexer, "select", same_selection)
    cache = q4.AttentionCache()
    ref = [attn(x, cache) for x in chunks]
    for got, want in zip(fast, ref):
        assert rel(got, want) < 1e-2, rel(got, want)


def test_index_selection_is_a_top_k_of_the_reference_scores():
    """index_select's blocks for each row are a top-``top`` of the reference's fp32 block scores (ties may go either
    way: it takes the lower block, argpartition any)."""

    from tensorfold.kernels.qwen.flash_next.v1 import attention as K

    mx.random.seed(9)
    rng = np.random.default_rng(9)
    rows, heads, dims, blocks, top = 64, 4, 128, 700, 512
    q = mx.array(rng.standard_normal((rows, heads, dims)).astype(np.float32)).astype(mx.bfloat16)
    pooled = mx.array(rng.standard_normal((blocks, dims)).astype(np.float32)).astype(mx.bfloat16)
    ends = [4 * blocks - rows + r + 1 for r in range(rows)]
    complete = [e // 4 for e in ends]
    ids = K.index_select(q, pooled, complete, ends, top=top)
    scores = np.maximum(np.einsum("rhd,bd->rhb", np.array(q.astype(mx.float32)), np.array(pooled.astype(mx.float32))),
                        0).sum(axis=1) / np.sqrt(dims)
    _assert_top_blocks(ids, scores, complete, top)


def test_attention_rows_gqa_matches_attention_rows():
    from tensorfold.kernels.qwen.flash_next.v1 import attention as K
    from tensorfold.kernels.qwen.flash_next.v1 import prefill as P

    rng = np.random.default_rng(6)
    rows, heads, kv, dims, cap = 48, 24, 2, 256, 2048
    q = mx.array(rng.standard_normal((rows, heads, dims)).astype(np.float32)).astype(mx.bfloat16)
    keys = mx.array(rng.standard_normal((1, kv, cap, dims)).astype(np.float32)).astype(mx.bfloat16)
    values = mx.array(rng.standard_normal((1, kv, cap, dims)).astype(np.float32)).astype(mx.bfloat16)
    sparse = [r % 3 != 0 for r in range(rows)]
    counts = [int(rng.integers(40, 300)) if sp else int(rng.integers(1, 900)) for sp in sparse]
    ids = np.zeros((rows, 300), dtype=np.int32)                         # a sparse row's key ids (dense rows: 0 .. n-1)
    for r in range(rows):
        if sparse[r]:
            ids[r, :counts[r]] = np.sort(rng.choice(cap, size=counts[r], replace=False))
    ids = mx.array(ids)
    ref = K.attention_rows(q, keys, values, counts, ids, sparse, dims ** -0.5)
    for parts in (1, 4):
        got = P.attention_rows_gqa(q, keys, values, counts, ids, sparse, dims ** -0.5, parts=parts)
        assert rel(got, ref) < 1e-3, (parts, rel(got, ref))


def test_ngram_lookup_through_the_fused_tables_is_bit_identical():
    from tensorfold.kernels.qwen.flash_next.v1 import embed as K

    mx.random.seed(7)
    emb = q4bits(q4.NGramEmbedding(config(), 0))
    tables = K.PleTables(emb)
    tokens = np.random.default_rng(8).integers(6, 97, size=(1, 200))
    tokens[0, 50] = 5
    ids = emb.ids(np.full((1, emb.context), emb.eos, dtype=np.int64), tokens)
    ref = emb(ids)
    emb.__dict__["fused_tables"] = tables
    assert bool(mx.array_equal(emb(ids), ref).item())


# -- the whole model: the fast prefill against the reference forward, then decode on its cache ------------------
def tiny_model(seed: int = 0) -> q4.Qwen4Exp:
    """Four layers (3 DeltaNet + 1 sparse attention, PLE on layer 1), 512 wide with 32 experts (the fused decode's
    expert kernels need K % 512 == 0 and a multiple of 32 experts), quantized as the checkpoint is (all but the router), with the fused decode attached as
    loading attaches it."""

    from mlx.utils import tree_flatten

    from tensorfold.families.qwen4_exp.decode import FusedDecode

    mx.random.seed(seed)
    model = q4.Qwen4Exp(config(hidden_size=512, num_experts=32))
    # centred norms start at zero; give them some weight so (1 + w) is exercised
    model.load_weights([(name, 0.1 * mx.random.normal(v.shape) if "norm" in name and name.endswith("weight") else v)
                        for name, v in tree_flatten(model.parameters())])
    q4bits(model, keep=("mlp.gate",))
    model.__dict__["fused"] = FusedDecode(model)
    return model


def test_whole_model_prefill_is_as_exact_as_the_reference(monkeypatch):
    """Two chunks (the second past the indexer's budget of 64 keys, extending the first's caches), then four decode
    steps on the caches each prefill left, against the same weights run in fp32. A random 4-layer MoE amplifies
    bf16 rounding (routing flips), so the fast path and the bf16 reference each land ~15% from fp32 here; the fast
    path must land as close as the reference (seeds 0-5: 0.99-1.09x its error, top-1 within 0.02). The decode steps
    run the reference forward: the fused decode's kernels are built for the checkpoint's shapes (48 value heads, 512
    experts, ...) and are not this test's subject."""

    tokens = np.random.default_rng(11).integers(6, 97, size=(1, 404))
    tokens[0, 100] = 5

    def run(model, fused=None):
        cache = model.make_cache()
        if fused is not None:
            model.__dict__["fused"] = fused
        logits = [model(tokens[:, a:b], cache) for a, b in ((0, 160), (160, 400))]
        model.__dict__.pop("fused", None)
        logits += [model(tokens[:, t:t + 1], cache) for t in range(400, 404)]
        return np.array(mx.concatenate(logits, axis=1)[0].astype(mx.float32))

    model = tiny_model()
    fused = model.__dict__["fused"]
    fast = run(model, fused)
    monkeypatch.setenv("TF_FLASH_PREFILL", "0")
    ref = run(model, fused)
    exact = tiny_model()                                                # the same weights, fp32 activations
    del exact.__dict__["fused"]
    for layer in exact.layers:
        if "ple" in layer:
            del layer.ple.ple_embedding.__dict__["fused_tables"]
    exact.set_dtype(mx.float32)
    truth = run(exact)

    def error(a):
        return np.linalg.norm(a - truth) / np.linalg.norm(truth), (a.argmax(-1) == truth.argmax(-1)).mean()

    (fast_err, fast_top), (ref_err, ref_top) = error(fast), error(ref)
    assert fast.shape == (404, 97) and np.isfinite(fast).all()
    assert fast_err < 1.2 * ref_err and fast_top > ref_top - 0.04, (fast_err, ref_err, fast_top, ref_top)


# -- prefix-snapshot keys name the prefill path ------------------------------------------------------------------
def test_snapshot_keys_name_the_prefill_path(monkeypatch):
    from types import SimpleNamespace

    from tensorfold import families
    from tensorfold.families.qwen4_exp.runtime import FlashNext

    runtime = FlashNext.__new__(FlashNext)                              # the key reads ``fused`` alone
    runtime.fused = object()
    fast = runtime.prefill_key
    monkeypatch.setenv("TF_FLASH_PREFILL", "0")
    reference = runtime.prefill_key
    monkeypatch.setenv("TF_FLASH_PREFILL", "1")
    runtime.fused = None                                                # no fused decode: MLX's forward
    assert fast != reference == runtime.prefill_key
    family = SimpleNamespace(package=SimpleNamespace(KERNEL_VERSION="v1"), module="tensorfold.families.qwen4_exp",
                             model_type="qwen4_exp")
    keys = {families.kernel_version(family, SimpleNamespace(prefill_key=k)) for k in (fast, reference)}
    keys.add(families.kernel_version(family, object()))                # a model without one: the kernels alone
    assert len(keys) == 3 and all(k.startswith("qwen4_exp-v1-") for k in keys)
