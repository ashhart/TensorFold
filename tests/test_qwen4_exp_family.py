"""Qwen3.8 Flash Next family on a tiny random config (CPU): consistency, hashing, the lane engine's family
rounds."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.qwen4_exp import model as q4  # noqa: E402

TEXT = {
    "hidden_size": 64, "num_hidden_layers": 4, "vocab_size": 97, "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1]},
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
    "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "output_gate_type": "sigmoid",
    "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32, "hc_count": 4, "hc_lowrank": 16,
    "indexer_n_heads": 2, "indexer_head_dim": 16, "indexer_budget": 8, "indexer_compress_ratio": 4,
    "ple_layer_ids": [2], "ple_embed_dim": 64, "ple_conv_kernel_size": 4, "ngram_size": 3,
    "heads_per_ngram": 2, "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8,
    "split_ngram_parts": 4, "eos_token_id": 5,
}


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny(seed: int = 0) -> q4.Qwen4Exp:
    mx.random.seed(seed)
    model = q4.Qwen4Exp(q4.Config.from_dict({"text_config": TEXT}))
    # centred norms start at zero; give them some weight so (1 + w) is exercised
    params = []
    for name, value in _flat(model.parameters()):
        if name.endswith("norm.weight") or "layernorm" in name or "hc_norm" in name or name.endswith("norm_key.weight"):
            value = 0.1 * mx.random.normal(value.shape)
        params.append((name, value))
    model.load_weights(params)
    mx.eval(model.parameters())
    return model


def _flat(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flat(v, f"{prefix}{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flat(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], tree


def _logits_whole(model, tokens):
    cache = model.make_cache()
    return model(np.array([tokens]), cache)[0, -1]


def _logits_stepwise(model, tokens):
    cache = model.make_cache()
    out = None
    for t in tokens:
        out = model(np.array([[t]]), cache)
    return out[0, -1]


@pytest.mark.parametrize("length", [7, 23])   # 23 keys: 5 complete blocks > 2 chosen, the sparse path
def test_whole_prompt_matches_token_by_token(length):
    model = tiny()
    rng = np.random.default_rng(1)
    tokens = [int(t) for t in rng.integers(6, 97, size=length)]
    tokens[4] = 5                                   # an EOS inside: the n-gram history restarts after it
    a = _logits_whole(model, tokens)
    b = _logits_stepwise(model, tokens)
    assert np.allclose(np.array(a), np.array(b), atol=2e-4), float(mx.abs(a - b).max())


def test_sparse_selection_is_used_past_the_budget():
    model = tiny()
    attn = model.layers[3].self_attn
    cache = model.make_cache()
    model(np.array([list(range(6, 30))]), cache)       # 24 keys, 6 blocks > 2
    assert cache[3].pooled is not None and cache[3].pooled.shape[1] == 6
    assert attn.indexer.top_blocks == 2


def _reference_ngram_ids(emb: q4.NGramEmbedding, history, tokens):
    """The reference's MLX formulation (shift with EOS resets, XOR of products, mod prime sizes)."""

    seq = mx.concatenate([mx.array(history, dtype=mx.int64), mx.array(tokens, dtype=mx.int64)], axis=-1)
    batch, width = seq.shape
    positions = mx.arange(width, dtype=mx.int64)
    eos_positions = mx.where(seq == emb.eos, positions, -1)
    previous = mx.concatenate([mx.full((batch, 1), -1, dtype=mx.int64), mx.cummax(eos_positions, axis=1)[:, :-1]], axis=1)
    in_segment = positions[None] - (previous + 1)
    shifted = []
    for shift in range(emb.n):
        source = positions - shift
        gathered = mx.take_along_axis(seq, mx.broadcast_to(mx.maximum(source, 0)[None], (batch, width)), axis=1)
        shifted.append(mx.where((in_segment >= shift) & (source[None] >= 0), gathered, emb.eos))
    mult = mx.array(emb.multipliers)
    blocks = []
    for ngram in range(2, emb.n + 1):
        first = (ngram - 2) * emb.per_ngram
        mixed = shifted[0] * mult[0]
        for p in range(1, ngram):
            mixed = mx.bitwise_xor(mixed, shifted[p] * mult[p])
        sizes = mx.array(emb.head_sizes[first:first + emb.per_ngram])
        offsets = mx.array(emb.head_offsets[first:first + emb.per_ngram])
        blocks.append(mixed[..., None] % sizes[None, None] + offsets[None, None])
    return np.array(mx.concatenate(blocks, axis=-1)[:, -tokens.shape[1]:])


def test_ngram_ids_match_the_reference_formula_at_full_vocab():
    cfg = q4.Config.from_dict({"text_config": {**TEXT, "vocab_size": 248320, "ngram_vocab_size_base": 20_000_000,
                                               "heads_per_ngram": 8, "ple_embed_dim": 2560 // 16 * 16,
                                               "make_ngram_vocab_size_divisible_by": 128,
                                               "split_ngram_parts": 128, "eos_token_id": 248044}})
    emb = q4.NGramEmbedding.__new__(q4.NGramEmbedding)
    # the tables are not needed for ids: build the hashing constants only
    emb.n, emb.context, emb.per_ngram = cfg.ngram_size, cfg.ngram_size - 1, cfg.heads_per_ngram
    emb.heads, emb.eos = emb.context * emb.per_ngram, cfg.ple_eos
    sizes, offsets, total = [], [], 0
    for head in range(emb.heads):
        size = q4._nth_prime_after(cfg.ngram_vocab_size_base - 1, head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    emb.head_sizes, emb.head_offsets = np.array(sizes, np.int64), np.array(offsets, np.int64)
    emb.multipliers = q4.layer_multipliers(cfg.vocab_size, cfg.ngram_size, 0, cfg.seed)
    assert (total + 127) // 128 * 128 == 320_001_536          # 128 shards of 2,500,012 rows, as shipped
    rng = np.random.default_rng(2)
    tokens = rng.integers(0, 248320, size=(2, 40))
    tokens[0, 7] = tokens[1, 0] = tokens[1, 1] = 248044
    history = np.array([[248044, 248044], [11, 248044]])
    assert np.array_equal(emb.ids(history, tokens), _reference_ngram_ids(emb, history, tokens))


def test_lane_engine_resumes_from_a_grid_checkpoint_bit_identically():
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.families.qwen4_exp.runtime import FlashNext

    model = FlashNext(tiny(), None, drafts=0)         # no fused kernels on the CPU: one token a round
    assert model.lane_family and model.exact_width == 1
    prompt = [int(t) for t in np.random.default_rng(3).integers(6, 97, size=18)]

    def engine_on(grid: int = 4) -> LaneEngine:
        engine = LaneEngine(model, retain_finished_caches=True)
        engine.prefill_plan = PrefillPlan(grid)       # the 2,048 grid, scaled to the tiny prompt
        assert engine.family
        return engine

    def run(engine, stream_id, ids, max_new, **kw):
        stream = LaneStream(stream_id=stream_id, prompt_ids=list(ids), max_new_tokens=max_new)
        engine.add_stream(stream, **kw)
        while engine.active_count:
            engine.step()
        return stream

    whole_engine = engine_on()
    whole = run(whole_engine, "s", prompt, 12)
    assert len(whole.emitted) == 12 and "s" not in whole_engine.finished_caches      # decoded states not kept
    split = run(engine_on(), "c", prompt, 12, checkpoints_at=(9,))
    tokens, cache = split.history_checkpoints[0]
    assert split.emitted == whole.emitted and tokens == prompt[:8]                    # the checkpoint moves to the grid
    # the next turn resumes from the grid checkpoint (the reply is prefilled again) like a fresh prefill
    follow = [*prompt, *whole.emitted, 7, 8]
    fresh = run(engine_on(), "b", follow, 4)
    resumed = run(engine_on(), "a", follow, 4, cache=LaneEngine.copy_single_cache(cache), cached_tokens=8)
    off_grid = run(engine_on(), "o", follow, 4, cache=LaneEngine.copy_single_cache(cache), cached_tokens=9)
    assert resumed.emitted == fresh.emitted == off_grid.emitted


def test_long_context_attention_in_query_parts(monkeypatch):
    """Past ``split_keys`` keys the reference attention runs its query rows in parts, each over the keys up to its
    last row: the same outputs up to rounding, and a resumed prompt still equals a fresh one bit for bit."""

    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.families.qwen4_exp.runtime import FlashNext

    model = tiny()
    prompt = [int(t) for t in np.random.default_rng(5).integers(6, 97, size=40)]
    whole = _logits_whole(model, prompt)
    monkeypatch.setattr(q4.SparseAttention, "split_keys", 8)
    monkeypatch.setattr(q4.SparseAttention, "split_rows", 3)
    parted = _logits_whole(model, prompt)
    assert np.allclose(np.array(parted.astype(mx.float32)), np.array(whole.astype(mx.float32)), atol=5e-2)

    flash = FlashNext(model, None, drafts=0)

    def run(ids, **kw):
        engine = LaneEngine(flash)
        engine.prefill_plan = PrefillPlan(8)
        stream = LaneStream(stream_id="x", prompt_ids=list(ids), max_new_tokens=4)
        engine.add_stream(stream, **kw)
        while engine.active_count:
            engine.step()
        return stream

    first = run(prompt[:30], checkpoints_at=(26,))
    tokens, cache = first.history_checkpoints[0]
    assert tokens == prompt[:24]
    fresh = run(prompt)
    resumed = run(prompt, cache=LaneEngine.copy_single_cache(cache), cached_tokens=24)
    assert resumed.emitted == fresh.emitted


def test_prompt_chunks_offer_8192_with_tensor_units(monkeypatch):
    from tensorfold.families import qwen3_5, qwen4_exp

    model = q4.Qwen4Exp(q4.Config.from_dict({"text_config": TEXT}))
    for units, steps in ((True, (8192, 4096, 2048)), (False, (4096, 2048))):
        monkeypatch.setattr(qwen3_5, "tensor_units", lambda units=units: units)
        settings = qwen4_exp.engine_settings(model)
        assert settings["prefill_steps"] == steps and "min_chunk" not in settings


def test_releasing_rounds_drops_the_last_forwards_rows():
    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.runtime import FlashNext

    fused = SimpleNamespace(row_states={0: [("conv", "ssm", 0)]}, last_streams="streams", _last_heads=["head"])
    runtime = SimpleNamespace(fused=fused, _streams="streams")
    FlashNext.release_rounds(runtime)
    assert fused.row_states == {} and fused.last_streams is None and fused._last_heads == [] and runtime._streams is None


def test_the_head_absorbs_every_prompt_row_into_its_cache_and_carries_only_the_last():
    from tensorfold.families.qwen4_exp.runtime import last_row_layer

    mx.random.seed(3)
    model = q4.Qwen4Exp(q4.Config.from_dict({"text_config": TEXT}))
    layer = next(layer for layer in model.layers if not layer.is_linear)
    wide = model.args.hc_count * model.args.hidden_size
    x = mx.random.normal((1, 24, wide))
    full_cache, trimmed_cache = q4.AttentionCache(), q4.AttentionCache()
    full = layer(x, None, full_cache)
    trimmed = last_row_layer(layer, x, trimmed_cache)
    for a, b in zip(full_cache.state, trimmed_cache.state):
        assert bool(mx.array_equal(a, b).item())
    assert trimmed.shape == (1, 1, wide)
    assert np.allclose(np.array(full[:, -1:]), np.array(trimmed), atol=1e-4)


def test_a_prompt_chunk_leaves_only_its_states_in_the_cache():
    model = tiny()
    tokens = (np.arange(1024, dtype=np.int64) % 90 + 6)[None]
    cache = model.make_cache()
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    out = model.hidden(tokens, cache)
    buffers = [a for c in cache for a in (
        (c.keys, c.values, c.index_keys, c.pooled) if isinstance(c, q4.AttentionCache) else (c.conv, c.ssm, c.ple_conv))
        if a is not None]
    mx.eval(out, *buffers)
    del out
    model.__dict__.pop("last_streams")
    mx.synchronize()
    mx.clear_cache()
    held = mx.get_active_memory() - before
    # the chunk's conv inputs (~2.5 MB here) must not stay behind views of their last rows; buffers round to pages
    assert held <= sum(a.nbytes + (16 << 10) for a in buffers) + (64 << 10)


def test_the_prompt_read_ahead_asks_for_each_chunks_lookup_ids():
    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.runtime import FlashNext

    model = tiny()
    emb = next(layer.ple.ple_embedding for layer in model.layers if "ple" in layer)
    asked = []
    emb.host = SimpleNamespace(read_ahead=asked.append)
    tokens = [int(t) for t in np.arange(20) * 7 % 90 + 6]
    tokens[9] = emb.eos                                      # an EOS inside the prompt resets the n-grams
    chunks = [(0, 5), (5, 12), (12, 20)]
    for begin, end in chunks:
        FlashNext.prefetch_prompt(SimpleNamespace(model=model), tokens, begin, end)
    history = np.full((1, emb.context), emb.eos, dtype=np.int64)       # as PLELayer carries it from chunk to chunk
    for (begin, end), ids in zip(chunks, asked, strict=True):
        chunk = np.array([tokens[begin:end]], dtype=np.int64)
        assert np.array_equal(ids, emb.ids(history, chunk))
        history = np.concatenate([history, chunk], axis=1)[:, -emb.context:]
