"""Qwen3.8 Flash Next family on a tiny random config (CPU): consistency, hashing, serial engine."""

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


def test_serial_engine_resumes_from_a_checkpoint_bit_identically():
    from tensorfold.engine.family_engine import SerialEngine
    from tensorfold.engine.lane_engine import LaneStream

    model = tiny()
    prompt = [int(t) for t in np.random.default_rng(3).integers(6, 97, size=18)]

    def run(checkpoints_at=()):
        engine = SerialEngine(model, retain_finished_caches=True)
        stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=12)
        engine.add_stream(stream, checkpoints_at=checkpoints_at)
        while engine.active_count:
            engine.step()
        return engine, stream

    engine, whole = run()
    assert len(whole.emitted) == 12
    tokens, cache = engine.finished_caches["s"]
    assert tokens == whole.context[:-1]
    # the reply's cache continues the conversation exactly like a fresh prefill of the same tokens
    follow = [*tokens, whole.emitted[-1], 7, 8]
    resumed = SerialEngine(model)
    a = LaneStream(stream_id="a", prompt_ids=follow, max_new_tokens=4)
    resumed.add_stream(a, cache=SerialEngine.copy_single_cache(cache), cached_tokens=len(tokens))
    while resumed.active_count:
        resumed.step()
    fresh = SerialEngine(model)
    b = LaneStream(stream_id="b", prompt_ids=follow, max_new_tokens=4)
    fresh.add_stream(b)
    while fresh.active_count:
        fresh.step()
    assert a.emitted == b.emitted
    _, split = run(checkpoints_at=(9,))
    assert split.emitted == whole.emitted and split.history_checkpoints[0][0] == prompt[:9]
