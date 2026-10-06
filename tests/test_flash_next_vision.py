"""Flash Next image prompts on MLX: the cache's rotary positions through the prompt path and the fused kernels."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.families.qwen4_exp import model as q4
from tensorfold.vision.rotary import attach_positions, frequency_axes, multimodal_rope, row_positions

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")


def image_table(tokens: int, begin: int, side: int) -> tuple[np.ndarray, int]:
    """Positions [3, tokens] of a prompt with a side x side image at row ``begin`` (Qwen's rope index) and its delta."""

    table = np.repeat(np.arange(tokens, dtype=np.int32)[None], 3, axis=0)
    end = begin + side * side
    table[0, begin:end] = begin
    table[1, begin:end] = begin + np.repeat(np.arange(side), side)
    table[2, begin:end] = begin + np.tile(np.arange(side), side)
    after = begin + side
    table[:, end:] = np.arange(after, after + tokens - end)
    return table, after - end


def test_rows_read_the_prompt_positions_then_continue_past_its_delta():
    table, delta = image_table(12, 2, 3)                 # rows 2..10 the image, 11 text after it
    cache = SimpleNamespace(keys=None)
    attach_positions([cache, SimpleNamespace()], table[:, None], delta)
    assert delta == -6 and cache.vision_rope_delta == -6
    np.testing.assert_array_equal(row_positions(cache, 0, 12), table)
    np.testing.assert_array_equal(row_positions(cache, 10, 4), [[2, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]])
    np.testing.assert_array_equal(row_positions(cache, 0, 4, step=4), [[0, 2, 2, 6], [0, 2, 4, 6], [0, 4, 2, 6]])
    assert row_positions(SimpleNamespace(keys=None), 0, 4) is None          # a text cache: rows at their index


def test_three_axis_rotation_picks_each_frequencys_axis():
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((1, 2, 5, 12)).astype(np.float32))
    positions = np.array([[3, 3, 3, 4, 9], [3, 4, 5, 4, 9], [3, 5, 4, 4, 9]], dtype=np.int32)
    got = np.array(multimodal_rope(x, 8, 1000.0, positions, [2, 1, 1]))
    rows = x.transpose(2, 1, 0, 3)
    per_axis = [np.array(mx.fast.rope(rows, 8, traditional=False, base=1000.0, scale=1.0,
                                      offset=mx.array(positions[a])).transpose(2, 1, 0, 3)) for a in range(3)]
    want = per_axis[0].copy()
    for column, axis in enumerate(frequency_axes(8, [2, 1, 1]) * 2):
        want[..., column] = per_axis[axis][..., column]
    assert np.array_equal(got, want)
    text = np.repeat(np.arange(4, 9, dtype=np.int32)[None], 3, axis=0)       # equal axes: plain rope at each row
    plain = mx.fast.rope(x, 8, traditional=False, base=1000.0, scale=1.0, offset=4)
    assert np.allclose(np.array(multimodal_rope(x, 8, 1000.0, text, [2, 1, 1])), np.array(plain), atol=1e-6)


# -- the prompt path (CPU, a tiny random model) ------------------------------------------------------------------
TINY = {
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


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny(seed: int = 0) -> q4.Qwen4Exp:
    from mlx.utils import tree_flatten

    mx.random.seed(seed)
    model = q4.Qwen4Exp(q4.Config.from_dict({"text_config": TINY}))
    model.load_weights([(name, 0.1 * mx.random.normal(v.shape) if "norm" in name and name.endswith("weight") else v)
                        for name, v in tree_flatten(model.parameters())])
    mx.eval(model.parameters())
    return model


def image_prompt(length: int = 27, begin: int = 3, side: int = 4, seed: int = 1):
    """Token ids (placeholder 7 on the image's rows), input rows (features there) and the rotary table."""

    rng = np.random.default_rng(seed)
    tokens = rng.integers(10, 97, size=length)
    tokens[begin:begin + side * side] = 7
    table, delta = image_table(length, begin, side)
    return tokens, table, delta


def rows_for(model, tokens, seed: int = 2):
    rows = model.model.embed_tokens(mx.array(tokens.astype(np.int32)))
    features = mx.array(np.random.default_rng(seed).standard_normal((len(tokens), rows.shape[-1])).astype(np.float32))
    image = mx.array((tokens == 7)[:, None])
    return mx.where(image, features.astype(rows.dtype), rows)


def test_an_image_prompt_whole_equals_row_by_row_and_decodes_past_its_delta(cpu):
    """27 rows, 6 complete blocks past a budget of 2: pooled blocks and queries sit at the image's positions."""

    model = tiny()
    tokens, table, delta = image_prompt()
    embeddings = rows_for(model, tokens)
    whole, stepwise = model.make_cache(), model.make_cache()
    for cache in (whole, stepwise):
        attach_positions(cache, table, delta)
    a = model.head(model.hidden(tokens[None], whole, embeddings))[0, -1]
    for t in range(len(tokens)):
        b = model.head(model.hidden(tokens[None, t:t + 1], stepwise, embeddings[t:t + 1]))[0, -1]
    assert whole[3].pooled.shape[1] == 6
    assert np.allclose(np.array(a), np.array(b), atol=2e-4), float(mx.abs(a - b).max())
    nxt = np.array([[11, 12]])                                          # decode: positions 27 + delta on
    a = model(nxt, whole)[0, -1]
    b = model(nxt[:, :1], stepwise)
    b = model(nxt[:, 1:], stepwise)[0, -1]
    assert np.allclose(np.array(a), np.array(b), atol=2e-4), float(mx.abs(a - b).max())
    # past the prompt a row sits at row + delta: as a table that already held the two rows, and not at the row
    longer, _ = image_table(len(tokens) + 2, 3, 4)
    outs = []
    for positions, shift in ((longer, 0), (table, 0)):
        cache = model.make_cache()
        attach_positions(cache, positions, shift)
        model.hidden(tokens[None], cache, embeddings)
        outs.append(np.array(model(nxt, cache)[0, -1]))
    assert np.allclose(np.array(a), outs[0], atol=2e-4) and not np.allclose(np.array(a), outs[1], atol=1e-3)


def test_text_positions_on_the_cache_give_the_text_prompt(cpu):
    """An image-free table and the tokens' own rows give the text forward."""

    model = tiny()
    tokens, _, _ = image_prompt()
    text, carried = model.make_cache(), model.make_cache()
    attach_positions(carried, np.repeat(np.arange(len(tokens), dtype=np.int32)[None], 3, axis=0), 0)
    a = model(tokens[None], text)[0]
    b = model.head(model.hidden(tokens[None], carried, model.model.embed_tokens(mx.array(tokens.astype(np.int32)))))[0]
    assert np.allclose(np.array(a), np.array(b), atol=1e-5)


def test_the_image_axes_reach_the_logits(cpu):
    model = tiny()
    tokens, table, delta = image_prompt()
    embeddings = rows_for(model, tokens)
    moved = table.copy()
    moved[1:, 3:19] = moved[0, 3:19]                                    # the image without its h and w spread
    outs = []
    for positions in (table, moved):
        cache = model.make_cache()
        attach_positions(cache, positions, delta)
        outs.append(np.array(model.head(model.hidden(tokens[None], cache, embeddings))[0, -1]))
    assert not np.allclose(outs[0], outs[1], atol=1e-4)


def record_positions(monkeypatch):
    """The three-axis rows each fused attn_prep and index_pool call receives, recorded, the calls run as before."""

    from tensorfold.kernels.qwen.flash_next.v1 import attention

    seen = {"attn": [], "pool": []}
    prep, pool = attention.attn_prep, attention.index_pool

    def rows(positions, count):
        return np.array(positions.tolist()[:3 * count]).reshape(count, 3).tolist()

    def attn_prep(projected, positions, *args, sections=None, **kwargs):
        if sections is not None:
            seen["attn"].append(rows(positions, projected.shape[0]))
        return prep(projected, positions, *args, sections=sections, **kwargs)

    def index_pool(raw, start, stop, *args, positions=None, sections=None, **kwargs):
        if sections is not None:
            seen["pool"].append(rows(positions, stop - start))
        return pool(raw, start, stop, *args, positions=positions, sections=sections, **kwargs)

    monkeypatch.setattr(attention, "attn_prep", attn_prep)
    monkeypatch.setattr(attention, "index_pool", index_pool)
    return seen


# -- the fused decode kernels (Metal) ---------------------------------------------------------------------------
@metal
def test_three_axis_kernels_with_equal_axes_keep_the_text_bits_and_pick_each_axis():
    from tensorfold.kernels.inputs import ints
    from tensorfold.kernels.qwen.flash_next.v1 import attention

    rng = np.random.default_rng(3)
    heads, kv_heads, dims, index_heads, index_dims = 4, 2, 256, 2, 128
    rows = 5
    width = heads * 2 * dims + 2 * kv_heads * dims + (index_heads + 1) * index_dims
    p = mx.array(rng.standard_normal((rows, width)).astype(np.float32)).astype(mx.bfloat16)
    norms = [mx.array(1 + 0.1 * rng.standard_normal(n).astype(np.float32)) for n in (dims, dims, index_dims)]
    eps = mx.array([1e-6], dtype=mx.float32)
    shape = {"q_heads": heads, "kv_heads": kv_heads, "head_dim": dims, "index_heads": index_heads,
             "index_dim": index_dims, "rotary_dim": 64, "base": 1e7}
    sections = (11, 11, 10)
    at = np.arange(40, 40 + rows)
    text = attention.attn_prep(p, mx.array(at, dtype=mx.int32), *norms, eps, **shape)
    equal = attention.attn_prep(p, ints(np.repeat(at, 3)), *norms, eps, **shape, sections=sections)
    assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(text, equal))
    three = np.stack([at, at + 7, at - 3])                              # [3, R]: each frequency on its own axis
    got = attention.attn_prep(p, ints(three.T.reshape(-1)), *norms, eps, **shape, sections=sections)
    per_axis = [attention.attn_prep(p, mx.array(three[a], dtype=mx.int32), *norms, eps, **shape) for a in range(3)]
    axes = frequency_axes(64, sections) * 2
    for out, parts in zip(got, zip(*per_axis)):
        want = np.array(parts[0].astype(mx.float32))
        for column, axis in enumerate(axes):
            want[..., column] = np.array(parts[axis].astype(mx.float32))[..., column]
        assert np.array_equal(np.array(out.astype(mx.float32)), want)
    raw = mx.array(rng.standard_normal((32, index_dims)).astype(np.float32)).astype(mx.bfloat16)
    pool = {"rotary_dim": 64, "base": 1e7}
    blocks = np.repeat(4 * np.arange(2, 8), 3)
    for relative in (False, True):
        source = raw[8:] if relative else raw
        plain = attention.index_pool(source, 2, 8, norms[2], eps, relative=relative, **pool)
        axes3 = attention.index_pool(source, 2, 8, norms[2], eps, relative=relative, positions=ints(blocks),
                                     sections=sections, **pool)
        assert bool(mx.array_equal(plain, axes3).item())


@metal
def test_an_image_stream_shares_a_forward_with_a_text_stream_at_its_own_bits(monkeypatch):
    """hidden_rows over an image stream and a text stream equals each stream alone."""

    from tensorfold.families.qwen4_exp.runtime import FlashNext
    from tests.test_qwen4_exp_prefill import tiny_model

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        monkeypatch.setattr(FlashNext, "check_windows", lambda self: (4, {1: 1.0, 4: 2.0}))
        model = tiny_model(vocab_size=128, moe_intermediate_size=512, shared_expert_intermediate_size=512,
                           rope_parameters={"rope_theta": 10000, "partial_rotary_factor": 0.25,
                                            "mrope_section": [11, 11, 10]})
        runtime = FlashNext(model, drafts=0)
        prompt = np.random.default_rng(4).integers(10, 128, size=(1, 40))
        table, delta = image_table(40, 4, 5)

        def prefilled(image: bool):
            cache = runtime.make_cache()
            if image:
                attach_positions(cache, table, delta)
            mx.eval(runtime.hidden(prompt, cache))
            return cache

        windows = [[7, 11, 13], [19, 23]]
        together = runtime.head(runtime.hidden_rows(windows, [prefilled(True), prefilled(False)]))
        alone = [runtime.head(runtime.hidden_rows([w], [prefilled(image)])) for w, image in zip(windows, (True, False))]
        assert bool(mx.array_equal(together[:, :3], alone[0]).item())
        assert bool(mx.array_equal(together[:, 3:], alone[1]).item())
        # the fused rows of an image stream past its prompt reach the kernels at row + delta on every axis
        seen = record_positions(monkeypatch)
        mx.eval(runtime.hidden(mx.array([[7, 11, 13]], dtype=mx.uint32), prefilled(True)))
        assert seen["attn"] and all(rows == [[40 + delta + r] * 3 for r in range(3)] for rows in seen["attn"])
    finally:
        mx.set_default_device(previous)


@metal
def test_a_short_last_chunk_with_image_rows_runs_the_fused_kernels_at_the_images_positions(monkeypatch):
    """A six-row last chunk (four image rows) and the decode after it reach the fused kernels at the image's axes."""

    from tensorfold.families.qwen4_exp.runtime import FlashNext
    from tests.test_qwen4_exp_prefill import tiny_model

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        monkeypatch.setattr(FlashNext, "check_windows", lambda self: (4, {1: 1.0, 4: 2.0}))
        model = tiny_model(vocab_size=128, moe_intermediate_size=512, shared_expert_intermediate_size=512,
                           rope_parameters={"rope_theta": 10000, "partial_rotary_factor": 0.25,
                                            "mrope_section": [11, 11, 10]})
        runtime = FlashNext(model, drafts=0)
        tokens = np.random.default_rng(5).integers(10, 128, size=24)
        tokens[6:22] = 7
        table, delta = image_table(24, 6, 4)
        embeddings = rows_for(model, tokens)
        encoded = SimpleNamespace(token_ids=tuple(int(t) for t in tokens), inputs_embeds=embeddings[None])
        cache = runtime.make_cache()
        attach_positions(cache, table, delta)
        runtime.prefill_vision(None, cache, encoded, 0, 18)
        seen = record_positions(monkeypatch)
        mx.eval(runtime.prefill_vision(None, cache, encoded, 18, 24))
        assert runtime._streams.shape[0] == 6 and runtime._streams is runtime.fused.last_streams
        assert seen["attn"] and all(rows == table[:, 18:24].T.tolist() for rows in seen["attn"])
        # the indexer's blocks 0..5 (past its budget of 4) pooled at each block's first row's axes
        assert seen["pool"] and all(rows == table[:, 0:24:4].T.tolist() for rows in seen["pool"])
        seen["attn"].clear()
        mx.eval(runtime.hidden(mx.array([[11, 12]], dtype=mx.uint32), cache))
        assert seen["attn"] and all(rows == [[24 + delta] * 3, [25 + delta] * 3] for rows in seen["attn"])
    finally:
        mx.set_default_device(previous)


@metal
def test_an_image_stream_drafts_with_the_mtp_head_to_its_serial_reply():
    """MTP drafts on an image prompt (an oracle draws, every third wrong) equal serial and shared runs."""

    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.families.qwen4_exp.model import CenteredRMSNorm
    from tensorfold.families.qwen4_exp.mtp import FlashMTP
    from tensorfold.families.qwen4_exp.mtp_cache import MTPCache
    from tensorfold.families.qwen4_exp.runtime import FlashNext
    from tests.test_qwen4_exp_prefill import q4bits, tiny_model

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        model = tiny_model(vocab_size=128, moe_intermediate_size=512, shared_expert_intermediate_size=512,
                           indexer_budget=32, rope_parameters={"rope_theta": 10000, "partial_rotary_factor": 0.25,
                                                               "mrope_section": [11, 11, 10]})
        mx.random.seed(1)
        head = FlashMTP(model.args)
        for _, module in head.named_modules():
            if isinstance(module, CenteredRMSNorm):
                module.weight = 0.1 * mx.random.normal(module.weight.shape)
        q4bits(head, keep=("mlp.gate",))
        runtime = FlashNext(model, head, drafts=3)
        if runtime.mtp is None:     # the load-time check: on an M5 this small model's rows are not always exact
            pytest.skip("this GPU's multi-row forward is not exact for the small model: the engine drafts nothing")
        tokens = [int(t) for t in np.random.default_rng(6).integers(10, 128, size=44)]
        tokens[26:42] = [7] * 16
        table, delta = image_table(44, 26, 4)
        embeddings = rows_for(model, np.array(tokens))
        attached = []

        class Tower:
            def encode(self, prepared):
                return SimpleNamespace(token_ids=tuple(tokens), inputs_embeds=embeddings[None], rope_delta=delta)

        runtime.vision = Tower()
        prepared = SimpleNamespace(token_ids=tuple(tokens), position_ids=table[:, None])
        encode = runtime.encode_vision
        runtime.encode_vision = lambda p, cache: (attached.append(cache), encode(p, cache))[1]

        def run(*specs):
            engine = LaneEngine(runtime)
            engine.prefill_plan = PrefillPlan(36)                    # chunks [0, 36) and [36, 44)
            streams = [LaneStream(stream_id=f"s{i}", prompt_ids=list(ids), max_new_tokens=24, prompt_data=data,
                                  drafts=drafts) for i, (ids, data, drafts) in enumerate(specs)]
            for stream in streams:
                engine.add_stream(stream)
            while engine.active_count:
                engine.step()
            return streams

        serial, = run((tokens, prepared, False))
        reply = serial.emitted

        def oracle(mixed, sampling, positions):
            mx.eval(mixed)                                  # the head's rows are computed, then the draw is chosen
            picks = []
            for p in positions:
                k = int(p) - len(tokens)
                right = reply[k] if 0 <= k < len(reply) else 0
                picks.append(right if k % 3 else (right + 1) % 128)
            return mx.array(picks, dtype=mx.uint32)

        runtime._draft_draw = oracle
        drafted, = run((tokens, prepared, True))
        text = [int(t) for t in np.random.default_rng(7).integers(10, 128, size=30)]
        shared, _ = run((tokens, prepared, True), (text, None, True))
        assert len(reply) == 24 and 0 < drafted.accepted < drafted.drafted         # drafts kept and rolled back
        assert drafted.emitted == reply == shared.emitted
        heads = [c[-1] for c in attached if isinstance(c[-1], MTPCache)]
        assert heads and all(h.vision_positions is not None and h.vision_rope_delta == delta for h in heads)
    finally:
        mx.set_default_device(previous)


