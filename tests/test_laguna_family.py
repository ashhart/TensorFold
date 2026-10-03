"""Laguna as a lane family on a tiny random oQ-layout checkpoint (Metal)."""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("the Laguna decode kernels are Metal kernels", allow_module_level=True)

from laguna_tiny import TINY, config, tiny_model, tokens  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402
from tensorfold.families.laguna.model import Laguna  # noqa: E402
from tensorfold.kernels.laguna.v1 import glue  # noqa: E402
from tensorfold.kernels.laguna.v1.matmul import Projection, regroup, tensor_units  # noqa: E402

BACKENDS = ["rows", pytest.param("lane", marks=pytest.mark.skipif(not tensor_units(), reason="needs tensor units"))]
copy = LaneEngine.copy_single_cache


@pytest.fixture(scope="module")
def model():
    return tiny_model()


def family(model, backend: str, width: int = 16) -> Laguna:
    out = Laguna(model, backend=backend, check=False)
    out.exact_width = width
    return out


def prefilled(lm: Laguna, n: int, seed: int = 3) -> list:
    cache = lm.make_cache()
    mx.eval(lm.prefill(mx.array([tokens(n, seed)], dtype=mx.uint32), cache))
    return cache


def steps(lm: Laguna, cache: list, window: list[int]) -> list:
    out = []
    for token in window:
        logits = lm.head(lm.hidden(mx.array([[token]], dtype=mx.uint32), cache))
        mx.eval(logits)
        out.append(logits[0, -1])
    return out


def same(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def close(a, b, tol: float) -> bool:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.abs(a - b).max().item()) <= tol * max(1e-6, float(mx.abs(b).max().item()))


# -- kernels against plain MLX ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("bits,group", [(4, 128), (5, 64), (6, 64), (8, 64), (8, 128), (4, 64)])
def test_projections_match_mlx_and_keep_each_rows_bits(backend, bits, group):
    import mlx.nn as nn

    mx.random.seed(bits * group)
    linear = nn.Linear(512, 192, bias=False)
    linear = nn.QuantizedLinear.from_linear(linear, group_size=group, bits=bits)
    linear.set_dtype(mx.bfloat16)
    proj = Projection([linear], backend)
    x = mx.random.normal((9, 512)).astype(mx.bfloat16)
    y = proj(x)
    assert close(y, linear(x), 0.02)
    for r in range(9):
        assert same(proj(x[r:r + 1])[0], y[r])
    assert same(proj(x[:4]), y[:4])


def test_regrouped_scales_dequantize_to_the_same_weights():
    import mlx.nn as nn

    mx.random.seed(1)
    q = nn.QuantizedLinear.from_linear(nn.Linear(256, 64, bias=False), group_size=128, bits=8)
    scales, biases = regroup(q.scales, q.biases, 128, 64)
    a = mx.dequantize(q.weight, q.scales, q.biases, group_size=128, bits=8)
    b = mx.dequantize(q.weight, scales, biases, group_size=64, bits=8)
    assert same(a, b)


def test_route_is_sigmoid_top_k_by_biased_scores_with_unbiased_weights():
    mx.random.seed(2)
    logits = (mx.random.normal((5, 64)) * 3).astype(mx.bfloat16)
    bias = (mx.random.uniform(shape=(64,)) * 0.5).astype(mx.bfloat16)
    ids, weights = glue.route(logits, bias, 6)
    mx.eval(ids, weights)
    scores = mx.sigmoid(logits.astype(mx.float32))
    pick = scores + bias.astype(mx.float32)
    for r in range(5):
        want = sorted(range(64), key=lambda e: (-float(pick[r, e].item()), e))[:6]
        got = [int(i) for i in ids[r * 6:r * 6 + 6].tolist()]
        assert got == want
        chosen = mx.array([float(scores[r, e].item()) for e in want])
        assert mx.allclose(weights[r * 6:r * 6 + 6], chosen / chosen.sum(), atol=1e-6)


def test_router_logits_match_mlx_and_keep_each_rows_bits():
    mx.random.seed(3)
    x = mx.random.normal((7, 512)).astype(mx.bfloat16)
    w = (mx.random.normal((64, 512)) * 0.05).astype(mx.bfloat16)
    y = glue.router_logits(x, w)
    assert close(y, x @ w.T, 0.02)
    for r in range(7):
        assert same(glue.router_logits(x[r:r + 1], w)[0], y[r])


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_decode_row_matches_mlx_s_forward(model, backend):
    """A one-token step through the kernels lies within mlx-lm's own spread (its step against its full forward)."""

    from mlx_lm.models.cache import KVCache, RotatingKVCache

    lm = family(model, backend)
    for n in (5, 40):                   # inside and past the sliding window
        ids = tokens(n + 1, seed=8)
        cache = prefilled(lm, n, seed=8)
        assert ids[:n] == tokens(n, seed=8)
        ours = lm.head(lm.hidden(mx.array([[ids[n]]], dtype=mx.uint32), cache))[0, -1].astype(mx.float32)
        ref = [RotatingKVCache(max_size=TINY["sliding_window"]) if layer.self_attn.is_sliding else KVCache()
               for layer in model.layers]
        model(mx.array([ids[:n]], dtype=mx.uint32), cache=ref)
        step = model(mx.array([[ids[n]]], dtype=mx.uint32), cache=ref)[0, -1].astype(mx.float32)
        full = model(mx.array([ids], dtype=mx.uint32))[0, -1].astype(mx.float32)
        spread = float(mx.abs(step - full).max().item())
        scale = float(mx.abs(full).max().item())
        assert float(mx.abs(ours - full).max().item()) <= max(2 * spread, 0.02 * scale), n


# -- the engine's exactness contract -----------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("prompt", [20, 150])
def test_windows_give_every_row_its_serial_bits(model, backend, prompt):
    lm = family(model, backend)
    base = prefilled(lm, prompt)
    window = tokens(14, seed=5)
    serial = steps(lm, copy(base), window)
    for width in range(2, len(window) + 1):
        logits = lm.head(lm.hidden(mx.array([window[:width]], dtype=mx.uint32), copy(base)))
        mx.eval(logits)
        assert all(same(logits[0, i], serial[i]) for i in range(width)), width


@pytest.mark.parametrize("backend", BACKENDS)
def test_rollback_then_steps_equal_serial_decoding(model, backend):
    lm = family(model, backend)
    base = prefilled(lm, 131)
    window = tokens(12, seed=5)
    serial = steps(lm, copy(base), window)
    work = copy(base)
    mx.eval(lm.hidden(mx.array([window[:3] + tokens(6, seed=9)], dtype=mx.uint32), work))
    lm.keep_rows(work, 9, 3)
    assert [c.offset for c in work] == [131 + 3] * len(work)
    assert all(same(a, b) for a, b in zip(steps(lm, work, window[3:]), serial[3:]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_shared_forward_gives_each_stream_its_own_bits(model, backend):
    lm = family(model, backend)
    assert lm.check_streams(None)
    bases = [prefilled(lm, n, seed=n) for n in (20, 133, 140)]
    windows = [tokens(n, seed=11 + n) for n in (3, 5, 1)]
    alone = []
    for base, window in zip(bases, windows):
        logits = lm.head(lm.hidden(mx.array([window], dtype=mx.uint32), copy(base)))
        mx.eval(logits)
        alone.append(logits[0])
    joint = lm.head(lm.hidden_rows(windows, [copy(b) for b in bases]))[0]
    mx.eval(joint)
    at = 0
    for window, own in zip(windows, alone):
        assert same(joint[at:at + len(window)], own)
        at += len(window)


def _reply(lm, prompt, n, *, sampling=None, cache=None, cached=0):
    engine = LaneEngine(lm, max_rows=16, max_draft=15)
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=n, sampling=sampling)
    engine.add_stream(stream, cache=cache, cached_tokens=cached)
    engine.run()
    return list(stream.emitted), engine


@pytest.mark.parametrize("backend", BACKENDS)
def test_concurrent_streams_emit_what_they_emit_alone(model, backend):
    lm = family(model, backend)
    prompts = [tokens(n, seed=30 + n) for n in (25, 140, 60, 9)]
    samplings = [None, Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95), None,
                 Sampling(seed=8, temperature=0.8, top_k=40, top_p=0.9)]
    alone = [_reply(lm, p, 20, sampling=s)[0] for p, s in zip(prompts, samplings)]
    engine = LaneEngine(lm, max_rows=16, max_draft=15)
    streams = []
    for i, (p, s) in enumerate(zip(prompts, samplings)):
        stream = LaneStream(stream_id=f"s{i}", prompt_ids=list(p), max_new_tokens=20, sampling=s)
        engine.add_stream(stream)
        streams.append(stream)
    engine.run()
    assert [list(s.emitted) for s in streams] == alone
    assert any(r.streams > 1 for r in engine.round_stats)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_prompt_resumed_from_a_grid_checkpoint_equals_a_fresh_one(model, backend, monkeypatch, tmp_path):
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    monkeypatch.setattr(LaneEngine, "prefill_plan", PrefillPlan(64))
    lm = family(model, backend)
    first = tokens(150, seed=40)
    reply, _ = _reply(lm, first, 12)
    engine = LaneEngine(lm, max_rows=16, max_draft=15, retain_finished_caches=True)
    stream = LaneStream(stream_id="a", prompt_ids=list(first), max_new_tokens=12)
    engine.add_stream(stream, checkpoints_at=(len(first),))
    engine.run()
    (at, checkpoint), = [(len(t), c) for t, c in stream.history_checkpoints]
    follow = first + reply + tokens(20, seed=41)
    fresh, _ = _reply(lm, follow, 16)
    resumed, _ = _reply(lm, follow, 16, cache=copy(checkpoint), cached=at)
    assert resumed == fresh
    path = save_snapshot(tmp_path, "laguna-test", first[:at], checkpoint)
    loaded_tokens, loaded = load_snapshot(path, "laguna-test")
    stored, _ = _reply(lm, follow, 16, cache=lm.adopt_cache(loaded), cached=at)
    assert stored == fresh


def test_load_time_checks_find_the_full_window(model):
    lm = Laguna(model, backend="rows", check=True)
    assert lm.exact_width == lm.fused_rows and lm.max_streams > 1


# -- checkpoints -------------------------------------------------------------------------------------------------------
def _write(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


OQ4E = {"group_size": 128, "bits": 4, "mode": "affine",
        "language_model.model.layers.1.self_attn.q_proj": {"group_size": 64, "bits": 8, "mode": "affine"},
        "language_model.model.layers.1.mlp.shared_expert.up_proj": {"group_size": 128, "bits": 8, "mode": "affine"},
        "language_model.lm_head": {"group_size": 64, "bits": 8, "mode": "affine"}}


def test_check_takes_oq_layouts_and_refuses_what_the_kernels_do_not_cover(tmp_path):
    from tensorfold import families
    from tensorfold.families import laguna

    path = _write(tmp_path, config(quantization=OQ4E, quantization_config=OQ4E))
    assert families.detect(path).module == "tensorfold.families.laguna"
    laguna.check(path)
    for change, match in (({"gating": False}, "gate"), ({"moe_router_score_func": "softmax"}, "sigmoid"),
                          ({"moe_intermediate_size": 2048}, "expert width")):
        _write(tmp_path, config(quantization=OQ4E, **change))
        with pytest.raises(ValueError, match=match):
            laguna.check(path)
    experts8 = dict(OQ4E, **{"language_model.model.layers.1.mlp.switch_mlp.up_proj": {"group_size": 64, "bits": 8}})
    _write(tmp_path, config(quantization=experts8))
    with pytest.raises(ValueError, match="not 4-bit"):
        laguna.check(path)
    _write(tmp_path, config(quantization={"group_size": 64, "bits": 4, "mode": "mxfp4"}))
    with pytest.raises(ValueError):
        laguna.check(path)


def test_load_reads_an_oq_checkpoint_with_language_model_prefixes(tmp_path):
    """Saved with oQ's ``language_model.`` prefix and per-layer widths, loaded back with every width in place."""

    from mlx.utils import tree_flatten

    from tensorfold.families.laguna.model import load_model

    original = tiny_model(seed=4)
    weights = {f"language_model.{k}": v for k, v in tree_flatten(original.parameters())}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    quant = {"group_size": 128, "bits": 4, "mode": "affine"}
    for name, module in original.named_modules():
        if hasattr(module, "bits") and (module.bits, module.group_size) != (4, 128):
            quant[f"language_model.{name}"] = {"group_size": module.group_size, "bits": module.bits, "mode": "affine"}
    _write(tmp_path, config(quantization=quant, quantization_config=quant))
    loaded, _ = load_model(tmp_path)
    x = mx.array([tokens(12, seed=2)], dtype=mx.uint32)
    assert same(loaded(x), original(x))
    assert loaded.model.layers[1].self_attn.q_proj.bits == 8
    assert not hasattr(loaded.model.layers[1].mlp.gate.proj, "scales")


# -- poolside's DFlash drafter ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sampled", [False, True])
def test_drafted_replies_equal_serial_ones(backend, sampled, tmp_path):
    from laguna_tiny import tiny_drafter

    from tensorfold.families.laguna.drafter import LagunaDrafter, is_laguna_drafter

    path = tiny_drafter(tmp_path / "draft")
    assert is_laguna_drafter(path)
    plain = tiny_model(seed=6)
    target = tiny_model(seed=6)
    draft = LagunaDrafter(target, str(path), bits=0)
    lm = Laguna(target, backend=backend, check=False, drafter=draft)
    lm.exact_width = 16
    ref = family(plain, backend)
    prompt = tokens(140, seed=21)
    sampling = Sampling(seed=5, temperature=1.0, top_k=20, top_p=0.95) if sampled else None
    serial, _ = _reply(ref, prompt, 40, sampling=sampling)
    engine = LaneEngine(lm, max_rows=16, max_draft=7)
    stream = LaneStream(stream_id="d", prompt_ids=list(prompt), max_new_tokens=40, sampling=sampling, drafts=True)
    engine.add_stream(stream)
    engine.run()
    assert list(stream.emitted) == serial
    assert engine.drafted > 0


def test_the_drafter_reads_each_tap_normed_and_the_context_through_each_layers_norm(tmp_path):
    """The draft block's hidden rows against a direct computation of vLLM's laguna_dflash for one layer."""

    from laguna_tiny import tiny_drafter

    from tensorfold.families.laguna.drafter import load_draft

    model = load_draft(tiny_drafter(tmp_path / "draft"))
    taps = mx.random.normal((1, 5, 512)).astype(mx.bfloat16)
    ctx = model.combine(taps)
    a, b = mx.split(taps, 2, axis=-1)
    want = model.hidden_norm(model.fc(mx.concatenate([model.aux_hidden_norms[0](a), model.aux_hidden_norms[1](b)],
                                                     axis=-1)))
    assert same(ctx, want)
    layer = model.layers[0]
    attn = layer.self_attn
    assert attn.is_causal and attn.is_sliding and attn.qkv_proj.weight.shape[0] == (4 + 2 * 2) * 128
