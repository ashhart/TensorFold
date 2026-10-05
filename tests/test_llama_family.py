"""Reproducible biased Llama fixtures exercise reference quality and lane exactness on Metal."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
llama = pytest.importorskip("mlx_lm.models.llama")
if not mx.metal.is_available():
    pytest.skip("Llama lane arithmetic needs Metal", allow_module_level=True)

from tensorfold.engine.lane_engine import LaneEngine, LaneStream

TINY = {"model_type": "llama", "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 64, "rms_norm_eps": 1e-5, "tie_word_embeddings": False,
        "attention_bias": True, "mlp_bias": True, "rope_theta": 1e6, "max_position_embeddings": 1024}
copy = LaneEngine.copy_single_cache


def tiny(*, quantized=False, kv=1, config=None):
    mx.random.seed(14)
    model = llama.Model(llama.ModelArgs.from_dict(config or dict(TINY, num_key_value_heads=kv)))
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name.endswith(("q_proj", "k_proj")):
                module.weight = module.weight * 2
            if "bias" in module:
                module.bias = mx.random.normal(module.bias.shape) * 0.15
    model.set_dtype(mx.bfloat16)
    if quantized:
        nn.quantize(model, bits=8, group_size=64,
                    class_predicate=lambda path, m: path.startswith("model.layers.") and isinstance(m, nn.Linear))
    mx.eval(model.parameters())
    return model


def family(model, **options):
    from tensorfold.families.llama.family import LlamaFamily

    return LlamaFamily(model, **options)


def tokens(n, seed=4):
    return [int(t) for t in np.random.default_rng(seed).integers(3, TINY["vocab_size"], size=n)]


def same(a, b):
    return bool(mx.array_equal(a, b).item())


def steps(model, cache, window):
    parts = [model.head(model.hidden(mx.array([[t]], dtype=mx.uint32), cache)) for t in window]
    out = mx.concatenate(parts, axis=1)
    mx.eval(out)
    return out


@pytest.mark.parametrize("kv", [1, 2])
def test_biased_rope_forward_matches_stock_llama(kv):
    stock = tiny(kv=kv)
    model = family(stock)
    prompt, window = tokens(19), tokens(8, seed=5)
    base = model.make_cache()
    pref = model.prefill(mx.array([prompt], dtype=mx.uint32), base)
    ref_cache = stock.make_cache()
    ref_pref = stock.model(mx.array([prompt], dtype=mx.uint32), ref_cache)
    assert same(pref, ref_pref)
    ref_hidden = mx.concatenate([stock.model(mx.array([[t]], dtype=mx.uint32), ref_cache) for t in window], axis=1)
    hidden = model.hidden(mx.array([window], dtype=mx.uint32), base)
    assert same(hidden, ref_hidden)
    ref_logits = mx.concatenate([stock.lm_head(ref_hidden[:, i:i + 1]) for i in range(len(window))], axis=1)
    assert same(model.head(hidden), ref_logits)
    # Each omission independently changes the trusted forward.
    for omit in ("theta", "bias"):
        other = tiny(kv=kv)
        other_cache = other.make_cache()
        for layer in other.model.layers:
            if omit == "theta":
                layer.self_attn.rope = nn.RoPE(64, traditional=False, base=10000)
            else:
                for _, module in layer.named_modules():
                    if isinstance(module, nn.Linear) and "bias" in module:
                        module.bias = mx.zeros_like(module.bias)
        other.model(mx.array([prompt]), other_cache)
        wrong = other.model(mx.array([window]), other_cache)
        assert not same(hidden, wrong), omit


@pytest.mark.parametrize("quantized", [False, True])
def test_all_widths_preserve_every_serial_row_and_cache(quantized):
    model = family(tiny(quantized=quantized))
    assert model.exact_width == 16
    assert set(model.window_costs) == set(range(1, 17))
    assert all(ms > 0 for ms in model.window_costs.values())
    base = model.make_cache()
    mx.eval(model.prefill(mx.array([tokens(31)]), base))
    window = tokens(16, seed=9)
    serial = steps(model, copy(base), window)
    for width in range(1, 17):
        work, ref = copy(base), copy(base)
        got = model.head(model.hidden(mx.array([window[:width]]), work))
        steps(model, ref, window[:width])
        assert same(got, serial[:, :width]), width
        assert all(same(a, b) for ca, cb in zip(work, ref) for a, b in zip(ca.state, cb.state)), width
    with pytest.raises(ValueError, match="width"):
        model.hidden(mx.array([tokens(17)]), copy(base))


@pytest.mark.parametrize("keep", [0, 1, 3, 8])
def test_partial_keep_restores_cache_and_continuation(keep):
    model = family(tiny(quantized=True))
    base = model.make_cache()
    mx.eval(model.prefill(mx.array([tokens(27)]), base))
    work, ref = copy(base), copy(base)
    window = tokens(8, seed=11)
    mx.eval(model.hidden(mx.array([window]), work))
    model.keep_rows(work, 8, list(range(keep)))
    if keep:
        steps(model, ref, window[:keep])
    assert [c.offset for c in work] == [27 + keep] * 2
    assert same(steps(model, work, [4, 5]), steps(model, ref, [4, 5]))


def test_uneven_shared_streams_keep_independent_prefixes():
    model = family(tiny(quantized=True))
    bases = [model.make_cache() for _ in range(3)]
    for n, cache in zip((13, 29, 41), bases):
        mx.eval(model.prefill(mx.array([tokens(n, seed=n)]), cache))
    windows = [tokens(n, seed=n + 10) for n in (3, 5, 1)]
    alone = [steps(model, copy(b), w) for b, w in zip(bases, windows)]
    shared = [copy(b) for b in bases]
    got = model.head(model.hidden_rows(windows, shared))
    assert same(got, mx.concatenate(alone, axis=1))
    model.keep_rows_streams(shared, [3, 5, 1], [1, 3, 0])
    for base, work, window, keep in zip(bases, shared, windows, [1, 3, 0]):
        ref = copy(base)
        if keep:
            steps(model, ref, window[:keep])
        assert same(steps(model, work, [7, 8]), steps(model, ref, [7, 8]))
    with pytest.raises(ValueError, match="chain"):
        model.hidden(mx.array([[1, 2, 3]]), copy(bases[0]), parents=[-1, 0, 0])


def test_quantized_projections_batch_rows_without_mutating_global_backend(monkeypatch):
    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    previous = row_matmul.BACKEND
    model = family(tiny(quantized=True))
    assert model.backend is not None
    projection = model.core.layers[0].self_attn.q_proj
    inputs = (mx.random.normal((1, 9, 128), key=mx.random.key(10)) * 0.2).astype(mx.bfloat16)
    expected = mx.concatenate([model.project(projection, inputs[:, i:i + 1]) for i in range(9)], axis=1)
    calls = []
    original = model.backend.qmm

    def counted(x, *args, **kwargs):
        calls.append(int(x.shape[-2]))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(model.backend, "qmm", counted)
    actual = model.project(projection, inputs)
    assert calls == [9]
    assert same(actual, expected)
    assert row_matmul.BACKEND is previous
    assert bool(mx.any(projection.bias != 0).item())
    assert projection.biases.shape != projection.bias.shape  # affine offsets are not model bias
    # Same quantized weights, stock arithmetic: report quality independently from lane bit equality.
    base = model.make_cache()
    mx.eval(model.prefill(mx.array([tokens(24)]), base))
    ours = steps(model, copy(base), tokens(8, seed=11))
    ref_cache = copy(base)
    stock = mx.concatenate([model.inner(mx.array([[t]]), ref_cache) for t in tokens(8, seed=11)], axis=1)
    error = float(mx.max(mx.abs(ours.astype(mx.float32) - stock.astype(mx.float32))).item())
    assert error < 0.03, error
    assert same(mx.argmax(ours, axis=-1), mx.argmax(stock, axis=-1))
    labels = mx.array(tokens(8, seed=12)).reshape(1, 8, 1)
    ours_nll = -mx.take_along_axis(nn.log_softmax(ours.astype(mx.float32)), labels, axis=-1).mean()
    ref_nll = -mx.take_along_axis(nn.log_softmax(stock.astype(mx.float32)), labels, axis=-1).mean()
    assert abs(float(ours_nll.item()) - float(ref_nll.item())) < 0.01
    stock_window = model.inner(mx.array([tokens(8, seed=11)]), copy(base))
    stock_variation = float(mx.max(mx.abs(stock_window.astype(mx.float32) - stock.astype(mx.float32))).item())
    assert stock_variation < 0.03, stock_variation


@pytest.mark.parametrize("quantized", [False, True])
def test_loader_uses_normalized_stock_model_with_local_weights(tmp_path, quantized):
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model

    from tensorfold.families.llama import load

    config = dict(TINY)
    config.pop("rope_theta")
    config["rope_parameters"] = {"rope_type": "default", "rope_theta": 1e6}
    if quantized:
        config["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine",
                                  "lm_head": False, "model.embed_tokens": False}
    config["eos_token_id"] = 1
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 7]}))
    original = tiny(quantized=quantized)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(tree_flatten(original.parameters())))
    # A public, nonpersonal vocabulary; no network or checkpoint download.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f"token{i}": i for i in range(64)}, unk_token="token0")), eos_token="token1", bos_token="token2")
    tokenizer.save_pretrained(tmp_path)
    model, loaded_tokenizer = load(tmp_path)
    assert model.inner.args.rope_theta == 1e6
    assert model.inner.args.attention_bias and model.inner.args.mlp_bias
    assert model.exact_width == 16
    assert loaded_tokenizer.encode("token7", add_special_tokens=False) == [7]
    assert loaded_tokenizer.eos_token_ids == {1, 7}
    assert same(model.inner.model.embed_tokens.weight, original.model.embed_tokens.weight)
    assert same(model.lm_head.weight, original.lm_head.weight)
    assert "scales" not in model.lm_head and "scales" not in model.core.embed_tokens
    stock, _ = load_model(tmp_path, model_config={"rope_theta": 1e6})
    ids = mx.array([tokens(21)])
    assert same(model.prefill(ids, model.make_cache()), stock.model(ids, stock.make_cache()))


class KnownReply:
    last_match = 1 << 30

    def __init__(self, expected):
        self.expected = list(expected)
        self.calls = 0

    def propose(self, context, max_draft):
        good = (6, 2, 0, 9, 3)[self.calls % 5]
        self.calls += 1
        out = self.expected[len(context):len(context) + max_draft]
        return [t if i < good else (t + i + 1) % 64 for i, t in enumerate(out)]


def reply(model, prompt, *, sampling=None, proposer=None, cache=None, cached=0):
    from tensorfold.engine.prefill_plan import PrefillPlan

    engine = LaneEngine(model, max_rows=16, max_draft=15, prefill_plan=PrefillPlan(16))
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=36, sampling=sampling,
                        drafts=proposer is not None, proposer=proposer)
    engine.add_stream(stream, cache=cache, cached_tokens=cached)
    engine.run()
    return list(stream.emitted), engine


@pytest.mark.parametrize("sampled", [False, True])
def test_engine_drafted_and_concurrent_equal_serial_with_real_counters(sampled):
    from tensorfold.engine.exact_sampling import Sampling

    model = family(tiny(quantized=True))
    prompt = tokens(41)
    sampling = Sampling(seed=7, temperature=0.9, top_k=30, top_p=0.95) if sampled else None
    serial, _ = reply(model, prompt, sampling=sampling)
    drafted, engine = reply(model, prompt, sampling=sampling, proposer=KnownReply(prompt + serial))
    assert serial == drafted
    assert engine.drafted > engine.accepted > 0
    assert max(r.rows for r in engine.round_stats) > 2
    prompts = [tokens(n, seed=n) for n in (23, 57, 32)]
    expected = [reply(model, p, sampling=sampling)[0] for p in prompts]
    concurrent = LaneEngine(model, max_rows=16, max_draft=15, prefill_plan=engine.prefill_plan)
    streams = [LaneStream(stream_id=str(i), prompt_ids=p, max_new_tokens=36, sampling=sampling,
                          drafts=True, proposer=KnownReply(p + want))
               for i, (p, want) in enumerate(zip(prompts, expected))]
    for stream in streams:
        concurrent.add_stream(stream)
    concurrent.run()
    assert [s.emitted for s in streams] == expected
    assert any(r.streams > 1 for r in concurrent.round_stats)
    assert concurrent.drafted > concurrent.accepted > 0


def test_resume_and_snapshot_keep_the_same_prefill_plan(tmp_path):
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    model = family(tiny(quantized=True))
    prompt = tokens(71)
    engine = LaneEngine(model, prefill_plan=PrefillPlan(16))
    stream = LaneStream(stream_id="prefix", prompt_ids=prompt, max_new_tokens=4)
    engine.add_stream(stream, checkpoints_at=(len(prompt),))
    (prefix, cache), = stream.history_checkpoints
    assert len(prefix) == 64
    fresh, _ = reply(model, prompt)
    resumed, _ = reply(model, prompt, cache=copy(cache), cached=len(prefix))
    assert fresh == resumed
    path = save_snapshot(tmp_path, "llama-fixture", prefix, cache)
    loaded_tokens, loaded_cache = load_snapshot(path, "llama-fixture")
    assert loaded_tokens == prefix
    stored, _ = reply(model, prompt, cache=model.adopt_cache(loaded_cache), cached=len(prefix))
    assert stored == fresh
    model.release_rounds()
    assert not model._last


def test_score_labels_uses_lane_engine_without_generating(monkeypatch):
    model = family(tiny(quantized=True))
    engine = LaneEngine(model)
    prompt, labels = tokens(43), [3, 7, 22]
    reference = model.head(model.prefill(mx.array([prompt]), model.make_cache()))[0, -1].astype(mx.float32)
    expected = reference[mx.array(labels)].tolist()
    total = float(mx.logsumexp(reference).item())

    def never_decode(*args, **kwargs):
        pytest.fail("label scoring must not decode or sample")

    monkeypatch.setattr(model, "hidden", never_decode)
    monkeypatch.setattr(engine, "_draw", never_decode)
    values, normalizer = engine.score_labels(prompt, labels)
    assert values == expected
    assert normalizer == total
    stream = LaneStream(stream_id="decision", prompt_ids=prompt, max_new_tokens=0)
    stream.label_ids = labels
    engine.add_stream(stream)
    assert stream.scored == (values, normalizer)
    assert stream.emitted == [] and stream.finish_reason == "decision"
    assert not engine.round_stats


@pytest.mark.parametrize("hidden,heads", [(1536, 12), (2048, 16)])
def test_basal_projection_geometry_keeps_bits_with_one_random_layer(hidden, heads):
    config = dict(TINY, hidden_size=hidden, intermediate_size=hidden * 2,
                  num_attention_heads=heads, num_key_value_heads=2, num_hidden_layers=1)
    model = family(tiny(quantized=True, config=config))
    assert model.exact_width == 16 and model.streams_exact
    cache = model.make_cache()
    mx.eval(model.prefill(mx.array([tokens(9)]), cache))
    wanted = steps(model, copy(cache), tokens(8, seed=19))
    actual = model.head(model.hidden(mx.array([tokens(8, seed=19)]), copy(cache)))
    assert same(actual, wanted)


def test_snapshot_key_tracks_reused_quantized_kernels(monkeypatch):
    from pathlib import Path

    from tensorfold import families
    from tensorfold.families import llama as package

    dependencies = package.KERNEL_DEPENDENCIES
    module = "tensorfold.kernels.qwen.dense.v1.simd_qmm_bits"
    assert module in dependencies
    before = families.kernel_source_version(families.families()["llama"])
    read_bytes = Path.read_bytes

    def changed(path):
        data = read_bytes(path)
        return data + b"\n# test revision\n" if path.name == "simd_qmm_bits.py" else data

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert families.kernel_source_version(families.families()["llama"]) != before


def test_prepared_quantized_paths_survive_other_backend_preparation(monkeypatch):
    from tensorfold import families
    from tensorfold.kernels.qwen.dense.v1 import affine_rows, row_matmul, simd_qmm_bits

    monkeypatch.setattr(simd_qmm_bits, "fallback", set())
    model = family(tiny(quantized=True))
    descriptor = families.families()["llama"]
    key = families.kernel_version(descriptor, model)
    projection = model.core.layers[0].self_attn.q_proj
    inputs = (mx.random.normal((1, 9, 128), key=mx.random.key(10)) * 0.2).astype(mx.bfloat16)
    expected = model.project(projection, inputs)
    mx.eval(expected)
    # Another model's failed scalar/MMA check changes the process-wide dispatcher.
    monkeypatch.setattr(simd_qmm_bits, "check", lambda *a, **k: False)
    row_matmul.simd_qmm_backend().prepare([
        (projection.weight, projection.scales, projection.biases, 64, 8)])
    assert (128, 128, 8, 64) in simd_qmm_bits.fallback
    calls = []
    original = affine_rows.qmm

    def counted(*args, **kwargs):
        calls.append("affine")
        return original(*args, **kwargs)

    monkeypatch.setattr(affine_rows, "qmm", counted)
    assert same(model.project(projection, inputs), expected)
    assert calls == []
    assert families.kernel_version(descriptor, model) == key
    other = family(tiny(quantized=True))
    assert families.kernel_version(descriptor, other) != key
    assert "qmm=" in key


def test_context_copy_proposer_is_used_by_existing_engine():
    from tensorfold.engine.lane_engine import SuffixLookupProposer
    from tensorfold.engine.prefill_plan import PrefillPlan

    model = family(tiny(quantized=True))
    chain = tokens(8, seed=21)
    prompt = chain + tokens(20, seed=23) + tokens(5, seed=26) + chain[:7]
    emitted = []
    for drafts in (False, True):
        proposer = SuffixLookupProposer(ngram=3, min_match=8, window=8)
        stream = LaneStream(stream_id="copy", prompt_ids=prompt, max_new_tokens=20, drafts=drafts, proposer=proposer)
        # The same forced first token completes the eight-token lookup in both runs.
        stream.force = [chain[7]]
        engine = LaneEngine(model, max_rows=16, prefill_plan=PrefillPlan(16))
        engine.add_stream(stream)
        engine.run()
        emitted.append(stream.emitted)
    assert emitted[0] == emitted[1]
    assert stream.proposer.proposals > 0
    assert engine.drafted > 0


def test_single_window_probe_failure_closes_the_loader(monkeypatch):
    from tensorfold.families.llama.family import LlamaFamily

    original = LlamaFamily.hidden_rows

    def wrong_one_row(self, windows, caches, parents=None):
        out = original(self, windows, caches, parents)
        return out + mx.array(1, dtype=out.dtype)

    # The serial reference and width-one check must not silently qualify inconsistent calls.
    calls = 0

    def wrong_on_repeat(self, windows, caches, parents=None):
        nonlocal calls
        calls += 1
        return wrong_one_row(self, windows, caches, parents) if calls == 17 else original(self, windows, caches, parents)

    monkeypatch.setattr(LlamaFamily, "hidden_rows", wrong_on_repeat)
    with pytest.raises(ValueError, match="width-one"):
        family(tiny())


@pytest.mark.parametrize("probe", ["window", "shared"])
def test_startup_checks_live_kv_not_only_outputs(monkeypatch, probe):
    from tensorfold.families.llama.family import LlamaFamily

    original = LlamaFamily.hidden_rows

    def corrupt_cache(self, windows, caches, parents=None):
        out = original(self, windows, caches, parents)
        corrupt = len(windows) > 1 if probe == "shared" else len(windows) == 1 and len(windows[0]) > 1
        if corrupt:
            for cache in caches:
                cache[0].keys = cache[0].keys + mx.array(1, dtype=mx.bfloat16)
        return out

    monkeypatch.setattr(LlamaFamily, "hidden_rows", corrupt_cache)
    model = family(tiny(quantized=True))
    if probe == "window":
        assert model.exact_width == model.batch_rows == 1
    else:
        assert not model.streams_exact


def test_startup_checks_continuation_after_keep(monkeypatch):
    from tensorfold.families.llama.family import LlamaFamily

    original = LlamaFamily.keep_rows

    def corrupt_keep(self, cache, rows, keep):
        original(self, cache, rows, keep)
        cache[0].values = cache[0].values + mx.array(1, dtype=mx.bfloat16)

    monkeypatch.setattr(LlamaFamily, "keep_rows", corrupt_keep)
    with pytest.raises(ValueError, match="rollback"):
        family(tiny(quantized=True))


def test_window_crossing_stock_kv_allocation_boundary_keeps_prefix_bits():
    model = family(tiny(quantized=True))
    base = model.make_cache()
    mx.eval(model.prefill(mx.array([tokens(252)]), base))
    window = tokens(12, seed=32)
    ref = steps(model, copy(base), window)
    work = copy(base)
    actual = model.head(model.hidden(mx.array([window]), work))
    assert same(actual, ref)
    model.keep_rows(work, 12, 6)
    assert same(steps(model, work, window[6:]), ref[:, 6:])


def test_startup_refuses_unsupported_width_and_context_copy_head_options(tmp_path):
    from tensorfold.families.llama import load
    from tensorfold.families.llama.family import LlamaFamily

    for width in (0, 17):
        with pytest.raises(ValueError, match="width"):
            LlamaFamily(tiny(), widest=width)
    (tmp_path / "config.json").write_text(json.dumps(TINY))
    with pytest.raises(ValueError, match="context-copy"):
        load(tmp_path, drafter="unsupported-head")


def test_negative_keep_is_refused_before_trimming():
    model = family(tiny())
    cache = model.make_cache()
    mx.eval(model.hidden(mx.array([[3, 4]]), cache))
    with pytest.raises(ValueError, match="prefix"):
        model.keep_rows(cache, 2, -1)
    assert [c.offset for c in cache] == [2, 2]
