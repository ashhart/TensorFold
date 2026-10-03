"""The GLM-5.3-Flash family on a tiny checkpoint: loading, both paths, exact windows, rollback, drafts, streams."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from glm5_fakes import TEXT, write_checkpoint  # noqa: E402
from tensorfold.families.glm5_next import caches, linear, mlp, weights  # noqa: E402
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
    return weights.load_backbone(checkpoint)


def tokens(n: int, seed: int = 1) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(6, TEXT["vocab_size"], size=n)]


def test_family_is_detected_and_checked(checkpoint):
    from tensorfold import families
    from tensorfold.families import glm5_next

    assert families.detect(checkpoint).module == "tensorfold.families.glm5_next"
    glm5_next.check(checkpoint)
    assert glm5_next.has_mtp(checkpoint)


def test_check_reads_json_only_and_names_the_mlx_it_needs(tmp_path, monkeypatch):
    """``check`` starts no MLX (the CLI runs it before the family's MLX environment) and refuses an old MLX."""

    import json
    import subprocess
    import sys
    from importlib import metadata

    from tensorfold.families import glm5_next

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm5_next", "text_config": TEXT,
                                                      "quantization": {"bits": 4, "group_size": 64}}))
    code = ("import sys; from tensorfold.families import glm5_next; glm5_next.check(sys.argv[1]); "
            "print('mlx.core' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "PYTHONPATH": ":".join(sys.path)})
    assert out.stdout.strip().splitlines()[-1] == "False"
    monkeypatch.setattr(metadata, "version", lambda name: "0.32.0")
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="mlx>=0.32.2"):
        glm5_next.check(tmp_path)


def test_loads_the_checkpoint_layout(checkpoint):
    model = backbone(checkpoint)
    kinds = ["kda" if layer.is_linear else "mla" for layer in model.layers]
    assert kinds == ["kda", "kda", "kda", "mla", "kda", "mla"]
    assert isinstance(model.layers[0].mlp, mlp.DenseMLP) and isinstance(model.layers[1].mlp, mlp.MoE)
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


def test_multimodal_embedding_prefill_matches_the_equivalent_token_embeddings(checkpoint):
    model = backbone(checkpoint)
    ids = mx.array([tokens(15, seed=12)], dtype=mx.uint32)
    embeddings = model.embed_tokens(ids.reshape(-1))
    token_cache, embedding_cache = model.make_cache(), model.make_cache()
    token_hidden = model.hidden(ids, token_cache)
    image_path_hidden = model.hidden(ids, embedding_cache, inputs_embeds=embeddings)
    assert bool(mx.array_equal(token_hidden, image_path_hidden).item())
    for left, right in zip(token_cache, embedding_cache):
        assert left.offset == right.offset == 15


def test_multimodal_embedding_prefill_rejects_wrong_shapes(checkpoint):
    model = backbone(checkpoint)
    ids = mx.array([tokens(3)], dtype=mx.uint32)
    with pytest.raises(ValueError, match="match the prompt rows"):
        model.hidden(ids, model.make_cache(), inputs_embeds=mx.zeros((2, TEXT["hidden_size"])))


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
            n = c1.offset if isinstance(c1, caches.MLACache) else None
            if n is not None and x.shape[0] >= n:
                x, y = x[:n], y[:n]
            assert bool(mx.array_equal(x, y).item())
    # and both continue alike
    la = model.head(model.hidden(mx.array([[21]]), a))
    lb = model.head(model.hidden(mx.array([[21]]), b))
    assert bool(mx.array_equal(la, lb).item())


def _run_engine(runtime, prompt, n, drafts=True):
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.glm5_next import engine_settings

    engine = LaneEngine(runtime, **engine_settings(runtime))
    assert engine.family
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=n, drafts=drafts)
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
    assert drafted.exact_width >= 2 and drafted.mtp_step_ms >= 0.0
    prompt = tokens(21, seed=4)
    engine_a, a = _run_engine(drafted, prompt, 24)
    engine_b, b = _run_engine(serial, prompt, 24)
    assert engine_a.family_mtp and not engine_b.family_mtp
    assert engine_a.drafted > 0
    assert a.emitted == b.emitted
    # the same model with drafts off for the request: the serial reference through the one-step-ahead rounds
    _, c = _run_engine(drafted, prompt, 24, drafts=False)
    assert c.emitted == b.emitted


@pytest.mark.parametrize(("device", "grid"), [("cpu", 8), ("gpu", 8), ("gpu", 32)])
def test_image_prefill_across_chunks_and_mtp_matches_the_equivalent_embeddings(checkpoint, device, grid):
    """An image crosses a chunk boundary; serial and MTP decode agree with a token-embedding reference."""
    from types import SimpleNamespace

    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.families.glm5_next import engine_settings
    from tensorfold.vision.glm_mlx import GLMVisionFrontend
    from tensorfold.vision.glm_processing import PreparedGLMVisionPrompt

    if device == "gpu":
        if not mx.metal.is_available():
            pytest.skip("needs Metal")
        mx.set_default_device(mx.gpu)
    model = backbone(checkpoint)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    if device == "cpu":
        runtime.exact_width = runtime.batch_rows = min(runtime.exact_width, 7)
    reference = tokens(70, seed=12)
    begin, end = grid - 2, grid + 4
    prompt = list(reference)
    prompt[begin:end] = [10] * 6
    features = model.embed_tokens(mx.array(reference[begin:end], dtype=mx.uint32))

    class ImageTower:
        patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.bfloat16)))

        def __call__(self, pixels, image_grid):
            return features

    config = {"image_token_id": 10, "vision_config": {"patch_size": 14, "temporal_patch_size": 2,
              "spatial_merge_size": 2, "out_hidden_size": TEXT["hidden_size"]}}
    processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 10),
                                image_processor=SimpleNamespace(patch_size=14, temporal_patch_size=2, merge_size=2))
    runtime.vision = GLMVisionFrontend(config, model.embed_tokens, ImageTower(), processor, mx)
    prepared = PreparedGLMVisionPrompt(tuple(prompt), np.zeros((24, 1176), dtype=np.float32),
                                      np.asarray([[1, 4, 6]], dtype=np.int64), ((begin, end),), ("image",))

    def run(ids, prompt_data=None, drafts=False):
        engine = LaneEngine(runtime, **engine_settings(runtime))
        engine.prefill_plan = PrefillPlan(grid)
        stream = LaneStream(stream_id="image", prompt_ids=list(ids), prompt_data=prompt_data,
                            max_new_tokens=20, drafts=drafts)
        engine.add_stream(stream, checkpoints_at=(grid,))
        while engine.active_count:
            engine.step()
        return stream, engine

    expected, _ = run(reference)
    serial, _ = run(prompt, prepared)
    drafted, engine = run(prompt, prepared, drafts=True)
    assert serial.emitted == drafted.emitted == expected.emitted
    assert engine.drafted > 0 and engine.prefill_chunks > 1
    assert not drafted.history_checkpoints


def test_draft_head_absorbs_vision_rows_instead_of_placeholder_ids(checkpoint):
    """Prompt absorb has to hand the draft head the vision rows, not embed(image token)."""
    from types import SimpleNamespace

    model = backbone(checkpoint)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=1)
    ids = mx.array([1, 10, 10, 2], dtype=mx.uint32)
    base = model.embed_tokens(ids)
    vision_rows = mx.ones((2, TEXT["hidden_size"]), dtype=base.dtype)
    embeds = mx.concatenate([base[:1], vision_rows, base[3:]], axis=0)
    cache = runtime.make_cache()
    runtime.prefill_vision(ids, cache, SimpleNamespace(inputs_embeds=embeds[None]), 0, 4)
    seen = {}
    real = runtime.mtp

    def wrapped(model, h, tokens, caches, lengths, decode, embeddings=None):
        seen["embeddings"] = embeddings
        return real(model, h, tokens, caches, lengths, decode, embeddings=embeddings)

    runtime.mtp = wrapped
    hidden = mx.zeros((1, 3, TEXT["hidden_size"]), dtype=base.dtype)
    runtime.absorb_draft_context(hidden, mx.array([10, 10, 2], dtype=mx.uint32), cache)
    got = seen["embeddings"]
    assert got is not None and bool(mx.array_equal(got, embeds[1:4]).item())
    placeholder = model.embed_tokens(mx.array([10, 10, 2], dtype=mx.uint32))
    assert not bool(mx.array_equal(got, placeholder).item())
    seen.clear()
    runtime.absorb_draft_context(hidden, mx.array([4, 5, 6], dtype=mx.uint32), cache)
    assert seen["embeddings"] is None


@pytest.mark.parametrize(("grid", "length", "cut", "kept"), [(8, 30, 26, 24), (32, 100, 80, 64)])
def test_lane_engine_resumes_from_a_chunk_start(checkpoint, tmp_path, grid, length, cut, kept):
    """A checkpoint at a chunk start, in memory or read back from disk, resumes exactly like a fresh prefill."""

    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    model = backbone(checkpoint)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=2)
    prompt = tokens(length, seed=5)

    def run(ids, **kw):
        engine = LaneEngine(runtime)
        engine.prefill_plan = PrefillPlan(grid)                     # 32-row chunks take the prompt path, 8-row decode's
        stream = LaneStream(stream_id="x", prompt_ids=list(ids), max_new_tokens=8)
        engine.add_stream(stream, **kw)
        while engine.active_count:
            engine.step()
        return stream

    first = run(prompt[:cut], checkpoints_at=(cut,))
    prefix, cache = first.history_checkpoints[0]
    assert prefix == prompt[:kept]                                 # the checkpoint moves to a chunk start
    path = save_snapshot(tmp_path, "glm-test", prefix, cache)
    got_tokens, stored = load_snapshot(path, "glm-test")
    assert got_tokens == prefix
    fresh = run(prompt).emitted
    assert run(prompt, cache=LaneEngine.copy_single_cache(cache), cached_tokens=kept).emitted == fresh
    assert run(prompt, cache=LaneEngine.copy_single_cache(stored), cached_tokens=kept).emitted == fresh


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("length", [20, 33, 70, 131, 200])
def test_prefill_resumed_at_every_grid_point_has_a_fresh_prefills_bits(checkpoint, monkeypatch, device, length):
    """Chunks on a 32-row grid (sub-chunks of 12 queries): a prompt resumed at any grid point gets fresh bits."""

    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.families.glm5_next import mla

    if device == "gpu":
        if not mx.metal.is_available():
            pytest.skip("needs Metal")
        mx.set_default_device(mx.gpu)
    monkeypatch.setattr(mla, "PREFILL_QUERIES", 12)
    model, prompt, grid = backbone(checkpoint), tokens(length, seed=7), 32

    def feed(ids, cache):
        for b in range(0, len(ids), grid):
            out = model.hidden(mx.array([ids[b:b + grid]], dtype=mx.uint32), cache)
            mx.eval(out)
        return np.array(model.head(out[:, -1:]).astype(mx.float32))

    fresh = feed(prompt, model.make_cache())
    for at in range(grid, length, grid):
        cache = model.make_cache()
        feed(prompt[:at], cache)
        assert np.array_equal(feed(prompt[at:], LaneEngine.copy_single_cache(cache)), fresh), at


def test_dense_prompt_chunks_attend_their_causal_prefix_as_decode_does(checkpoint, monkeypatch):
    """Chunks inside the indexer's reach take one causal attention over the latent, close to decode's dense rows."""

    from dataclasses import replace

    from tensorfold.families.glm5_next import config as C
    from tensorfold.families.glm5_next.mla import MLA

    model = backbone(checkpoint)
    attn = next(layer.attn for layer in model.layers if isinstance(layer.attn, MLA))
    monkeypatch.setattr(attn, "cfg", replace(attn.cfg, index_topk=64))           # 48 positions: all dense
    mx.random.seed(4)
    x = (0.5 * mx.random.normal((48, TEXT["hidden_size"]))).astype(mx.bfloat16)

    def run(chunk):
        cache, outs = caches.MLACache(), []
        for s in range(0, 48, chunk):
            part = x[s:s + chunk]
            outs.append(attn(part, [cache], (int(part.shape[0]),), int(part.shape[0]) <= C.DECODE_ROWS))
        return np.array(mx.concatenate(outs).astype(mx.float32))

    prompt, steps = run(24), run(1)
    assert np.abs(prompt - steps).max() < 0.05 * np.abs(steps).max()


def test_prompt_attention_matches_the_decode_path(checkpoint):
    """Prompt chunks attend over the keys each query's decode step would choose (bits aside: the paths differ)."""

    from tensorfold.families.glm5_next import config as C
    from tensorfold.families.glm5_next.mla import MLA

    model = backbone(checkpoint)
    attn = next(layer.attn for layer in model.layers if isinstance(layer.attn, MLA))
    mx.random.seed(3)
    x = (0.5 * mx.random.normal((200, TEXT["hidden_size"]))).astype(mx.bfloat16)

    def run(chunk):
        cache, outs = caches.MLACache(), []
        for s in range(0, 200, chunk):
            part = x[s:s + chunk]
            outs.append(attn(part, [cache], (int(part.shape[0]),), int(part.shape[0]) <= C.DECODE_ROWS))
        return np.array(mx.concatenate(outs).astype(mx.float32))

    prompt, steps = run(40), run(1)
    assert np.abs(prompt - steps).max() < 0.1 * np.abs(steps).max()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_shared_forward_gives_each_stream_its_own_bits(checkpoint, device):
    """Streams of different lengths in one forward, sparse attention on, equal their own calls, also after a keep."""

    if device == "gpu":
        if not mx.metal.is_available():
            pytest.skip("needs Metal")
        mx.set_default_device(mx.gpu)
    runtime = GLMFlash(backbone(checkpoint), check=True)
    assert runtime.multi_row_exact, runtime.check_report
    assert runtime.check_streams() and runtime.max_streams > 1 and runtime.batch_rows == runtime.exact_width


def _run_streams(runtime, specs, together):
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.glm5_next import engine_settings

    def stream(i, spec):
        prompt, n, sampling, drafts = spec
        return LaneStream(stream_id=f"s{i}", prompt_ids=list(prompt), max_new_tokens=n, sampling=sampling,
                          drafts=drafts)

    if not together:
        out = []
        for i, spec in enumerate(specs):
            engine = LaneEngine(runtime, **engine_settings(runtime))
            s = stream(i, spec)
            engine.add_stream(s)
            while engine.active_count:
                engine.step()
            out.append(s.emitted)
        return out, None
    engine = LaneEngine(runtime, **engine_settings(runtime))
    streams = [stream(i, spec) for i, spec in enumerate(specs)]
    for s in streams:
        engine.add_stream(s)
    while engine.active_count:
        engine.step()
    return [s.emitted for s in streams], engine


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_concurrent_streams_emit_what_they_emit_alone(checkpoint, device):
    """Streams sharing rounds (drafted and serial, greedy and sampled) each emit exactly their tokens alone."""

    from tensorfold.engine.exact_sampling import Sampling

    if device == "gpu":
        if not mx.metal.is_available():
            pytest.skip("needs Metal")
        mx.set_default_device(mx.gpu)
    model = backbone(checkpoint)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.max_streams > 1
    if device == "cpu":
        # MLX's CPU rms_norm (fp32) gives a row other bits once a call holds 8 rows or more: CPU rounds stay under 8
        runtime.exact_width = runtime.batch_rows = min(runtime.exact_width, 7)
    specs = [
        (tokens(21, seed=4), 20, None, True),
        (tokens(9, seed=5), 14, Sampling(seed=3, temperature=0.8, top_k=40, top_p=0.9), True),
        (tokens(33, seed=6), 17, None, False),
        (tokens(14, seed=7), 11, Sampling(seed=9), True),
    ]
    alone, _ = _run_streams(runtime, specs, together=False)
    shared, engine = _run_streams(runtime, specs, together=True)
    assert shared == alone
    assert engine._shared_rounds > 0 and engine.drafted > 0
    assert any(r.streams > 1 for r in engine.round_stats)


def test_qmv_rows_gives_mlx_one_row_bits():
    """On Metal: every row of a qmv_rows call equals MLX's one-row quantized matmul."""

    from tensorfold.kernels.glm.flash.v1 import kernels

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    mx.set_default_device(mx.gpu)
    assert kernels.metal()
    mx.random.seed(3)
    w = linear.Q(*mx.quantize((0.05 * mx.random.normal((256, 1024))).astype(mx.bfloat16), group_size=64, bits=4),
                 bits=4, group=64)
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
    """The real checkpoint's first six layers: 2/3/4/8-row windows get one-row bits and drafts change speed only."""

    import os

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    mx.set_default_device(mx.gpu)
    path = os.environ["TF_GLM5_MODEL"]
    model = weights.load_backbone(path, layers=6)
    assert ["kda" if layer.is_linear else "mla" for layer in model.layers] == ["kda"] * 3 + ["mla", "kda", "kda"]
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.multi_row_exact, runtime.check_report
    prompt = [int(t) for t in np.random.default_rng(7).integers(1000, 100_000, size=40)]
    engine_a, a = _run_engine(runtime, prompt, 16)
    _, b = _run_engine(GLMFlash(model, None, drafts=0), prompt, 16)
    assert engine_a.drafted > 0 and a.emitted == b.emitted


def test_bf16_abliterated_output_projections_keep_prefill_and_mtp_working(tmp_path):
    """A TensorFold derivative keeps quantized inputs/experts but stores attention outputs, including MTP, in BF16."""
    import json
    from tensorfold.families import glm5_next

    folder = write_checkpoint(tmp_path / 'bf16-output')
    index = json.loads((folder / 'model.safetensors.index.json').read_text())['weight_map']
    config = json.loads((folder / 'config.json').read_text())
    for layer in (0, 3, TEXT['num_hidden_layers']):
        prefix = f'model.language_model.layers.{layer}.self_attn.o_proj'
        parts = {}
        shards = {index[f'{prefix}.{suffix}'] for suffix in ('weight', 'scales', 'biases')}
        for shard in shards:
            parts.update(mx.load(str(folder / shard)))
        dense = mx.dequantize(parts[prefix + '.weight'], parts[prefix + '.scales'],
                              parts[prefix + '.biases'], bits=4, group_size=64).astype(mx.bfloat16)
        for shard in shards:
            tensors = mx.load(str(folder / shard))
            for suffix in ('scales', 'biases'):
                key = f'{prefix}.{suffix}'
                tensors.pop(key, None)
                index.pop(key, None)
            if prefix + '.weight' in tensors:
                tensors[prefix + '.weight'] = dense
            mx.eval(tensors)
            staged = folder / (shard + '.new.safetensors')
            mx.save_safetensors(str(staged), tensors)
            staged.replace(folder / shard)
        config['quantization'][prefix] = False
    (folder / 'config.json').write_text(json.dumps(config))
    (folder / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
    glm5_next.check(folder)
    model = backbone(folder)
    head = glm_mtp.load(model)
    assert isinstance(model.layers[0].attn.o_proj, linear.Dense)
    assert isinstance(model.layers[3].attn.o_proj, linear.Dense)
    assert isinstance(head.layer.attn.o_proj, linear.Dense)
    ids = tokens(9)
    a = model.head(model.hidden(mx.array([ids]), model.make_cache()))[0, -1]
    cache = model.make_cache()
    for token in ids:
        h = model.hidden(mx.array([[token]]), cache)
    b = model.head(h)[0, -1]
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert int(a.argmax()) == int(b.argmax())
    assert np.max(np.abs(a - b)) < 0.05 * np.max(np.abs(b)) + 0.05
    drafted = head(model, h.reshape(-1, TEXT['hidden_size']), mx.array([ids[-1]]),
                   [head.make_cache()], (1,), True)
    assert bool(mx.all(mx.isfinite(head.logits(model, drafted))).item())


def test_unquantized_inputs_still_rejected(tmp_path):
    from tensorfold.families import glm5_next
    folder = write_checkpoint(tmp_path / 'unsupported', stated={
        'model.language_model.layers.0.self_attn.q_proj': False})
    with pytest.raises(ValueError, match='module'):
        glm5_next.check(folder)
