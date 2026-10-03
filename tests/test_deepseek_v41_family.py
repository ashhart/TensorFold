"""The DeepSeek-V4.1-Flash family on a tiny checkpoint in oMLX's converted layout."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from dsv41_fakes import TEXT, TOKEN_MAP, VOCAB, write_checkpoint  # noqa: E402
from tensorfold.families.deepseek_v41 import weights  # noqa: E402
from tensorfold.families.deepseek_v41.runtime import DeepSeekV41Flash  # noqa: E402

ROWS = 9          # wider than the fake's window of 8: every row of a window still gets its one-row bits


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
        return write_checkpoint(tmp_path_factory.mktemp("dsv41"))
    finally:
        mx.set_default_device(previous)


@pytest.fixture(scope="module")
def model(checkpoint):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return weights.load_backbone(checkpoint, token_map=TOKEN_MAP)
    finally:
        mx.set_default_device(previous)


def tokens(n: int, seed: int = 1) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(3, VOCAB, size=n)]


def logits_of(model, ids, cache):
    return model.head(model.hidden(mx.array([ids], dtype=mx.uint32), cache))[0]


# -- the package and the layout ---------------------------------------------------------------
def test_family_is_detected_and_checked(checkpoint):
    from tensorfold import families
    from tensorfold.families import deepseek_v41

    family = families.detect(checkpoint)
    assert family.module == "tensorfold.families.deepseek_v41" and family.lanes
    deepseek_v41.check(checkpoint)
    families.require_readable(family, json.loads((checkpoint / "config.json").read_text()), "mlx")


def test_check_refuses_deepseeks_fp8_release(checkpoint, tmp_path):
    from tensorfold.families import deepseek_v41

    config = json.loads((checkpoint / "config.json").read_text())
    del config["omlx_deepseek_v41"]
    config["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [32, 32]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="oMLX's converted layout"):
        deepseek_v41.check(tmp_path)


def test_check_refuses_an_unread_format(checkpoint, tmp_path):
    from tensorfold.families import deepseek_v41

    config = json.loads((checkpoint / "config.json").read_text())
    config["omlx_deepseek_v41"]["quantized_modules"]["language_model.layers.2.attn.wq_a"] = {"bits": 4,
                                                                                             "mode": "nvfp4"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="layers.2.attn.wq_a"):
        deepseek_v41.check(tmp_path)


def test_the_layers_follow_the_sources(model):
    cfg = model.args
    assert [cfg.kv_source(i) for i in range(6)] == [None, None, 2, 2, 4, 4]
    assert [cfg.index_source(i) for i in range(6)] == [None, None, 2, 2, 4, 5]
    assert [b.attn.compressor is not None for b in model.layers] == [False, False, True, False, True, False]
    assert [b.attn.indexer is not None for b in model.layers] == [False, False, True, False, True, True]
    assert [b.attn.candidates for b in model.layers] == [False] * 5 + [True]
    assert [b.engram is not None for b in model.layers] == [False, True, False, True, False, False]


def test_weight_bytes_leave_the_engram_tables_out(checkpoint):
    from tensorfold.families import deepseek_v41
    from tensorfold.families.deepseek_v41.weights import engram_bytes

    total = sum(p.stat().st_size for p in checkpoint.glob("*.safetensors"))
    assert 0 < engram_bytes(checkpoint) < total
    assert deepseek_v41.weight_bytes(checkpoint) == total - engram_bytes(checkpoint)


# -- the formats --------------------------------------------------------------------------------
def test_fp8_and_fp4_round_trips_are_idempotent_and_row_local():
    from tensorfold.families.deepseek_v41.quant import quantize_activation

    x = (mx.random.normal((6, 128)) * mx.array([1e-3, 1.0, 50.0, 3e2, 1e-30, 7.0])[:, None]).astype(mx.float32)
    for bits, group, e4m3 in ((8, 32, False), (4, 32, False), (4, 16, True)):
        q = quantize_activation(x, bits, group, e4m3)
        assert mx.array_equal(quantize_activation(q, bits, group, e4m3), q).item()
        rows = mx.concatenate([quantize_activation(x[r:r + 1], bits, group, e4m3) for r in range(6)])
        assert mx.array_equal(rows, q).item()


def test_engram_hashes_continue_across_calls(model):
    """A stream's lookups computed in one call equal those of the same tokens split over calls (history ring)."""

    ids = np.array(tokens(30, seed=3))
    whole = model.make_cache()
    one = model._lookups(ids, whole[0], 0)
    parts = model.make_cache()
    got = [model._lookups(ids[a:b], parts[0], a) for a, b in ((0, 1), (1, 2), (2, 11), (11, 30))]
    assert np.array_equal(np.concatenate(got), one)
    assert one.shape == (30, 2, 4) and one.min() >= 0
    rows = model.args.engram_num_embeddings
    assert all(one[:, k].max() < rows[k] for k in range(2))


# -- forward paths -----------------------------------------------------------------------------
def agree(a, b):
    """Two paths' logits [L, V] over a prompt are the same function to rounding on most rows."""

    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    top = (a.argmax(-1) == b.argmax(-1)).mean()
    rel = np.median(np.abs(a - b).max(-1) / np.abs(b).max(-1))
    return top >= 0.8 and rel < 0.05


@pytest.mark.parametrize("length", [5, 23, 70])     # inside the window; past top-k; past the candidate blocks
def test_prefill_path_agrees_with_decode_path(model, length):
    ids = tokens(length)
    a = logits_of(model, ids, model.make_cache())
    step = model.make_cache()
    b = mx.stack([logits_of(model, [t], step)[-1] for t in ids])
    assert agree(a, b)
    assert all(c.offset == length for c in step)


def test_chunked_prefill_agrees_with_one_chunk(model):
    ids = tokens(61, seed=3)
    a = logits_of(model, ids, model.make_cache())
    parts = model.make_cache()
    b = mx.concatenate([logits_of(model, ids[lo:hi], parts) for lo, hi in ((0, 17), (17, 32), (32, 61))])
    assert agree(a, b)


def serial_logits(model, base, window):
    from tensorfold.engine.lane_engine import LaneEngine

    cache = LaneEngine.copy_single_cache(base)
    return [logits_of(model, [t], cache)[-1] for t in window], cache


@pytest.mark.parametrize("prompt", [0, 3, 6, 15, 40])   # empty, partial blocks, the window boundary, the candidates
def test_decode_windows_give_one_row_bits(model, prompt):
    from tensorfold.engine.lane_engine import LaneEngine

    base = model.make_cache()
    if prompt:
        mx.eval(model.hidden(mx.array([tokens(prompt, seed=5)], dtype=mx.uint32), base))
    window = tokens(ROWS, seed=6)
    serial, _ = serial_logits(model, base, window)
    joint = logits_of(model, window, LaneEngine.copy_single_cache(base))
    for i in range(ROWS):
        assert mx.array_equal(joint[i], serial[i]).item(), f"row {i}"


@pytest.mark.parametrize("keep", [1, 2, 5])
def test_keep_rows_then_continue_equals_serial(model, keep):
    """Rejected rows leave nothing behind (half blocks, pools, the token ring), then decoding continues exactly."""

    from tensorfold.engine.lane_engine import LaneEngine

    base = model.make_cache()
    mx.eval(model.hidden(mx.array([tokens(29, seed=7)], dtype=mx.uint32), base))
    window, after = tokens(ROWS, seed=8), tokens(ROWS, seed=9)
    drafted = LaneEngine.copy_single_cache(base)
    mx.eval(logits_of(model, window, drafted))
    model.keep_rows(drafted, ROWS, keep)
    joint = logits_of(model, after, drafted)
    serial, _ = serial_logits(model, base, window[:keep] + after)
    for i in range(ROWS):
        assert mx.array_equal(joint[i], serial[keep + i]).item(), f"row {i}"


def test_streams_in_one_forward_equal_their_own(model):
    from tensorfold.engine.lane_engine import LaneEngine

    bases = []
    for n, seed in ((7, 1), (20, 2), (33, 3)):
        c = model.make_cache()
        mx.eval(model.hidden(mx.array([tokens(n, seed=seed)], dtype=mx.uint32), c))
        bases.append(c)
    wins = [tokens(3, 11), tokens(1, 12), tokens(4, 13)]
    alone = [logits_of(model, w, LaneEngine.copy_single_cache(b)) for w, b in zip(wins, bases)]
    flat = mx.array([t for w in wins for t in w], dtype=mx.uint32)
    joint = model.head(model.hidden_rows(flat, [LaneEngine.copy_single_cache(b) for b in bases],
                                         [len(w) for w in wins]))[0]
    at = 0
    for w, a in zip(wins, alone):
        assert mx.array_equal(joint[at:at + len(w)], a).item()
        at += len(w)


def test_runtime_checks_windows_and_streams(model):
    runtime = DeepSeekV41Flash(model, None, drafts=0, check=False)
    width, costs = runtime.check_windows(widest=ROWS)
    assert width == ROWS and set(costs) == set(range(1, ROWS + 1))
    runtime.exact_width = width
    assert runtime.check_streams()


def cpu_runtime(model):
    runtime = DeepSeekV41Flash(model, None, drafts=0, check=False)
    runtime.exact_width = runtime.batch_rows = ROWS
    runtime.multi_row_exact = True
    runtime.max_streams = ROWS
    return runtime


def test_concurrent_streams_emit_what_they_emit_alone(model):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.deepseek_v41 import engine_settings

    runtime = cpu_runtime(model)
    specs = [(tokens(21, seed=4), 12, None), (tokens(9, seed=5), 10, Sampling(seed=3, temperature=0.8)),
             (tokens(33, seed=6), 11, None)]

    def streams():
        return [LaneStream(stream_id=f"s{i}", prompt_ids=list(p), max_new_tokens=n, sampling=smp)
                for i, (p, n, smp) in enumerate(specs)]

    alone = []
    for s in streams():
        engine = LaneEngine(runtime, **engine_settings(runtime))
        engine.add_stream(s)
        while engine.active_count:
            engine.step()
        alone.append(s.emitted)
    engine = LaneEngine(runtime, **engine_settings(runtime))
    together = streams()
    for s in together:
        engine.add_stream(s)
    while engine.active_count:
        engine.step()
    assert [s.emitted for s in together] == alone
    assert engine._shared_rounds > 0


@pytest.mark.parametrize(("grid", "length", "cut", "kept"), [(8, 30, 26, 24), (16, 50, 45, 32)])
def test_lane_engine_resumes_from_a_chunk_start(model, tmp_path, grid, length, cut, kept):
    """A checkpoint at a chunk start, in memory or read back from disk, resumes exactly like a fresh prefill."""

    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    runtime = cpu_runtime(model)
    prompt = tokens(length, seed=5)

    def run(ids, **kw):
        engine = LaneEngine(runtime)
        engine.prefill_plan = PrefillPlan(grid)
        stream = LaneStream(stream_id="x", prompt_ids=list(ids), max_new_tokens=8)
        engine.add_stream(stream, **kw)
        while engine.active_count:
            engine.step()
        return stream

    first = run(prompt[:cut], checkpoints_at=(cut,))
    prefix, cache = first.history_checkpoints[0]
    assert prefix == prompt[:kept]
    path = save_snapshot(tmp_path, "dsv41-test", prefix, cache)
    got_tokens, stored = load_snapshot(path, "dsv41-test")
    assert got_tokens == prefix
    fresh = run(prompt).emitted
    assert run(prompt, cache=LaneEngine.copy_single_cache(cache), cached_tokens=kept).emitted == fresh
    assert run(prompt, cache=LaneEngine.copy_single_cache(stored), cached_tokens=kept).emitted == fresh


# -- oMLX's model as the reference ---------------------------------------------------------------
def _reference(checkpoint):
    import dsv41_omlx

    if not dsv41_omlx.available() or not mx.metal.is_available():
        pytest.skip("needs an oMLX checkout on OMLX_SRC and Metal")
    mx.set_default_device(mx.gpu)
    return dsv41_omlx, dsv41_omlx.load_reference(checkpoint, TOKEN_MAP), weights.load_backbone(checkpoint,
                                                                                            token_map=TOKEN_MAP)


@pytest.mark.parametrize(("length", "seed"), [(40, 1), (150, 2), (250, 5)])
def test_prompt_logits_match_omlx(checkpoint, length, seed):
    """Teacher-forced logits equal oMLX's own model's on the same weights (Metal), chunks below 256 rows."""

    ref_mod, ref, model = _reference(checkpoint)
    ids = tokens(length, seed=seed)
    a = np.array(ref_mod.reference_logits(ref, ids))
    b = np.array(logits_of(model, ids, model.make_cache()))
    assert (a.argmax(-1) == b.argmax(-1)).all()
    assert np.abs(a - b).max() <= 1e-5 * np.abs(a).max()


def test_decode_steps_match_omlx(checkpoint):
    """A 37-token prompt, then 40 one-token steps through both engines' caches: the same logits every step."""

    _, ref, model = _reference(checkpoint)
    ids = tokens(77, seed=4)
    rc, mc = ref.make_cache(), model.make_cache()
    a = [ref(mx.array([ids[:37]], dtype=mx.int32), cache=rc)[0, -1]]
    b = [logits_of(model, ids[:37], mc)[-1]]
    for t in ids[37:]:
        a.append(ref(mx.array([[t]], dtype=mx.int32), cache=rc)[0, -1])
        b.append(logits_of(model, [t], mc)[-1])
    a, b = np.array(mx.stack(a)), np.array(mx.stack(b))
    assert (a.argmax(-1) == b.argmax(-1)).all()
    assert np.abs(a - b).max() <= 1e-5 * np.abs(a).max()


def test_config_reads_the_released_text_config():
    from tensorfold.families.deepseek_v41.config import Config

    cfg = Config.from_dict({"model_type": "deepseek_v41", "text_config": {**TEXT}})
    assert cfg.num_hidden_layers == 6 and cfg.compress_ratios[:6] == [0, 0, 2, 2, 1, 1]
    assert cfg.rms_norm_eps == 1e-20 and cfg.dspark_n_routed_experts == 4


# -- the checkpoint's DSpark drafter ---------------------------------------------------------------
@pytest.fixture(scope="module")
def drafter(model):
    from tensorfold.families.deepseek_v41 import dspark

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return dspark.load(model, model.weights)
    finally:
        mx.set_default_device(previous)


def dspark_runtime(model, drafter):
    from tensorfold.families.deepseek_v41.runtime import DSparkV41Flash

    runtime = DSparkV41Flash(model, drafter, check=False)
    runtime.exact_width = runtime.batch_rows = ROWS
    runtime.multi_row_exact = True
    runtime.max_streams = ROWS
    runtime.dspark = runtime.mtp = drafter
    return runtime


def test_dspark_drafts_a_block(model, drafter):
    rings = drafter.make_cache()
    drafter.absorb(mx.zeros((3, len(drafter.taps) * TEXT["hidden_size"]), dtype=mx.bfloat16), rings)
    token = mx.array([7], dtype=mx.uint32)
    drafts = drafter.draw(drafter.logits(model, token, rings), token, drafter.size, lambda row, j: mx.argmax(row, -1))
    assert drafts.shape == (TEXT["dspark_block_size"],) and rings[0].offset == 3
    assert drafter.taps == tuple(TEXT["dspark_target_layer_ids"])
    assert [b.block.moe.top for b in drafter.blocks] == [TEXT["dspark_num_experts_per_tok"]] * 3


def test_dspark_drafts_change_speed_only(model, drafter):
    """Drafted replies equal the serial run's, greedy and sampled."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.deepseek_v41 import engine_settings

    runtime, serial = dspark_runtime(model, drafter), cpu_runtime(model)
    try:
        for sampling in (None, Sampling(seed=5, temperature=0.9)):
            got = []
            for rt, drafts in ((runtime, True), (serial, False)):
                engine = LaneEngine(rt, **engine_settings(rt))
                stream = LaneStream(stream_id="s", prompt_ids=tokens(37, seed=11), max_new_tokens=24,
                                    sampling=sampling, drafts=drafts)
                engine.add_stream(stream)
                while engine.active_count:
                    engine.step()
                got.append(stream.emitted)
                if drafts:
                    assert engine.family_mtp and engine.drafted > 0
            assert got[0] == got[1]
    finally:
        model.tap_layers = ()


def test_dspark_concurrent_streams_emit_what_they_emit_alone(model, drafter):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.deepseek_v41 import engine_settings

    runtime = dspark_runtime(model, drafter)
    specs = [(tokens(21, seed=4), 14, None, True), (tokens(9, seed=5), 12, Sampling(seed=3, temperature=0.8), True),
             (tokens(33, seed=6), 10, None, False)]

    def streams():
        return [LaneStream(stream_id=f"s{i}", prompt_ids=list(p), max_new_tokens=n, sampling=smp, drafts=d)
                for i, (p, n, smp, d) in enumerate(specs)]

    try:
        alone = []
        for s in streams():
            engine = LaneEngine(runtime, **engine_settings(runtime))
            engine.add_stream(s)
            while engine.active_count:
                engine.step()
            alone.append(s.emitted)
        engine = LaneEngine(runtime, **engine_settings(runtime))
        together = streams()
        for s in together:
            engine.add_stream(s)
        while engine.active_count:
            engine.step()
        assert [s.emitted for s in together] == alone
        assert engine.drafted > 0
    finally:
        model.tap_layers = ()


# -- the same contract on Metal (the row kernels) ------------------------------------------------------
@pytest.fixture(scope="module")
def gpu_model(checkpoint):
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        return weights.load_backbone(checkpoint, token_map=TOKEN_MAP)
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("prompt", [0, 6, 40])
def test_metal_windows_give_one_row_bits(gpu_model, prompt):
    from tensorfold.engine.lane_engine import LaneEngine

    mx.set_default_device(mx.gpu)
    model = gpu_model
    base = model.make_cache()
    if prompt:
        mx.eval(model.hidden(mx.array([tokens(prompt, seed=5)], dtype=mx.uint32), base))
    window = tokens(16, seed=6)
    serial, _ = serial_logits(model, base, window)
    joint = logits_of(model, window, LaneEngine.copy_single_cache(base))
    for i in range(16):
        assert mx.array_equal(joint[i], serial[i]).item(), f"row {i}"


def test_metal_keep_rows_and_streams(gpu_model):
    from tensorfold.engine.lane_engine import LaneEngine

    mx.set_default_device(mx.gpu)
    model = gpu_model
    base = model.make_cache()
    mx.eval(model.hidden(mx.array([tokens(29, seed=7)], dtype=mx.uint32), base))
    window, after = tokens(16, seed=8), tokens(16, seed=9)
    drafted = LaneEngine.copy_single_cache(base)
    mx.eval(logits_of(model, window, drafted))
    model.keep_rows(drafted, 16, 3)
    joint = logits_of(model, after, drafted)
    serial, _ = serial_logits(model, base, window[:3] + after)
    for i in range(16):
        assert mx.array_equal(joint[i], serial[3 + i]).item(), f"row {i}"
    runtime = DeepSeekV41Flash(model, None, drafts=0, check=False)
    width, _ = runtime.check_windows(widest=16)
    assert width == 16
    runtime.exact_width = width
    assert runtime.check_streams()
