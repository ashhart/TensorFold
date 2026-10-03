"""The DeepSeek-V4-Flash family on a tiny checkpoint: loading, both paths, exact windows, rollback, streams."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from dsv4_fakes import (DSPARK, TEXT, write_checkpoint, write_dspark, write_mtp, write_official_dspark,  # noqa: E402
                        write_official_mtp)
from tensorfold.families.deepseek_v4 import mtp as ds_mtp  # noqa: E402
from tensorfold.families.deepseek_v4 import weights  # noqa: E402
from tensorfold.families.deepseek_v4.runtime import DeepSeekFlash  # noqa: E402

CPU_ROWS = 7          # MLX's CPU fp32 rms_norm is row-invariant below 8 rows


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
        return write_checkpoint(tmp_path_factory.mktemp("dsv4"))
    finally:
        mx.set_default_device(previous)


@pytest.fixture(scope="module")
def model(checkpoint):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return weights.load_backbone(checkpoint)
    finally:
        mx.set_default_device(previous)


def tokens(n: int, seed: int = 1) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(2, TEXT["vocab_size"], size=n)]


def logits_of(model, ids, cache):
    return model.head(model.hidden(mx.array([ids], dtype=mx.uint32), cache))[0]


def test_family_is_detected_and_checked(checkpoint):
    from tensorfold import families
    from tensorfold.families import deepseek_v4

    assert families.detect(checkpoint).module == "tensorfold.families.deepseek_v4"
    deepseek_v4.check(checkpoint)


def test_check_refuses_affine_experts(checkpoint, tmp_path):
    import shutil

    from tensorfold.families import deepseek_v4

    shutil.copy(checkpoint / "config.json", tmp_path / "config.json")
    config = json.loads((tmp_path / "config.json").read_text())
    for key in ("quantization", "quantization_config"):
        config[key]["model.layers.2.ffn.switch_mlp.up_proj"] = {"group_size": 64, "bits": 4}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="switch_mlp.up_proj"):
        deepseek_v4.check(tmp_path)


def test_layers_follow_the_ratios(model):
    ratios = [layer.attn.ratio for layer in model.layers]
    assert ratios == TEXT["compress_ratios"][:TEXT["num_hidden_layers"]]
    assert [layer.attn.indexer is not None for layer in model.layers] == [r == 4 for r in ratios]
    assert model.layers[0].moe.table is not None and model.layers[1].moe.table is None


@pytest.mark.parametrize("length", [6, 40, 300])   # the window, top-k past 4 pooled rows, a ratio-128 block
def test_prefill_path_agrees_with_decode_path(model, length):
    """One prompt chunk and one-row steps compute the same function (to rounding)."""

    ids = tokens(length)
    whole = model.make_cache()
    a = logits_of(model, ids, whole)[-1]
    step = model.make_cache()
    for t in ids:
        b = logits_of(model, [t], step)[-1]
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert int(a.argmax()) == int(b.argmax())
    assert np.max(np.abs(a - b)) < 0.05 * np.max(np.abs(b)) + 0.05
    for c1, c2 in zip(whole, step):
        assert c1.offset == c2.offset == length and c1.pool_rows == c2.pool_rows


def test_chunked_prefill_agrees_with_one_chunk(model):
    ids = tokens(90, seed=3)
    one = model.make_cache()
    a = logits_of(model, ids, one)[-1]
    parts = model.make_cache()
    for lo, hi in ((0, 33), (33, 61), (61, 90)):
        b = logits_of(model, ids[lo:hi], parts)[-1]
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert int(a.argmax()) == int(b.argmax())
    assert np.max(np.abs(a - b)) < 0.05 * np.max(np.abs(b)) + 0.05


def serial_logits(model, base, window):
    from tensorfold.engine.lane_engine import LaneEngine

    cache = LaneEngine.copy_single_cache(base)
    out = []
    for t in window:
        out.append(logits_of(model, [t], cache)[-1])
    return out, cache


@pytest.mark.parametrize("prompt", [5, 38, 124])   # a ratio-128 block completes in the last window
def test_decode_windows_give_one_row_bits(model, prompt):
    """A window of up to 7 rows gives each row its one-row logits bit for bit, across pool emissions."""

    from tensorfold.engine.lane_engine import LaneEngine

    base = model.make_cache()
    mx.eval(model.hidden(mx.array([tokens(prompt, seed=5)], dtype=mx.uint32), base))
    window = tokens(CPU_ROWS, seed=6)
    serial, _ = serial_logits(model, base, window)
    joint = logits_of(model, window, LaneEngine.copy_single_cache(base))
    for i in range(CPU_ROWS):
        assert mx.array_equal(joint[i], serial[i]).item(), f"row {i}"


def test_keep_rows_then_continue_equals_serial(model):
    """Rejected rows leave nothing behind: keep 2 of 7, continue, and match the serial run of the kept tokens."""

    from tensorfold.engine.lane_engine import LaneEngine

    base = model.make_cache()
    mx.eval(model.hidden(mx.array([tokens(29, seed=7)], dtype=mx.uint32), base))
    window, after = tokens(CPU_ROWS, seed=8), tokens(CPU_ROWS, seed=9)
    drafted = LaneEngine.copy_single_cache(base)
    mx.eval(logits_of(model, window, drafted))
    model.keep_rows(drafted, CPU_ROWS, 2)
    joint = logits_of(model, after, drafted)
    serial, _ = serial_logits(model, base, window[:2] + after)
    for i in range(CPU_ROWS):
        assert mx.array_equal(joint[i], serial[2 + i]).item(), f"row {i}"


def test_runtime_checks_windows_and_streams(model):
    runtime = DeepSeekFlash(model, None, drafts=0, check=False)
    width, costs = runtime.check_windows(widest=CPU_ROWS)
    assert width == CPU_ROWS and set(costs) == set(range(1, CPU_ROWS + 1))
    runtime.exact_width = width
    assert runtime.check_streams()


@pytest.fixture(scope="module")
def drafter(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return write_mtp(tmp_path_factory.mktemp("dsv4-mtp"))
    finally:
        mx.set_default_device(previous)


def test_official_mtp_shard_converts_to_the_mlx_layout(tmp_path):
    """FP8 linears become 4-bit affine rows of their exact bf16 values; the FP4 experts keep their bytes."""

    from tensorfold.families.deepseek_v4.convert import convert_mtp, e8m0

    raw = write_official_mtp(tmp_path / "official.safetensors")
    folder = convert_mtp(tmp_path / "official.safetensors", tmp_path / "mtp")
    assert json.loads((folder / "config.json").read_text()) == {"model_type": "deepseek_v4_mtp"}
    out = mx.load(str(folder / "model.safetensors"))
    w = mx.from_fp8(raw["mtp.0.e_proj.weight"], dtype=mx.float32) * e8m0(raw["mtp.0.e_proj.scale"])[0, 0]
    got = mx.dequantize(out["mtp.e_proj.weight"], out["mtp.e_proj.scales"], out["mtp.e_proj.biases"], group_size=64,
                        bits=4)
    assert float(mx.abs(got - w).max().item()) <= float(mx.abs(w).max().item()) / 7
    codes = out["mtp.ffn.switch_mlp.up_proj.weight"]
    assert codes.dtype == mx.uint32 and codes.shape == (TEXT["n_routed_experts"], TEXT["moe_intermediate_size"],
                                                         TEXT["hidden_size"] // 8)
    assert mx.array_equal(codes[3].view(mx.uint8), raw["mtp.0.ffn.experts.3.w3.weight"]).item()
    assert mx.array_equal(out["mtp.ffn.gate.e_score_correction_bias"], raw["mtp.0.ffn.gate.bias"]).item()


def _run_engine(runtime, prompt, n, drafts=True):
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.deepseek_v4 import engine_settings

    engine = LaneEngine(runtime, **engine_settings(runtime))
    assert engine.family
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=n, drafts=drafts)
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    return engine, stream


def cpu_runtime(model, head, drafts):
    runtime = DeepSeekFlash(model, None, drafts=drafts, check=False)
    runtime.exact_width = runtime.batch_rows = CPU_ROWS
    runtime.multi_row_exact = True
    if head is not None and drafts:
        runtime.mtp = head
    return runtime


def test_mtp_drafts_change_speed_only(model, drafter):
    head = ds_mtp.load(model, drafter / "model.safetensors")
    drafted, serial = cpu_runtime(model, head, 3), cpu_runtime(model, None, 0)
    prompt = tokens(41, seed=4)
    engine_a, a = _run_engine(drafted, prompt, 30)
    engine_b, b = _run_engine(serial, prompt, 30)
    assert engine_a.family_mtp and not engine_b.family_mtp and engine_a.drafted > 0
    assert a.emitted == b.emitted
    _, c = _run_engine(drafted, prompt, 30, drafts=False)
    assert c.emitted == b.emitted


@pytest.mark.parametrize(("grid", "length", "cut", "kept"), [(8, 30, 26, 24), (32, 150, 140, 128)])
def test_lane_engine_resumes_from_a_chunk_start(model, drafter, tmp_path, grid, length, cut, kept):
    """A checkpoint at a chunk start, in memory or read back from disk, resumes exactly like a fresh prefill."""

    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    runtime = cpu_runtime(model, ds_mtp.load(model, drafter / "model.safetensors"), 2)
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
    path = save_snapshot(tmp_path, "dsv4-test", prefix, cache)
    got_tokens, stored = load_snapshot(path, "dsv4-test")
    assert got_tokens == prefix
    fresh = run(prompt).emitted
    assert run(prompt, cache=LaneEngine.copy_single_cache(cache), cached_tokens=kept).emitted == fresh
    assert run(prompt, cache=LaneEngine.copy_single_cache(stored), cached_tokens=kept).emitted == fresh


def test_concurrent_streams_emit_what_they_emit_alone(model, drafter):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.deepseek_v4 import engine_settings

    runtime = cpu_runtime(model, ds_mtp.load(model, drafter / "model.safetensors"), 3)
    runtime.max_streams = CPU_ROWS
    specs = [(tokens(21, seed=4), 16, None, True), (tokens(9, seed=5), 12, Sampling(seed=3, temperature=0.8), True),
             (tokens(33, seed=6), 14, None, False)]

    def streams():
        return [LaneStream(stream_id=f"s{i}", prompt_ids=list(p), max_new_tokens=n, sampling=smp, drafts=d)
                for i, (p, n, smp, d) in enumerate(specs)]

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
    assert engine._shared_rounds > 0 and engine.drafted > 0


def test_dspark_drafts_change_speed_only(model, tmp_path):
    """DSpark drafts a block after each round's read; the reply equals the serial run's, sampled or greedy."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.deepseek_v4 import dspark as ds_dspark
    from tensorfold.families.deepseek_v4 import engine_settings
    from tensorfold.families.deepseek_v4.runtime import DSparkFlash

    folder = write_dspark(tmp_path / "dspark")
    drafter = ds_dspark.load(model, folder / "model.safetensors", json.loads((folder / "config.json").read_text()))
    runtime = DSparkFlash(model, drafter, check=False)
    runtime.exact_width = runtime.batch_rows = CPU_ROWS
    runtime.multi_row_exact = True
    runtime.dspark = runtime.mtp = drafter
    serial = cpu_runtime(model, None, 0)
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


def test_official_dspark_converts_and_drafts(model, tmp_path):
    """DSpark's shards convert block by block, load with its config's fields and draft a block."""

    from tensorfold.families.deepseek_v4 import dspark as ds_dspark
    from tensorfold.families.deepseek_v4.convert import convert_dspark

    folder = convert_dspark(write_official_dspark(tmp_path / "official"), tmp_path / "dspark")
    config = json.loads((folder / "config.json").read_text())
    assert config == {"model_type": "deepseek_v4_dspark", **DSPARK}
    names = set(mx.load(str(folder / "model.safetensors")))
    assert {"dspark.0.main_proj.weight", "dspark.1.markov_head.markov_w2.weight", "dspark.1.hc_head.fn",
            "dspark.0.ffn.switch_mlp.down_proj.scales"} <= names
    drafter = ds_dspark.load(model, folder / "model.safetensors", config)
    rings = drafter.make_cache()
    drafter.absorb(mx.zeros((3, len(drafter.taps) * TEXT["hidden_size"]), dtype=mx.bfloat16), rings)
    token = mx.array([7], dtype=mx.uint32)
    drafts = drafter.draw(drafter.logits(model, token, rings), token, drafter.size, lambda row, j: mx.argmax(row, -1))
    assert drafts.shape == (DSPARK["dspark_block_size"],) and rings[0].offset == 3


def test_a_loaded_runtime_decodes_on_another_thread(model):
    """The server decodes on its own thread: nothing lazy from the loading thread may reach it."""

    import threading

    from tensorfold.engine.family_common import cache_arrays
    from tensorfold.families.deepseek_v4.runtime import materialize

    runtime = DeepSeekFlash(model, None, drafts=0, check=False)
    assert materialize(runtime) > 100
    cache = runtime.make_cache()
    mx.eval(runtime.hidden(mx.array([tokens(5)], dtype=mx.uint32), cache), *cache_arrays(cache))
    errors = []

    def other():
        try:
            mx.set_default_device(mx.cpu)
            mx.eval(runtime.head(runtime.hidden(mx.array([tokens(2, seed=9)], dtype=mx.uint32), cache)))
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert not errors, errors


def test_a_drafter_folder_names_its_head(tmp_path):
    """model.safetensors beside a config.json whose model_type is the head; anything else is refused before loading."""

    from tensorfold.families.deepseek_v4.runtime import drafter_config

    assert drafter_config(write_mtp(tmp_path / "mtp"))["model_type"] == "deepseek_v4_mtp"
    assert drafter_config(write_dspark(tmp_path / "dspark")) == {"model_type": "deepseek_v4_dspark", **DSPARK}
    old = tmp_path / "old"
    old.mkdir()
    (old / "dspark.safetensors").write_bytes(b"")
    (old / "config.json").write_text(json.dumps(DSPARK))
    other = write_mtp(tmp_path / "other")
    (other / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    for folder in (old, other, tmp_path / "missing"):
        with pytest.raises(ValueError, match="deepseek_v4_dspark or deepseek_v4_mtp"):
            drafter_config(folder)


def test_the_published_dspark_head_is_the_default_drafter(tmp_path, monkeypatch, capsys):
    """`--drafter auto` takes the pulled DSpark repo: its snapshot layout counts as complete weights."""

    from tensorfold import cli, families, hub
    from tensorfold.families.deepseek_v4.runtime import drafter_config

    family = families.families()["deepseek_v4"]
    assert family.package.DRAFTER == "TensorFold/DeepSeek-V4-Flash-DSpark-MLX"
    snapshot = write_dspark(tmp_path / "snapshot")
    monkeypatch.setattr(hub, "cached", lambda repo, **kw: snapshot if repo == family.package.DRAFTER else None)
    assert cli._drafter(family, "auto") == str(snapshot)
    assert drafter_config(snapshot)["model_type"] == "deepseek_v4_dspark"
    monkeypatch.setattr(hub, "cached", lambda repo, **kw: None)
    assert cli._drafter(family, "auto") == ""
    assert "tensorfold pull TensorFold/DeepSeek-V4-Flash-DSpark-MLX" in capsys.readouterr().out
