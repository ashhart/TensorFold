"""Gemma 4 as a lane family on a tiny random checkpoint (Metal): windows, rollback, streams, resumes and drafts."""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.models.gemma4_text")
if not mx.metal.is_available():
    pytest.skip("the Gemma decode kernels are Metal kernels", allow_module_level=True)

from gemma4_tiny import TINY, KnownReply, tiny_text, tokens  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402
from tensorfold.families.gemma4.model import Gemma4  # noqa: E402
from tensorfold.kernels.gemma.v1.matmul import tensor_units  # noqa: E402

BACKENDS = ["rows", pytest.param("lane", marks=pytest.mark.skipif(not tensor_units(), reason="needs tensor units"))]
copy = LaneEngine.copy_single_cache


@pytest.fixture(scope="module")
def text():
    return tiny_text()


def family(text, backend: str, width: int = 16) -> Gemma4:
    model = Gemma4(text, backend=backend, check=False)
    model.exact_width = width
    return model


def prefilled(model: Gemma4, n: int, seed: int = 3) -> list:
    cache = model.make_cache()
    mx.eval(model.prefill(mx.array([tokens(n, seed)], dtype=mx.uint32), cache))
    return cache


def steps(model: Gemma4, cache: list, window: list[int]) -> list:
    out = []
    for token in window:
        logits = model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache))
        mx.eval(logits)
        out.append(logits[0, -1])
    return out


def same(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


# 150 tokens pass the ring (window 8 + 128 slots): windows and rollbacks write across its end
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("prompt", [20, 150])
def test_windows_give_every_row_its_serial_bits(text, backend, prompt):
    model = family(text, backend)
    base = prefilled(model, prompt)
    window = tokens(14, seed=5)
    serial = steps(model, copy(base), window)
    for width in range(2, len(window) + 1):
        logits = model.head(model.hidden(mx.array([window[:width]], dtype=mx.uint32), copy(base)))
        mx.eval(logits)
        assert all(same(logits[0, i], serial[i]) for i in range(width)), width


@pytest.mark.parametrize("backend", BACKENDS)
def test_rollback_then_steps_equal_serial_decoding(text, backend):
    model = family(text, backend)
    base = prefilled(model, 131)                      # the window's rows straddle the ring's end
    window = tokens(12, seed=5)
    serial = steps(model, copy(base), window)
    work = copy(base)
    mx.eval(model.hidden(mx.array([window[:3] + tokens(6, seed=9)], dtype=mx.uint32), work))
    model.keep_rows(work, 9, 3)
    assert [c.offset for c in work] == [131 + 3] * len(work)
    assert all(same(a, b) for a, b in zip(steps(model, work, window[3:]), serial[3:]))
    # a window kept whole, passed as a chain's path (a one-stream round commits trees this way)
    whole = copy(base)
    mx.eval(model.hidden(mx.array([window[:4]], dtype=mx.uint32), whole))
    model.keep_rows(whole, 4, [0, 1, 2, 3])
    assert same(steps(model, whole, window[4:5])[0], serial[4])


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_shared_forward_gives_each_stream_its_own_bits(text, backend):
    model = family(text, backend)
    assert model.check_streams(None)
    # streams at different sides of the ring's end, each window kept in part
    bases = [prefilled(model, n, seed=n) for n in (20, 133, 140)]
    windows = [tokens(n, seed=11 + n) for n in (3, 5, 1)]
    alone = []
    for base, window in zip(bases, windows):
        logits = model.head(model.hidden(mx.array([window], dtype=mx.uint32), copy(base)))
        mx.eval(logits)
        alone.append(logits[0])
    joint = model.head(model.hidden_rows(windows, [copy(b) for b in bases]))[0]
    mx.eval(joint)
    at = 0
    for window, own in zip(windows, alone):
        assert same(joint[at:at + len(window)], own)
        at += len(window)


def _reply(model: Gemma4, prompt: list[int], n: int, *, sampling=None, proposer=None, cache=None, cached: int = 0,
           engine: LaneEngine | None = None) -> tuple[list[int], LaneEngine]:
    engine = engine or LaneEngine(model, max_rows=16, max_draft=15)
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=n, sampling=sampling,
                        proposer=proposer, drafts=proposer is not None)
    engine.add_stream(stream, cache=cache, cached_tokens=cached)
    engine.run()
    return list(stream.emitted), engine


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sampled", [False, True])
def test_drafted_replies_equal_serial_ones(text, backend, sampled):
    model = family(text, backend)
    prompt = tokens(140, seed=21)
    sampling = Sampling(seed=5, temperature=1.0, top_k=20, top_p=0.95) if sampled else None
    serial, _ = _reply(model, prompt, 40, sampling=sampling)
    drafted, engine = _reply(model, prompt, 40, sampling=sampling, proposer=KnownReply(prompt + serial))
    assert drafted == serial
    assert engine.drafted > 0 and 0 < engine.accepted < engine.drafted     # windows kept whole, in part and not
    assert max(r.rows for r in engine.round_stats) > 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_pipelined_one_token_rounds_equal_synchronous_ones(text, backend):
    model = family(text, backend)
    prompt = tokens(30, seed=4)
    ahead, _ = _reply(model, prompt, 24)
    model.gpu_tokens = False
    try:
        synchronous, _ = _reply(model, prompt, 24)
    finally:
        model.gpu_tokens = True
    assert ahead == synchronous


@pytest.mark.parametrize("backend", BACKENDS)
def test_concurrent_streams_emit_what_they_emit_alone(text, backend):
    model = family(text, backend)
    prompts = [tokens(n, seed=30 + n) for n in (25, 140, 60, 9)]
    samplings = [None, Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95), None,
                 Sampling(seed=8, temperature=0.8, top_k=40, top_p=0.9)]
    alone = [_reply(model, p, 20, sampling=s)[0] for p, s in zip(prompts, samplings)]
    engine = LaneEngine(model, max_rows=16, max_draft=15)
    streams = []
    for i, (p, s, want) in enumerate(zip(prompts, samplings, alone)):
        proposer = KnownReply(p + want, good=(3, 0, 7)) if i % 2 == 0 else None
        stream = LaneStream(stream_id=f"s{i}", prompt_ids=list(p), max_new_tokens=20, sampling=s, proposer=proposer,
                            drafts=proposer is not None)
        engine.add_stream(stream)
        streams.append(stream)
    engine.run()
    assert [list(s.emitted) for s in streams] == alone
    assert any(r.streams > 1 for r in engine.round_stats)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_prompt_resumed_from_a_grid_checkpoint_equals_a_fresh_one(text, backend, monkeypatch, tmp_path):
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    from tensorfold.engine.prefill_plan import PrefillPlan

    monkeypatch.setattr(LaneEngine, "prefill_plan", PrefillPlan(64))          # chunks of 64 from position 0
    model = family(text, backend)
    first = tokens(150, seed=40)
    reply, engine = _reply(model, first, 12)
    engine = LaneEngine(model, max_rows=16, max_draft=15, retain_finished_caches=True)
    stream = LaneStream(stream_id="a", prompt_ids=list(first), max_new_tokens=12)
    engine.add_stream(stream, checkpoints_at=(len(first),))
    engine.run()
    (at, checkpoint), = [(len(t), c) for t, c in stream.history_checkpoints]
    assert at == 128 and not engine.finished_caches            # on the grid only; decoded states are not kept
    follow = first + reply + tokens(20, seed=41)
    fresh, _ = _reply(model, follow, 16)
    resumed, _ = _reply(model, follow, 16, cache=copy(checkpoint), cached=at)
    assert resumed == fresh
    path = save_snapshot(tmp_path, "gemma-test", first[:at], checkpoint)
    loaded_tokens, loaded = load_snapshot(path, "gemma-test")
    assert loaded_tokens == first[:at]
    stored, _ = _reply(model, follow, 16, cache=model.adopt_cache(loaded), cached=at)
    assert stored == fresh


def test_the_ring_is_fixed_memory_and_full_layers_grow():
    from tensorfold.server.memory_budget import CacheMemory

    model = Gemma4(tiny_text(), backend="rows", check=False)
    cache = prefilled(model, 300)
    memory = CacheMemory.from_cache(cache)
    full = sum(1 for kind in TINY["layer_types"] if kind == "full_attention")
    per_position = full * 2 * TINY["num_global_key_value_heads"] * TINY["global_head_dim"] * 2   # keys and values, bf16
    assert memory.bytes_per_token in (per_position, 2 * per_position)       # the spare buffer counted up front or not
    ring = next(c for c in cache if hasattr(c, "ring_keys"))
    assert memory.fixed_bytes >= ring.nbytes


def test_forward_runs_on_another_thread(text):
    """The server loads the model on the main thread and decodes on its engine thread."""

    import threading

    model = family(text, "rows")
    errors: list[BaseException] = []

    def work():
        try:
            cache = model.make_cache()
            mx.eval(model.prefill(mx.array([tokens(6)], dtype=mx.uint32), cache))
            mx.eval(model.head(model.hidden(mx.array([[7]], dtype=mx.uint32), cache)))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert not errors, errors


def test_check_refuses_the_layouts_the_kernels_do_not_cover(tmp_path):
    from tensorfold import families
    from tensorfold.families import gemma4

    config = {"model_type": "gemma4", "text_config": dict(TINY, model_type="gemma4_text"),
              "quantization": {"group_size": 64, "bits": 4, "mode": "affine",
                               "language_model.model.layers.0.router.proj": {"group_size": 64, "bits": 8}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert families.detect(tmp_path).module == "tensorfold.families.gemma4"
    gemma4.check(tmp_path)
    for change in ({"enable_moe_block": False}, {"hidden_size_per_layer_input": 256}, {"num_kv_shared_layers": 2}):
        (tmp_path / "config.json").write_text(json.dumps(dict(config, text_config={**config["text_config"],
                                                                                    **change})))
        with pytest.raises(ValueError, match="MoE checkpoints"):
            gemma4.check(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(dict(config, quantization={"group_size": 64, "bits": 8})))
    with pytest.raises(ValueError, match="4-bit"):
        gemma4.check(tmp_path)
