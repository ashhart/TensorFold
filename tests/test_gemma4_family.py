"""Gemma 4 family on a tiny random config (CPU): the hidden/head split is mlx_lm's forward, sliding-window caches
decode past the window, and the serial engine resumes from a checkpoint like a fresh prefill."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
gemma4_text = pytest.importorskip("mlx_lm.models.gemma4_text")

from tensorfold.families.gemma4.model import Gemma4  # noqa: E402

TEXT = {
    "model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 32,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
    "num_global_key_value_heads": 1, "attention_k_eq_v": True, "vocab_size": 97, "vocab_size_per_layer_input": 97,
    "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "sliding_window": 8,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "enable_moe_block": True, "num_experts": 4, "top_k_experts": 2, "moe_intermediate_size": 16,
    "use_double_wide_mlp": False, "final_logit_softcapping": 30.0, "tie_word_embeddings": True,
}


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny(seed: int = 0) -> Gemma4:
    mx.random.seed(seed)
    model = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(TEXT))
    mx.eval(model.parameters())
    return Gemma4(model)


def prompt_of(n: int, seed: int = 3) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(6, 97, size=n)]


def test_split_equals_mlx_lm_forward():
    model = tiny()
    ids = mx.array([prompt_of(20)], dtype=mx.uint32)
    reference = model.text(ids, cache=model.text.make_cache())
    split = model.head(model.hidden(ids, model.make_cache()))
    assert bool(mx.array_equal(reference, split).item())


def test_forward_runs_on_another_thread():
    """The server loads the model on the main thread and decodes on its engine thread."""

    import threading

    if not mx.metal.is_available():
        pytest.skip("the failure needs a GPU stream (CPU streams are shared across threads)")
    mx.set_default_device(mx.gpu)                  # the _cpu fixture restores the device afterwards
    model = tiny()
    errors: list[BaseException] = []

    def work():
        try:
            mx.eval(model(mx.array([prompt_of(6)], dtype=mx.uint32), model.make_cache()))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert not errors, errors


@pytest.mark.parametrize("length", [5, 20])      # 20 > sliding_window: the rotating caches wrap
def test_whole_prompt_matches_token_by_token(length):
    model = tiny()
    tokens = prompt_of(length)
    whole = model(mx.array([tokens], dtype=mx.uint32), model.make_cache())
    cache = model.make_cache()
    steps = mx.concatenate([model(mx.array([[t]], dtype=mx.uint32), cache) for t in tokens], axis=1)
    assert np.allclose(np.array(whole), np.array(steps), atol=1e-4)


@pytest.mark.parametrize("pipeline", ["0", "1"])
def test_serial_engine_resumes_from_a_checkpoint(monkeypatch, pipeline):
    monkeypatch.setenv("TF_GEMMA4_PIPELINE", pipeline)
    from tensorfold.engine.family_engine import SerialEngine
    from tensorfold.engine.lane_engine import LaneStream

    model = tiny()
    assert model.gpu_tokens == (pipeline == "1")
    prompt = prompt_of(18)

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
    follow = [*tokens, *whole.context[len(tokens):], 7, 8]
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
    assert split.emitted == whole.emitted


def test_pipelined_and_synchronous_decode_agree(monkeypatch):
    from tensorfold.engine.family_engine import SerialEngine
    from tensorfold.engine.lane_engine import LaneStream

    out = {}
    for pipeline in ("0", "1"):
        monkeypatch.setenv("TF_GEMMA4_PIPELINE", pipeline)
        engine = SerialEngine(tiny())
        stream = LaneStream(stream_id="s", prompt_ids=prompt_of(18), max_new_tokens=16)
        engine.add_stream(stream)
        while engine.active_count:
            engine.step()
        out[pipeline] = list(stream.emitted)
    assert out["0"] == out["1"]
