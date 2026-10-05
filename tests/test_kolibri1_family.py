"""Kolibri 1 as a lane family on a tiny random checkpoint (Metal): the vendored forward, ring caches and rollback."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Kolibri 1 runs on MLX's Metal kernels", allow_module_level=True)

from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402
from tensorfold.families.kolibri1.model import Kolibri1, register  # noqa: E402

register()
from mlx_lm.models import kolibri1  # noqa: E402

TINY = {
    "model_type": "kolibri1", "hidden_size": 128, "num_hidden_layers": 5, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 64, "rms_norm_eps": 1e-6, "vocab_size": 256, "num_experts": 8,
    "num_experts_per_tok": 2, "moe_intermediate_size": 128, "shared_expert_intermediate_size": 128,
    "sliding_window": 7, "rope_theta": 10000.0, "tie_word_embeddings": False,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"],
}
copy = LaneEngine.copy_single_cache


@pytest.fixture(scope="module")
def model():
    """The vendored Kolibri 1 on random weights, quantized as the MLX checkpoints are (router bf16, head 8-bit)."""

    import mlx.nn as nn

    mx.random.seed(0)
    model = kolibri1.Model(kolibri1.ModelArgs.from_dict(TINY))
    nn.quantize(model, group_size=64, bits=4, class_predicate=lambda path, m: hasattr(m, "to_quantized")
                and model.quant_predicate(path, m))
    model.set_dtype(mx.bfloat16)
    for layer in model.layers:                         # the router's bias stays fp32, as mlx_lm loads it
        layer.mlp.expert_bias = mx.random.normal((TINY["num_experts"],)) * 0.1
    mx.eval(model.parameters())
    return model


def tokens(n: int, seed: int = 3) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(1, TINY["vocab_size"], size=n)]


def prefilled(family: Kolibri1, n: int, seed: int = 3) -> list:
    cache = family.make_cache()
    mx.eval(family.prefill(mx.array([tokens(n, seed)], dtype=mx.uint32), cache))
    return cache


def steps(family: Kolibri1, cache: list, window: list[int]) -> list:
    out = []
    for token in window:
        logits = family(mx.array([[token]], dtype=mx.uint32), cache)
        mx.eval(logits)
        out.append(logits[0, -1])
    return out


def same(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


# prompts shorter than, equal to and past the window (7), decoding across its end
@pytest.mark.parametrize("prompt", [3, 7, 20])
def test_the_family_decodes_what_the_reference_forward_decodes(model, prompt):
    family = Kolibri1(model)
    ids, window = tokens(prompt), tokens(12, seed=5)
    reference = model.make_cache()
    expected = [model(mx.array([ids]), cache=reference)[0, -1]]
    for token in window:
        expected.append(model(mx.array([[token]]), cache=reference)[0, -1])
    cache = family.make_cache()
    got = [family.head(family.prefill(mx.array([ids], dtype=mx.uint32), cache))[0, -1]]
    got += steps(family, cache, window)
    # the same kernels on the same keys: a window off by one row is off by about the logits' own size
    assert all(same(a, b) for a, b in zip(got, expected))


def test_a_prompt_in_chunks_equals_it_whole(model):
    family = Kolibri1(model)
    ids = tokens(30)
    whole = family.make_cache()
    a = family.head(family.prefill(mx.array([ids], dtype=mx.uint32), whole))[0, -1]
    chunked = family.make_cache()
    for at in range(0, 30, 10):
        b = family.head(family.prefill(mx.array([ids[at:at + 10]], dtype=mx.uint32), chunked))[0, -1]
    assert float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()) < 0.05
    window = tokens(9, seed=8)
    assert [int(mx.argmax(x).item()) for x in steps(family, whole, window)] == \
        [int(mx.argmax(x).item()) for x in steps(family, chunked, window)]


@pytest.mark.parametrize("prompt", [5, 20])
def test_rollback_then_steps_equal_serial_decoding(model, prompt):
    family = Kolibri1(model)
    base = prefilled(family, prompt)
    window = tokens(12, seed=5)
    serial = steps(family, copy(base), window)
    work = copy(base)
    steps(family, work, window[:3] + tokens(4, seed=9))       # 7 one-row calls, the last 4 rejected
    family.keep_rows(work, 4, 0)
    assert [c.offset for c in work] == [prompt + 3] * len(work)
    assert all(same(a, b) for a, b in zip(steps(family, work, window[3:]), serial[3:]))


def test_engine_replies_equal_serial_decoding(model):
    family = Kolibri1(model)
    prompt = tokens(25, seed=21)
    engine = LaneEngine(family, max_rows=1, max_draft=0)
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=16)
    engine.add_stream(stream)
    engine.run()
    reply = []
    reference = model.make_cache()
    step = model(mx.array([prompt]), cache=reference)[0, -1]
    for _ in range(16):
        token = int(mx.argmax(step).item())
        reply.append(token)
        step = model(mx.array([[token]]), cache=reference)[0, -1]
    assert list(stream.emitted) == reply


def test_check_takes_the_mlx_checkpoints_and_refuses_others(tmp_path):
    from tensorfold import families
    from tensorfold.families import kolibri1 as package

    config = dict(TINY, quantization={"group_size": 64, "bits": 4, "mode": "affine",
                                      "lm_head": {"group_size": 64, "bits": 8}})
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert families.detect(tmp_path).module == "tensorfold.families.kolibri1"
    package.check(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(dict(config, quantization={"group_size": 64, "bits": 3})))
    with pytest.raises(ValueError, match="4-bit or 8-bit"):
        package.check(tmp_path)
