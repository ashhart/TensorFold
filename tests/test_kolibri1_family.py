"""Kolibri 1 as a lane family on a tiny random checkpoint (Metal): the vendored forward, ring caches, rollback and
the row-exact decode's windows, streams and drafts."""

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
    "num_experts_per_tok": 2, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
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
    family = Kolibri1(model, backend=None)
    ids, window = tokens(prompt), tokens(12, seed=5)
    reference = model.make_cache()
    expected = [model(mx.array([ids]), cache=reference)[0, -1]]
    for token in window:
        expected.append(model(mx.array([[token]]), cache=reference)[0, -1])
    cache = family.make_cache()
    got = [family.head(family.prefill(mx.array([ids], dtype=mx.uint32), cache))[0, -1]]
    got += steps(family, cache, window)
    # the same kernels on the same keys, though mlx_lm's full rotating cache hands them over rotated (another sum
    # order, an ulp or two); a window off by one row is off by about the logits' own size
    for a, b in zip(got, expected):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        assert int(mx.argmax(a).item()) == int(mx.argmax(b).item())
        assert float(mx.abs(a - b).max().item()) < 0.05 * float(mx.abs(b).max().item())


def test_a_prompt_in_chunks_equals_it_whole(model):
    family = Kolibri1(model, backend=None)
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


@pytest.mark.parametrize("backend", [None, "rows"])
@pytest.mark.parametrize("prompt", [5, 20, 131])
def test_rollback_then_steps_equal_serial_decoding(model, prompt, backend):
    family = Kolibri1(model, backend=backend, check=False)
    base = prefilled(family, prompt)
    window = tokens(12, seed=5)
    serial = steps(family, copy(base), window)
    work = copy(base)
    steps(family, work, window[:3] + tokens(4, seed=9))       # 7 one-row calls, the last 4 rejected
    family.keep_rows(work, 4, 0)
    assert [c.offset for c in work] == [prompt + 3] * len(work)
    assert all(same(a, b) for a, b in zip(steps(family, work, window[3:]), serial[3:]))


def test_engine_replies_equal_serial_decoding(model):
    family = Kolibri1(model, backend=None)
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


def _write_safetensors(path, tensors: dict) -> None:
    """A safetensors file with the official dtypes (F8_E4M3 codes as uint8 bytes, BF16, F32)."""

    import struct

    header, blobs, at = {}, [], 0
    for name, (dtype, array) in tensors.items():
        raw = np.asarray(array).tobytes()
        header[name] = {"dtype": dtype, "shape": list(np.asarray(array).shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    path.write_bytes(struct.pack("<Q", len(data)) + data + b"".join(blobs))


def _bf16(a) -> np.ndarray:
    return np.asarray(mx.array(a).astype(mx.bfloat16).view(mx.uint16))


def test_an_fp8_checkpoint_converts_to_an_mlx_one_that_decodes_like_its_weights(model, tmp_path):
    """The official layout (FP8 codes, fp32 block scales, per-expert tensors) -> MLX 8-bit: the converted model's
    logits are bit for bit those of the exact bf16 weights quantized by mlx_lm (the reference dequant, then 8-bit)."""

    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from tensorfold.families.kolibri1.convert import convert
    from tensorfold.families.kolibri1.model import load

    mx.random.seed(1)
    exact = kolibri1.Model(kolibri1.ModelArgs.from_dict(TINY))       # bf16, filled from the FP8 checkpoint below
    exact.set_dtype(mx.bfloat16)
    params = dict(tree_flatten(exact.parameters()))
    src = tmp_path / "fp8"
    src.mkdir()
    stored, loaded = {}, {}
    for name, value in params.items():
        layer = name.split(".mlp.switch_mlp.")
        if name.endswith("expert_bias"):
            bias = mx.random.normal(value.shape) * 0.1
            stored[name.replace(".mlp.expert_bias", ".moe.router.expert_bias")] = ("BF16", _bf16(bias))
            loaded[name] = bias.astype(mx.bfloat16)
        elif value.ndim >= 2 and "norm" not in name and not name.endswith("mlp.gate.weight") \
                and "embed_tokens" not in name and not name.startswith("lm_head"):
            weights = [value[e] for e in range(value.shape[0])] if value.ndim == 3 else [value]
            exacts = []
            for e, w in enumerate(weights):
                rows, cols = w.shape
                scale = mx.random.uniform(0.001, 0.01, ((rows + 127) // 128, (cols + 127) // 128))
                big = mx.repeat(mx.repeat(scale, 128, axis=0)[:rows], 128, axis=1)[:, :cols]
                codes = mx.to_fp8(w.astype(mx.float32) / big)
                key = f"{layer[0]}.mlp.experts.{e}.{layer[1]}" if value.ndim == 3 else name
                stored[key] = ("F8_E4M3", np.asarray(codes))
                stored[key.replace(".weight", ".weight_scale_inv")] = ("F32", np.asarray(scale))
                exacts.append((mx.from_fp8(codes, dtype=mx.bfloat16) * big).astype(mx.bfloat16))   # the reference's dequant
            loaded[name] = mx.stack(exacts) if value.ndim == 3 else exacts[0]
        else:
            stored[name] = ("BF16", _bf16(value))
            loaded[name] = value
    _write_safetensors(src / "model-00001-of-00001.safetensors", stored)
    (src / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001-of-00001.safetensors" for k in stored}}))
    (src / "config.json").write_text(json.dumps(dict(TINY, quantization_config={
        "quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]})))
    exact.load_weights(list(loaded.items()))
    mx.eval(exact.parameters())

    out = convert(src, tmp_path / "mlx", bits=8)
    config = json.loads((out / "config.json").read_text())
    assert config["quantization"]["bits"] == 8 and "quant_method" not in config["quantization_config"]
    from tensorfold.families import kolibri1 as package

    package.check(out)
    (out / "tokenizer.json").unlink(missing_ok=True)
    import mlx_lm.utils

    original = mlx_lm.utils.load_tokenizer
    mlx_lm.utils.load_tokenizer = lambda *a, **k: None                # no tokenizer in the tiny checkpoint
    try:
        family, _ = load(out, backend=None)        # 8-bit: mlx_lm's forward
    finally:
        mlx_lm.utils.load_tokenizer = original
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    assert isinstance(family.model.layers[0].mlp.switch_mlp.gate_proj, QuantizedSwitchLinear)
    assert not isinstance(family.model.layers[0].mlp.gate, nn.QuantizedLinear)
    nn.quantize(exact, group_size=64, bits=8, class_predicate=lambda path, m: hasattr(m, "to_quantized")
                and exact.quant_predicate(path, m))
    ids = tokens(20)
    want = exact(mx.array([ids]))[0]
    got = family.head(family.prefill(mx.array([ids], dtype=mx.uint32), family.make_cache()))[0]
    assert same(got, want)


# -- the row-exact decode (4-bit) ----------------------------------------------------------------------------------------
def rows_family(model, width: int = 16) -> Kolibri1:
    family = Kolibri1(model, check=False)
    family.exact_width = width
    return family


# 150 tokens pass the ring (window 7 + 128 slots): windows write across its end
@pytest.mark.parametrize("prompt", [5, 20, 150])
def test_windows_give_every_row_its_serial_bits(model, prompt):
    family = rows_family(model)
    base = prefilled(family, prompt)
    window = tokens(14, seed=5)
    serial = steps(family, copy(base), window)
    for width in range(2, len(window) + 1):
        logits = family.head(family.hidden(mx.array([window[:width]], dtype=mx.uint32), copy(base)))
        mx.eval(logits)
        assert all(same(logits[0, i], serial[i]) for i in range(width)), width


def test_the_row_decode_follows_the_reference_forward(model):
    """Its own kernels round differently from mlx_lm's, but decode the same tokens on a random model's clear margins."""

    rows, plain = rows_family(model), Kolibri1(model, backend=None)
    ids, window = tokens(20), tokens(12, seed=5)
    got = steps(rows, prefilled(rows, 20), window)
    want = steps(plain, prefilled(plain, 20), window)
    for a, b in zip(got, want):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        assert float(mx.abs(a - b).max().item()) < 0.05 * float(mx.abs(b).max().item())
    del ids


def test_load_checks_find_every_width_exact_and_streams_shared(model):
    family = Kolibri1(model)
    assert family.exact_width == family.fused_rows and family.max_streams > 1 and family.streams_exact
    engine = LaneEngine(family, max_rows=16, max_draft=15)
    assert engine.family_streams and engine.batch_streams == family.max_streams


def test_one_row_a_step_shares_no_forward(model):
    family = Kolibri1(model, backend=None)
    assert not family.streams_exact and family.max_streams == 1


def test_the_router_gives_every_row_its_one_row_bits():
    """At Kolibri 1's router shape (2,560 -> 384) MLX's bf16 matmul gives a row among 6, 15 or 32 other bits than
    alone, so a row among others could route to other experts (#434); the router's gemv rows give each its own."""

    from tensorfold.kernels.glm.flash.v1.kernels import matmul_rows

    router = (mx.random.normal((384, 2560), key=mx.random.key(1)) * 0.05).astype(mx.bfloat16)
    x = (mx.random.normal((32, 2560), key=mx.random.key(3)) * 4).astype(mx.bfloat16)
    alone = mx.concatenate([x[r:r + 1] @ router.T for r in range(32)])
    for rows in (1, 2, 6, 15, 16, 32):
        assert same(matmul_rows(x[:rows], router, transposed=False), alone[:rows]), rows


def test_a_shared_forward_gives_each_stream_its_own_bits(model):
    family = rows_family(model)
    assert family.check_streams(None)
    bases = [prefilled(family, n, seed=n) for n in (20, 133, 140)]
    windows = [tokens(n, seed=11 + n) for n in (3, 5, 1)]
    alone = []
    for base, window in zip(bases, windows):
        logits = family.head(family.hidden(mx.array([window], dtype=mx.uint32), copy(base)))
        mx.eval(logits)
        alone.append(logits[0])
    joint = family.head(family.hidden_rows(windows, [copy(b) for b in bases]))[0]
    mx.eval(joint)
    at = 0
    for window, own in zip(windows, alone):
        assert same(joint[at:at + len(window)], own)
        at += len(window)


def _reply(family, prompt, n, *, sampling=None, proposer=None):
    engine = LaneEngine(family, max_rows=16, max_draft=15)
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=n, sampling=sampling,
                        proposer=proposer, drafts=proposer is not None)
    engine.add_stream(stream)
    engine.run()
    return list(stream.emitted), engine


@pytest.mark.parametrize("sampled", [False, True])
def test_drafted_replies_equal_serial_ones(model, sampled):
    from gemma4_tiny import KnownReply

    from tensorfold.engine.exact_sampling import Sampling

    family = rows_family(model)
    prompt = tokens(140, seed=21)
    sampling = Sampling(seed=5, temperature=1.0, top_k=20, top_p=0.95) if sampled else None
    serial, _ = _reply(family, prompt, 40, sampling=sampling)
    drafted, engine = _reply(family, prompt, 40, sampling=sampling, proposer=KnownReply(prompt + serial))
    assert drafted == serial
    assert engine.drafted > 0 and 0 < engine.accepted < engine.drafted
    assert max(r.rows for r in engine.round_stats) > 2
