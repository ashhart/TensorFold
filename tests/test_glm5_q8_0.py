"""GLM-5.3-Flash from a Q8_0 GGUF: the lossless 8-bit / group-32 encoding, and a checkpoint in that layout loads and
decodes exactly (unquantised small projections, fp16 scales, the decay rate stored as A)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glm5_q8_0_gguf_to_mlx as tool  # noqa: E402
from glm5_fakes import write_checkpoint  # noqa: E402
from test_glm5_next_family import _run_engine, tokens  # noqa: E402
from tensorfold.families import glm5_next  # noqa: E402
from tensorfold.families.glm5_next import linear, weights  # noqa: E402
from tensorfold.families.glm5_next import mtp as glm_mtp  # noqa: E402
from tensorfold.families.glm5_next.runtime import GLMFlash  # noqa: E402

DENSE = ("f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "b_proj", "indexer.wq_b", "indexer.wk", "indexer.weights_proj")


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_q8_0_blocks_encode_exactly():
    rng = np.random.default_rng(0)
    blocks = np.zeros((64, 16), dtype=tool.Q8_DTYPE)
    blocks["d"] = (rng.standard_normal(blocks.shape) * 1e-3).astype(np.float16)
    blocks["d"][0] = np.float16(2 ** -24)                                   # subnormal scales
    blocks["d"][1] = np.float16(3.0)
    blocks["q"] = rng.integers(-128, 128, size=blocks.shape + (32,), dtype=np.int8)
    blocks["q"][2] = -128
    blocks["q"][3] = 127
    packed, scales, biases = tool.q8_to_affine8(blocks)
    got = mx.dequantize(mx.array(packed), mx.array(scales).astype(mx.float32), mx.array(biases).astype(mx.float32),
                        group_size=32, bits=8)
    assert np.array_equal(np.array(got), tool.q8_values(blocks))            # value for value (+0 == -0)


def test_dense_tensors_keep_every_value():
    a = np.array([1.5, -2.0, 2 ** -20, 1 + 2 ** -10], dtype=np.float32)
    data, dtype, _ = tool.dense(a)
    assert dtype == "F32" and np.array_equal(np.frombuffer(data, np.float32), a)
    data, dtype, _ = tool.dense(np.array([1.5, -2.0, 0.25], dtype=np.float16))
    assert dtype == "BF16"


@pytest.fixture(scope="module")
def q8_checkpoint(tmp_path_factory):
    """The tiny checkpoint re-encoded as a Q8_0 conversion writes it."""

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        folder = write_checkpoint(tmp_path_factory.mktemp("glm5q8"))
        index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
        tensors = {}
        for shard in sorted(set(index.values())):
            tensors.update(mx.load(str(folder / shard)))
        out = {}
        for name, value in tensors.items():
            if name.endswith(".scales") or name.endswith(".biases"):
                continue
            base = name[: -len(".weight")] if name.endswith(".weight") else name
            if f"{base}.scales" in tensors:
                w = mx.dequantize(value, tensors[f"{base}.scales"], tensors[f"{base}.biases"], group_size=64, bits=4)
                if "self_attn." in base and base.split("self_attn.", 1)[1] in DENSE:
                    out[f"{base}.weight"] = w.astype(mx.bfloat16)
                else:
                    q, s, b = mx.quantize(w.astype(mx.float16), group_size=32, bits=8)
                    out[f"{base}.weight"], out[f"{base}.scales"], out[f"{base}.biases"] = q, s, b
            elif name.endswith(".A_log"):
                out[name[: -len("A_log")] + "A"] = mx.exp(value.astype(mx.float32))
            else:
                out[name] = value
        mx.eval(out)
        for shard in set(index.values()):
            (folder / shard).unlink()
        mx.save_safetensors(str(folder / "model-00001-of-00001.safetensors"), out)
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in out}}))
        config = json.loads((folder / "config.json").read_text())
        config["quantization"] = {"bits": 8, "group_size": 32}
        (folder / "config.json").write_text(json.dumps(config))
        return folder
    finally:
        mx.set_default_device(previous)


def test_the_mac_engine_admits_it(q8_checkpoint, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    glm5_next.check(q8_checkpoint)
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError):
        glm5_next.check(q8_checkpoint)                                    # the CUDA engine reads 4-bit / 64 only


def test_it_loads_as_stored(q8_checkpoint):
    model = weights.load_backbone(q8_checkpoint)
    kda, mla = model.layers[0].attn, model.layers[3].attn
    assert isinstance(kda.f_b, linear.Dense) and isinstance(mla.ik_proj, linear.Dense)
    assert model.layers[1].mlp.gate.scales.dtype == mx.float32               # fp16 scales widened at load
    ids = tokens(12)
    assert model.head(model.hidden(mx.array([ids]), model.make_cache())).dtype == mx.bfloat16


@pytest.mark.parametrize("length", [9, 40])
def test_prefill_agrees_with_decode(q8_checkpoint, length):
    model = weights.load_backbone(q8_checkpoint)
    ids = tokens(length)
    a = model.head(model.hidden(mx.array([ids]), model.make_cache()))[0, -1]
    step = model.make_cache()
    for t in ids:
        b = model.head(model.hidden(mx.array([[t]]), step))[0, -1]
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert int(a.argmax()) == int(b.argmax())
    assert np.max(np.abs(a - b)) < 0.05 * np.max(np.abs(b)) + 0.05


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_rows_are_exact_and_drafts_change_speed_only(q8_checkpoint, device):
    if device == "gpu":
        if not mx.metal.is_available():
            pytest.skip("needs Metal")
        mx.set_default_device(mx.gpu)
    model = weights.load_backbone(q8_checkpoint)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.multi_row_exact, runtime.check_report
    prompt = tokens(21, seed=4)
    engine_a, a = _run_engine(runtime, prompt, 24)
    _, b = _run_engine(GLMFlash(model, None, drafts=0), prompt, 24)
    assert engine_a.drafted > 0 and a.emitted == b.emitted
