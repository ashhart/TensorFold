"""Offline EXL3 vision conversion and externally supplied tower admission."""
import json

import numpy as np
import pytest

from tensorfold.vision.exl3_convert import convert, convert_tensors


def _group(rng, prefix, bits=6):
    return {prefix + ".trellis": rng.integers(-32768, 32767, (8, 8, 16 * bits), dtype=np.int16),
            prefix + ".suh": np.full(128, 0.1, dtype=np.float16),
            prefix + ".svh": np.full(128, 0.2, dtype=np.float16),
            prefix + ".mul1": np.array([-2082672339], dtype=np.int32),
            prefix + ".bias": np.arange(128, dtype=np.float16) * np.float16(0.01)}


def test_converter_preserves_represented_linears_and_qkv_order():
    from tensorfold.cuda.exl3 import format as fmt

    rng = np.random.default_rng(31)
    source = {}
    for proj in ("q", "k", "v"):
        source.update(_group(rng, f"model.visual.blocks.0.attn.{proj}_proj"))
    source["model.visual.blocks.0.attn.qkv.weight"] = np.zeros((384, 128), dtype=np.float16)
    source["model.visual.blocks.0.attn.qkv.bias"] = np.zeros(384, dtype=np.float16)
    result = convert_tensors(source)
    weight = result["vision_tower.blocks.0.attn.qkv.weight"]
    bias = result["vision_tower.blocks.0.attn.qkv.bias"]
    x = rng.normal(size=(2, 128)).astype(np.float16)
    refs = []
    for proj in ("q", "k", "v"):
        p = f"model.visual.blocks.0.attn.{proj}_proj"
        refs.append(fmt.forward(x, source[p + ".trellis"], source[p + ".suh"], source[p + ".svh"], 6,
                                "mul1", source[p + ".bias"]))
    np.testing.assert_allclose(x.astype(np.float64) @ weight.astype(np.float64).T + bias,
                               np.concatenate(refs, axis=1), atol=0.001, rtol=0.002)
    assert weight.shape == (384, 128) and weight.dtype == np.float16
    assert not any("q_proj" in name or "trellis" in name for name in result)


def test_converter_is_hashed_reusable_and_never_overwrites_source(tmp_path):
    from safetensors.numpy import save_file
    from safetensors import safe_open

    source, output = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    tensors = _group(np.random.default_rng(1), "model.visual.attn.proj")
    save_file(tensors, str(source))
    original = source.read_bytes()
    assert convert(source, output) == output
    timestamp = output.stat().st_mtime_ns
    assert convert(source, output) == output and output.stat().st_mtime_ns == timestamp
    with safe_open(str(output), framework="np") as artifact:
        assert len(artifact.metadata()["source_sha256"]) == 64
    with pytest.raises(ValueError, match="immutable"):
        convert(source, source)
    tensors["model.visual.attn.proj.suh"][0] *= 2
    save_file(tensors, str(source))
    with pytest.raises(ValueError, match="different source"):
        convert(source, output)
    assert original != source.read_bytes()


def test_external_tower_is_read_without_language_payloads_and_counted_once(tmp_path, monkeypatch):
    from test_vision_cuda import _checkpoint
    from tensorfold.cuda.capacity import Geometry
    from tensorfold.vision.qwen_cuda import checkpoint_vision, capacity_geometry, weight_transform

    _, size = _checkpoint(tmp_path)
    tower = tmp_path / "external.safetensors"
    (tmp_path / "model.safetensors").rename(tower)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"language.weight": "absent"}}))
    monkeypatch.setenv("TENSORFOLD_VISION_WEIGHTS", str(tower))
    assert checkpoint_vision(tmp_path)[1] == size
    base = lambda text: Geometry(lambda slots: 100 + slots, 2)
    assert capacity_geometry(base, tmp_path, True, 0, 50)({}).bytes_at(10) == 160 + size
    assert weight_transform(lambda n, i: (7, 0), True, 0)("model.visual.test", {}) == (0, 0)
