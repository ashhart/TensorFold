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


def test_converter_removes_only_configured_mlp_padding():
    rng = np.random.default_rng(8)
    source = {}
    for part in ("linear_fc1", "linear_fc2"):
        source.update(_group(rng, "model.visual.blocks.0.mlp." + part))
    config = {"depth": 1, "hidden_size": 128, "intermediate_size": 112}
    full = convert_tensors(source)
    trimmed = convert_tensors(source, config)
    prefix = "vision_tower.blocks.0.mlp."
    np.testing.assert_array_equal(trimmed[prefix + "linear_fc1.weight"], full[prefix + "linear_fc1.weight"][:112])
    np.testing.assert_array_equal(trimmed[prefix + "linear_fc1.bias"], full[prefix + "linear_fc1.bias"][:112])
    np.testing.assert_array_equal(trimmed[prefix + "linear_fc2.weight"], full[prefix + "linear_fc2.weight"][:, :112])
    with pytest.raises(ValueError, match="padding"):
        convert_tensors(source, {**config, "intermediate_size": 129})


def test_conversion_hashes_config_beside_snapshot_symlink(tmp_path):
    from safetensors.numpy import save_file
    from safetensors import safe_open
    import hashlib

    blob = tmp_path / "blob.safetensors"
    save_file({"model.visual.pos_embed.weight": np.zeros((4, 8), np.float16)}, str(blob))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = snapshot / "vision.safetensors"
    source.symlink_to(blob)
    config = b'{"vision_config": {"depth": 0, "hidden_size": 8, "intermediate_size": 8}}'
    (snapshot / "config.json").write_bytes(config)
    output = tmp_path / "converted.safetensors"
    convert(source, output)
    with safe_open(str(output), framework="np") as artifact:
        assert artifact.metadata()["config_sha256"] == hashlib.sha256(config).hexdigest()


def _bf16(values):
    """Float32 values truncated to bfloat16 bits (exact for these test values)."""
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _write(path, tensors):
    """A safetensors file from {name: array}, with uint16 arrays stored as BF16 (safetensors.numpy can't)."""
    import struct

    header, blobs, offset = {}, [], 0
    for name, value in tensors.items():
        data = np.ascontiguousarray(value).tobytes()
        dtype = {"uint16": "BF16", "float16": "F16", "float32": "F32", "int16": "I16", "int32": "I32"}[value.dtype.name]
        header[name] = {"dtype": dtype, "shape": list(value.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def _pack(tmp_path, language):
    """A two-shard pack whose second shard holds language weights beside a quantized and a BF16 vision tensor."""
    rng = np.random.default_rng(5)
    vision = _group(rng, "model.visual.blocks.0.attn.proj")
    vision["model.visual.pos_embed.weight"] = _bf16([[0.5, -2.0], [3.0, 0.25]])
    _write(tmp_path / "model-00001-of-00002.safetensors", {"model.embed_tokens.weight": language})
    _write(tmp_path / "model-00002-of-00002.safetensors", {"model.norm.weight": language, **vision})
    shard = {name: "model-00002-of-00002.safetensors" for name in ("model.norm.weight", *vision)}
    index = {"weight_map": {"model.embed_tokens.weight": "model-00001-of-00002.safetensors", **shard}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return vision


def test_converter_reads_the_tower_from_a_packs_shared_shards(tmp_path):
    from safetensors.numpy import load_file

    pack, output = tmp_path / "pack", tmp_path / "vision-f16.safetensors"
    pack.mkdir()
    vision = _pack(pack, np.ones(4, np.float16))
    convert(pack, output)
    result = load_file(str(output))
    assert set(result) == {"vision_tower.pos_embed.weight", "vision_tower.blocks.0.attn.proj.weight",
                           "vision_tower.blocks.0.attn.proj.bias"}
    np.testing.assert_array_equal(result["vision_tower.pos_embed.weight"],
                                  np.array([[0.5, -2.0], [3.0, 0.25]], np.float16))
    expected = convert_tensors({k: v for k, v in vision.items() if "pos_embed" not in k})
    np.testing.assert_array_equal(result["vision_tower.blocks.0.attn.proj.weight"],
                                  expected["vision_tower.blocks.0.attn.proj.weight"])


def test_pack_hash_covers_the_tower_and_ignores_language_weights(tmp_path):
    pack, output = tmp_path / "pack", tmp_path / "vision-f16.safetensors"
    pack.mkdir()
    _pack(pack, np.ones(4, np.float16))
    convert(pack, output)
    _pack(pack, np.zeros(4, np.float16))                    # language weights changed: the artifact still fits
    assert convert(pack, output) == output
    shard = pack / "model-00002-of-00002.safetensors"
    tensors = {k: v for k, v in _read_raw(shard).items()}
    tensors["model.visual.pos_embed.weight"] = _bf16([[1.0, 1.0], [1.0, 1.0]])
    _write(shard, tensors)
    with pytest.raises(ValueError, match="different source"):
        convert(pack, output)


def _read_raw(path):
    from tensorfold.vision.qwen_checkpoint import _header, read_vision_tensor

    header, begin = _header(path)
    return {name: read_vision_tensor(name, path, item, begin) for name, item in header.items()}


def test_converter_reads_bf16_sidecars(tmp_path):
    from safetensors.numpy import load_file

    source, output = tmp_path / "vision.safetensors", tmp_path / "vision-f16.safetensors"
    _write(source, {"model.visual.merger.norm.bias": _bf16([1.5, -0.125])})
    convert(source, output)
    np.testing.assert_array_equal(load_file(str(output))["vision_tower.merger.norm.bias"],
                                  np.array([1.5, -0.125], np.float16))
