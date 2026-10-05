from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import safe_open, save_file

from tensorfold.families.deepseek_v41.convert import (
    convert,
    dequant_fp4_e8m0,
    dequant_fp8_blockwise,
    dequantize_packed,
    pack_bits,
    quantize_pack,
    unpack_bits,
)


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("width", [64, 128, 192])
def test_affine_packing_round_trips_across_word_boundaries(bits, width):
    codes = torch.randint(
        0, 1 << bits, (5, width), dtype=torch.int64, generator=torch.Generator().manual_seed(bits * width)
    )
    packed = pack_bits(codes, bits)
    assert torch.equal(unpack_bits(packed, bits, width), codes)


def test_affine_quantization_respects_bf16_storage_error_bound():
    source = torch.randn((7, 512), generator=torch.Generator().manual_seed(22)) * 0.02
    for bits in (3, 4):
        packed, scales, biases = quantize_pack(source, bits, 64)
        reconstructed = dequantize_packed(packed, scales, biases, bits, 64, width=512)
        storage_error = ((2**bits - 1) * scales.float() + biases.float().abs()) * (2**-8)
        bound = (scales.float() / 2 + storage_error).repeat_interleave(64, dim=1)
        assert torch.all((source - reconstructed).abs() <= bound)


def test_official_fp4_nibbles_and_fp8_block_scales_decode():
    fp4 = torch.full((1, 32), 0x21, dtype=torch.int8)
    fp4_scales = torch.full((1, 2), 128, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    assert torch.equal(dequant_fp4_e8m0(fp4, fp4_scales), torch.tensor([[1.0, 2.0] * 32]))

    fp8 = torch.ones((64, 64), dtype=torch.float8_e4m3fn)
    row_scales = torch.full((64, 2), 0.25)
    block_scales = torch.full((2, 2), 0.125)
    assert torch.equal(dequant_fp8_blockwise(fp8, row_scales), torch.full((64, 64), 0.25))
    assert torch.equal(dequant_fp8_blockwise(fp8, block_scales), torch.full((64, 64), 0.125))


def test_complete_conversion_writes_a_loadable_index_after_atomic_shard_conversion(tmp_path):
    source, output = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    weight = torch.ones((64, 64), dtype=torch.float8_e4m3fn)
    scale = torch.full((64, 2), 0.25)
    norm = torch.ones((64,), dtype=torch.bfloat16)
    hc_scale = torch.full((64,), 0.5, dtype=torch.float32)
    save_file(
        {"layer.weight": weight, "layer.scale": scale, "norm.weight": norm, "layer.hc.scale": hc_scale},
        str(source / shard_name),
    )
    (source / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": shard_name,
                    "layer.scale": shard_name,
                    "norm.weight": shard_name,
                    "layer.hc.scale": shard_name,
                }
            }
        )
    )

    config_path = convert(source, output)

    assert config_path == output / "config.json"
    config = json.loads(config_path.read_text())
    assert config["quantization"] == {"bits": 3, "group_size": 64, "mode": "affine"}
    output_index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(output_index["weight_map"]) == {
        "layer.weight",
        "layer.scales",
        "layer.biases",
        "norm.weight",
        "layer.hc.scale",
    }
    assert output_index["metadata"]["total_size"] == (output / shard_name).stat().st_size
    with safe_open(output / shard_name, framework="pt") as converted:
        decoded = dequantize_packed(
            converted.get_tensor("layer.weight"),
            converted.get_tensor("layer.scales"),
            converted.get_tensor("layer.biases"),
            3,
            64,
            width=64,
        )
        assert torch.allclose(decoded, torch.full((64, 64), 0.25), atol=1e-3)
        assert torch.equal(converted.get_tensor("layer.hc.scale"), hc_scale.bfloat16())
    assert (source / shard_name).is_file()


def test_partial_conversion_never_publishes_a_runnable_checkpoint(tmp_path):
    source, output = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    names = {}
    for number in (1, 2):
        shard_name = f"model-{number:05d}-of-00002.safetensors"
        save_file({f"norm{number}.weight": torch.ones((64,), dtype=torch.bfloat16)}, str(source / shard_name))
        names[f"norm{number}.weight"] = shard_name
    (source / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": names}))

    result = convert(source, output, shards=[1])

    assert result is None
    assert not (output / "model.safetensors.index.json").exists()
    assert json.loads((output / "converted-shards.json").read_text())["complete"] is False
