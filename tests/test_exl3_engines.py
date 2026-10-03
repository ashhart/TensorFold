"""EXL3 packs on the 27B and Flash Next CUDA engines, host side: metadata, shards, admission bytes and refusals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safetensors.numpy import save_file  # noqa: E402

EXL3 = {"quant_method": "exl3", "version": "1.4.2", "bits": 3.0, "head_bits": 6, "codebook": "mul1"}


def _config(tmp: Path, **extra) -> Path:
    config = {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5"}, **extra}
    (tmp / "config.json").write_text(json.dumps(config))
    return tmp


@pytest.mark.parametrize("where", ["top", "text", "text_quantization", "sidecar"])
def test_the_27b_reads_exl3_metadata_wherever_the_gate_accepts_it(tmp_path, where):
    from tensorfold.families import quant_method, read_config
    from tensorfold.families.qwen3_5.cuda.exl3_load import quant_config

    if where == "top":
        _config(tmp_path, quantization_config=EXL3)
    elif where == "text":
        _config(tmp_path, text_config={"model_type": "qwen3_5", "quantization_config": EXL3})
    elif where == "text_quantization":
        _config(tmp_path, text_config={"model_type": "qwen3_5", "quantization": EXL3})
    else:
        _config(tmp_path)
        (tmp_path / "quantization_config.json").write_text(json.dumps(EXL3))
    assert quant_config(tmp_path)["quant_method"] == "exl3"
    if where != "sidecar":
        assert quant_method(read_config(tmp_path)) == "exl3"


def test_an_mlx_checkpoint_is_not_taken_for_exl3(tmp_path):
    from tensorfold.families.qwen3_5.cuda.exl3_load import quant_config

    _config(tmp_path, quantization={"bits": 4, "group_size": 64})
    assert quant_config(tmp_path) is None


def _group(k: int, n: int, bits: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"trellis": rng.integers(-2**15, 2**15, size=(k // 16, n // 16, 16 * bits)).astype(np.int16),
            "suh": (rng.standard_normal(k) * 0.05).astype(np.float16),
            "svh": (rng.standard_normal(n) * 0.05).astype(np.float16),
            "mul1": np.array(0x83DCD12D - 2**32, dtype=np.int32)}


def test_a_group_split_across_shards_loads_from_each_part_s_own_file(tmp_path):
    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.families.qwen3_5.cuda.exl3_load import _files, _read_groups, _where

    prefix = "model.language_model.layers.0.mlp.down_proj"
    g = _group(256, 128, 3, 1)
    save_file({f"{prefix}.trellis": g["trellis"], f"{prefix}.mul1": g["mul1"]}, str(tmp_path / "a.safetensors"))
    save_file({f"{prefix}.suh": g["suh"], f"{prefix}.svh": g["svh"]}, str(tmp_path / "b.safetensors"))
    _config(tmp_path, quantization_config=EXL3)
    ckpt = fmt.scan(tmp_path, read_markers=False)
    assert set(ckpt.groups[prefix].files) == {"a.safetensors", "b.safetensors"}
    layer = _read_groups(_files(tmp_path, _where(tmp_path)), ckpt.groups, "cpu")[prefix].layer
    assert torch.equal(layer.suh, torch.from_numpy(g["suh"])) and torch.equal(layer.svh, torch.from_numpy(g["svh"]))
    assert layer.k == 256 and layer.n == 128 and layer.bits == 3 and layer.codebook == "mul1"


def test_admission_counts_an_exl3_pack_as_loaded():
    from tensorfold.cuda.geometry import exl3_weights, indexed_weights

    head = {"dtype": "I16", "shape": [320, 15520, 96]}
    assert exl3_weights("lm_head.trellis", head)[0] == 320 * 15520 * 96 * 2 * 7 // 5
    assert exl3_weights("lm_head.suh", {"dtype": "F16", "shape": [5120]})[0] == 5120 * 2
    for skipped in ("model.visual.blocks.0.attn.qkv.weight", "mtp.fc.trellis", "model.language_model.mtp.x.suh"):
        assert exl3_weights(skipped, {"dtype": "BF16", "shape": [64, 64]}) == (0, 0)
    table = {"dtype": "I16", "shape": [1000, 81]}
    name = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.trellis"
    assert indexed_weights(1, True)(name, table) == (0, 1000 * 81 * 2)
    assert indexed_weights(1, True)("model.visual.merger.fc.weight", {"dtype": "BF16", "shape": [8, 8]}) == (0, 0)


@pytest.mark.parametrize("suffix", ["shard_0.trellis", "trellis"])
def test_extra_files_add_their_mapped_pages(tmp_path, suffix):
    from tensorfold.cuda.capacity import estimate_weights
    from tensorfold.cuda.geometry import indexed_weights

    shard = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding." + suffix
    save_file({shard: np.zeros((100, 51), dtype=np.int16),
               "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.head_bias":
                   np.zeros((16, 160), dtype=np.float16)}, str(tmp_path / "ngram_embedding.safetensors"))
    got = estimate_weights(tmp_path, indexed_weights(1, True), files=[tmp_path / "ngram_embedding.safetensors"])
    assert got.mapped == 100 * 51 * 2 and got.resident == 16 * 160 * 2


def test_two_ranks_refuse_an_exl3_pack_before_any_setup(tmp_path):
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    _config(tmp_path, quantization_config=EXL3)
    with pytest.raises(ValueError, match="one GPU"):
        Qwen27Engine(tmp_path, None, tp=2, rank=1, master="192.0.2.1")
    with pytest.raises(ValueError, match="one GPU"):
        FlashNextEngine(tmp_path, tp=2, rank=1, master="192.0.2.1")
    with pytest.raises(ValueError, match="--ple-on-ssd"):
        FlashNextEngine(tmp_path, ple_on_ssd=True)
