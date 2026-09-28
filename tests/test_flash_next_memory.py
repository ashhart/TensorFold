"""Flash Next startup counts file-backed n-grams separately, without loading checkpoint tensors or MLX."""

from __future__ import annotations

import json
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tensorfold import cli
from tensorfold.families import qwen4_exp
from tensorfold.families.qwen4_exp import host_table
from tensorfold.server import memory_budget

GIB = 1024**3
NGRAM = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"


def _file(path, tensors):
    header = {"__metadata__": {"format": "mlx"}}
    offset = 0
    for name, size in tensors.items():
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        offset += size
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        stream.truncate(8 + len(data) + offset)     # sparse: only headers take disk space
    return path.stat().st_size


def _checkpoint(path, main=4096, mapped=1600):
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 32},
        "max_position_embeddings": 262144,
    }))
    total = 0
    for shard in range(2):
        total += _file(path / f"model-{shard}.safetensors", {
            f"{NGRAM}.shard_{shard}.weight": mapped * 2 // 5,
            f"{NGRAM}.shard_{shard}.scales": mapped // 20,
            f"{NGRAM}.shard_{shard}.biases": mapped // 20,
            f"language_model.model.layers.{shard}.ple.key_proj.weight": main // 2,
        })
    total += _file(path / "mtp.safetensors", {"language_model.mtp.fc_hidden.weight": 64})
    return total


@pytest.fixture
def runtime(monkeypatch):
    core = ModuleType("mlx.core")
    core.device_info = lambda: {"max_recommended_working_set_size": 120 * GIB}
    core.set_memory_limit = lambda value: None
    core.set_cache_limit = lambda value: None
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.delenv("TF_NGRAM_HOST", raising=False)
    monkeypatch.delenv("TENSORFOLD_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.setattr(memory_budget, "physical_memory_bytes", lambda: 128 * GIB)
    monkeypatch.setattr("faulthandler.register", lambda *a, **k: None)
    return core


@pytest.mark.parametrize("flag, fits", [(None, True), ("1", True), ("0", False)])
def test_serve_fits_host_ngrams_within_the_original_budget(tmp_path, monkeypatch, runtime, flag, fits):
    _checkpoint(tmp_path, main=75 * GIB, mapped=30 * GIB)
    if flag is not None:
        monkeypatch.setenv("TF_NGRAM_HOST", flag)
    budgets = []
    monkeypatch.setattr(cli, "_serve_mlx", lambda args, family, path, context, required, budget, *rest:
                        budgets.append(budget) or 0)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "mlx", "--no-update-check"])
    if fits:
        assert cli.cmd_serve(args) == 0
        assert budgets == [int(0.70 * 128 * GIB)]
    else:
        with pytest.raises(ValueError, match=r"weights \(105\.0 GiB\) do not fit.*89\.6 GiB"):
            cli.cmd_serve(args)
        assert budgets == []


@pytest.mark.parametrize("flag", [None, "1"])
def test_host_mode_still_refuses_resident_weights_past_the_limit(tmp_path, monkeypatch, runtime, flag):
    _checkpoint(tmp_path, main=75 * GIB, mapped=30 * GIB)
    monkeypatch.setenv("TENSORFOLD_MEMORY_LIMIT_GB", "78")    # exactly 75 GiB for buffers, no room for the head
    if flag is not None:
        monkeypatch.setenv("TF_NGRAM_HOST", flag)
    monkeypatch.setattr(cli, "_serve_mlx", lambda *a: pytest.fail("weights loaded past the budget"))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "mlx", "--no-update-check"])
    with pytest.raises(ValueError, match=r"weights \(75\.0 GiB\) do not fit.*78\.0 GiB"):
        cli.cmd_serve(args)


def test_serve_passes_a_raised_budget_to_the_allocator_and_server(tmp_path, monkeypatch, runtime, capsys):
    _checkpoint(tmp_path, main=75 * GIB, mapped=30 * GIB)
    monkeypatch.setenv("TENSORFOLD_MEMORY_LIMIT_GB", "110")
    allocations, budgets = [], []
    runtime.set_memory_limit = allocations.append
    monkeypatch.setattr(cli, "_serve_mlx", lambda args, family, path, context, required, budget, *rest:
                        budgets.append((budget, context, args.max_tokens)) or 0)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "mlx", "--no-update-check",
                                         "--context", "262144", "--max-tokens", "131072"])
    assert cli.cmd_serve(args) == 0
    assert allocations == [107 * GIB]
    assert budgets == [(110 * GIB, 262144, 131072)]
    assert "memory budget 110.0 GiB: MLX's buffers up to 107.0 GiB" in capsys.readouterr().out


def test_only_mapped_table_tensors_are_subtracted(tmp_path, monkeypatch, runtime):
    total = _checkpoint(tmp_path)
    total += _file(tmp_path / "model-extra.safetensors", {
        f"{NGRAM}.shard_0.unrecognized": 128,
        "vision.ple_embedding.ngram_embedding.shard_0.weight": 256,
        "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets": 64,
    })
    total += _file(tmp_path / "unused.safetensors", {f"{NGRAM}.shard_9.weight": 512})
    monkeypatch.setenv("TF_NGRAM_HOST", "1")
    assert qwen4_exp.weight_bytes(tmp_path) == total - 1600
    monkeypatch.setenv("TF_NGRAM_HOST", "0")
    monkeypatch.setattr(host_table, "read_header", lambda *a: pytest.fail("resident tables need no header scan"))
    assert qwen4_exp.weight_bytes(tmp_path) == total


@pytest.mark.parametrize("flag, expected", [(None, True), ("1", True), ("0", False)])
@pytest.mark.parametrize("legacy", [False, True])
def test_host_selection_uses_the_loaders_threshold_and_overrides(tmp_path, monkeypatch, runtime, flag, expected, legacy):
    _checkpoint(tmp_path)
    model_size = sum(p.stat().st_size for p in tmp_path.glob("model*.safetensors"))
    runtime.device_info = lambda: {"max_recommended_working_set_size": model_size}
    if legacy:
        runtime.metal = SimpleNamespace(device_info=runtime.device_info)
        del runtime.device_info
    if flag is not None:
        monkeypatch.setenv("TF_NGRAM_HOST", flag)
    assert host_table.ngrams_on_host(tmp_path) is expected
    info = lambda: {"max_recommended_working_set_size": model_size * 2}
    if legacy:
        runtime.metal.device_info = info
    else:
        runtime.device_info = info
    assert host_table.ngrams_on_host(tmp_path) is (flag == "1")
