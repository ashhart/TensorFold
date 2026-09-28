"""Family metadata gates and CLI routing are exercised without loading accelerator frameworks."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from tensorfold import cli, families
from tensorfold.families import nemotron_h, qwen3_5, qwen4_exp


@pytest.fixture(autouse=True)
def block_accelerators(monkeypatch):
    for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "torch", "triton"):
        monkeypatch.setitem(sys.modules, name, None)


def configuration(bits=4, group=64, **overrides):
    return {"quantization": {"bits": bits, "group_size": group, **overrides}}


def qwen_family():
    return families.Family("qwen3_5", qwen3_5.TITLE, qwen3_5.__name__, True)


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_qwen_affine_metadata_accepts_all_declared_combinations(backend, bits, group):
    value = configuration(bits, group)
    qwen3_5.check_quantization(value, backend)
    families.require_readable(qwen_family(), value, backend)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_native_tensor_unit_routing_is_narrower_than_format_admission(bits, group):
    assert qwen3_5.native_lanes(configuration(bits, group)) is (group == 64 or (bits == 4 and group == 32))


@pytest.mark.parametrize("entry,native", [({"bits": 8, "group_size": 128}, False),
                                         ({"bits": 5}, True), ({"group_size": 32}, True),
                                         ({}, True), (False, True), (True, True)])
def test_native_routing_resolves_language_projection_overrides(entry, native):
    value = configuration(**{"model.layers.0.self_attn.q_proj": entry})
    assert qwen3_5.native_lanes(value) is native


def test_embedding_and_vision_format_overrides_do_not_force_language_rows():
    value = configuration(**{"model.embed_tokens": {"bits": 8, "group_size": 128},
                             "vision_tower.blocks.0.proj": {"bits": 5, "group_size": 32}})
    assert qwen3_5.native_lanes(value)


def test_family_metadata_prioritizes_modern_quantization_and_skips_empty_overrides():
    value = configuration(8, 128, **{"model.layers.0.q_proj": {}, "model.layers.0.k_proj": False,
                                    "model.layers.0.v_proj": {"bits": 3}})
    value["quantization_config"] = {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    assert families.quant_method(value) == "mlx" and families.quantization(value) == (8, 128)
    assert families.layer_quantization(value) == {"model.layers.0.v_proj": (3, 64, "affine")}
    assert "8-bit" in families.describe_quantization(value)


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("value", [configuration(7, 64), configuration(4, 16),
                                  configuration(mode="mxfp4"), configuration(mode="nvfp4"),
                                  configuration(mode="mxfp8"), configuration(quant_method="gptq")])
def test_qwen_family_refuses_unsupported_global_formats(backend, value):
    with pytest.raises(ValueError):
        families.require_readable(qwen_family(), value, backend)


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_qwen_family_validates_per_module_overrides(backend):
    value = configuration(**{"model.layers.0.self_attn.q_proj": {"bits": 7}})
    with pytest.raises(ValueError, match="bits"):
        families.require_readable(qwen_family(), value, backend)


@pytest.mark.parametrize("package,group", [(nemotron_h, 64), (qwen4_exp, 32)])
@pytest.mark.parametrize("bits", [2, 3, 5, 6, 8])
def test_moe_families_still_refuse_other_bit_widths_on_cuda(package, group, bits):
    family = families.Family(package.MODEL_TYPES[0], package.TITLE, package.__name__, True)
    with pytest.raises(ValueError, match="4-bit"):
        families.require_readable(family, configuration(bits, group), "cuda")


@pytest.mark.parametrize("package,group", [(nemotron_h, 64), (qwen4_exp, 32)])
@pytest.mark.parametrize("bits", [2, 3, 5, 6, 8])
def test_moe_family_metal_preflight_still_refuses_other_widths(tmp_path, package, group, bits):
    (tmp_path / "config.json").write_text(json.dumps(configuration(bits, group)))
    with pytest.raises(ValueError, match="4-bit"):
        package.check(tmp_path)


@pytest.mark.parametrize("lane_kernels", ["auto", "on", "off"])
def test_cli_passes_kernel_choice_to_family_loader_without_allocating(tmp_path, monkeypatch, lane_kernels):
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--lane-kernels", lane_kernels,
                                          "--no-drafts", "--no-update-check"])
    core, mlx = ModuleType("mlx.core"), ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    calls = []

    class AtLoader(Exception):
        pass

    def load(path, **options):
        calls.append((path, options))
        raise AtLoader

    family = NS(package=NS(load=load), title="Qwen dense", model_type="qwen3_5")
    with pytest.raises(AtLoader):
        cli._serve_mlx(args, family, tmp_path, 4096, (), 1024**3)
    assert calls[0][0] == tmp_path and calls[0][1]["lane_kernels"] == lane_kernels
    assert calls[0][1]["drafter"] == ""
