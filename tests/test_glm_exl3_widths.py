"""GLM-5.3-Flash on CUDA reads EXL3 routed experts at a width per tensor from their headers, and refuses by name the
groups its loader does not read (synthetic safetensors headers; no weights, no GPU)."""

from __future__ import annotations

import json
import struct

import pytest

from tensorfold.families import glm5_next
from tensorfold.families.glm5_next.cuda.exl3_widths import describe, widths

L = "model.language_model.layers."
D, I = 256, 128
QUANT = {"quant_method": "exl3", "codebook": "mcg", "scope": "glm53_routed_experts_only"}


def _group(prefix: str, k: int, n: int, bits: float, scales=("suh", "svh"), marker="mcg") -> dict:
    out = {f"{prefix}.trellis": ("I16", [k // 16, n // 16, int(16 * bits)])}
    out[f"{prefix}.{scales[0]}"] = ("F16", [k]) if scales[0] == "suh" else ("I16", [k // 16])
    out[f"{prefix}.{scales[1]}"] = ("F16", [n]) if scales[1] == "svh" else ("I16", [n // 16])
    out[f"{prefix}.{marker}"] = ("I32", [])
    return out


def _experts(widths_of: dict[tuple[int, int, str], float]) -> dict:
    tensors = {}
    for (layer, e, proj), bits in widths_of.items():
        k, n = (I, D) if proj == "down" else (D, I)
        tensors.update(_group(f"{L}{layer}.mlp.experts.{e}.{proj}_proj", k, n, bits))
    return tensors


def _write(folder, tensors: dict, bits=4) -> None:
    """config.json and one safetensors file of zeros whose header lists ``tensors`` (name: (dtype, shape))."""

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"model_type": "glm5_next",
                                                    "quantization_config": {**QUANT, "bits": bits}}))
    size = {"I16": 2, "F16": 2, "I32": 4, "BF16": 2}
    header, at = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = size[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + n]}
        at += n
    raw = json.dumps(header).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(at))


MIXED = {(3, e, p): (3 if p != "down" else 4) if e % 2 else 2 for e in range(4) for p in ("gate", "up", "down")}


def test_each_expert_tensor_keeps_its_own_width(tmp_path):
    _write(tmp_path, _experts(MIXED), bits=3.0)
    found = widths(tmp_path)
    assert found == {2: 6 * D * I, 3: 4 * D * I, 4: 2 * D * I}
    assert describe(found) == "2-bit 50.0%, 3-bit 33.3%, 4-bit 16.7%"


@pytest.mark.parametrize("bits", [4, 3, 3.3333, "mixed_k34_per_tensor"])
def test_check_takes_any_stated_bits_and_names_a_mix(tmp_path, capsys, bits):
    uniform = {k: 4 for k in MIXED}
    _write(tmp_path / "u", _experts(uniform), bits=bits)
    glm5_next.check(tmp_path / "u")
    assert "mixed widths" not in capsys.readouterr().out
    _write(tmp_path / "m", _experts(MIXED), bits=bits)
    glm5_next.check(tmp_path / "m")
    assert "EXL3 routed experts of mixed widths: 2-bit 50.0%, 3-bit 33.3%, 4-bit 16.7%" in capsys.readouterr().out


def test_config_alone_passes_before_the_weights_are_here(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm5_next", "quantization_config":
                                                      {**QUANT, "bits": "mixed_k34_per_tensor"}}))
    glm5_next.check(tmp_path)


@pytest.mark.parametrize("extra,why", [
    (_group(L + "3.self_attn.o_proj", D, D, 4), "routed experts only"),
    (_group(L + "3.mlp.experts.9.gate_proj", D, I, 4, marker="mul1"), "mul1 codebook"),
    (_group(L + "3.mlp.experts.9.gate_proj", D, I, 4, scales=("su", "sv")), "su/sv scales"),
    (_group(L + "3.mlp.experts.9.gate_proj", D, I, 2.5), "need the mul1 codebook"),
])
def test_groups_the_loader_does_not_read_are_refused_by_name(tmp_path, extra, why):
    _write(tmp_path, {**_experts(MIXED), **extra})
    with pytest.raises(ValueError, match=why) as refused:
        glm5_next.check(tmp_path)
    assert "recipe book" in str(refused.value) and next(iter(extra)).rpartition(".")[0] in str(refused.value)
