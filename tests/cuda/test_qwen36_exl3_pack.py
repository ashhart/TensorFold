"""Qwen3.6 MoE on CUDA: EXL3 packs pass check and admission; MTP loads from a side file or one tensor an expert."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from safetensors.torch import save_file  # noqa: E402

from tensorfold import families  # noqa: E402
from tensorfold.cuda.capacity import Geometry  # noqa: E402
from tensorfold.families import qwen3_5_moe  # noqa: E402
from tensorfold.families.qwen3_5.cuda import exl3_load  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.weights import load_mtp  # noqa: E402

D, NI, E, TOP, HEADS, HD = 256, 64, 4, 2, 2, 128

# UnstableLlama/Qwen3.6-35B-A3B-exl3-4.00bpw's text config: MoE widths only, no dense intermediate_size
TEXT = {"model_type": "qwen3_5_moe_text", "hidden_size": 2048, "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512, "num_attention_heads": 16, "head_dim": 256,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_num_value_heads": 32,
        "linear_value_head_dim": 128}
EXL3 = {"model_type": "qwen3_5_moe", "text_config": TEXT,
        "quantization_config": {"quant_method": "exl3", "version": "0.0.6", "bits": 4.0, "head_bits": 6}}


def test_cuda_takes_an_exl3_pack(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    (tmp_path / "config.json").write_text(json.dumps(EXL3))
    qwen3_5_moe.check(tmp_path)
    families.require_readable(families.families()["qwen3_5_moe"], EXL3, "cuda")


def test_exl3_admission_sizes_a_moe_text_config():
    with_workspace, _ = exl3_load.admission(lambda text: Geometry(lambda slots: 0, 0))
    widest = 2 * TEXT["num_attention_heads"] * TEXT["head_dim"]        # attention's q and gate, wider than experts
    assert with_workspace(TEXT).bytes_at(0) >= 2 * TEXT["hidden_size"] * widest


def _layer(fused: bool) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(7)

    def r(*shape):
        return (torch.randn(*shape, generator=gen) * 0.05).bfloat16()

    p, m = "mtp.layers.0.", "mtp.layers.0.mlp."
    t = {"mtp.fc.weight": r(D, 2 * D), "mtp.pre_fc_norm_embedding.weight": r(D), "mtp.pre_fc_norm_hidden.weight": r(D),
         "mtp.norm.weight": r(D), p + "input_layernorm.weight": r(D), p + "post_attention_layernorm.weight": r(D),
         p + "self_attn.q_proj.weight": r(2 * HEADS * HD, D), p + "self_attn.k_proj.weight": r(HD, D),
         p + "self_attn.v_proj.weight": r(HD, D), p + "self_attn.o_proj.weight": r(D, HEADS * HD),
         p + "self_attn.q_norm.weight": r(HD), p + "self_attn.k_norm.weight": r(HD),
         m + "gate.weight": r(E, D), m + "shared_expert_gate.weight": r(1, D),
         m + "shared_expert.gate_proj.weight": r(NI, D), m + "shared_expert.up_proj.weight": r(NI, D),
         m + "shared_expert.down_proj.weight": r(D, NI)}
    gate_up, down = r(E, 2 * NI, D), r(E, D, NI)
    if fused:
        t |= {m + "experts.gate_up_proj": gate_up, m + "experts.down_proj": down}
    else:
        for e in range(E):
            t |= {m + f"experts.{e}.gate_proj.weight": gate_up[e, :NI].clone(),
                  m + f"experts.{e}.up_proj.weight": gate_up[e, NI:].clone(),
                  m + f"experts.{e}.down_proj.weight": down[e].clone()}
    return t


def _indexed(path, tensors):
    path.mkdir()
    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: "model.safetensors" for name in tensors}}))
    return path


def _weights(quant: str):
    return SimpleNamespace(config=SimpleNamespace(hidden=D, experts=E, top_k=TOP), quant=quant, head=None)


def _same_experts(a, b):
    assert torch.equal(a.moe.router, b.moe.router)
    for name in ("up", "down", "up_scale", "down_scale"):
        assert torch.equal(getattr(a.moe.experts, name), getattr(b.moe.experts, name)), name


def test_mtp_experts_one_tensor_an_expert_load_as_the_fused_layout(tmp_path):
    fused = load_mtp(_indexed(tmp_path / "fused", _layer(True)), _weights("nvfp4"))
    split = load_mtp(_indexed(tmp_path / "split", _layer(False)), _weights("nvfp4"))
    _same_experts(split, fused)


def test_an_exl3_pack_drafts_with_its_bf16_mtp_side_file(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    save_file(_layer(True), str(pack / "mtp.safetensors"))
    mtp = load_mtp(pack, _weights("exl3"))
    assert mtp is not None
    _same_experts(mtp, load_mtp(_indexed(tmp_path / "fused", _layer(True)), _weights("nvfp4")))
