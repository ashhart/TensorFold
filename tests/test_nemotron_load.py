"""Nemotron's load gate: MLX 4-bit weights in groups of 32 or 64 everywhere, refused from the config and after load."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

from tensorfold import families  # noqa: E402
from tensorfold.families import nemotron_h  # noqa: E402


def _tiny(bits):
    """A 3-layer Nemotron-H (mamba, attention, experts) at small widths, quantized at ``bits`` in groups of 64."""

    from mlx_lm.models.nemotron_h import Model, ModelArgs

    args = ModelArgs(model_type="nemotron_h", vocab_size=512, hidden_size=256, intermediate_size=256,
                     num_hidden_layers=3, max_position_embeddings=1024, num_attention_heads=4, num_key_value_heads=2,
                     attention_bias=False, mamba_num_heads=8, mamba_head_dim=32, mamba_proj_bias=False,
                     ssm_state_size=16, conv_kernel=4, n_groups=2, mlp_bias=False, layer_norm_epsilon=1e-5,
                     use_bias=False, use_conv_bias=True, hybrid_override_pattern=["M", "*", "E"], head_dim=64,
                     moe_intermediate_size=128, moe_shared_expert_intermediate_size=128, n_group=1,
                     n_routed_experts=4, n_shared_experts=1, topk_group=1, num_experts_per_tok=2,
                     norm_topk_prob=True, routed_scaling_factor=2.5)
    mx.random.seed(3)
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("quant,named", [
    (None, r"none \(unquantized weights\)"),
    ({"bits": 8, "group_size": 64}, "MLX 8-bit, groups of 64"),
    ({"bits": 5, "group_size": 64}, "MLX 5-bit, groups of 64"),
    ({"bits": 4, "group_size": 128}, "MLX 4-bit, groups of 128"),
    ({"bits": 4, "group_size": 64, "backbone.layers.2.mixer.switch_mlp.fc1": {"bits": 8, "group_size": 64}},
     "with layers at 8-bit g64"),
    ({"bits": 4, "group_size": 64}, None),
    ({"bits": 4, "group_size": 32}, None),
    ({"bits": 4, "group_size": 64, "backbone.embeddings": {"bits": 8, "group_size": 64}}, None),   # a lookup
    ({"bits": 4, "group_size": 64, "lm_head": {"group_size": 64}}, None),              # MLX's default: 4 bits
])
def test_check_refuses_other_widths_before_the_download(tmp_path, quant, named):
    config = {"model_type": "nemotron_h", **({"quantization": quant} if quant else {})}
    (tmp_path / "config.json").write_text(json.dumps(config))
    if named is None:
        nemotron_h.check(tmp_path)
    else:
        with pytest.raises(ValueError, match=named):
            nemotron_h.check(tmp_path)


@pytest.mark.parametrize("bits", [4, 8])
def test_the_loaded_widths_are_named(bits):
    found = nemotron_h.unreadable(_tiny(bits))
    if bits == 4:
        assert found == {}
    else:
        assert found.pop("8-bit g64 expert tables") == 2 and list(found) == ["8-bit g64 linears"]


def test_load_refuses_weights_the_config_does_not_name(monkeypatch):
    from tensorfold.families.nemotron_h import model as nemotron_model

    tiny = _tiny(8)
    monkeypatch.setattr(families, "read_config", lambda path: {"quantization": {"bits": 4, "group_size": 64}})
    monkeypatch.setattr(nemotron_model, "load", lambda path, **_: (SimpleNamespace(model=tiny), "tokenizer"))
    with pytest.raises(SystemExit, match="2 8-bit g64 expert tables"):
        nemotron_h.load("unused")
