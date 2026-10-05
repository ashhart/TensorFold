"""Llama discovery and checkpoint refusal must work without accelerator imports."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


def base_config():
    return {"model_type": "llama", "hidden_size": 128, "intermediate_size": 256,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
            "rms_norm_eps": 1e-5, "vocab_size": 64, "tie_word_embeddings": False,
            "attention_bias": True, "mlp_bias": True}


def test_llama_is_discovered_without_importing_accelerators(tmp_path):
    config = {"model_type": "llama", "hidden_size": 1536, "intermediate_size": 4096,
              "num_hidden_layers": 32, "num_attention_heads": 12, "num_key_value_heads": 2,
              "rms_norm_eps": 1e-5, "vocab_size": 32768, "tie_word_embeddings": False,
              "attention_bias": True, "mlp_bias": True,
              "rope_parameters": {"rope_type": "default", "rope_theta": 1000000},
              "quantization": {"bits": 8, "group_size": 64, "mode": "affine"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    code = """
import sys
from tensorfold import families
family = families.detect(sys.argv[1])
assert family.module == 'tensorfold.families.llama' and family.lanes
family.package.check(sys.argv[1])
assert not any(n.split('.')[0] in ('mlx', 'mlx_lm', 'torch', 'triton') for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("quant", [
    {"bits": 4, "group_size": 64},
    {"bits": 8, "group_size": 128},
    {"bits": 8, "group_size": 64, "lm_head": True},
])
def test_hub_refuses_unsupported_affine_layout_before_download(quant):
    from tensorfold import families

    config = dict(base_config(), quantization=quant)
    with pytest.raises(ValueError, match="8-bit|lm_head"):
        families.require_readable(families.families()["llama"], config, "mlx")


def test_transformers5_rope_is_normalized_without_changing_input():
    from tensorfold.families.llama.config import normalize

    config = dict(base_config(), rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    normalized = normalize(config)
    assert normalized["rope_theta"] == 1e6
    assert "rope_theta" not in config
    assert normalize(dict(config, rope_theta=1e6))["rope_theta"] == 1e6
    assert normalize(base_config())["rope_theta"] == 10000


@pytest.mark.parametrize("change,reason", [
    ({"rope_parameters": {"rope_type": "linear", "rope_theta": 1e6}}, "RoPE"),
    ({"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}, "rope_theta": 10000}, "conflict"),
    ({"rope_scaling": {"type": "linear", "factor": 2}}, "RoPE"),
    ({"rope_traditional": True}, "RoPE"),
    ({"partial_rotary_factor": 0.5}, "RoPE"),
    ({"rope_theta": float("nan")}, "RoPE"),
    ({"tie_word_embeddings": True}, "untied"),
    ({"sliding_window": 4096}, "full attention"),
    ({"layer_types": ["full_attention", "sliding_attention"]}, "full attention"),
    ({"hidden_act": "gelu"}, "SwiGLU"),
    ({"num_attention_heads": 3}, "geometry"),
    ({"num_key_value_heads": 3}, "geometry"),
    ({"head_dim": 63}, "geometry"),
    ({"attention_bias": "true"}, "boolean"),
    ({"quantization": {"bits": 4, "group_size": 64}}, "8-bit"),
    ({"quantization": {"bits": 8, "group_size": 128}}, "8-bit"),
    ({"quantization": {"bits": 4, "group_size": 16, "mode": "nvfp4"}}, "8-bit"),
    ({"quantization": {"bits": 8, "group_size": 64, "lm_head": {"bits": 8}}}, "lm_head"),
    ({"quantization": {"bits": 8, "group_size": 64, "model.embed_tokens": {"bits": 8}}}, "embed_tokens"),
    ({"torch_dtype": "float16"}, "bf16"),
    ({"quantize_activations": True}, "unsupported"),
    ({"model_file": "custom.py"}, "unsupported"),
])
def test_unsupported_config_is_refused_before_weights(tmp_path, change, reason):
    from tensorfold.families.llama import check

    (tmp_path / "config.json").write_text(json.dumps(dict(base_config(), **change)))
    with pytest.raises(ValueError, match=reason):
        check(tmp_path)


@pytest.mark.parametrize("key", ["quantization", "quantization_config"])
@pytest.mark.parametrize("block", [{}, [], False, True, 8, "affine", [["bits", 8]]])
def test_malformed_quantization_declarations_fail_at_config_gate(tmp_path, key, block):
    from tensorfold.families.llama import check

    (tmp_path / "config.json").write_text(json.dumps(dict(base_config(), **{key: block})))
    with pytest.raises(ValueError, match="8-bit"):
        check(tmp_path)


@pytest.mark.parametrize("key", ["rope_parameters", "rope_scaling"])
@pytest.mark.parametrize("block", [[], False, True, 0, "", "default"])
def test_malformed_rope_declarations_fail_at_config_gate(tmp_path, key, block):
    from tensorfold.families.llama import check

    (tmp_path / "config.json").write_text(json.dumps(dict(base_config(), **{key: block})))
    with pytest.raises(ValueError, match="RoPE"):
        check(tmp_path)


@pytest.mark.parametrize("change", [
    {"bits": 8.0}, {"bits": 8, "group_size": 64.0},
    {"bits": 8, "model.layers.0.self_attn.q_proj": []},
    {"bits": 8, "model.layers.0.self_attn.q_proj": "8bit"},
    {"bits": 8, "model.layers.0.self_attn.q_proj": 8},
])
def test_quantization_declaration_types_are_not_coerced(change):
    from tensorfold.families.llama.config import normalize

    with pytest.raises(ValueError, match="8-bit"):
        normalize(dict(base_config(), quantization=change))


def test_quantized_head_in_weight_index_is_refused(tmp_path):
    from tensorfold.families.llama import check

    (tmp_path / "config.json").write_text(json.dumps(base_config()))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.scales": "a"}}))
    with pytest.raises(ValueError, match="lm_head"):
        check(tmp_path)


@pytest.mark.parametrize("name,dtype,shape,reason", [
    ("lm_head.weight", "F16", [64, 128], "bf16"),
    ("model.embed_tokens.weight", "F32", [64, 128], "bf16"),
    ("lm_head.weight", "BF16", [63, 128], "shape"),
    ("model.embed_tokens.scales", "BF16", [64, 2], "embed_tokens"),
    ("model.layers.0.self_attn.q_proj.weight", "F16", [128, 128], "bf16"),
])
def test_safetensor_header_refuses_unsupported_weights_without_backend(tmp_path, name, dtype, shape, reason):
    import struct

    from tensorfold.families.llama import check

    header = {"lm_head.weight": {"dtype": "BF16", "shape": [64, 128], "data_offsets": [0, 0]},
              "model.embed_tokens.weight": {"dtype": "BF16", "shape": [64, 128], "data_offsets": [0, 0]}}
    header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
    data = json.dumps(header).encode()
    (tmp_path / "config.json").write_text(json.dumps(base_config()))
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(data)) + data)
    with pytest.raises(ValueError, match=reason):
        check(tmp_path)


@pytest.mark.parametrize("change", [
    {"num_key_value_heads": 0}, {"head_dim": 0}, {"pretraining_tp": 2},
    {"use_sliding_window": True}, {"num_local_experts": 4}, {"qk_norm": True},
    {"layer_types": []}, {"rope_theta": True}, {"attention_multiplier": 2},
    {"residual_multiplier": 2}, {"embedding_multiplier": 2}, {"tie_word_embeddings": 0},
    {"rope_parameters": {"rope_type": "default", "rope_theta": 1e6, "factor": 2}},
])
def test_other_unsupported_features_are_not_silently_ignored(tmp_path, change):
    from tensorfold.families.llama import check

    (tmp_path / "config.json").write_text(json.dumps(dict(base_config(), **change)))
    with pytest.raises(ValueError):
        check(tmp_path)


def test_partial_shard_headers_do_not_require_unavailable_head(tmp_path):
    import struct

    from tensorfold.families.llama import check

    (tmp_path / "config.json").write_text(json.dumps(base_config()))
    entries = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [64, 128], "data_offsets": [0, 0]}}
    data = json.dumps(entries).encode()
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(struct.pack("<Q", len(data)) + data)
    check(tmp_path)  # the CLI must still be able to download the second shard


def test_quantized_head_packed_tensor_is_refused_even_without_scales(tmp_path):
    import struct

    from tensorfold.families.llama import check

    (tmp_path / "config.json").write_text(json.dumps(dict(base_config(), quantization={"bits": 8})))
    header = {"lm_head.weight": {"dtype": "U32", "shape": [64, 32], "data_offsets": [0, 0]}}
    data = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(data)) + data)
    with pytest.raises(ValueError, match="unquantized"):
        check(tmp_path)


def test_quantized_norm_weight_is_refused_in_header(tmp_path):
    import struct

    from tensorfold.families.llama import check

    (tmp_path / "config.json").write_text(json.dumps(dict(base_config(), quantization={"bits": 8})))
    header = {"model.norm.weight": {"dtype": "U32", "shape": [128], "data_offsets": [0, 0]}}
    data = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(data)) + data)
    with pytest.raises(ValueError, match="shape"):
        check(tmp_path)


def test_scale_only_partial_shard_can_be_completed(tmp_path):
    import struct

    from tensorfold.families.llama import check

    (tmp_path / "config.json").write_text(json.dumps(dict(base_config(), quantization={"bits": 8})))
    header = {"model.layers.0.self_attn.q_proj.scales": {"dtype": "BF16", "shape": [128, 2], "data_offsets": [0, 0]}}
    data = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(data)) + data)
    check(tmp_path)


@pytest.mark.parametrize("context", [1, 32, 64, 65, 66, 256, 512, 575])
def test_native_context_too_small_for_server_probes_is_refused_early(context):
    from tensorfold.families.llama.config import normalize

    with pytest.raises(ValueError, match="at least 576"):
        normalize(dict(base_config(), max_position_embeddings=context))


def test_marker_context_512_is_refused_before_weights(tmp_path, monkeypatch):
    from tensorfold.families.llama import config, load

    (tmp_path / "config.json").write_text(json.dumps(dict(base_config(), max_position_embeddings=512)))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for m in messages %}{{ '<|message|>' + m['role'] + m['content'] }}{% endfor %}"}))

    def never(*args, **kwargs):
        pytest.fail("unsupported context reached weight headers")

    monkeypatch.setattr(config, "check_headers", never)
    with pytest.raises(ValueError, match="at least 576"):
        load(tmp_path)


def test_default_quant_group_is_explicit_in_normalized_config():
    from tensorfold.families.llama.config import normalize

    config = dict(base_config(), quantization={"bits": 8})
    assert normalize(config)["quantization"]["group_size"] == 64
    assert config["quantization"] == {"bits": 8}


@pytest.mark.parametrize("part,shape", [("weight", [128, 16]), ("scales", [128, 1]), ("biases", [128, 1])])
def test_header_gate_checks_actual_affine_geometry(tmp_path, part, shape):
    import struct

    from tensorfold.families.llama import check

    config = dict(base_config(), quantization={"bits": 8, "group_size": 64})
    prefix = "model.layers.0.self_attn.q_proj"
    headers = {prefix + "." + key: {"dtype": dtype, "shape": dims, "data_offsets": [0, 0]}
               for key, dtype, dims in [("weight", "U32", [128, 32]), ("scales", "BF16", [128, 2]),
                                        ("biases", "BF16", [128, 2])]}
    headers[prefix + "." + part]["shape"] = shape
    data = json.dumps(headers).encode()
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(data)) + data)
    with pytest.raises(ValueError, match="shape"):
        check(tmp_path)
