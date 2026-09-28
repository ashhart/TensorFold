"""Affine metadata and packed tensor headers are checked without loading weights or frameworks."""

from __future__ import annotations

import copy
import sys

import pytest

from tensorfold.quantization import AffineSpec, resolve_affine, validate_shapes


@pytest.fixture(autouse=True)
def block_accelerators(monkeypatch):
    for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "torch", "triton"):
        monkeypatch.setitem(sys.modules, name, None)


def config(**entries):
    return {"quantization": {"bits": 8, "group_size": 128, **entries}}


def test_global_quantization_precedes_legacy_configuration():
    value = config()
    value["quantization_config"] = {"quant_method": "gptq", "bits": 4, "group_size": 32}
    assert resolve_affine(value) == AffineSpec(8, 128)


def test_legacy_and_nested_quantization_are_available_when_top_level_is_absent():
    assert resolve_affine({"quantization_config": {"bits": 6, "group_size": 32}}) == AffineSpec(6, 32)
    assert resolve_affine({"text_config": config(bits=5, group_size=64)}) == AffineSpec(5, 64)
    assert resolve_affine({"text_config": {"quantization_config": {"bits": 3, "group_size": 32}}}) == AffineSpec(3, 32)


def test_top_level_block_precedes_nested_block():
    value = config()
    value["text_config"] = config(bits=2, group_size=32)
    assert resolve_affine(value) == AffineSpec(8, 128)


def test_missing_global_group_uses_affine_default_but_bits_are_required():
    assert resolve_affine({"quantization": {"bits": 8}}) == AffineSpec(8, 64)
    with pytest.raises(ValueError):
        resolve_affine({"quantization": {"group_size": 64}})


def test_absent_quantization_returns_no_affine_spec():
    assert resolve_affine({}) is None
    assert resolve_affine({"text_config": {"hidden_size": 128}}, "model.layers.0.q_proj") is None


@pytest.mark.parametrize("entry,expected", [
    ({"bits": 6}, AffineSpec(6, 64)),
    ({"group_size": 32}, AffineSpec(4, 32)),
    ({"mode": "affine"}, AffineSpec(4, 64)),
    ({"bits": 3, "group_size": 128}, AffineSpec(3, 128)),
    (True, AffineSpec(8, 128)),
    (False, None),
    ({}, None),
])
def test_module_overrides_match_mlx_predicate_semantics(entry, expected):
    value = config(**{"model.layers.0.self_attn.q_proj": entry})
    assert resolve_affine(value, "model.layers.0.self_attn.q_proj") == expected
    assert resolve_affine(value, "model.layers.0.self_attn.k_proj") == AffineSpec(8, 128)


@pytest.mark.parametrize("prefix", ["", "model.", "language_model.", "language_model.model.",
                                    "model.language_model."])
@pytest.mark.parametrize("query_prefix", ["", "model.", "language_model.model.", "model.language_model."])
def test_model_prefix_aliases_resolve_the_same_module(prefix, query_prefix):
    leaf = "layers.0.self_attn.q_proj"
    value = config(**{prefix + leaf: {"bits": 5, "group_size": 32}})
    assert resolve_affine(value, query_prefix + leaf) == AffineSpec(5, 32)


@pytest.mark.parametrize("prefix", ["", "model.", "language_model.", "language_model.model.",
                                    "model.language_model."])
def test_head_prefix_aliases_resolve_without_matching_other_heads(prefix):
    value = config(**{prefix + "lm_head": {"bits": 4, "group_size": 32},
                      "mtp.lm_head": {"bits": 2, "group_size": 64}})
    assert resolve_affine(value, "lm_head") == AffineSpec(4, 32)
    assert resolve_affine(value, "mtp.lm_head") == AffineSpec(2, 64)


def test_equal_aliases_are_unambiguous():
    value = config(**{"model.lm_head": {"bits": 4}, "language_model.lm_head": {"bits": 4, "group_size": 64}})
    assert resolve_affine(value, "lm_head") == AffineSpec(4, 64)


@pytest.mark.parametrize("first,second", [({"bits": 4}, {"bits": 8}), (False, {"bits": 4}),
                                         ({}, True), ({"group_size": 32}, {"group_size": 64})])
def test_conflicting_aliases_are_refused_even_when_one_matches_exactly(first, second):
    value = config(**{"model.lm_head": first, "language_model.lm_head": second})
    with pytest.raises(ValueError):
        resolve_affine(value, "model.lm_head")


def test_alias_matching_does_not_use_arbitrary_suffixes():
    value = config(**{"layers.1.self_attn.q_proj": {"bits": 2}, "mtp.layers.0.q_proj": {"bits": 3}})
    assert resolve_affine(value, "layers.0.self_attn.q_proj") == AffineSpec(8, 128)
    assert resolve_affine(value, "layers.0.q_proj") == AffineSpec(8, 128)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_supported_affine_combinations_have_matching_packed_headers(bits, group):
    spec = resolve_affine(config(bits=bits, group_size=group))
    assert spec == AffineSpec(bits, group)
    n, k = 7, group * 2
    assert validate_shapes((n, k * bits // 32), (n, 2), (n, 2), spec) == (n, k)


@pytest.mark.parametrize("bits", [3, 5, 6])
def test_non_power_of_two_widths_use_contiguous_bits_across_words(bits):
    # MLX stores 32 values in exactly bits words, without padding each uint32 word.
    spec = AffineSpec(bits, 32)
    assert validate_shapes((2, bits), (2, 1), (2, 1), spec) == (2, 32)
    padded_words = (32 + (32 // bits) - 1) // (32 // bits)
    assert padded_words != bits
    with pytest.raises(ValueError):
        validate_shapes((2, padded_words), (2, 1), (2, 1), spec)


@pytest.mark.parametrize("entry", [
    {"bits": 1}, {"bits": 7}, {"bits": 16}, {"bits": 0}, {"bits": -4}, {"bits": True}, {"bits": 4.5},
    {"group_size": 16}, {"group_size": 48}, {"group_size": 256}, {"group_size": 0}, {"group_size": True},
    {"mode": "mxfp4"}, {"mode": "nvfp4"}, {"mode": "mxfp8"}, {"mode": "unknown"},
])
def test_invalid_global_quantization_is_refused(entry):
    with pytest.raises(ValueError):
        resolve_affine(config(**entry))


@pytest.mark.parametrize("entry", [{"bits": 7}, {"bits": False}, {"group_size": 16}, {"mode": "mxfp4"}])
def test_invalid_override_is_not_hidden_by_valid_global_quantization(entry):
    with pytest.raises(ValueError):
        resolve_affine(config(**{"model.lm_head": entry}), "lm_head")


@pytest.mark.parametrize("method", ["gptq", "awq", "compressed-tensors", "bitnet", "mxfp4"])
def test_foreign_storage_is_not_reinterpreted_as_mlx_affine(method):
    with pytest.raises(ValueError):
        resolve_affine({"quantization_config": {"quant_method": method, "bits": 4, "group_size": 64}})


@pytest.mark.parametrize("weight,scales,biases", [
    ((2, 8), (2, 2), (2, 2)),
    ((2, 8), (3, 1), (3, 1)),
    ((2, 8), (2, 1), (2, 2)),
    ((2, 8), (2,), (2,)),
    ((1, 2, 8), (1, 2, 1), (1, 2, 1)),
    ((2,), (2, 1), (2, 1)),
    ((0, 8), (0, 1), (0, 1)),
    ((2, 0), (2, 0), (2, 0)),
    ((2, -8), (2, 1), (2, 1)),
    ((True, 8), (1, 1), (1, 1)),
    ((2, 8.0), (2, 1), (2, 1)),
])
def test_malformed_or_inconsistent_tensor_headers_are_refused(weight, scales, biases):
    with pytest.raises(ValueError):
        validate_shapes(weight, scales, biases, AffineSpec(4, 64))


def test_resolving_overrides_does_not_mutate_configuration():
    value = config(**{"model.layers.0.self_attn.q_proj": {"bits": 6}, "model.lm_head": False})
    original = copy.deepcopy(value)
    assert resolve_affine(value, "layers.0.self_attn.q_proj") == AffineSpec(6, 64)
    assert resolve_affine(value, "lm_head") is None
    assert value == original
