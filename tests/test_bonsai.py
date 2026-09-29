"""Ternary Bonsai 2: the pack contract, the pre-M5 forms (exact widening) and the row-exact rotation kernels."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorfold.families.bonsai import pack

mx = pytest.importorskip("mlx.core")


def config(**change):
    base = {"schema_version": 2, "model_type": "prism_hadamard_qwen35", "base_model_type": "qwen3_5",
            "hadamard_config": "hadamard.json", "gdn_activation_layout": "grouped",
            "components": {"text": True, "vision": True, "mtp": False},
            "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
            "text_config": {"model_type": "qwen3_5_text", "tie_word_embeddings": False},
            "modules": [{"path": "lm_head", "block": 1024, "embedding": False, "dtype": "float16"}]}
    base.update(change)
    return base


def hadamard(**change):
    base = {"prism.hadamard.version": 1, "prism.hadamard.block_size": 1024,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension", "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.gdn_v_grouped": True, "prism.hadamard.sign_widths": [1024, 2048],
            "prism.hadamard.sign_values": [1.0, -1.0] * 1536,
            "prism.hadamard.inverse_weight_names": ["language_model.model.embed_tokens.weight"]}
    base.update(change)
    return base


def test_the_packs_contract_is_read(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(config()))
    (tmp_path / "hadamard.json").write_text(json.dumps(hadamard()))
    got, transform = pack.contract(tmp_path)
    assert got["model_type"] == "prism_hadamard_qwen35" and set(pack.signs_by_width(transform)) == {1024, 2048}
    assert len(pack.fingerprint(tmp_path)) == 12


@pytest.mark.parametrize("change", [{"model_type": "qwen3_5"}, {"quantization": {"bits": 2, "group_size": 64}},
                                    {"quantization": {"bits": 4, "group_size": 128}},
                                    {"gdn_activation_layout": "interleaved"},
                                    {"components": {"text": True, "mtp": True}},
                                    {"text_config": {"model_type": "qwen3_5_text", "tie_word_embeddings": True}},
                                    {"modules": [{"path": "lm_head", "block": 512, "dtype": "float16"}]}])
def test_packs_outside_the_contract_are_refused(tmp_path, change):
    (tmp_path / "config.json").write_text(json.dumps(config(**change)))
    with pytest.raises(ValueError, match="cannot run this checkpoint"):
        pack.contract(tmp_path)


@pytest.mark.parametrize("change", [{"prism.hadamard.block_size": 512},
                                    {"prism.hadamard.transform": "randomized-hadamard"},
                                    {"prism.hadamard.sign_mode": "seeded"},
                                    {"prism.hadamard.gdn_v_grouped": False},
                                    {"prism.hadamard.sign_widths": [1024, 1024]},
                                    {"prism.hadamard.sign_values": [2.0] * 3072},
                                    {"prism.hadamard.inverse_weight_names": []}])
def test_transforms_outside_the_contract_are_refused(tmp_path, change):
    (tmp_path / "config.json").write_text(json.dumps(config()))
    (tmp_path / "hadamard.json").write_text(json.dumps(hadamard(**change)))
    with pytest.raises(ValueError, match="cannot run this checkpoint"):
        pack.contract(tmp_path)


def test_widening_keeps_every_code_and_value():
    rng = np.random.default_rng(18)
    codes = rng.integers(0, 4, (48, 512), dtype=np.uint32)
    words = np.bitwise_or.reduce(codes.reshape(48, -1, 16) << (2 * np.arange(16, dtype=np.uint32)), axis=-1)
    scales = rng.uniform(0.01, 0.1, (48, 4)).astype(np.float16)
    with mx.stream(mx.cpu):
        wide = np.array(pack.widen(mx.array(words), chunk=16))
        unpacked = ((wide[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(48, 512)
        np.testing.assert_array_equal(unpacked, codes)
        s, b = mx.array(scales), mx.array(-scales)
        two = mx.dequantize(mx.array(words), s, b, group_size=128, bits=2)
        four = mx.dequantize(mx.array(wide), mx.repeat(s, 2, axis=1), mx.repeat(b, 2, axis=1), group_size=64, bits=4)
        np.testing.assert_array_equal(np.array(two), np.array(four))


def test_every_form_holds_the_packs_values():
    rng = np.random.default_rng(19)
    codes = rng.integers(0, 3, (32, 1024), dtype=np.uint32)
    words = np.bitwise_or.reduce(codes.reshape(32, -1, 16) << (2 * np.arange(16, dtype=np.uint32)), axis=-1)
    scales = rng.uniform(0.01, 0.1, (32, 8)).astype(np.float16)
    with mx.stream(mx.cpu):
        want = np.array(mx.dequantize(mx.array(words), mx.array(scales), mx.array(-scales), group_size=128, bits=2)
                        .astype(mx.float32))
        for form, bits, group, dtype in (("lanes", 2, 64, mx.bfloat16), ("widened", 4, 64, mx.bfloat16),
                                         ("packed", 2, 128, mx.float16)):
            q = pack._linear(mx.array(words), mx.array(scales), mx.array(-scales), form)
            assert (q.bits, q.group_size, q.scales.dtype) == (bits, group, dtype)
            got = mx.dequantize(q.weight, q.scales, q.biases, group_size=group, bits=bits).astype(mx.float32)
            rounded = scales.astype(np.float32) if form == "packed" else np.array(
                mx.array(scales).astype(mx.bfloat16).astype(mx.float32))
            expect = (codes.astype(np.float32) - 1) * np.repeat(rounded, 128, axis=1)
            np.testing.assert_array_equal(np.array(got), expect)
        assert np.abs(want - expect).max() < 1e-3


def test_widened_codes_are_chosen_only_where_they_fit(tmp_path):
    import struct

    (tmp_path / "config.json").write_text(json.dumps(config()))
    (tmp_path / "hadamard.json").write_text(json.dumps(hadamard()))
    header = {"language_model.lm_head.weight": {"dtype": "U32", "shape": [4, 64], "data_offsets": [0, 1000]},
              "language_model.lm_head.scales": {"dtype": "F16", "shape": [4, 8], "data_offsets": [1000, 1100]},
              "language_model.lm_head.biases": {"dtype": "F16", "shape": [4, 8], "data_offsets": [1100, 1200]},
              "language_model.lm_head.signs": {"dtype": "F32", "shape": [1024], "data_offsets": [1200, 1300]},
              "language_model.model.norm.weight": {"dtype": "F32", "shape": [25], "data_offsets": [1300, 1400]},
              "vision_tower.x": {"dtype": "F16", "shape": [50], "data_offsets": [1400, 1500]}}
    raw = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert pack.sizes(tmp_path) == (1400, 1200)
    assert pack.pre_m5_form(tmp_path, 2600 + pack.ROOM) == "widened"
    assert pack.pre_m5_form(tmp_path, 2599 + pack.ROOM) == "packed"


def fwht_ref(x: np.ndarray) -> np.ndarray:
    """Sylvester-order Walsh-Hadamard over 1024-blocks in float64, normalized."""

    x = x.astype(np.float64).reshape(*x.shape[:-1], -1, 1024).copy()
    h = 1
    while h < 1024:
        y = x.reshape(*x.shape[:-1], 1024 // (2 * h), 2, h)
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y[..., 0, :], y[..., 1, :] = a + b, a - b
        h *= 2
    return (x / 32.0).reshape(*x.shape[:-2], -1)


def metal():
    try:
        return mx.metal.is_available()
    except Exception:  # noqa: BLE001 - no Metal: the kernel tests do not apply
        return False


needs_metal = pytest.mark.skipif(not metal(), reason="needs a Metal GPU")


@needs_metal
@pytest.mark.parametrize("k", [1024, 5120, 6144])
def test_rotation_matches_the_transform_and_every_row_is_its_own(k):
    from tensorfold.kernels.qwen.prism.v1 import rotate

    rng = np.random.default_rng(k)
    x = mx.array(rng.normal(0, 1, (40, k)).astype(np.float32)).astype(mx.bfloat16)
    signs = mx.array(rng.choice([-1.0, 1.0], k).astype(np.float32))
    y = rotate.rotate_rows(x, signs)
    want = fwht_ref(np.array(x.astype(mx.float32)) * np.array(signs))
    np.testing.assert_allclose(np.array(y.astype(mx.float32)), want, rtol=1e-2, atol=1e-2)
    ref = np.array(mx.hadamard_transform((x.astype(mx.float32) * signs).reshape(-1, 1024), scale=1 / 32))
    np.testing.assert_allclose(np.array(y.astype(mx.float32)).reshape(-1, 1024), ref, rtol=1e-2, atol=1e-2)
    for rows in (1, 2, 3, 17, 33):
        part = rotate.rotate_rows(x[:rows], signs)
        assert bool(mx.array_equal(part, y[:rows]).item())
    one = [rotate.rotate_rows(x[i:i + 1], signs) for i in range(40)]
    assert bool(mx.array_equal(mx.concatenate(one), y).item())


@needs_metal
def test_embedding_rows_rotate_back_and_do_not_depend_on_the_batch():
    from tensorfold.kernels.qwen.prism.v1 import rotate

    rng = np.random.default_rng(7)
    vocab, k = 64, 2048
    codes = rng.integers(0, 3, (vocab, k), dtype=np.uint32)
    words = np.bitwise_or.reduce(codes.reshape(vocab, -1, 16) << (2 * np.arange(16, dtype=np.uint32)), axis=-1)
    scales = rng.uniform(0.01, 0.1, (vocab, k // 128)).astype(np.float16)
    signs = rng.choice([-1.0, 1.0], k).astype(np.float32)
    ids = mx.array([5, 0, 63, 5, 17], dtype=mx.uint32)
    out = rotate.embed_rows(ids, mx.array(words), mx.array(scales), mx.array(-scales), mx.array(signs), 128)
    rows = (codes.astype(np.float64) - 1) * np.repeat(scales.astype(np.float64), 128, axis=1)
    want = fwht_ref(rows[np.array(ids)]) * signs
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), want, rtol=1e-2, atol=1e-3)
    alone = rotate.embed_rows(ids[2:3], mx.array(words), mx.array(scales), mx.array(-scales), mx.array(signs), 128)
    assert bool(mx.array_equal(alone[0], out[2]).item()) and bool(mx.array_equal(out[0], out[3]).item())


@needs_metal
def test_dense_rows_are_row_exact():
    from tensorfold.kernels.qwen.prism.v1 import rotate

    rng = np.random.default_rng(3)
    x = mx.array(rng.normal(0, 1, (33, 5120)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.normal(0, 0.02, (48, 5120)).astype(np.float32))
    y = rotate.dense_rows(x[None], w)[0]
    np.testing.assert_allclose(np.array(y.astype(mx.float32)),
                               np.array(x.astype(mx.float32)) @ np.array(w).T, rtol=2e-2, atol=2e-2)
    for rows in (1, 2, 17):
        assert bool(mx.array_equal(rotate.dense_rows(x[:rows], w), y[:rows]).item())


@needs_metal
def test_a_projection_with_the_packs_fp16_scales_returns_its_input_dtype():
    import mlx.nn as nn

    from tensorfold.families.bonsai.modules import RotatedLinear

    inner = nn.QuantizedLinear(1024, 32, bias=False, group_size=128, bits=2)
    inner.scales, inner.biases = inner.scales.astype(mx.float16), inner.biases.astype(mx.float16)
    x = mx.ones((1, 3, 1024), dtype=mx.bfloat16)
    assert RotatedLinear(inner, mx.ones((1024,)))(x).dtype == mx.bfloat16


@needs_metal
def test_fp32_gates_keep_row_bits_in_decode_windows_and_batch_prompt_chunks():
    from tensorfold.families.bonsai.modules import ROW_EXACT_ROWS, RowDense

    from tensorfold.kernels.qwen.prism.v1 import rotate

    rng = np.random.default_rng(5)
    x = mx.array(rng.normal(0, 1, (ROW_EXACT_ROWS + 72, 1024)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.normal(0, 0.02, (48, 1024)).astype(np.float32))
    gate = RowDense(w)
    window = gate(x[:ROW_EXACT_ROWS])
    assert bool(mx.array_equal(window, rotate.dense_rows(x[:ROW_EXACT_ROWS], w)).item())
    chunk = gate(x)
    np.testing.assert_allclose(np.array(chunk.astype(mx.float32)), np.array(x.astype(mx.float32)) @ np.array(w).T,
                               rtol=2e-2, atol=2e-2)
    assert chunk.dtype == mx.bfloat16


def test_the_row_decoder_runs_a_projection_that_brings_its_own_rows():
    from types import SimpleNamespace

    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    own = SimpleNamespace(project_rows=lambda x: ("own", x))
    assert row_matmul.project(own, "rows") == ("own", "rows")
    assert row_matmul.logits(own, "rows") == ("own", "rows")


def test_the_draft_head_reads_the_matmul_under_a_rotated_head():
    import mlx.nn as nn

    from tensorfold.drafters.dflash_drafter import DFlashDrafter
    from tensorfold.families.bonsai.modules import RotatedLinear

    inner = nn.QuantizedLinear(64, 32, bias=False, group_size=64, bits=4)
    drafter = DFlashDrafter.__new__(DFlashDrafter)
    drafter.model = type("M", (), {})()
    drafter.model.lm_head = RotatedLinear(inner, mx.ones((64,)))
    assert drafter._matmul_head() is inner
    drafter.model.lm_head = inner
    assert drafter._matmul_head() is inner
