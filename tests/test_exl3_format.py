"""TensorFold's generic EXL3 format module (``cuda/exl3/format.py``) on the CPU: the decoder against a bit-by-bit one
written from the format description for every codebook and width, the codebooks against their definitions, the
layer's weight against its rotated forward, tensor-group parsing and the checkpoint inspector on a synthetic
checkpoint. No GPU and no torch needed.

Against ExLlamaV3's own ``reconstruct`` (every codebook and width, and real MiMo-V2.6 tensors of every width it
holds), see ``tests/cuda/test_exl3_linear.py``."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import inspect as exl3_inspect

WIDTHS = [(cb, b) for cb in fmt.CODEBOOKS for b in fmt.BITS if float(b).is_integer() or cb == "mul1"]


def _slow_value(s: int, codebook: str) -> np.float16:
    if codebook == "mul1":
        x = (s * 0x83DCD12D) & 0xFFFFFFFF
        h = 1024 + sum((x >> (8 * i)) & 255 for i in range(4))
        scale = float(np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0])
        bias = float(np.array([0xC931], dtype=np.uint16).view(np.float16)[0])
        return np.float16(h * scale + bias)                  # float64 is exact here: one rounding, like the fma
    x = (s * 0xCBAC1FED) & 0xFFFFFFFF if codebook == "mcg" else (s * 89226354 + 64248484) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = np.array([x & 0xFFFF], dtype=np.uint16).view(np.float16)[0]
    hi = np.array([x >> 16], dtype=np.uint16).view(np.float16)[0]
    return np.float16(np.float64(lo) + np.float64(hi))


def _slow_tile(tile_words: np.ndarray, bits: float, codebook: str) -> np.ndarray:
    """One tile, bit by bit: int16 words -> 256 fp16 values in stream order. Half-integer widths: value p takes
    KA new bits, plus one more at odd p (the prefix sum of the steps gives where each window ends)."""

    stream = []
    for j in range(0, len(tile_words), 2):
        word = (int(tile_words[j]) & 0xFFFF) | ((int(tile_words[j + 1]) & 0xFFFF) << 16)
        stream += [(word >> (31 - b)) & 1 for b in range(32)]
    n = len(stream)
    assert n == 256 * bits
    out, end = [], 0
    for p in range(256):
        end += int(bits) + (1 if (not float(bits).is_integer() and p % 2 == 1) else 0)
        s = 0
        for b in range(end - 16, end):
            s = (s << 1) | stream[b % n]
        out.append(_slow_value(s, codebook))
    assert end == n
    return np.array(out, dtype=np.float16)


def _trellis(kt: int, nt: int, bits: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-2**15, 2**15, size=(kt, nt, fmt.tile_words(bits)), dtype=np.int64).astype(np.int16)


def test_codebooks_are_their_definitions():
    rng = np.random.default_rng(0)
    for cb in fmt.CODEBOOKS:
        table = fmt.codebook(cb)
        assert table.dtype == np.float16 and table.shape == (65536,) and np.isfinite(table).all()
        assert 0.5 < float(np.std(table.astype(np.float64))) < 2.0        # unit-scale Gaussian-like values
        for s in [0, 1, 0xFFFF, *rng.integers(0, 65536, size=200).tolist()]:
            assert table[s].view(np.int16) == _slow_value(int(s), cb).view(np.int16), (cb, s)
    with pytest.raises(ValueError):
        fmt.codebook("2mad")


@pytest.mark.parametrize("codebook,bits", WIDTHS)
def test_decoder_matches_bit_by_bit(codebook, bits):
    t = _trellis(2, 2, bits, seed=int(10 * bits) + len(codebook))
    wq = fmt.unpack(t, bits, codebook)
    rows, cols = fmt.tile_positions()
    for k in range(2):
        for n in range(2):
            slow = _slow_tile(t[k, n], bits, codebook)
            tile = wq[16 * k:16 * k + 16, 16 * n:16 * n + 16]
            assert np.array_equal(tile[rows, cols].view(np.int16), slow.view(np.int16)), (k, n)


def test_stream_ends_and_widths():
    for bits in fmt.BITS:
        e = fmt.stream_ends(bits)
        assert e[-1] == 256 * bits and np.all(np.diff(e) >= 1)
        steps = np.diff(np.concatenate([[0], e]))
        if float(bits).is_integer():
            assert set(steps.tolist()) == {int(bits)}
        else:
            assert steps[0::2].tolist() == [int(bits)] * 128 and steps[1::2].tolist() == [int(bits) + 1] * 128
        assert fmt.bits_of((4, 4, fmt.tile_words(bits))) == bits
    for bad in (0, 9, 4.5, 2.25):
        with pytest.raises(ValueError):
            fmt.check_bits(bad)
    rows, cols = fmt.tile_positions()
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(r, c) for r in range(16) for c in range(16)]


def test_the_glm_reference_is_the_4_bit_mcg_case():
    torch = pytest.importorskip("torch")
    from tensorfold.families.glm5_next.cuda import exl3 as glm

    t = _trellis(3, 2, 4, seed=5)
    assert np.array_equal(glm.unpack(torch.from_numpy(t)).numpy().view(np.int16),
                          fmt.unpack(t, 4, "mcg").view(np.int16))


def test_weight_and_rotated_forward_agree():
    K, N = 256, 384
    rng = np.random.default_rng(3)
    for cb, bits in (("mul1", 2.5), ("mcg", 3), ("3inst", 6)):
        t = _trellis(K // 16, N // 16, bits, seed=2)
        suh = (rng.standard_normal(K) * 0.02).astype(np.float16)
        svh = (rng.standard_normal(N) * 0.5).astype(np.float16)
        bias = (rng.standard_normal(N) * 0.1).astype(np.float16)
        x = rng.standard_normal((5, K))
        w = fmt.dequantize(t, suh, svh, bits, cb)
        y = fmt.forward(x, t, suh, svh, bits, cb, bias)
        assert np.allclose(x @ w + bias.astype(np.float64), y, rtol=1e-12, atol=1e-12)
    h = fmt.hadamard(128) / np.sqrt(128)
    assert np.allclose(h @ h, np.eye(128), atol=1e-12)


def test_packed_signs():
    signs = np.array([0b1000_0000_0000_0101, 0], dtype=np.uint16).view(np.int16)
    got = fmt.unpack_signs(signs)
    want = np.ones(32, dtype=np.float16)
    want[[0, 2, 15]] = -1
    assert np.array_equal(got, want)


def _entry(dtype: str, shape: list[int]) -> dict:
    return {"dtype": dtype, "shape": shape}


def test_parse_group_reads_each_tensor_and_refuses_bad_ones():
    ok = {"trellis": _entry("I16", [256, 768, 32]), "suh": _entry("F16", [4096]), "svh": _entry("F16", [12288]),
          "mul1": _entry("I32", [])}
    g = fmt.parse_group("model.layers.0.self_attn.q_proj", ok)
    assert (g.bits, g.codebook, g.k, g.n, g.in_scales, g.out_scales) == (2, "mul1", 4096, 12288, "suh", "svh")
    assert g.trellis_bytes == 4096 * 12288 * 2 // 8
    g = fmt.parse_group("p", {**ok, "trellis": _entry("I16", [256, 768, 40])})
    assert g.bits == 2.5
    g = fmt.parse_group("p", {"trellis": _entry("I16", [8, 8, 64]), "su": _entry("I16", [8]), "sv": _entry("I16", [8]),
                              "bias": _entry("F16", [128])})
    assert (g.bits, g.codebook, g.in_scales, g.out_scales, g.bias) == (4, "3inst", "su", "sv", True)
    for parts, why in (
        ({**ok, "mcg": _entry("I32", [])}, "both"),
        ({k: v for k, v in ok.items() if k != "suh"}, "input scales"),
        ({**ok, "svh": _entry("F16", [100])}, "svh"),
        ({**ok, "trellis": _entry("I16", [256, 768, 72])}, "bits"),
        ({**ok, "trellis": _entry("I16", [256, 768, 40]), "mul1": None}, "mul1"),
        ({**ok, "trellis": _entry("I16", [4, 768, 32]), "suh": _entry("F16", [64])}, "multiples"),
    ):
        parts = {k: v for k, v in parts.items() if v is not None}
        with pytest.raises(ValueError, match=why):
            fmt.parse_group("p", parts)


def _write_safetensors(path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, blob = {}, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + blob)


def test_scan_and_inspect_a_mixed_checkpoint(tmp_path, capsys):
    """Mixed widths within one MoE layer, two codebooks' markers, plain bf16 tensors and one broken group."""

    def group(prefix, k, n, bits, marker="mul1", value=fmt.MUL1_MUL):
        tw = fmt.tile_words(bits)
        out = {f"{prefix}.trellis": ("I16", [k // 16, n // 16, tw], bytes(2 * (k // 16) * (n // 16) * tw)),
               f"{prefix}.suh": ("F16", [k], bytes(2 * k)), f"{prefix}.svh": ("F16", [n], bytes(2 * n))}
        if marker:
            out[f"{prefix}.{marker}"] = ("I32", [], struct.pack("<I", value))
        return out

    tensors = {}
    tensors.update(group("model.layers.0.mlp.experts.0.gate_proj", 256, 128, 2))
    tensors.update(group("model.layers.0.mlp.experts.1.gate_proj", 256, 128, 3.5))
    tensors.update(group("model.layers.1.mlp.experts.0.gate_proj", 256, 128, 5))
    tensors.update(group("model.layers.0.self_attn.q_proj", 128, 256, 4, "mcg", fmt.MCG_MUL))
    tensors.update(group("lm_head", 128, 512, 6))
    tensors["model.embed_tokens.weight"] = ("BF16", [512, 128], bytes(2 * 512 * 128))
    tensors["model.layers.0.input_layernorm.weight"] = ("BF16", [128], bytes(256))
    tensors["model.layers.1.self_attn.o_proj.trellis"] = ("I16", [8, 8, 32], bytes(2 * 8 * 8 * 32))  # no scales
    _write_safetensors(tmp_path / "model.safetensors", tensors)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fake", "quantization_config": {
        "quant_method": "exl3", "version": "1.5.0", "bits": 3.1, "head_bits": 6, "codebook": "mul1"}}))

    ck = fmt.scan(tmp_path)
    assert len(ck.groups) == 5 and list(ck.bad) == ["model.layers.1.self_attn.o_proj"]
    assert ck.groups["model.layers.0.mlp.experts.1.gate_proj"].bits == 3.5
    assert ck.markers["lm_head.mul1"] == fmt.MUL1_MUL
    assert set(ck.plain) == {"model.embed_tokens.weight", "model.layers.0.input_layernorm.weight"}

    s = exl3_inspect.summarize(tmp_path)
    assert s["categories"]["model.layers.*.mlp.experts.*.gate_proj"] == {"2": 1, "3.5": 1, "5": 1}
    assert s["codebooks"] == {"mul1": 4, "mcg": 1} and s["config"]["version"] == "1.5.0"
    assert exl3_inspect.main([str(tmp_path)]) == 1                     # the broken group is reported
    out = capsys.readouterr().out
    assert "UNSUPPORTED" in out and "o_proj" in out and "no input scales" in out

# -- the config check families declare ---------------------------------------------

QWEN = {"quantization_config": {"quant_method": "exl3", "version": "1.4.2", "bits": 4.0, "head_bits": 6,
                                "codebook": "mul1", "mtp_bits": 4}}
MIMO = {"quantization_config": {"quant_method": "exl3", "version": "1.5.1", "bits": 2.5078, "head_bits": 6,
                                "codebook": "mul1"}}


def test_config_fields_reads_the_quantization_block():
    assert fmt.config_fields(QWEN)["codebook"] == "mul1"
    assert fmt.config_fields({"text_config": QWEN})["version"] == "1.4.2"
    assert fmt.config_fields({"quantization": {"quant_method": "exl3", "bits": 3}})["bits"] == 3
    assert fmt.config_fields({"quantization_config": {"quant_method": "mlx", "bits": 4}}) == {}


def test_require_config_accepts_every_codebook_and_width():
    for config in (QWEN, MIMO, {"quantization_config": {"quant_method": "exl3", "codebook": "3inst", "bits": 2.5}},
                   {"quantization_config": {"quant_method": "exl3"}},                     # no fields at all
                   {"quantization_config": {"quant_method": "exl3", "bits": 4.15}}):      # an average like Sage's
        fmt.require_config(config, where="Fake Model on NVIDIA GPUs (CUDA)", tested="owner/tested")
    assert "mul1" in fmt.describe_config(fmt.config_fields(QWEN))
    assert "1.4.2" in fmt.describe_config(fmt.config_fields(QWEN)) and "head 6" in fmt.describe_config(
        fmt.config_fields(QWEN))


def test_require_config_refuses_what_the_module_does_not_read():
    bad = [{"quantization_config": {"quant_method": "exl3", "codebook": "mcg2"}},
           {"quantization_config": {"quant_method": "exl3", "head_bits": 2.5}},           # a half-width head
           {"quantization_config": {"quant_method": "exl3", "bits": 9}},
           {"quantization_config": {"quant_method": "exl3", "head_bits": 0}}]
    for config in bad:
        with pytest.raises(ValueError) as refused:
            fmt.require_config(config, where="Fake Model on CUDA", tested="owner/tested", help="See RUNBOOK.md.")
        message = str(refused.value)
        assert "does not read" in message and "RUNBOOK.md" in message and "owner/tested" in message
