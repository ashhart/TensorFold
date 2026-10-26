"""gemma4_e4b_convert.py: the quantizer on hand-worked groups, and a tiny E-series checkpoint through the converter."""
# needs numpy (skipped without it): python3 -m unittest discover -s tools/zig -p gemma4_e4b_convert_test.py

import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as e:
    raise unittest.SkipTest("numpy is not installed") from e

import gemma4_e4b_convert as c

BF16_ONE, BF16_MINUS_EIGHT, BF16_MINUS_128 = 0x3F80, 0xC100, 0xC300


def bf16(values) -> np.ndarray:
    return c.bf16_round(np.array(values, dtype=np.float32))


class Quantizer(unittest.TestCase):
    def test_four_bit_group_of_sixteen_values_repeated(self):
        # -8..7 four times: scale (7 + 8) / 15 = 1, the edge -8 is code 0 of a bias -8, so code i holds i - 8
        w = (np.tile(np.arange(16, dtype=np.float32), 4) - 8).reshape(1, 64)
        words, scales, biases, _ = c.quantize(w, 4)
        self.assertEqual(words.tolist(), [[0x76543210, 0xFEDCBA98] * 4])
        self.assertEqual(scales.tolist(), [[BF16_ONE]])
        self.assertEqual(biases.tolist(), [[BF16_MINUS_EIGHT]])

    def test_eight_bit_group_at_both_ends(self):
        # alternating -128 and 127: scale 255 / 255 = 1, the edge -128 is code 0, so the codes are 0 and 255
        w = np.where(np.arange(64) % 2 == 0, -128, 127).astype(np.float32).reshape(1, 64)
        words, scales, biases, _ = c.quantize(w, 8)
        self.assertEqual(words.tolist(), [[0xFF00FF00] * 16])
        self.assertEqual(scales.tolist(), [[BF16_ONE]])
        self.assertEqual(biases.tolist(), [[BF16_MINUS_128]])

    def test_constant_groups(self):
        # zeros keep the scale floor 1e-7 / 15; a constant 3 takes scale and bias from its edge; every code is 0
        w = np.concatenate([np.zeros((1, 64)), np.full((1, 64), 3.0)]).astype(np.float32)
        words, scales, biases, q = c.quantize(w, 4)
        self.assertEqual(words.tolist(), [[0] * 8] * 2)
        self.assertEqual(scales[0].tolist(), bf16([np.float32(1e-7) / np.float32(15)]).tolist())
        self.assertEqual(biases.tolist(), [[0], [int(bf16([3.0])[0])]])
        np.testing.assert_array_equal(c.dequantize(q, scales, biases), w)

    def test_bf16_rounds_ties_to_even(self):
        # 1 + 2^-8 is half a step above 1 (to the even 1); 1 + 3 * 2^-8 is half above 1 + 2^-7 (to 1 + 2^-6)
        self.assertEqual(bf16([1 + 2**-8, 1 + 3 * 2**-8, -(1 + 2**-8)]).tolist(), [0x3F80, 0x3F82, 0xBF80])

    def test_the_round_trip_stays_within_half_a_step(self):
        rng = np.random.default_rng(1222)
        w = c.bf16_float(bf16(rng.normal(size=(8, 256))))
        for bits in (4, 8):
            _, s, b, q = c.quantize(w, bits)
            step = np.repeat(c.bf16_float(s), c.GROUP, axis=1)
            error = np.abs(c.dequantize(q, s, b) - w)
            # half a code step, and the bf16 rounding of the stored scale and bias
            self.assertTrue((error <= step * 0.5 + np.abs(w).max() * 2**-7 * (1 << bits)).all(), bits)


def write_source(path: Path) -> dict[str, tuple[int, ...]]:
    """Six layers, the last two reading earlier caches; those two have their k, v and k_norm in a frozen shard."""
    rng = np.random.default_rng(7)
    hidden, ple, inter, vocab, layers, shared = 128, 64, 128, 8, 6, 2
    t: dict[str, tuple[int, ...]] = {"embed_tokens.weight": (vocab, hidden), "embed_tokens_per_layer.weight": (vocab, layers * ple),
                                     "norm.weight": (hidden,), "per_layer_model_projection.weight": (layers * ple, hidden),
                                     "per_layer_projection_norm.weight": (ple,)}
    for i in range(layers):
        p = f"layers.{i}."
        t.update({p + "self_attn.q_proj.weight": (64, hidden), p + "self_attn.o_proj.weight": (hidden, 64), p + "self_attn.q_norm.weight": (32,),
                  p + "mlp.gate_proj.weight": (inter, hidden), p + "mlp.up_proj.weight": (inter, hidden), p + "mlp.down_proj.weight": (hidden, inter),
                  p + "per_layer_input_gate.weight": (ple, hidden), p + "per_layer_projection.weight": (hidden, ple),
                  p + "post_per_layer_input_norm.weight": (hidden,), p + "input_layernorm.weight": (hidden,), p + "layer_scalar": (1,),
                  p + "self_attn.k_proj.weight": (32, hidden), p + "self_attn.v_proj.weight": (32, hidden), p + "self_attn.k_norm.weight": (32,)})
    text = {c.HF_PREFIX + k: v for k, v in t.items()}
    towers = {"model.vision_tower.a.weight": (64, 64), "model.audio_tower.b.weight": (64, 64), "model.embed_vision.embedding_projection.weight": (64, 128)}
    frozen = {k for k in text if (c.layer_of(k) or 0) >= layers - shared and k.endswith(("k_proj.weight", "v_proj.weight", "k_norm.weight"))}
    weight_map = {}
    for file, names in (("model-00001-of-00001.safetensors", [k for k in {**text, **towers} if k not in frozen]), ("model-frozen-shared-kv.safetensors", sorted(frozen))):
        items = [(k, "BF16", list({**text, **towers}[k]), c.bf16_round(rng.normal(size=int(np.prod({**text, **towers}[k]))).astype(np.float32)).tobytes()) for k in names]
        c.write_safetensors(path / file, items, {"format": "pt"})
        weight_map.update({k: file for k in names})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    config = {"model_type": "gemma4", "vision_config": {"x": 1}, "audio_config": {"y": 2}, "eos_token_id": [1, 106],
              "text_config": {"hidden_size_per_layer_input": ple, "num_hidden_layers": layers, "num_kv_shared_layers": shared}}
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text("{}")
    return {k: v for k, v in text.items() if k not in frozen}


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    header.pop("__metadata__")
    return header


class Converter(unittest.TestCase):
    def convert(self, **options):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src, out = Path(tmp.name) / "src", Path(tmp.name) / "out"
        src.mkdir()
        kept = write_source(src)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(c.convert(src, out, **options), 0)
        return out, read_header(out / "model.safetensors"), json.loads((out / "config.json").read_text()), kept

    def test_names_follow_mlx_lm_and_the_unread_tensors_are_not_written(self):
        out, header, _, kept = self.convert()
        self.assertTrue(all(n.startswith("language_model.model.") for n in header))
        for n in ("layers.5.self_attn.q_proj.weight", "layers.1.self_attn.k_proj.weight", "layers.3.self_attn.v_proj.weight", "layers.3.self_attn.k_norm.weight"):
            self.assertIn("language_model.model." + n, header, n)
        for n in ("layers.4.self_attn.k_proj", "layers.5.self_attn.v_proj", "layers.5.self_attn.k_norm"):
            self.assertFalse(any(h.startswith("language_model.model." + n) for h in header), n)
        self.assertFalse(any("tower" in h or "embed_vision" in h for h in header))
        self.assertTrue((out / "tokenizer.json").exists())
        written = {n for n in header if not n.endswith((".scales", ".biases"))}
        self.assertEqual(written, {c.PREFIX + k.removeprefix(c.HF_PREFIX) for k in kept})

    def test_quantized_tensors_are_a_triple_and_the_norms_stay_bf16(self):
        _, header, _, _ = self.convert()
        p = "language_model.model."
        self.assertEqual((header[p + "layers.0.mlp.up_proj.weight"]["dtype"], header[p + "layers.0.mlp.up_proj.weight"]["shape"]), ("U32", [128, 16]))
        self.assertEqual((header[p + "layers.0.mlp.up_proj.scales"]["dtype"], header[p + "layers.0.mlp.up_proj.scales"]["shape"]), ("BF16", [128, 2]))
        self.assertEqual(header[p + "layers.0.mlp.up_proj.biases"]["shape"], [128, 2])
        self.assertEqual(header[p + "embed_tokens_per_layer.weight"]["shape"], [8, 6 * 64 // 8])
        for n in ("norm.weight", "layers.2.layer_scalar", "layers.2.self_attn.q_norm.weight", "per_layer_projection_norm.weight"):
            self.assertEqual(header[p + n]["dtype"], "BF16", n)

    def test_the_default_table_is_uniform_and_both_spellings_agree(self):
        _, _, config, _ = self.convert()
        self.assertEqual(config["quantization"], {"group_size": 64, "bits": 4, "mode": "affine"})
        self.assertEqual(config["quantization"], config["quantization_config"])
        self.assertNotIn("vision_config", config)
        self.assertNotIn("audio_config", config)
        self.assertEqual(config["eos_token_id"], [1, 106])

    def test_dense_ple_writes_the_per_layer_linears_as_bf16_and_marks_them_false(self):
        _, header, config, _ = self.convert(dense_ple=True)
        p = "language_model.model."
        for n in ("layers.3.per_layer_input_gate", "layers.3.per_layer_projection", "per_layer_model_projection"):
            self.assertEqual(header[p + n + ".weight"]["dtype"], "BF16", n)
            self.assertNotIn(p + n + ".scales", header)
            self.assertIs(config["quantization"][p + n], False, n)
        self.assertEqual(sum(v is False for v in config["quantization"].values()), 6 * 2 + 1)
        self.assertIn(p + "embed_tokens_per_layer.scales", header)  # the table stays 4-bit
        self.assertEqual(config["quantization"], config["quantization_config"])

    def test_attn_bits_eight_names_only_the_projections_that_exist(self):
        _, header, config, _ = self.convert(attn_bits=8)
        p = "language_model.model."
        entries = {k for k, v in config["quantization"].items() if isinstance(v, dict)}
        want = {p + f"layers.{i}.self_attn.{m}_proj" for i in range(6) for m in "qo"} | {p + f"layers.{i}.self_attn.{m}_proj" for i in range(4) for m in "kv"}
        self.assertEqual(entries, want)
        self.assertEqual(config["quantization"][p + "layers.0.self_attn.q_proj"], {"group_size": 64, "bits": 8, "mode": "affine"})
        self.assertEqual(header[p + "layers.0.self_attn.q_proj.weight"]["shape"], [64, 32])
        self.assertEqual(header[p + "layers.0.mlp.up_proj.weight"]["shape"], [128, 16])  # the MLP stays 4-bit

    def test_two_runs_write_the_same_bytes(self):
        a, _, _, _ = self.convert(dense_ple=True, attn_bits=8)
        b, _, _, _ = self.convert(dense_ple=True, attn_bits=8)
        self.assertEqual((a / "model.safetensors").read_bytes(), (b / "model.safetensors").read_bytes())


if __name__ == "__main__":
    unittest.main()
