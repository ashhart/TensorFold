"""qwen35_bf16_norms.py: f32 GDN norms round to bf16 as torch does, every other tensor and file is unchanged."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("norms", Path(__file__).resolve().parents[1] / "qwen35_bf16_norms.py")
norms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(norms)


def f32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def shard(path, tensors):
    """A safetensors file of (name, dtype, shape, raw bytes)."""
    header, body, at = {}, b"", 0
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(raw)]}
        body += raw
        at += len(raw)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + body)


class Bf16NormsTest(unittest.TestCase):
    def test_rounding_is_nearest_ties_to_even(self):
        self.assertEqual(norms.bf16_bits(f32(1.0 + 2 ** -8)), 0x3F80)  # a tie: down to even
        self.assertEqual(norms.bf16_bits(f32(1.0 + 3 * 2 ** -8)), 0x3F82)  # a tie: up to even
        self.assertEqual(norms.bf16_bits(f32(1.0 + 2 ** -7 + 2 ** -12)), 0x3F81)  # nearest
        self.assertEqual(norms.bf16_bits(f32(-1.0)), 0xBF80)
        self.assertEqual(norms.bf16_bits(0x7FC00001) & 0x7FC0, 0x7FC0)  # NaN stays NaN

    def test_only_f32_gdn_norms_change(self):
        tmp = Path(tempfile.mkdtemp())
        src, out = tmp / "src", tmp / "out"
        src.mkdir()
        norm = struct.pack("<2f", 1.0 + 2 ** -8, -0.5)
        other = struct.pack("<2f", 1.0 + 2 ** -8, 2.0)
        shard(src / "model-00001-of-00002.safetensors", [
            ("model.layers.0.linear_attn.norm.weight", "F32", [2], norm),
            ("model.layers.0.linear_attn.A_log", "F32", [2], other)])
        shard(src / "model-00002-of-00002.safetensors", [("model.norm.weight", "BF16", [2], b"\x80\x3f\x00\x40")])
        (src / "config.json").write_text("{}")
        (src / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": {}}))
        self.assertEqual(norms.main([str(src), str(out)]), 0)
        header, base = norms.read(out / "model-00001-of-00002.safetensors")
        body = (out / "model-00001-of-00002.safetensors").read_bytes()[base:]
        n = header["model.layers.0.linear_attn.norm.weight"]
        self.assertEqual(n["dtype"], "BF16")
        self.assertEqual(body[n["data_offsets"][0]:n["data_offsets"][1]], struct.pack("<2H", 0x3F80, 0xBF00))
        a = header["model.layers.0.linear_attn.A_log"]
        self.assertEqual((a["dtype"], body[a["data_offsets"][0]:a["data_offsets"][1]]), ("F32", other))
        for name in ("model-00002-of-00002.safetensors", "config.json"):
            self.assertEqual((out / name).read_bytes(), (src / name).read_bytes())
        total = json.loads((out / "model.safetensors.index.json").read_text())["metadata"]["total_size"]
        self.assertEqual(total, sum(p.stat().st_size for p in out.glob("*.safetensors")))

    def test_a_checkpoint_without_f32_norms_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "src").mkdir()
        shard(tmp / "src" / "model.safetensors", [("model.norm.weight", "BF16", [1], b"\x80\x3f")])
        self.assertEqual(norms.main([str(tmp / "src"), str(tmp / "out")]), 1)
        self.assertFalse((tmp / "out").exists())


if __name__ == "__main__":
    unittest.main()
