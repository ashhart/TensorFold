"""Synthetic merged/base checkpoints: the adapter recovers a low-rank delta, and non-projection changes are refused."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "clef_adapter.py"
HAVE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None


def write(path, tensors):
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)


@unittest.skipUnless(HAVE, "needs torch and safetensors")
class ClefAdapterTest(unittest.TestCase):
    def setUp(self):
        import torch

        torch.manual_seed(0)
        self.tmp = Path(tempfile.mkdtemp())
        p = "model.language_model."
        self.base = {p + "layers.0.mlp.up_proj.weight": torch.randn(48, 32).bfloat16(),
                     p + "layers.1.self_attn.q_proj.weight": torch.randn(32, 32).bfloat16(),
                     p + "embed_tokens.weight": torch.randn(64, 32).bfloat16(),
                     p + "layers.1.input_layernorm.weight": torch.ones(32).bfloat16(),
                     "mtp.fc.weight": torch.randn(32, 64).bfloat16()}
        delta = (torch.randn(32, 3) @ torch.randn(3, 32)) * 0.05
        self.merged = {k: v for k, v in self.base.items() if not k.startswith("mtp.")}
        self.merged[p + "layers.1.self_attn.q_proj.weight"] = (self.base[p + "layers.1.self_attn.q_proj.weight"].double()
                                                               + delta.double()).bfloat16()
        (self.tmp / "m").mkdir()
        (self.tmp / "m" / "joint_head_config.json").write_text("{}")

    def test_recovers_the_delta_and_names_it(self):
        import torch
        from safetensors import safe_open

        write(self.tmp / "m", self.merged)
        write(self.tmp / "b", self.base)
        r = run("--merged", str(self.tmp / "m"), "--base", str(self.tmp / "b"), "--out", str(self.tmp / "o"), "--rank", "8")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        key = "layers.1.self_attn.q_proj"
        with safe_open(str(self.tmp / "o" / "adapter.safetensors"), "pt") as f:
            self.assertEqual(sorted(f.keys()), [key + ".lora_A", key + ".lora_B"])
            a, b = f.get_tensor(key + ".lora_A").double(), f.get_tensor(key + ".lora_B").double()
        self.assertEqual(tuple(a.shape), (8, 32))
        torch.testing.assert_close(a @ a.T, torch.eye(8, dtype=torch.float64), atol=2e-2, rtol=0)
        p = "model.language_model.layers.1.self_attn.q_proj.weight"
        want = self.merged[p].double() - self.base[p].double()
        self.assertLess(((b @ a - want).norm() / want.norm()).item(), 0.02)
        manifest = json.loads((self.tmp / "o" / "manifest.json").read_text())
        self.assertEqual(manifest["hf_key_prefix"], "model.language_model.")
        self.assertEqual(manifest["modules"][0]["rank"], 8)
        self.assertTrue((self.tmp / "o" / "joint_head_config.json").exists())
        self.assertTrue((self.tmp / "o" / "lm_head_rows.safetensors").exists())
        self.assertIn("adapter.safetensors", json.loads((self.tmp / "o" / "SHA256SUMS.json").read_text()))

    def test_a_changed_norm_is_refused(self):
        import torch

        p = "model.language_model.layers.1.input_layernorm.weight"
        self.merged[p] = torch.full((32,), 2.0).bfloat16()
        write(self.tmp / "m", self.merged)
        write(self.tmp / "b", self.base)
        r = run("--merged", str(self.tmp / "m"), "--base", str(self.tmp / "b"), "--out", str(self.tmp / "o"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not projection weights", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
