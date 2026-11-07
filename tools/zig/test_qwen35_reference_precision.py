#!/usr/bin/env python3
"""Prevent silent FP32 checkpoint-parameter downcasts in the independent model reference."""
import json
from pathlib import Path
import tempfile
import unittest
try:
    import torch
    from safetensors.torch import save_file
except ModuleNotFoundError as error:
    if error.name not in {"torch", "safetensors"}:
        raise
    torch = None

if torch is not None:
    from qwen35_reference_forward import restore_stored_fp32


@unittest.skipIf(torch is None, "requires torch and safetensors for the CPU reference")
class ReferencePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.Linear(2, 1, bias=False).bfloat16()
        self.model.projection = torch.nn.Linear(2, 1, bias=False).bfloat16()
        self.original = torch.tensor([[0.123456789, -1.23456789]], dtype=torch.float32)
        self.projection = self.model.projection.weight.detach().clone()
        self.write({"norm.weight": self.original, "projection.weight": self.projection})

    def write(self, weights):
        save_file(weights, self.path / "weights.safetensors")
        index = {"weight_map": {name: "weights.safetensors" for name in weights}}
        (self.path / "model.safetensors.index.json").write_text(json.dumps(index))

    def test_original_fp32_values_and_bf16_projection_preserved(self):
        self.model.norm.weight.data.copy_(self.original)
        self.assertFalse(torch.equal(self.model.norm.weight.detach().float(), self.original))
        result = restore_stored_fp32(self.model, self.path)
        self.assertEqual(result["count"], 1)
        self.assertTrue(result["tensors"][0]["auto_dtype_changed_values"])
        self.assertEqual(self.model.norm.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(self.model.norm.weight.detach(), self.original))
        self.assertEqual(self.model.projection.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(self.model.projection.weight.detach(), self.projection))

    def test_unmapped_fp32_is_not_silently_ignored(self):
        self.write({"missing.weight": self.original})
        with self.assertRaises(ValueError):
            restore_stored_fp32(self.model, self.path)

    def test_wrong_shape_is_refused(self):
        self.write({"norm.weight": self.original.T.contiguous()})
        with self.assertRaises(ValueError):
            restore_stored_fp32(self.model, self.path)


if __name__ == "__main__":
    unittest.main()
