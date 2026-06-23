"""Construction + model-type-gate tests for NemotronHStreamingForwardRunner."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.fixtures.tiny_nemotron import build_tiny_nemotron, tiny_config


class NemotronResidentForwardTests(unittest.TestCase):
    def _stock_logits(self, path, ids):
        import mlx.core as mx
        from pathlib import Path
        from smarttensor.adapters.mlx import build_mlx_model_shell
        from smarttensor.manifest import SmartTensorManifest

        cfg = json.loads((Path(path) / "config.json").read_text())
        manifest = SmartTensorManifest.from_safetensors(
            [str(Path(path) / "model.safetensors")]
        )
        model = build_mlx_model_shell(cfg, manifest)
        w = model.sanitize(mx.load(str(Path(path) / "model.safetensors")))
        model.load_weights(list(w.items()))
        out = model(mx.array([ids]), cache=model.make_cache())
        mx.eval(out)
        return out

    def test_resident_forward_matches_stock(self) -> None:
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        ids = [5, 9, 1, 17, 3]
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            expected = self._stock_logits(path, ids)
            runner = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=False
            )
            self.addCleanup(runner.close)
            got = runner.forward_logits(ids)
            mx.eval(got)
            self.assertEqual(tuple(got.shape), tuple(expected.shape))
            self.assertEqual(
                float(
                    mx.max(mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)))
                ),
                0.0,
            )


class NemotronRunnerConstructTests(unittest.TestCase):
    def test_constructs_against_tiny_model(self) -> None:
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            runner = NemotronHStreamingForwardRunner(str(path), pin_policy="all")
            self.addCleanup(runner.close)
            self.assertEqual(len(runner.session.model.backbone.layers), 6)
            self.assertEqual(len(runner.session.model.make_cache()), 3)  # M,*,M -> 3

    def test_rejects_wrong_model_type(self) -> None:
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        with tempfile.TemporaryDirectory() as d:
            cfg = tiny_config()
            cfg["model_type"] = "gpt_oss"
            (Path(d) / "config.json").write_text(json.dumps(cfg, indent=2))
            with self.assertRaises(ValueError):
                NemotronHStreamingForwardRunner(d)


if __name__ == "__main__":
    unittest.main()
