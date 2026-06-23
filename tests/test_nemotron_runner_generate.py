"""Decode-exactness tests for NemotronHStreamingForwardRunner.generate_greedy.

Validates multi-step autoregressive greedy decode + cache handling by matching
stock mlx_lm greedy generation token-for-token. Decode-exact (identical token
list) is the gate: it proves the single persistent cache (mamba SSM state + KV)
is advanced correctly across prefill and every decode step — the riskiest
unverified part of the runner.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class NemotronGenerateTests(unittest.TestCase):
    def _stock_greedy(self, path, prompt, n):
        import mlx.core as mx

        from smarttensor.adapters.mlx import build_mlx_model_shell
        from smarttensor.manifest import SmartTensorManifest

        cfg = json.loads((Path(path) / "config.json").read_text())
        man = SmartTensorManifest.from_safetensors(
            [str(Path(path) / "model.safetensors")]
        )
        m = build_mlx_model_shell(cfg, man)
        w = m.sanitize(mx.load(str(Path(path) / "model.safetensors")))
        m.load_weights(list(w.items()))
        cache = m.make_cache()
        cur = mx.array([prompt])
        out = []
        for _ in range(n):
            lg = m(cur, cache=cache)
            mx.eval(lg)
            nxt = int(mx.argmax(lg[:, -1, :], axis=-1)[0])
            out.append(nxt)
            cur = mx.array([[nxt]])
        return out

    def test_greedy_decode_matches_stock_quantized(self) -> None:
        """Paged (page_experts=True), 4-bit quantized toy."""
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
        from tests.fixtures.tiny_nemotron import build_tiny_nemotron_quantized

        prompt = [5, 9, 1, 17, 3]
        n = 8
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            expected = self._stock_greedy(path, prompt, n)
            runner = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True
            )
            self.addCleanup(runner.close)
            res = runner.generate_greedy(prompt, n)
            self.assertEqual(res["tokens"], expected)  # decode token-for-token exact
            self.assertIn("decode_tok_s", res)
            self.assertIn("summary", res)
            self.assertEqual(len(res["tokens"]), n)
            self.assertGreater(res["prefill_s"], 0.0)
            self.assertGreater(res["decode_s"], 0.0)

    def test_greedy_decode_matches_stock_unquantized_resident(self) -> None:
        """Resident (page_experts=False), unquantized toy."""
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
        from tests.fixtures.tiny_nemotron import build_tiny_nemotron

        prompt = [5, 9, 1, 17, 3]
        n = 8
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            expected = self._stock_greedy(path, prompt, n)
            runner = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=False
            )
            self.addCleanup(runner.close)
            res = runner.generate_greedy(prompt, n)
            self.assertEqual(res["tokens"], expected)  # decode token-for-token exact
            self.assertIn("decode_tok_s", res)
            self.assertIn("summary", res)
            self.assertEqual(len(res["tokens"]), n)


if __name__ == "__main__":
    unittest.main()
