"""Fixed per-layer resident hot-set + native on-GPU gather: token-exact decode.

The decode-hot fix: build, ONCE, a per-layer FIXED resident stacked switch_mlp
of the top-K experts and NEVER update it per token. Per token, when the routed
experts are a subset of that fixed set, run the model's NATIVE MoE forward (one
on-GPU gather, zero Python expert assembly); the rare "cold" pass (a routed
expert not resident) falls back to the existing exact page path.

The hard gate is TOKEN-EXACTNESS: the native-hit path runs the SAME
``NemotronHMoE.__call__`` the page path runs, against the SAME expert rows (just
sourced from the fixed stack), with the SAME ``NemotronHotsetGateAdapter`` remap.
So the output must be bit-identical to (a) stock mlx_lm and (b) the existing page
path (``fixed_hotset_experts=None``) — on BOTH fixtures, for BOTH the native-hit
branch (fixed set covers routing) AND the cold-fallback branch (fixed set misses
a routed expert).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.fixtures.tiny_nemotron import (
    build_tiny_nemotron,
    build_tiny_nemotron_quantized,
)


def _stock_logits(path, ids):
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
    out = m(mx.array([ids]), cache=m.make_cache())
    mx.eval(out)
    return out


def _stock_greedy(path, prompt, n):
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
    out: list[int] = []
    for _ in range(n):
        lg = m(cur, cache=cache)
        mx.eval(lg)
        nxt = int(mx.argmax(lg[:, -1, :], axis=-1)[0])
        out.append(nxt)
        cur = mx.array([[nxt]])
    return out


def _spy_load_selected_experts(testcase):
    """Count ``_load_selected_experts`` calls (delegates to the real method).

    The native-hit path must NOT assemble a compact table per token, so on the
    all-resident path this counter stays at its one-time-build value during
    decode. The old page path calls it on EVERY MoE layer EVERY pass.
    """
    from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

    original = NemotronHStreamingForwardRunner._load_selected_experts
    counter = {"calls": 0}

    def counting(self, *args, _orig=original, _counter=counter, **kwargs):
        _counter["calls"] += 1
        return _orig(self, *args, **kwargs)

    NemotronHStreamingForwardRunner._load_selected_experts = counting
    testcase.addCleanup(
        setattr,
        NemotronHStreamingForwardRunner,
        "_load_selected_experts",
        original,
    )
    return counter


# The tiny fixture's two MoE layers live at backbone indices 1 and 4 and route
# to the non-contiguous union {2,5,7} for the standard token sequence.
MOE_LAYERS = (1, 4)
ROUTED_UNION = {2, 5, 7}
PROMPT = [5, 9, 1, 17, 3, 8]


class NemotronFixedHotsetExactnessTests(unittest.TestCase):
    def test_native_hit_forward_logits_exact_both_fixtures(self) -> None:
        """forward_logits with a COVERING fixed set == stock == page path (0.0).

        Override the fixed set to a SUPERSET of the routed union ({2,5,7,...}) so
        every MoE layer takes the native-hit branch. The result must be
        bit-identical to stock mlx_lm and to the page path on both fixtures.
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        # Covering order: superset of {2,5,7}, deliberately NON-sorted and with
        # an extra resident expert (1) so the remap is a genuine permutation and
        # residency strictly exceeds the routed union.
        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_logits(path, PROMPT)

                    page = NemotronHStreamingForwardRunner(
                        str(path), pin_policy="all", page_experts=True
                    )
                    self.addCleanup(page.close)
                    page_logits = page.forward_logits(PROMPT)
                    mx.eval(page_logits)

                    fixed = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=4,
                    )
                    self.addCleanup(fixed.close)
                    fixed.build_fixed_hotset(PROMPT, override=cover)
                    got = fixed.forward_logits(PROMPT)
                    mx.eval(got)

                    self.assertEqual(tuple(got.shape), tuple(stock.shape))
                    self.assertEqual(
                        float(
                            mx.max(
                                mx.abs(
                                    got.astype(mx.float32)
                                    - stock.astype(mx.float32)
                                )
                            )
                        ),
                        0.0,
                        "native-hit prefill diverged from stock mlx_lm",
                    )
                    self.assertEqual(
                        float(
                            mx.max(
                                mx.abs(
                                    got.astype(mx.float32)
                                    - page_logits.astype(mx.float32)
                                )
                            )
                        ),
                        0.0,
                        "native-hit prefill diverged from the page path",
                    )
                    # The covering set really did take the native branch on
                    # every MoE layer (no cold fallback).
                    self.assertGreater(fixed._hotset_stats["native_hits"], 0)
                    self.assertEqual(fixed._hotset_stats["cold_fallbacks"], 0)

    def test_cold_fallback_forward_logits_exact_both_fixtures(self) -> None:
        """forward_logits with a MISSING routed expert == stock == page path (0.0).

        Override the fixed set to {2,5} while the layer routes to {2,5,7}: expert
        7 is not resident, so every MoE layer must take the COLD fallback (the
        exact page path). Still bit-identical to stock and to the page path.
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        # Missing order: 7 is routed but NOT resident -> forces cold fallback.
        miss = {layer: [2, 5] for layer in MOE_LAYERS}

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_logits(path, PROMPT)

                    page = NemotronHStreamingForwardRunner(
                        str(path), pin_policy="all", page_experts=True
                    )
                    self.addCleanup(page.close)
                    page_logits = page.forward_logits(PROMPT)
                    mx.eval(page_logits)

                    fixed = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=2,
                    )
                    self.addCleanup(fixed.close)
                    fixed.build_fixed_hotset(PROMPT, override=miss)
                    got = fixed.forward_logits(PROMPT)
                    mx.eval(got)

                    self.assertEqual(tuple(got.shape), tuple(stock.shape))
                    self.assertEqual(
                        float(
                            mx.max(
                                mx.abs(
                                    got.astype(mx.float32)
                                    - stock.astype(mx.float32)
                                )
                            )
                        ),
                        0.0,
                        "cold-fallback prefill diverged from stock mlx_lm",
                    )
                    self.assertEqual(
                        float(
                            mx.max(
                                mx.abs(
                                    got.astype(mx.float32)
                                    - page_logits.astype(mx.float32)
                                )
                            )
                        ),
                        0.0,
                        "cold-fallback prefill diverged from the page path",
                    )
                    # Expert 7 missing on every MoE layer -> all cold fallbacks.
                    self.assertGreater(fixed._hotset_stats["cold_fallbacks"], 0)
                    self.assertEqual(fixed._hotset_stats["native_hits"], 0)

    def test_native_hit_generate_greedy_exact_both_fixtures(self) -> None:
        """Multi-token greedy with a COVERING fixed set == stock == page path.

        Decode-exactness (identical token list across all steps) is the gate: any
        drift in the fixed-stack weights or the native-hit remap would change a
        downstream argmax. Checked vs stock AND vs the page path, both fixtures.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_greedy(path, PROMPT, steps)

                    page = NemotronHStreamingForwardRunner(
                        str(path), pin_policy="all", page_experts=True
                    )
                    self.addCleanup(page.close)
                    page_tokens = page.generate_greedy(PROMPT, steps)["tokens"]

                    fixed = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=4,
                    )
                    self.addCleanup(fixed.close)
                    fixed.build_fixed_hotset(PROMPT, override=cover)
                    fixed_tokens = fixed.generate_greedy(PROMPT, steps)["tokens"]

                    self.assertEqual(
                        fixed_tokens,
                        stock,
                        "native-hit decode diverged from stock mlx_lm",
                    )
                    self.assertEqual(
                        fixed_tokens,
                        page_tokens,
                        "native-hit decode diverged from the page path",
                    )

    def test_cold_fallback_generate_greedy_exact_both_fixtures(self) -> None:
        """Multi-token greedy with a MISSING routed expert == stock == page path."""
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        miss = {layer: [2, 5] for layer in MOE_LAYERS}

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_greedy(path, PROMPT, steps)

                    page = NemotronHStreamingForwardRunner(
                        str(path), pin_policy="all", page_experts=True
                    )
                    self.addCleanup(page.close)
                    page_tokens = page.generate_greedy(PROMPT, steps)["tokens"]

                    fixed = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=2,
                    )
                    self.addCleanup(fixed.close)
                    fixed.build_fixed_hotset(PROMPT, override=miss)
                    fixed_tokens = fixed.generate_greedy(PROMPT, steps)["tokens"]

                    self.assertEqual(
                        fixed_tokens,
                        stock,
                        "cold-fallback decode diverged from stock mlx_lm",
                    )
                    self.assertEqual(
                        fixed_tokens,
                        page_tokens,
                        "cold-fallback decode diverged from the page path",
                    )

    def test_mixed_native_and_cold_in_one_forward_is_exact(self) -> None:
        """One layer native-hit + the other cold-fallback in the SAME pass == stock.

        The realistic production case and the sharpest test of state restoration:
        MoE layer 1 gets a COVERING fixed set ({2,5,7,1}) so it takes the native
        branch, while MoE layer 4 gets a MISSING set ({2,5}) so it takes the cold
        page branch — within the same forward. The result must still be
        bit-identical to stock and the page path, and the per-pass telemetry must
        show BOTH branches firing.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        mixed = {1: [7, 2, 5, 1], 4: [2, 5]}  # layer 1 covers, layer 4 misses 7
        steps = 6
        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_greedy(path, PROMPT, steps)

                    page = NemotronHStreamingForwardRunner(
                        str(path), pin_policy="all", page_experts=True
                    )
                    self.addCleanup(page.close)
                    page_tokens = page.generate_greedy(PROMPT, steps)["tokens"]

                    fixed = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=4,
                    )
                    self.addCleanup(fixed.close)
                    fixed.build_fixed_hotset(PROMPT, override=mixed)
                    fixed_tokens = fixed.generate_greedy(PROMPT, steps)["tokens"]

                    self.assertEqual(
                        fixed_tokens, stock, "mixed-branch decode diverged from stock"
                    )
                    self.assertEqual(
                        fixed_tokens,
                        page_tokens,
                        "mixed-branch decode diverged from the page path",
                    )
                    # BOTH branches fired: layer 1 native, layer 4 cold, every pass.
                    self.assertGreater(fixed._hotset_stats["native_hits"], 0)
                    self.assertGreater(fixed._hotset_stats["cold_fallbacks"], 0)

    def test_frequency_built_hotset_is_exact(self) -> None:
        """A hot-set built from MEASURED routing frequency (no override) is exact.

        Exercises ``build_fixed_hotset`` end-to-end: it runs a warmup decode,
        records per-(MoE-layer, expert) routing frequency, and selects the top-K.
        With K = full union size the built set covers routing -> native-hit
        decode, bit-identical to stock.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            stock = _stock_greedy(path, PROMPT, steps)

            fixed = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=3,  # == |{2,5,7}|, so the union is covered
            )
            self.addCleanup(fixed.close)
            built = fixed.build_fixed_hotset(PROMPT, warmup_tokens=8)

            # The warmup discovered the {2,5,7} union on every MoE layer.
            for layer in MOE_LAYERS:
                self.assertIn(layer, built)
                self.assertEqual(set(built[layer]["order"]), ROUTED_UNION)

            fixed_tokens = fixed.generate_greedy(PROMPT, steps)["tokens"]
            self.assertEqual(
                fixed_tokens,
                stock,
                "frequency-built native-hit decode diverged from stock",
            )
            # Covered routing -> native-hit, no cold fallback.
            self.assertEqual(fixed._hotset_stats["cold_fallbacks"], 0)
            self.assertGreater(fixed._hotset_stats["native_hits"], 0)


class NemotronFixedHotsetNativeProofTests(unittest.TestCase):
    def test_native_hit_does_not_assemble_per_token(self) -> None:
        """On the all-resident path the per-token compact assembly is GONE.

        The old page path calls ``_load_selected_experts`` on every MoE layer
        every pass (2 layers x #passes). With a covering fixed set, the native-hit
        path makes ZERO such calls during decode — the one-time build's loads are
        counted separately and the per-token gather is eliminated.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 10
        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)

            # Reference: the page path's per-token assembly count.
            page_counter = _spy_load_selected_experts(self)
            page = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True
            )
            self.addCleanup(page.close)
            page.generate_greedy(PROMPT, steps)
            page_calls = page_counter["calls"]

            passes = 1 + (steps - 1)  # prefill + decode steps
            self.assertEqual(
                page_calls,
                2 * passes,
                "page path is expected to assemble every MoE layer every pass",
            )

            # Fixed hot-set: build first (one-time loads), then count assembly
            # calls DURING decode only.
            fixed = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=4,
            )
            self.addCleanup(fixed.close)
            fixed.build_fixed_hotset(PROMPT, override=cover)

            decode_counter = _spy_load_selected_experts(self)
            fixed.generate_greedy(PROMPT, steps)

            self.assertEqual(
                decode_counter["calls"],
                0,
                "native-hit decode still assembled a compact table per token "
                "(the per-token gather was not eliminated)",
            )
            # And the native branch genuinely ran every MoE layer every pass.
            self.assertEqual(
                fixed._hotset_stats["native_hits"],
                2 * passes,
                "native-hit count does not cover every MoE layer every pass",
            )
            self.assertEqual(fixed._hotset_stats["cold_fallbacks"], 0)


class NemotronFixedHotsetDefaultPathTests(unittest.TestCase):
    def test_feature_off_is_unchanged_page_path(self) -> None:
        """``fixed_hotset_experts=None`` (default) leaves the page path untouched.

        No fixed hot-set is built, the runner exposes no residency, and decode is
        bit-identical to stock — i.e. the feature is fully opt-in and the default
        behavior is preserved.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 6
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            stock = _stock_greedy(path, PROMPT, steps)

            runner = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True
            )
            self.addCleanup(runner.close)
            # Feature off: no fixed hot-set built.
            self.assertFalse(runner._fixed_hotset)
            tokens = runner.generate_greedy(PROMPT, steps)["tokens"]
            self.assertEqual(tokens, stock)


if __name__ == "__main__":
    unittest.main()
