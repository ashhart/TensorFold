"""Mixed-precision COLD-EXPERT tier: a resident-fit lever distinct from buddy substitution.

Cold-expert BUDDY substitution (``test_nemotron_runner_cold_substitution.py``)
keeps the fast path off the redo by computing a cold expert with a *different*
resident expert's weights (changes routing -> drift). This module tests the
SIBLING lever: keep a LOWER-PRECISION copy of the ACTUAL cold experts resident,
so the routed cold expert is computed by ITS OWN weights, just re-quantized to a
coarser bit-width (e.g. 2-bit/3-bit). That fits more experts in the same RAM (a
2-bit gs128 expert is ~42% of a 4-bit gs32 one) WITHOUT a full disk reload and
WITHOUT swapping in a foreign expert.

How it differs from substitution, precisely:

* substitution: cold expert ``c`` -> resident buddy ``b``'s weights (wrong expert,
  exact bits). Drift = (W_b - W_c) @ x.
* cold tier (this): cold expert ``c`` -> ``c``'s OWN weights re-quantized to fewer
  bits. Drift = (Q_lo(W_c) - Q_hi(W_c)) @ x, i.e. just the extra quantization
  error of the coarser grid. Same expert, coarser precision.

Serve semantics (the PoC boundary). MLX's ``gather_qmm`` takes SCALAR
``bits``/``group_size`` and the packed weight shape differs per bit-width, so a
single native ``SwitchMLP`` gather cannot mix the 4-bit hot stack and the low-bit
cold stack. The PoC therefore serves a pass from the cold tier only when EVERY
routed expert is in the tier (a "pure-cold" pass): the cold stack is swapped into
``switch_mlp`` and the native ``__call__`` runs against it. A MIXED pass (some
routed experts 4-bit-resident, some cold-tier) takes the exact page fallback.

The quality gate is the NEAR-EXACT lane (coherent + BOUNDED drift, NOT 0.0):

* **Scoped drift** — the cold tier is consulted ONLY when a routed expert is not in
  the 4-bit hot set. A token routing entirely to the 4-bit hot set is
  BIT-IDENTICAL to exact (the tier is never gathered). We assert byte-identity on
  all-resident tokens and bounded, finite drift only where a cold-tier expert was
  served.
* **Real memory delta** — the per-layer low-bit stack is genuinely smaller than the
  same experts at 4-bit. We assert the byte accounting the build records.
* **Default off == exact** — ``cold_tier_bits=None`` leaves the proven synced fixed
  path byte-identical (the tier is purely additive, opt-in).
* **Quantized-only** — the tier re-quantizes a quantized resident tier, so it
  no-ops on an unquantized layer (the real Ultra is fully quantized).

The cold-tier expert is the SAME expert at coarser precision, so its routing SCORE
is the stock score; only its weight grid changes. Drift is therefore confined to
the re-quantization error of cold-routed experts.
"""
from __future__ import annotations

import tempfile
import unittest

from tests.fixtures.tiny_nemotron import (
    build_tiny_nemotron,
    build_tiny_nemotron_quantized,
)
from tests.test_nemotron_runner_fixed_hotset import (
    MOE_LAYERS,
    PROMPT,
    ROUTED_UNION,
    _stock_greedy,
)

# The toy routes every MoE pass to this union (see tiny_nemotron gate planting).
# A hot set drawn from its COMPLEMENT makes every pass "pure-cold" so the tier path
# is exercised; a hot set covering the union keeps every pass on the exact 4-bit
# path so the tier is never gathered.
NON_ROUTED = sorted(set(range(8)) - ROUTED_UNION)  # [0, 1, 3, 4, 6]


class NemotronColdTierMechanismTests(unittest.TestCase):
    """The cold tier serves a pure-cold pass at low precision (no disk redo)."""

    def test_cold_tier_serves_pure_cold_pass_at_low_precision(self) -> None:
        """Routed union held in the low-bit tier (hot set holds none of it).

        Hot set (4-bit) = two NON-routed experts; cold tier (2-bit) = the routed
        union {2,5,7}. Every MoE pass routes entirely into the tier, so it is served
        from the resident low-bit stack: ``cold_tier_hits`` fires every MoE pass and
        the slow disk-reload page fallback NEVER does. The run still produces a full
        valid token sequence (coherent, in-vocab).
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        hot = {layer: NON_ROUTED[:2] for layer in MOE_LAYERS}              # {0,1}
        cold = {layer: sorted(ROUTED_UNION) for layer in MOE_LAYERS}        # {2,5,7}

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)

            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=2,
                cold_tier_bits=2,
            )
            self.addCleanup(runner.close)
            built = runner.build_fixed_hotset(
                PROMPT, override=hot, cold_tier_override=cold
            )

            out = runner.generate_greedy(PROMPT, steps)

            # The cold tier served the cold-routed experts (every MoE pass).
            self.assertGreater(
                runner._cold_tier_stats["cold_tier_hits"], 0,
                "the cold tier was never used (test would be vacuous)",
            )
            # And the slow exact disk-reload fallback never fired.
            self.assertEqual(
                runner._hotset_stats["cold_fallbacks"], 0,
                "a pure-cold pass covered by the tier must not hit the page fallback",
            )
            # A full, valid token sequence (right length, in-vocab ids).
            self.assertEqual(len(out["tokens"]), steps)
            for tok in out["tokens"]:
                self.assertIsInstance(tok, int)
                self.assertGreaterEqual(tok, 0)

            # The tier holds the ACTUAL routed experts at the low bit-width.
            for layer in MOE_LAYERS:
                entry = built[layer]
                self.assertIn("cold_tier", entry)
                self.assertEqual(entry["cold_tier"]["order"], sorted(ROUTED_UNION))
                self.assertEqual(entry["cold_tier"]["bits"], 2)
                # Every routed expert is in the tier's slot map.
                for e in ROUTED_UNION:
                    self.assertIn(e, entry["cold_tier"]["global_to_slot"])

    def test_mixed_hot_cold_pass_falls_back_to_exact(self) -> None:
        """A MIXED pass (some 4-bit-hot, some cold-tier) takes the exact page path.

        Hot set = {2,5} (covers part of the routed union); cold tier = {7}. The toy
        routes to {2,5,7}: experts 2,5 are 4-bit-resident, 7 is cold-tier. A single
        native gather cannot span both grids, so the documented PoC boundary applies:
        the pass takes the exact page fallback (NOT the tier). This guards that the
        tier never silently serves a mixed pass (which would drift the hot experts).
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 6
        hot = {layer: [2, 5] for layer in MOE_LAYERS}
        cold = {layer: [7] for layer in MOE_LAYERS}

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            stock = _stock_greedy(path, PROMPT, steps)

            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=2,
                cold_tier_bits=2,
            )
            self.addCleanup(runner.close)
            runner.build_fixed_hotset(PROMPT, override=hot, cold_tier_override=cold)

            out = runner.generate_greedy(PROMPT, steps)

            # Mixed pass -> exact page fallback, NOT the (drifting) tier.
            self.assertEqual(
                runner._cold_tier_stats["cold_tier_hits"], 0,
                "a mixed hot+cold pass must not be served by the tier",
            )
            self.assertGreater(
                runner._hotset_stats["cold_fallbacks"], 0,
                "a mixed pass should take the exact page fallback",
            )
            # The exact page fallback is bit-exact -> matches stock.
            self.assertEqual(out["tokens"], stock)

    def test_cold_tier_no_op_on_unquantized_layer(self) -> None:
        """The tier is a re-quant lever -> it is NOT built for an unquantized layer.

        The unquantized toy has a plain ``SwitchLinear`` (weight only, no
        bits/group_size/scales), so there is nothing to coarsen cheaply. The build
        must skip the tier entirely (no ``cold_tier`` entry), and decode stays on the
        exact path. The real Ultra is fully quantized, so this only no-ops here.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 6
        hot = {layer: NON_ROUTED[:2] for layer in MOE_LAYERS}
        cold = {layer: sorted(ROUTED_UNION) for layer in MOE_LAYERS}

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)  # UNQUANTIZED
            stock = _stock_greedy(path, PROMPT, steps)

            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=2,
                cold_tier_bits=2,
            )
            self.addCleanup(runner.close)
            built = runner.build_fixed_hotset(
                PROMPT, override=hot, cold_tier_override=cold
            )

            for layer in MOE_LAYERS:
                self.assertNotIn(
                    "cold_tier", built[layer],
                    "tier must not be built for an unquantized layer",
                )
            self.assertEqual(runner._cold_tier_stats["cold_tier_experts"], 0)
            # Decode is unaffected (exact page fallback) -> matches stock.
            out = runner.generate_greedy(PROMPT, steps)
            self.assertEqual(out["tokens"], stock)
            self.assertEqual(runner._cold_tier_stats["cold_tier_hits"], 0)


class NemotronColdTierMemoryTests(unittest.TestCase):
    """The low-bit cold tier is genuinely smaller than the same experts at 4-bit."""

    def test_cold_tier_saves_bytes_vs_four_bit(self) -> None:
        """The recorded cold-tier byte total < the same experts held at 4-bit.

        The whole point of the lever is fit: the same cold experts cost fewer bytes
        at a coarser bit-width. ``build_fixed_hotset`` records, per layer, both the
        low-bit cold-tier byte total and what those experts WOULD cost at 4-bit; the
        low-bit total must be strictly smaller (the fit win).
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        hot = {layer: [2, 5] for layer in MOE_LAYERS}
        cold = {layer: [7, 0, 1, 3] for layer in MOE_LAYERS}  # a real tail

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=2,
                cold_tier_bits=2,
            )
            self.addCleanup(runner.close)
            built = runner.build_fixed_hotset(
                PROMPT, override=hot, cold_tier_override=cold
            )

            total_low = 0
            total_ref4 = 0
            for layer in MOE_LAYERS:
                ct = built[layer]["cold_tier"]
                self.assertGreater(ct["bytes"], 0)
                self.assertGreater(ct["bytes_at_4bit"], 0)
                # Low-bit is strictly cheaper than the same experts at 4-bit.
                self.assertLess(
                    ct["bytes"], ct["bytes_at_4bit"],
                    f"layer {layer}: 2-bit cold tier not smaller than 4-bit",
                )
                total_low += ct["bytes"]
                total_ref4 += ct["bytes_at_4bit"]

            # Aggregate fit win is real and recorded in the runner stats.
            self.assertLess(total_low, total_ref4)
            self.assertEqual(runner._cold_tier_stats["cold_tier_bytes"], total_low)
            self.assertEqual(
                runner._cold_tier_stats["cold_tier_bytes_at_4bit"], total_ref4
            )
            self.assertEqual(
                runner._cold_tier_stats["cold_tier_experts"],
                len(cold[MOE_LAYERS[0]]) * len(MOE_LAYERS),
            )

    def test_coarser_group_size_saves_more(self) -> None:
        """A coarser cold-tier group size shrinks the per-group overhead -> more savings.

        Same experts, same bit-width, two group sizes (32 vs 64): the gs=64 tier must
        record strictly fewer bytes than gs=32 (fewer scale/bias groups). This is the
        documented fit knob -- the cold tier need not share the on-disk group size.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        hot = {layer: [2, 5] for layer in MOE_LAYERS}
        cold = {layer: [7, 0, 1, 3] for layer in MOE_LAYERS}

        def _tier_bytes(group_size):
            with tempfile.TemporaryDirectory() as d:
                path = build_tiny_nemotron_quantized(d)
                runner = NemotronHStreamingForwardRunner(
                    str(path),
                    pin_policy="all",
                    page_experts=True,
                    fixed_hotset_experts=2,
                    cold_tier_bits=2,
                    cold_tier_group_size=group_size,
                )
                self.addCleanup(runner.close)
                runner.build_fixed_hotset(
                    PROMPT, override=hot, cold_tier_override=cold
                )
                return runner._cold_tier_stats["cold_tier_bytes"]

        # The toy latent dims (fc1 in=32, fc2 in=64) both divide 32; fc2 also
        # divides 64, so gs=64 reduces fc2's group count -> fewer bytes overall.
        bytes_gs32 = _tier_bytes(32)
        bytes_gs64 = _tier_bytes(64)
        self.assertLess(
            bytes_gs64, bytes_gs32,
            "coarser group size should reduce the cold-tier byte total",
        )


class NemotronColdTierDriftScopeTests(unittest.TestCase):
    """Drift is SCOPED to cold-tier serving; all-resident tokens stay byte-exact."""

    def test_all_resident_token_is_byte_identical_to_exact(self) -> None:
        """A 4-bit hot set covering the routed union -> tier never gathered -> exact.

        Hot set {2,5,7,1} covers the routed union {2,5,7}; the cold tier holds only
        never-routed experts, so the low-bit stack is never gathered. The
        cold-tier-on output must be byte-identical to both stock and the exact
        (tier-off) output -> the drift is SCOPED.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
        cold = {layer: [0, 3, 4, 6] for layer in MOE_LAYERS}

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            stock = _stock_greedy(path, PROMPT, steps)

            # Exact synced fixed (cold tier OFF).
            exact = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=4,
                cold_tier_bits=None,
            )
            self.addCleanup(exact.close)
            exact.build_fixed_hotset(PROMPT, override=cover)
            exact_tokens = exact.generate_greedy(PROMPT, steps)["tokens"]

            # Cold tier ON (routed union fully 4-bit resident -> tier unused).
            tiered = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=4,
                cold_tier_bits=2,
            )
            self.addCleanup(tiered.close)
            built = tiered.build_fixed_hotset(
                PROMPT, override=cover, cold_tier_override=cold
            )
            out = tiered.generate_greedy(PROMPT, steps)

            for layer in MOE_LAYERS:
                resident = set(built[layer]["order"])
                self.assertTrue(ROUTED_UNION <= resident)
                for cold_id in built[layer]["cold_tier"]["order"]:
                    self.assertNotIn(cold_id, ROUTED_UNION)
            self.assertEqual(
                out["tokens"], exact_tokens,
                "cold-tier-on output diverged from exact on all-resident tokens",
            )
            self.assertEqual(
                out["tokens"], stock,
                "cold-tier-on output diverged from stock on all-resident tokens",
            )
            self.assertEqual(
                tiered._cold_tier_stats["cold_tier_hits"], 0,
                "cold tier should NOT fire when routing is all-resident",
            )

    def test_drift_is_scoped_and_bounded_at_logit_level(self) -> None:
        """Cold-tier serving drifts logits ONLY when a cold-tier expert is served.

        Two cold-tier-ON runners on the SAME prompt:
          * COVERING 4-bit set {2,5,7,1} -> routed union resident -> the tier is never
            gathered -> logits BIT-IDENTICAL to the exact (tier-off) logits;
          * PURE-COLD set (hot = non-routed {0,1}, tier = routed {2,5,7}) -> every pass
            served at low precision -> logits DRIFT from exact, but stay finite.
        Same model, same input: the ONLY difference is whether the tier was gathered,
        isolating the drift to cold-tier serving exactly (scoped), and showing the
        all-resident case stays exact.
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
        hot = {layer: NON_ROUTED[:2] for layer in MOE_LAYERS}
        cold = {layer: sorted(ROUTED_UNION) for layer in MOE_LAYERS}

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)

            # Exact reference logits (cold tier off, covering set).
            exact = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=4,
                cold_tier_bits=None,
            )
            self.addCleanup(exact.close)
            exact.build_fixed_hotset(PROMPT, override=cover)
            exact_logits = exact.forward_logits(PROMPT)
            mx.eval(exact_logits)

            # Cold tier ON, covering set: routed experts all 4-bit resident -> exact.
            tier_cover = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=4,
                cold_tier_bits=2,
            )
            self.addCleanup(tier_cover.close)
            tier_cover.build_fixed_hotset(
                PROMPT, override=cover,
                cold_tier_override={layer: [0, 3] for layer in MOE_LAYERS},
            )
            cover_logits = tier_cover.forward_logits(PROMPT)
            mx.eval(cover_logits)

            # Cold tier ON, pure-cold routing -> served at 2-bit -> drift expected.
            tier_cold = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=2,
                cold_tier_bits=2,
            )
            self.addCleanup(tier_cold.close)
            tier_cold.build_fixed_hotset(
                PROMPT, override=hot, cold_tier_override=cold
            )
            cold_logits = tier_cold.forward_logits(PROMPT)
            mx.eval(cold_logits)

            # SCOPED: all-resident routing -> byte-identical to exact.
            self.assertEqual(
                float(mx.max(mx.abs(
                    cover_logits.astype(mx.float32) - exact_logits.astype(mx.float32)
                ))),
                0.0,
                "cold tier drifted on an ALL-RESIDENT token (should be exact)",
            )
            self.assertEqual(tier_cover._cold_tier_stats["cold_tier_hits"], 0)

            # COLD-TIER serving -> bounded, finite drift (coarser grid did something
            # but did not blow up). The tier genuinely fired.
            self.assertGreater(tier_cold._cold_tier_stats["cold_tier_hits"], 0)
            drift = float(mx.max(mx.abs(
                cold_logits.astype(mx.float32) - exact_logits.astype(mx.float32)
            )))
            self.assertGreater(
                drift, 0.0,
                "expected drift when a cold-tier (low-bit) expert is served",
            )
            self.assertTrue(
                bool(mx.all(mx.isfinite(cold_logits)).item()),
                "cold-tier (low-bit) logits must stay finite/coherent",
            )

    def test_higher_bits_drift_less_than_lower_bits(self) -> None:
        """A 3-bit cold tier drifts LESS than a 2-bit one (monotone in precision).

        Same pure-cold routing, two tier bit-widths. The 3-bit tier is a finer grid,
        so its re-quant error -- and thus the logit drift vs the exact 4-bit output
        -- must be smaller than the 2-bit tier's. This shows ``cold_tier_bits`` is a
        real quality/fit dial (more bits = less drift, more bytes).
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
        hot = {layer: NON_ROUTED[:2] for layer in MOE_LAYERS}
        cold = {layer: sorted(ROUTED_UNION) for layer in MOE_LAYERS}

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)

            exact = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True,
                fixed_hotset_experts=4, cold_tier_bits=None,
            )
            self.addCleanup(exact.close)
            exact.build_fixed_hotset(PROMPT, override=cover)
            exact_logits = exact.forward_logits(PROMPT).astype(mx.float32)
            mx.eval(exact_logits)

            def _drift(bits):
                r = NemotronHStreamingForwardRunner(
                    str(path), pin_policy="all", page_experts=True,
                    fixed_hotset_experts=2, cold_tier_bits=bits,
                )
                self.addCleanup(r.close)
                r.build_fixed_hotset(PROMPT, override=hot, cold_tier_override=cold)
                lg = r.forward_logits(PROMPT).astype(mx.float32)
                mx.eval(lg)
                self.assertGreater(r._cold_tier_stats["cold_tier_hits"], 0)
                return float(mx.max(mx.abs(lg - exact_logits)))

            drift_2bit = _drift(2)
            drift_3bit = _drift(3)
            self.assertLess(
                drift_3bit, drift_2bit,
                "3-bit cold tier should drift less than 2-bit",
            )


class NemotronColdTierDefaultOffTests(unittest.TestCase):
    """``cold_tier_bits=None`` is byte-identical to the current synced fixed path."""

    def test_default_off_matches_exact_cold_fallback_behaviour(self) -> None:
        """Default (off) on a MISSING set behaves exactly as the current fixed path.

        With the cold tier OFF and a fixed set that misses experts in the routed
        union, the synced fixed path must STILL fall back to the exact page path and
        match stock, with NO cold-tier entry built. Guards that the opt-in tier did
        not perturb the proven default behaviour.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        miss = {layer: [2, 5] for layer in MOE_LAYERS}

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_greedy(path, PROMPT, steps)

                    runner = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=2,
                        cold_tier_bits=None,  # default; explicit for clarity
                    )
                    self.addCleanup(runner.close)
                    built = runner.build_fixed_hotset(PROMPT, override=miss)

                    # Default off: NO cold-tier entry built.
                    for layer in MOE_LAYERS:
                        self.assertNotIn("cold_tier", built[layer])

                    out = runner.generate_greedy(PROMPT, steps)
                    # Exact (cold-fallback) behaviour preserved: == stock.
                    self.assertEqual(out["tokens"], stock)
                    self.assertEqual(runner._cold_tier_stats["cold_tier_hits"], 0)

    def test_cold_tier_override_requires_bits(self) -> None:
        """Passing ``cold_tier_override`` without ``cold_tier_bits`` is rejected."""
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=2,
                cold_tier_bits=None,
            )
            self.addCleanup(runner.close)
            with self.assertRaises(ValueError):
                runner.build_fixed_hotset(
                    PROMPT,
                    override={layer: [2, 5] for layer in MOE_LAYERS},
                    cold_tier_override={layer: [7] for layer in MOE_LAYERS},
                )


if __name__ == "__main__":
    unittest.main()
