"""Cold-expert BUDDY substitution: near-exact deferred decode with ZERO redos.

The deferred fixed-hot-set forward (``test_nemotron_runner_deferred.py``) runs at
the fast ~16 tok/s rate ONLY when every routed expert is resident; a routed COLD
(non-resident) expert sets the on-GPU cold flag, which forces a full-token
rollback + exact synced redo. On fresh generation (and for speculation's wide
verify blocks) routed experts go cold often, so the redos dominate and the fast
path is lost.

This module tests the OPT-IN unlock: substitute each cold expert with its nearest
RESIDENT *buddy* (most-similar gate behaviour) instead of the sentinel. With
substitution on, ``g2s`` maps EVERY global id to a valid resident slot, so the
remap gate's ``local == sentinel`` cold signal NEVER fires -> no redo -> the
deferred path runs fast on ANY generation. The cost: a routed cold expert is
computed by a similar resident one, so the output DRIFTS slightly from exact (the
user-accepted ~0.1% PPL tradeoff).

The quality gate here is COHERENCE + LOW DRIFT, NOT 0.0:

* **Scoped drift** — the substitution touches ONLY cold experts. A token routing
  entirely to resident experts is still BIT-IDENTICAL to exact (substitution is a
  no-op for resident ids). We assert byte-identity on all-resident tokens and that
  divergence appears only where a cold expert was routed.
* **Default off == exact** — ``cold_substitution=False`` is byte-identical to the
  current deferred/fixed path (the substitution is purely additive, opt-in). The
  existing exactness suites cover this; here we re-assert it directly.

Scores are preserved: substitution changes only WHICH expert computes a given
routed slot (cold -> buddy), never the routing SCORE applied to it, so the
weighted-sum structure is unchanged — only the substituted expert's contribution
drifts.
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


class NemotronColdSubstitutionMechanismTests(unittest.TestCase):
    """Substitution removes ALL cold -> zero redos, valid tokens, no sentinel g2s."""

    def test_substitution_eliminates_cold_redos(self) -> None:
        """A fixed set that MISSES routed experts runs with ZERO cold redos.

        Override the fixed set to {2,5} while the layer routes to {2,5,7}: expert
        7 is cold. WITHOUT substitution this forces a redo every token (proven by
        ``test_nemotron_runner_deferred``). WITH substitution, expert 7 maps to a
        resident buddy slot, so the deferred path never goes cold: ``cold_redos``
        stays 0, the run still produces a full valid token sequence, and at least
        one cold expert was actually substituted.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        miss = {layer: [2, 5] for layer in MOE_LAYERS}

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)

                    runner = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=2,
                        cold_substitution=True,
                    )
                    self.addCleanup(runner.close)
                    built = runner.build_fixed_hotset(PROMPT, override=miss)

                    out = runner.generate_greedy_deferred(PROMPT, steps)

                    # The whole point: cold experts were substituted, so NO redo.
                    self.assertEqual(
                        out["cold_redos"], 0,
                        "substitution must eliminate cold redos",
                    )
                    self.assertGreater(
                        out["deferred_tokens"], 0,
                        "all tokens should be served by the fast deferred path",
                    )
                    # A full, valid token sequence (right length, in-vocab ids).
                    self.assertEqual(len(out["tokens"]), steps)
                    for tok in out["tokens"]:
                        self.assertIsInstance(tok, int)
                        self.assertGreaterEqual(tok, 0)

                    # Substitution genuinely FIRED (cold experts -> buddies > 0).
                    self.assertGreater(
                        out["cold_substitutions"], 0,
                        "no cold expert was substituted (test would be vacuous)",
                    )
                    # And there is NO sentinel left in any layer's g2s -> every
                    # global id maps to a valid resident slot.
                    for layer in MOE_LAYERS:
                        entry = built[layer]
                        self.assertEqual(
                            int((entry["g2s"] == entry["sentinel"]).sum().item()),
                            0,
                            f"layer {layer} g2s still has sentinel entries",
                        )


class NemotronColdSubstitutionBuddySanityTests(unittest.TestCase):
    """Each cold expert's buddy is resident, deterministic, and 'nearest'."""

    def test_buddy_is_resident_and_deterministic(self) -> None:
        """Every cold expert maps to a RESIDENT slot; the map is reproducible."""
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        miss = {layer: [2, 5] for layer in MOE_LAYERS}

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)

            def _build():
                r = NemotronHStreamingForwardRunner(
                    str(path),
                    pin_policy="all",
                    page_experts=True,
                    fixed_hotset_experts=2,
                    cold_substitution=True,
                )
                self.addCleanup(r.close)
                return r.build_fixed_hotset(PROMPT, override=miss)

            first = _build()
            second = _build()

            for layer in MOE_LAYERS:
                order = first[layer]["order"]
                k = len(order)
                buddy_map = first[layer]["buddy_map"]
                self.assertTrue(buddy_map, "expected at least one cold->buddy entry")
                for cold_id, buddy_slot in buddy_map.items():
                    # The cold expert is genuinely NOT resident ...
                    self.assertNotIn(cold_id, order)
                    # ... and its buddy is a RESIDENT slot (< K).
                    self.assertGreaterEqual(buddy_slot, 0)
                    self.assertLess(
                        buddy_slot, k,
                        f"buddy slot {buddy_slot} is not resident (K={k})",
                    )
                # Deterministic across two independent builds.
                self.assertEqual(buddy_map, second[layer]["buddy_map"])

    def test_buddy_cosine_at_least_as_good_as_random_resident(self) -> None:
        """The chosen buddy's gate-row cosine >= cosine to any resident expert.

        Sanity that the buddy is the argmax-cosine resident, not arbitrary. (On
        the toy the gate rows are orthogonal blocks so cosines tie at 0; argmax is
        then the deterministic tie-break, and ``>=`` still holds against every
        resident alternative.)
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        miss = {layer: [2, 5] for layer in MOE_LAYERS}

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=2,
                cold_substitution=True,
            )
            self.addCleanup(runner.close)
            built = runner.build_fixed_hotset(PROMPT, override=miss)

            backbone = runner.session.model.backbone
            for layer in MOE_LAYERS:
                order = built[layer]["order"]
                buddy_map = built[layer]["buddy_map"]
                gate_w = backbone.layers[layer].mixer.gate.weight  # [E, hidden]
                gate_w = gate_w.astype(mx.float32)
                norms = mx.sqrt((gate_w * gate_w).sum(axis=1)) + 1e-12

                def _cos(a: int, b: int) -> float:
                    dot = float((gate_w[a] * gate_w[b]).sum().item())
                    return dot / (float(norms[a].item()) * float(norms[b].item()))

                for cold_id, buddy_slot in buddy_map.items():
                    buddy_id = order[buddy_slot]
                    chosen = _cos(cold_id, buddy_id)
                    for slot, resident_id in enumerate(order):
                        self.assertGreaterEqual(
                            chosen + 1e-6,
                            _cos(cold_id, resident_id),
                            f"buddy for {cold_id} (slot {buddy_slot}) not argmax-"
                            f"cosine vs resident {resident_id}",
                        )

    def test_buddy_picks_most_parallel_resident_on_nonorthogonal_gate(self) -> None:
        """With NON-orthogonal gate rows, the buddy is the most-parallel resident.

        The toy's stock gate rows are orthogonal blocks, so cosine ties at 0 and the
        buddy is just the tie-break -- which does NOT exercise the metric's
        discrimination. Here we OVERWRITE the gate weight on both MoE layers with a
        hand-crafted matrix where cold expert 7's row is nearly PARALLEL to resident
        expert 5's row (and far from the others), then assert the buddy of 7 is the
        slot holding expert 5. This proves the gate-row cosine genuinely selects the
        most-similar resident on a real (non-degenerate) gate, exactly the metric
        the real 550B model exercises.
        """
        import numpy as np
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        miss = {layer: [2, 5, 1] for layer in MOE_LAYERS}  # 7 cold; 5 resident

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=3,
                cold_substitution=True,
            )
            self.addCleanup(runner.close)

            # Force the gate bases resident, then plant a non-orthogonal weight:
            # expert 7 ~parallel to expert 5, everything else distinct directions.
            backbone = runner.session.model.backbone
            n_experts = runner._n_routed_experts()
            hidden = runner.session.config["hidden_size"]
            rng = np.random.default_rng(0)
            planted = rng.standard_normal((n_experts, hidden)).astype(np.float32)
            planted[7] = planted[5] * 0.97 + 0.03 * planted[7]  # 7 nearly == 5
            for layer in MOE_LAYERS:
                runner._load_layer_base(layer)
                backbone.layers[layer].mixer.gate.weight = mx.array(planted)

            built = runner.build_fixed_hotset(PROMPT, override=miss)
            for layer in MOE_LAYERS:
                order = built[layer]["order"]
                buddy_map = built[layer]["buddy_map"]
                self.assertIn(7, buddy_map, "expert 7 should be cold here")
                self.assertEqual(
                    order[buddy_map[7]], 5,
                    "buddy of 7 should be its most-parallel resident (5)",
                )


class NemotronColdSubstitutionDriftScopeTests(unittest.TestCase):
    """Drift is SCOPED to cold experts; all-resident tokens stay byte-exact."""

    def test_all_resident_token_is_byte_identical_to_exact(self) -> None:
        """A fixed set covering the ROUTED union -> substitution never used -> exact.

        Use a fixed set that covers the routed union {2,5,7}. The buddy map MAY be
        non-empty (experts that are neither resident nor ever routed still get a
        buddy slot, so the deferred path can't go cold even on a surprise routing),
        but since routing only ever hits RESIDENT experts the buddies are never
        gathered. So the substitution-on output must be byte-identical to both stock
        and the exact (substitution-off) deferred output. This proves the drift is
        SCOPED: substitution changes nothing on tokens that route to resident
        experts only.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_greedy(path, PROMPT, steps)

                    # Exact deferred (substitution OFF).
                    exact = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=4,
                        cold_substitution=False,
                    )
                    self.addCleanup(exact.close)
                    exact.build_fixed_hotset(PROMPT, override=cover)
                    exact_tokens = exact.generate_greedy_deferred(PROMPT, steps)["tokens"]

                    # Substituted deferred (substitution ON; routed experts all
                    # resident, so the buddies are never gathered).
                    subbed = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=4,
                        cold_substitution=True,
                    )
                    self.addCleanup(subbed.close)
                    built = subbed.build_fixed_hotset(PROMPT, override=cover)
                    out = subbed.generate_greedy_deferred(PROMPT, steps)

                    # Every buddy maps a NON-resident expert to a RESIDENT slot, and
                    # the routed union {2,5,7} is fully resident, so none of these
                    # buddies is ever used during the decode.
                    for layer in MOE_LAYERS:
                        resident = set(built[layer]["order"])
                        for cold_id in built[layer]["buddy_map"]:
                            self.assertNotIn(cold_id, resident)
                        self.assertTrue(ROUTED_UNION <= resident)
                    self.assertEqual(
                        out["tokens"], exact_tokens,
                        "covering substitution-on output diverged from exact deferred",
                    )
                    self.assertEqual(
                        out["tokens"], stock,
                        "covering substitution-on output diverged from stock",
                    )

    def test_drift_is_scoped_to_cold_routing_at_logit_level(self) -> None:
        """Substitution drifts logits ONLY when a cold expert is routed; else exact.

        Two substitution-ON runners on the SAME prompt:
          * COVERING set {2,5,7,1} -> routed union {2,5,7} all resident -> the buddy
            map is never gathered -> logits must be BIT-IDENTICAL to the exact
            (substitution-off) covering logits;
          * MISSING set {2,5,1} -> expert 7 is cold-routed -> the buddy computes
            slot 7's contribution -> logits DRIFT from exact.
        Same model, same input: the ONLY difference is whether a cold expert was
        routed, so this isolates the drift to cold routing exactly (scoped), and
        shows the all-resident case stays exact (not 0.0 asserted globally -- the
        covering case is byte-exact, the cold case is allowed to differ).
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
        miss = {layer: [2, 5, 1] for layer in MOE_LAYERS}  # expert 7 cold

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)

            # Exact reference logits (substitution off, covering set).
            exact = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=4,
                cold_substitution=False,
            )
            self.addCleanup(exact.close)
            exact.build_fixed_hotset(PROMPT, override=cover)
            exact_logits = exact.forward_logits_deferred(PROMPT)
            mx.eval(exact_logits)

            # Substitution ON, covering set: routed experts all resident -> exact.
            sub_cover = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=4,
                cold_substitution=True,
            )
            self.addCleanup(sub_cover.close)
            sub_cover.build_fixed_hotset(PROMPT, override=cover)
            cover_logits = sub_cover.forward_logits_deferred(PROMPT)
            mx.eval(cover_logits)

            # Substitution ON, missing expert 7: cold-routed -> drift expected.
            sub_miss = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=3,
                cold_substitution=True,
            )
            self.addCleanup(sub_miss.close)
            sub_miss.build_fixed_hotset(PROMPT, override=miss)
            miss_logits = sub_miss.forward_logits_deferred(PROMPT)
            mx.eval(miss_logits)

            # SCOPED: all-resident routing -> byte-identical to exact.
            self.assertEqual(
                float(mx.max(mx.abs(
                    cover_logits.astype(mx.float32) - exact_logits.astype(mx.float32)
                ))),
                0.0,
                "substitution drifted on an ALL-RESIDENT token (should be exact)",
            )
            # COLD routing -> drift exists (the substitution did something). It must
            # be a real but BOUNDED change, not a blow-up: finite logits.
            drift = float(mx.max(mx.abs(
                miss_logits.astype(mx.float32) - exact_logits.astype(mx.float32)
            )))
            self.assertGreater(
                drift, 0.0,
                "expected drift when a cold expert is substituted",
            )
            self.assertTrue(
                bool(mx.all(mx.isfinite(miss_logits)).item()),
                "substituted (cold) logits must stay finite/coherent",
            )


class NemotronColdSubstitutionDefaultOffTests(unittest.TestCase):
    """``cold_substitution=False`` is byte-identical to the current path."""

    def test_default_off_matches_exact_cold_redo_behaviour(self) -> None:
        """Default (off) on a MISSING set behaves exactly as the current deferred path.

        With substitution OFF and a fixed set that misses expert 7, the deferred
        path must STILL detect cold and redo exactly (matching stock), exactly as
        ``test_nemotron_runner_deferred`` asserts. This guards that adding the
        opt-in did not perturb the default behaviour.
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
                        cold_substitution=False,  # default; explicit for clarity
                    )
                    self.addCleanup(runner.close)
                    built = runner.build_fixed_hotset(PROMPT, override=miss)

                    # Default off: NO buddy map built, g2s still carries sentinels.
                    for layer in MOE_LAYERS:
                        self.assertEqual(built[layer].get("buddy_map", {}), {})
                        entry = built[layer]
                        self.assertGreater(
                            int((entry["g2s"] == entry["sentinel"]).sum().item()),
                            0,
                            "default-off g2s should still carry cold sentinels",
                        )

                    out = runner.generate_greedy_deferred(PROMPT, steps)
                    # Exact (cold-redo) behaviour preserved: == stock, redos fired.
                    self.assertEqual(out["tokens"], stock)
                    self.assertGreater(out["cold_redos"], 0)
                    self.assertEqual(out["cold_substitutions"], 0)


class NemotronColdSubstitutionSpeculativeTests(unittest.TestCase):
    """Speculation's wide verify blocks never go cold under substitution either."""

    def test_speculative_verify_never_cold_with_substitution(self) -> None:
        """A MISSING set + substitution -> ZERO cold verify blocks in spec decode.

        The substitution lives in the build-time ``g2s`` table, so the speculative
        deferred path's batched k-token verify (which routes through the same
        deferred forward) inherits it for free: a routed cold expert maps to a
        resident buddy, the block's OR-cold flag never fires, and the slow exact-
        verify fallback is never taken. Without substitution this MISS set makes
        every verify block cold (see ``test_nemotron_runner_speculative_deferred``).
        """
        from smarttensor.block_verify import StaticTokenBlockSource
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 12
        miss = {layer: [2, 5] for layer in MOE_LAYERS}

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)

            # A reference greedy tape so most blocks propose plausible tokens (so
            # verify blocks actually run, exercising the cold path under MISS).
            ref = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True
            )
            self.addCleanup(ref.close)
            tape = ref.generate_greedy(PROMPT, steps)["tokens"]

            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=2,
                cold_substitution=True,
            )
            self.addCleanup(runner.close)
            runner.build_fixed_hotset(PROMPT, override=miss)

            drafter = StaticTokenBlockSource(
                tokens=list(PROMPT) + list(tape), name="ref-tape"
            )
            out = runner.generate_greedy_speculative_deferred(
                PROMPT, steps, drafter=drafter, block_size=4
            )

            # The unlock for speculation: no verify block fell back to exact.
            self.assertEqual(
                out["verify_cold_blocks"], 0,
                "substitution must keep speculative verify blocks off the cold path",
            )
            self.assertGreater(out["cold_substitutions"], 0)
            self.assertEqual(len(out["tokens"]), steps)


if __name__ == "__main__":
    unittest.main()
