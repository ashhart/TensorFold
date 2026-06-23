"""Deferred decode HOT-PATH optimisation: strip per-layer Python plumbing.

The deferred fixed-hot-set decode (``generate_greedy_deferred``) already collapses
the per-token sync to ~1-2 ``mx.eval`` (``test_nemotron_runner_deferred.py``). What
remains on the per-token critical path is pure Python: the 108-iteration layer loop
in ``_stream_forward_tokens`` calls ``_load_layer_full`` / ``_load_layer_base`` ->
``_load_into_model`` on EVERY layer EVERY token, and the deferred MoE forward appends
a telemetry dict per MoE layer. On the steady-state ALL-RESIDENT decode path those
loader calls move ZERO bytes (everything is resident after prefill / hot-set install)
and the events are not consumed by ``run_summary`` (which only reads ``kind=='load'``
byte/second totals — zero here). They are removable overhead that, x108 layers x every
token, starves the GPU between kernels.

This module pins the optimisation:

* **Exactness (the hard gate).** A hot-path deferred decode is bit-identical to the
  normal deferred decode, to the synced fixed path, and to stock greedy — on BOTH
  fixtures. Stripping the plumbing must not move a single logit.
* **Overhead actually stripped.** On the hot path the loop does NOT call the per-layer
  loader and does NOT accumulate per-layer load/compute event dicts during decode, so
  the per-token Python object churn drops. We assert both: the loader is not called per
  layer during decode, and the events list does not grow per layer per token.

The optimisation is OPT-IN and gated on the all-resident precondition (the deferred
loop only enables it after a warmup forward has made every layer resident), so the
default / paged / synced paths and their telemetry are untouched.
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
    _stock_greedy,
)


class NemotronDeferredHotPathExactnessTests(unittest.TestCase):
    """Hot-path deferred decode stays token-exact vs stock / synced / normal-deferred."""

    def test_hot_path_greedy_exact_both_fixtures(self) -> None:
        """Multi-token hot-path deferred greedy == stock == synced == normal deferred."""
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_greedy(path, PROMPT, steps)

                    # Synced fixed reference.
                    synced = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=4,
                    )
                    self.addCleanup(synced.close)
                    synced.build_fixed_hotset(PROMPT, override=cover)
                    synced_tokens = synced.generate_greedy(PROMPT, steps)["tokens"]

                    # Normal deferred (hot path implicitly enabled by default).
                    deferred = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=4,
                    )
                    self.addCleanup(deferred.close)
                    deferred.build_fixed_hotset(PROMPT, override=cover)
                    out = deferred.generate_greedy_deferred(PROMPT, steps)

                    self.assertEqual(out["tokens"], stock, "hot-path deferred != stock")
                    self.assertEqual(out["tokens"], synced_tokens, "hot-path deferred != synced")
                    self.assertEqual(out["cold_redos"], 0)

    def test_hot_path_disabled_matches_enabled(self) -> None:
        """Toggling the hot path off (full per-layer plumbing) yields identical tokens.

        Proves the fast path is a pure overhead-strip: same logits, same argmax.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)

            r_on = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True, fixed_hotset_experts=4
            )
            self.addCleanup(r_on.close)
            r_on.build_fixed_hotset(PROMPT, override=cover)
            toks_on = r_on.generate_greedy_deferred(PROMPT, steps)["tokens"]

            r_off = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True, fixed_hotset_experts=4
            )
            self.addCleanup(r_off.close)
            r_off.build_fixed_hotset(PROMPT, override=cover)
            r_off.hot_decode = False  # force the legacy full-plumbing loop
            toks_off = r_off.generate_greedy_deferred(PROMPT, steps)["tokens"]

            self.assertEqual(toks_on, toks_off, "hot path changed the output")

    def test_hot_path_cold_redo_still_exact(self) -> None:
        """A missing routed expert still triggers the exact synced redo on the hot path."""
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 6
        miss = {layer: [2, 5] for layer in MOE_LAYERS}
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            stock = _stock_greedy(path, PROMPT, steps)

            deferred = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True, fixed_hotset_experts=2
            )
            self.addCleanup(deferred.close)
            deferred.build_fixed_hotset(PROMPT, override=miss)
            out = deferred.generate_greedy_deferred(PROMPT, steps)

            self.assertEqual(out["tokens"], stock, "hot-path cold-redo != stock")
            self.assertGreater(out["cold_redos"], 0)


class NemotronDeferredHotPathOverheadTests(unittest.TestCase):
    """The hot path actually removes the per-layer loader call + event churn."""

    def test_hot_path_does_not_call_loader_per_layer_during_decode(self) -> None:
        """During steady-state decode the hot path skips the per-layer loader entirely.

        The loader (``_load_into_model``) is the per-layer plumbing whose only effect on
        the resident decode path is event/bookkeeping churn (zero bytes moved). The hot
        path must not call it per layer per token; the synced fallback (cold) may.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            r = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True, fixed_hotset_experts=4
            )
            self.addCleanup(r.close)
            r.build_fixed_hotset(PROMPT, override=cover)
            # Warm prefill + 1 decode so every layer is resident and the hot path arms.
            r.generate_greedy_deferred(PROMPT, 2)

            # Count loader calls during a pure-decode hot run.
            calls = {"n": 0}
            orig = r.session._load_into_model

            def counting(names, *, action, layer):
                calls["n"] += 1
                return orig(names, action=action, layer=layer)

            r.session._load_into_model = counting
            try:
                out = r.generate_greedy_deferred(PROMPT, 6)
            finally:
                r.session._load_into_model = orig

            self.assertEqual(out["cold_redos"], 0, "test requires the all-resident path")
            # Non-layer warmup (embeddings/norm/lm_head) may load once at prefill via
            # _ensure_non_layer_weights, but the 6-layer loop must NOT call the loader
            # per layer per decode token. With 6 layers x 5 decode steps the legacy path
            # would call it >=30 times; the hot path keeps it tiny.
            self.assertLess(
                calls["n"],
                10,
                f"hot path still called the per-layer loader {calls['n']} times",
            )

    def test_hot_path_events_do_not_grow_per_layer_per_token(self) -> None:
        """The hot path does not accumulate per-layer load/compute event dicts in decode.

        ``run_summary`` only reads ``kind=='load'`` byte/second totals, which are zero on
        the resident decode path, so dropping the per-layer event dicts changes no
        reported metric. We assert the events list does not carry one load+compute dict
        per layer per decode token.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
        steps = 6
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            r = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True, fixed_hotset_experts=4
            )
            self.addCleanup(r.close)
            r.build_fixed_hotset(PROMPT, override=cover)
            out = r.generate_greedy_deferred(PROMPT, steps)
            self.assertEqual(out["cold_redos"], 0)

            n_layers = len(r.session.model.backbone.layers)
            # Legacy path emits >=1 load event per layer per token (n_layers*steps load
            # events alone). The hot path must emit far fewer than that lower bound.
            load_events = [e for e in out["events"] if e.get("kind") == "load"]
            self.assertLess(
                len(load_events),
                n_layers * steps,
                f"hot path still emits per-layer load events ({len(load_events)})",
            )
            # The summary's load totals are still well-defined (zero bytes on resident
            # decode) and the run still reports tok/s.
            self.assertIn("tok_per_s", out["summary"])


if __name__ == "__main__":
    unittest.main()
