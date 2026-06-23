"""Compose: speculative VERIFY through the DEFERRED fixed-hot-set forward.

Two token-exact mechanisms already live on this runner:

1. ``generate_greedy_deferred`` — decode as ~one lazy GPU graph (on-GPU
   global->slot remap, ZERO per-layer eval), with an end-of-token cold flag and
   a roll-back-and-redo on cold. The fast bandwidth-bound base.
2. ``generate_greedy_speculative`` — a drafter proposes k tokens; the runner
   verifies them in ONE batched forward, accepts the longest greedy-matching
   prefix + a bonus token, rolls the cache back, and re-forwards the committed
   tokens. Token-exact, but verifies via the SLOW synced path.

``generate_greedy_speculative_deferred`` composes them: the batched k-token
verify runs through ``_stream_forward_tokens_deferred`` (the deferred MoE
dispatch) instead of the synced ``_stream_forward_tokens``. The block's lazy
cold flag (OR over all k tokens x all layers) is evaluated alongside the verify
argmaxes — the single sync the speculative loop already does. If the verify
block is COLD (some routed expert not resident), the optimistic per-position
predictions are invalid, so the block's verify is REDONE on the proven-exact
synced path before acceptance. Acceptance / bonus / rollback are identical to
``generate_greedy_speculative``.

THE HARD GATE is token-exactness. Speculation only ever emits tokens matching
the model's own greedy; the deferred verify is exact on all-resident blocks
(on-GPU remap == numpy remap, already proven == stock == page) and falls back to
the exact synced verify on cold blocks. So the emitted tokens are bit-identical
to plain ``generate_greedy`` (== stock mlx_lm) for EVERY token, on BOTH fixtures,
under a COVERING fixed set (all-resident verify) AND a MISSING-expert fixed set
(cold verify -> exact fallback), with a CORRECT drafter tape (high acceptance,
exercises commit/rollback) AND a WRONG tape (all-reject every block).
"""
from __future__ import annotations

import tempfile
import unittest

from smarttensor.block_verify import StaticTokenBlockSource
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
from tests.test_nemotron_runner_speculative import _LastTokenFlipTape, _MixedTape

# A fixed set that COVERS the routed union (superset, non-sorted, with an extra
# resident expert) so every verify block is all-resident -> pure deferred verify.
COVER = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
# A fixed set MISSING a routed expert (routes to {2,5,7}, resident {2,5}) so every
# verify block is COLD -> the exact synced-verify fallback fires every block.
MISS = {layer: [2, 5] for layer in MOE_LAYERS}
N = 12


def _spy_deferred_verify(testcase):
    """Count ``_stream_forward_tokens_deferred`` calls (the verify-via-deferred proof).

    The compose point is that the batched k-token VERIFY routes through the
    deferred forward. We spy the deferred stream entrypoint so a non-zero count
    proves the verify actually used it (rather than silently falling back to the
    synced path). Delegates to the real method.
    """
    from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

    original = NemotronHStreamingForwardRunner._stream_forward_tokens_deferred
    counter = {"calls": 0}

    def counting(self, *args, _orig=original, _counter=counter, **kwargs):
        _counter["calls"] += 1
        return _orig(self, *args, **kwargs)

    NemotronHStreamingForwardRunner._stream_forward_tokens_deferred = counting
    testcase.addCleanup(
        setattr,
        NemotronHStreamingForwardRunner,
        "_stream_forward_tokens_deferred",
        original,
    )
    return counter


def _count_evals():
    """Patch ``mlx.core.eval`` to tally calls; returns (counter, restore)."""
    import mlx.core as mx

    original = mx.eval
    counter = {"calls": 0}

    def counting(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    mx.eval = counting

    def restore() -> None:
        mx.eval = original

    return counter, restore


class NemotronSpeculativeDeferredExactnessTests(unittest.TestCase):
    """spec-deferred output == plain greedy == stock, both fixtures, both branches."""

    def _runner(self, path, *, fixed_hotset_experts):
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        runner = NemotronHStreamingForwardRunner(
            str(path),
            pin_policy="all",
            page_experts=True,
            fixed_hotset_experts=fixed_hotset_experts,
        )
        self.addCleanup(runner.close)
        return runner

    def _exact_gate(self, builder, *, override, fixed_hotset_experts, expect_cold):
        """spec-deferred == plain greedy == stock under correct AND wrong tapes.

        ``override`` selects the all-resident (COVER) or cold (MISS) verify
        branch; ``expect_cold`` asserts the cold-verify fallback actually fired
        (MISS) or never fired (COVER), so each branch is genuinely exercised.
        """
        with tempfile.TemporaryDirectory() as d:
            path = builder(d)

            # Ground truth: stock mlx_lm greedy, and the runner's own plain greedy
            # (which is already proven == stock by the generate/paged tests).
            stock = _stock_greedy(path, PROMPT, N)
            base = self._runner(path, fixed_hotset_experts=fixed_hotset_experts)
            greedy = base.generate_greedy(PROMPT, N)["tokens"]
            self.assertEqual(greedy, stock, "runner plain greedy diverged from stock")

            # --- Correct tape: replays the model's own greedy, so most blocks
            # commit in full (high acceptance) -> exercises commit + rollback with
            # the deferred verify.
            correct = StaticTokenBlockSource(
                tokens=list(PROMPT) + list(greedy), name="correct-tape"
            )
            r1 = self._runner(path, fixed_hotset_experts=fixed_hotset_experts)
            r1.build_fixed_hotset(PROMPT, override=override)
            out_correct = r1.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=correct, block_size=4
            )
            self.assertEqual(
                out_correct["tokens"],
                stock,
                "spec-deferred (correct tape) diverged from stock/greedy",
            )
            self.assertEqual(len(out_correct["tokens"]), N)

            # --- Wrong tape: every proposal mismatches -> reject at offset 0 every
            # block, fall back to the bonus token. Output must STILL equal greedy
            # (proves rollback restores the cache bit-exactly after a batched
            # deferred verify, including the Mamba SSM state).
            wrong = StaticTokenBlockSource(
                tokens=list(PROMPT) + [(int(t) + 1) % 128 for t in greedy],
                name="wrong-tape",
            )
            r2 = self._runner(path, fixed_hotset_experts=fixed_hotset_experts)
            r2.build_fixed_hotset(PROMPT, override=override)
            out_wrong = r2.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=wrong, block_size=4
            )
            self.assertEqual(
                out_wrong["tokens"],
                stock,
                "spec-deferred (wrong tape) diverged from stock/greedy",
            )

            # The intended verify branch was genuinely taken.
            if expect_cold:
                self.assertGreater(
                    out_correct["verify_cold_blocks"],
                    0,
                    "MISS fixture: cold-verify fallback never fired",
                )
                self.assertGreater(out_wrong["verify_cold_blocks"], 0)
            else:
                self.assertEqual(
                    out_correct["verify_cold_blocks"],
                    0,
                    "COVER fixture: a verify block was unexpectedly cold",
                )
                self.assertEqual(out_wrong["verify_cold_blocks"], 0)

    # --- All-resident verify (COVER): pure deferred verify, no cold fallback.
    def test_all_resident_unquantized(self) -> None:
        self._exact_gate(
            build_tiny_nemotron,
            override=COVER,
            fixed_hotset_experts=4,
            expect_cold=False,
        )

    def test_all_resident_quantized(self) -> None:
        self._exact_gate(
            build_tiny_nemotron_quantized,
            override=COVER,
            fixed_hotset_experts=4,
            expect_cold=False,
        )

    # --- Cold verify (MISS): every verify block has a non-resident routed expert
    #     -> the exact synced-verify fallback fires before acceptance.
    def test_cold_verify_unquantized(self) -> None:
        self._exact_gate(
            build_tiny_nemotron,
            override=MISS,
            fixed_hotset_experts=2,
            expect_cold=True,
        )

    def test_cold_verify_quantized(self) -> None:
        self._exact_gate(
            build_tiny_nemotron_quantized,
            override=MISS,
            fixed_hotset_experts=2,
            expect_cold=True,
        )

    def test_block_size_one_matches_greedy(self) -> None:
        """Degenerate k=1 spec-deferred is still exact (both branches)."""
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            stock = _stock_greedy(path, PROMPT, N)
            drafter = StaticTokenBlockSource(
                tokens=list(PROMPT) + list(stock), name="correct-tape"
            )
            r = self._runner(path, fixed_hotset_experts=4)
            r.build_fixed_hotset(PROMPT, override=COVER)
            out = r.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=drafter, block_size=1
            )
            self.assertEqual(out["tokens"], stock)

    def test_frequency_built_is_exact(self) -> None:
        """A frequency-built (no override) spec-deferred decode is exact + all-resident."""
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            stock = _stock_greedy(path, PROMPT, N)
            r = self._runner(path, fixed_hotset_experts=3)  # == |{2,5,7}|, covers union
            built = r.build_fixed_hotset(PROMPT, warmup_tokens=8)
            for layer in MOE_LAYERS:
                self.assertEqual(set(built[layer]["order"]), ROUTED_UNION)
            drafter = StaticTokenBlockSource(
                tokens=list(PROMPT) + list(stock), name="correct-tape"
            )
            out = r.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=drafter, block_size=4
            )
            self.assertEqual(out["tokens"], stock)
            self.assertEqual(out["verify_cold_blocks"], 0)

    def test_requires_fixed_hotset(self) -> None:
        """spec-deferred without a built hot-set raises (opt-in, like deferred)."""
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            r = self._runner(path, fixed_hotset_experts=None)
            drafter = StaticTokenBlockSource(tokens=list(PROMPT), name="t")
            with self.assertRaises(Exception):
                r.generate_greedy_speculative_deferred(
                    PROMPT, 4, drafter=drafter, block_size=4
                )


class NemotronSpeculativeDeferredMechanismTests(unittest.TestCase):
    """Prove the verify pass actually routes through the deferred forward."""

    def _runner(self, path, *, fixed_hotset_experts):
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        runner = NemotronHStreamingForwardRunner(
            str(path),
            pin_policy="all",
            page_experts=True,
            fixed_hotset_experts=fixed_hotset_experts,
        )
        self.addCleanup(runner.close)
        return runner

    def test_verify_uses_deferred_forward(self) -> None:
        """The batched verify invokes ``_stream_forward_tokens_deferred``.

        Spy the deferred stream entrypoint; a multi-block spec-deferred run must
        call it at least once per verify block (the compose point). The synced
        verify path would never touch it.
        """
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            stock = _stock_greedy(path, PROMPT, N)
            drafter = StaticTokenBlockSource(
                tokens=list(PROMPT) + list(stock), name="correct-tape"
            )
            r = self._runner(path, fixed_hotset_experts=4)
            r.build_fixed_hotset(PROMPT, override=COVER)
            counter = _spy_deferred_verify(self)
            out = r.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=drafter, block_size=4
            )
            self.assertEqual(out["tokens"], stock)
            self.assertGreater(out["blocks"], 0)
            self.assertGreaterEqual(
                counter["calls"],
                out["blocks"],
                "verify did not route through the deferred forward",
            )

    def _verify_block_evals(self, runner, block, *, deferred):
        """Count ``mx.eval`` calls for ONE k-token verify forward of ``block``.

        Isolates the verify itself (no snapshot / accept / commit) so the proof is
        a clean apples-to-apples eval count: the same k-token block forwarded once
        on the SYNCED fixed path vs the DEFERRED path. Mirrors the deferred eval-
        count harness (``_decode_one_token_synced`` / ``_decode_one_token_deferred``)
        but over a batched k-token block. The forward advances a FRESH cache, so it
        does not perturb any persistent state.
        """
        import mlx.core as mx

        model = runner.session.model
        norm_f = model.backbone.norm_f
        lm_head = model.lm_head

        if deferred:
            runner._install_deferred_hotset()
            try:
                counter, restore = _count_evals()
                try:
                    h = runner._stream_forward_tokens_deferred(
                        [block], cache=model.make_cache(), events=[],
                        pass_kind="verify",
                    )
                    preds = mx.argmax(lm_head(norm_f(h))[0], axis=-1)
                    cold = runner._deferred_cold_flag()
                    mx.eval(preds, cold)  # single per-block sync
                finally:
                    restore()
            finally:
                runner._uninstall_deferred_hotset()
        else:
            # Synced fixed-path verify: dispatch picks _nemotron_moe_layer_forward_
            # fixed, which evals once per MoE layer (the per-layer mx.eval(inds)).
            counter, restore = _count_evals()
            try:
                h = runner._stream_forward_tokens(
                    [block], cache=model.make_cache(), events=[], pass_kind="verify",
                )
                preds = mx.argmax(lm_head(norm_f(h))[0], axis=-1)
                mx.eval(preds)
            finally:
                restore()
        return counter["calls"]

    def test_verify_block_does_few_evals(self) -> None:
        """An all-resident k-token verify block evals fewer than the synced verify.

        The deferred verify's whole point is one lazy graph: the k-token block
        forward composes to a SINGLE end-of-block eval (verify argmaxes + cold
        flag), NOT one ``mx.eval(inds)`` per MoE layer. We forward the SAME block
        once on each path (fresh cache, verify only) and assert:

        * the synced verify pays at least one eval per MoE layer (>= len(MOE_LAYERS));
        * the deferred verify pays strictly fewer, and a small constant (~1-2);
        * the deferred eval count does NOT grow with block_size (a per-layer scheme
          would be constant in k too, but a per-position one would not — this pins
          the lazy-graph property).
        """
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            stock = _stock_greedy(path, PROMPT, N)
            block4 = stock[:4]
            block8 = stock[:8]

            r = self._runner(path, fixed_hotset_experts=4)
            r.build_fixed_hotset(PROMPT, override=COVER)
            # Warm the lazy base loads OUTSIDE the per-call counters (steady state).
            r.generate_greedy(PROMPT, 1)

            synced_evals = self._verify_block_evals(r, block4, deferred=False)
            deferred_evals4 = self._verify_block_evals(r, block4, deferred=True)
            deferred_evals8 = self._verify_block_evals(r, block8, deferred=True)

            self.assertGreaterEqual(
                synced_evals,
                len(MOE_LAYERS),
                "synced verify should eval at least once per MoE layer",
            )
            self.assertLess(
                deferred_evals4,
                synced_evals,
                f"deferred verify ({deferred_evals4}) not fewer than synced "
                f"({synced_evals})",
            )
            self.assertLessEqual(
                deferred_evals4,
                2,
                f"deferred verify did {deferred_evals4} evals (target ~1-2)",
            )
            # Lazy-graph property: a bigger block does NOT cost more verify evals.
            self.assertEqual(
                deferred_evals4,
                deferred_evals8,
                "deferred verify evals scaled with block size (not a single graph)",
            )


class NemotronSpeculativeDeferredFullAcceptSkipTests(unittest.TestCase):
    """Full-accept-skip optimization: token-exact AND fewer forwards.

    When a verify block is FULLY accepted (j == block_size), the verify forward
    already advanced the cache to the exact post-block state, so the rollback +
    commit re-forward is redundant. Skipping it must (a) keep the output
    bit-identical to plain greedy == stock and (b) genuinely save a forward
    (1 forward/block on the skip path vs 2 on the rollback path).
    """

    def _runner(self, path, *, fixed_hotset_experts):
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        runner = NemotronHStreamingForwardRunner(
            str(path),
            pin_policy="all",
            page_experts=True,
            fixed_hotset_experts=fixed_hotset_experts,
        )
        self.addCleanup(runner.close)
        return runner

    def _gate(self, builder) -> None:
        """oracle (full-accept) / partial / wrong tapes all == stock, no cold.

        Uses the COVERING fixed set (a superset of the routed union), which is
        genuinely token-EXACT (== stock == page, proven by the fixed-hotset
        tests). Every verify block is all-resident so ``verify_cold_blocks == 0``
        — isolating the pure skip path from the cold-verify fallback WITHOUT the
        near-exact buddy substitution (which would change routing and the output).
        """
        with tempfile.TemporaryDirectory() as d:
            path = builder(d)
            stock = _stock_greedy(path, PROMPT, N)

            # ORACLE tape: replays the model's own greedy -> every block FULLY
            # accepted -> exercises the skip path on every block.
            r_full = self._runner(path, fixed_hotset_experts=4)
            r_full.build_fixed_hotset(PROMPT, override=COVER)
            oracle = StaticTokenBlockSource(
                tokens=list(PROMPT) + list(stock), name="oracle-tape"
            )
            out_full = r_full.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=oracle, block_size=4
            )
            self.assertEqual(out_full["tokens"], stock, "full-accept skip diverged")
            self.assertEqual(out_full["verify_cold_blocks"], 0)
            # The skip path actually fired, and on full-accept blocks only.
            self.assertGreater(
                out_full["full_accept_skips"], 0, "skip path never fired on oracle tape"
            )
            self.assertEqual(out_full["rollback_blocks"], 0)

            # PARTIAL tape: every block accepts j == k-1 (last token flipped) ->
            # exercises the rollback+commit path; bonus re-syncs to greedy.
            r_part = self._runner(path, fixed_hotset_experts=4)
            r_part.build_fixed_hotset(PROMPT, override=COVER)
            out_part = r_part.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=_LastTokenFlipTape(stock), block_size=4
            )
            self.assertEqual(
                out_part["tokens"], stock, "partial-accept (rollback) diverged"
            )
            self.assertEqual(out_part["verify_cold_blocks"], 0)
            self.assertGreater(
                out_part["rollback_blocks"], 0, "rollback path never fired on partial tape"
            )

            # WRONG tape: j == 0 every block -> all rollback, still exact.
            r_wrong = self._runner(path, fixed_hotset_experts=4)
            r_wrong.build_fixed_hotset(PROMPT, override=COVER)
            wrong = StaticTokenBlockSource(
                tokens=list(PROMPT) + [(int(t) + 1) % 128 for t in stock],
                name="wrong-tape",
            )
            out_wrong = r_wrong.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=wrong, block_size=4
            )
            self.assertEqual(out_wrong["tokens"], stock, "wrong-tape diverged")
            self.assertEqual(out_wrong["full_accept_skips"], 0)

    def test_exact_unquantized(self) -> None:
        self._gate(build_tiny_nemotron)

    def test_exact_quantized(self) -> None:
        self._gate(build_tiny_nemotron_quantized)

    def test_exact_cold_verify_skip(self) -> None:
        """A cold verify block that re-verifies + fully accepts still stays exact.

        With a MISSING fixed set every verify block hits the synced re-verify
        fallback. On the oracle tape it then accepts fully — the skip path must
        either fire correctly off the cold-re-verified cache OR fall back to
        rollback, but the OUTPUT must still equal stock greedy either way.
        """
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            stock = _stock_greedy(path, PROMPT, N)
            r = self._runner(path, fixed_hotset_experts=2)
            r.build_fixed_hotset(PROMPT, override=MISS)
            oracle = StaticTokenBlockSource(
                tokens=list(PROMPT) + list(stock), name="oracle-tape"
            )
            out = r.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=oracle, block_size=4
            )
            self.assertEqual(out["tokens"], stock, "cold-verify full-accept diverged")
            self.assertGreater(out["verify_cold_blocks"], 0)

    def test_exact_mixed_skip_and_rollback_in_one_run(self) -> None:
        """A run that MIXES skip + rollback blocks stays exact (skip<->rollback).

        The kept-cache full-accept skip's riskiest interaction is alternating
        between skipping (keep verify cache, carry pending) and rolling back
        (restore + commit). The mixed tape forces both within one decode; the
        output must still equal stock greedy, and BOTH counters must be non-zero.
        """
        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_greedy(path, PROMPT, N)
                    r = self._runner(path, fixed_hotset_experts=4)
                    r.build_fixed_hotset(PROMPT, override=COVER)
                    out = r.generate_greedy_speculative_deferred(
                        PROMPT, N, drafter=_MixedTape(stock), block_size=3
                    )
                    self.assertEqual(out["tokens"], stock, "mixed run diverged")
                    self.assertEqual(out["verify_cold_blocks"], 0)
                    self.assertGreater(out["full_accept_skips"], 0)
                    self.assertGreater(out["rollback_blocks"], 0)

    def test_skip_path_does_one_forward_per_block(self) -> None:
        """Oracle run: full-accept blocks do 1 forward each, not 2.

        Spy ALL stream-forward entrypoints (synced + deferred) and tally
        invocations during decode only (post-prefill). On the oracle tape every
        block is fully accepted, so with the skip optimization each block costs
        exactly ONE forward (the verify), versus TWO without it (verify + commit
        re-forward). We assert total decode forwards == number of blocks (skip),
        strictly fewer than the 2x a rollback-every-block run would pay. Uses the
        COVERING (exact) set so the verify never re-verifies on the synced path —
        so EVERY forward (prefill, verify, commit) goes through the deferred
        entrypoint and we count it alone (it internally calls
        ``_stream_forward_tokens``; spying both would double-count).
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            stock = _stock_greedy(path, PROMPT, N)

            r = self._runner(path, fixed_hotset_experts=4)
            r.build_fixed_hotset(PROMPT, override=COVER)
            oracle = StaticTokenBlockSource(
                tokens=list(PROMPT) + list(stock), name="oracle-tape"
            )

            # Count DEFERRED forwards only (prefill + verify + any commit). Prefill
            # is one; subtract it to isolate decode forwards.
            counter = {"calls": 0}
            orig_def = NemotronHStreamingForwardRunner._stream_forward_tokens_deferred

            def cdef(self, *a, _o=orig_def, _c=counter, **k):
                _c["calls"] += 1
                return _o(self, *a, **k)

            NemotronHStreamingForwardRunner._stream_forward_tokens_deferred = cdef
            self.addCleanup(
                setattr,
                NemotronHStreamingForwardRunner,
                "_stream_forward_tokens_deferred",
                orig_def,
            )

            out = r.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=oracle, block_size=4
            )
            self.assertEqual(out["tokens"], stock)

            blocks = out["blocks"]
            self.assertGreater(blocks, 0)
            # Every block was fully accepted -> skip path on all of them (a final
            # block may break on truncation before classification).
            self.assertEqual(out["rollback_blocks"], 0)
            self.assertGreaterEqual(out["full_accept_skips"], blocks - 1)
            # Decode forwards = total - 1 prefill. Skip path => exactly one forward
            # per block (the verify); the non-skip path would do 2x (verify +
            # commit re-forward).
            decode_forwards = counter["calls"] - 1
            self.assertEqual(
                decode_forwards,
                blocks,
                f"expected {blocks} decode forwards (1/block on skip), got "
                f"{decode_forwards}",
            )


if __name__ == "__main__":
    unittest.main()
