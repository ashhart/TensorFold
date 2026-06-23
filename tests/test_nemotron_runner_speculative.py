"""Token-exactness gate for NemotronHStreamingForwardRunner speculative decode.

Batched speculative greedy decoding is a PURE COMPUTE-SKIP: a drafter proposes a
block of k tokens, the runner verifies them in ONE batched forward, accepts the
longest prefix that matches the model's own greedy, and emits the accepted tokens
plus one bonus token. The output MUST equal plain ``generate_greedy`` token-for-
token — speculation only avoids redundant single-token forwards, it never changes
which tokens are produced.

THE GATE (``test_speculative_*``): for the SAME prompt + max_tokens, the
speculative token list equals the plain greedy token list on BOTH the unquantized
and 4-bit quantized toy, under BOTH a correct drafter tape (high acceptance,
exercises the commit/rollback path) and a deliberately-wrong tape (forces
rejection + rollback every block). Passing under both acceptance and rejection
proves the cache snapshot/restore + batched verify off-by-one + re-forward
rollback are all exact.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smarttensor.block_verify import StaticTokenBlockSource
from tests.fixtures.tiny_nemotron import (
    build_tiny_nemotron,
    build_tiny_nemotron_quantized,
)


class _LastTokenFlipTape:
    """Replay the model's own greedy tape but FLIP the last token of each block.

    Proposes the correct greedy continuation of ``context`` for the first
    ``k-1`` positions and a deliberately-wrong ``k``-th token, so against the
    runner's acceptance rule every full block accepts ``j == k-1``: the first
    ``k-1`` drafts match (accepted), the last is rejected, then a bonus is
    emitted. Every block is thus PARTIALLY accepted (``0 < j < k``) — exercising
    the rollback+commit re-forward path while still re-syncing exactly to greedy.
    The output must equal plain greedy regardless, since speculation only ever
    emits the model's own tokens. Shared by the synced + deferred spec tests.
    """

    name = "last-token-flip"

    def __init__(self, greedy: list[int]) -> None:
        self._greedy = [int(t) for t in greedy]

    def propose(self, context, max_tokens: int):
        if max_tokens < 1:
            return []
        emitted = self._emitted_count(context)
        block = [int(t) for t in self._greedy[emitted : emitted + max_tokens]]
        if len(block) == max_tokens:
            block[-1] = (block[-1] + 1) % 128  # corrupt last -> j == k-1
        return block

    def _emitted_count(self, context) -> int:
        ctx = [int(t) for t in context]
        g = self._greedy
        for n in range(min(len(ctx), len(g)), 0, -1):
            if ctx[-n:] == g[:n]:
                return n
        return 0


class _MixedTape:
    """Alternate FULL-accept and PARTIAL-accept blocks to mix both paths in one run.

    On odd ``propose`` calls it returns the exact greedy continuation (full accept
    -> skip path); on even calls it flips the last token (partial accept ->
    rollback path). Exercises the skip -> rollback -> skip transitions within a
    single decode (the riskiest interaction for the kept-cache full-accept skip).
    """

    name = "mixed-tape"

    def __init__(self, greedy: list[int]) -> None:
        self._greedy = [int(t) for t in greedy]
        self._calls = 0

    def propose(self, context, max_tokens: int):
        if max_tokens < 1:
            return []
        ctx = [int(t) for t in context]
        emitted = 0
        for n in range(min(len(ctx), len(self._greedy)), 0, -1):
            if ctx[-n:] == self._greedy[:n]:
                emitted = n
                break
        block = [int(t) for t in self._greedy[emitted : emitted + max_tokens]]
        self._calls += 1
        if len(block) == max_tokens and self._calls % 2 == 0:
            block[-1] = (block[-1] + 1) % 128  # partial-accept block
        return block


class NemotronSpeculativeDecodeTests(unittest.TestCase):
    PROMPT = [5, 9, 1, 17, 3]
    N = 12

    def _runner(self, path, *, page_experts):
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        runner = NemotronHStreamingForwardRunner(
            str(path), pin_policy="all", page_experts=page_experts
        )
        self.addCleanup(runner.close)
        return runner

    def _exact_gate(self, builder, *, page_experts) -> None:
        """spec output == plain greedy, under correct AND wrong drafter tapes."""
        with tempfile.TemporaryDirectory() as d:
            path = builder(d)

            # Plain greedy is the ground truth this speculative run must match.
            base_runner = self._runner(path, page_experts=page_experts)
            greedy = base_runner.generate_greedy(self.PROMPT, self.N)["tokens"]
            self.assertEqual(len(greedy), self.N)

            # --- Correct tape: the prompt-lookup drafter replays a PRIOR greedy
            # generation of this exact prompt, so its proposals match the model's
            # own greedy and most blocks commit in full (high acceptance length).
            correct_tape = list(self.PROMPT) + list(greedy)
            correct_drafter = StaticTokenBlockSource(
                tokens=correct_tape, name="correct-tape"
            )
            spec_runner = self._runner(path, page_experts=page_experts)
            correct = spec_runner.generate_greedy_speculative(
                self.PROMPT, self.N, drafter=correct_drafter, block_size=4
            )
            self.assertEqual(
                correct["tokens"],
                greedy,
                "speculative decode (correct tape) diverged from plain greedy",
            )
            self.assertEqual(len(correct["tokens"]), self.N)

            # --- Wrong tape: proposals never match the model, so EVERY block
            # rejects at offset 0 and falls back to the bonus token. The output
            # must STILL equal plain greedy (proves rollback restores the cache
            # bit-exactly after a fully-rejected batched verify).
            wrong_tape = list(self.PROMPT) + [
                (int(tok) + 1) % 128 for tok in greedy
            ]
            wrong_drafter = StaticTokenBlockSource(
                tokens=wrong_tape, name="wrong-tape"
            )
            wrong_runner = self._runner(path, page_experts=page_experts)
            wrong = wrong_runner.generate_greedy_speculative(
                self.PROMPT, self.N, drafter=wrong_drafter, block_size=4
            )
            self.assertEqual(
                wrong["tokens"],
                greedy,
                "speculative decode (wrong tape) diverged from plain greedy",
            )

    def test_speculative_matches_greedy_unquantized_resident(self) -> None:
        self._exact_gate(build_tiny_nemotron, page_experts=False)

    def test_speculative_matches_greedy_quantized_paged(self) -> None:
        self._exact_gate(build_tiny_nemotron_quantized, page_experts=True)

    def test_acceptance_telemetry(self) -> None:
        """Correct tape accepts >1 tok/block; wrong tape is all-fallback (AL~0).

        The loop reports ``accepted_length`` block stats. With the correct tape
        speculation genuinely fires (mean accepted length per block > 1), proving
        the batched verify committed proposed tokens rather than silently falling
        back. With the wrong tape every proposal is rejected at offset 0 (mean
        accepted length ~0) yet the run still terminates and equals plain greedy.
        """
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            greedy = self._runner(path, page_experts=False).generate_greedy(
                self.PROMPT, self.N
            )["tokens"]

            correct_drafter = StaticTokenBlockSource(
                tokens=list(self.PROMPT) + list(greedy), name="correct-tape"
            )
            correct = self._runner(
                path, page_experts=False
            ).generate_greedy_speculative(
                self.PROMPT, self.N, drafter=correct_drafter, block_size=4
            )
            self.assertEqual(correct["tokens"], greedy)
            self.assertIn("accepted_length_mean", correct)
            self.assertIn("blocks", correct)
            self.assertGreater(
                correct["accepted_length_mean"],
                1.0,
                "correct tape never accepted >1 token/block (no real speculation)",
            )
            # At least one block accepted its full proposal (k draft tokens).
            self.assertGreater(correct["accepted_tokens_total"], 0)

            wrong_tape = list(self.PROMPT) + [
                (int(tok) + 1) % 128 for tok in greedy
            ]
            wrong_drafter = StaticTokenBlockSource(
                tokens=wrong_tape, name="wrong-tape"
            )
            wrong = self._runner(
                path, page_experts=False
            ).generate_greedy_speculative(
                self.PROMPT, self.N, drafter=wrong_drafter, block_size=4
            )
            self.assertEqual(wrong["tokens"], greedy)
            # Every block rejected at offset 0 -> zero accepted draft tokens.
            self.assertEqual(
                wrong["accepted_tokens_total"],
                0,
                "wrong tape accepted a draft token (verify off-by-one?)",
            )
            self.assertAlmostEqual(wrong["accepted_length_mean"], 0.0)

    def test_full_accept_skip_is_exact_and_fewer_forwards(self) -> None:
        """Full-accept-skip: synced spec output == greedy AND skips a re-forward.

        On an oracle (correct) tape every block is fully accepted, so the verify
        forward already advanced the cache to the exact post-block state and the
        rollback+commit re-forward is redundant. Skipping it must keep the output
        bit-identical to plain greedy and cost ONE forward per block (the verify)
        instead of two. We spy ``_stream_forward_tokens`` and check decode
        forwards == blocks.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            greedy = self._runner(path, page_experts=False).generate_greedy(
                self.PROMPT, self.N
            )["tokens"]

            r = self._runner(path, page_experts=False)
            oracle = StaticTokenBlockSource(
                tokens=list(self.PROMPT) + list(greedy), name="oracle-tape"
            )

            counter = {"calls": 0}
            orig = NemotronHStreamingForwardRunner._stream_forward_tokens

            def counting(self, *a, _o=orig, _c=counter, **k):
                _c["calls"] += 1
                return _o(self, *a, **k)

            NemotronHStreamingForwardRunner._stream_forward_tokens = counting
            self.addCleanup(
                setattr,
                NemotronHStreamingForwardRunner,
                "_stream_forward_tokens",
                orig,
            )

            out = r.generate_greedy_speculative(
                self.PROMPT, self.N, drafter=oracle, block_size=4
            )
            self.assertEqual(out["tokens"], greedy, "full-accept skip diverged")
            blocks = out["blocks"]
            self.assertGreater(blocks, 0)
            self.assertEqual(out["rollback_blocks"], 0, "a block rolled back on oracle tape")
            # Every block was full-accept; a final block may break on truncation
            # before classification, so allow one unclassified.
            self.assertGreaterEqual(out["full_accept_skips"], blocks - 1)
            # Decode forwards = total - 1 prefill; skip path => one per block (the
            # verify), vs two (verify + commit) without the optimization.
            self.assertEqual(
                counter["calls"] - 1,
                blocks,
                "full-accept skip did not collapse to one forward per block",
            )

    def test_partial_accept_still_rolls_back_exact(self) -> None:
        """A partial-accept (j<k) tape still rolls back + re-forwards, exactly."""
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            greedy = self._runner(path, page_experts=False).generate_greedy(
                self.PROMPT, self.N
            )["tokens"]
            r = self._runner(path, page_experts=False)
            out = r.generate_greedy_speculative(
                self.PROMPT,
                self.N,
                drafter=_LastTokenFlipTape(greedy),
                block_size=4,
            )
            self.assertEqual(out["tokens"], greedy, "partial-accept diverged")
            self.assertGreater(out["rollback_blocks"], 0)

    def test_block_size_one_matches_greedy(self) -> None:
        """Degenerate k=1: still exact (verify-of-one + bonus == two greedy steps)."""
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            greedy = self._runner(path, page_experts=False).generate_greedy(
                self.PROMPT, self.N
            )["tokens"]
            drafter = StaticTokenBlockSource(
                tokens=list(self.PROMPT) + list(greedy), name="correct-tape"
            )
            res = self._runner(path, page_experts=False).generate_greedy_speculative(
                self.PROMPT, self.N, drafter=drafter, block_size=1
            )
            self.assertEqual(res["tokens"], greedy)

    def test_speculative_exact_with_persistent_table_capped(self) -> None:
        """Persistent expert table stays exact under speculative verify+rollback.

        The riskiest interaction for the persistent per-layer table: the batched
        verify advances over a k-token block, then the cache is rolled back and
        the committed tokens are re-forwarded. The table lives on the runner (not
        the cache), so it legitimately persists across the rollback — but it must
        never change WHICH expert weights a global id maps to, and the cap/evict
        machinery must not corrupt anything. A tight per-layer cap (2) keeps the
        eviction path live during the speculative run; the emitted tokens must
        still equal plain greedy under the SAME config (which itself matches
        stock — see the paged decode-exact test). Run under correct (commit) AND
        wrong (full-reject rollback) tapes.

        (Note: a single batched-verify pass routes to the UNION over its whole
        block, which for this prompt is the full {2,5,7}; with cap=2 < 3 the
        eviction routine pins all routed experts that pass — so dedicated
        eviction-firing coverage lives in the greedy single-token-decode test,
        where per-pass unions are 2-subsets. Here the gate is purely: the capped
        persistent table is exact through speculative rollback.)
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        def runner():
            r = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                persist_expert_tables=True,
                max_resident_experts_per_layer=2,
            )
            self.addCleanup(r.close)
            return r

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)

            greedy_runner = runner()
            greedy = greedy_runner.generate_greedy(self.PROMPT, self.N)["tokens"]
            # The capped persistent table was genuinely active (it assembled
            # rows for this run rather than no-op'ing).
            self.assertGreater(
                greedy_runner._assembled_stats["loaded_experts"],
                0,
                "persistent table never loaded any experts (path inactive)",
            )

            # Correct tape: high acceptance, exercises commit + rollback while
            # the capped table is reused/extended.
            correct = StaticTokenBlockSource(
                tokens=list(self.PROMPT) + list(greedy), name="correct-tape"
            )
            out = runner().generate_greedy_speculative(
                self.PROMPT, self.N, drafter=correct, block_size=4
            )
            self.assertEqual(
                out["tokens"],
                greedy,
                "speculative + capped persistent table diverged from greedy",
            )

            # Wrong tape: full rejection + rollback every block, still exact.
            wrong = StaticTokenBlockSource(
                tokens=list(self.PROMPT) + [(int(t) + 1) % 128 for t in greedy],
                name="wrong-tape",
            )
            wrong_out = runner().generate_greedy_speculative(
                self.PROMPT, self.N, drafter=wrong, block_size=4
            )
            self.assertEqual(
                wrong_out["tokens"],
                greedy,
                "speculative (wrong tape) + persistent-table diverged from greedy",
            )


if __name__ == "__main__":
    unittest.main()
