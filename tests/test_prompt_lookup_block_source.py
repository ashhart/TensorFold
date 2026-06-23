"""PromptLookupBlockSource: a REAL (non-oracle) prompt-lookup drafter.

``StaticTokenBlockSource`` replays a KNOWN token tape — for a measurement it is
an oracle (hand it the generation's own greedy output and every block commits in
full by construction, so any acceptance/throughput number it yields is leakage).

``PromptLookupBlockSource`` is the honest counterpart. It is handed ONLY the
generated-so-far context and proposes the next ``k`` tokens by matching the
LONGEST suffix of that context against an EARLIER occurrence in the SAME context
and proposing the tokens that followed the last such match (classic
prompt-lookup / n-gram decoding). No external tape, no future tokens — it sees
exactly what the decoder has already emitted.

Two guarantees are tested here:

1. EXACTNESS (the hard gate). Wired as the ``drafter`` for
   ``generate_greedy_speculative`` AND ``generate_greedy_speculative_deferred``,
   the emitted tokens are bit-identical to plain ``generate_greedy`` (== stock
   mlx_lm) on BOTH tiny fixtures. block_verify guarantees this regardless of
   draft quality: speculation only ever emits the model's own greedy.

2. NO-ORACLE / acceptance. Pure-unit checks that every proposed token comes from
   the context's own history (never a token the source could not have seen), and
   a repetitive toy run reporting the realised accepted-length (modest, NOT the
   100%-accept an oracle tape fakes).
"""
from __future__ import annotations

import tempfile
import unittest

from smarttensor.block_verify import (
    PromptLookupBlockSource,
    StaticTokenBlockSource,
    propose_candidate_block,
)
from tests.fixtures.tiny_nemotron import (
    build_tiny_nemotron,
    build_tiny_nemotron_quantized,
)
from tests.test_nemotron_runner_fixed_hotset import (
    MOE_LAYERS,
    PROMPT,
    _stock_greedy,
)

# A fixed set that COVERS the routed union so every verify block is all-resident
# (pure deferred verify, no cold fallback) — see the spec-deferred test.
COVER = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
N = 12


# --------------------------------------------------------------------------- #
# (2) NO-ORACLE / acceptance — pure unit, no model.
# --------------------------------------------------------------------------- #
class PromptLookupBlockSourceUnitTests(unittest.TestCase):
    """The source proposes ONLY from context history; it never sees the future."""

    def test_is_a_token_block_source(self) -> None:
        """Conforms to the block_verify drafter contract (name + propose)."""
        source = PromptLookupBlockSource()
        self.assertIsInstance(source.name, str)
        self.assertTrue(callable(source.propose))

    def test_no_proposal_without_a_repeated_suffix(self) -> None:
        """A context whose tail never recurred earlier yields no candidate."""
        source = PromptLookupBlockSource(ngram=2)
        self.assertEqual(list(source.propose([1, 2, 3, 4, 5], 4)), [])

    def test_empty_or_zero_budget_is_empty(self) -> None:
        source = PromptLookupBlockSource()
        self.assertEqual(list(source.propose([], 4)), [])
        self.assertEqual(list(source.propose([1, 2, 1, 2], 0)), [])

    def test_proposes_continuation_of_last_earlier_match(self) -> None:
        """``... a b c`` recurring as ``a b`` -> the suffix ``a b`` matched
        earlier was followed by ``c``, so ``c`` leads the proposal."""
        # context: a b c  a b  -> suffix (a b) last occurred at index 0, followed
        # by [c, a, b]. The trailing (a b) itself must NOT match against itself.
        # With budget 3 the whole earlier continuation [c, a, b] is proposed; the
        # FIRST proposed token (c) is the guess for the immediate next token.
        ctx = [10, 11, 12, 10, 11]
        source = PromptLookupBlockSource(ngram=2)
        self.assertEqual(list(source.propose(ctx, 3)), [12, 10, 11])
        # Budget 1 -> just the immediate continuation token.
        self.assertEqual(list(source.propose(ctx, 1)), [12])

    def test_proposed_tokens_are_all_from_history(self) -> None:
        """EVERY proposed token previously appeared in the context (no leakage).

        This is the anti-oracle invariant: a prompt-lookup drafter can only echo
        tokens it has already seen. We drive a repetitive context and assert each
        proposed token is a member of the context it was given — a token the
        source could not have invented or peeked at.
        """
        source = PromptLookupBlockSource(ngram=3)
        # A periodic stream: every continuation the source can propose is a copy
        # of an earlier token, never something outside the seen alphabet.
        ctx: list[int] = [1, 2, 3, 4] * 5
        for cut in range(4, len(ctx)):
            window = ctx[:cut]
            proposal = list(source.propose(window, 4))
            for token in proposal:
                self.assertIn(
                    token,
                    window,
                    f"proposed token {token} was not in the seen context",
                )

    def test_prefers_longest_then_most_recent_match(self) -> None:
        """Longest suffix match wins; the most recent earlier match breaks ties."""
        # Two earlier occurrences of suffix (7 8): the FIRST (idx 1) is part of a
        # longer (6 7 8) match, the SECOND (idx 5) is only (7 8). The current tail
        # is (6 7 8). With ngram=3 the longer 3-gram match (idx 0) wins and its
        # continuation (9) is proposed — longest match beats the more-recent but
        # shorter (7 8) at idx 5 (whose continuation would be 1).
        ctx = [6, 7, 8, 9, 0, 7, 8, 1, 6, 7, 8]
        self.assertEqual(list(PromptLookupBlockSource(ngram=3).propose(ctx, 1)), [9])
        # Capped at ngram=2 it can only see (7 8); the MOST RECENT earlier (idx 5)
        # then wins on the tie -> its continuation (1).
        self.assertEqual(list(PromptLookupBlockSource(ngram=2).propose(ctx, 1)), [1])

    def test_propose_via_block_verify_helper(self) -> None:
        """Works through ``propose_candidate_block`` (the loop's entrypoint)."""
        # suffix (4 5) earlier at idx 0 -> continuation [6, 4, 5] (budget 4 caps
        # at the 3 available continuation tokens).
        ctx = [4, 5, 6, 4, 5]
        source = PromptLookupBlockSource(ngram=2, name="pl")
        result = propose_candidate_block(source, context=ctx, max_tokens=4)
        self.assertEqual(result["tokens"], [6, 4, 5])
        self.assertEqual(result["source"], "pl")

    @staticmethod
    def _run_accept_loop(sequence_next, *, ngram, block_size, seed, rounds):
        """Drive a guarded-decode-style accept loop with a deterministic verifier.

        ``sequence_next(prefix) -> int`` is the ground-truth next token for the
        SEQUENCE (it stands in for the model in a pure-unit harness). The drafter
        is fed ONLY the running prefix — it must rediscover structure from history
        — and we accept exactly as ``block_verify`` would: the matching prefix,
        then the exact bonus token. Returns the per-round accepted-lengths, the
        realised continuation, and the drafter telemetry. The accepted-length is
        whatever the real n-gram matcher earns, NOT the unconditional full-accept
        a known-tape oracle reports by construction.
        """
        source = PromptLookupBlockSource(ngram=ngram)
        emitted = list(seed)
        accepted_lengths: list[int] = []
        for _ in range(rounds):
            proposal = list(source.propose(list(emitted), block_size))
            accepted = 0
            for offset, token in enumerate(proposal):
                if token == sequence_next(emitted + proposal[:offset]):
                    accepted += 1
                else:
                    break
            accepted_lengths.append(accepted)
            committed = proposal[:accepted]
            committed.append(sequence_next(emitted + committed))  # exact bonus
            emitted.extend(committed)
        return accepted_lengths, emitted, source.telemetry()

    def test_reported_accepted_length_clean_period(self) -> None:
        """A perfectly periodic stream: a real n-gram drafter fills the block.

        This is the EASY end of honest acceptance — once the period is in
        history, the suffix match is exact, so AL saturates at the block size.
        That is a genuine number (the drafter saw only the prefix), not leakage:
        the win is real precisely because the text is exactly repetitive.
        """
        period = [1, 2, 3, 4]
        block_size = 4
        accepted_lengths, _, telemetry = self._run_accept_loop(
            lambda prefix: period[len(prefix) % len(period)],
            ngram=3,
            block_size=block_size,
            seed=[period[i % 4] for i in range(8)],
            rounds=6,
        )
        mean_al = sum(accepted_lengths) / len(accepted_lengths)
        self.assertGreater(mean_al, 0.0, "drafter never accepted on a periodic toy")
        self.assertLessEqual(max(accepted_lengths), block_size)
        print(
            f"[prompt-lookup toy: clean period] accepted_lengths={accepted_lengths} "
            f"mean_AL={mean_al:.2f} telemetry={telemetry}"
        )

    def test_reported_accepted_length_quasi_repetitive(self) -> None:
        """A quasi-repetitive stream: a recurring motif with periodic breaks.

        This is the REALISTIC end. The drafter still mines the motif, but every
        few tokens the sequence diverges from any earlier continuation, so the
        block is only partially accepted — a MODEST, data-dependent AL strictly
        between 0 and the block size. That is exactly the honest shape a real
        drafter shows on real text (no oracle full-accept).
        """
        motif = [11, 12, 13]

        def sequence_next(prefix: list[int]) -> int:
            # motif motif ... but every 7th position is a position-dependent
            # "surprise" token that no earlier motif continuation predicts, so a
            # block straddling it cannot be fully accepted.
            i = len(prefix)
            if i % 7 == 6:
                return 90 + (i // 7) % 5  # varies -> not in the motif history
            return motif[i % 3]

        block_size = 4
        seed = [sequence_next(list(range(j))) for j in range(14)]
        accepted_lengths, _, telemetry = self._run_accept_loop(
            sequence_next,
            ngram=3,
            block_size=block_size,
            seed=seed,
            rounds=10,
        )
        mean_al = sum(accepted_lengths) / len(accepted_lengths)
        # Honest middle ground: the drafter helps (mean AL > 0) but the breaks
        # keep it well under a full block. The exact value is reported, not
        # tightly gated, because honest acceptance is data-dependent.
        self.assertGreater(mean_al, 0.0, "drafter earned no acceptance at all")
        self.assertLess(
            mean_al,
            block_size,
            "quasi-repetitive toy fully accepted — the break tokens are not "
            "interrupting acceptance as intended",
        )
        print(
            f"[prompt-lookup toy: quasi-repetitive] accepted_lengths={accepted_lengths} "
            f"mean_AL={mean_al:.2f} telemetry={telemetry}"
        )

    def test_distinct_from_static_oracle_on_unseen_continuation(self) -> None:
        """The honest source CANNOT fabricate an unseen continuation; the oracle
        tape can. This is the leakage difference, made concrete.

        Context ``[1,2,3]`` has no repeated suffix, so prompt-lookup proposes
        nothing. A StaticTokenBlockSource seeded with the *future* ``[4,5,6]``
        would happily propose it — because it was handed the answer.
        """
        ctx = [1, 2, 3]
        honest = PromptLookupBlockSource(ngram=2)
        oracle = StaticTokenBlockSource(tokens=[1, 2, 3, 4, 5, 6])
        self.assertEqual(list(honest.propose(ctx, 3)), [])
        self.assertEqual(list(oracle.propose(ctx, 3)), [4, 5, 6])


# --------------------------------------------------------------------------- #
# (1) EXACTNESS — model-backed, both fixtures, both speculative generators.
# --------------------------------------------------------------------------- #
class PromptLookupBlockSourceExactnessTests(unittest.TestCase):
    """spec(prompt-lookup) == plain greedy == stock, both fixtures."""

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

    def _exact_gate(self, builder) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = builder(d)

            # Ground truth: stock mlx_lm greedy and the runner's own plain greedy.
            stock = _stock_greedy(path, PROMPT, N)
            base = self._runner(path, fixed_hotset_experts=4)
            greedy = base.generate_greedy(PROMPT, N)["tokens"]
            self.assertEqual(greedy, stock, "runner plain greedy diverged from stock")

            def n_diffs(produced: list[int]) -> int:
                # Token-level "max-abs-diff": the count of positions that differ
                # from stock. The exactness gate is n_diffs == 0 (bit-exact).
                return sum(1 for a, b in zip(produced, stock) if a != b) + abs(
                    len(produced) - len(stock)
                )

            # --- Synced speculative path with the HONEST drafter.
            r_sync = self._runner(path, fixed_hotset_experts=4)
            out_sync = r_sync.generate_greedy_speculative(
                PROMPT, N, drafter=PromptLookupBlockSource(), block_size=4
            )
            self.assertEqual(
                out_sync["tokens"],
                stock,
                "spec(prompt-lookup) synced diverged from stock/greedy",
            )
            self.assertEqual(n_diffs(out_sync["tokens"]), 0)  # bit-exact gate
            self.assertEqual(len(out_sync["tokens"]), N)

            # --- Deferred speculative path with the HONEST drafter, constructed
            # via the documented adapter-namespace re-export (the wiring path a
            # caller uses), proving it drops in as ``drafter=`` unchanged.
            from smarttensor.adapters.mlx import (
                PromptLookupBlockSource as AdapterPromptLookupBlockSource,
            )

            r_def = self._runner(path, fixed_hotset_experts=4)
            r_def.build_fixed_hotset(PROMPT, override=COVER)
            out_def = r_def.generate_greedy_speculative_deferred(
                PROMPT, N, drafter=AdapterPromptLookupBlockSource(), block_size=4
            )
            self.assertEqual(
                out_def["tokens"],
                stock,
                "spec(prompt-lookup) deferred diverged from stock/greedy",
            )
            self.assertEqual(n_diffs(out_def["tokens"]), 0)  # bit-exact gate
            self.assertEqual(len(out_def["tokens"]), N)

    def test_exact_unquantized(self) -> None:
        self._exact_gate(build_tiny_nemotron)

    def test_exact_quantized(self) -> None:
        self._exact_gate(build_tiny_nemotron_quantized)

    def test_fresh_drafter_proposes_only_from_running_context(self) -> None:
        """End-to-end no-oracle proof: the drafter is constructed with NO tape,
        sees only the decoder's running context, and the run is still exact.

        We spy ``propose`` to capture every (context, proposal) pair and assert
        each proposed token already appears in the context it was handed — so the
        drafter provably never peeked at a future/unseen token, yet output stayed
        bit-exact. This is the structural guarantee the oracle tape lacks.
        """
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            stock = _stock_greedy(path, PROMPT, N)

            source = PromptLookupBlockSource()
            seen: list[tuple[list[int], list[int]]] = []
            original_propose = source.propose

            def spy(context, max_tokens, _orig=original_propose, _seen=seen):
                proposal = list(_orig(context, max_tokens))
                _seen.append((list(context), proposal))
                return proposal

            source.propose = spy  # type: ignore[method-assign]

            r = self._runner(path, fixed_hotset_experts=4)
            out = r.generate_greedy_speculative(
                PROMPT, N, drafter=source, block_size=4
            )
            self.assertEqual(out["tokens"], stock)

            self.assertTrue(seen, "drafter.propose was never called")
            for context, proposal in seen:
                for token in proposal:
                    self.assertIn(
                        token,
                        context,
                        "drafter proposed a token absent from its context "
                        "(would indicate oracle leakage)",
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
