from __future__ import annotations

import unittest
from unittest.mock import patch

from smarttensor.block_verify import (
    CallbackGuardedBlockRuntime,
    StaticTokenBlockSource,
    StepwiseGreedyRuntime,
    block_commit_decision,
    compare_candidate_tokens,
    propose_candidate_block,
    run_guarded_block_decode,
)
from smarttensor.adapters.mlx import StreamingChatRunner


class FakeBlockRuntime:
    def __init__(
        self,
        *,
        verify_outputs: list[list[int]] | None = None,
        fallback_outputs: list[list[int]] | None = None,
        verify_telemetry: dict[str, object] | None = None,
    ) -> None:
        self.verify_outputs = list(verify_outputs or [])
        self.fallback_outputs = list(fallback_outputs or [])
        self.verify_telemetry_payload = verify_telemetry
        self.actions: list[str] = []
        self.snapshots = 0
        self.allow_fast_prefix_commit = True

    def snapshot(self) -> str:
        self.snapshots += 1
        token = f"snapshot-{self.snapshots}"
        self.actions.append("snapshot")
        return token

    def verify_candidate_block(self, candidate_tokens: list[int]) -> list[int]:
        self.actions.append(f"verify:{candidate_tokens}")
        return self.verify_outputs.pop(0)

    def verify_telemetry(self) -> dict[str, object] | None:
        if self.verify_telemetry_payload is not None:
            self.actions.append("telemetry")
        return self.verify_telemetry_payload

    def commit_verified_block(self, snapshot: str) -> None:
        del snapshot
        self.actions.append("commit")

    def restore(self, snapshot: str) -> None:
        self.actions.append(f"restore:{snapshot}")

    def fallback_exact(self, token_count: int) -> list[int]:
        self.actions.append(f"fallback:{token_count}")
        return self.fallback_outputs.pop(0)


class SequentialBlockSource:
    name = "sequential-block-source"

    def __init__(self, blocks: list[list[int]]) -> None:
        self.blocks = list(blocks)

    def propose(self, context: list[int], max_tokens: int) -> list[int]:
        del context
        if not self.blocks:
            return []
        return self.blocks.pop(0)[:max_tokens]


class StatefulBlockSource:
    name = "stateful-block-source"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def propose(self, context: list[int], max_tokens: int) -> list[int]:
        del context, max_tokens
        raise AssertionError("stateful source should receive runtime draft state")

    def propose_guarded_with_state(self, **kwargs: object) -> list[int]:
        self.calls.append(dict(kwargs))
        return [7, 8, 9]


class ContextLatchBlockSource:
    name = "context-latch-block-source"

    def __init__(
        self,
        *,
        latch_suffix: list[int],
        proposal: list[int],
    ) -> None:
        self.latch_suffix = [int(token) for token in latch_suffix]
        self.proposal = [int(token) for token in proposal]
        self.calls: list[list[int]] = []

    def propose(self, context: list[int], max_tokens: int) -> list[int]:
        rendered = [int(token) for token in context]
        self.calls.append(rendered)
        if len(rendered) >= len(self.latch_suffix) and rendered[-len(self.latch_suffix) :] == self.latch_suffix:
            return self.proposal[:max_tokens]
        return []


class TokenAwareCommitRuntime(FakeBlockRuntime):
    def commit_verified_block(
        self,
        snapshot: str,
        *,
        committed_tokens: list[int],
        terminal: bool,
    ) -> None:
        self.actions.append(
            f"commit-tokens:{snapshot}:{committed_tokens}:terminal={terminal}"
        )


class CommitCleanupRuntime(TokenAwareCommitRuntime):
    def after_verified_block_commit(
        self,
        *,
        committed_tokens: list[int],
        terminal: bool,
    ) -> None:
        self.actions.append(f"cleanup:{committed_tokens}:terminal={terminal}")


class PrefixAwareFallbackRuntime(FakeBlockRuntime):
    def fallback_exact_after_accepted_prefix(
        self,
        *,
        accepted_prefix: list[int],
        token_count: int,
    ) -> list[int]:
        self.actions.append(f"fallback-prefix:{accepted_prefix}:{token_count}")
        return [*accepted_prefix, 42][:token_count]


class RecoveringVerifyRuntime(FakeBlockRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.allow_fast_prefix_commit = False

    def verify_candidate_block(self, candidate_tokens: list[int]) -> list[int]:
        self.actions.append(f"verify:{candidate_tokens}")
        raise RuntimeError("route-plan miss")

    def recover_verify_candidate_block_error(
        self,
        error: BaseException,
        candidate_tokens: list[int],
    ) -> list[int]:
        self.actions.append(f"recover:{type(error).__name__}:{candidate_tokens}")
        if candidate_tokens == [13]:
            return [13]
        return [10, 11, 10, 13]

    def fallback_exact_after_accepted_prefix(
        self,
        *,
        accepted_prefix: list[int],
        token_count: int,
    ) -> list[int]:
        self.actions.append(f"fallback-prefix:{accepted_prefix}:{token_count}")
        return [*accepted_prefix, 10][:token_count]


class MinimalStreamingRunner(StreamingChatRunner):
    def _stream_forward_tokens(self, *args, **kwargs):  # pragma: no cover - not invoked
        raise AssertionError("not invoked")

    def _logits_from_hidden(self, *args, **kwargs):  # pragma: no cover - not invoked
        raise AssertionError("not invoked")


class ReplayCountingStreamingRunner(StreamingChatRunner):
    qwen_route_union_fast_prefix_commit_experimental = False

    def __init__(self) -> None:
        self.forward_tokens: list[int] = []
        self.output_projection_hiddens: list[str] = []
        self.after_steps: list[int] = []

    def _before_decode_forward_tokens(
        self,
        step_tokens: list[int],
        *,
        token_step: int,
        measuring: bool,
    ) -> None:
        del token_step, measuring
        self.forward_tokens.extend(int(token) for token in step_tokens)

    def _stream_forward_tokens(
        self,
        token_rows: list[list[int]],
        **kwargs,
    ) -> str:
        del kwargs
        return f"hidden-{int(token_rows[0][0])}"

    def _tokens_from_hidden_for_guarded(self, hidden: str, events: list[dict]) -> list[int]:
        del events
        self.output_projection_hiddens.append(hidden)
        return [len(self.output_projection_hiddens)]

    def _after_decode_forward(
        self,
        *,
        token_step: int,
        measured_token_count: int,
        events: list[dict],
    ) -> None:
        del token_step, events
        self.after_steps.append(int(measured_token_count))

    def _logits_from_hidden(self, *args, **kwargs):  # pragma: no cover - token-only path
        raise AssertionError("not invoked")


class BlockVerifyTests(unittest.TestCase):
    def test_compare_candidate_tokens_accepts_exact_block(self) -> None:
        result = compare_candidate_tokens(
            candidate_tokens=[10, 11, 12],
            verifier_tokens=[10, 11, 12],
        )

        self.assertTrue(result["all_match"])
        self.assertIsNone(result["first_mismatch"])
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["verifier_count"], 3)

    def test_compare_candidate_tokens_reports_first_mismatch(self) -> None:
        result = compare_candidate_tokens(
            candidate_tokens=[10, 99, 12],
            verifier_tokens=[10, 11, 12],
        )

        self.assertFalse(result["all_match"])
        self.assertEqual(
            result["first_mismatch"],
            {
                "offset": 1,
                "candidate": 99,
                "verifier": 11,
                "match": False,
            },
        )

    def test_compare_candidate_tokens_rejects_short_verifier_output(self) -> None:
        result = compare_candidate_tokens(
            candidate_tokens=[10, 11, 12],
            verifier_tokens=[10, 11],
        )

        self.assertFalse(result["all_match"])
        self.assertIsNone(result["first_mismatch"])
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["verifier_count"], 2)

    def test_block_commit_decision_commits_exact_block(self) -> None:
        decision = block_commit_decision(
            candidate_tokens=[10, 11],
            verifier_tokens=[10, 11],
        )

        self.assertTrue(decision["committed"])
        self.assertEqual(decision["accepted_count"], 2)
        self.assertEqual(decision["committed_tokens"], [10, 11])
        self.assertFalse(decision["requires_fallback"])

    def test_block_commit_decision_rejects_from_first_mismatch(self) -> None:
        decision = block_commit_decision(
            candidate_tokens=[10, 99, 12],
            verifier_tokens=[10, 11, 12],
        )

        self.assertFalse(decision["committed"])
        self.assertEqual(decision["accepted_count"], 1)
        self.assertEqual(decision["reject_offset"], 1)
        self.assertEqual(decision["accepted_prefix"], [10])
        self.assertTrue(decision["requires_fallback"])

    def test_static_token_block_source_proposes_budgeted_blocks(self) -> None:
        source = StaticTokenBlockSource([10, 11, 12, 13])

        first = propose_candidate_block(source, context=[1, 2], max_tokens=3)
        second = propose_candidate_block(source, context=[1, 2, 10, 11, 12], max_tokens=3)

        self.assertEqual(first["tokens"], [10, 11, 12])
        self.assertEqual(first["candidate_count"], 3)
        self.assertEqual(first["source"], "static-token-block")
        self.assertEqual(second["tokens"], [13])
        self.assertEqual(second["candidate_count"], 1)

    def test_propose_candidate_block_rejects_empty_candidate(self) -> None:
        source = StaticTokenBlockSource([])

        result = propose_candidate_block(source, context=[1, 2], max_tokens=4)

        self.assertEqual(result["tokens"], [])
        self.assertEqual(result["candidate_count"], 0)
        self.assertFalse(result["complete"])

    def test_propose_candidate_block_passes_runtime_state_to_stateful_source(self) -> None:
        class RuntimeWithDraftState:
            def draft_state(self) -> dict[str, object]:
                return {"runner": "runner", "hidden": "hidden", "logits": "logits"}

        source = StatefulBlockSource()

        result = propose_candidate_block(
            source,
            context=[1, 2],
            max_tokens=2,
            runtime=RuntimeWithDraftState(),
        )

        self.assertEqual(result["tokens"], [7, 8])
        self.assertTrue(result["stateful"])
        self.assertEqual(
            source.calls,
            [
                {
                    "context": [1, 2],
                    "budget": 2,
                    "runner": "runner",
                    "hidden": "hidden",
                    "logits": "logits",
                }
            ],
        )

    def test_guarded_block_decode_commits_matching_candidate_block(self) -> None:
        runtime = FakeBlockRuntime(verify_outputs=[[10, 11, 12]])
        source = StaticTokenBlockSource([10, 11, 12])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=3,
            block_size=3,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11, 12])
        self.assertEqual(result["committed_blocks"], 1)
        self.assertEqual(result["restored_blocks"], 0)
        self.assertEqual(result["fallback_blocks"], 0)
        self.assertEqual(runtime.actions, ["snapshot", "verify:[10, 11, 12]", "commit"])

    def test_guarded_block_decode_trims_committed_block_at_stop_token(self) -> None:
        runtime = TokenAwareCommitRuntime(
            verify_outputs=[[10, 99, 12]],
        )
        source = StaticTokenBlockSource([10, 99, 12])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=3,
            block_size=3,
            stop_tokens={99},
        )

        self.assertEqual(result["emitted_tokens"], [10, 99])
        self.assertTrue(result["stopped"])
        self.assertEqual(
            runtime.actions,
            [
                "snapshot",
                "verify:[10, 99, 12]",
                "commit-tokens:snapshot-1:[10, 99]:terminal=True",
            ],
        )

    def test_guarded_block_decode_fallback_checks_stop_token_incrementally(self) -> None:
        runtime = FakeBlockRuntime(
            fallback_outputs=[[10], [99], [12]],
        )
        source = StaticTokenBlockSource([])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=5,
            block_size=5,
            stop_tokens={99},
        )

        self.assertEqual(result["emitted_tokens"], [10, 99])
        self.assertTrue(result["stopped"])
        self.assertEqual(runtime.actions, ["fallback:1", "fallback:1"])

    def test_guarded_block_decode_passes_tokens_and_terminal_to_token_aware_commit(self) -> None:
        runtime = TokenAwareCommitRuntime(
            verify_outputs=[[10, 11], [12, 13]],
        )
        source = SequentialBlockSource([[10, 11], [12, 13]])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=4,
            block_size=2,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11, 12, 13])
        self.assertEqual(
            runtime.actions,
            [
                "snapshot",
                "verify:[10, 11]",
                "commit-tokens:snapshot-1:[10, 11]:terminal=False",
                "snapshot",
                "verify:[12, 13]",
                "commit-tokens:snapshot-2:[12, 13]:terminal=True",
            ],
        )

    def test_guarded_block_decode_runs_optional_cleanup_after_committed_block(self) -> None:
        runtime = CommitCleanupRuntime(
            verify_outputs=[[10, 11], [12, 13]],
        )
        source = SequentialBlockSource([[10, 11], [12, 13]])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=4,
            block_size=2,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11, 12, 13])
        self.assertEqual(
            runtime.actions,
            [
                "snapshot",
                "verify:[10, 11]",
                "commit-tokens:snapshot-1:[10, 11]:terminal=False",
                "cleanup:[10, 11]:terminal=False",
                "snapshot",
                "verify:[12, 13]",
                "commit-tokens:snapshot-2:[12, 13]:terminal=True",
                "cleanup:[12, 13]:terminal=True",
            ],
        )

    def test_guarded_block_decode_includes_optional_verify_telemetry(self) -> None:
        runtime = FakeBlockRuntime(
            verify_outputs=[[10, 11, 12]],
            verify_telemetry={
                "engine": "route-union",
                "layer_allclose": True,
                "tok_s": 69.69,
            },
        )
        source = StaticTokenBlockSource([10, 11, 12])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=3,
            block_size=3,
        )

        self.assertEqual(
            result["events"][0]["verifier_telemetry"],
            {
                "engine": "route-union",
                "layer_allclose": True,
                "tok_s": 69.69,
            },
        )
        self.assertEqual(result["verifier_telemetry"], [result["events"][0]["verifier_telemetry"]])
        self.assertEqual(
            runtime.actions,
            ["snapshot", "verify:[10, 11, 12]", "telemetry", "commit"],
        )

    def test_guarded_block_decode_replays_only_prefix_plus_correction_on_reject(self) -> None:
        runtime = FakeBlockRuntime(
            verify_outputs=[[10, 11, 12], [10], [12]],
            fallback_outputs=[[11]],
        )
        source = SequentialBlockSource([[10, 99, 12], [12]])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=3,
            block_size=3,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11, 12])
        self.assertEqual(result["committed_blocks"], 1)
        self.assertEqual(result["accepted_prefix_blocks"], 1)
        self.assertEqual(result["accepted_prefix_tokens"], 1)
        self.assertEqual(result["restored_blocks"], 1)
        self.assertEqual(result["fallback_blocks"], 1)
        self.assertFalse(result["garbage_committed"])
        self.assertEqual(
            runtime.actions,
            [
                "snapshot",
                "verify:[10, 99, 12]",
                "restore:snapshot-1",
                "snapshot",
                "verify:[10]",
                "commit",
                "fallback:1",
                "snapshot",
                "verify:[12]",
                "commit",
            ],
        )
        self.assertEqual(result["events"][0]["reject_offset"], 1)
        self.assertEqual(result["events"][0]["fallback_request_tokens"], 1)
        self.assertTrue(result["events"][0]["accepted_prefix_replayed"])
        self.assertTrue(result["events"][0]["accepted_prefix_fast_committed"])

    def test_guarded_block_decode_recommits_accepted_prefix_before_correction(self) -> None:
        runtime = FakeBlockRuntime(
            verify_outputs=[
                [10, 11, 10, 13],
                [10, 11],
                [13],
            ],
            fallback_outputs=[[10, 11, 10], [13]],
        )
        source = SequentialBlockSource([[10, 11, 99, 13], [13]])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=4,
            block_size=4,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11, 10, 13])
        self.assertEqual(result["committed_blocks"], 1)
        self.assertEqual(result["accepted_prefix_blocks"], 1)
        self.assertEqual(result["accepted_prefix_tokens"], 2)
        self.assertEqual(result["fallback_blocks"], 1)
        self.assertEqual(
            runtime.actions,
            [
                "snapshot",
                "verify:[10, 11, 99, 13]",
                "restore:snapshot-1",
                "snapshot",
                "verify:[10, 11]",
                "commit",
                "fallback:1",
                "snapshot",
                "verify:[13]",
                "commit",
            ],
        )
        first_event = result["events"][0]
        self.assertTrue(first_event["accepted_prefix_replayed"])
        self.assertTrue(first_event["accepted_prefix_fast_committed"])
        self.assertEqual(first_event["fallback_request_tokens"], 1)
        self.assertEqual(first_event["fallback_tokens"], [10])

    def test_guarded_block_decode_stops_when_fast_prefix_correction_hits_stop_token(self) -> None:
        runtime = FakeBlockRuntime(
            verify_outputs=[
                [10, 11, 88],
                [10, 11],
            ],
            fallback_outputs=[[88]],
        )
        source = SequentialBlockSource([[10, 11, 99], [12]])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=4,
            block_size=3,
            stop_tokens={88},
        )

        self.assertEqual(result["emitted_tokens"], [10, 11, 88])
        self.assertTrue(result["stopped"])
        self.assertEqual(result["fallback_blocks"], 1)
        self.assertEqual(result["committed_blocks"], 0)
        self.assertEqual(result["accepted_prefix_blocks"], 1)
        self.assertEqual(
            runtime.actions,
            [
                "snapshot",
                "verify:[10, 11, 99]",
                "restore:snapshot-1",
                "snapshot",
                "verify:[10, 11]",
                "commit",
                "fallback:1",
            ],
        )

    def test_guarded_block_decode_respects_runtime_disabling_fast_prefix_commit(self) -> None:
        runtime = FakeBlockRuntime(
            verify_outputs=[
                [10, 11, 10, 13],
                [13],
            ],
            fallback_outputs=[[10, 11, 10], [13]],
        )
        runtime.allow_fast_prefix_commit = False
        source = SequentialBlockSource([[10, 11, 99, 13], [13]])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=4,
            block_size=4,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11, 10, 13])
        self.assertEqual(result["accepted_prefix_blocks"], 1)
        self.assertEqual(result["accepted_prefix_tokens"], 2)
        self.assertEqual(result["fallback_blocks"], 1)
        self.assertEqual(
            runtime.actions,
            [
                "snapshot",
                "verify:[10, 11, 99, 13]",
                "restore:snapshot-1",
                "fallback:3",
                "snapshot",
                "verify:[13]",
                "commit",
            ],
        )
        first_event = result["events"][0]
        self.assertTrue(first_event["accepted_prefix_replayed"])
        self.assertFalse(first_event["accepted_prefix_fast_committed"])
        self.assertEqual(first_event["fallback_request_tokens"], 3)
        self.assertEqual(first_event["fallback_tokens"], [10, 11, 10])

    def test_guarded_block_decode_uses_prefix_aware_fallback_when_available(self) -> None:
        runtime = PrefixAwareFallbackRuntime(
            verify_outputs=[[10, 11, 42], [13]],
            fallback_outputs=[[10, 11, 42], [13]],
        )
        runtime.allow_fast_prefix_commit = False
        source = SequentialBlockSource([[10, 11, 99], [13]])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=4,
            block_size=3,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11, 42, 13])
        self.assertEqual(
            runtime.actions,
            [
                "snapshot",
                "verify:[10, 11, 99]",
                "restore:snapshot-1",
                "fallback-prefix:[10, 11]:3",
                "snapshot",
                "verify:[13]",
                "commit",
            ],
        )

    def test_guarded_block_decode_restores_before_recovering_verify_exception(self) -> None:
        runtime = RecoveringVerifyRuntime()
        source = SequentialBlockSource([[10, 11, 99, 13], [13]])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=4,
            block_size=4,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11, 10, 13])
        self.assertEqual(result["verify_exception_recoveries"], 2)
        self.assertEqual(result["accepted_prefix_blocks"], 1)
        self.assertEqual(result["fallback_blocks"], 1)
        self.assertEqual(
            runtime.actions,
            [
                "snapshot",
                "verify:[10, 11, 99, 13]",
                "restore:snapshot-1",
                "recover:RuntimeError:[10, 11, 99, 13]",
                "restore:snapshot-1",
                "fallback-prefix:[10, 11]:3",
                "snapshot",
                "verify:[13]",
                "restore:snapshot-2",
                "recover:RuntimeError:[13]",
                "commit",
            ],
        )
        first_event = result["events"][0]
        self.assertTrue(first_event["verify_exception_recovered"])
        self.assertEqual(first_event["verify_exception_type"], "RuntimeError")
        self.assertFalse(first_event["accepted_prefix_fast_committed"])

    def test_guarded_block_decode_can_stop_after_first_reject_without_fallback(self) -> None:
        runtime = FakeBlockRuntime(
            verify_outputs=[[10, 11, 12]],
            fallback_outputs=[[10, 11]],
        )
        source = StaticTokenBlockSource([10, 99, 12])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=3,
            block_size=3,
            stop_after_reject=True,
        )

        self.assertEqual(result["emitted_tokens"], [])
        self.assertEqual(result["committed_blocks"], 0)
        self.assertEqual(result["restored_blocks"], 1)
        self.assertEqual(result["fallback_blocks"], 0)
        self.assertTrue(result["stopped_after_reject"])
        self.assertEqual(result["events"][0]["action"], "reject_stop")
        self.assertEqual(result["events"][0]["accepted_count"], 1)
        self.assertEqual(
            runtime.actions,
            ["snapshot", "verify:[10, 99, 12]", "restore:snapshot-1"],
        )

    def test_guarded_block_decode_marks_when_fallback_matches_rejected_candidate(self) -> None:
        runtime = FakeBlockRuntime(
            verify_outputs=[[10, 11], [10]],
            fallback_outputs=[[99]],
        )
        source = StaticTokenBlockSource([10, 99, 12])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=2,
            block_size=3,
        )

        self.assertEqual(result["emitted_tokens"], [10, 99])
        self.assertEqual(result["events"][0]["fallback_tokens"], [99])
        self.assertEqual(result["events"][0]["fallback_request_tokens"], 1)
        self.assertTrue(result["events"][0]["fallback_matches_candidate"])
        self.assertTrue(result["events"][0]["verifier_diverged_from_fallback"])

    def test_guarded_block_decode_falls_back_when_source_has_no_candidate(self) -> None:
        runtime = FakeBlockRuntime(fallback_outputs=[[10, 11]])
        source = StaticTokenBlockSource([])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=2,
            block_size=4,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11])
        self.assertEqual(result["committed_blocks"], 0)
        self.assertEqual(result["fallback_blocks"], 1)
        self.assertEqual(runtime.actions, ["fallback:2"])

    def test_guarded_block_decode_can_reprobe_after_capped_no_candidate_fallback(self) -> None:
        runtime = FakeBlockRuntime(
            verify_outputs=[[12, 13]],
            fallback_outputs=[[10], [11]],
        )
        source = ContextLatchBlockSource(latch_suffix=[10, 11], proposal=[12, 13])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=4,
            block_size=4,
            no_candidate_fallback_tokens=1,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11, 12, 13])
        self.assertEqual(result["fallback_blocks"], 2)
        self.assertEqual(result["committed_blocks"], 1)
        self.assertEqual(
            runtime.actions,
            ["fallback:1", "fallback:1", "snapshot", "verify:[12, 13]", "commit"],
        )
        self.assertEqual(source.calls, [[1, 2], [1, 2, 10], [1, 2, 10, 11]])

    def test_guarded_block_decode_fails_closed_when_fallback_returns_no_tokens(self) -> None:
        runtime = FakeBlockRuntime(fallback_outputs=[[]])
        source = StaticTokenBlockSource([])

        with self.assertRaisesRegex(RuntimeError, "exact fallback returned no tokens"):
            run_guarded_block_decode(
                source,
                runtime=runtime,
                context=[1, 2],
                max_tokens=2,
                block_size=4,
            )

    def test_callback_guarded_runtime_plugs_generation_callbacks(self) -> None:
        actions: list[str] = []

        runtime = CallbackGuardedBlockRuntime(
            snapshot=lambda: actions.append("snapshot") or "state-1",
            verify_candidate_block=(
                lambda candidate: actions.append(f"verify:{list(candidate)}") or [10, 11]
            ),
            restore=lambda snapshot: actions.append(f"restore:{snapshot}"),
            fallback_exact=lambda token_count: (
                actions.append(f"fallback:{token_count}") or [10] * token_count
            ),
            commit_verified_block=lambda snapshot: actions.append(f"commit:{snapshot}"),
        )
        source = StaticTokenBlockSource([10, 11])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=2,
            block_size=2,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11])
        self.assertEqual(result["committed_blocks"], 1)
        self.assertEqual(actions, ["snapshot", "verify:[10, 11]", "commit:state-1"])

    def test_stepwise_greedy_runtime_restores_before_exact_fallback(self) -> None:
        tape = [10, 11, 12]
        state = {"offset": 0}
        actions: list[str] = []

        def snapshot() -> int:
            actions.append(f"snapshot:{state['offset']}")
            return int(state["offset"])

        def restore(snapshot_offset: int) -> None:
            actions.append(f"restore:{snapshot_offset}")
            state["offset"] = snapshot_offset

        def greedy_next() -> int:
            token = tape[int(state["offset"])]
            actions.append(f"greedy:{token}")
            return token

        def advance_exact(token: int) -> None:
            actions.append(f"advance:{token}")
            state["offset"] += 1

        runtime = StepwiseGreedyRuntime(
            snapshot=snapshot,
            restore=restore,
            greedy_next=greedy_next,
            advance_exact=advance_exact,
        )
        source = StaticTokenBlockSource([10, 99])

        result = run_guarded_block_decode(
            source,
            runtime=runtime,
            context=[1, 2],
            max_tokens=2,
            block_size=2,
        )

        self.assertEqual(result["emitted_tokens"], [10, 11])
        self.assertEqual(result["restored_blocks"], 1)
        self.assertEqual(result["fallback_blocks"], 1)
        self.assertEqual(state["offset"], 2)
        self.assertEqual(
            actions,
            [
                "snapshot:0",
                "greedy:10",
                "advance:10",
                "greedy:11",
                "restore:0",
                "snapshot:0",
                "greedy:10",
                "advance:10",
                "greedy:11",
                "advance:11",
            ],
        )


if __name__ == "__main__":
    unittest.main()
