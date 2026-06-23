"""Block verification helpers for exact candidate-span decoding.

These helpers are intentionally MLX-free. The runtime-specific verifier can
produce greedy target tokens however it wants: serial target replay, route-union
layer streaming, a custom Metal verifier, or a future MLX extension. This module
owns the exact commit/reject contract shared by those paths.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
import inspect
from typing import Protocol
from typing import Any


class TokenBlockSource(Protocol):
    name: str

    def propose(self, context: Sequence[int], max_tokens: int) -> Sequence[int]:
        """Return up to ``max_tokens`` candidate tokens for the current context."""


class GuardedBlockRuntime(Protocol):
    def snapshot(self) -> Any:
        """Capture mutable generation state before speculative block verify."""

    def verify_candidate_block(self, candidate_tokens: Sequence[int]) -> Sequence[int]:
        """Return exact verifier greedy tokens for the candidate block."""

    def restore(self, snapshot: Any) -> None:
        """Restore mutable generation state to ``snapshot``."""

    def fallback_exact(self, token_count: int) -> Sequence[int]:
        """Generate ``token_count`` exact tokens from current state."""


@dataclass
class CallbackGuardedBlockRuntime:
    """Guarded runtime adapter backed by model-specific callbacks."""

    snapshot_callback: Callable[[], Any]
    verify_candidate_block_callback: Callable[[Sequence[int]], Sequence[int]]
    restore_callback: Callable[[Any], None]
    fallback_exact_callback: Callable[[int], Sequence[int]]
    commit_verified_block_callback: Callable[[Any], None] | None = None

    def __init__(
        self,
        *,
        snapshot: Callable[[], Any],
        verify_candidate_block: Callable[[Sequence[int]], Sequence[int]],
        restore: Callable[[Any], None],
        fallback_exact: Callable[[int], Sequence[int]],
        commit_verified_block: Callable[[Any], None] | None = None,
    ) -> None:
        self.snapshot_callback = snapshot
        self.verify_candidate_block_callback = verify_candidate_block
        self.restore_callback = restore
        self.fallback_exact_callback = fallback_exact
        self.commit_verified_block_callback = commit_verified_block

    def snapshot(self) -> Any:
        return self.snapshot_callback()

    def verify_candidate_block(self, candidate_tokens: Sequence[int]) -> Sequence[int]:
        return self.verify_candidate_block_callback(candidate_tokens)

    def restore(self, snapshot: Any) -> None:
        self.restore_callback(snapshot)

    def fallback_exact(self, token_count: int) -> Sequence[int]:
        return self.fallback_exact_callback(token_count)

    def commit_verified_block(self, snapshot: Any) -> None:
        if self.commit_verified_block_callback is not None:
            self.commit_verified_block_callback(snapshot)


@dataclass
class StepwiseGreedyRuntime:
    """Guarded runtime for exact token-by-token greedy verification."""

    snapshot_callback: Callable[[], Any]
    restore_callback: Callable[[Any], None]
    greedy_next_callback: Callable[[], int]
    advance_exact_callback: Callable[[int], None]

    def __init__(
        self,
        *,
        snapshot: Callable[[], Any],
        restore: Callable[[Any], None],
        greedy_next: Callable[[], int],
        advance_exact: Callable[[int], None],
    ) -> None:
        self.snapshot_callback = snapshot
        self.restore_callback = restore
        self.greedy_next_callback = greedy_next
        self.advance_exact_callback = advance_exact

    def snapshot(self) -> Any:
        return self.snapshot_callback()

    def verify_candidate_block(self, candidate_tokens: Sequence[int]) -> Sequence[int]:
        verifier_tokens: list[int] = []
        for token in candidate_tokens:
            verifier = int(self.greedy_next_callback())
            verifier_tokens.append(verifier)
            if verifier != int(token):
                break
            self.advance_exact_callback(int(token))
        return verifier_tokens

    def restore(self, snapshot: Any) -> None:
        self.restore_callback(snapshot)

    def fallback_exact(self, token_count: int) -> Sequence[int]:
        tokens: list[int] = []
        for _ in range(token_count):
            token = int(self.greedy_next_callback())
            tokens.append(token)
            self.advance_exact_callback(token)
        return tokens

    def commit_verified_block(self, snapshot: Any) -> None:
        del snapshot


@dataclass
class StaticTokenBlockSource:
    """Replay a known token tape as a candidate block source."""

    tokens: Sequence[int]
    name: str = "static-token-block"

    def propose(self, context: Sequence[int], max_tokens: int) -> Sequence[int]:
        if max_tokens < 1:
            return []
        offset = _matched_replay_offset(context, self.tokens)
        return [int(token) for token in self.tokens[offset : offset + max_tokens]]


def _matched_replay_offset(context: Sequence[int], tape: Sequence[int]) -> int:
    context_tokens = [int(token) for token in context]
    tape_tokens = [int(token) for token in tape]
    max_len = min(len(context_tokens), len(tape_tokens))
    for length in range(max_len, 0, -1):
        if context_tokens[-length:] == tape_tokens[:length]:
            return length
    return 0


@dataclass
class PromptLookupBlockSource:
    """Honest (non-oracle) prompt-lookup candidate block source.

    The counterpart to :class:`StaticTokenBlockSource`. Where that replays a
    KNOWN tape — for a measurement an oracle, since handing it the generation's
    own greedy output makes every block commit by construction — this source is
    handed ONLY the generated-so-far context and proposes the next ``k`` tokens
    by classic prompt-lookup / n-gram decoding:

    * take the longest suffix of the context (up to ``ngram`` tokens) that
      occurred EARLIER in the SAME context;
    * propose the tokens that FOLLOWED that earlier occurrence (most recent
      earlier occurrence wins ties).

    It has no tape and never sees a future token, so it cannot leak the answer.
    When the tail never recurred it proposes nothing (the loop falls back to one
    exact token). Proposals are only guesses — the block verifier keeps output
    bit-exact regardless of draft quality — so the matcher is deliberately
    aggressive. It satisfies the :class:`TokenBlockSource` protocol AND the
    ``drafter.propose(context, max_draft)`` interface the speculative generators
    use, so it drops in as a ``drafter=`` with no other change.

    MLX-free and stateless across proposals (each call is a fresh scan of the
    context it is handed); ``reset`` / ``observe_result`` / ``telemetry`` exist
    only for drafter-interface parity and minimal acceptance telemetry.
    """

    ngram: int = 3
    min_ngram: int = 1
    name: str = "prompt-lookup-block"

    def __post_init__(self) -> None:
        if self.ngram < 1:
            raise ValueError("ngram must be positive")
        if self.min_ngram < 1:
            raise ValueError("min_ngram must be positive")
        if self.min_ngram > self.ngram:
            raise ValueError("min_ngram must not exceed ngram")
        self.reset()

    def reset(self) -> None:
        self.matches_attempted = 0
        self.matches_found = 0
        self.match_length_total = 0

    def propose(self, context: Sequence[int], max_tokens: int) -> list[int]:
        if max_tokens < 1:
            return []
        tokens = [int(token) for token in context]
        if len(tokens) <= self.min_ngram:
            return []
        self.matches_attempted += 1
        # Longest suffix first: a longer matched n-gram is a more specific (and
        # so stronger) guess. Within an n-gram size the LAST earlier occurrence
        # wins (closest in context = most likely to repeat next).
        upper = min(self.ngram, len(tokens) - 1)
        for size in range(upper, self.min_ngram - 1, -1):
            pattern = tokens[-size:]
            # Earlier occurrences only: the search window excludes the trailing
            # copy of the suffix itself (start can be at most len-size-1).
            for start in range(len(tokens) - size - 1, -1, -1):
                if tokens[start : start + size] == pattern:
                    continuation = tokens[start + size : start + size + max_tokens]
                    if continuation:
                        self.matches_found += 1
                        self.match_length_total += size
                        return continuation
                    break
        return []

    def observe_result(self, proposed: int, accepted: int) -> None:
        # Stateless matcher: nothing to adapt. Present for drafter parity so the
        # speculative loop's optional ``observe_result`` call is a no-op here.
        del proposed, accepted

    def telemetry(self) -> dict[str, Any]:
        return {
            "matches_attempted": self.matches_attempted,
            "matches_found": self.matches_found,
            "average_match_length": (
                self.match_length_total / self.matches_found
                if self.matches_found
                else 0.0
            ),
        }


def propose_candidate_block(
    source: TokenBlockSource,
    *,
    context: Sequence[int],
    max_tokens: int,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Ask a source for one candidate block and render verifier telemetry."""

    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    state = None
    state_fn = getattr(runtime, "draft_state", None) if runtime is not None else None
    stateful_propose = getattr(source, "propose_guarded_with_state", None)
    if callable(state_fn) and callable(stateful_propose):
        state = state_fn()
        tokens = [
            int(token)
            for token in stateful_propose(
                context=context,
                budget=max_tokens,
                **(state or {}),
            )
        ]
    else:
        tokens = [int(token) for token in source.propose(context, max_tokens)]
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    result = {
        "source": getattr(source, "name", type(source).__name__),
        "tokens": tokens,
        "candidate_count": len(tokens),
        "max_tokens": int(max_tokens),
        "complete": len(tokens) == max_tokens and max_tokens > 0,
    }
    if state is not None:
        result["stateful"] = True
    return result


def run_guarded_block_decode(
    source: TokenBlockSource,
    *,
    runtime: GuardedBlockRuntime,
    context: Sequence[int],
    max_tokens: int,
    block_size: int,
    stop_after_reject: bool = False,
    stop_tokens: set[int] | None = None,
    no_candidate_fallback_tokens: int | None = None,
) -> dict[str, Any]:
    """Run a guarded block decode loop around an exact verifier runtime."""

    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if no_candidate_fallback_tokens is not None and int(no_candidate_fallback_tokens) < 1:
        raise ValueError("no_candidate_fallback_tokens must be positive when set")

    emitted: list[int] = []
    events: list[dict[str, Any]] = []
    verifier_telemetry: list[Any] = []
    committed_blocks = 0
    accepted_prefix_blocks = 0
    accepted_prefix_tokens = 0
    restored_blocks = 0
    fallback_blocks = 0
    stopped_after_reject = False
    stopped = False
    verify_exception_recoveries = 0
    stops = {int(token) for token in (stop_tokens or set())}
    generated_context = [int(token) for token in context]

    def trim_at_stop(tokens: Sequence[int]) -> tuple[list[int], bool]:
        rendered = [int(token) for token in tokens]
        if not stops:
            return rendered, False
        for index, token in enumerate(rendered):
            if int(token) in stops:
                return rendered[: index + 1], True
        return rendered, False

    def fallback_exact(token_count: int) -> list[int]:
        fallback_tokens = [
            int(token)
            for token in runtime.fallback_exact(token_count)
        ][:token_count]
        if token_count > 0 and not fallback_tokens:
            raise RuntimeError("exact fallback returned no tokens")
        return fallback_tokens

    def fallback_exact_after_prefix(
        accepted_prefix_tokens: Sequence[int],
        token_count: int,
    ) -> list[int]:
        fallback_with_prefix = getattr(
            runtime,
            "fallback_exact_after_accepted_prefix",
            None,
        )
        if callable(fallback_with_prefix) and accepted_prefix_tokens:
            fallback_tokens = [
                int(token)
                for token in fallback_with_prefix(
                    accepted_prefix=[int(token) for token in accepted_prefix_tokens],
                    token_count=token_count,
                )
            ][:token_count]
            if token_count > 0 and not fallback_tokens:
                raise RuntimeError("exact fallback returned no tokens")
            return fallback_tokens
        return fallback_exact(token_count)

    while len(emitted) < max_tokens:
        remaining = max_tokens - len(emitted)
        request = min(block_size, remaining)
        proposal = propose_candidate_block(
            source,
            context=generated_context,
            max_tokens=request,
            runtime=runtime,
        )
        candidate_tokens = [int(token) for token in proposal["tokens"]]
        if not candidate_tokens:
            fallback_request = request
            if no_candidate_fallback_tokens is not None:
                fallback_request = min(fallback_request, int(no_candidate_fallback_tokens))
            if stops:
                fallback_tokens = []
                hit_stop = False
                for _ in range(fallback_request):
                    token = fallback_exact(1)[0]
                    fallback_tokens.append(token)
                    if int(token) in stops:
                        hit_stop = True
                        break
            else:
                fallback_tokens = fallback_exact(fallback_request)
                fallback_tokens, hit_stop = trim_at_stop(fallback_tokens)
            emitted.extend(fallback_tokens)
            generated_context.extend(fallback_tokens)
            fallback_blocks += 1
            events.append(
                {
                    "action": "fallback_no_candidate",
                    "requested_tokens": request,
                    "fallback_request_tokens": fallback_request,
                    "fallback_tokens": fallback_tokens,
                }
            )
            if hit_stop:
                stopped = True
                break
            continue

        snapshot = runtime.snapshot()
        verify_exception: BaseException | None = None
        verify_exception_recovered = False
        try:
            raw_verifier_tokens = runtime.verify_candidate_block(candidate_tokens)
        except Exception as error:
            recover = getattr(runtime, "recover_verify_candidate_block_error", None)
            if not callable(recover):
                raise
            runtime.restore(snapshot)
            restored_blocks += 1
            raw_verifier_tokens = recover(error, candidate_tokens)
            if raw_verifier_tokens is None:
                raise
            verify_exception = error
            verify_exception_recovered = True
            verify_exception_recoveries += 1
        verifier_tokens = [int(token) for token in raw_verifier_tokens]
        telemetry = _runtime_verify_telemetry(runtime)
        if telemetry is not None:
            verifier_telemetry.append(telemetry)
        decision = block_commit_decision(
            candidate_tokens=candidate_tokens,
            verifier_tokens=verifier_tokens,
        )
        if decision["committed"]:
            committed = [int(token) for token in decision["committed_tokens"]]
            committed, hit_stop = trim_at_stop(committed)
            _commit_verified_block(
                runtime,
                snapshot,
                committed_tokens=committed,
                terminal=(hit_stop or len(emitted) + len(committed) >= max_tokens),
            )
            _after_verified_block_commit(
                runtime,
                committed_tokens=committed,
                terminal=(hit_stop or len(emitted) + len(committed) >= max_tokens),
            )
            emitted.extend(committed)
            generated_context.extend(committed)
            committed_blocks += 1
            event = {
                "action": "commit",
                "candidate_tokens": candidate_tokens,
                "verifier_tokens": verifier_tokens,
                "committed_tokens": committed,
            }
            if verify_exception_recovered:
                event["verify_exception_recovered"] = True
                event["verify_exception_type"] = type(verify_exception).__name__
            if telemetry is not None:
                event["verifier_telemetry"] = telemetry
            events.append(event)
            if hit_stop:
                stopped = True
                break
            continue

        runtime.restore(snapshot)
        restored_blocks += 1
        accepted_count = int(decision["accepted_count"])
        accepted_prefix = [int(token) for token in decision["accepted_prefix"]]
        event = {
            "action": "reject_stop" if stop_after_reject else "restore_and_fallback_exact",
            "candidate_tokens": candidate_tokens,
            "verifier_tokens": verifier_tokens,
            "accepted_prefix": accepted_prefix,
            "accepted_count": accepted_count,
            "reject_offset": decision["reject_offset"],
        }
        if verify_exception_recovered:
            event["verify_exception_recovered"] = True
            event["verify_exception_type"] = type(verify_exception).__name__
        if telemetry is not None:
            event["verifier_telemetry"] = telemetry
        if stop_after_reject:
            stopped_after_reject = True
            events.append(event)
            break
        accepted_prefix_fast_committed = False
        prefix_verifier_tokens: list[int] = []
        allow_fast_prefix_commit = bool(
            getattr(runtime, "allow_fast_prefix_commit", True)
        )
        if accepted_count > 0 and allow_fast_prefix_commit:
            prefix_snapshot = runtime.snapshot()
            prefix_verifier_tokens = [
                int(token)
                for token in runtime.verify_candidate_block(accepted_prefix)
            ]
            prefix_decision = block_commit_decision(
                candidate_tokens=accepted_prefix,
                verifier_tokens=prefix_verifier_tokens,
            )
            if prefix_decision["committed"]:
                accepted_prefix, hit_stop = trim_at_stop(accepted_prefix)
                emitted.extend(accepted_prefix)
                generated_context.extend(accepted_prefix)
                _commit_verified_block(
                    runtime,
                    prefix_snapshot,
                    committed_tokens=accepted_prefix,
                    terminal=(hit_stop or len(emitted) >= max_tokens),
                )
                _after_verified_block_commit(
                    runtime,
                    committed_tokens=accepted_prefix,
                    terminal=(hit_stop or len(emitted) >= max_tokens),
                )
                accepted_prefix_blocks += 1
                accepted_prefix_tokens += len(accepted_prefix)
                accepted_prefix_fast_committed = True
                if hit_stop:
                    event.update({
                        "accepted_prefix_replayed": True,
                        "accepted_prefix_fast_committed": True,
                        "accepted_prefix_verifier_tokens": prefix_verifier_tokens,
                        "fallback_request_tokens": 0,
                        "fallback_tokens": [],
                        "fallback_matches_candidate": False,
                        "verifier_diverged_from_fallback": False,
                    })
                    events.append(event)
                    stopped = True
                    break
            else:
                runtime.restore(prefix_snapshot)
        if accepted_prefix_fast_committed:
            fallback_request = 1
            fallback_tokens = fallback_exact(fallback_request)
            fallback_tokens, hit_stop = trim_at_stop(fallback_tokens)
            fallback_candidate_window = candidate_tokens[
                accepted_count : accepted_count + len(fallback_tokens)
            ]
            fallback_matches_candidate = fallback_tokens == fallback_candidate_window
            verifier_fallback_window = verifier_tokens[
                accepted_count : accepted_count + len(fallback_tokens)
            ]
            verifier_diverged_from_fallback = (
                fallback_matches_candidate
                and verifier_fallback_window != fallback_tokens
            )
            emitted.extend(fallback_tokens)
            generated_context.extend(fallback_tokens)
            fallback_blocks += 1
            event.update({
                "fallback_tokens": fallback_tokens,
                "accepted_prefix_replayed": True,
                "accepted_prefix_fast_committed": True,
                "accepted_prefix_verifier_tokens": prefix_verifier_tokens,
                "fallback_request_tokens": fallback_request,
                "fallback_matches_candidate": fallback_matches_candidate,
                "verifier_diverged_from_fallback": verifier_diverged_from_fallback,
            })
            events.append(event)
            if hit_stop:
                stopped = True
                break
            continue
        fallback_request = min(len(candidate_tokens), accepted_count + 1)
        fallback_tokens = fallback_exact_after_prefix(
            accepted_prefix,
            fallback_request,
        )
        fallback_tokens, hit_stop = trim_at_stop(fallback_tokens)
        replayed_prefix = fallback_tokens[:accepted_count]
        accepted_prefix_replayed = (
            len(replayed_prefix) == accepted_count
            and replayed_prefix == accepted_prefix
        )
        if accepted_count > 0 and accepted_prefix_replayed:
            accepted_prefix_blocks += 1
            accepted_prefix_tokens += accepted_count
        fallback_matches_candidate = fallback_tokens == candidate_tokens[: len(fallback_tokens)]
        verifier_diverged_from_fallback = (
            fallback_matches_candidate
            and verifier_tokens[: len(fallback_tokens)] != fallback_tokens
        )
        emitted.extend(fallback_tokens)
        generated_context.extend(fallback_tokens)
        fallback_blocks += 1
        event.update({
            "fallback_tokens": fallback_tokens,
            "accepted_prefix_replayed": accepted_prefix_replayed,
            "accepted_prefix_fast_committed": False,
            "fallback_request_tokens": fallback_request,
            "fallback_matches_candidate": fallback_matches_candidate,
            "verifier_diverged_from_fallback": verifier_diverged_from_fallback,
        })
        events.append(event)
        if hit_stop:
            stopped = True
            break

    return {
        "emitted_tokens": emitted[:max_tokens],
        "events": events,
        "verifier_telemetry": verifier_telemetry,
        "committed_blocks": committed_blocks,
        "accepted_prefix_blocks": accepted_prefix_blocks,
        "accepted_prefix_tokens": accepted_prefix_tokens,
        "restored_blocks": restored_blocks,
        "fallback_blocks": fallback_blocks,
        "verify_exception_recoveries": verify_exception_recoveries,
        "stopped_after_reject": stopped_after_reject,
        "stopped": stopped,
        "garbage_committed": False,
    }


def _runtime_verify_telemetry(runtime: GuardedBlockRuntime) -> Any | None:
    telemetry_fn = getattr(runtime, "verify_telemetry", None)
    if telemetry_fn is None:
        return None
    telemetry = telemetry_fn()
    if telemetry is None:
        return None
    if isinstance(telemetry, dict):
        return dict(telemetry)
    return telemetry


def _commit_verified_block(
    runtime: GuardedBlockRuntime,
    snapshot: Any,
    *,
    committed_tokens: Sequence[int],
    terminal: bool,
) -> None:
    commit = getattr(runtime, "commit_verified_block", None)
    if commit is None:
        return
    try:
        signature = inspect.signature(commit)
    except (TypeError, ValueError):
        commit(snapshot)
        return
    if "committed_tokens" not in signature.parameters:
        commit(snapshot)
        return
    commit(
        snapshot,
        committed_tokens=[int(token) for token in committed_tokens],
        terminal=bool(terminal),
    )


def _after_verified_block_commit(
    runtime: GuardedBlockRuntime,
    *,
    committed_tokens: Sequence[int],
    terminal: bool,
) -> None:
    cleanup = getattr(runtime, "after_verified_block_commit", None)
    if cleanup is None:
        return
    try:
        signature = inspect.signature(cleanup)
    except (TypeError, ValueError):
        cleanup()
        return
    if "committed_tokens" not in signature.parameters:
        cleanup()
        return
    cleanup(
        committed_tokens=[int(token) for token in committed_tokens],
        terminal=bool(terminal),
    )


def compare_candidate_tokens(
    *,
    candidate_tokens: Sequence[int],
    verifier_tokens: Sequence[int],
) -> dict[str, Any]:
    """Compare a proposed token block with exact verifier greedy tokens."""

    candidates = [int(token) for token in candidate_tokens]
    verifiers = [int(token) for token in verifier_tokens]
    comparisons: list[dict[str, Any]] = []
    for offset, (candidate, verifier) in enumerate(
        zip(candidates, verifiers, strict=False)
    ):
        comparisons.append(
            {
                "offset": offset,
                "candidate": candidate,
                "verifier": verifier,
                "match": candidate == verifier,
            }
        )
    first_mismatch = next(
        (entry for entry in comparisons if not entry["match"]),
        None,
    )
    return {
        "candidate_count": len(candidates),
        "verifier_count": len(verifiers),
        "all_match": (
            len(candidates) == len(verifiers)
            and first_mismatch is None
        ),
        "first_mismatch": first_mismatch,
        "comparisons": comparisons,
    }


def block_commit_decision(
    *,
    candidate_tokens: Sequence[int],
    verifier_tokens: Sequence[int],
) -> dict[str, Any]:
    """Return the exact commit/replay decision for one candidate block."""

    comparison = compare_candidate_tokens(
        candidate_tokens=candidate_tokens,
        verifier_tokens=verifier_tokens,
    )
    candidates = [int(token) for token in candidate_tokens]
    if comparison["all_match"]:
        return {
            "committed": True,
            "accepted_count": len(candidates),
            "accepted_prefix": candidates,
            "committed_tokens": candidates,
            "reject_offset": None,
            "requires_fallback": False,
            "comparison": comparison,
        }

    first_mismatch = comparison["first_mismatch"]
    if first_mismatch is None:
        accepted_count = min(
            int(comparison["candidate_count"]),
            int(comparison["verifier_count"]),
        )
    else:
        accepted_count = int(first_mismatch["offset"])
    return {
        "committed": False,
        "accepted_count": accepted_count,
        "accepted_prefix": candidates[:accepted_count],
        "committed_tokens": [],
        "reject_offset": (
            int(first_mismatch["offset"]) if first_mismatch is not None else None
        ),
        "requires_fallback": True,
        "comparison": comparison,
    }
