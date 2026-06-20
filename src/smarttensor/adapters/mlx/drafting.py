"""Drafting and cache-transaction helpers for generated-token verification."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

def prompt_lookup_draft(
    context: list[int],
    max_draft: int,
    *,
    max_ngram: int = 4,
    min_ngram: int = 2,
) -> list[int]:
    """Propose draft tokens by continuing the most recent earlier n-gram match.

    This is draft-model-free speculation: when the tail of the context already
    appeared earlier (repeated identifiers, echoed code, quoted text), the
    tokens that followed it last time are a strong guess for what comes next.
    """

    if max_draft <= 0:
        return []
    for ngram in range(max_ngram, min_ngram - 1, -1):
        if len(context) <= ngram:
            continue
        pattern = context[-ngram:]
        for start in range(len(context) - ngram - 1, -1, -1):
            if context[start : start + ngram] == pattern:
                continuation = context[start + ngram : start + ngram + max_draft]
                if continuation:
                    return list(continuation)
                break
    return []


def count_accepted_drafts(
    draft_tokens: list[int],
    greedy_tokens: list[int],
    *,
    gaps: list[float] | None = None,
    margin: float = 0.0,
) -> int:
    """Length of the draft prefix the target's greedy choices agree with.

    ``greedy_tokens[i]`` is the target argmax at the position that predicts
    ``draft_tokens[i]``. When ``gaps`` (top-1 minus top-2 logit per position)
    and a ``margin`` are given, near-tie agreements are rejected too: the
    verify pass evaluates tokens in one chunk, whose numerics can flip
    near-ties relative to single-step decoding, and drafters propose biased
    continuations — so ambiguous positions must go through the careful path.
    """

    accepted = 0
    for index, (draft_token, greedy_token) in enumerate(zip(draft_tokens, greedy_tokens)):
        if draft_token != greedy_token:
            break
        if gaps is not None and margin > 0 and gaps[index] < margin:
            break
        accepted += 1
    return accepted


def _copy_cache_value(value: Any) -> Any:
    """Deep-ish copy of one cache attribute for an exact rollback snapshot.

    mlx_lm caches mutate in place: KVCache index-assigns into a preallocated
    key/value buffer, and several caches track ``offset``/``lengths`` ints
    alongside the arrays. ``.state`` only exposes the array payload, so a true
    snapshot must copy every attribute, forcing a real copy of each array
    (MLX index assignment mutates the original buffer otherwise).
    """

    import mlx.core as mx

    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, list):
        return [_copy_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_cache_value(item) for item in value)
    return value


def snapshot_cache_states(cache: list[Any]) -> list[dict[str, Any]]:
    """Capture full cache object state so a speculative pass can be rolled back.

    SSM/linear-attention recurrent state cannot be trimmed after the fact, so
    partial draft acceptance on a hybrid model requires restoring the exact
    pre-verify state. Each cache item's whole ``__dict__`` is copied (arrays
    forced to fresh copies); SSM states are tiny and KV buffers are the only
    sizeable copy.
    """

    import mlx.core as mx

    snapshots: list[dict[str, Any]] = []
    arrays_to_eval: list[Any] = []
    for item in cache:
        copied = {key: _copy_cache_value(value) for key, value in vars(item).items()}
        snapshots.append(copied)
        arrays_to_eval.extend(value for value in copied.values() if isinstance(value, mx.array))
    if arrays_to_eval:
        mx.eval(arrays_to_eval)
    return snapshots


def restore_cache_states(cache: list[Any], snapshots: list[dict[str, Any]]) -> None:
    if len(cache) != len(snapshots):
        raise ValueError("cache/state length mismatch")
    for item, snapshot in zip(cache, snapshots):
        for key, value in snapshot.items():
            setattr(item, key, _copy_cache_value(value))


def _repeat_cache_value(value: Any, batch_size: int) -> Any:
    import mlx.core as mx

    if isinstance(value, mx.array):
        if len(value.shape) > 0 and value.shape[0] == 1 and batch_size > 1:
            return mx.concatenate([value] * batch_size, axis=0)
        return mx.array(value)
    if isinstance(value, list):
        return [_repeat_cache_value(item, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(_repeat_cache_value(item, batch_size) for item in value)
    return value


def restore_repeated_cache_states(
    cache: list[Any],
    snapshots: list[dict[str, Any]],
    batch_size: int,
) -> None:
    if len(cache) != len(snapshots):
        raise ValueError("cache/state length mismatch")
    for item, snapshot in zip(cache, snapshots):
        for key, value in snapshot.items():
            setattr(item, key, _repeat_cache_value(value, batch_size))


def _select_cache_value(value: Any, row: int) -> Any:
    import mlx.core as mx

    if isinstance(value, mx.array):
        if len(value.shape) > 0 and value.shape[0] > row:
            return value[row : row + 1]
        return mx.array(value)
    if isinstance(value, list):
        return [_select_cache_value(item, row) for item in value]
    if isinstance(value, tuple):
        return tuple(_select_cache_value(item, row) for item in value)
    return value


def restore_cache_batch_row(target_cache: list[Any], batch_cache: list[Any], row: int) -> None:
    if len(target_cache) != len(batch_cache):
        raise ValueError("cache length mismatch")
    snapshots: list[dict[str, Any]] = []
    for item in batch_cache:
        snapshots.append({key: _select_cache_value(value, row) for key, value in vars(item).items()})
    restore_cache_states(target_cache, snapshots)


class CacheTransaction:
    """Transactional boundary around mutable generation cache state.

    Speculative paths must never leave durable generation state mutated
    unless the verifier committed it: snapshot up front, run the speculative
    block, then either ``commit()`` (keep the mutated cache) or ``rollback()``
    (restore the pre-transaction state exactly). Hybrid caches make this
    mandatory — recurrent SSM state cannot be trimmed after the fact.
    """

    def __init__(self, cache: list[Any]) -> None:
        self._cache = cache
        self._snapshot: list[dict[str, Any]] | None = snapshot_cache_states(cache)

    @property
    def open(self) -> bool:
        return self._snapshot is not None

    def commit(self) -> None:
        if self._snapshot is None:
            raise RuntimeError("transaction already closed")
        self._snapshot = None

    def rollback(self) -> None:
        if self._snapshot is None:
            raise RuntimeError("transaction already closed")
        restore_cache_states(self._cache, self._snapshot)
        self._snapshot = None


class PromptLookupDrafter:
    """Token-level LZ drafter: continue the longest recent suffix match.

    Maintains an incremental n-gram index over the whole context — system and
    user prompt, tool/file content, and generated text alike. Each proposal
    finds where the current suffix occurred before, prefers the longest match
    (most recent on ties), and proposes the tokens that followed it.
    Proposals are only guesses — exact verification keeps output correct —
    so the matcher is deliberately aggressive. Repeated full rejections shrink
    the proposal budget to cut wasted verify passes; acceptance restores it.
    """

    name = "prompt-lookup"
    MATCH_EXTENSION_CAP = 64

    def __init__(
        self,
        *,
        ngram: int = 3,
        max_candidates: int = 8,
        min_draft: int = 2,
    ) -> None:
        if ngram < 1:
            raise ValueError("ngram must be positive")
        self.ngram = ngram
        self.max_candidates = max_candidates
        self.min_draft = min_draft
        # Tokens mined from earlier requests in the same serve session
        # (prior prompts and prior model outputs). They are prepended to the
        # per-request context so historical spans stay mineable across turns;
        # reset() preserves them so a new request keeps the session index.
        self._session_prefix: list[int] = []
        self.reset()

    def reset(self) -> None:
        self._index: dict[tuple[int, ...], list[int]] = {}
        self._indexed_count = 0
        self._last_indexed_token: int | None = None
        self._draft_scale = 1.0
        self.matches_attempted = 0
        self.matches_found = 0
        self.match_length_total = 0

    def seed_session(self, tokens: list[int]) -> None:
        """Fold tokens from a prior request into the persistent session index.

        Used by the server between requests so prompt-lookup mines the whole
        session — earlier prompts and earlier completions — not just the
        current request's context. Proposals are still guesses verified
        exactly downstream, so growing the corpus only ever adds candidates;
        it cannot change emitted tokens.
        """

        if not tokens:
            return
        self._session_prefix.extend(int(token) for token in tokens)
        # The combined-context view changed underneath any built index; force a
        # rebuild on the next proposal by invalidating the indexed cursor.
        self._index = {}
        self._indexed_count = 0
        self._last_indexed_token = None

    def reset_session(self) -> None:
        """Drop all cross-request history (new session / context cleared)."""

        self._session_prefix = []
        self.reset()

    def _combined(self, context: list[int]) -> list[int]:
        """The token sequence the index and proposals operate over."""

        if not self._session_prefix:
            return context
        return self._session_prefix + context

    def _extend_index(self, context: list[int]) -> None:
        start = self._indexed_count
        if start > len(context) or (
            start and context[start - 1] != self._last_indexed_token
        ):
            # Context changed underneath us (no reset between requests):
            # rebuild the index but keep telemetry and the adaptive scale.
            stats = (self.matches_attempted, self.matches_found, self.match_length_total)
            scale = self._draft_scale
            self.reset()
            self.matches_attempted, self.matches_found, self.match_length_total = stats
            self._draft_scale = scale
            start = 0
        smallest = min(2, self.ngram)
        for position in range(max(start, smallest - 1), len(context)):
            # Index every n-gram size the tree drafter can query: shorter
            # fallback lookups are dead unless their keys exist in the index.
            for size in range(smallest, self.ngram + 1):
                if position - size + 1 < 0:
                    continue
                key = tuple(context[position - size + 1 : position + 1])
                positions = self._index.setdefault(key, [])
                positions.append(position)
                if len(positions) > self.max_candidates * 4:
                    del positions[0]
        self._indexed_count = len(context)
        self._last_indexed_token = context[-1] if context else None

    def _suffix_match_length(self, context: list[int], position: int) -> int:
        length = self.ngram
        earlier = position - self.ngram
        suffix = len(context) - 1 - self.ngram
        while (
            earlier >= 0
            and suffix >= 0
            and length < self.MATCH_EXTENSION_CAP
            and context[earlier] == context[suffix]
        ):
            earlier -= 1
            suffix -= 1
            length += 1
        return length

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        if max_draft <= 0 or not context:
            return []
        combined = self._combined(context)
        if len(combined) <= self.ngram:
            return []
        self._extend_index(combined)
        self.matches_attempted += 1

        key = tuple(combined[-self.ngram :])
        candidates = [
            position
            for position in self._index.get(key, ())
            if position < len(combined) - 1
        ]
        if not candidates:
            return []

        best_position = -1
        best_length = -1
        for position in candidates[-self.max_candidates :]:
            length = self._suffix_match_length(combined, position)
            if length >= best_length:
                best_position = position
                best_length = length

        budget = min(max_draft, max(self.min_draft, round(max_draft * self._draft_scale)))
        continuation = combined[best_position + 1 : best_position + 1 + budget]
        if not continuation:
            return []
        self.matches_found += 1
        self.match_length_total += best_length
        return list(continuation)

    def observe_result(self, proposed: int, accepted: int) -> None:
        if proposed <= 0:
            return
        if accepted >= proposed:
            self._draft_scale = min(1.0, self._draft_scale * 2.0)
        elif accepted == 0:
            self._draft_scale = max(0.25, self._draft_scale * 0.5)

    def telemetry(self) -> dict[str, Any]:
        return {
            "matches_attempted": self.matches_attempted,
            "matches_found": self.matches_found,
            "average_match_length": (
                self.match_length_total / self.matches_found if self.matches_found else 0.0
            ),
        }


class PromptLookupTreeDrafter(PromptLookupDrafter):
    """Return several historical continuations for batched tree verification."""

    name = "prompt-lookup-tree"

    def propose_branches(
        self,
        context: list[int],
        max_draft: int,
        max_branches: int,
    ) -> list[list[int]]:
        if max_draft <= 0 or max_branches <= 0 or len(context) <= 1:
            return []
        combined = self._combined(context)
        self._extend_index(combined)
        self.matches_attempted += 1

        # One verifier lane per distinct historical continuation. Mine the
        # combined session corpus (prior requests + this context) at every
        # n-gram size; longer keys rank first (a more specific match is a
        # stronger guess). A continuation shorter than max_draft is still a
        # valid lane: the old `len == max_draft` filter discarded every match
        # whose source span ran into the end of the corpus, which on
        # copy-edit-class text is most of them — that starved the verifier.
        ranked: list[tuple[int, int, list[int]]] = []
        upper = len(combined) - 1
        for ngram in range(min(self.ngram, len(combined) - 1), 0, -1):
            key = tuple(combined[-ngram:])
            for position in self._index.get(key, ())[-self.max_candidates * 4 :]:
                if position >= upper:
                    continue
                continuation = combined[position + 1 : position + 1 + max_draft]
                if continuation:
                    ranked.append((ngram, position, list(continuation)))

        if not ranked:
            return []
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        branches: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for _, _, continuation in ranked:
            key = tuple(continuation)
            # Two source spans can yield the identical continuation; collapse
            # them so each verifier lane carries a distinct guess (a duplicate
            # lane is pure wasted verify width — see the verify-width probe).
            if key in seen:
                continue
            seen.add(key)
            branches.append(continuation)
            if len(branches) >= max_branches:
                break
        if branches:
            self.matches_found += 1
            self.match_length_total += len(branches[0])
        return branches


class ModelDrafter:
    """Drafter backed by a small fully-resident MLX model."""

    name = "model"

    def __init__(self, model_dir: str | Path, target_tokenizer: Any) -> None:
        from mlx_lm import load

        self.model_dir = Path(model_dir)
        self.model, self.tokenizer = load(self.model_dir)
        self._verify_tokenizer(target_tokenizer)
        self.cache: list[Any] | None = None
        self.fed = 0

    def _verify_tokenizer(self, target_tokenizer: Any) -> None:
        probes = ("Hello, world!", "def main():\n    return 0", "The 42 robots painted.")
        for probe in probes:
            if list(self.tokenizer.encode(probe)) != list(target_tokenizer.encode(probe)):
                raise ValueError(
                    f"draft model tokenizer at {self.model_dir} does not match the target tokenizer"
                )

    def reset(self) -> None:
        self.cache = None
        self.fed = 0

    def _forward(self, token_ids: list[int]) -> Any:
        import mlx.core as mx

        logits = self.model(mx.array([token_ids]), cache=self.cache)
        mx.eval(logits)
        return logits

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        import mlx.core as mx

        if max_draft <= 0:
            return []
        if self.cache is None:
            from mlx_lm.models.cache import make_prompt_cache

            self.cache = make_prompt_cache(self.model)
            self.fed = 0

        delta = context[self.fed :]
        if not delta:
            return []
        logits = self._forward(delta)
        self.fed = len(context)

        snapshot = snapshot_cache_states(self.cache)
        drafted: list[int] = []
        try:
            for _ in range(max_draft):
                token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                drafted.append(token)
                if len(drafted) < max_draft:
                    logits = self._forward([token])
        finally:
            restore_cache_states(self.cache, snapshot)
        return drafted


class ExternalProcessDrafter:
    """Drafter backed by a line-oriented helper process.

    This is the bridge for non-MLX sidecars: Core ML/ANE, Swift, or another
    Python runtime can propose guesses while the target MLX model remains the
    exact verifier. The protocol is JSON-lines request/response:

    - ``{"type": "propose", "context": [...], "max_draft": N}``
      -> ``{"tokens": [...]}``
    - ``{"type": "reset"}``, ``{"type": "observe", ...}``,
      ``{"type": "seed_session", ...}``, ``{"type": "reset_session"}``,
      ``{"type": "telemetry"}``
      -> ``{"ok": true}`` or a telemetry object.
    """

    name = "external"

    def __init__(self, command: str | list[str] | tuple[str, ...]) -> None:
        if isinstance(command, str):
            argv = shlex.split(command)
        else:
            argv = [str(part) for part in command]
        if not argv:
            raise ValueError("external drafter command must not be empty")
        self.command = tuple(argv)
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._requests = 0
        self._tokens_proposed = 0

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("external drafter process is not connected")
        if self._process.poll() is not None:
            raise RuntimeError(
                f"external drafter exited with code {self._process.returncode}"
            )
        self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("external drafter closed stdout")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"external drafter returned invalid JSON: {line!r}") from exc
        if not isinstance(response, dict):
            raise RuntimeError("external drafter response must be a JSON object")
        if "error" in response:
            raise RuntimeError(f"external drafter error: {response['error']}")
        return response

    def reset(self) -> None:
        self._request({"type": "reset"})

    def seed_session(self, tokens: list[int]) -> None:
        if tokens:
            self._request({"type": "seed_session", "tokens": [int(token) for token in tokens]})

    def reset_session(self) -> None:
        self._request({"type": "reset_session"})

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        if max_draft <= 0:
            return []
        response = self._request(
            {
                "type": "propose",
                "context": [int(token) for token in context],
                "max_draft": int(max_draft),
            }
        )
        tokens = response.get("tokens", [])
        if not isinstance(tokens, list):
            raise RuntimeError("external drafter response must include list field 'tokens'")
        proposed = [int(token) for token in tokens[:max_draft]]
        self._requests += 1
        self._tokens_proposed += len(proposed)
        return proposed

    def observe_result(self, proposed: int, accepted: int) -> None:
        self._request(
            {
                "type": "observe",
                "proposed": int(proposed),
                "accepted": int(accepted),
            }
        )

    def telemetry(self) -> dict[str, Any]:
        stats = {
            "requests": self._requests,
            "tokens_proposed": self._tokens_proposed,
            "command": list(self.command),
        }
        try:
            response = self._request({"type": "telemetry"})
        except RuntimeError:
            return stats
        if isinstance(response.get("telemetry"), dict):
            stats.update(response["telemetry"])
        return stats

    def close(self) -> None:
        process = self._process
        if process.poll() is None:
            try:
                self._request({"type": "close"})
            except RuntimeError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def expert_merge_plan(
    predicted: list[int],
    true_ids: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Plan an exact expert table from prefetched rows plus a fallback load.

    Returns ``(ordering, hits, missing)`` where ``ordering`` is the row order of
    the assembled table (all predicted rows first, then missing rows), ``hits``
    are true experts covered by the prediction, and ``missing`` are true experts
    that still need a synchronous load.
    """

    predicted_set = set(predicted)
    hits = [expert for expert in true_ids if expert in predicted_set]
    missing = [expert for expert in true_ids if expert not in predicted_set]
    return list(predicted) + missing, hits, missing
