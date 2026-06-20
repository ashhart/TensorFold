"""Result, telemetry, and policy records for the MLX runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

def resolve_weight_page_policy(
    model_type: str,
    policy: str,
    *,
    has_weight_page_budget: bool,
) -> str:
    """Resolve user-facing row-page cache policy aliases.

    DeepSeek's layer-ordered expert scans are hostile to LRU once the selected
    route is slightly larger than the cache. The frequency policy is the
    measured safe default for that path; other models keep the historical LRU
    default unless the caller opts in.
    """

    if policy == "auto":
        if model_type == "deepseek_v3" and has_weight_page_budget:
            return "frequency"
        return "lru"
    return policy


@dataclass(frozen=True)
class MlxTensorBatch:
    """A group of tensors loaded into MLX arrays."""

    names: tuple[str, ...]
    nbytes: int
    seconds: float
    arrays: dict[str, Any]
    weight_page_cache_bytes: int = 0
    transient_page_bytes: int = 0

    @property
    def tensor_count(self) -> int:
        return len(self.names)

    def to_summary(self) -> dict[str, Any]:
        return {
            "tensor_count": self.tensor_count,
            "nbytes": self.nbytes,
            "seconds": self.seconds,
            "weight_page_cache_bytes": self.weight_page_cache_bytes,
            "transient_page_bytes": self.transient_page_bytes,
            "names": list(self.names),
        }


@dataclass(frozen=True)
class MlxStreamEvent:
    """One resident-set transition in an MLX streaming session."""

    action: str
    seconds: float
    resident_bytes: int
    requested: tuple[str, ...] = ()
    loaded: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    evicted: tuple[str, ...] = ()
    nbytes_loaded: int = 0
    transient_page_bytes: int = 0
    layer: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "layer": self.layer,
            "seconds": self.seconds,
            "resident_bytes": self.resident_bytes,
            "requested_count": len(self.requested),
            "loaded_count": len(self.loaded),
            "skipped_count": len(self.skipped),
            "evicted_count": len(self.evicted),
            "nbytes_loaded": self.nbytes_loaded,
            "transient_page_bytes": self.transient_page_bytes,
            "requested": list(self.requested),
            "loaded": list(self.loaded),
            "skipped": list(self.skipped),
            "evicted": list(self.evicted),
        }


class PassTraceCollector:
    """Accumulate per-bucket wall time across streamed passes (`--trace-pass`).

    Buckets are attributed by call site. MLX is lazy, so "compute" largely
    materializes inside the sync buckets (router_indices_sync, logits_eval,
    select_token_eval, embedding); the residual `python_other` is the
    orchestration tail the Sprint-1 cuts target.
    """

    def __init__(self) -> None:
        self.bucket_seconds: dict[str, float] = {}
        self.bucket_calls: dict[str, int] = {}
        self.pass_seconds = 0.0
        self.passes = 0

    def add(self, bucket: str, seconds: float) -> None:
        self.bucket_seconds[bucket] = self.bucket_seconds.get(bucket, 0.0) + seconds
        self.bucket_calls[bucket] = self.bucket_calls.get(bucket, 0) + 1

    def add_pass(self, seconds: float) -> None:
        self.pass_seconds += seconds
        self.passes += 1

    def to_dict(self) -> dict[str, Any]:
        attributed = sum(self.bucket_seconds.values())
        return {
            "note": (
                "lazy compute materializes inside sync buckets; python_other is "
                "unattributed orchestration time inside streamed passes"
            ),
            "passes": self.passes,
            "pass_seconds": self.pass_seconds,
            "attributed_seconds": attributed,
            "python_other_seconds": max(self.pass_seconds - attributed, 0.0),
            "buckets": {
                name: {
                    "seconds": self.bucket_seconds[name],
                    "calls": self.bucket_calls[name],
                    "ms_per_pass": (
                        1000.0 * self.bucket_seconds[name] / self.passes if self.passes else 0.0
                    ),
                }
                for name in sorted(
                    self.bucket_seconds, key=lambda key: self.bucket_seconds[key], reverse=True
                )
            },
        }



@dataclass(frozen=True)
class MlxForwardResult:
    """Result from a streaming forward pass."""

    prompt: str
    prompt_tokens: int
    completed_layers: int
    total_layers: int
    seconds: float
    next_token: int | None
    next_text: str | None
    resident_peak_bytes: int
    events: tuple[dict[str, Any], ...]

    @property
    def complete(self) -> bool:
        return self.completed_layers == self.total_layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_tokens": self.prompt_tokens,
            "completed_layers": self.completed_layers,
            "total_layers": self.total_layers,
            "complete": self.complete,
            "seconds": self.seconds,
            "next_token": self.next_token,
            "next_text": self.next_text,
            "resident_peak_bytes": self.resident_peak_bytes,
            "events": list(self.events),
        }


@dataclass(frozen=True)
class MlxGenerationResult:
    """Result from a streaming generation loop."""

    prompt: str
    prompt_tokens: int
    generated_tokens: tuple[int, ...]
    generated_text: str
    seconds: float
    resident_peak_bytes: int
    kv_cache_bytes: int
    events: tuple[dict[str, Any], ...]
    expert_prefetch: dict[str, Any] | None = None
    decode_scheduler: dict[str, Any] | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        generated_count = len(self.generated_tokens)
        tokens_per_second = generated_count / self.seconds if self.seconds > 0 else 0.0
        return {
            "prompt": self.prompt,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": list(self.generated_tokens),
            "generated_text": self.generated_text,
            "generated_token_count": generated_count,
            "seconds": self.seconds,
            "tokens_per_second": tokens_per_second,
            "resident_peak_bytes": self.resident_peak_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "expert_prefetch": self.expert_prefetch,
            "decode_scheduler": self.decode_scheduler,
            "finish_reason": self.finish_reason,
            "events": list(self.events),
        }


@dataclass(frozen=True)
class MlxBatchGenerationResult:
    """Result from a batched streaming generation loop."""

    prompts: tuple[str, ...]
    prompt_tokens: int
    generated_tokens: tuple[tuple[int, ...], ...]
    generated_texts: tuple[str, ...]
    seconds: float
    resident_peak_bytes: int
    kv_cache_bytes: int
    events: tuple[dict[str, Any], ...]
    expert_prefetch: dict[str, Any] | None = None
    decode_scheduler: dict[str, Any] | None = None
    finish_reasons: tuple[str, ...] | None = None
    speculative: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        batch_size = len(self.prompts)
        total_generated = sum(len(tokens) for tokens in self.generated_tokens)
        tokens_per_second = total_generated / self.seconds if self.seconds > 0 else 0.0
        return {
            "prompts": list(self.prompts),
            "batch_size": batch_size,
            "prompt_tokens_per_request": self.prompt_tokens,
            "generated_tokens": [list(tokens) for tokens in self.generated_tokens],
            "generated_texts": list(self.generated_texts),
            "generated_token_count": total_generated,
            "seconds": self.seconds,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_per_request": tokens_per_second / batch_size if batch_size else 0.0,
            "resident_peak_bytes": self.resident_peak_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "expert_prefetch": self.expert_prefetch,
            "decode_scheduler": self.decode_scheduler,
            "finish_reasons": list(self.finish_reasons) if self.finish_reasons else None,
            "speculative": self.speculative,
            "events": list(self.events),
        }


@dataclass
class ExpertPrefetchStats:
    """Aggregate telemetry for predictive expert prefetch."""

    attempted_layers: int = 0
    skipped_no_history: int = 0
    skipped_over_cap: int = 0
    skipped_over_budget: int = 0
    table_prefetch_downgrades: int = 0
    full_hits: int = 0
    predicted_rows: int = 0
    true_rows: int = 0
    hit_rows: int = 0
    wasted_rows: int = 0
    missing_rows: int = 0
    prefetched_bytes: int = 0
    wasted_bytes: int = 0
    fallback_bytes: int = 0
    prefetch_load_seconds: float = 0.0
    join_wait_seconds: float = 0.0
    fallback_load_seconds: float = 0.0
    assemble_seconds: float = 0.0
    max_assemble_temporary_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        coverage = self.hit_rows / self.true_rows if self.true_rows else 0.0
        full_hit_rate = self.full_hits / self.attempted_layers if self.attempted_layers else 0.0
        waste_rate = self.wasted_rows / self.predicted_rows if self.predicted_rows else 0.0
        return {
            "attempted_layers": self.attempted_layers,
            "skipped_no_history": self.skipped_no_history,
            "skipped_over_cap": self.skipped_over_cap,
            "skipped_over_budget": self.skipped_over_budget,
            "table_prefetch_downgrades": self.table_prefetch_downgrades,
            "full_hits": self.full_hits,
            "full_hit_rate": full_hit_rate,
            "predicted_rows": self.predicted_rows,
            "true_rows": self.true_rows,
            "hit_rows": self.hit_rows,
            "wasted_rows": self.wasted_rows,
            "missing_rows": self.missing_rows,
            "coverage": coverage,
            "waste_rate": waste_rate,
            "prefetched_bytes": self.prefetched_bytes,
            "wasted_bytes": self.wasted_bytes,
            "fallback_bytes": self.fallback_bytes,
            "prefetch_load_seconds": self.prefetch_load_seconds,
            "join_wait_seconds": self.join_wait_seconds,
            "fallback_load_seconds": self.fallback_load_seconds,
            "assemble_seconds": self.assemble_seconds,
            "max_assemble_temporary_bytes": self.max_assemble_temporary_bytes,
            "hidden_load_seconds": max(self.prefetch_load_seconds - self.join_wait_seconds, 0.0),
            "exposed_wait_seconds": self.join_wait_seconds + self.fallback_load_seconds + self.assemble_seconds,
        }


@dataclass
class DeepSeekExpertSlotArena:
    """Persistent slot table for one DeepSeek routed layer."""

    layer_index: int
    capacity: int
    arrays: dict[str, Any]
    slot_to_expert: list[int | None]
    expert_to_slot: dict[int, int]
    nbytes: int
    last_used: int = 0


@dataclass
class DeepSeekExpertSlotArenaStats:
    """Aggregate telemetry for slot-indexed selected expert tables."""

    attempts: int = 0
    arena_hits: int = 0
    arena_misses: int = 0
    arena_creates: int = 0
    arena_updates: int = 0
    fallback_direct: int = 0
    evictions: int = 0
    selected_rows: int = 0
    hit_rows: int = 0
    missing_rows: int = 0
    loaded_bytes: int = 0
    update_seconds: float = 0.0
    load_seconds: float = 0.0
    slot_update_ops: int = 0

    def to_dict(self, *, resident_bytes: int, arena_count: int, capacity: int) -> dict[str, Any]:
        hit_rate = self.hit_rows / self.selected_rows if self.selected_rows else 0.0
        return {
            "attempts": self.attempts,
            "arena_hits": self.arena_hits,
            "arena_misses": self.arena_misses,
            "arena_creates": self.arena_creates,
            "arena_updates": self.arena_updates,
            "fallback_direct": self.fallback_direct,
            "evictions": self.evictions,
            "selected_rows": self.selected_rows,
            "hit_rows": self.hit_rows,
            "missing_rows": self.missing_rows,
            "hit_rate": hit_rate,
            "loaded_bytes": self.loaded_bytes,
            "load_seconds": self.load_seconds,
            "update_seconds": self.update_seconds,
            "slot_update_ops": self.slot_update_ops,
            "resident_bytes": resident_bytes,
            "arena_count": arena_count,
            "capacity": capacity,
        }


@dataclass
class SpeculativeStats:
    """Telemetry for speculative decoding rounds."""

    rounds: int = 0
    single_steps: int = 0
    verify_passes: int = 0
    refeed_passes: int = 0
    full_accept_rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    draft_seconds: float = 0.0
    snapshot_seconds: float = 0.0
    verify_seconds: float = 0.0
    refeed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        passes = self.single_steps + self.verify_passes + self.refeed_passes
        return {
            "rounds": self.rounds,
            "single_steps": self.single_steps,
            "verify_passes": self.verify_passes,
            "refeed_passes": self.refeed_passes,
            "full_accept_rounds": self.full_accept_rounds,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "emitted_tokens": self.emitted_tokens,
            "acceptance_rate": self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0,
            "streamed_passes": passes,
            "tokens_per_pass": self.emitted_tokens / passes if passes else 0.0,
            "accepted_tokens_per_pass": self.emitted_tokens / passes if passes else 0.0,
            "draft_seconds": self.draft_seconds,
            "snapshot_seconds": self.snapshot_seconds,
            "verify_seconds": self.verify_seconds,
            "refeed_seconds": self.refeed_seconds,
        }


@dataclass
class TreeSpeculativeStats:
    """Telemetry for batched branch verification."""

    rounds: int = 0
    single_steps: int = 0
    verify_passes: int = 0
    refeed_passes: int = 0
    full_accept_rounds: int = 0
    branch_rounds: int = 0
    branches_verified: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    draft_seconds: float = 0.0
    snapshot_seconds: float = 0.0
    verify_seconds: float = 0.0
    refeed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        passes = self.single_steps + self.verify_passes + self.refeed_passes
        return {
            "rounds": self.rounds,
            "single_steps": self.single_steps,
            "verify_passes": self.verify_passes,
            "refeed_passes": self.refeed_passes,
            "full_accept_rounds": self.full_accept_rounds,
            "branch_rounds": self.branch_rounds,
            "branches_verified": self.branches_verified,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "emitted_tokens": self.emitted_tokens,
            "acceptance_rate": self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0,
            "average_branches": self.branches_verified / self.branch_rounds if self.branch_rounds else 0.0,
            "streamed_passes": passes,
            "tokens_per_pass": self.emitted_tokens / passes if passes else 0.0,
            "accepted_tokens_per_pass": self.emitted_tokens / passes if passes else 0.0,
            "draft_seconds": self.draft_seconds,
            "snapshot_seconds": self.snapshot_seconds,
            "verify_seconds": self.verify_seconds,
            "refeed_seconds": self.refeed_seconds,
        }


EXACTNESS_MODES = ("target-verified", "exact-strict")


@dataclass(frozen=True)
class ExactnessModeSettings:
    """Resolved runner settings for a requested exactness mode.

    ``mode`` is the user-visible label; ``sliding_cache`` and ``speculation``
    are the concrete levers the runner is wired with; ``reason`` records why,
    so logs and telemetry can explain the choice without re-deriving it.
    """

    mode: str
    sliding_cache: str
    speculation: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sliding_cache": self.sliding_cache,
            "speculation": self.speculation,
            "reason": self.reason,
        }


def select_exactness_mode(
    model_type: str | None,
    mode: str = "target-verified",
    *,
    sliding_cache: str = "rotating",
) -> ExactnessModeSettings:
    """Map a requested exactness mode to concrete runner settings.

    Decision record (do not relitigate):
    - ``target-verified`` (default): every emitted token came from a
      target-model verification path; rare near-tie differences vs the
      single-token baseline are documented and counted in telemetry. The
      operator's ``--sliding-cache`` choice is honored as given.
    - ``exact-strict`` (the honest bitwise mode):
        * GPT-OSS: speculation ON + ``--sliding-cache temporal`` — proven
          bitwise exact (forensics gate max_abs=0.0). Any other sliding-cache
          request is overridden to ``temporal`` because rotating/kv are not
          bitwise exact under chunked verify.
        * Qwen (qwen3_5_moe) and everything else: speculation OFF — SSM chunk
          exactness is unproven, so bitwise exactness is only guaranteed by
          plain single-token greedy decoding.

    This is pure settings logic so the mode -> settings mapping is testable on
    CPU without touching the GPU or constructing a model.
    """

    if mode not in EXACTNESS_MODES:
        raise ValueError(
            f"unknown exactness mode {mode!r}; expected one of {list(EXACTNESS_MODES)}"
        )

    if mode == "target-verified":
        return ExactnessModeSettings(
            mode=mode,
            sliding_cache=sliding_cache,
            speculation=True,
            reason="target-verified: emitted tokens are target-verified; "
            "near-ties counted in telemetry",
        )

    # exact-strict
    if model_type == "gpt_oss":
        if sliding_cache != "temporal":
            reason = (
                "exact-strict on gpt_oss requires temporal sliding cache "
                f"(overrode requested {sliding_cache!r}); speculation stays on, "
                "bitwise exact (forensics max_abs=0.0)"
            )
        else:
            reason = (
                "exact-strict on gpt_oss: speculation + temporal sliding cache, "
                "bitwise exact (forensics max_abs=0.0)"
            )
        return ExactnessModeSettings(
            mode=mode,
            sliding_cache="temporal",
            speculation=True,
            reason=reason,
        )

    return ExactnessModeSettings(
        mode=mode,
        sliding_cache=sliding_cache,
        speculation=False,
        reason=(
            f"exact-strict on {model_type!r}: speculation off — SSM/chunk "
            "exactness unproven, only single-token greedy is bitwise exact"
        ),
    )


class AdaptiveDraftGate:
    """Escalating gate that pauses speculation on consecutive zero-accept rounds.

    A verify round with zero accepted tokens costs ~2 streamed passes for one
    token — pure loss. A round with >=1 accepted token is at worst break-even
    (j+1 tokens for 2 passes), so it must NOT count against the drafter: the
    earlier <=1 trigger measured rewrite-file 1.15x -> 0.99x by gating spans
    that were paying for themselves.

    Ladder, per request: two consecutive zero-accept rounds trigger a mild
    cooldown; the next trigger a hard cooldown; the third disables speculation
    for the rest of the request. An accepted round resets the streak and walks
    the ladder back one rung, so mixed workloads recover; the disabled rung is
    only reachable through repeated zero-accept evidence.
    """

    TRIGGER_STREAK = 2
    MILD_COOLDOWN = 4
    HARD_COOLDOWN = 16
    DISABLED_LEVEL = 3

    def __init__(self) -> None:
        self.zero_streak = 0
        self.level = 0
        self.cooldown_remaining = 0
        self.triggers = 0
        self.gated_rounds = 0
        self.disabled_rounds = 0
        self.zero_accept_rounds = 0

    @property
    def disabled(self) -> bool:
        return self.level >= self.DISABLED_LEVEL

    def allow_draft(self) -> bool:
        """One round wants to draft: count down cooldowns, refuse when gated."""
        if self.disabled:
            self.gated_rounds += 1
            self.disabled_rounds += 1
            return False
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.gated_rounds += 1
            return False
        return True

    def observe(self, accepted: int) -> None:
        """Record one verify round's accepted-token count."""
        if accepted > 0:
            self.zero_streak = 0
            if self.level > 0 and not self.disabled:
                self.level -= 1
            return
        self.zero_accept_rounds += 1
        self.zero_streak += 1
        if self.zero_streak < self.TRIGGER_STREAK:
            return
        self.zero_streak = 0
        self.level = min(self.level + 1, self.DISABLED_LEVEL)
        self.triggers += 1
        if self.level == 1:
            self.cooldown_remaining = self.MILD_COOLDOWN
        elif self.level == 2:
            self.cooldown_remaining = self.HARD_COOLDOWN

    def telemetry(self) -> dict[str, Any]:
        return {
            "gate_level": self.level,
            "gate_triggers": self.triggers,
            "gate_disabled": self.disabled,
            "gated_rounds": self.gated_rounds,
            "disabled_rounds": self.disabled_rounds,
            "zero_accept_rounds": self.zero_accept_rounds,
            # Each gated round skips a draft that recent evidence says would
            # zero-accept, i.e. a verify+refeed double-pass that emits one
            # token. Single-stepping it instead costs one pass, so each gated
            # round avoids ~1 wasted streamed pass. Conservative lower bound:
            # the real failed round can cost up to 2 passes for one token.
            "estimated_regression_avoided_passes": self.gated_rounds,
        }
