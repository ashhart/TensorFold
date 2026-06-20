"""Architecture-independent streaming decode loops."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .drafting import CacheTransaction, PromptLookupDrafter
from .records import (
    AdaptiveDraftGate,
    MlxBatchGenerationResult,
    MlxGenerationResult,
    MlxStreamEvent,
    PassTraceCollector,
    SpeculativeStats,
    TreeSpeculativeStats,
)
from .utils import compact_event, kv_cache_nbytes

class TemporalSlidingKVCache:
    """Temporal-order sliding cache for exact chunk verification.

    MLX's saturated ``RotatingKVCache`` returns physical ring-buffer order for
    one-token updates and temporal order for chunk updates. GPT-OSS chunk
    verification needs both paths to see the same temporal K/V sequence. This
    cache keeps only the temporal union required by a chunk of length ``N``:
    ``[offset - window + 1, ..., offset + N - 1]``.
    """

    def __init__(self, max_size: int):
        self.keys = None
        self.values = None
        self.offset = 0
        self.start_position = 0
        self.max_size = max_size

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        previous_offset = self.offset
        length = keys.shape[2]
        if self.keys is None:
            combined_keys = keys
            combined_values = values
            combined_start = previous_offset
        else:
            combined_keys = mx.concatenate([self.keys, keys], axis=2)
            combined_values = mx.concatenate([self.values, values], axis=2)
            combined_start = self.start_position

        self.offset = previous_offset + length
        new_start = max(0, previous_offset - self.max_size + 1)
        drop = max(0, new_start - combined_start)
        if drop:
            combined_keys = combined_keys[..., drop:, :]
            combined_values = combined_values[..., drop:, :]

        self.keys = combined_keys
        self.values = combined_values
        self.start_position = new_start
        return self.keys, self.values

    def make_mask(
        self,
        N: int,
        window_size: int | None = None,
        return_array: bool = False,
    ) -> Any:
        if N == 1 and not return_array:
            return None

        import mlx.core as mx

        window = window_size or self.max_size
        start = max(0, self.offset - window + 1)
        end = self.offset + N
        rinds = mx.arange(start, end)
        linds = mx.arange(self.offset, end)[:, None]
        mask = linds >= rinds[None]
        mask = mask & (linds < rinds[None] + window)
        return mask

    def size(self) -> int:
        return self.offset - self.start_position

    def empty(self) -> bool:
        return self.keys is None

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        # Unlike KVCache, this cache's keys/values array IS the exact window
        # (no over-allocated slab indexed by offset): update_and_fetch
        # concatenates new tokens and slices the front, so the invariant
        # keys.shape[2] == offset - start_position must hold. A bare
        # ``offset -= n`` (the KVCache contract) would desync the physical
        # array from offset and corrupt the next update's window math, so the
        # tail must be sliced off the arrays too.
        n = min(self.offset - self.start_position, n)
        if n <= 0:
            return 0
        self.offset -= n
        if self.keys is not None:
            kept = self.keys.shape[2] - n
            if kept <= 0:
                self.keys = None
                self.values = None
                self.start_position = self.offset
            else:
                self.keys = self.keys[..., :kept, :]
                self.values = self.values[..., :kept, :]
        return n

    @property
    def state(self) -> tuple[Any, Any]:
        return self.keys, self.values

    @state.setter
    def state(self, value: tuple[Any, Any]) -> None:
        self.keys, self.values = value
        self.offset = self.start_position + self.keys.shape[2]

    @property
    def meta_state(self) -> tuple[str, str, str]:
        return tuple(map(str, (self.max_size, self.start_position, self.offset)))

    @meta_state.setter
    def meta_state(self, value: tuple[str, str, str]) -> None:
        self.max_size, self.start_position, self.offset = map(int, value)

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class StreamingChatRunner:
    """Shared generation loops over an architecture-specific streamed forward.

    Subclasses provide ``_stream_forward_tokens`` (rows in, hidden out),
    ``_logits_from_hidden``, ``_embed_module``/``_embed_prefix`` for sliced
    embedding loads, and the pin/warm attributes. The loops themselves —
    batched greedy/temperature decoding and speculative decoding with exact
    rollback — are architecture-independent.

    ``_trace`` defaults to the legacy eventful behavior; runners that support
    lean mode override it per instance (lean skips per-layer/per-token event
    construction in the hot loop). ``_pass_trace`` carries the optional
    ``--trace-pass`` bucket collector.
    """

    _trace = True
    _pass_trace: PassTraceCollector | None = None

    def pass_trace(self) -> dict[str, Any] | None:
        return self._pass_trace.to_dict() if self._pass_trace is not None else None

    def _reset_stream_state(self) -> None:
        return None

    def _expert_telemetry(self) -> dict[str, Any] | None:
        return None

    def _make_cache(self) -> list[Any]:
        make_cache = getattr(self.session.model, "make_cache", None)
        if make_cache is not None:
            return make_cache()

        from mlx_lm.models.cache import make_prompt_cache

        core = getattr(self.session.model, "model", self.session.model)
        return make_prompt_cache(core)

    def _pin_for_run(self) -> MlxStreamEvent:
        if self.pin_policy == "phase":
            return self.session.pin_small()
        return self.session.pin()

    def _embed_module(self) -> Any:
        raise NotImplementedError

    def _embedding_slice_names(self) -> tuple[str, ...]:
        cached = getattr(self, "_embedding_names_cache", None)
        if cached is None:
            prefix = self._embed_prefix + "."
            cached = tuple(
                sorted(
                    name
                    for name in self.session.loader.manifest.tensors
                    if name.startswith(prefix)
                )
            )
            self._embedding_names_cache = cached
        return cached

    def _load_embedding_slices(
        self,
        token_rows: list[list[int]],
    ) -> tuple[MlxStreamEvent | None, list[list[int]]]:
        started = time.perf_counter()
        selected_tokens = sorted({int(token_id) for row in token_rows for token_id in row})
        names = self._embedding_slice_names()
        batch = self.session.loader.load_first_dim_slices(
            names,
            selected_tokens,
            evaluate=self.session.evaluate,
        )
        module = self._embed_module()
        for name in names:
            setattr(module, name.rsplit(".", 1)[1], batch.arrays[name])
        self._set_external_resident_bytes(batch.nbytes)

        local_by_token = {token: index for index, token in enumerate(selected_tokens)}
        local_token_rows = [
            [local_by_token[int(token_id)] for token_id in row] for row in token_rows
        ]
        if self._pass_trace is not None:
            self._pass_trace.add("embedding_slice_load", time.perf_counter() - started)
        if not self._trace:
            return None, local_token_rows
        event = MlxStreamEvent(
            action="load-embedding-slices",
            layer=None,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names,
            nbytes_loaded=batch.nbytes,
        )
        self.session.events.append(event)
        return event, local_token_rows

    def _clear_embedding_slices(self) -> None:
        import mlx.core as mx

        module = self._embed_module()
        manifest = self.session.loader.manifest
        for name in self._embedding_slice_names():
            record = manifest.tensors[name]
            empty_shape = (0,) * len(record.shape)
            setattr(
                module,
                name.rsplit(".", 1)[1],
                mx.zeros(empty_shape, dtype=mlx_dtype(record.dtype)),
            )
        self._set_external_resident_bytes()

    def _resident_sidecar_bytes(self) -> int:
        return getattr(self, "_expert_cache_bytes", 0)

    def _set_external_resident_bytes(self, temporary_bytes: int = 0) -> None:
        total = self._resident_sidecar_bytes() + temporary_bytes
        if total:
            self.session.set_external_resident_bytes(total)
        else:
            self.session.clear_external_resident_bytes()

    def _async_lookahead_support(self) -> tuple[bool, str]:
        return False, "runner does not expose an async-lookahead-safe forward"

    def _select_decode_scheduler(
        self,
        *,
        prompt_count: int,
        temperature: float,
        max_tokens: int,
        stop_tokens_present: bool = False,
        on_step: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        requested = getattr(self, "decode_scheduler", "auto")
        if requested not in {"auto", "serial", "async-lookahead"}:
            raise ValueError(
                "decode_scheduler must be 'auto', 'serial', or 'async-lookahead'"
            )

        telemetry: dict[str, Any] = {
            "requested": requested,
            "active": "serial",
            "reason": "",
            "reason_code": "",
            "prompt_count": prompt_count,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop_tokens_present": stop_tokens_present,
            "on_step_present": on_step is not None,
            "async_submissions": 0,
            "async_drains": 0,
            "async_sync_seconds": 0.0,
            "unused_lookahead_submissions": 0,
            "stopped_after_lookahead": False,
        }
        if requested == "serial":
            telemetry["reason"] = "serial requested"
            telemetry["reason_code"] = "serial_requested"
            return False, telemetry
        if max_tokens <= 0:
            telemetry["reason"] = "no decode tokens requested"
            telemetry["reason_code"] = "no_decode"
            return False, telemetry
        if prompt_count != 1:
            telemetry["reason"] = "async-lookahead currently supports one request"
            telemetry["reason_code"] = "batch_size"
            if requested == "async-lookahead":
                raise ValueError(telemetry["reason"])
            return False, telemetry
        if temperature != 0:
            telemetry["reason"] = "async-lookahead currently supports greedy decoding only"
            telemetry["reason_code"] = "temperature"
            if requested == "async-lookahead":
                raise ValueError(telemetry["reason"])
            return False, telemetry
        if on_step is not None and not getattr(on_step, "smarttensor_async_safe", False):
            telemetry["reason"] = "async-lookahead requires an async-safe on_step callback"
            telemetry["reason_code"] = "on_step"
            if requested == "async-lookahead":
                raise ValueError(telemetry["reason"])
            return False, telemetry
        supported, reason = self._async_lookahead_support()
        if not supported:
            telemetry["reason"] = reason
            telemetry["reason_code"] = "runner_support"
            if requested == "async-lookahead":
                raise ValueError(f"async-lookahead decode is unsafe: {reason}")
            return False, telemetry

        telemetry["active"] = "async-lookahead"
        telemetry["reason"] = reason
        telemetry["reason_code"] = "active"
        return True, telemetry

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        stop_tokens: set[int] | None = None,
    ) -> MlxGenerationResult:
        result = self.generate_batch(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=stop_tokens,
        )
        return MlxGenerationResult(
            prompt=prompt,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens[0],
            generated_text=result.generated_texts[0],
            seconds=result.seconds,
            resident_peak_bytes=result.resident_peak_bytes,
            kv_cache_bytes=result.kv_cache_bytes,
            events=result.events,
            expert_prefetch=result.expert_prefetch,
            decode_scheduler=result.decode_scheduler,
            finish_reason=result.finish_reasons[0] if result.finish_reasons else None,
        )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        stop_tokens: set[int] | None = None,
        on_step: Any | None = None,
        cache: Any | None = None,
        cached_tokens: int = 0,
        prompt_token_rows: list[list[int]] | None = None,
    ) -> MlxBatchGenerationResult:
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if not prompts:
            raise ValueError("at least one prompt is required")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        import mlx.core as mx

        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        self._reset_stream_state()

        pin_event = self._pin_for_run()
        if self._trace:
            events.append({"kind": "load", **compact_event(pin_event)})

        if prompt_token_rows is not None:
            if len(prompt_token_rows) != len(prompts):
                raise ValueError("prompt_token_rows must match prompts")
            token_rows = [list(row) for row in prompt_token_rows]
        else:
            token_rows = [self.tokenizer.encode(prompt) for prompt in prompts]
        if any(not row for row in token_rows):
            raise ValueError("every prompt must produce at least one token")
        row_lengths = {len(row) for row in token_rows}
        if len(row_lengths) != 1:
            raise ValueError(
                "batched generation requires prompts that tokenize to equal lengths; "
                f"got lengths {sorted(len(row) for row in token_rows)}"
            )
        prompt_length = row_lengths.pop()
        if cached_tokens < 0 or cached_tokens >= prompt_length:
            raise ValueError("cached_tokens must leave at least one prompt token to process")
        if cached_tokens and cache is None:
            raise ValueError("cached_tokens requires the matching cache")

        embeddings_warmed = False
        if self.pin_policy == "phase" and self.warm_embeddings:
            load_embedding = self.session.load_embedding()
            if self._trace:
                events.append({"kind": "load", "pass": "warm", **compact_event(load_embedding)})
            embeddings_warmed = True

        output_warmed = False
        if self.warm_output:
            load_output = self.session.load_output()
            if self._trace:
                events.append({"kind": "load", "pass": "warm-output", **compact_event(load_output)})
            output_warmed = True

        if cache is None:
            cache = self._make_cache()
        try:
            fresh_rows = [row[cached_tokens:] for row in token_rows]
            if len(fresh_rows[0]) > 1:
                self._stream_forward_tokens(
                    [row[:-1] for row in fresh_rows],
                    cache=cache,
                    events=events,
                    pass_kind="prefill-cache",
                    manage_embedding=not embeddings_warmed,
                )

            hidden = self._stream_forward_tokens(
                [[row[-1]] for row in fresh_rows],
                cache=cache,
                events=events,
                pass_kind="prefill",
                manage_embedding=not embeddings_warmed,
            )
            logits = self._logits_from_hidden(hidden, events)

            generated: list[list[int]] = [[] for _ in prompts]
            finished = [False] * len(prompts)
            finish_reasons = ["length"] * len(prompts)

            overlap, decode_scheduler = self._select_decode_scheduler(
                prompt_count=len(prompts),
                temperature=temperature,
                max_tokens=max_tokens,
                stop_tokens_present=bool(stop_tokens),
                on_step=on_step,
            )
            if overlap:
                cur = mx.argmax(logits[:, -1, :], axis=-1)
                mx.async_eval(cur)
                decode_scheduler["async_submissions"] += 1
                nxt = None
                for step in range(max_tokens):
                    if step < max_tokens - 1:
                        hidden = self._stream_forward_tokens(
                            [[0]],
                            cache=cache,
                            events=events,
                            pass_kind="decode",
                            token_step=step,
                            manage_embedding=False,
                            input_array=cur.reshape(1, 1),
                        )
                        nlogits = self._logits_from_hidden(hidden, events)
                        nxt = mx.argmax(nlogits[:, -1, :], axis=-1)
                        mx.async_eval(nxt)
                        decode_scheduler["async_submissions"] += 1
                    token = int(cur.item())
                    generated[0].append(token)
                    stopped = bool(stop_tokens and token in stop_tokens)
                    if stopped:
                        finished[0] = True
                        finish_reasons[0] = "stop"
                    if on_step is not None:
                        on_step([token], list(finished))
                    if stopped or step == max_tokens - 1:
                        if stopped and nxt is not None:
                            sync_started = time.perf_counter()
                            mx.eval(nxt)
                            decode_scheduler["async_drains"] += 1
                            decode_scheduler["async_sync_seconds"] += (
                                time.perf_counter() - sync_started
                            )
                            decode_scheduler["unused_lookahead_submissions"] += 1
                            decode_scheduler["stopped_after_lookahead"] = True
                        break
                    cur = nxt
            else:
                for step in range(max_tokens):
                    select_started = time.perf_counter()
                    if temperature > 0:
                        token_array = mx.random.categorical(logits[:, -1, :] * (1 / temperature))
                    else:
                        token_array = mx.argmax(logits[:, -1, :], axis=-1)
                    mx.eval(token_array)
                    if self._pass_trace is not None:
                        self._pass_trace.add("select_token_eval", time.perf_counter() - select_started)
                    if self._trace:
                        events.append(
                            {
                                "kind": "compute",
                                "action": "select-token",
                                "token_step": step,
                                "seconds": time.perf_counter() - select_started,
                            }
                        )
                    step_tokens = [int(token) for token in token_array.tolist()]
                    for request_index, token in enumerate(step_tokens):
                        if finished[request_index]:
                            continue
                        generated[request_index].append(token)
                        if stop_tokens and token in stop_tokens:
                            finished[request_index] = True
                            finish_reasons[request_index] = "stop"
                    if on_step is not None:
                        on_step(step_tokens, list(finished))
                    if all(finished):
                        break

                    if step < max_tokens - 1:
                        hidden = self._stream_forward_tokens(
                            [[token] for token in step_tokens],
                            cache=cache,
                            events=events,
                            pass_kind="decode",
                            token_step=step,
                            manage_embedding=not embeddings_warmed,
                        )
                        logits = self._logits_from_hidden(hidden, events)
        finally:
            if embeddings_warmed:
                evict_embedding = self.session.evict_embedding()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm", **compact_event(evict_embedding)})
            if output_warmed:
                evict_output = self.session.evict_output()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm-output", **compact_event(evict_output)})

        return MlxBatchGenerationResult(
            prompts=tuple(prompts),
            prompt_tokens=prompt_length,
            generated_tokens=tuple(tuple(tokens) for tokens in generated),
            generated_texts=tuple(self.tokenizer.decode(tokens) for tokens in generated),
            seconds=time.perf_counter() - started,
            resident_peak_bytes=self.session.peak_resident_bytes,
            kv_cache_bytes=kv_cache_nbytes(cache),
            events=tuple(events),
            expert_prefetch=self._expert_telemetry(),
            decode_scheduler=decode_scheduler,
            finish_reasons=tuple(finish_reasons),
        )

    def generate_speculative(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        stop_tokens: set[int] | None = None,
        drafter: Any | None = None,
        max_draft: int = 8,
        draft_margin: float = 0.5,
        on_step: Any | None = None,
        cache: Any | None = None,
        cached_tokens: int = 0,
        prompt_token_rows: list[list[int]] | None = None,
    ) -> MlxBatchGenerationResult:
        """Greedy speculative decoding: one streamed pass verifies many tokens.

        A drafter proposes up to ``max_draft`` tokens; the target verifies the
        whole block in a single streamed forward pass. Accepted tokens cost one
        pass instead of one pass each. On partial acceptance the cache is
        rolled back (recurrent states cannot be trimmed) and the accepted
        prefix is re-fed, so a partial round costs two passes for j+1 tokens.
        ``draft_margin`` rejects near-tie agreements (top-1/top-2 logit gap
        below the margin) so chunked-pass numerics cannot steer ambiguous
        positions toward the drafter's bias.
        """

        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if len(prompts) != 1:
            raise ValueError("speculative decoding currently supports a single request")
        if max_draft < 1:
            raise ValueError("max_draft must be positive")

        import mlx.core as mx

        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        stats = SpeculativeStats()
        self._reset_stream_state()

        drafter = drafter or PromptLookupDrafter()
        drafter.reset()
        stops = stop_tokens or set()

        pin_event = self._pin_for_run()
        if self._trace:
            events.append({"kind": "load", **compact_event(pin_event)})

        if prompt_token_rows is not None:
            if len(prompt_token_rows) != 1:
                raise ValueError("prompt_token_rows must match prompts")
            prompt_ids = list(prompt_token_rows[0])
        else:
            prompt_ids = list(self.tokenizer.encode(prompts[0]))
        if not prompt_ids:
            raise ValueError("the prompt must produce at least one token")
        if cached_tokens < 0 or cached_tokens >= len(prompt_ids):
            raise ValueError("cached_tokens must leave at least one prompt token to process")
        if cached_tokens and cache is None:
            raise ValueError("cached_tokens requires the matching cache")

        embeddings_warmed = False
        if self.pin_policy == "phase" and self.warm_embeddings:
            load_embedding = self.session.load_embedding()
            if self._trace:
                events.append({"kind": "load", "pass": "warm", **compact_event(load_embedding)})
            embeddings_warmed = True

        output_warmed = False
        if self.warm_output:
            load_output = self.session.load_output()
            if self._trace:
                events.append({"kind": "load", "pass": "warm-output", **compact_event(load_output)})
            output_warmed = True

        if cache is None:
            cache = self._make_cache()

        emitted: list[int] = []
        finish_reason = "length"
        try:
            fresh = prompt_ids[cached_tokens:]
            if len(fresh) > 1:
                self._stream_forward_tokens(
                    [fresh[:-1]],
                    cache=cache,
                    events=events,
                    pass_kind="prefill-cache",
                    manage_embedding=not embeddings_warmed,
                )
            hidden = self._stream_forward_tokens(
                [[fresh[-1]]],
                cache=cache,
                events=events,
                pass_kind="prefill",
                manage_embedding=not embeddings_warmed,
            )
            logits = self._logits_from_hidden(hidden, events)

            context = list(prompt_ids)

            def emit(token: int) -> bool:
                """Record one committed token; returns True when generation must stop."""
                nonlocal finish_reason
                emitted.append(token)
                context.append(token)
                stopped = token in stops
                if stopped:
                    finish_reason = "stop"
                if on_step is not None:
                    on_step([token], [stopped or len(emitted) >= max_tokens])
                return stopped or len(emitted) >= max_tokens

            while len(emitted) < max_tokens:
                stats.rounds += 1
                token_array = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(token_array)
                next_token = int(token_array.item())
                if emit(next_token):
                    break

                draft_started = time.perf_counter()
                budget = min(max_draft, max_tokens - len(emitted))
                drafts = drafter.propose(context, budget)
                stats.draft_seconds += time.perf_counter() - draft_started

                if not drafts:
                    step_started = time.perf_counter()
                    hidden = self._stream_forward_tokens(
                        [[next_token]],
                        cache=cache,
                        events=events,
                        pass_kind="decode",
                        token_step=len(emitted),
                        manage_embedding=not embeddings_warmed,
                    )
                    logits = self._logits_from_hidden(hidden, events)
                    stats.single_steps += 1
                    stats.verify_seconds += time.perf_counter() - step_started
                    continue

                stats.drafted_tokens += len(drafts)
                snapshot_started = time.perf_counter()
                transaction = CacheTransaction(cache)
                stats.snapshot_seconds += time.perf_counter() - snapshot_started

                verify_started = time.perf_counter()
                hidden = self._stream_forward_tokens(
                    [[next_token, *drafts]],
                    cache=cache,
                    events=events,
                    pass_kind="verify",
                    token_step=len(emitted),
                    manage_embedding=not embeddings_warmed,
                )
                verify_logits = self._logits_from_hidden(hidden, events)
                greedy_array = mx.argmax(verify_logits[0], axis=-1)
                top2 = mx.topk(verify_logits[0].astype(mx.float32), 2, axis=-1)
                gap_array = mx.abs(top2[..., 0] - top2[..., 1])
                mx.eval(greedy_array, gap_array)
                greedy_tokens = [int(token) for token in greedy_array.tolist()]
                gaps = [float(gap) for gap in gap_array.tolist()]
                stats.verify_passes += 1
                stats.verify_seconds += time.perf_counter() - verify_started

                accepted = count_accepted_drafts(
                    drafts,
                    greedy_tokens[: len(drafts)],
                    gaps=gaps[: len(drafts)],
                    margin=draft_margin,
                )
                stats.accepted_tokens += accepted
                if hasattr(drafter, "observe_result"):
                    drafter.observe_result(len(drafts), accepted)

                stopped = False
                for token in drafts[:accepted]:
                    if emit(token):
                        stopped = True
                        break

                if stopped:
                    transaction.commit()
                    break
                if accepted == len(drafts):
                    stats.full_accept_rounds += 1
                    transaction.commit()
                    logits = verify_logits
                    continue

                refeed_started = time.perf_counter()
                transaction.rollback()
                hidden = self._stream_forward_tokens(
                    [[next_token, *drafts[:accepted]]],
                    cache=cache,
                    events=events,
                    pass_kind="refeed",
                    token_step=len(emitted),
                    manage_embedding=not embeddings_warmed,
                )
                logits = self._logits_from_hidden(hidden, events)
                stats.refeed_passes += 1
                stats.refeed_seconds += time.perf_counter() - refeed_started
        finally:
            if embeddings_warmed:
                evict_embedding = self.session.evict_embedding()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm", **compact_event(evict_embedding)})
            if output_warmed:
                evict_output = self.session.evict_output()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm-output", **compact_event(evict_output)})

        stats.emitted_tokens = len(emitted)
        speculative = stats.to_dict()
        speculative["draft_policy"] = getattr(drafter, "name", type(drafter).__name__)
        speculative["drafter"] = getattr(drafter, "name", type(drafter).__name__)
        # Linear (non-tree) speculation has no adaptive gate and no near-tie
        # guard; report the fields as zero so API consumers see a uniform
        # telemetry schema across both speculative paths.
        speculative.setdefault("gated_rounds", 0)
        speculative.setdefault("disabled_rounds", 0)
        speculative.setdefault("zero_accept_rounds", 0)
        speculative.setdefault("near_tie_events", 0)
        speculative.setdefault("estimated_regression_avoided_passes", 0)
        if hasattr(drafter, "telemetry"):
            speculative["drafter_stats"] = drafter.telemetry()
        return MlxBatchGenerationResult(
            prompts=tuple(prompts),
            prompt_tokens=len(prompt_ids),
            generated_tokens=(tuple(emitted),),
            generated_texts=(self.tokenizer.decode(emitted),),
            seconds=time.perf_counter() - started,
            resident_peak_bytes=self.session.peak_resident_bytes,
            kv_cache_bytes=kv_cache_nbytes(cache),
            events=tuple(events),
            expert_prefetch=self._expert_telemetry(),
            finish_reasons=(finish_reason,),
            speculative=speculative,
        )

    def generate_tree_speculative(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        stop_tokens: set[int] | None = None,
        drafter: Any | None = None,
        max_draft: int = 8,
        max_branches: int = 16,
        draft_margin: float = 0.5,
        cache: Any | None = None,
        cached_tokens: int = 0,
        prompt_token_rows: list[list[int]] | None = None,
    ) -> MlxBatchGenerationResult:
        """Experimental tree speculation: verify many guessed futures at once.

        The prefix cache is cloned across ``max_branches`` rows, each row
        verifies a different continuation, and the winning branch's cache row is
        committed only when the whole branch is accepted. Partial accepts refeed
        the accepted prefix through the normal single-row cache path.
        """

        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if len(prompts) != 1:
            raise ValueError("tree speculation currently supports a single request")
        if max_draft < 1:
            raise ValueError("max_draft must be positive")
        if max_branches < 1:
            raise ValueError("max_branches must be positive")

        import mlx.core as mx

        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        stats = TreeSpeculativeStats()
        self._reset_stream_state()

        drafter = drafter or PromptLookupTreeDrafter()
        drafter.reset()
        stops = stop_tokens or set()

        pin_event = self._pin_for_run()
        if self._trace:
            events.append({"kind": "load", **compact_event(pin_event)})

        if prompt_token_rows is not None:
            if len(prompt_token_rows) != 1:
                raise ValueError("prompt_token_rows must match prompts")
            prompt_ids = list(prompt_token_rows[0])
        else:
            prompt_ids = list(self.tokenizer.encode(prompts[0]))
        if not prompt_ids:
            raise ValueError("the prompt must produce at least one token")
        if cached_tokens < 0 or cached_tokens >= len(prompt_ids):
            raise ValueError("cached_tokens must leave at least one prompt token to process")
        if cached_tokens and cache is None:
            raise ValueError("cached_tokens requires the matching cache")

        embeddings_warmed = False
        if self.pin_policy == "phase" and self.warm_embeddings:
            load_embedding = self.session.load_embedding()
            if self._trace:
                events.append({"kind": "load", "pass": "warm", **compact_event(load_embedding)})
            embeddings_warmed = True

        output_warmed = False
        if self.warm_output:
            load_output = self.session.load_output()
            if self._trace:
                events.append({"kind": "load", "pass": "warm-output", **compact_event(load_output)})
            output_warmed = True

        if cache is None:
            cache = self._make_cache()

        emitted: list[int] = []
        finish_reason = "length"
        try:
            fresh = prompt_ids[cached_tokens:]
            if len(fresh) > 1:
                self._stream_forward_tokens(
                    [fresh[:-1]],
                    cache=cache,
                    events=events,
                    pass_kind="prefill-cache",
                    manage_embedding=not embeddings_warmed,
                )
            hidden = self._stream_forward_tokens(
                [[fresh[-1]]],
                cache=cache,
                events=events,
                pass_kind="prefill",
                manage_embedding=not embeddings_warmed,
            )
            logits = self._logits_from_hidden(hidden, events)

            context = list(prompt_ids)
            reservoir: list[list[int]] = []
            gate = AdaptiveDraftGate()
            near_tie_events = 0

            def emit(token: int) -> bool:
                nonlocal finish_reason
                emitted.append(token)
                context.append(token)
                stopped = token in stops
                if stopped:
                    finish_reason = "stop"
                return stopped or len(emitted) >= max_tokens

            while len(emitted) < max_tokens:
                stats.rounds += 1
                token_array = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(token_array)
                next_token = int(token_array.item())
                if emit(next_token):
                    break

                budget = min(max_draft, max_tokens - len(emitted))
                draft_started = time.perf_counter()
                if not gate.allow_draft():
                    # Adaptive gate: consecutive zero-accept rounds proved this
                    # span undraftable; single-step through the cooldown (or for
                    # the rest of the request once disabled) so speculation is
                    # never a throughput regression.
                    branches = []
                elif hasattr(drafter, "propose_branches"):
                    branches = drafter.propose_branches(context, budget, max_branches)
                else:
                    branch = drafter.propose(context, budget)
                    branches = [branch] if branch else []
                stats.draft_seconds += time.perf_counter() - draft_started

                # Recycled verify tails augment rounds where the drafter found
                # real matches. They must never CREATE a verify round: measured,
                # stale tails accept ~0 on novel text, so turning a cheap
                # single-step into a failed verify+refeed double-pass halves
                # throughput on low-match workloads.
                if branches:
                    seen_branches = {tuple(branch) for branch in branches}
                    for seed in reservoir:
                        candidate = list(seed[:budget])
                        if candidate and len(candidate) < budget:
                            candidate = candidate + [candidate[-1]] * (budget - len(candidate))
                        key = tuple(candidate)
                        if candidate and key not in seen_branches:
                            seen_branches.add(key)
                            branches.append(candidate)
                    branches = branches[:max_branches]

                if not branches:
                    step_started = time.perf_counter()
                    hidden = self._stream_forward_tokens(
                        [[next_token]],
                        cache=cache,
                        events=events,
                        pass_kind="decode",
                        token_step=len(emitted),
                        manage_embedding=not embeddings_warmed,
                    )
                    logits = self._logits_from_hidden(hidden, events)
                    stats.single_steps += 1
                    stats.verify_seconds += time.perf_counter() - step_started
                    continue

                stats.branch_rounds += 1
                stats.branches_verified += len(branches)
                stats.drafted_tokens += sum(len(branch) for branch in branches)

                snapshot_started = time.perf_counter()
                prefix_snapshot = snapshot_cache_states(cache)
                verify_cache = self._make_cache()
                restore_repeated_cache_states(verify_cache, prefix_snapshot, len(branches))
                stats.snapshot_seconds += time.perf_counter() - snapshot_started

                verify_started = time.perf_counter()
                # Branches may have unequal length (a historical span can run
                # into the end of the corpus). A batched forward needs
                # rectangular rows, so right-pad each branch to the round's
                # widest length by repeating its own last token. Causal masking
                # means a padding column at position p only ever feeds positions
                # > p, so the logits at every real position 0..len(branch) are
                # byte-identical to an unpadded run — padding never changes
                # acceptance. Acceptance and tail extraction below index by the
                # branch's TRUE length, so padded columns are simply ignored.
                verify_width = max(len(branch) for branch in branches)
                verify_rows = [
                    [next_token, *branch, *([branch[-1]] * (verify_width - len(branch)) if branch else [])]
                    for branch in branches
                ]
                hidden = self._stream_forward_tokens(
                    verify_rows,
                    cache=verify_cache,
                    events=events,
                    pass_kind="tree-verify",
                    token_step=len(emitted),
                    manage_embedding=not embeddings_warmed,
                )
                verify_logits = self._logits_from_hidden(hidden, events)
                greedy_array = mx.argmax(verify_logits, axis=-1)
                top2 = mx.topk(verify_logits.astype(mx.float32), 2, axis=-1)
                gap_array = mx.abs(top2[..., 0] - top2[..., 1])
                mx.eval(greedy_array, gap_array)
                greedy_rows = greedy_array.tolist()
                gap_rows = gap_array.tolist()
                stats.verify_passes += 1
                stats.verify_seconds += time.perf_counter() - verify_started

                best_index = 0
                best_accepted = -1
                for index, branch in enumerate(branches):
                    accepted = count_accepted_drafts(
                        branch,
                        [int(token) for token in greedy_rows[index][: len(branch)]],
                        gaps=[float(gap) for gap in gap_rows[index][: len(branch)]],
                        margin=draft_margin,
                    )
                    if accepted > best_accepted:
                        best_index = index
                        best_accepted = accepted

                best_branch = branches[best_index]
                tail_from = max(best_accepted, 0) + 1
                winner_tail = [
                    int(token)
                    for token in greedy_rows[best_index][tail_from : tail_from + max_draft]
                ]
                # Only recycle tails from rounds with real acceptance; a dead
                # round's tail is conditioned on rejected context.
                reservoir = [winner_tail] if winner_tail and best_accepted >= 1 else []
                gate.observe(max(best_accepted, 0))
                stats.accepted_tokens += max(best_accepted, 0)
                stopped = False
                for token in best_branch[:best_accepted]:
                    if emit(token):
                        stopped = True
                        break
                if hasattr(drafter, "observe_result"):
                    drafter.observe_result(len(best_branch), max(best_accepted, 0))

                if stopped:
                    break
                decision_position = min(max(best_accepted, 0), len(gap_rows[best_index]) - 1)
                decision_guarded = float(gap_rows[best_index][decision_position]) < draft_margin
                if decision_guarded:
                    # The next-token decision fell inside the margin under
                    # chunked numerics — a near-tie that could flip vs the
                    # single-token baseline. Counted whether or not the guard
                    # path runs this round, so the figure tracks how often
                    # chunked numerics put a decision on the knife's edge.
                    near_tie_events += 1

                if (
                    best_accepted == len(best_branch)
                    and len(best_branch) == verify_width
                    and not decision_guarded
                ):
                    # Fast path only when the winner spans the full verify
                    # width: then verify_logits[best_index][-1] is its true
                    # last position and the committed cache row carries no
                    # padding KV. A shorter winner falls through to refeed,
                    # which re-runs exactly [next_token, *committed] on the
                    # single-row cache and recomputes correct logits.
                    stats.full_accept_rounds += 1
                    restore_cache_batch_row(cache, verify_cache, best_index)
                    logits = verify_logits[best_index : best_index + 1]
                    continue

                refeed_started = time.perf_counter()
                committed = best_branch[: max(best_accepted, 0)]
                if decision_guarded and committed:
                    # The next-token decision sits inside the margin under
                    # chunked numerics. End the round with a single-token pass
                    # so that decision uses single-step numerics — this is the
                    # exactness guard for the deterministic near-tie flips.
                    self._stream_forward_tokens(
                        [[next_token, *committed[:-1]]],
                        cache=cache,
                        events=events,
                        pass_kind="tree-refeed",
                        token_step=len(emitted),
                        manage_embedding=not embeddings_warmed,
                    )
                    hidden = self._stream_forward_tokens(
                        [[committed[-1]]],
                        cache=cache,
                        events=events,
                        pass_kind="tree-refeed-guard",
                        token_step=len(emitted),
                        manage_embedding=not embeddings_warmed,
                    )
                    stats.refeed_passes += 2
                else:
                    hidden = self._stream_forward_tokens(
                        [[next_token, *committed]],
                        cache=cache,
                        events=events,
                        pass_kind="tree-refeed",
                        token_step=len(emitted),
                        manage_embedding=not embeddings_warmed,
                    )
                    stats.refeed_passes += 1
                logits = self._logits_from_hidden(hidden, events)
                stats.refeed_seconds += time.perf_counter() - refeed_started
        finally:
            if embeddings_warmed:
                evict_embedding = self.session.evict_embedding()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm", **compact_event(evict_embedding)})
            if output_warmed:
                evict_output = self.session.evict_output()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm-output", **compact_event(evict_output)})

        stats.emitted_tokens = len(emitted)
        speculative = stats.to_dict()
        speculative.update(gate.telemetry())
        speculative["near_tie_events"] = near_tie_events
        speculative["draft_policy"] = getattr(drafter, "name", type(drafter).__name__)
        speculative["drafter"] = getattr(drafter, "name", type(drafter).__name__)
        if hasattr(drafter, "telemetry"):
            speculative["drafter_stats"] = drafter.telemetry()
        return MlxBatchGenerationResult(
            prompts=tuple(prompts),
            prompt_tokens=len(prompt_ids),
            generated_tokens=(tuple(emitted),),
            generated_texts=(self.tokenizer.decode(emitted),),
            seconds=time.perf_counter() - started,
            resident_peak_bytes=self.session.peak_resident_bytes,
            kv_cache_bytes=kv_cache_nbytes(cache),
            events=tuple(events),
            expert_prefetch=self._expert_telemetry(),
            finish_reasons=(finish_reason,),
            speculative=speculative,
        )
