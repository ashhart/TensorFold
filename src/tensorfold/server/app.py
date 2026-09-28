"""The chat app renders requests and queues them onto one lane engine thread."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable
import uuid

from tensorfold.engine.lane_engine import LaneEngine, SuffixLookupProposer
from tensorfold.server.checkpoints import CheckpointStore, longest_common_prefix, save_conversations
from tensorfold.server.cancellation import Cancellation
from tensorfold.server.errors import RequestError
from tensorfold.server.request_options import RequestOptions
from tensorfold.server.http import served_model_ids
from tensorfold.server.scheduler import ChatJob, Scheduler
from tensorfold.server.stopping import StopPolicy
from tensorfold.server.text import (
    IncrementalText,
    _LockedTokenizer,
    eos_ids_of,
    hide_tool_calls,
    is_title_request,
    parse_harmony_output,
    render_prompt_ids,
    split_thinking, think_markers,
    streaming_visible_text,
    template_late_system,
    strip_trailing_stops,
)


_REQUEST = threading.local()


def _mlx_version() -> str:
    import mlx.core as mx

    return str(mx.__version__)


def _token_sha(tokens: list[int]) -> str:
    """A short fingerprint of a reply's tokens: two runs match byte for byte iff these match."""

    import hashlib

    return hashlib.sha256(",".join(str(int(t)) for t in tokens).encode()).hexdigest()[:12]


class ChatApp(RequestOptions):
    """One model behind the OpenAI endpoint (``server.http.make_handler``)."""

    accepts_sampling = True
    accepts_cancellation = True
    accepts_raw_prompt = True
    # the request's tool list never stops prose from streaming
    streams_prose_with_tools = True

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        served_name: str,
        engine_factory: Callable[..., Any] | None = None,
        lanes: int = 1,
        max_rows: int = 16,
        max_draft: int = 32,
        min_match: int = 4,
        default_max_tokens: int = 4096,
        context_window: int = 0,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
        thinking_budget: int = 0,
        default_sampling: dict[str, Any] | None = None,
        max_snapshots: int = 3,
        checkpoint_slots: int | None = None,
        checkpoint_budget_bytes: int | None = 16 * 1024**3,
        memory_budget_bytes: int | None = None,
        memory_runtime: Any = None,
        model_aliases: list[str] | None = None,
        use_proposer: bool = True,
        snapshot_dir: Path | None = None,
        model_id: str = "",
        memory_fraction: float | None = None,
        memory_overhead_bytes: int | None = None,
        fit_context: bool = False,
    ) -> None:
        # three candidate entries per conversation (history boundary, stable prefix, reply end)
        if checkpoint_slots is None:
            checkpoint_slots = max(3 * int(lanes), 8)
        self._model = model
        self.served_name = served_name
        self.model_ids = served_model_ids(served_name, model_aliases)
        self.max_batch_size = int(lanes)
        self.tokenizer = tokenizer
        self.tokenizer_lock = threading.Lock()
        self.default_max_tokens = int(default_max_tokens)
        # prompt + reply tokens a request may use (0: no limit)
        self.context_window = int(context_window)
        self.enable_thinking = bool(enable_thinking)
        self.reasoning_effort = reasoning_effort
        # the default thinking budget (0: none; a request's "thinking_budget" overrides it)
        self.thinking_budget = int(thinking_budget)
        self._think_tokens: tuple[tuple[int, ...], int] | None = None
        self.think_markers = think_markers(tokenizer)
        # ``exact_sampling.Sampling`` fields used when a request names none (None: greedy)
        self.default_sampling = dict(default_sampling) if default_sampling else None
        self.max_snapshots = int(max_snapshots)
        factory = engine_factory or LaneEngine
        self.exact_mode = {
            "mode": "exact",
            "engine": "lanes",
            "note": "every token is the model's own sample at its position; drafts only change speed",
        }
        self.stop_ids = eos_ids_of(tokenizer)
        self.late_system = template_late_system(tokenizer)
        self.engine = factory(model, max_rows=int(max_rows), max_draft=int(max_draft),
                              retain_finished_caches=int(checkpoint_slots) > 0)
        from tensorfold.server.memory_budget import cache_nbytes

        self.checkpoints = (
            CheckpointStore(int(checkpoint_slots), copier=self.engine.copy_single_cache,
                            budget_bytes=checkpoint_budget_bytes,
                            sizer=cache_nbytes if memory_budget_bytes is not None else self.engine.cache_nbytes,
                            pinned_slots=max(3, self.max_snapshots))
            if int(checkpoint_slots) > 0 else None
        )
        self.requests_completed = 0
        # Delay background requests while foreground requests arrive and prepare so the session turn is admitted first.
        self.background_grace_s = 0.15
        self._preparing = 0
        self._preparing_lock = threading.Lock()
        self.min_match = int(min_match)
        self.use_proposer = bool(use_proposer)
        self.prompt_memory: Any = None
        self.context_fitted = False       # the window is what the memory budget fits, below the configured one
        if memory_budget_bytes is not None:
            from tensorfold.server.prompt_memory import PromptMemory, probe_tokens

            self.prompt_memory = PromptMemory(memory_budget_bytes, model, runtime=memory_runtime,
                                              store=self.checkpoints, window_tokens=self.context_window,
                                              chunk_rows=getattr(self.engine.prefill_plan, "step",
                                                                 self.engine.prefill_step),
                                              **({} if memory_overhead_bytes is None
                                                 else {"overhead_bytes": memory_overhead_bytes}))
            if self.checkpoints is not None:
                # admission evicts on demand, so a long conversation keeps its newest prefix past the budget
                self.checkpoints.admit_oversize = True
        measure = lambda: self._admission(float(memory_fraction), int(lanes)) if memory_fraction and lanes > 1 else None
        admission = measure() if self.prompt_memory is None else self.prompt_memory.sized(
            self.engine, measure, probe_tokens(tokenizer))
        if self.prompt_memory is not None:
            self.context_window, self.context_fitted = self.prompt_memory.fit_window(self.context_window, fit_context)
        self.scheduler = Scheduler(
            self.engine,
            lanes=int(lanes),
            admission=admission,
            eos_ids=self.stop_ids,
            checkpoints=self.checkpoints,
            proposer_factory=(lambda: SuffixLookupProposer(min_match=self.min_match)) if use_proposer else None,
            snapshot_dir=snapshot_dir,
            session_dir=None if snapshot_dir is None else Path(snapshot_dir).parent / "session-snapshots",
            model_id=model_id,
            prompt_memory=self.prompt_memory,
        )
        loaded_count = 0
        if snapshot_dir is not None and self.checkpoints is not None:
            from tensorfold.engine.prefix_snapshots import load_snapshots

            loaded_at = time.perf_counter()
            allow = None if self.prompt_memory is None else lambda path: self.prompt_memory.allow_load(path.stat().st_size)
            loaded = list(load_snapshots(snapshot_dir, model_id, limit=self.max_snapshots, allow=allow))
            for tokens, cache in reversed(loaded):        # the newest ends up most recently used
                self.checkpoints.insert(tokens, cache, last_prompt=tokens, pinned=True)
                print(f"[tensorfold] loaded system-block snapshot tokens={len(tokens)} "
                      f"in {time.perf_counter() - loaded_at:.1f}s", flush=True)
            loaded_count = len(loaded)
            del loaded
        self.warming = False
        self.scheduler.start()
        if snapshot_dir is not None and self.checkpoints is not None and not loaded_count:
            # only when these kernels have no block yet: a warmed block is pinned after the loaded ones
            self._warm_known_blocks(snapshot_dir, model_id)

    def _admission(self, fraction: float, lanes: int) -> Any:
        """Admit concurrent streams only within ``fraction`` of RAM minus memory held elsewhere on the machine."""

        from tensorfold.engine import memory

        stream = memory.measure(self.engine)
        used = memory._mlx_used()
        ram = memory.ram_bytes()
        elsewhere = memory.used_elsewhere(used)
        share = self.prompt_memory.budget if self.prompt_memory is not None else int(fraction * ram)
        admission = memory.Admission(min(int(fraction * ram) - elsewhere, share), stream,
                                     used=None if self.prompt_memory is None else self.prompt_memory.held)
        tokens = self.default_max_tokens + 4096
        gib, mib = 1024**3, 1024**2
        print(f"[tensorfold] concurrency: up to {lanes} requests share each round; memory budget "
              f"{admission.budget / gib:.1f} GB (MLX's share {share / gib:.1f} GB, or {fraction:.0%} of "
              f"{ram / gib:.0f} GB less {elsewhere / gib:.1f} GB in use elsewhere); a stream "
              f"{stream.short / mib:.0f} MB at {stream.short_tokens} tokens, "
              f"{stream.long / mib:.0f} MB at {stream.long_tokens:,}, then {stream.per_token / 1024:.1f} KB a token; "
              f"a shared round up to {stream.round_bytes / gib:.2f} GB; {admission.fitting(tokens)} streams of "
              f"{tokens:,} tokens fit now (more wait their turn)", flush=True)
        return admission

    def render(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        thinking: bool | None = None,
    ) -> tuple[list[int], int]:
        """Prompt ids plus the length of the rendered history that prefixes them."""

        thinking = self.enable_thinking if thinking is None else bool(thinking)
        effort = (getattr(_REQUEST, "sampling", None) or {}).get("reasoning_effort", self.reasoning_effort)
        with self.tokenizer_lock:
            prompt = render_prompt_ids(self.tokenizer, messages, tools=tools, enable_thinking=thinking,
                                       reasoning_effort=effort, late_system=self.late_system)
            history = render_prompt_ids(self.tokenizer, messages, tools=tools, enable_thinking=thinking,
                                        reasoning_effort=effort, add_generation_prompt=False,
                                        late_system=self.late_system)
        history_len = len(history) if 0 < len(history) < len(prompt) and prompt[: len(history)] == history else 0
        return prompt, history_len

    def system_prefix_len(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
        prompt_ids: list[int], thinking: bool | None = None,
    ) -> int:
        """Find a reusable system prefix by substituting a probe for the first user message; return zero for short matches."""

        effort = (getattr(_REQUEST, "sampling", None) or {}).get("reasoning_effort", self.reasoning_effort)
        first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
        if first_user is None:
            return 0
        probe = [*messages[:first_user], {"role": "user", "content": "⁣probe"}]
        try:
            with self.tokenizer_lock:
                other = render_prompt_ids(
                    self.tokenizer, probe, tools=tools,
                    enable_thinking=self.enable_thinking if thinking is None else bool(thinking),
                    reasoning_effort=effort, late_system=self.late_system)
        except Exception:  # noqa: BLE001 - a template quirk must not fail the request
            return 0
        shared = longest_common_prefix(prompt_ids, other)
        return shared if shared >= 512 else 0

    def _warm_known_blocks(self, snapshot_dir: Path, model_id: str) -> None:
        """Compute the newest system block saved by other kernels in the background, a prompt chunk a job."""

        from tensorfold.engine.prefill_plan import block_jobs
        from tensorfold.engine.prefix_snapshots import blocks_to_warm

        blocks = blocks_to_warm(snapshot_dir, model_id)[:1]
        if not blocks:
            return
        pad = int(self.tokenizer.encode("\n", add_special_tokens=False)[-1])

        def warm() -> None:
            try:
                warm_blocks()
            finally:
                self.warming = False

        def warm_blocks() -> None:
            for tokens in blocks:
                started = time.perf_counter()
                jobs = block_jobs(self.engine.prefill_plan, tokens, pad)
                for i, (prompt, at) in enumerate(jobs):
                    final = i == len(jobs) - 1
                    while True:
                        job = ChatJob(
                            job_id=f"warm-{uuid.uuid4().hex[:8]}", prompt_ids=prompt, max_tokens=1,
                            temperature=0.0, history_len=at, shared_prefix_lens=(at,) if final else (),
                            drafts=False, background=True)
                        self.scheduler.submit(job)
                        while job.chunks.get() is not None:
                            pass
                        if not job.preempted:
                            break
                print(f"[tensorfold] warmed system block tokens={jobs[-1][1] if jobs else 0} of {len(tokens)} in "
                      f"{time.perf_counter() - started:.1f}s", flush=True)

        print(f"[tensorfold] warming {len(blocks)} saved system block(s) for these kernels in the background: "
              "until it ends, a request first waits for one prompt chunk (GET /health reports warming)", flush=True)
        self.warming = True
        threading.Thread(target=warm, name="warm-blocks", daemon=True).start()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        on_delta: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        sampling: dict[str, Any] | None = None,
        cancellation: Cancellation | None = None,
        prompt: str | list[int] | None = None,
    ) -> dict[str, Any]:
        """A reply to ``messages``, or with ``prompt`` (text or token ids) a raw completion: no template, no thinking."""

        _REQUEST.sampling = sampling   # per HTTP thread
        received_at = time.perf_counter()
        fields = sampling or {}
        limit = max(1, int(max_tokens or self.default_max_tokens))
        background = is_title_request(messages, tools) or fields.get("priority") == "background"
        preparing = None if background else self._Preparing(self)
        cancellation = cancellation or Cancellation()
        try:
            return self._chat_prepared(messages, max_tokens=limit, temperature=temperature, on_delta=on_delta,
                                       tools=tools, received_at=received_at, background=background,
                                       preparing=preparing, reply_limit_explicit=max_tokens is not None,
                                       cancellation=cancellation, prompt=prompt)
        except BaseException:
            self.scheduler.cancel(cancellation)
            raise
        finally:
            if preparing is not None:
                preparing.release()

    class _Preparing:
        """A."""

        def __init__(self, app: "ChatApp") -> None:
            self.app = app
            with app._preparing_lock:
                app._preparing += 1
            self.released = False

        def release(self) -> None:
            with self.app._preparing_lock:
                if not self.released:
                    self.released = True
                    self.app._preparing -= 1

    def _chat_prepared(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        on_delta: Any | None,
        tools: list[dict[str, Any]] | None,
        received_at: float,
        background: bool,
        preparing: "ChatApp._Preparing | None",
        reply_limit_explicit: bool = True,
        cancellation: Cancellation | None = None,
        prompt: str | list[int] | None = None,
    ) -> dict[str, Any]:
        # a request's chat_template_kwargs.enable_thinking (via the sampling fields) overrides the server's
        cancellation = cancellation or Cancellation()
        cancellation.check()
        fields = getattr(_REQUEST, "sampling", None) or {}
        stops = StopPolicy(fields, self.tokenizer, self.tokenizer_lock, self.stop_ids)
        requested = fields.get("enable_thinking")
        thinking = self.enable_thinking if requested is None else bool(requested)
        if prompt is not None:
            thinking, history_len = False, 0
            if isinstance(prompt, str):
                with self.tokenizer_lock:
                    prompt_ids = [int(t) for t in self.tokenizer.encode(prompt)]
            else:
                prompt_ids = [int(t) for t in prompt]
        else:
            prompt_ids, history_len = self.render(messages, tools, thinking=thinking)
        cancellation.check()
        if not prompt_ids:
            raise RequestError("rendered prompt is empty")
        limit = max(1, int(max_tokens))
        if self.context_window:
            room = self.context_window - len(prompt_ids)
            if room < 1:
                why = ", the most this server's memory budget fits" if self.context_fitted else ""
                raise RequestError(f"This server's maximum context length is {self.context_window:,} tokens{why}, "
                                   f"but the rendered prompt has {len(prompt_ids):,} tokens and leaves no room for a "
                                   "reply. Compact or shorten the conversation.")
            if reply_limit_explicit and limit > room:
                raise RequestError(
                    f"the rendered prompt has {len(prompt_ids)} tokens and requests {limit} reply tokens; "
                    f"this server's context window is {self.context_window}. Reduce the prompt to at most "
                    f"{max(0, self.context_window - limit)} prompt tokens or request at most {room} reply tokens, "
                    "including chat template and thinking tokens."
                )
            limit = min(limit, room)
        system_len = 0 if prompt is not None else self.system_prefix_len(messages, tools, prompt_ids, thinking=thinking)
        spec = self._resolve_sampling(fields, temperature, prompt_ids)
        drafts = self.use_proposer and fields.get("draft", True) is not False

        def make_job() -> ChatJob:
            job = ChatJob(
                job_id=f"req-{uuid.uuid4().hex[:12]}",
                prompt_ids=prompt_ids,
                max_tokens=limit,
                temperature=float(temperature),
                history_len=history_len,
                # Snapshot before the system block ends to retain reusable prefixes when session-specific tails differ.
                shared_prefix_lens=tuple(n for n in (system_len - 2048, system_len - 512, system_len)
                                         if n >= 512) if system_len else (),
                sampling=spec,
                background=background,
                drafts=drafts,
                ignore_eos=stops.ignore_eos, stop_check=stops if stops.strings else None,
                cancellation=cancellation,
            )
            budget = int(fields.get("thinking_budget") or self.thinking_budget) if thinking else 0
            if budget > 0:
                job.think_close, job.think_end = self._think_close()
                job.think_budget = budget if job.think_end >= 0 else 0
            if drafts:
                base: Any = SuffixLookupProposer(min_match=self.min_match)
                if tools:
                    from tensorfold.engine.tool_draft import ToolCallProposer

                    base = ToolCallProposer(_LockedTokenizer(self.tokenizer, self.tokenizer_lock), tools,
                                            len(prompt_ids), fallback=base)
                job.proposer = base
            return job

        job = make_job()
        if background:
            # Let the session's foreground turn arrive and render before submitting background work.
            deadline = received_at + self.background_grace_s
            while time.perf_counter() < deadline or (self._preparing and time.perf_counter() < received_at + 2.0):
                cancellation.check()
                time.sleep(0.005)
        cancellation.check()
        self.scheduler.submit(job)
        if preparing is not None:
            preparing.release()           # submitted: a waiting background request may go now

        collected: list[int] = []
        streamed = ""
        streamed_reasoning = ""
        streaming_done = False
        first_token_at = 0.0
        # tool calls stream as OpenAI tool_call deltas while they are written
        from tensorfold.engine.tool_draft import ToolCallStreamer

        calls_stream = ToolCallStreamer(tools) if (tools and on_delta is not None) else None
        visible_text = IncrementalText(self.tokenizer, self.tokenizer_lock)
        replay: list[int] = []        # tokens a preempted job already delivered, owed again by its rerun
        preemptions = 0
        while True:
            cancellation.check()
            try:
                chunk = job.chunks.get(timeout=0.05)
            except queue.Empty:
                continue
            cancellation.check()
            if chunk is None:
                if job.preempted and job.error is None:
                    # Replay preempted work with the same prompt and sampling, dropping tokens already delivered.
                    preemptions += 1
                    replay = list(collected)
                    job = make_job()
                    self.scheduler.submit(job)
                    continue
                break
            if replay:
                owed = replay[:len(chunk)]
                if list(chunk[:len(owed)]) != owed:
                    print(f"[tensorfold] rerun of a preempted request diverged: {chunk[:len(owed)]} != {owed}",
                          flush=True)
                replay = replay[len(owed):]
                chunk = chunk[len(owed):]
                if not chunk:
                    continue
            if not first_token_at:
                first_token_at = time.perf_counter()
            collected.extend(chunk)
            if on_delta is None or streaming_done:
                continue
            fresh = []
            for token in chunk:
                if token in stops.eos_ids:
                    streaming_done = True
                    break
                fresh.append(token)
            text = stops.visible(visible_text.extend(fresh), partial=True)
            if len(visible_text.tokens) and visible_text._read < len(visible_text.tokens):
                continue                    # a character still split across tokens: wait for the rest
            answer = text
            if thinking:
                # the prompt opened a think block: reasoning streams as reasoning_content until </think>
                reasoning_so_far, answer = split_thinking(text, finished=False, markers=self.think_markers)
                piece = reasoning_so_far[len(streamed_reasoning):]
                if piece:
                    streamed_reasoning = reasoning_so_far
                    on_delta({"reasoning_content": piece})
            shown = hide_tool_calls(answer, finished=False) if calls_stream is not None else answer
            visible = streaming_visible_text(shown)
            delta = visible[len(streamed):]
            if delta:
                streamed = visible
                on_delta(delta)
            if calls_stream is not None:
                for call_delta in calls_stream.feed(answer):   # never the reasoning: a call it mentions is not made
                    on_delta(call_delta)
        if job.error is not None:
            raise job.error

        content_tokens = strip_trailing_stops(collected, set(stops.eos_ids))
        with self.tokenizer_lock:
            text = stops.visible(self.tokenizer.decode(content_tokens))
        if thinking:
            reasoning_text, content = split_thinking(text, finished=True, markers=self.think_markers)
            reasoning = reasoning_text.strip() or None
        else:
            content, reasoning = parse_harmony_output(text)
        stops.flush(on_delta, content, reasoning, streamed, streamed_reasoning, calls_stream is not None)
        stream = job.stream
        seconds = max(0.0, job.finished_at - job.submitted_at)
        decode_seconds = max(0.0, job.finished_at - job.prefilled_at) if job.prefilled_at else 0.0
        decode_tokens = max(0, len(collected) - 1)
        reply: dict[str, Any] = {
            "content": content,
            "reasoning": reasoning,
            "tool_calls_streamed": bool(calls_stream is not None and calls_stream.streamed),
            "finish_reason": stream.finish_reason if stream is not None else "length",
            "prompt_tokens": len(prompt_ids),
            "cached_tokens": int(job.cached_tokens),
            "completion_tokens": len(collected),
            "seconds": seconds,
            "runtime": {
                "enable_thinking": thinking,
                "reasoning_effort": fields.get("reasoning_effort", self.reasoning_effort) if thinking else "none",
                "engine": self.exact_mode["engine"],
                "tokens_per_second": (decode_tokens / decode_seconds) if decode_seconds > 0 else 0.0,
                "seconds": seconds,
                "prefill_seconds": max(0.0, job.prefilled_at - job.submitted_at) if job.prefilled_at else None,
                "time_to_first_token": (first_token_at - received_at) if first_token_at else None,
                "sampling": "exact" if spec is not None else "greedy",
                "drafts": bool(job.drafts),
                # the reply's token ids, hashed: drafted and ``"draft": false`` replies must match
                "token_sha": _token_sha(collected),
                # the narrowest verify window of the reply's rounds (drafted replies: 2 or more)
                "min_rows": int(getattr(job.stream, "min_rows", 0) or 0),
            },
        }
        if stream is not None:
            speculative: dict[str, Any] = {
                "rounds": stream.rounds,
                "drafted": stream.drafted,
                "accepted": stream.accepted,
                "acceptance_rate": (stream.accepted / stream.drafted) if stream.drafted else 0.0,
                "tokens_per_round": (len(collected) / stream.rounds) if stream.rounds else 0.0,
            }
            if stream.proposer is not None and hasattr(stream.proposer, "telemetry"):
                speculative["proposer"] = stream.proposer.telemetry()
            reply["speculative"] = speculative
        self.requests_completed += 1
        store = self.checkpoints
        print(
            f"[tensorfold] done {job.job_id} prompt={len(prompt_ids)} cached={job.cached_tokens} "
            f"thinking={thinking} effort={reply['runtime']['reasoning_effort']} "
            f"tokens={len(collected)} sha={_token_sha(collected)} finish={reply['finish_reason']} "
            f"tok/s={reply['runtime']['tokens_per_second']:.1f} "
            f"ttft={(first_token_at - received_at) if first_token_at else -1:.2f}s "
            f"prefill={max(0.0, job.prefilled_at - job.started_at) if job.started_at and job.prefilled_at else -1:.2f}s "
            + (f"background preemptions={preemptions} " if background else "")
            + f"rounds={stream.rounds if stream else 0} "
            f"accepted={stream.accepted if stream else 0}/{stream.drafted if stream else 0} "
            + self._round_profile(stream)
            + (
                f"checkpoints={len(store)} ({store.nbytes / 1024**3:.2f} GiB, "
                f"hits={store.hits} misses={store.misses} evictions={store.evictions})"
                if store is not None
                else "checkpoints=off"
            ),
            flush=True,
        )
        return reply

    def _round_profile(self, stream: Any) -> str:
        """Mean ms per round of this stream's last rounds."""

        rounds = int(getattr(stream, "rounds", 0) or 0) if stream is not None else 0
        stats = list(self.scheduler.engine.round_stats[-rounds:]) if rounds else []
        if not stats or self.scheduler.active:
            return ""
        n = len(stats)
        return (f"ms/round={sum(r.total_ms for r in stats) / n:.1f} "
                f"forward={sum(r.forward_ms for r in stats) / n:.1f} "
                f"draft={sum(r.draft_ms for r in stats) / n:.1f} post={sum(r.post_ms for r in stats) / n:.1f} "
                f"rows={sum(r.width for r in stats) / n:.1f} ")

    def close(self) -> None:
        self.scheduler.on_stop = self.save_sessions    # saved by the scheduler thread, which owns the arrays
        self.scheduler.stop(timeout=120.0)

    def save_sessions(self) -> int:
        """At shutdown, the most recent conversations' checkpoints to disk (read back on demand)."""

        if self.scheduler.session_dir is None or self.checkpoints is None:
            return 0
        return save_conversations(self.checkpoints, Path(self.scheduler.session_dir), self.scheduler.model_id)
