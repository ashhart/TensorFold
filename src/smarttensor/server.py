"""OpenAI-compatible chat server backed by SmartTensor streaming runners.

This exposes the minimal surface that OpenAI-style clients (opencode, curl,
SDKs) need: ``GET /v1/models`` and ``POST /v1/chat/completions`` with optional
server-sent-event streaming. Requests are served one at a time behind a lock;
the most recent conversation's KV cache is kept live so a follow-up request
that extends the same conversation only prefills the new suffix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from smarttensor.adapters.mlx import (
    DeepSeekV3StreamingForwardRunner,
    GptOssStreamingForwardRunner,
    NemotronHStreamingForwardRunner,
    Qwen35MoeStreamingForwardRunner,
)

# Model-type -> runner dispatch table. Hoisted to module scope so it is
# importable and testable (tests/test_nemotron_serve_dispatch.py); run_server
# resolves the runner via RUNNERS.get(model_type).
RUNNERS = {
    "deepseek_v3": DeepSeekV3StreamingForwardRunner,
    "glm_moe_dsa": DeepSeekV3StreamingForwardRunner,
    "qwen3_5_moe": Qwen35MoeStreamingForwardRunner,
    "gpt_oss": GptOssStreamingForwardRunner,
    "nemotron_h": NemotronHStreamingForwardRunner,
}


def longest_reusable_prefix(cached: list[int], requested: list[int]) -> int:
    """Return how many cached tokens can seed the new request.

    The live cache is reusable only when every cached token is a strict prefix
    of the request, leaving at least one request token to produce logits.
    """

    if not cached or len(cached) >= len(requested):
        return 0
    if requested[: len(cached)] != cached:
        return 0
    return len(cached)


def strip_trailing_stops(tokens: list[int], stop_ids: set[int]) -> list[int]:
    end = len(tokens)
    while end > 0 and tokens[end - 1] in stop_ids:
        end -= 1
    return tokens[:end]


def pass_economics(speculative: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize a speculative-stats dict into the API pass-economics block.

    The unit of currency is the streamed pass: effective speed scales with
    accepted tokens per pass over cost per pass. This surfaces the few numbers
    that tell the caller whether speculation paid off on their workload — and,
    honestly, when it did not (zero-accept rounds, branches that went unused).
    Tolerates both the single-branch (SpeculativeStats) and the tree
    (TreeSpeculativeStats) shapes; missing keys default rather than raise.
    """

    if not speculative:
        return None
    return {
        "drafter": speculative.get("drafter"),
        "accepted_tokens_per_pass": speculative.get("accepted_tokens_per_pass", 0.0),
        "acceptance_rate": speculative.get("acceptance_rate", 0.0),
        "streamed_passes": speculative.get("streamed_passes", 0),
        "emitted_tokens": speculative.get("emitted_tokens", 0),
        "accepted_tokens": speculative.get("accepted_tokens", 0),
        "branch_rounds": speculative.get("branch_rounds", 0),
        "branches_verified": speculative.get("branches_verified", 0),
        "average_branches": speculative.get("average_branches", 0.0),
        "verify_passes": speculative.get("verify_passes", 0),
        "refeed_passes": speculative.get("refeed_passes", 0),
        "single_steps": speculative.get("single_steps", 0),
        "zero_accept_rounds": speculative.get("zero_accept_rounds", 0),
        "gate_disabled": speculative.get("gate_disabled", False),
    }


HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"
HARMONY_ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
HARMONY_TERMINATORS = ("<|return|>", "<|end|>", "<|call|>", "<|start|>")


def parse_harmony_output(text: str) -> tuple[str, str | None]:
    """Split harmony-format output (GPT-OSS) into (content, reasoning).

    Non-harmony text passes through unchanged with no reasoning. When the
    final channel never arrived (token budget spent inside analysis), content
    is empty and the partial analysis is surfaced as reasoning.
    """

    if "<|channel|>" not in text:
        return text, None

    reasoning: str | None = None
    if HARMONY_ANALYSIS_MARKER in text:
        reasoning = text.split(HARMONY_ANALYSIS_MARKER, 1)[1]
        for terminator in (HARMONY_FINAL_MARKER, *HARMONY_TERMINATORS):
            reasoning = reasoning.split(terminator, 1)[0]

    if HARMONY_FINAL_MARKER not in text:
        return "", reasoning
    content = text.split(HARMONY_FINAL_MARKER, 1)[1]
    for terminator in HARMONY_TERMINATORS:
        content = content.split(terminator, 1)[0]
    return content, reasoning


def streaming_visible_text(text: str) -> str:
    """The part of partially-decoded output that should stream to the client."""

    if "<|channel|>" not in text:
        return text
    content, _ = parse_harmony_output(text)
    return content


@dataclass
class QueuedChatRequest:
    prompt_ids: list[int]
    max_tokens: int
    temperature: float
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    error: BaseException | None = None

    @property
    def batch_key(self) -> tuple[int, int, float]:
        return (len(self.prompt_ids), self.max_tokens, self.temperature)


class SmartTensorChat:
    """Serialize chat requests onto one streaming runner with cache reuse."""

    def __init__(
        self,
        runner_factory: Any,
        *,
        served_name: str,
        default_max_tokens: int = 512,
        enable_thinking: bool = False,
        draft: str = "prompt-lookup",
        draft_model: Any | None = None,
        draft_command: str | None = None,
        max_draft: int = 8,
        speculative_scheduler: str = "linear",
        max_branches: int = 16,
        draft_margin: float = 0.5,
        reasoning_effort: str = "low",
        max_batch_size: int = 1,
        batch_wait_seconds: float = 0.0,
        exact_mode: dict[str, Any] | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if batch_wait_seconds < 0:
            raise ValueError("batch_wait_seconds must be non-negative")
        if speculative_scheduler not in {"linear", "tree"}:
            raise ValueError("speculative_scheduler must be 'linear' or 'tree'")
        if max_branches < 1:
            raise ValueError("max_branches must be positive")
        # MLX streams are bound to the thread that first used them, so the
        # runner must be CONSTRUCTED on the same dedicated thread that will
        # serve every request (HTTP handler threads only submit work to it).
        self._inference = ThreadPoolExecutor(max_workers=1)
        self.runner = self._inference.submit(runner_factory).result()
        self.tokenizer = self.runner.tokenizer
        self.served_name = served_name
        self.default_max_tokens = default_max_tokens
        self.enable_thinking = enable_thinking
        self.draft = draft
        self.draft_model = draft_model
        self.draft_command = draft_command
        self.max_draft = max_draft
        self.speculative_scheduler = speculative_scheduler
        self.max_branches = max_branches
        self.draft_margin = draft_margin
        self.reasoning_effort = reasoning_effort
        self.max_batch_size = max_batch_size
        self.batch_wait_seconds = batch_wait_seconds
        self.exact_mode = exact_mode or {"mode": "target-verified"}
        # Built lazily on the inference thread: MLX work must stay there.
        self._drafter: Any | None = None
        self.lock = threading.Lock()
        self.tokenizer_lock = threading.Lock()
        self._batch_condition = threading.Condition()
        self._batch_pending: list[QueuedChatRequest] = []
        self._batch_closed = False
        self._batcher = (
            threading.Thread(target=self._batch_loop, name="smarttensor-batcher", daemon=True)
            if max_batch_size > 1
            else None
        )
        # Checkpoint of the KV/SSM cache at the end of rendered *history*.
        # Generation-prompt suffixes (e.g. injected empty <think> blocks) are
        # not part of the next turn's rendering, so only history is reusable.
        self.checkpoint: dict[str, Any] | None = None
        self.stop_ids: set[int] = set(getattr(self.tokenizer, "eos_token_ids", None) or ())
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            self.stop_ids.add(int(eos_token_id))
        if self._batcher is not None:
            self._batcher.start()

    def _snapshot_cache(self, cache: list[Any]) -> list[tuple[Any, Any]] | None:
        import mlx.core as mx

        try:
            states = []
            for item in cache:
                state = item.state
                arrays = state if isinstance(state, (list, tuple)) else [state]
                mx.eval([array for array in arrays if array is not None])
                states.append((state, getattr(item, "meta_state", None)))
            return states
        except Exception:
            return None

    def _restore_cache(self, states: list[tuple[Any, Any]]) -> Any | None:
        try:
            cache = self.runner._make_cache()
            if len(cache) != len(states):
                return None
            for item, (state, meta_state) in zip(cache, states):
                item.state = state
                if meta_state is not None:
                    item.meta_state = meta_state
            return cache
        except Exception:
            return None

    def _checkpoint_cache_enabled(self) -> bool:
        return getattr(self.runner, "model_type", None) != "glm_moe_dsa"

    # Cap on the persistent session corpus the prompt-lookup drafter mines
    # across requests. Keeps the most recent ~24k tokens of prior prompts and
    # completions so a later request can still draft from earlier copied code,
    # tool output, or completions even when the client drops them from history;
    # the cap bounds index-build cost per request.
    SESSION_INDEX_CAP = 24_000

    def _get_drafter(self) -> Any | None:
        if self.draft == "off":
            return None
        if self._drafter is None:
            from smarttensor.adapters.mlx import (
                ExternalProcessDrafter,
                ModelDrafter,
                PromptLookupTreeDrafter,
            )

            if self.draft == "model":
                self._drafter = ModelDrafter(self.draft_model, self.tokenizer)
            elif self.draft == "external":
                if not self.draft_command:
                    raise ValueError("--draft external requires --draft-command")
                self._drafter = ExternalProcessDrafter(self.draft_command)
            else:
                # The tree drafter is a strict superset of PromptLookupDrafter
                # (adds propose_branches) and carries the persistent
                # session-wide n-gram index across requests.
                self._drafter = PromptLookupTreeDrafter()
        return self._drafter

    def _seed_session_index(self, *token_runs: list[int]) -> None:
        """Fold prior prompt/output tokens into the drafter's session corpus.

        Proposals are guesses verified exactly downstream, so enlarging the
        mined corpus can only add candidate branches — never change emitted
        tokens. Trims the oldest tokens past SESSION_INDEX_CAP.
        """

        drafter = self._drafter
        if drafter is None or not hasattr(drafter, "seed_session"):
            return
        for tokens in token_runs:
            if tokens:
                drafter.seed_session(list(tokens))
        prefix = getattr(drafter, "_session_prefix", None)
        if prefix is not None and len(prefix) > self.SESSION_INDEX_CAP:
            drafter.reset_session()
            drafter.seed_session(prefix[-self.SESSION_INDEX_CAP :])

    def _render_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        with self.tokenizer_lock:
            return list(
                self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                    reasoning_effort=self.reasoning_effort,
                )
            )

    def _batchable(self, *, on_delta: Any | None, temperature: float) -> bool:
        # Streaming needs per-token callbacks and speculative decoding has its
        # own single-request rollback path. Batch only plain non-streaming runs.
        return (
            self.max_batch_size > 1
            and on_delta is None
            and self.draft == "off"
            and temperature >= 0
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        on_delta: Any | None = None,
    ) -> dict[str, Any]:
        if self._batchable(on_delta=on_delta, temperature=temperature):
            prompt_ids = self._render_prompt_ids(messages)
            limit = max_tokens if max_tokens and max_tokens > 0 else self.default_max_tokens
            return self._chat_batched(prompt_ids, limit, temperature)

        def run() -> dict[str, Any]:
            from smarttensor.adapters.mlx import clear_mlx_memory

            try:
                return self._chat_on_inference_thread(messages, max_tokens, temperature, on_delta)
            finally:
                clear_mlx_memory()

        return self._inference.submit(run).result()

    def _chat_batched(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        request = QueuedChatRequest(
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        with self._batch_condition:
            if self._batch_closed:
                raise RuntimeError("SmartTensor batcher is closed")
            self._batch_pending.append(request)
            self._batch_condition.notify()
        request.event.wait()
        if request.error is not None:
            raise request.error
        if request.response is None:
            raise RuntimeError("SmartTensor batcher completed without a response")
        return request.response

    def _batch_loop(self) -> None:
        while True:
            with self._batch_condition:
                while not self._batch_pending and not self._batch_closed:
                    self._batch_condition.wait()
                if not self._batch_pending and self._batch_closed:
                    return
                deadline = time.perf_counter() + self.batch_wait_seconds
                while (
                    not self._batch_closed
                    and len(self._batch_pending) < self.max_batch_size
                    and self.batch_wait_seconds > 0
                ):
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    self._batch_condition.wait(remaining)
                requests = self._pop_compatible_batch_locked()

            try:
                future = self._inference.submit(self._chat_batch_on_inference_thread, requests)
                responses = future.result()
                for request, response in zip(requests, responses):
                    request.response = response
            except BaseException as exc:
                for request in requests:
                    request.error = exc
            finally:
                for request in requests:
                    request.event.set()

    def _pop_compatible_batch_locked(self) -> list[QueuedChatRequest]:
        seed = self._batch_pending[0]
        key = seed.batch_key
        selected: list[QueuedChatRequest] = []
        remaining: list[QueuedChatRequest] = []
        for request in self._batch_pending:
            if len(selected) < self.max_batch_size and request.batch_key == key:
                selected.append(request)
            else:
                remaining.append(request)
        self._batch_pending = remaining
        return selected

    def _chat_batch_on_inference_thread(
        self,
        requests: list[QueuedChatRequest],
    ) -> list[dict[str, Any]]:
        from smarttensor.adapters.mlx import clear_mlx_memory

        started = time.perf_counter()
        try:
            result = self.runner.generate_batch(
                [f"<chat:{len(request.prompt_ids)} tokens>" for request in requests],
                max_tokens=requests[0].max_tokens,
                temperature=requests[0].temperature,
                stop_tokens=self.stop_ids,
                prompt_token_rows=[request.prompt_ids for request in requests],
            )
            seconds = time.perf_counter() - started
            responses: list[dict[str, Any]] = []
            for index, request in enumerate(requests):
                generated = list(result.generated_tokens[index])
                content_tokens = strip_trailing_stops(generated, self.stop_ids)
                content, reasoning = parse_harmony_output(self.tokenizer.decode(content_tokens))
                responses.append(
                    {
                        "content": content,
                        "reasoning": reasoning,
                        "finish_reason": (result.finish_reasons or ("length",))[index],
                        "prompt_tokens": len(request.prompt_ids),
                        "cached_tokens": 0,
                        "completion_tokens": len(generated),
                        "seconds": seconds,
                        "batch_size": len(requests),
                    }
                )
            if len(requests) > 1:
                total_tokens = sum(response["completion_tokens"] for response in responses)
                tok_per_s = total_tokens / seconds if seconds > 0 else 0.0
                print(
                    f"[smarttensor] batch size={len(requests)} "
                    f"tokens={total_tokens} seconds={seconds:.3f} tok/s={tok_per_s:.1f}",
                    flush=True,
                )
            return responses
        finally:
            clear_mlx_memory()

    def _chat_on_inference_thread(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float,
        on_delta: Any | None,
    ) -> dict[str, Any]:
        prompt_ids = list(
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                reasoning_effort=self.reasoning_effort,
            )
        )
        history_ids = list(
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
                reasoning_effort=self.reasoning_effort,
            )
        )
        canonical = (
            0 < len(history_ids) < len(prompt_ids)
            and prompt_ids[: len(history_ids)] == history_ids
        )
        limit = max_tokens if max_tokens and max_tokens > 0 else self.default_max_tokens

        with self.lock:
            started = time.perf_counter()
            cache = None
            reused = 0
            checkpoint_enabled = self._checkpoint_cache_enabled()
            if not checkpoint_enabled:
                self.checkpoint = None
            if checkpoint_enabled and self.checkpoint is not None:
                checkpoint_len = longest_reusable_prefix(self.checkpoint["tokens"], prompt_ids)
                if checkpoint_len:
                    cache = self._restore_cache(self.checkpoint["states"])
                    reused = checkpoint_len if cache is not None else 0
            if cache is None:
                cache = self.runner._make_cache()
                reused = 0

            if checkpoint_enabled and canonical and len(history_ids) > reused:
                # Advance the cache to the end of rendered history and
                # checkpoint there: that prefix re-renders identically on the
                # next turn, unlike generation-prompt suffixes and sampled text.
                self.runner.generate_batch(
                    [f"<history:{len(history_ids)} tokens>"],
                    max_tokens=0,
                    cache=cache,
                    cached_tokens=reused,
                    prompt_token_rows=[history_ids],
                )
                states = self._snapshot_cache(cache)
                self.checkpoint = (
                    {"tokens": list(history_ids), "states": states} if states else None
                )
                reused_for_generation = len(history_ids)
            else:
                reused_for_generation = reused

            collected: list[int] = []
            emitted = [""]
            done = [False]

            def on_step(step_tokens: list[int], finished: list[bool]) -> None:
                if on_delta is None or done[0]:
                    return
                token = step_tokens[0]
                if token in self.stop_ids:
                    done[0] = True
                    return
                collected.append(token)
                text = self.tokenizer.decode(collected)
                if text.endswith("�"):
                    return
                visible = streaming_visible_text(text)
                delta = visible[len(emitted[0]) :]
                if delta:
                    emitted[0] = visible
                    on_delta(delta)
            on_step.smarttensor_async_safe = True

            drafter = self._get_drafter() if temperature == 0 else None
            if drafter is not None and hasattr(drafter, "reset_session") and reused == 0:
                # A request that reused no checkpoint prefix is a fresh topic
                # (new conversation or a prompt unrelated to the last turn):
                # drop the cross-request corpus so stale spans don't crowd the
                # verifier's lanes. Continued turns keep the accumulated index.
                drafter.reset_session()
            if (
                drafter is not None
                and self.speculative_scheduler == "tree"
                and on_delta is None
                and hasattr(self.runner, "generate_tree_speculative")
            ):
                result = self.runner.generate_tree_speculative(
                    [f"<chat:{len(prompt_ids)} tokens>"],
                    max_tokens=limit,
                    stop_tokens=self.stop_ids,
                    drafter=drafter,
                    max_draft=self.max_draft,
                    max_branches=self.max_branches,
                    draft_margin=self.draft_margin,
                    cache=cache,
                    cached_tokens=reused_for_generation,
                    prompt_token_rows=[prompt_ids],
                )
            elif drafter is not None:
                result = self.runner.generate_speculative(
                    [f"<chat:{len(prompt_ids)} tokens>"],
                    max_tokens=limit,
                    stop_tokens=self.stop_ids,
                    drafter=drafter,
                    max_draft=self.max_draft,
                    draft_margin=self.draft_margin,
                    on_step=on_step,
                    cache=cache,
                    cached_tokens=reused_for_generation,
                    prompt_token_rows=[prompt_ids],
                )
            else:
                result = self.runner.generate_batch(
                    [f"<chat:{len(prompt_ids)} tokens>"],
                    max_tokens=limit,
                    temperature=temperature,
                    stop_tokens=self.stop_ids,
                    on_step=on_step,
                    cache=cache,
                    cached_tokens=reused_for_generation,
                    prompt_token_rows=[prompt_ids],
                )
            seconds = time.perf_counter() - started

            generated = list(result.generated_tokens[0])
            if drafter is not None and generated:
                # Persist this turn's completion in the session corpus so the
                # next request can draft from it even if the client trims it
                # from rendered history. Exact verification keeps output
                # correct regardless of what is indexed.
                self._seed_session_index(generated)
            content_tokens = strip_trailing_stops(generated, self.stop_ids)
            content, reasoning = parse_harmony_output(self.tokenizer.decode(content_tokens))
            speculative = getattr(result, "speculative", None)
            if speculative:
                print(
                    f"[smarttensor] speculative policy={speculative.get('draft_policy', speculative.get('drafter'))} "
                    f"passes={speculative.get('streamed_passes')} "
                    f"accepted/pass={speculative.get('accepted_tokens_per_pass', 0):.2f} "
                    f"acceptance={speculative.get('acceptance_rate', 0):.2f} "
                    f"zero_accept={speculative.get('zero_accept_rounds', 0)} "
                    f"disabled_rounds={speculative.get('disabled_rounds', 0)} "
                    f"near_ties={speculative.get('near_tie_events', 0)} "
                    f"regression_avoided={speculative.get('estimated_regression_avoided_passes', 0)}",
                    flush=True,
                )
            return {
                "content": content,
                "reasoning": reasoning,
                "finish_reason": (result.finish_reasons or ("length",))[0],
                "prompt_tokens": len(prompt_ids),
                "cached_tokens": reused,
                "completion_tokens": len(generated),
                "seconds": seconds,
                "speculative": speculative,
                "pass_economics": pass_economics(speculative),
            }

    def close(self) -> None:
        with self._batch_condition:
            self._batch_closed = True
            self._batch_condition.notify_all()
        if self._batcher is not None:
            self._batcher.join()

        def close_inference_objects() -> None:
            if self._drafter is not None and hasattr(self._drafter, "close"):
                self._drafter.close()
            self.runner.close()

        self._inference.submit(close_inference_objects).result()
        self._inference.shutdown(wait=True)


def make_handler(app: SmartTensorChat) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[smarttensor] {self.address_string()} {format % args}")

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _route(self) -> str:
            # Tolerate query strings, trailing slashes, and clients that join
            # the base URL differently (with or without the /v1 prefix).
            return self.path.split("?", 1)[0].rstrip("/")

        def do_GET(self) -> None:
            route = self._route()
            if route in {"", "/health"}:
                self._send_json(
                    {
                        "status": "ok",
                        "model": app.served_name,
                        "max_batch_size": app.max_batch_size,
                    }
                )
                return
            if route.endswith("/models") or route == "/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": app.served_name,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "smarttensor",
                            }
                        ],
                    }
                )
                return
            self._send_json({"error": {"message": f"unknown path {self.path}"}}, status=404)

        def do_POST(self) -> None:
            if not self._route().endswith("/chat/completions"):
                self._send_json({"error": {"message": f"unknown path {self.path}"}}, status=404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty list")
                max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
                temperature = float(body.get("temperature") or 0.0)
                stream = bool(body.get("stream", False))
            except Exception as exc:
                self._send_json({"error": {"message": str(exc)}}, status=400)
                return

            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    def emit(payload: dict[str, Any]) -> None:
                        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    def on_delta(delta: str) -> None:
                        emit(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": app.served_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )

                    reply = app.chat(
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        on_delta=on_delta,
                    )
                    emit(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": app.served_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": reply["finish_reason"],
                                }
                            ],
                        }
                    )
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return

                reply = app.chat(messages, max_tokens=max_tokens, temperature=temperature)
                message: dict[str, Any] = {"role": "assistant", "content": reply["content"]}
                if reply.get("reasoning"):
                    message["reasoning_content"] = reply["reasoning"]
                self._send_json(
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": app.served_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": reply["finish_reason"],
                            }
                        ],
                        "usage": {
                            "prompt_tokens": reply["prompt_tokens"],
                            "completion_tokens": reply["completion_tokens"],
                            "total_tokens": reply["prompt_tokens"] + reply["completion_tokens"],
                            "prompt_tokens_details": {"cached_tokens": reply["cached_tokens"]},
                        },
                        **(
                            {
                                "smarttensor": {
                                    "batch_size": reply["batch_size"],
                                    "seconds": reply["seconds"],
                                }
                            }
                            if reply.get("batch_size")
                            else {}
                        ),
                        "exact_mode": app.exact_mode.get("mode", "target-verified"),
                        **({"speculative": reply["speculative"]} if reply.get("speculative") else {}),
                        **(
                            {"pass_economics": reply["pass_economics"]}
                            if reply.get("pass_economics")
                            else {}
                        ),
                    }
                )
            except BrokenPipeError:
                pass
            except Exception as exc:  # surface runner errors to the client
                try:
                    self._send_json({"error": {"message": str(exc)}}, status=500)
                except Exception:
                    pass

    return Handler


def run_server(args: Any) -> int:
    from smarttensor.adapters import mlx as mlx_adapters
    from smarttensor.adapters.mlx import (
        load_mlx_config,
        select_exactness_mode,
    )
    from smarttensor.planner import parse_bytes

    config = load_mlx_config(args.model_dir)
    model_type = config.get("model_type")
    runner_class = RUNNERS.get(model_type)
    if runner_class is None:
        raise ValueError(
            f"serve currently supports {sorted(RUNNERS)} only, got {model_type!r}"
        )
    # Re-resolve by name against the live adapter module so test doubles that
    # patch smarttensor.adapters.mlx.<Runner> take effect (RUNNERS itself binds
    # the default classes at import time for dispatch/introspection).
    runner_class = getattr(mlx_adapters, runner_class.__name__, runner_class)

    # Resolve the exactness contract to concrete levers. exact-strict on
    # GPT-OSS overrides the sliding cache to temporal (bitwise exact); on Qwen
    # and others it forces speculation off (SSM chunk-exactness unproven).
    exact_mode = select_exactness_mode(
        model_type,
        getattr(args, "exact_mode", "target-verified"),
        sliding_cache=args.sliding_cache,
    )
    sliding_cache = exact_mode.sliding_cache
    draft = args.draft if exact_mode.speculation else "off"
    print(f"exactness: {exact_mode.reason}", flush=True)

    retain_layers = set(range(args.retain_layers)) if args.retain_layers is not None else None
    resident_budget = parse_bytes(args.resident_budget) if args.resident_budget else None

    print(f"loading {args.model_dir} ...", flush=True)

    def build_runner() -> Any:
        runner_kwargs = {
            "evaluate": True,
            "retain_layers": retain_layers,
            "resident_budget_bytes": resident_budget,
            "backend": args.loader_backend,
            "clear_on_evict": False,
            "pin_policy": args.pin_policy,
        }
        if model_type == "gpt_oss":
            runner_kwargs["expert_hot_set"] = args.expert_hot_set
            runner_kwargs["sliding_cache"] = sliding_cache
            runner_kwargs["native_layers"] = getattr(args, "native_layers", 0)
            runner_kwargs["pack_dir"] = getattr(args, "pack_dir", None)
            runner_kwargs["pack_read_workers"] = getattr(args, "pack_read_workers", 1)
            runner_kwargs["weight_page_budget_bytes"] = (
                parse_bytes(args.weight_page_budget) if getattr(args, "weight_page_budget", None) else None
            )
            runner_kwargs["weight_page_policy"] = getattr(args, "weight_page_policy", "auto")
            runner_kwargs["weight_page_rows"] = getattr(args, "weight_page_rows", 1)
            runner_kwargs["decode_scheduler"] = getattr(args, "decode_scheduler", "auto")
            runner_kwargs["expert_compute_mode"] = getattr(args, "expert_compute_mode", "table")
            runner_kwargs["expert_prefetch"] = getattr(args, "expert_prefetch", "off")
            runner_kwargs["expert_prefetch_cap"] = getattr(args, "expert_prefetch_cap", 32)
            if getattr(args, "expert_prefetch", "off") not in {"off", "previous"}:
                raise ValueError("--expert-prefetch on GPT-OSS supports off or previous")
            if getattr(args, "expert_compute_mode", "table") not in {"table", "direct_qmm"}:
                raise ValueError(
                    "--expert-compute-mode on GPT-OSS supports table or direct_qmm"
                )
            if getattr(args, "expert_slot_capacity", None) is not None:
                raise ValueError("--expert-slot-capacity currently applies to DeepSeek only")
        elif model_type in {"deepseek_v3", "glm_moe_dsa"}:
            runner_kwargs["pack_dir"] = getattr(args, "pack_dir", None)
            runner_kwargs["pack_read_workers"] = getattr(args, "pack_read_workers", 1)
            runner_kwargs["weight_page_budget_bytes"] = (
                parse_bytes(args.weight_page_budget) if getattr(args, "weight_page_budget", None) else None
            )
            runner_kwargs["weight_page_policy"] = getattr(args, "weight_page_policy", "auto")
            runner_kwargs["weight_page_rows"] = getattr(args, "weight_page_rows", 1)
            runner_kwargs["expert_prefetch"] = getattr(args, "expert_prefetch", "off")
            runner_kwargs["expert_prefetch_cap"] = getattr(args, "expert_prefetch_cap", 32)
            expert_compute_mode = getattr(args, "expert_compute_mode", "table")
            if model_type == "glm_moe_dsa" and expert_compute_mode == "table":
                expert_compute_mode = "direct_qmm"
            runner_kwargs["expert_compute_mode"] = expert_compute_mode
            runner_kwargs["expert_slot_capacity"] = getattr(args, "expert_slot_capacity", None)
            if getattr(args, "native_layers", 0):
                raise ValueError("--native-layers currently applies to GPT-OSS only")
            if args.expert_hot_set:
                raise ValueError("--expert-hot-set currently applies to GPT-OSS only")
            if sliding_cache != "rotating":
                raise ValueError("--sliding-cache currently applies to GPT-OSS only")
            if getattr(args, "decode_scheduler", "auto") != "auto":
                raise ValueError("--decode-scheduler currently applies to GPT-OSS only")
        elif model_type == "nemotron_h":
            runner_kwargs["page_experts"] = getattr(args, "page_experts", True)
            runner_kwargs["weight_page_budget_bytes"] = (
                parse_bytes(args.weight_page_budget) if getattr(args, "weight_page_budget", None) else None
            )
            runner_kwargs["weight_page_policy"] = getattr(args, "weight_page_policy", "auto")
            runner_kwargs["weight_page_rows"] = getattr(args, "weight_page_rows", 1)
        elif getattr(args, "native_layers", 0):
            raise ValueError("--native-layers currently applies to GPT-OSS only")
        elif getattr(args, "weight_page_budget", None):
            raise ValueError("--weight-page-budget currently applies to GPT-OSS/DeepSeek/GLM only")
        elif getattr(args, "pack_dir", None):
            raise ValueError("--pack-dir currently applies to GPT-OSS/DeepSeek/GLM only")
        elif getattr(args, "weight_page_policy", "auto") not in {"auto", "lru"}:
            raise ValueError("--weight-page-policy currently applies to GPT-OSS/DeepSeek only")
        elif getattr(args, "weight_page_rows", 1) != 1:
            raise ValueError("--weight-page-rows currently applies to GPT-OSS/DeepSeek/GLM only")
        elif model_type != "gpt_oss" and getattr(args, "expert_prefetch", "off") != "off":
            raise ValueError("--expert-prefetch currently applies to DeepSeek only")
        elif model_type != "gpt_oss" and getattr(args, "expert_compute_mode", "table") != "table":
            raise ValueError("--expert-compute-mode currently applies to GPT-OSS/DeepSeek/GLM only")
        elif getattr(args, "expert_slot_capacity", None) is not None:
            raise ValueError("--expert-slot-capacity currently applies to DeepSeek only")
        elif args.expert_hot_set:
            raise ValueError("--expert-hot-set currently applies to GPT-OSS only")
        elif sliding_cache != "rotating":
            raise ValueError("--sliding-cache currently applies to GPT-OSS only")
        return runner_class(args.model_dir, **runner_kwargs)

    served_name = args.served_name or args.model_dir.name
    app = SmartTensorChat(
        build_runner,
        served_name=served_name,
        default_max_tokens=args.max_tokens_default,
        enable_thinking=args.enable_thinking,
        draft=draft,
        draft_model=args.draft_model,
        draft_command=args.draft_command,
        max_draft=args.max_draft,
        speculative_scheduler=args.speculative_scheduler,
        max_branches=args.max_branches,
        draft_margin=args.draft_margin,
        reasoning_effort=args.reasoning_effort,
        max_batch_size=args.max_batch_size,
        batch_wait_seconds=args.batch_wait_ms / 1000,
        exact_mode=exact_mode.to_dict(),
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(
        f"smarttensor serving {served_name} at http://{args.host}:{args.port}/v1 "
        f"(base layers retained: {len(app.runner.base_retain_layers)}, "
        f"max batch: {app.max_batch_size})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()
    return 0
