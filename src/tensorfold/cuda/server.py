"""OpenAI server for the CUDA engines: a family's ``cuda_engine`` gives ``eos``, ``generate`` and ``follow``."""
from __future__ import annotations

import hashlib
import inspect
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tensorfold.engine import grammar
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import CONTEXT_LIMIT, RequestError, refusal
from tensorfold.server.messages import validate_modalities
from tensorfold.server.probabilities import TokenBytes, probability_options
from tensorfold.server.request_options import heard_effort, parse_numbers, thinking_fields
from tensorfold.server.stopping import matched_stop, stop_options
from tensorfold.server.token_routes import flag, token_ids
from tensorfold.server.tool_policy import ToolCallPolicy
from tensorfold.engine.call_gate import CallGate, ThinkBudget, call_format, generate_gated
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server.tools import active_tool_specs, tool_choice_requires_call

from tensorfold.cuda import health
from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.cuda.reply_text import StopStrings, StreamDecoder, hide_tool_calls, parse_tool_calls
from tensorfold.cuda.turns import Turns, Yield
from tensorfold.server.text import is_title_request, reasoning_count, split_thinking
from tensorfold.vision.images import DEFAULT_LIMITS, ImageLimits


# -- requests --------------------------------------------------------------------------------

_MADE = threading.Lock()                # guards the lazily made per-app ``Turns``

_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seed")


@dataclass(slots=True)
class PreparedRequest:
    prompt: list[int]
    max_tokens: int
    tools: list[dict[str, Any]]
    thinking: bool
    sampling: Any          # the engine's ``Sampling``, or None for greedy decoding
    ignore_eos: bool = False
    stop: tuple[str, ...] = ()
    vision: Any = None
    grammar: Any = None     # (spec, compiled grammar) of the request's response_format, or None
    think_budget: int = 0   # reply tokens before the server closes a think block the reply leaves open (0: no limit)


def _native_context(model_dir: Path) -> int:
    path = model_dir / "config.json"
    if not path.exists():
        return 0
    config = json.loads(path.read_text())
    text = config.get("text_config") or config
    limit = text.get("max_position_embeddings") or config.get("max_position_embeddings")
    return int(limit) if isinstance(limit, int) and limit > 0 else 0


class App:
    """Serve one engine with sampling and reply-length defaults for requests that omit them."""

    reads_ignore_eos = False            # True where the engine reads ``ignore_eos`` itself; a ``stop_eos`` engine is given it

    def __init__(self, engine, model_dir: Path, served: str, *, default_thinking: bool = False,
                 sampling: dict[str, Any] | None = None, max_tokens: int = 4096,
                 context_window: int | None = None, reasoning_effort: str | None = None, thinking_budget: int = 0,
                 aliases: tuple[str, ...] | list[str] = (), vision_max_images: int | None = None):
        from tokenizers import Tokenizer

        self.engine = engine
        self.vision = getattr(engine, "vision", None)
        self.image_limits = DEFAULT_LIMITS if vision_max_images is None else ImageLimits(max_images=vision_max_images)
        self.served = served
        self.aliases = tuple(str(alias).strip() for alias in aliases if str(alias).strip())
        self.model_dir = Path(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.template = ChatTemplate(model_dir)
        self.default_thinking = default_thinking
        self.reasoning_effort, self.thinking_budget = reasoning_effort, int(thinking_budget)   # the Mac's defaults
        self.sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, **(sampling or {})}
        self.max_tokens = int(max_tokens)
        self.native_context_window = _native_context(model_dir)
        self.context_window = self.native_context_window if context_window is None else int(context_window)
        if self.context_window < 0:
            raise ValueError("context_window must be 0 or a positive token count")
        self.turns = Turns()                # one request at a time where the engine decodes one

    @property
    def model_ids(self) -> list[str]:
        """The ids this endpoint answers to, as the MLX server lists them: ``--name`` first, then each ``--alias``."""

        ids: list[str] = []
        for model_id in (self.served, *getattr(self, "aliases", ())):
            if model_id and model_id not in ids:
                ids.append(model_id)
        return ids

    def reply_model(self, body: Any) -> str:
        """The id a reply names: the one the request asked for when this endpoint answers to it, else ``--name``."""

        asked = body.get("model") if isinstance(body, dict) else None
        return asked if isinstance(asked, str) and asked in self.model_ids else self.served

    def _check_fields(self, body: dict[str, Any]) -> str | None:
        import inspect

        if not isinstance(body, dict):
            return "the request body must be a JSON object"
        try:
            probability_options(body, supported=bool(getattr(self.engine, "supports_logprobs", False)))
        except RequestError as exc:
            return str(exc)
        if body.get("draft", True) is False and "draft" not in inspect.signature(self.engine.generate).parameters:
            return "this model's CUDA engine has no serial switch (\"draft\": false)"
        if not isinstance(body.get("messages", []), list):
            return "messages must be a list"
        problem = grammar.refusal(body)                 # a malformed grammar field, or one beside a required call
        if problem is None and grammar.request_spec(body) and "constraint" not in inspect.signature(
                self.engine.generate).parameters:
            problem = "this model's engine does not enforce structured output"
        return problem

    def _grammars(self) -> grammar.Grammars:
        return grammar.compiler(self, getattr(self, "model_dir", None), self.engine.eos)

    def _engine_capacity(self) -> int | None:
        capacities = []
        for name in ("context_window", "limit"):
            limit = getattr(self.engine, name, None)
            if isinstance(limit, int):
                capacities.append(max(0, limit))
        return min(capacities) if capacities else None

    def _restart(self, need: int, ranks: str = "") -> str:
        """A larger ``--context`` to restart with, only where the startup admission would accept it."""

        largest = (getattr(self.engine, "capacity_plan", None) or {}).get("largest_window")
        if largest is None or need > largest:
            return ""
        return f", or restart{ranks} with --context {need} or more (this memory admits up to {largest})"

    def _context_limit(self) -> int | None:
        limits = [self.context_window] if self.context_window > 0 else []
        capacity = self._engine_capacity()
        if capacity is not None:
            limits.append(capacity)
        return min(limits) if limits else None

    @property
    def effective_context_window(self) -> int | None:
        """Safe prompt-plus-reply capacity; None is unlimited, while zero refuses every prompt."""

        return self._context_limit()

    def decisions(self, body: dict[str, Any]) -> dict[str, Any]:
        """Answer typed questions from next-token label logits. No text is generated."""

        from jinja2.exceptions import TemplateError

        from tensorfold.server.decisions import DecisionError, build_response, prompts_for

        if not hasattr(self.engine, "score_labels"):
            raise RequestError("this model's CUDA engine does not score decision labels")

        def render(content: str) -> str:
            try:
                return self.template.render([{"role": "user", "content": content}], tools=None, enable_thinking=False)
            except TemplateError as exc:
                raise DecisionError(f"the chat template failed: {exc}") from exc

        def encode(text: str) -> list[int]:
            return [int(token) for token in self.tok.encode(text, add_special_tokens=False).ids]

        try:
            prepared = prompts_for(body, render, encode, context_len=self.effective_context_window)
        except DecisionError as exc:
            raise RequestError(str(exc)) from exc
        turns = self._turns()
        turns.take(False)
        try:
            scored = []
            for item in prepared:
                try:
                    scored.append(self.engine.score_labels(item.prompt_ids, item.label_ids))
                except ValueError as exc:
                    raise RequestError(f"question {item.id!r}: {exc}") from exc
            return build_response(body, prepared, scored)
        finally:
            turns.give()

    def _requested_tokens(self, body: dict[str, Any]) -> int:
        for name in ("max_tokens", "max_completion_tokens"):
            value = body.get(name)
            if value is not None:
                try:
                    int(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RequestError(f"{name} must be an integer token count") from exc
        return max(1, int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens))

    def _prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        from jinja2.exceptions import TemplateError

        validate_modalities(body)
        ToolCallPolicy(body)
        ignore_eos, stop = stop_options(body)
        max_tokens = self._requested_tokens(body)
        try:
            tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
        except ValueError as exc:
            raise RequestError(str(exc)) from None
        kwargs = body.get("chat_template_kwargs")
        if kwargs is None:                   # absent or null: the server's defaults
            kwargs = {}
        elif not isinstance(kwargs, dict):   # [], "", false and 0 included
            raise RequestError("chat_template_kwargs must be a JSON object or null")
        kwargs = dict(kwargs)
        # reasoning_effort and enable_thinking as the Mac server reads them: the template hears an effort when thinking
        levels = getattr(self.template, "efforts", frozenset())
        fields = thinking_fields(body, levels)
        kwargs.pop("enable_thinking", None)
        kwargs.pop("reasoning_effort", None)
        thinking = bool(fields.get("enable_thinking", self.default_thinking))
        effort = heard_effort(fields.get("reasoning_effort"), getattr(self, "reasoning_effort", None), levels)
        if thinking and effort:
            kwargs["reasoning_effort"] = effort
        budget = parse_numbers({"thinking_budget": body.get("thinking_budget")})["thinking_budget"]
        budget = int(budget or getattr(self, "thinking_budget", 0)) if chat and thinking else 0     # 0: the default
        spec = grammar.request_spec(body)
        top = probability_options(body, supported=bool(getattr(self.engine, "supports_logprobs", False)))
        if top is not None:
            if not chat or body.get("stream") or thinking or tools or stop or spec is not None or budget:
                raise RequestError("logprobs support nonstreamed text chat with thinking off, without tools, "
                                   "stop strings or structured output")
            if not hasattr(self, "_probability_decoder"):
                self._probability_decoder = TokenBytes(self.tok)
        compiled = (spec, self._grammars().compile(spec)) if spec is not None else None
        if chat:
            if not isinstance(body.get("messages"), list):
                raise RequestError("messages must be a list")
            from tensorfold.server.prompts import has_images, prepare_images

            def render(messages: list[dict[str, Any]], **images: bool) -> str:   # text renders as it always has
                try:
                    return self.template.render(messages, tools=tools, enable_thinking=thinking, extra=kwargs,
                                                **images)
                except TemplateError as exc:     # the checkpoint's template refuses the request (``raise_exception``)
                    raise RequestError(f"the chat template rejected the request: {exc}") from exc

            if has_images(body["messages"]):
                rendered = prepare_images(self.vision, body["messages"],
                                          lambda messages: render(messages, allow_images=True),
                                          context_limit=self._context_limit(),
                                          limits=getattr(self, "image_limits", DEFAULT_LIMITS))
                return PreparedRequest(rendered.tokens, max_tokens, tools, thinking,
                                       self.sampling_for(body, rendered.tokens), ignore_eos=ignore_eos, stop=stop,
                                       vision=rendered.vision, grammar=compiled, think_budget=budget)
            text = render(body["messages"])
            prompt = self.tok.encode(text, add_special_tokens=False).ids
        elif isinstance(body.get("prompt"), list):       # token ids (vLLM's and OpenAI's form): served as given
            prompt = self.token_ids(body["prompt"])
        else:
            text = body.get("prompt")
            if not isinstance(text, str):
                raise RequestError("prompt must be a string or a list of token ids")
            prompt = self.tok.encode(text, add_special_tokens=flag(body, "add_special_tokens", False)).ids
        if not prompt:
            raise RequestError("rendered prompt is empty")
        # sampling is resolved here, so a malformed control is refused before a stream opens
        return PreparedRequest(prompt, max_tokens, tools, thinking, self.sampling_for(body, prompt),
                               ignore_eos=ignore_eos, stop=stop, grammar=compiled, think_budget=budget)

    def token_ids(self, value: Any, field: str = "prompt") -> list[int]:
        """Token ids as a request gives them (a list, or a list holding one list); RequestError outside the
        vocabulary."""

        size = getattr(self.tok, "get_vocab_size", None)
        return token_ids(value, size(with_added_tokens=True) if size is not None else None, field)

    def tokenize(self, body: dict[str, Any]) -> dict[str, Any]:
        """vLLM's ``/tokenize`` (``server.token_routes``): a prompt's ids (``add_special_tokens``, default true, as
        vLLM's), or ``messages``' as the chat route renders them (``add_generation_prompt``, default true)."""

        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        strings = flag(body, "return_token_strs", False)
        if "messages" in body:
            fields = dict(body)
            if "add_generation_prompt" in body:
                kwargs = body.get("chat_template_kwargs")
                kwargs = {} if kwargs is None else kwargs
                if not isinstance(kwargs, dict):
                    raise RequestError("chat_template_kwargs must be a JSON object or null")
                fields["chat_template_kwargs"] = {**kwargs,
                                                  "add_generation_prompt": flag(body, "add_generation_prompt", True)}
            ids = self._prepare(fields, True).prompt
        else:
            text = body.get("prompt")
            if not isinstance(text, str):
                raise RequestError("prompt must be a string (or send messages)")
            ids = self.tok.encode(text, add_special_tokens=flag(body, "add_special_tokens", True)).ids
        limit = self._context_limit()
        reply: dict[str, Any] = {"count": len(ids), "tokens": [int(t) for t in ids],
                                 "max_model_len": limit if limit is not None else self.native_context_window}
        if strings:
            reply["token_strs"] = [self.tok.id_to_token(t) for t in reply["tokens"]]
        return reply

    def detokenize(self, body: dict[str, Any]) -> dict[str, Any]:
        """vLLM's ``/detokenize``: the text of ``tokens``, special tokens included."""

        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        return {"prompt": self.tok.decode(self.token_ids(body.get("tokens"), "tokens"), skip_special_tokens=False)}

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Why the request cannot run, or None; rendered before a stream's headers are sent."""

        problem = self._check_fields(body)
        if problem:
            return problem
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        limit = self._context_limit()
        if limit is not None and len(prepared.prompt) >= limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            native = f" (model window: {self.native_context_window} tokens)" if self.native_context_window else ""
            return (f"{CONTEXT_LIMIT} {limit} tokens: the rendered prompt has {len(prepared.prompt)} tokens and leaves "
                    f"no room for a reply in the server's {limit}-token {kind}{native}, which exceeds the context "
                    f"window; shorten the prompt"
                    f"{self._restart(len(prepared.prompt) + 1)}")
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        if limit is not None and asked and len(prepared.prompt) + prepared.max_tokens > limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            return (f"{CONTEXT_LIMIT} {limit} tokens: the rendered prompt has {len(prepared.prompt)} tokens and "
                    f"requests {prepared.max_tokens} reply tokens, which exceeds the context window (the server's "
                    f"{limit}-token {kind}); reduce the prompt or reply length"
                    f"{self._restart(len(prepared.prompt) + prepared.max_tokens)}")
        return None

    def prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        problem = self._check_fields(body)
        if problem:
            raise refusal(problem)
        prepared = self._prepare(body, chat)
        problem = self.check(body, prepared=prepared)
        if problem:
            raise refusal(problem)
        limit = self._context_limit()
        if limit is not None:
            prepared.max_tokens = min(prepared.max_tokens, limit - len(prepared.prompt))
        return prepared

    def sampling_for(self, body: dict[str, Any], prompt: list[int]):
        """Keyed sampling (the seed, else one drawn from the prompt), or None for greedy; RequestError if malformed."""

        from tensorfold.engine.exact_sampling import Sampling, seed_for

        fields = parse_numbers({k: body[k] for k in _SAMPLING_FIELDS if body.get(k) is not None})
        temp = float(fields.get("temperature", self.sampling["temperature"]))
        if temp <= 0:
            return None
        seed = fields.get("seed")
        top_k = fields.get("top_k", self.sampling["top_k"])
        top_p = fields.get("top_p", self.sampling["top_p"])
        min_p = fields.get("min_p", self.sampling.get("min_p", 0.0))
        return Sampling(int(seed) if seed is not None else seed_for(prompt), temp, int(top_k), float(top_p),
                        float(min_p))

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        """One reply; once ``cancelled()`` holds, a waiting request raises ``RequestCancelled`` unstarted, a running one stops at its next round and raises it after ``generate``."""

        arrived = time.perf_counter()
        prepared = prepared if prepared is not None else self.prepare(body, chat)
        prompt, max_tokens = prepared.prompt, prepared.max_tokens
        tools, thinking = prepared.tools, prepared.thinking
        policy = ToolCallPolicy(body)
        sampling = prepared.sampling
        # end tokens end the reply and stay out of its text, unless it asks ignore_eos of an engine that reads it
        takes_stop_eos = "stop_eos" in inspect.signature(self.engine.generate).parameters
        ends = () if prepared.ignore_eos and (self.reads_ignore_eos or takes_stop_eos) else tuple(self.engine.eos)
        stops = StopStrings(prepared.stop, self.tok, ends)
        out: list[int] = []
        sent = {"reasoning": 0, "content": 0}
        stopped = {"client": False, "stop": False}
        failed: list[Exception] = []
        stream = StreamDecoder(self.tok, ends)
        # calls stream as argument deltas while written (as on the Mac); one-call requests keep the end parser
        calls_stream = ToolCallStreamer(tools) if tools and not policy.single else None
        answer_raw = [""]

        def visible(finished: bool) -> tuple[str, str]:
            raw = stream.final() if finished else stream.text
            # stop strings match the generated text, reasoning included, before it is split (as on the Mac)
            raw = stops.visible(raw, partial=not finished) if stops.strings else raw
            if chat and thinking:
                reasoning, answer = split_thinking(raw, finished=finished)
            else:
                reasoning, answer = "", raw
            answer_raw[0] = answer
            if tools:
                answer = (policy.content(answer, finished=finished) if policy.single
                          else hide_tool_calls(answer, finished=finished))
            return reasoning, answer

        serving: list[Any] = [None]

        def on_tokens(new: list[int]) -> bool:
            # True stops the engine after this round; engines that finish on both ranks keep calling and get True
            if stopped["client"] or stopped["stop"] or failed:
                return True
            try:
                if stops.strings:
                    kept = []
                    for token in new:             # token by token: the round's width cannot move the cut
                        kept.append(token)
                        out.append(token)
                        if stops.hit(out):
                            stopped["stop"] = True
                            break
                    new = kept
                else:
                    out.extend(new)
                stream.add(new)
                reasoning, answer = visible(False)
                delta: dict[str, Any] = {}
                if len(reasoning) > sent["reasoning"]:
                    delta["reasoning_content"] = reasoning[sent["reasoning"]:]
                    sent["reasoning"] = len(reasoning)
                if len(answer) > sent["content"]:
                    delta["content"] = answer[sent["content"]:]
                    sent["content"] = len(answer)
                if delta and not emit(delta):
                    stopped["client"] = True
                if calls_stream is not None and not stopped["client"]:
                    for call_delta in calls_stream.feed(answer_raw[0]):   # never the reasoning
                        if not emit(call_delta):
                            stopped["client"] = True
                            break
                if not stopped["client"] and cancelled is not None and cancelled():   # every round, text or not
                    stopped["client"] = True
                if serving[0] is not None:
                    serving[0].saw()
            except Exception as exc:        # noqa: BLE001  raised after generate returns, never into the engine
                failed.append(exc)
                return True
            return stopped["client"] or stopped["stop"]

        draft = body.get("draft", True) is not False
        gate = self._call_gate(prompt, tools) if tools and tool_choice_requires_call(body.get("tool_choice")) else None

        options: dict[str, Any] = {} if draft else {"draft": False}
        probabilities = None
        if body.get("logprobs"):
            from tensorfold.engine.probabilities import Probabilities

            probabilities = Probabilities(body.get("top_logprobs") or 0, len(prompt), max_tokens)
            options["probabilities"] = probabilities
        if takes_stop_eos:
            options["stop_eos"] = not prepared.ignore_eos
        shaped = prepared.grammar is not None or prepared.think_budget > 0
        think_end = self.tok.token_to_id("</think>") if chat and thinking and shaped else None
        budget = self._think_budget(prepared, think_end)
        # priority "background" (or a session-title request): after the others, as on the Mac
        background = body.get("priority") == "background" or (chat and is_title_request(body.get("messages"), tools))
        concurrent = getattr(self.engine, "concurrent", False)
        turns = None if concurrent else self._turns()
        if concurrent and background and "background" in inspect.signature(self.engine.generate).parameters:
            options["background"] = True            # the engine's scheduler orders its lanes and prompts
        # one engine at a time: a background reply yields between rounds (not on two ranks, which decode to the end)
        yielding = background and turns is not None and getattr(self.engine, "tp", 1) == 1
        gates = [g for g in (gate, budget, Yield(turns) if yielding else None) if g is not None]

        cached: list[int] = []              # the prompt tokens the first run found cached (usage's cached_tokens)

        def generate(ids: list[int], count: int, feed: Callable[[list[int]], bool]) -> Any:
            if yielding and cached:                 # a run after a cut: foreground requests waiting go first
                turns.give()
                turns.take(True, cancelled)
            extra = dict(options)
            if prepared.grammar is not None:    # response_format: a fresh grammar state, after </think> when thinking
                spec, compiled = prepared.grammar
                # a run after the thinking budget's close (it ends at </think> under a grammar) starts in the grammar
                end = None if think_end is None or think_end in ids[len(prompt):] else think_end
                extra["constraint"] = self._grammars().constraint(compiled, think_end=end, spec=spec)
            if prepared.vision is not None:          # a gate's continuation keeps the images, positions extended
                from tensorfold.vision.qwen_processing import continued

                same = list(ids) == list(prepared.vision.token_ids)
                extra["vision"] = prepared.vision if same else continued(prepared.vision, ids,
                                                                       self.vision.frontend.config)
            stats = self.engine.generate(ids, count, sampling, feed, **extra)
            if not cached:
                cached.append(int((stats or {}).get("cached") or 0))
            return stats

        # an engine that decodes concurrent requests together (``concurrent``) takes them as they come
        if turns is not None:
            turns.take(background, cancelled)
        try:
            if cancelled is not None and cancelled():                # the client left while this request waited
                raise RequestCancelled("the client left before the request started")
            with health.of(self).running(len(prompt), out, arrived) as request:  # /health reads ``out``; rounds never call in
                serving[0] = request
                try:
                    stats = request.stats = generate_gated(generate, prompt, max_tokens, gates, on_tokens)
                finally:
                    serving[0] = None
        finally:
            if turns is not None:
                turns.give()
        if failed:
            raise failed[0]
        if stopped["client"]:                                        # as the Mac server: nothing more is written
            raise RequestCancelled("the client left during the reply")
        stats = {**(stats or {}), "token_sha": token_sha(out)}
        reasoning, answer = visible(True)
        final: dict[str, Any] = {}
        if len(reasoning) > sent["reasoning"]:
            final["reasoning_content"] = reasoning[sent["reasoning"]:]
        raw_text = self.tok.decode([t for t in out if t not in ends], skip_special_tokens=False)
        text = stops.visible(raw_text)
        raw_answer = split_thinking(text, finished=True)[1] if chat and thinking else text
        content, calls = parse_tool_calls(raw_answer, tools, max_calls=policy.max_calls) if tools else (answer, None)
        content = policy.content(content) if tools else content
        tail = content[sent["content"]:] if content.startswith(answer[:sent["content"]]) else ""
        if tail:
            final["content"] = tail
        finish = "tool_calls" if calls else ("stop" if stopped["stop"] or (out and out[-1] in ends) else "length")
        print_done(len(prompt), (cached or [0])[0], thinking, out, finish, stats, request)
        if body.get("return_token_ids"):              # the reply's ids in the "tensorfold" block, for exactness checks
            stats = {**(stats or {}), "token_ids": [int(t) for t in out]}
        logprobs = (self._probability_decoder.format(probabilities.emitted(out), ends)
                    if probabilities is not None else None)
        # the calls already sent as deltas; the handler sends the rest (a call the streamer could not follow)
        streamed = calls_stream.index + 1 if calls_stream is not None and calls_stream.streamed else 0
        return {"final": final, "calls": calls, "finish": finish, "content": content, "reasoning": reasoning,
                "stop_sequence": matched_stop(raw_text, stops.strings),
                **({"logprobs": logprobs} if logprobs is not None else {}),
                "prompt_tokens": len(prompt), "completion_tokens": len(out), "cached_tokens": (cached or [0])[0],
                "reasoning_tokens": reasoning_count(out, self.tok.token_to_id("</think>") if chat and thinking else None),
                "stats": stats, "calls_streamed": streamed}

    def _turns(self) -> Turns:
        """The engine's turns (one request at a time, background ones last), made on first use."""

        with _MADE:
            return self.__dict__.setdefault("turns", Turns())

    def _think_budget(self, prepared: PreparedRequest, think_end: int | None) -> ThinkBudget | None:
        """The Mac's thinking budget: the cut becomes a newline, </think> and a blank line (</think> alone under a grammar)."""

        if prepared.think_budget <= 0 or think_end is None:
            return None
        close = [*self.tok.encode("\n", add_special_tokens=False).ids, think_end]
        if prepared.grammar is None:
            close += self.tok.encode("\n\n", add_special_tokens=False).ids
        return ThinkBudget(prepared.think_budget, close, think_end)

    def _call_gate(self, prompt: list[int], tools: list[dict[str, Any]]) -> CallGate:
        """The gate a required tool call needs, from this template's call markup and the rendered prompt."""

        if not hasattr(self, "_form"):
            probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "tfprobe_fn", "arguments": {}}}]}]
            try:
                text = self.template.render(probe, tools=None, enable_thinking=False)
            except Exception:  # noqa: BLE001 - a template that renders no calls: the opener alone
                text = ""
            openers = [o for o in ("<tool_call>", "<|tool_call>") if self.tok.token_to_id(o) is not None]
            self._form = call_format(text, "tfprobe_fn", openers) or ((openers[0], None, None) if openers else None)
        if self._form is None:
            raise RequestError('tool_choice "required" or a named function needs a chat template that marks tool calls '
                               '(<tool_call> or <|tool_call>), and this one does not: send "auto"')
        opener, lead, tail = self._form
        eos = set(self.engine.eos)

        def text(token: int) -> str:
            return self.tok.decode([token], skip_special_tokens=False)

        def blank(token: int) -> bool:
            return token not in eos and not text(token).strip()

        think = [-1 if self.tok.token_to_id(t) is None else self.tok.token_to_id(t) for t in ("<think>", "</think>")]
        names = [str((t.get("function") or t).get("name") or "") for t in tools] if lead is not None else []
        return CallGate.after_prompt(prompt, self.tok.token_to_id(opener), blank, think_open=think[0],
                                     think_end=think[1], text=text, lead=lead or "", names=names, tail=tail or "",
                                     encode=lambda t: list(self.tok.encode(t, add_special_tokens=False).ids))


def token_sha(tokens: list[int]) -> str:
    """A reply's token ids, hashed as the Mac server does: drafted and ``"draft": false`` replies must match."""

    return hashlib.sha256(",".join(str(int(t)) for t in tokens).encode()).hexdigest()[:12]


def print_done(prompt: int, cached: int, thinking: bool, out: list[int], finish: str, stats: dict[str, Any],
               request: Any) -> None:
    """The Mac server's ``done`` line for a finished reply; tok/s runs from the first token to the last."""

    ended = time.perf_counter()
    first, started = getattr(request, "first", None), getattr(request, "started", ended)
    decode = ended - first if first is not None else 0.0
    rate = (len(out) - 1) / decode if decode > 0 and len(out) > 1 else 0.0
    print(f"[tensorfold] done req-{uuid.uuid4().hex[:12]} prompt={prompt} cached={cached} thinking={thinking} "
          f"tokens={len(out)} sha={token_sha(out)} finish={finish} tok/s={rate:.1f} "
          f"ttft={(first - started) if first is not None else -1:.2f}s prefill={stats.get('prefill_s', -1):.2f}s "
          f"rounds={stats.get('rounds', 0)} accepted={stats.get('accepted', 0)}/{stats.get('drafted', 0)}",
          flush=True)


from tensorfold.cuda.http import Server, make_handler, serve, usage_of  # noqa: E402,F401  (the HTTP side)
