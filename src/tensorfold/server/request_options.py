"""Numeric request validation and model defaults shared by the HTTP layer and chat app."""

from __future__ import annotations

import math
import re
from typing import Any

from tensorfold.server.errors import RequestError
from tensorfold.server.stopping import stop_options


_INTEGER_FIELDS = {"seed", "top_k", "thinking_budget", "max_tokens", "max_completion_tokens"}


def parse_numbers(fields: dict[str, Any]) -> dict[str, Any]:
    stop_options(fields)
    parsed = dict(fields)
    for name in (*sorted(_INTEGER_FIELDS), "temperature", "top_p", "min_p"):
        value = fields.get(name)
        if value is None:
            continue
        integer = name in _INTEGER_FIELDS
        try:
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError
            number = int(value) if integer else float(value)
            if integer and isinstance(value, float) and value != number:
                raise ValueError
            if not integer and not math.isfinite(number):
                raise ValueError
        except (ValueError, TypeError, OverflowError) as exc:
            kind = "an integer" if integer else "a finite number"
            raise RequestError(f"{name} must be {kind} or null") from exc
        if name == "min_p" and not 0.0 <= number <= 1.0:
            raise RequestError("min_p must be between 0 and 1, or null")
        parsed[name] = max(0, number) if name == "top_k" else number
    return parsed


EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


_EFFORT_ORDER = ("xhigh", "high", "medium", "low", "minimal")   # highest first


def nearest_named_effort(effort: str, levels: frozenset[str]) -> str:
    """The named level nearest effort, the higher one when two are as near; effort itself when the
    template names no level, or when effort is not on the five-level ladder (none is not a level).

    A template that names none of the five keeps the effort as sent: its else-branch is the
    template's own default. GLM-5.3 names low and high but no medium, whose requests used to
    fall out of the template's else-branch as Max -- its own ceiling, not the level asked for.
    """

    if not levels or effort not in _EFFORT_ORDER:
        return effort
    want = _EFFORT_ORDER.index(effort)
    return min(levels, key=lambda name: (abs(_EFFORT_ORDER.index(name) - want), _EFFORT_ORDER.index(name)))


def effort_levels(template: str | None) -> frozenset[str]:
    """The efforts a chat template names: Qwen3.8's low, medium and xhigh; GLM-5.3's low and high."""

    return frozenset(re.findall(r"""['"](minimal|low|medium|high|xhigh)['"]""", template or ""))


def coerce_effort(effort: str | None, levels: frozenset[str] = frozenset()) -> str | None:
    """The name the template hears. None stays None, so an omitted effort keeps the template's own default."""

    if effort is None:
        return None
    # a medium the template does not name lands on the nearest named level, the higher when two are as near
    if effort == "medium" and effort not in levels:
        return nearest_named_effort(effort, levels)
    # OpenAI's "high" and "minimal" are "xhigh" and "low" unless the template names them (GLM-5.3 names "high")
    if effort in levels or effort not in ("high", "minimal"):
        return effort
    return "xhigh" if effort == "high" else "low"


def heard_effort(explicit: str | None, default: str | None, levels: frozenset[str]) -> str | None:
    """A request's effort when it set one, otherwise the server default, both in the template's names."""

    return coerce_effort(default if explicit is None else explicit, levels)


def thinking_fields(body: dict[str, Any], levels: frozenset[str] = frozenset()) -> dict[str, Any]:
    """A request's ``reasoning_effort`` and ``enable_thinking`` where it sets them; unset is the server's default."""

    kwargs = body.get("chat_template_kwargs") or {}
    fields: dict[str, Any] = {}
    effort = body.get("reasoning_effort")
    if effort is None and isinstance(kwargs, dict):
        effort = kwargs.get("reasoning_effort")           # where vLLM's clients put it
    if effort is not None:
        if not isinstance(effort, str) or effort not in EFFORTS:
            raise RequestError("reasoning_effort must be none, minimal, low, medium, high or xhigh")
        fields["reasoning_effort"] = coerce_effort(effort, levels)
        fields["enable_thinking"] = effort != "none"
    if isinstance(kwargs, dict) and "enable_thinking" in kwargs:          # an explicit switch wins
        fields["enable_thinking"] = bool(kwargs["enable_thinking"])
        if fields["enable_thinking"] and fields.get("reasoning_effort") == "none":
            fields.pop("reasoning_effort")
    return fields


class RequestOptions:
    """Resolve sampling and thinking controls before a request reaches the engine."""

    @property
    def effort_levels(self) -> frozenset[str]:
        """The reasoning efforts this model's chat template names (``effort_levels``), read once."""

        found = self.__dict__.get("_effort_levels")
        if found is None:
            template = getattr(self.tokenizer, "chat_template", None)
            if isinstance(template, dict):                   # named templates: any of them may name a level
                template = " ".join(str(t) for t in template.values())
            found = self.__dict__["_effort_levels"] = effort_levels(template if isinstance(template, str) else None)
        return found

    def effort_for(self, explicit: str | None) -> str | None:
        """The effort the template hears: the request's name, or this server's default."""

        return heard_effort(explicit, getattr(self, "reasoning_effort", None), self.effort_levels)

    def _resolve_sampling(self, fields: dict[str, Any] | None, temperature: float,
                          prompt_ids: list[int]) -> Any:
        """Omitted or null fields keep model defaults; an omitted seed is keyed to the prompt."""

        from tensorfold.engine.exact_sampling import Sampling, seed_for

        options = {k: v for k, v in parse_numbers(self.default_sampling or {}).items() if v is not None}
        options.update({k: v for k, v in parse_numbers(fields or {}).items() if v is not None})
        temp = options.get("temperature", 0.0)
        if temp <= 0.0:
            return None
        return Sampling(seed=options.get("seed", seed_for(prompt_ids)), temperature=temp,
                        top_k=options.get("top_k", 0), top_p=options.get("top_p", 1.0),
                        min_p=options.get("min_p", 0.0))

    def _call_gate(self, fields: dict[str, Any], prompt_ids: list[int], tools: Any) -> Any:
        """The gate that opens a required tool call (``tool_choice`` "required" or a named function), else None."""

        from tensorfold.engine.call_gate import CallGate

        if not fields.get("tool_call_required"):
            return None
        form = self._call_form()
        opener = self._token_id(form[0]) if form else -1
        if opener < 0:
            raise RequestError('tool_choice "required" or a named function needs a chat template that marks tool calls '
                               '(<tool_call> or <|tool_call>), and this one does not: send "auto"')
        opens, closes = getattr(self, "think_markers", ("", "</think>"))
        if opens:                           # Gemma 4's thought channel opens with a token, then a word
            with self.tokenizer_lock:
                think_open = int(self.tokenizer.encode(opens, add_special_tokens=False)[0])
        else:
            think_open = self._token_id("<think>")
        names = [str((t.get("function") or t).get("name") or "") for t in tools or ()]
        return CallGate.after_prompt(prompt_ids, opener, self._blank, think_open=think_open,
                                     think_end=self._token_id(closes), text=self._text, encode=self._encode,
                                     lead=form[1] or "", names=names if form[1] is not None else (), tail=form[2] or "")

    def _call_form(self) -> tuple[str, str, str] | None:
        """(opener, lead, tail) of this template's tool calls, read once from a rendered call."""

        from tensorfold.engine.call_gate import call_format
        from tensorfold.server.text import _CALLS, render_prompt_ids

        if not hasattr(self, "_form"):
            probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "tfprobe_fn", "arguments": {}}}]}]
            with self.tokenizer_lock:
                try:
                    text = self.tokenizer.decode(render_prompt_ids(self.tokenizer, probe, add_generation_prompt=False))
                except Exception:  # noqa: BLE001 - a template that renders no calls: the opener alone
                    text = ""
            self._form = call_format(text, "tfprobe_fn", [o for o, _ in _CALLS]) or next(
                ((o, None, None) for o, _ in _CALLS if self._token_id(o) >= 0), None)      # no names: format unknown
        return self._form

    def _text(self, token: int) -> str:
        with self.tokenizer_lock:
            return self.tokenizer.decode([int(token)], skip_special_tokens=False)

    def _encode(self, text: str) -> list[int]:
        with self.tokenizer_lock:
            return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def _token_id(self, text: str) -> int:
        """The id of a token the tokenizer has whole, else -1."""

        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)     # a tokenizer without it has no whole tokens
        with self.tokenizer_lock:
            found = convert(text) if convert is not None else None
            unk = getattr(self.tokenizer, "unk_token_id", None)
        return int(found) if isinstance(found, int) and found >= 0 and found != unk else -1

    def _blank(self, token: int) -> bool:
        """Whether a token is whitespace only, which a required call's answer may begin with (never an end token)."""

        if token in self.stop_ids:
            return False
        with self.tokenizer_lock:
            return not self.tokenizer.decode([int(token)], skip_special_tokens=False).strip()

    def _think_close(self) -> tuple[tuple[int, ...], int]:
        """The forced close and its end token, or -1 when the tokenizer has no think-end token."""

        if self._think_tokens is None:
            convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
            with self.tokenizer_lock:
                end = convert("</think>") if convert is not None else None
                unk = getattr(self.tokenizer, "unk_token_id", None)
                if not isinstance(end, int) or end < 0 or end == unk:
                    self._think_tokens = ((), -1)
                else:
                    lead = self.tokenizer.encode("\n", add_special_tokens=False)
                    trail = self.tokenizer.encode("\n\n", add_special_tokens=False)
                    self._think_tokens = ((*lead, end, *trail), int(end))
        return self._think_tokens
