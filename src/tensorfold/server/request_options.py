"""Numeric request validation and model defaults shared by the HTTP layer and chat app."""

from __future__ import annotations

import math
from typing import Any

from tensorfold.server.errors import RequestError
from tensorfold.server.stopping import stop_options


_INTEGER_FIELDS = {"seed", "top_k", "thinking_budget", "max_tokens", "max_completion_tokens"}


def parse_numbers(fields: dict[str, Any]) -> dict[str, Any]:
    stop_options(fields)
    parsed = dict(fields)
    for name in (*sorted(_INTEGER_FIELDS), "temperature", "top_p"):
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
        parsed[name] = max(0, number) if name == "top_k" else number
    return parsed


class RequestOptions:
    """Resolve sampling and thinking controls before a request reaches the engine."""

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
                        top_k=options.get("top_k", 0), top_p=options.get("top_p", 1.0))

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

        with self.tokenizer_lock:
            found = self.tokenizer.convert_tokens_to_ids(text)
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
            with self.tokenizer_lock:
                end = self.tokenizer.convert_tokens_to_ids("</think>")
                unk = getattr(self.tokenizer, "unk_token_id", None)
                if not isinstance(end, int) or end < 0 or end == unk:
                    self._think_tokens = ((), -1)
                else:
                    lead = self.tokenizer.encode("\n", add_special_tokens=False)
                    trail = self.tokenizer.encode("\n\n", add_special_tokens=False)
                    self._think_tokens = ((*lead, end, *trail), int(end))
        return self._think_tokens
