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
