"""``/v1/decisions`` in SGLang prompt-format version 1: one prefill, label logits, no generated text.

The wording, single-token label check and probability math follow
``sglang.srt.entrypoints.openai.serving_decisions`` (prompt format version 1).
TensorFold reads the logits from its own prefill instead of SGLang's ``score_prompts``.
"""

from __future__ import annotations

import json
import math
import string
from dataclasses import dataclass
from typing import Any

from tensorfold.server.text import render_prompt_ids

PROMPT_FORMAT_VERSION = 1

_TOP_FIELDS = frozenset({
    "input", "questions", "temperature", "chat_template_kwargs",
    "prompt_format_version", "return_prompt_token_ids", "model",
})
_CHOICE_FIELDS = frozenset({"id", "type", "question", "options"})
_SCORE_FIELDS = frozenset({"id", "type", "question", "levels"})
_YES_NO_FIELDS = frozenset({"id", "type", "question", "yes", "no"})
_OPTION_FIELDS = frozenset({"name", "description"})


class DecisionError(ValueError):
    """A request this route refuses before scoring."""


@dataclass(frozen=True)
class PreparedQuestion:
    id: str
    kind: str
    names: list[str]
    prompt_ids: list[int]
    label_ids: list[int]


def prepare(tokenizer: Any, body: dict[str, Any], *, context_len: int | None = None) -> list[PreparedQuestion]:
    """Render every question and check that each answer label is one token."""

    _validate_body(body)
    text = _render_text(body.get("input"))
    if not text.strip():
        raise DecisionError("input must not be blank")
    prepared = []
    for index, question in enumerate(body["questions"]):
        try:
            prepared.append(_prepare_question(tokenizer, text, question, context_len=context_len))
        except DecisionError as exc:
            ident = question.get("id") if isinstance(question, dict) else None
            where = repr(ident) if isinstance(ident, str) and ident else f"at position {index}"
            raise DecisionError(f"question {where}: {exc}") from exc
    return prepared


def build_response(
    body: dict[str, Any],
    prepared: list[PreparedQuestion],
    scored: list[tuple[list[float], float]],
) -> dict[str, Any]:
    """SGLang's decision response. ``scored`` is label logits and the full-vocabulary logsumexp, one row a question."""

    temperature = float(body.get("temperature") or 1.0)
    answers = {}
    prompt_tokens = 0
    for item, (logits, logsumexp) in zip(prepared, scored):
        probabilities = _softmax(logits, temperature)
        logprobs = [logit - logsumexp for logit in logits]
        answer: dict[str, Any] = {
            "type": item.kind,
            "probabilities": dict(zip(item.names, probabilities)),
            "label_mass": math.fsum(math.exp(logprob) for logprob in logprobs),
        }
        if item.kind == "choice":
            answer["choice"] = item.names[probabilities.index(max(probabilities))]
        elif item.kind == "score":
            answer["score"] = math.fsum(index * probability for index, probability in enumerate(probabilities))
        if body.get("return_prompt_token_ids"):
            answer["prompt_token_ids"] = item.prompt_ids
            answer["label_token_ids"] = item.label_ids
        answers[item.id] = answer
        prompt_tokens += len(item.prompt_ids)
    return {
        "object": "decisions",
        "model": body.get("model") or "default",
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "answers": answers,
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens},
    }


def _validate_body(body: dict[str, Any]) -> None:
    unknown = sorted(set(body) - _TOP_FIELDS)
    if unknown:
        raise DecisionError(f"unknown field {unknown[0]!r}")
    version = body.get("prompt_format_version")
    if version is not None and version != PROMPT_FORMAT_VERSION:
        raise DecisionError(
            f"prompt_format_version {version} is not served, this server uses version {PROMPT_FORMAT_VERSION}"
        )
    temperature = body.get("temperature", 1)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise DecisionError("temperature must be a number above 0")
    kwargs = body.get("chat_template_kwargs") or {}
    if not isinstance(kwargs, dict):
        raise DecisionError("chat_template_kwargs must be an object")
    if "enable_thinking" in kwargs and kwargs["enable_thinking"] is not False:
        raise DecisionError("decisions need enable_thinking false or unset")
    questions = body.get("questions")
    if not isinstance(questions, list) or not questions:
        raise DecisionError("questions must contain at least one question")
    seen: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            raise DecisionError("each question must be an object")
        ident = question.get("id")
        if not isinstance(ident, str) or not ident.strip():
            raise DecisionError("a question id must not be blank")
        if ident in seen:
            raise DecisionError(f"question id {ident!r} is repeated")
        seen.add(ident)


def _prepare_question(
    tokenizer: Any, text: str, question: dict[str, Any], *, context_len: int | None,
) -> PreparedQuestion:
    kind, names, labels, content = _wording(text, question)
    prompt, prompt_ids = _chat_prompt(tokenizer, content)
    return _finish(
        question["id"], kind, names, labels, prompt, prompt_ids, context_len,
        lambda rendered: _encode(tokenizer, rendered),
    )


def prompts_for(
    body: dict[str, Any], render: Any, encode: Any, *, context_len: int | None = None,
) -> list[PreparedQuestion]:
    """The same questions as prepare, for a server that renders text and encodes it itself (the CUDA GLM template)."""

    _validate_body(body)
    text = _render_text(body.get("input"))
    if not text.strip():
        raise DecisionError("input must not be blank")
    prepared = []
    for index, question in enumerate(body["questions"]):
        try:
            prepared.append(_from_text(render, encode, text, question, context_len=context_len))
        except DecisionError as exc:
            ident = question.get("id") if isinstance(question, dict) else None
            where = repr(ident) if isinstance(ident, str) and ident else f"at position {index}"
            raise DecisionError(f"question {where}: {exc}") from exc
    return prepared


def _from_text(
    render: Any, encode: Any, text: str, question: dict[str, Any], *, context_len: int | None,
) -> PreparedQuestion:
    kind, names, labels, content = _wording(text, question)
    prompt = render(content)
    prompt_ids = [int(token) for token in encode(prompt)]
    return _finish(question["id"], kind, names, labels, prompt, prompt_ids, context_len, encode)


def _wording(text: str, question: dict[str, Any]) -> tuple[str, list[str], list[str], str]:
    kind = question.get("type")
    if kind == "choice":
        _unknown(question, _CHOICE_FIELDS)
        names, details, labels = _choice(question)
        closing = "Answer with the letter of one option only."
        lines = [_question_line(question)]
        for label, name, detail in zip(labels, names, details):
            lines.append(f"{label}: {name} - {detail}" if detail else f"{label}: {name}")
    elif kind == "score":
        _unknown(question, _SCORE_FIELDS)
        names, details, labels = _score(question)
        closing = "Answer with the number of one level only."
        lines = [_question_line(question), *[f"{label}: {detail}" for label, detail in zip(labels, details)]]
    elif kind == "yes_no":
        _unknown(question, _YES_NO_FIELDS)
        names, details, labels = ["yes", "no"], [_optional_text(question.get("yes")), _optional_text(question.get("no"))], ["yes", "no"]
        closing = "Answer with yes or no only."
        lead = _render_text(question.get("question"))
        if not lead.strip():
            raise DecisionError("a question must not be blank")
        lines = [f"Is the following true? {lead}"]
        for label, detail in zip(labels, details):
            if detail:
                lines.append(f"{label}: {detail}")
    else:
        raise DecisionError(f"unknown question type {kind!r}")
    if kind != "yes_no":
        lines = [line for line in lines if line]
    lines.append(closing)
    return kind, names, labels, "\n".join([text, "", *lines])


def _finish(
    question_id: str, kind: str, names: list[str], labels: list[str], prompt: str, prompt_ids: list[int],
    context_len: int | None, encode: Any,
) -> PreparedQuestion:
    if context_len and len(prompt_ids) >= context_len:
        raise DecisionError(
            f"the prompt has {len(prompt_ids)} tokens, which does not fit the context length of {context_len} tokens"
        )
    return PreparedQuestion(
        id=question_id, kind=kind, names=names, prompt_ids=prompt_ids,
        label_ids=_label_ids(encode, prompt, prompt_ids, labels),
    )


def _choice(question: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    options = question.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 26:
        raise DecisionError("a choice needs 2 to 26 options")
    names, details = [], []
    seen: set[str] = set()
    for option in options:
        if not isinstance(option, dict):
            raise DecisionError("each option must be an object")
        _unknown(option, _OPTION_FIELDS)
        name = option.get("name")
        if not isinstance(name, str) or _bad_name(name):
            raise DecisionError("option names must not be blank or contain line breaks")
        folded = name.strip().casefold()
        if folded in seen:
            raise DecisionError(f"option name {name!r} repeats another name")
        seen.add(folded)
        names.append(name.strip())
        details.append(_optional_text(option.get("description")))
    return names, details, list(string.ascii_uppercase[: len(names)])


def _score(question: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    levels = question.get("levels")
    if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
        raise DecisionError("a score needs 2 to 10 levels")
    details = []
    for level in levels:
        text = _render_text(level)
        if not text.strip():
            raise DecisionError("a level must not be blank")
        details.append(text)
    names = [str(index) for index in range(len(details))]
    return names, details, names


def _question_line(question: dict[str, Any]) -> str:
    text = _render_text(question.get("question"))
    if not text.strip():
        raise DecisionError("a question must not be blank")
    return f"Question: {text}"


def _chat_prompt(tokenizer: Any, content: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": content}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        text = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(text, str):
        raise DecisionError("the chat template did not return text")
    try:
        prompt_ids = render_prompt_ids(tokenizer, messages, enable_thinking=False)
    except Exception as exc:
        raise DecisionError(f"the chat template failed: {exc}") from exc
    encoded = _encode(tokenizer, text)
    if prompt_ids != encoded:
        if prompt_ids[: len(encoded)] == encoded:
            text += tokenizer.decode(prompt_ids[len(encoded):])
            encoded = _encode(tokenizer, text)
        if prompt_ids != encoded:
            raise DecisionError("this tokenizer does not encode the rendered chat text back to the same ids")
    if text.rfind("<think>") > text.rfind("</think>"):
        raise DecisionError("the chat template leaves a reasoning block open at the answer position")
    return text, prompt_ids


def _label_ids(encode: Any, prompt: str, prompt_ids: list[int], labels: list[str]) -> list[int]:
    found = []
    for label in labels:
        ids = [int(token) for token in encode(prompt + label)]
        if len(ids) != len(prompt_ids) + 1 or ids[:-1] != prompt_ids or ids[-1] in found:
            raise DecisionError(
                f"the answer label {label!r} is not one distinct token after the chat prompt for this tokenizer, "
                "so this model is not supported"
            )
        found.append(ids[-1])
    return found


def reduce_vocab_shards(rows: list[list[float]], label_ids: list[int], shard: int) -> tuple[list[float], float]:
    """Join per-rank vocabulary shards into the label logits and the full-vocabulary logsumexp."""

    if shard < 1 or not rows or any(not row for row in rows):
        raise ValueError("label scoring needs a positive shard width and one row per rank")
    peak = max(max(row) for row in rows)
    total = math.fsum(math.exp(value - peak) for row in rows for value in row)
    if not math.isfinite(peak) or total <= 0 or not math.isfinite(total):
        raise ValueError("label scoring produced a non-finite logit")
    logsumexp = peak + math.log(total)
    logits = []
    for token in label_ids:
        rank, column = divmod(int(token), shard)
        if rank < 0 or rank >= len(rows) or column >= len(rows[rank]):
            raise ValueError(f"label token {token} is outside the vocabulary")
        logits.append(float(rows[rank][column]))
    if not math.isfinite(logsumexp) or any(not math.isfinite(value) for value in logits):
        raise ValueError("label scoring produced a non-finite logit")
    return logits, logsumexp


def _softmax(logits: list[float], temperature: float) -> list[float]:
    # Shift before dividing: finite logits / a tiny positive temperature can
    # overflow, whereas the maximum's shifted value stays exactly zero.
    peak = max(logits)
    weights = [math.exp((logit - peak) / temperature) for logit in logits]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def _render_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise DecisionError("text must be a string, object, or array")


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    return _render_text(value)


def _unknown(obj: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise DecisionError(f"unknown field {unknown[0]!r}")


def _bad_name(name: str) -> bool:
    stripped = name.strip()
    return not stripped or any(ord(char) < 32 for char in stripped)
