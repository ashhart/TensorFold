"""Prompt rendering and streamed reply text with think blocks, tool markup and locked tokenizer access."""

from __future__ import annotations

from typing import Any

from tensorfold.server.messages import _normalize_tool_call_arguments, late_system_role, normalize_messages

_THINK_END = "</think>"
_THINK_OPEN = "<think>"                  # a reply that opens its own block writes this (Kolibri 1); stripped
# (what a reply writes to open its think block, what closes it): Qwen's prompt opens the block; Gemma 4's reply does
THINK_MARKERS = ("", _THINK_END)
CHANNEL_MARKERS = ("<|channel>thought", "<channel|>")
# (opener, closer) of a tool call's markup: Qwen's, Gemma 4's, DeepSeek-V4's DSML block
_CALLS = (("<tool_call>", "</tool_call>"), ("<|tool_call>", "<tool_call|>"),
          ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"))


def _partial_tag(text: str, tag: str) -> int:
    """How many characters at the end of ``text`` could be the start of ``tag``."""

    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


def split_thinking(text: str, *, finished: bool, markers: tuple[str, str] = THINK_MARKERS) -> tuple[str, str]:
    """(reasoning, answer) of a thinking reply; while it streams, a tail that could begin a marker is held back."""

    # Gemma 4 can open its thought channel after visible text. Strip that block wherever it opens.
    opener, closer = markers
    if opener and not text.startswith(opener):
        start = text.find(opener)
        if start > 0:                                        # a thought channel opened after some visible text
            prefix = text[:start]
            prefix = prefix[: len(prefix) - _partial_tag(prefix, opener)]   # drop a stray/doubled partial opener
            reasoning, answer = split_thinking(text[start:], finished=finished, markers=markers)
            return reasoning, prefix + answer
        if not finished:                                     # no opener yet: hold a tail that could begin one
            return "", text[: len(text) - max(_partial_tag(text, tag)
                                               for tag in (opener, closer, *(o for o, _ in _CALLS)))]
        return "", text                                      # a finished reply that never opened the block
    if opener:
        text = text[len(opener):].lstrip("\n")
    elif text.startswith(_THINK_OPEN):                   # a reply that opens its own block (Kolibri 1's template)
        text = text[len(_THINK_OPEN):].lstrip("\n")
    elif not finished and _THINK_OPEN.startswith(text):  # a tail that may still become that opener: held
        return "", ""
    end = text.find(closer)
    if end >= 0:
        return text[:end], text[end + len(closer):].lstrip("\n")
    call = min((at for at in (text.find(opener) for opener, _ in _CALLS) if at >= 0), default=-1)
    if call >= 0:
        # a call written before the block closes: held while the block may still close, the answer if the reply ends
        if not finished:
            return text[:call], ""
        if any(text.startswith(opener, call) and close in text[call:] for opener, close in _CALLS):
            return text[:call], text[call:]
        return text, ""
    held = 0 if finished else max(_partial_tag(text, tag) for tag in (closer, *(opener for opener, _ in _CALLS)))
    return text[: len(text) - held], ""


def reasoning_count(tokens: list[int], think_end: int | None) -> int:
    """A thinking reply's reasoning tokens: through its close ``think_end`` (None or -1: not thinking), else all."""

    if think_end is None or think_end < 0:
        return 0
    return tokens.index(think_end) + 1 if think_end in tokens else len(tokens)


def think_markers(tokenizer: Any) -> tuple[str, str]:
    """Gemma 4's thought channel when the tokenizer has its ``<channel|>`` token, else Qwen's ``</think>``."""

    try:
        close = tokenizer.convert_tokens_to_ids(CHANNEL_MARKERS[1])
    except Exception:  # noqa: BLE001 - a tokenizer without the lookup: the default markers
        return THINK_MARKERS
    unk = getattr(tokenizer, "unk_token_id", None)
    return CHANNEL_MARKERS if isinstance(close, int) and close >= 0 and close != unk else THINK_MARKERS


def hide_tool_calls(text: str, *, finished: bool) -> str:
    """Hide tool-call blocks and partial opening tags while streaming so visible text only grows and calls arrive as deltas."""

    out: list[str] = []
    pos = 0
    while True:
        found = [(text.find(opener, pos), opener, closer) for opener, closer in _CALLS]
        found = [f for f in found if f[0] >= 0]
        if not found:
            tail = text[pos:]
            held = 0 if finished else max(_partial_tag(tail, opener) for opener, _ in _CALLS)
            out.append(tail[: len(tail) - held])
            return "".join(out)
        start, opener, closer = min(found)
        out.append(text[pos:start])
        end = text.find(closer, start + len(opener))
        if end < 0:
            return "".join(out)
        pos = end + len(closer)


def render_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
    reasoning_effort: str | None = None,
    late_system: str | None = None,
) -> list[int]:
    """Render chat tokens; Qwen3.8 thinking effort ``medium`` preserves the system block, while default ``xhigh`` adds an instruction."""

    if late_system is None:
        late_system = template_late_system(tokenizer)
    messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=late_system))
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "thinking_mode": "thinking" if enable_thinking else "chat",   # DeepSeek-V4's templates read this switch
    }
    if enable_thinking and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if tools:
        kwargs["tools"] = tools
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("tools", None)
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(rendered, str):
        rendered = tokenizer.encode(rendered)
    ids = [int(t) for t in rendered]
    if not enable_thinking and add_generation_prompt:
        ids = _close_open_think(tokenizer, ids)
    return ids


def _close_open_think(tokenizer: Any, ids: list[int]) -> list[int]:
    """Close a trailing bare ``<think>`` when thinking is off: GLM-5.3's template writes one whatever the switch."""

    if not ids:
        return ids
    try:
        if tokenizer.decode([ids[-1]]).strip() != "<think>":
            return ids
        close = tokenizer.encode("</think>", add_special_tokens=False)
    except Exception:  # noqa: BLE001 - a tokenizer without these tokens keeps the prompt as rendered
        return ids
    return ids + [int(t) for t in close] if len(close) == 1 else ids


def template_late_system(tokenizer: Any) -> str:
    """The role a later system message renders as in this tokenizer's chat template (see ``late_system_role``)."""

    return late_system_role(lambda messages: tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False))


def eos_ids_of(tokenizer: Any) -> frozenset[int]:
    values = getattr(tokenizer, "eos_token_ids", None)
    if values:
        return frozenset(int(t) for t in values)
    single = getattr(tokenizer, "eos_token_id", None)
    return frozenset({int(single)}) if single is not None else frozenset()


def is_title_request(messages: list[dict[str, Any]], tools: Any) -> bool:
    """Identify short session-title requests without tools so they can yield to foreground turns."""

    if tools or not messages or messages[0].get("role") != "system":
        return False
    size = sum(len(m["content"]) if isinstance(m.get("content"), str) else 4096 for m in messages)
    text = messages[0].get("content")
    return size < 4096 and isinstance(text, str) and "title" in text.lower()



class IncrementalText:
    """Decode only new tokens with the previous chunk as context so multi-byte characters split across tokens remain whole."""

    def __init__(self, tokenizer: Any, lock: Any) -> None:
        self.tokenizer = tokenizer
        self.lock = lock
        self.tokens: list[int] = []
        self.text = ""
        self._prefix = 0
        self._read = 0

    def extend(self, tokens: list[int]) -> str:
        self.tokens.extend(int(t) for t in tokens)
        with self.lock:
            before = self.tokenizer.decode(self.tokens[self._prefix:self._read])
            after = self.tokenizer.decode(self.tokens[self._prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            self.text += after[len(before):]
            self._prefix, self._read = self._read, len(self.tokens)
        return self.text


class _LockedTokenizer:
    """The app's tokenizer behind its lock, for proposers on the scheduler thread."""

    def __init__(self, tokenizer: Any, lock: Any) -> None:
        self._tokenizer = tokenizer
        self._lock = lock

    def decode(self, ids: list[int]) -> str:
        with self._lock:
            return self._tokenizer.decode(ids)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        with self._lock:
            return self._tokenizer.encode(text, **kwargs)

    def convert_tokens_to_ids(self, token: str) -> Any:
        with self._lock:
            return self._tokenizer.convert_tokens_to_ids(token)


def strip_trailing_stops(tokens: list[int], stop_ids: set[int]) -> list[int]:
    end = len(tokens)
    while end > 0 and tokens[end - 1] in stop_ids:
        end -= 1
    return tokens[:end]


HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"
HARMONY_ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
HARMONY_TERMINATORS = ("<|return|>", "<|end|>", "<|call|>", "<|start|>")


def parse_harmony_output(text: str) -> tuple[str, str | None]:
    """Split Harmony content and reasoning, pass other text unchanged, and return empty content when the final channel is absent."""

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
