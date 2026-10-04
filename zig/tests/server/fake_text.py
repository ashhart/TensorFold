"""The parity tests' tokenizer and template for the Python server, byte for byte as fake_text.zig."""

from __future__ import annotations

import json
from typing import Any

SOURCE = "{# fake template: 'low' 'medium' 'xhigh' enable_thinking #}"


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def render(messages: list[dict[str, Any]], tools: Any = None, add_generation_prompt: bool = True,
           enable_thinking: bool = False, reasoning_effort: str | None = None, **_: Any) -> str:
    """Tools, each message, then the assistant header and an open think block when thinking."""
    out = []
    if tools:
        out.append("<|im_start|>system\n# Tools\n" + json.dumps(tools, ensure_ascii=False) + "<|im_end|>\n")
    for m in messages:
        role = _text(m.get("role"))
        out.append(f"<|im_start|>{role}\n")
        if m.get("reasoning_content"):
            out.append(f"<think>\n{_text(m['reasoning_content'])}\n</think>\n\n")
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.append("".join(_text(p.get("text")) if isinstance(p, dict) else "" for p in content))
        calls = m.get("tool_calls")
        for call in calls if isinstance(calls, list) else []:
            fn = call.get("function") if isinstance(call, dict) else None
            fn = fn if isinstance(fn, dict) else {}
            body = json.dumps({"name": fn.get("name"), "arguments": fn.get("arguments")}, ensure_ascii=False)
            out.append(f"\n<tool_call>\n{body}\n</tool_call>")
        if role == "tool":
            out.append(f"\n[call {m.get('tool_call_id')}]")
        out.append("<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
        if enable_thinking:
            out.append("<think>\n")
            if reasoning_effort:
                out.append(f"[effort {reasoning_effort}]\n")
    return "".join(out)


class FakeTokenizer:
    """Ids 0-255 are bytes, then whole pieces; encoding takes the longest piece at each byte."""

    unk_token_id = None
    chat_template = SOURCE

    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces
        self.raw = [p.encode() for p in pieces]
        self.ids = {p: 256 + i for i, p in enumerate(pieces)}
        self.eos_token_ids = {self.ids["<|im_end|>"], self.ids["<|endoftext|>"]}

    def __len__(self) -> int:
        return 256 + len(self.pieces)

    def encode(self, text: str, add_special_tokens: bool = True, **_: Any) -> list[int]:
        data, out, i = text.encode(), [], 0
        while i < len(data):
            best = None
            for j, p in enumerate(self.raw):
                if len(p) >= 2 and data.startswith(p, i) and (best is None or len(p) > len(self.raw[best])):
                    best = j
            if best is None:
                out.append(data[i])
                i += 1
            else:
                out.append(256 + best)
                i += len(self.raw[best])
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = False, **_: Any) -> str:
        data = b"".join(bytes([t]) if t < 256 else self.raw[t - 256] if t - 256 < len(self.raw) else b"" for t in ids)
        return data.decode("utf-8", errors="replace")

    def convert_tokens_to_ids(self, text: str) -> int | None:
        if text in self.ids:
            return self.ids[text]
        return ord(text) if len(text) == 1 and ord(text) < 0x80 else None

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        out = []
        for t in ids:
            if t >= 256:
                out.append(self.pieces[t - 256] if t - 256 < len(self.pieces) else "")
            elif 0x20 <= t < 0x7F:
                out.append(chr(t))
            else:
                out.append(f"<0x{t:02X}>")
        return out

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = True, **kwargs: Any) -> Any:
        text = render(messages, **kwargs)
        return self.encode(text) if tokenize else text
