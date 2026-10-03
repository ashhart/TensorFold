# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
# Adapted for tensorfold dsv41-cuda: own value typing, max_calls, unknown-tool fallback, no NaN.
"""DeepSeek-V4.1 replies: reasoning, content and DSML tool calls, whole (``parse``) or while they stream (``Stream``).

A V4.1 reply (thinking mode; the prompt ends with ``<think>``):

    reasoning</think>content

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="get_weather">
    <｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
    <｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls><｜end▁of▁sentence｜>

(chat mode: no reasoning part). DeepSeek's reference parser is strict; this one is lenient where models slip:

- a calls block inside an unclosed think block ends the reasoning there;
- ``<｜DSML｜calls>`` / V4's ``tool_calls`` / ``function_calls`` block names, missing blank lines, an unclosed last
  invoke or block (a reply cut by max_tokens) are read;
- values: ``string="true"`` keeps the text unless the schema types the parameter and the text spells that type;
  ``string="false"`` is JSON (finite numbers only), closed when it stops a bracket short (#87), else read as
  ``string="true"`` is;
- a block whose invokes all name tools the request did not offer stays in the content, whole.

Blank lines after ``</think>`` are dropped, as ``split_thinking`` drops them for every other family.

Arguments are compact JSON (``{"city":"Paris","days":3}``); while streaming, each completed parameter is sent as one
fragment (``{`` + key + value, ``,`` + key + value ..., ``}``), so the fragments concatenate to the final arguments.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tensorfold.tool_parameters import closed_json, decode_parameter

DSML = "｜DSML｜"
THINK_END = "</think>"

_BLOCK_OPEN = re.compile(r"<｜DSML｜\s?(?:calls|tool_calls|function_calls)>")
_BLOCK_CLOSE = re.compile(r"</｜DSML｜\s?(?:calls|tool_calls|function_calls)>")
_INVOKE = re.compile(r'<｜DSML｜\s?invoke\s+name="([^"]*)"\s*>')
_INVOKE_END = re.compile(r"</｜DSML｜\s?invoke>")
_PARAM = re.compile(r'<｜DSML｜\s?parameter\s+name="([^"]*)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜\s?parameter>',
                    re.DOTALL)
OPEN_TAGS = tuple(f"{sep}<{DSML}{name}>" for sep in ("", "\n\n") for name in (" calls", "calls", "tool_calls",
                                                                                  "function_calls"))
BARE_TAGS = OPEN_TAGS[:4]


def _partial(text: str, tag: str) -> int:
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


def _hold(text: str) -> int:
    """Trailing characters that may still become a calls block's opening tag (with its blank line)."""

    return max(_partial(text, tag) for tag in OPEN_TAGS)


def _reasoning_hold(text: str) -> int:
    """Trailing characters of an open think block that may still become ``</think>`` or a calls block's opener: the
    newlines before an opener are not reasoning either."""

    k = max(_partial(text, THINK_END), max(_partial(text, tag) for tag in BARE_TAGS))
    if k == 0 or any(_partial(text, tag) == k for tag in BARE_TAGS):
        head = text[:len(text) - k]
        k += len(head) - len(head.rstrip("\n"))
    return k


def _schemas(tools: Sequence[dict] | None) -> dict[str, dict]:
    out = {}
    for t in tools or ():
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn, dict) and fn.get("name"):
            params = fn.get("parameters") or fn.get("input_schema") or {}
            props = params.get("properties") if isinstance(params, dict) else None
            out[str(fn["name"])] = props if isinstance(props, dict) else {}
    return out


def _known(name: str, schemas: dict[str, dict]) -> str | None:
    if name in schemas:
        return name
    low = {k.lower(): k for k in schemas}
    return low.get(name.lower()) or low.get(name.split("::")[-1].lower())


def _finite(text: str) -> tuple[bool, Any]:
    try:
        v = json.loads(text)
        json.dumps(v, allow_nan=False)      # NaN / Infinity would make the streamed arguments invalid JSON
    except (ValueError, TypeError, RecursionError):
        return False, None
    return True, v


def value(raw: str, string: str, prop: Any) -> Any:
    """A parameter's value (see the module docstring)."""

    prop = prop if isinstance(prop, dict) else {}
    if string == "false":
        ok, v = _finite(raw)
        if ok:
            return v
        fixed = closed_json(raw.strip())
        if fixed is not None:
            ok, v = _finite(fixed)
            if ok:
                return v
    return decode_parameter(raw, prop, python=False)


def fragment(first: bool, key: str, val: Any) -> str:
    return ("{" if first else ",") + json.dumps(key, ensure_ascii=False) + ":" + \
        json.dumps(val, ensure_ascii=False, separators=(",", ":"))


def new_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


@dataclass
class Call:
    name: str
    args: list[tuple[str, Any]] = field(default_factory=list)
    closed: bool = False                     # its ``</｜DSML｜ invoke>`` was written (False: cut by max_tokens)
    id: str = field(default_factory=new_id)

    def arguments(self) -> str:
        if not self.args:
            return "{}"
        return "".join(fragment(i == 0, k, v) for i, (k, v) in enumerate(self.args)) + "}"

    def openai(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments()}}


def _calls(block: str, schemas: dict[str, dict]) -> list[Call]:
    """Invokes of a calls block body, complete parameters only (the first of a repeated name wins)."""

    out: list[Call] = []
    pos = 0
    while True:
        m = _INVOKE.search(block, pos)
        if m is None:
            break
        name = m.group(1)
        end = _INVOKE_END.search(block, m.end())
        nxt = _INVOKE.search(block, m.end())
        stop = end.start() if end is not None and (nxt is None or end.start() < nxt.start()) else \
            (nxt.start() if nxt is not None else len(block))
        call = Call(name)
        props = schemas.get(_known(name, schemas) or name, {})
        seen = set()
        for p in _PARAM.finditer(block, m.end(), stop):
            key = p.group(1)
            if key in seen:
                continue
            seen.add(key)
            call.args.append((key, value(p.group(3), p.group(2), props.get(key))))
        call.closed = end is not None and stop == end.start()
        out.append(call)
        pos = end.end() if call.closed else stop
        if not call.closed and nxt is None:
            break
    return out


def _usable(calls: list[Call], schemas: dict[str, dict], max_calls: int | None) -> list[Call]:
    """The calls of offered tools (names as offered), the first ``max_calls`` of them."""

    out = []
    for c in calls:
        known = _known(c.name, schemas)
        if known is None:
            continue
        if max_calls is not None and len(out) >= max_calls:
            break
        c.name = known
        out.append(c)
    return out


@dataclass
class Reply:
    reasoning: str
    content: str
    calls: list[Call]


def _split(text: str, thinking: bool, calls: bool) -> tuple[str, str]:
    """(reasoning, the rest)."""

    if not thinking:
        return "", text
    end = text.find(THINK_END)
    blk = _BLOCK_OPEN.search(text) if calls else None
    if end >= 0 and (blk is None or end < blk.start()):
        return text[:end], text[end + len(THINK_END):].lstrip("\n")
    if blk is not None:                      # calls inside an unclosed think block: the reasoning ends there
        reasoning = text[:blk.start()].rstrip("\n")
        return reasoning, text[len(reasoning):]
    return text, ""


def parse(text: str, *, thinking: bool, tools: Sequence[dict] | None = None,
          max_calls: int | None = None) -> Reply:
    """A finished reply's parts (EOS already removed); the text after a calls block is dropped."""

    reasoning, rest = _split(text, thinking, bool(tools))
    blk = _BLOCK_OPEN.search(rest) if tools else None
    if blk is None:
        return Reply(reasoning, rest, [])
    close = _BLOCK_CLOSE.search(rest, blk.end())
    body = rest[blk.end():close.start() if close else len(rest)]
    schemas = _schemas(tools)
    usable = _usable(_calls(body, schemas), schemas, max_calls)
    if not usable:
        return Reply(reasoning, rest, [])
    return Reply(reasoning, rest[:blk.start()].removesuffix("\n\n"), usable)


class Stream:
    """A reply's deltas while it grows: ``feed(text)`` (the whole decoded text so far) -> OpenAI delta dicts
    (``reasoning`` / ``content`` text, ``tool_calls`` entries); ``finish(text)`` releases what was held back.
    Concatenating the deltas gives ``parse(text)``'s parts (tool-call arguments included)."""

    def __init__(self, *, thinking: bool, tools: Sequence[dict] | None, max_calls: int | None = None) -> None:
        self.thinking = thinking
        self.tools = list(tools or [])
        self.schemas = _schemas(self.tools)
        self.max_calls = max_calls
        self.sent_reasoning = 0
        self.sent_content = 0
        self.calls: list[Call] = []
        self.sent_args: list[int] = []          # parameters sent, per call
        self.sent_close: list[bool] = []
        self.in_calls = False

    def _content_limit(self, rest: str, finished: bool) -> tuple[int, Any]:
        """How much of the rest is content now, and the calls block's match (or None)."""

        blk = _BLOCK_OPEN.search(rest) if self.tools else None
        if blk is not None:
            cut = blk.start()
            if rest[:cut].endswith("\n\n"):
                cut -= 2
            return cut, blk
        if finished:
            return len(rest), None
        return len(rest) - (_hold(rest) if self.tools else 0), None

    def feed(self, text: str, finished: bool = False) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.thinking:
            end = text.find(THINK_END)
            blk = _BLOCK_OPEN.search(text) if self.tools else None
            if end < 0 and blk is None:
                hold = _reasoning_hold(text) if self.tools else _partial(text, THINK_END)
                upto = len(text) if finished else len(text) - hold
                if upto > self.sent_reasoning:
                    out.append({"reasoning": text[self.sent_reasoning:upto]})
                    self.sent_reasoning = upto
                return out
        reasoning, rest = _split(text, self.thinking, bool(self.tools))
        if len(reasoning) > self.sent_reasoning:
            out.append({"reasoning": reasoning[self.sent_reasoning:]})
            self.sent_reasoning = len(reasoning)
        limit, blk = self._content_limit(rest, finished)
        if limit > self.sent_content and not self.in_calls:
            out.append({"content": rest[self.sent_content:limit]})
            self.sent_content = limit
        if blk is None:
            return out
        self.in_calls = True
        close = _BLOCK_CLOSE.search(rest, blk.end())
        body = rest[blk.end():close.start() if close else len(rest)]
        for k, c in enumerate(_usable(_calls(body, self.schemas), self.schemas, self.max_calls)):
            if k == len(self.calls):
                self.calls.append(c)
                self.sent_args.append(0)
                self.sent_close.append(False)
                out.append({"tool_calls": [{"index": k, "id": c.id, "type": "function",
                                            "function": {"name": c.name, "arguments": ""}}]})
            mine = self.calls[k]
            mine.args, mine.closed = c.args, c.closed
            frag = "".join(fragment(j == 0, key, val) for j, (key, val) in enumerate(c.args[self.sent_args[k]:],
                                                                                       start=self.sent_args[k]))
            self.sent_args[k] = len(c.args)
            if (c.closed or finished) and not self.sent_close[k]:
                frag += "}" if c.args else "{}"
                self.sent_close[k] = True
            if frag:
                out.append({"tool_calls": [{"index": k, "function": {"arguments": frag}}]})
        if finished and not self.calls:
            # no offered tool in the block: nothing of it was sent, so it goes out as content, as ``parse`` keeps it
            out.append({"content": rest[self.sent_content:]})
            self.sent_content = len(rest)
        return out

    def finish(self, text: str) -> list[dict[str, Any]]:
        return self.feed(text, finished=True)
