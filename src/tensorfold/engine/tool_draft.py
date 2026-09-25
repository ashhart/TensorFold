"""Drafts for tool calls: the call's fixed structure, proposed ahead of the model.

Qwen 3.x writes a call as

    <tool_call>
    <function=NAME>
    <parameter=ARG>
    value
    </parameter>
    </function>
    </tool_call>

Most of a short call is that scaffolding. Once the model has picked a tool,
the tags, the rest of a unique tool or parameter name, the header of the next
schema parameter and the closing tags follow from the schema, so a verify
window can absorb them in one forward instead of one token per round. The
values (paths, patterns, code) come from the copy-span proposer, which finds
them earlier in the conversation.

Every proposal is only a draft: the lane engine keeps a drafted token only
where it equals the target's own argmax, so the call is exactly the one the
model would have written token by token.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

_OPEN = "<tool_call>"
_END = "\x00end"  # placeholder: after a closed call the draft is the end-of-turn token
_SINGLE_LINE = re.compile(r"path|file|dir|name|pattern|glob|url|query|id|mode|lang|cwd|limit|offset",
                          re.IGNORECASE)


def tool_schema(tools: Sequence[dict[str, Any]] | None) -> dict[str, list[str]]:
    """Tool name -> its parameter names, required ones first, in schema order."""

    out: dict[str, list[str]] = {}
    for tool in tools or []:
        spec: dict[str, Any] = tool["function"] if isinstance(tool.get("function"), dict) else tool
        name = str(spec.get("name") or "")
        if not name:
            continue
        params: dict[str, Any] = spec.get("parameters") or spec.get("input_schema") or {}
        props = list((params.get("properties") or {}).keys())
        required = [p for p in (params.get("required") or []) if p in props]
        out[name] = required + [p for p in props if p not in required]
    return out


def _unique_completion(partial: str, options: Sequence[str]) -> str | None:
    matches = [o for o in options if o.startswith(partial)]
    if len(matches) == 1:
        return matches[0][len(partial):]
    return None


class ToolCallProposer:
    """Structure drafts for tool calls, copy-span drafts for everything else."""

    name = "tool-call"

    def __init__(self, tokenizer: Any, tools: Sequence[dict[str, Any]] | None, prompt_len: int,
                 fallback: Any = None, *, open_at_start: bool = True) -> None:
        self.tokenizer = tokenizer
        self.schema = tool_schema(tools)
        self.prompt_len = int(prompt_len)
        self.fallback = fallback
        self.open_at_start = bool(open_at_start)
        end = getattr(tokenizer, "convert_tokens_to_ids", None)
        end_id = end("<|im_end|>") if callable(end) else None
        self.end_ids = [int(end_id)] if isinstance(end_id, int) and end_id >= 0 else []
        self.end_text = bool(self.end_ids)
        self._decoded_upto = 0
        self._text = ""
        # the output's text is only needed from its last <tool_call> on (and whether it has any text):
        # decoding all of it every round cost 5.8 ms a round at 32k tokens (2026-09-23)
        open_id = end("<tool_call>") if callable(end) else None
        self._open_id = int(open_id) if isinstance(open_id, int) and open_id >= 0 else None
        self._last_open: int | None = None      # emitted index of the last <tool_call> token
        self._seen_text = False                 # any non-whitespace output yet
        self._last_structural = False
        self.structural_proposals = 0
        self.structural_tokens = 0
        self.structural_accepted = 0

    # -- the text written so far -------------------------------------------------
    def _output(self, context: Sequence[int]) -> str:
        """What ``structure`` needs of the output: '' while it is blank, text without a call opening while
        no call has opened, else the text from the last <tool_call> on (the same answers as the full text)."""

        n = len(context) - self.prompt_len
        if n == self._decoded_upto:
            return self._text
        if self._open_id is None or n < self._decoded_upto:
            emitted = context[self.prompt_len:]
            self._text = self.tokenizer.decode([int(t) for t in emitted])
            self._decoded_upto = n
            return self._text
        new = [int(t) for t in context[self.prompt_len + self._decoded_upto:self.prompt_len + n]]
        for i, token in enumerate(new):
            if token == self._open_id:
                self._last_open = self._decoded_upto + i
        if not self._seen_text and new and self.tokenizer.decode(new).strip():
            self._seen_text = True
        self._decoded_upto = n
        if self._last_open is None:
            self._text = " ." if self._seen_text else ""
        else:
            self._text = self.tokenizer.decode(
                [int(t) for t in context[self.prompt_len + self._last_open:self.prompt_len + n]])
        return self._text

    def _encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    # -- what must come next ---------------------------------------------------------
    def structure(self, text: str) -> str | None:
        """The text that must follow ``text`` inside a tool call, if it is determined."""

        if not text.strip():
            return f"{_OPEN}\n<function=" if self.open_at_start and self.schema else None
        start = text.rfind(_OPEN)
        if start < 0:
            return None
        call = text[start:]
        if call.rstrip("\n").endswith("</tool_call>") and call.count("</tool_call>") == 1:
            return _END if self.end_text else None
        if "</tool_call>" in call:
            return None
        if call.endswith("</function>"):
            return "\n</tool_call>"
        func = re.search(r"<function=([^>\n]*)(>?)", call)
        if func is None:
            return "\n<function=" if call.strip() == _OPEN else None
        name, closed = func.group(1), bool(func.group(2))
        params = self.schema.get(name) if closed else None
        if not closed:
            rest = _unique_completion(name, list(self.schema))
            if rest is None:
                return None
            first = self.schema[name + rest]
            return rest + ">\n" + (self._header(first[0]) if first else "</function>\n</tool_call>")
        if params is None:
            return None
        used = re.findall(r"<parameter=([^>\n]+)>", call)
        remaining = [p for p in params if p not in used]
        tail = call[func.end():]
        if not tail.strip():
            return "\n" + (self._header(remaining[0]) if remaining else "</function>\n</tool_call>")
        opened = re.search(r"<parameter=([^>\n]*)(>?)$", call)
        if opened and not opened.group(2):
            rest = _unique_completion(opened.group(1), remaining)
            return None if rest is None else rest + ">\n"
        if call.endswith("</parameter>"):
            return "\n" + (self._header(remaining[0]) if remaining else "</function>\n</tool_call>")
        last = re.search(r"<parameter=([^>\n]+)>\n([\s\S]*)$", call)
        if last and last.group(2).endswith("\n") and "</parameter>" not in last.group(2):
            value = last.group(2)
            if _SINGLE_LINE.search(last.group(1)) and value.count("\n") == 1 and value.strip():
                # a one-line value has ended: close it and open the next part
                return "</parameter>\n" + (self._header(remaining[0]) if remaining
                                             else "</function>\n</tool_call>")
        return None

    @staticmethod
    def _header(param: str) -> str:
        # the whole header: this tokenizer writes '=path' as one token, so a draft
        # that stops at '<parameter=' misses the model's next token
        return f"<parameter={param}>\n"

    # -- the proposer protocol the lane engine calls -----------------------------------
    @property
    def last_confident(self) -> bool:
        return self._last_structural or bool(getattr(self.fallback, "last_confident", False))

    def propose(self, context: Sequence[int], max_draft: int) -> list[int]:
        self._last_structural = False
        if self.fallback is not None:
            self.fallback.last_confident = False
        if max_draft <= 0:
            return []
        text = self._output(context)
        follow = self.structure(text)
        if follow:
            draft = (self.end_ids if follow == _END else self._encode(follow))[:max_draft]
            if draft:
                self._last_structural = True
                self.structural_proposals += 1
                self.structural_tokens += len(draft)
                return draft
        if self.fallback is None:
            return []
        return self.fallback.propose(context, max_draft)

    def observe(self, proposed: int, accepted: int) -> None:
        if self._last_structural:
            self.structural_accepted += int(accepted)
            return
        observe = getattr(self.fallback, "observe", None)
        if callable(observe):
            observe(proposed, accepted)

    # lane-engine hooks for a fallback that reads hidden states (DFlash2)
    @property
    def sink(self) -> Any:
        """The fallback's hidden-state feed (the relay drafter's), which the engine routes each forward to."""
        return getattr(self.fallback, "sink", None)

    def on_prefill(self, prompt_len: int) -> None:
        hook = getattr(self.fallback, "on_prefill", None)
        if callable(hook):
            hook(prompt_len)

    def on_round(self, row: int, keep: int) -> None:
        hook = getattr(self.fallback, "on_round", None)
        if callable(hook):
            hook(row, keep)

    def invalidate(self) -> None:
        hook = getattr(self.fallback, "invalidate", None)
        if callable(hook):
            hook()

    def on_rows(self, rows: Any) -> None:
        hook = getattr(self.fallback, "on_rows", None)
        if callable(hook):
            hook(rows)

    def capture_target(self, positions: Any, cand: Any, vals: Any) -> None:
        """The target's candidates behind committed tokens, for the drafter's capture (if it keeps one)."""

        forward = getattr(self.fallback, "capture_target", None)
        if callable(forward):
            forward(positions, cand, vals)

    def propose_tree(self, context: Any, max_nodes: int) -> tuple[list[int], list[int]]:
        """The call's known structure as a chain, else the fallback's tree."""

        self._last_structural = False
        if self.fallback is not None:
            self.fallback.last_confident = False
        if max_nodes <= 0:
            return [], []
        follow = self.structure(self._output(context))
        if follow:
            draft = (self.end_ids if follow == _END else self._encode(follow))[:max_nodes]
            if draft:
                self._last_structural = True
                self.structural_proposals += 1
                self.structural_tokens += len(draft)
                return draft, list(range(-1, len(draft) - 1))
        tree = getattr(self.fallback, "propose_tree", None)
        return tree(context, max_nodes) if callable(tree) else ([], [])

    def telemetry(self) -> dict[str, Any]:
        base = self.fallback.telemetry() if hasattr(self.fallback, "telemetry") else {}
        return {**base, "structural_proposals": self.structural_proposals,
                "structural_tokens": self.structural_tokens,
                "structural_accepted": self.structural_accepted}




class ToolCallStreamer:
    """The model's XML tool calls, as they are written, as OpenAI ``tool_calls`` deltas.

    Fed the whole decoded reply so far after every round, it emits the call's
    name as soon as ``<function=NAME>`` is complete, then the arguments object
    piece by piece: ``{"path":"`` when a parameter opens, the value's text as it
    grows (JSON-escaped; the last few characters held back in case they begin
    ``</parameter>``, trailing whitespace held until the value closes), ``"}``
    when the function closes. The concatenated argument deltas are the same
    JSON the end-of-reply parser builds (all strings; of a value's whitespace
    only the template's one framing newline a side is dropped, so the client's
    resent history is the tokens the model wrote), so a client can show a file
    being written instead of a silent minute.
    """

    _TAIL = "</parameter>"

    def __init__(self, tools: Sequence[dict[str, Any]] | None) -> None:
        self.known = {name.lower(): name for name in tool_schema(tools)}
        self.pos = 0
        self.state = "outside"
        self.index = -1
        self.args_open = False
        self.lead = False       # the framing newline after <parameter=NAME> has not arrived yet
        self.held = ""
        self.streamed = False

    def _args(self, fragment: str) -> dict[str, Any]:
        return {"tool_calls": [{"index": self.index, "function": {"arguments": fragment}}]}

    @staticmethod
    def _esc(text: str) -> str:
        import json

        return json.dumps(text, ensure_ascii=False)[1:-1]

    def feed(self, text: str) -> list[dict[str, Any]]:
        import uuid

        out: list[dict[str, Any]] = []
        while self.state != "abort":
            rest = text[self.pos:]
            if self.state == "outside":
                j = rest.find(_OPEN)
                if j < 0:
                    break
                self.pos += j + len(_OPEN)
                self.state = "call"
            elif self.state == "call":
                m = re.match(r"\s*<function=([^>\n]+)>\n?", rest)
                if m is None:
                    if len(rest) > 256 or (rest.strip() and not rest.strip().startswith("<")):
                        self.state = "abort"  # not a call we can follow: the end parser decides
                    break
                name = self.known.get(m.group(1).strip().lower())
                if name is None:
                    self.state = "abort"
                    break
                self.index += 1
                self.streamed = True
                out.append({"tool_calls": [{"index": self.index, "id": f"call_{uuid.uuid4().hex[:24]}",
                                            "type": "function",
                                            "function": {"name": name, "arguments": ""}}]})
                self.args_open = False
                self.pos += m.end()
                self.state = "function"
            elif self.state == "function":
                m = re.match(r"\s*<parameter=([^>\n]+)>\n?", rest)
                if m is not None:
                    out.append(self._args(("," if self.args_open else "{")
                                          + self._esc_key(m.group(1).strip()) + ':"'))
                    self.args_open = True
                    self.pos += m.end()
                    self.state = "value"
                    self.lead = not m.group(0).endswith("\n")
                    self.held = ""
                    continue
                m = re.match(r"\s*</function>", rest)
                if m is not None:
                    out.append(self._args("}" if self.args_open else "{}"))
                    self.pos += m.end()
                    self.state = "closing"
                    continue
                break
            elif self.state == "value":
                end = rest.find(self._TAIL)
                chunk = rest[:end] if end >= 0 else rest[:max(0, len(rest) - len(self._TAIL))]
                self.pos += len(chunk)
                piece = self.held + chunk
                if self.lead and piece:
                    piece = piece[1:] if piece.startswith("\n") else piece
                    self.lead = False
                if end >= 0:
                    keep = piece[:-1] if piece.endswith("\n") else piece   # the framing newline
                    self.held = ""
                else:
                    keep = piece.rstrip()   # its last newline may be the framing one
                    self.held = piece[len(keep):]
                if keep:
                    out.append(self._args(self._esc(keep)))
                if end < 0:
                    break
                out.append(self._args('"'))
                self.pos += len(self._TAIL)
                self.state = "function"
            elif self.state == "closing":
                m = re.match(r"\s*</tool_call>", rest)
                if m is None:
                    break
                self.pos += m.end()
                self.state = "outside"
        return out

    def _esc_key(self, key: str) -> str:
        import json

        return json.dumps(key, ensure_ascii=False)


__all__ = ["ToolCallProposer", "ToolCallStreamer", "tool_schema"]
