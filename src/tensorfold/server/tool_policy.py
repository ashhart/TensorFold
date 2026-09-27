import re
from typing import Any

from tensorfold.server.errors import RequestError


class _ToolTextFilter:
    """Hide tool envelopes while retaining only a partial tag between prose deltas."""

    def __init__(self):
        self.pending = ""
        self.hidden = None

    def feed(self, text: str) -> str:
        self.pending += text
        out = []
        while self.pending:
            if self.hidden:
                close = f"</{self.hidden}>"
                end = self.pending.lower().find(close)
                if end < 0:
                    self.pending = self.pending[-(len(close) - 1):]
                    break
                self.pending = self.pending[end + len(close):]
                self.hidden = None
                continue
            start = self.pending.find("<")
            if start < 0:
                out.append(self.pending)
                self.pending = ""
                break
            out.append(self.pending[:start])
            self.pending = self.pending[start:]
            end = self.pending.find(">")
            if end < 0:
                name = self.pending[1:]
                prefix, separator, suffix = name.partition(":")
                possible = (not name or re.fullmatch(r"[A-Za-z_][\w.-]*", name)
                            or (separator and re.fullmatch(r"[A-Za-z_][\w.-]*", prefix)
                                and "tool_call".startswith(suffix.lower())))
                if possible:
                    break
                out.append("<")
                self.pending = self.pending[1:]
                continue
            name = self.pending[1:end]
            if re.fullmatch(r"(?:[A-Za-z_][\w.-]*:)?tool_call", name, flags=re.I):
                self.hidden = name.lower()
            else:
                out.append(self.pending[:end + 1])
            self.pending = self.pending[end + 1:]
        return "".join(out)

    def finish(self) -> str:
        tail = "" if self.hidden else self.pending
        self.pending = ""
        return tail


class ToolCallPolicy:
    def __init__(self, body: dict[str, Any]):
        parallel = body.get("parallel_tool_calls")
        parallel = True if parallel is None else parallel
        if type(parallel) is not bool:
            raise RequestError("parallel_tool_calls must be a boolean")
        self.single = not parallel
        self._text = _ToolTextFilter() if self.single else None

    @property
    def max_calls(self) -> int | None:
        return 1 if self.single else None

    def limit(self, calls):
        return calls[:1] if self.single and calls else calls

    def delta(self, delta):
        if self.single:
            if isinstance(delta, str):
                return self._text.feed(delta)
            if isinstance(delta, dict):
                result = {k: v for k, v in delta.items() if k != "tool_calls"}
                if isinstance(result.get("content"), str):
                    content = self._text.feed(result.pop("content"))
                    if content:
                        result["content"] = content
                return result
        return delta

    def flush(self) -> str:
        return self._text.finish() if self.single else ""

    def content(self, text: str, *, finished: bool = True) -> str:
        if self.single:
            filtered = _ToolTextFilter()
            return filtered.feed(text) + (filtered.finish() if finished else "")
        return text

    def finish(self, reply, tools, parse):
        if not tools:
            return reply
        content, calls = parse(str(reply.get("content") or ""), tools, max_calls=self.max_calls)
        calls = calls or reply.get("tool_calls")
        if not calls:
            return {**reply, "content": self.content(content)} if self.single else reply
        result = {**reply, "content": self.content(content), "tool_calls": self.limit(calls), "finish_reason": "tool_calls"}
        if self.single:
            result["tool_calls_streamed"] = False
        return result
