"""Validated usage and output fingerprints shared by the HTTP benchmark clients."""
import hashlib
import json
import re


def count(value):
    """A server-reported nonnegative integer; missing, boolean and coerced numbers stay unknown."""
    return value if type(value) is int and value >= 0 else None


def tool_pieces(delta):
    """A delta's tool-call fragments as (index, type, name, arguments); a missing index is the fragment's position."""
    calls = delta.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("invalid_tool_chunk")
    out = []
    for position, call in enumerate(calls):
        function = (call.get("function") or {}) if isinstance(call, dict) else None
        index = call.get("index", position) if isinstance(call, dict) else None
        if not isinstance(function, dict) or type(index) is not int or index < 0:
            raise ValueError("invalid_tool_chunk")
        parts = (call.get("type") or "", function.get("name") or "", function.get("arguments") or "")
        if not all(isinstance(part, str) for part in parts):
            raise ValueError("invalid_tool_chunk")
        out.append((index, *parts))
    return out


class Evidence:
    """Hash each choice's content, reasoning and tool calls apart so network chunk boundaries cannot alter them."""

    def __init__(self):
        self.choices = {}
        self.tools = {}
        self.observed_output = False

    def add(self, choice):
        """Take one choice delta; returns the output kinds it carried ("content", "reasoning", "tool")."""
        delta = choice.get("delta") or {}
        index = choice.get("index", 0)
        content = choice.get("text") or delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or ""
        if not isinstance(content, str) or not isinstance(reasoning, str) or type(index) is not int or index < 0:
            raise ValueError("invalid_text_chunk")
        calls = tool_pieces(delta)
        self.observed_output |= bool(content or reasoning)
        text, thought = self.choices.setdefault(index, (hashlib.sha256(), hashlib.sha256()))
        text.update(content.encode("utf-8"))
        thought.update(reasoning.encode("utf-8"))
        for at, kind, name, arguments in calls:
            parts = self.tools.setdefault((index, at), ["", [], []])
            parts[0] = parts[0] or kind
            parts[1].append(name)
            parts[2].append(arguments)
        tool = any(name or arguments for _, _, name, arguments in calls)
        return {kind for kind, seen in (("content", content), ("reasoning", reasoning), ("tool", tool)) if seen}

    def digest(self):
        """One choice: SHA-256 of its content and reasoning digests; several: of each choice's, in index order."""
        empty = (hashlib.sha256(), hashlib.sha256())
        parts = [self.choices[i] for i in sorted(self.choices)] or [empty]
        each = [hashlib.sha256(text.digest() + thought.digest()) for text, thought in parts]
        return each[0].hexdigest() if len(each) == 1 else hashlib.sha256(b"".join(d.digest() for d in each)).hexdigest()

    def tool_digest(self):
        """SHA-256 of the assembled calls (choice, index, type, name, arguments); server call ids are left out."""
        calls = [[*key, kind, "".join(name), "".join(arguments)] for key in sorted(self.tools)
                 for kind, name, arguments in [self.tools[key]]]
        return hashlib.sha256(json.dumps(calls, separators=(",", ":")).encode("utf-8")).hexdigest() if calls else None

    def finish(self, usage, runtime, complete):
        usage = usage if isinstance(usage, dict) else {}
        runtime = runtime if isinstance(runtime, dict) else {}
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, dict) else {}
        prompt = count(usage.get("prompt_tokens"))
        cached = count(details.get("cached_tokens"))
        tokens = count(usage.get("completion_tokens"))
        valid = prompt is not None and cached is not None and cached <= prompt
        state = "cold" if valid and cached == 0 else "reused" if valid else "unverified"
        if not complete:
            state = "unverified"
        token_sha = runtime.get("token_sha")
        if not isinstance(token_sha, str) or not re.fullmatch(r"(?:[0-9a-f]{12}|[0-9a-f]{64})", token_sha):
            token_sha = None
        token_sha256 = runtime.get("token_sha256")
        if not isinstance(token_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", token_sha256):
            token_sha256 = None
        if token_sha and token_sha256 and not token_sha256.startswith(token_sha[:12]):
            token_sha256 = None
        verified_output = complete and self.observed_output
        return {"prompt_tokens": prompt, "cached_tokens": cached, "tokens": tokens, "cache_state": state,
                "complete": complete, "token_sha": token_sha if verified_output else None,
                "token_sha256": token_sha256 if verified_output else None,
                "output_sha256": self.digest() if verified_output else None,
                "tool_calls": len(self.tools), "tool_sha256": self.tool_digest() if complete else None}


def token_equal(left, right):
    """Missing token hashes or failed/incomplete streams are unverified, never an equality pass."""
    if left.get("error") or right.get("error") or not left.get("complete") or not right.get("complete"):
        return None
    a, b = left.get("token_sha256"), right.get("token_sha256")
    if not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in (a, b)):
        return None
    return a == b
