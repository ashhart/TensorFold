"""Validated usage and output fingerprints shared by the HTTP benchmark clients."""
import hashlib
import re


def count(value):
    """A server-reported nonnegative integer; missing, boolean and coerced numbers stay unknown."""
    return value if type(value) is int and value >= 0 else None


class Evidence:
    """Hash content and reasoning independently so network chunk boundaries cannot alter the fingerprint."""

    def __init__(self):
        self.content = hashlib.sha256()
        self.reasoning = hashlib.sha256()

    def add(self, choice):
        delta = choice.get("delta") or {}
        content = choice.get("text") or delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or ""
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise ValueError("invalid_text_chunk")
        self.content.update(content.encode("utf-8"))
        self.reasoning.update(reasoning.encode("utf-8"))

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
        digest = hashlib.sha256(self.content.digest() + self.reasoning.digest()).hexdigest()
        return {"prompt_tokens": prompt, "cached_tokens": cached, "tokens": tokens, "cache_state": state,
                "complete": complete, "token_sha": token_sha if complete else None,
                "output_sha256": digest if complete else None}


def token_equal(left, right):
    """Missing token hashes or failed/incomplete streams are unverified, never an equality pass."""
    if left.get("error") or right.get("error") or not left.get("complete") or not right.get("complete"):
        return None
    if not left.get("token_sha") or not right.get("token_sha"):
        return None
    return left["token_sha"] == right["token_sha"]
