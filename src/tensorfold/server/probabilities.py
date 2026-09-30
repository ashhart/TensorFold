"""Validate probability requests before generation or stream headers."""

import json
from typing import Any

from tensorfold.server.errors import RequestError


def probability_options(body: dict[str, Any], *, supported: bool = False) -> int | None:
    count = body.get("n")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count != 1):
        raise RequestError("n must be 1; multiple choices are not supported")
    enabled = body.get("logprobs")
    if enabled is not None and not isinstance(enabled, bool):
        raise RequestError("logprobs must be a boolean or null")
    top = body.get("top_logprobs")
    if top is not None and (isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= 20):
        raise RequestError("top_logprobs must be an integer between 0 and 20, or null")
    if top is not None and not enabled:
        raise RequestError("top_logprobs requires logprobs: true")
    if not enabled:
        return None
    if not supported:
        raise RequestError("logprobs are not supported by this model or backend")
    return top or 0


class TokenBytes:
    """ByteLevel token bytes remain intact when a token contains only part of UTF-8."""

    def __init__(self, tokenizer):
        try:
            decoder = json.loads(tokenizer.decoder.__getstate__())
        except (AttributeError, TypeError, ValueError):
            decoder = {}
        if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
            raise RequestError("logprobs require a ByteLevel tokenizer")
        present = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        chars = list(present)
        missing = [b for b in range(256) if b not in present]
        self.decode = dict(zip(map(chr, chars + list(range(256, 256 + len(missing)))), present + missing))
        self.tokenizer = tokenizer
        self.added = tokenizer.get_added_tokens_decoder()
        self.cache = {}

    def token(self, token_id):
        if token_id not in self.cache:
            added = self.added.get(token_id)
            piece = self.tokenizer.id_to_token(token_id)
            if piece is None:
                value = {"token": "", "bytes": None}
            else:
                raw = b"".join(bytes([self.decode[c]]) if c in self.decode else c.encode("utf-8") for c in piece)
                value = {"token": raw.decode("utf-8", errors="replace"),
                         "bytes": None if added is not None and added.special else list(raw)}
            self.cache[token_id] = value
        return self.cache[token_id]

    def format(self, rows, ends):
        return {"content": [{**self.token(row["id"]), "logprob": row["logprob"],
                             "top_logprobs": [{**self.token(token), "logprob": value}
                                             for token, value in row["top"]]}
                            for row in rows if row["id"] not in ends]}
