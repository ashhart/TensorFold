"""Structured output for the CUDA engines: a request's JSON schema as a token mask that keeps replies exact.

A request asks with OpenAI's ``response_format`` (``{"type": "json_schema", "json_schema": {"schema": ...}}`` or
``{"type": "json_object"}``), or with vLLM's ``guided_json`` / ``structured_outputs: {"json": ...}``.
``request_spec`` reads that from the body without xgrammar; ``Grammars`` compiles it (``pip install
'tensorfold[grammar]'``), and ``Constraint`` is one reply's grammar state, which an engine drives where it chooses
tokens:

- ``window(tokens, parents)`` before a verify forward. Row 0 is the pending token (chosen, its successor not yet),
  row r > 0 a draft under ``parents[r]``. The tree is walked depth first, the matcher accepting a row's token on the
  way down and rolling it back on the way up, and each row gets the tokens its path allows next. A draft the grammar
  rejects, a stop token, and the rows under either are dropped: the parent's masked row cannot choose the first, and
  the reply ends at the second, so no accepted path holds them.
- ``mask(logits, window)`` after the forward: each constrained row's disallowed tokens become -inf before sampling.
  A row's logits do not depend on the other rows, and its mask depends only on its path, so a drafted row and the
  serial step with the same path choose the same token, greedy or keyed.
- ``advance(tokens)``: the chosen tokens only. ``window`` leaves the matcher at the chosen tokens.

The grammar decides when a reply ends: the mask allows a stop token only once the JSON value is complete. With
thinking on, it applies from the token after ``</think>``.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tensorfold.server.errors import RequestError

EXTRA = "tensorfold[grammar]"             # the optional dependency that brings xgrammar
CACHE_BYTES = 256 << 20                   # compiled grammars kept for repeated schemas
OBJECT = '{"type": "object"}'             # json_object: any JSON object (OpenAI's contract), not an array


@dataclass(frozen=True)
class Spec:
    """A request's structured output: ``kind`` "json" (any JSON object) or "json_schema" (``schema`` as JSON text)."""

    kind: str
    schema: str = ""
    field: str = "response_format"        # the request field it came from, for error messages


class GrammarError(RuntimeError):
    """A reply's grammar failed: that request ends with this error, the server and other requests go on."""


def _schema_text(value: Any, where: str) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RequestError(f"{where} is not valid JSON: {exc.msg}") from None
    if isinstance(value, bool):
        value = {} if value else None
    if not isinstance(value, dict):
        raise RequestError(f"{where} must be a JSON schema object")
    return json.dumps(value, ensure_ascii=False)


def request_spec(body: dict[str, Any]) -> Spec | None:
    """The body's structured-output request, None for plain text; RequestError (HTTP 400) when it is malformed."""

    rf = body.get("response_format")
    if rf is not None:
        if not isinstance(rf, dict):
            raise RequestError('response_format must be an object such as {"type": "json_object"}')
        kind = rf.get("type")
        if kind == "json_object":
            return Spec("json")
        if kind == "json_schema":
            js = rf.get("json_schema")
            if not isinstance(js, dict) or js.get("schema") is None:
                raise RequestError("response_format json_schema needs json_schema.schema (a JSON schema object)")
            return Spec("json_schema", _schema_text(js["schema"], "response_format json_schema.schema"))
        if kind != "text":
            raise RequestError(f"response_format type must be text, json_object or json_schema, not {kind!r}")
    if body.get("guided_json") is not None:
        return Spec("json_schema", _schema_text(body["guided_json"], "guided_json"), "guided_json")
    for name in ("guided_regex", "guided_choice", "guided_grammar"):
        if body.get(name) is not None:
            raise RequestError(f"{name} is not supported: constrain replies with a JSON schema (response_format)")
    so = body.get("structured_outputs")
    if so is not None:
        if not isinstance(so, dict):
            raise RequestError('structured_outputs must be an object such as {"json": {...}}')
        other = sorted(k for k, v in so.items() if v is not None and k not in ("json", "json_object"))
        if other:
            raise RequestError(f"structured_outputs {', '.join(other)} is not supported: use structured_outputs.json")
        if so.get("json") is not None:
            return Spec("json_schema", _schema_text(so["json"], "structured_outputs.json"), "structured_outputs")
        if so.get("json_object"):
            return Spec("json", field="structured_outputs")
    return None


def _message(exc: Exception) -> str:
    """xgrammar's error without its timestamp and source location."""

    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return re.sub(r"^\[[^\]]*\]\s*\S+:\d+:\s*", "", text)


class Grammars:
    """One tokenizer's grammar compiler (xgrammar), with compiled grammars cached by schema."""

    def __init__(self, info, *, think_end: int | None = None) -> None:
        import xgrammar as xgr

        self.xgr = xgr
        self.info = info
        self.vocab_size = int(info.vocab_size)
        self.think_end = think_end
        self.compiler = xgr.GrammarCompiler(info, max_threads=8, cache_limit_bytes=CACHE_BYTES)
        self.lock = threading.Lock()              # requests compile on their own HTTP threads

    @classmethod
    def for_model(cls, model_dir: str | Path, vocab_size: int, stop_ids: Sequence[int]) -> "Grammars":
        """From the checkpoint's ``tokenizer.json``: ``vocab_size`` is the logits' width, ``stop_ids`` its eos ids."""

        try:
            import xgrammar as xgr
            from transformers import PreTrainedTokenizerFast       # xgrammar depends on transformers
        except ImportError:
            raise RequestError(f"structured output needs xgrammar on the server: pip install '{EXTRA}'") from None
        tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(model_dir) / "tokenizer.json"))
        info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=int(vocab_size), stop_token_ids=list(stop_ids))
        think_end = tok.convert_tokens_to_ids("</think>")
        return cls(info, think_end=think_end if isinstance(think_end, int) and think_end >= 0 else None)

    def compile(self, spec: Spec):
        """The compiled grammar, or RequestError naming what the schema gets wrong."""

        try:
            with self.lock:
                return self.compiler.compile_json_schema(OBJECT if spec.kind == "json" else spec.schema)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise RequestError(f"{spec.field}: the JSON schema cannot be enforced: {_message(exc)}") from None

    def constraint(self, compiled, *, after_think: bool = False) -> "Constraint":
        return Constraint(self.xgr, compiled, self.vocab_size, think_end=self.think_end if after_think else None)


_MODELS: dict[tuple[str, int, tuple[int, ...]], Grammars] = {}
_BUILD = threading.Lock()


def for_model(model_dir: str | Path, vocab_size: int, stop_ids: Sequence[int]) -> Grammars:
    """``Grammars.for_model``, built once per checkpoint (the first structured request pays a second or two)."""

    key = (str(Path(model_dir).resolve()), int(vocab_size), tuple(int(t) for t in stop_ids))
    with _BUILD:
        found = _MODELS.get(key)
        if found is None:
            found = _MODELS[key] = Grammars.for_model(model_dir, vocab_size, stop_ids)
        return found


def vocab_size(model_dir: str | Path) -> int | None:
    """The checkpoint's logits width from ``config.json`` (``text_config`` first), or None."""

    path = Path(model_dir) / "config.json"
    if not path.is_file():
        return None
    config = json.loads(path.read_text())
    for part in (config.get("text_config") or {}, config):
        if isinstance(part.get("vocab_size"), int):
            return int(part["vocab_size"])
    return None


@dataclass
class Window:
    """A verify window the grammar can keep: its rows, and the allowed-token bits of the rows it constrains."""

    tokens: list[int]
    parents: list[int]
    rows: list[int] = field(default_factory=list)     # constrained rows, in window order
    bits: Any = None                                  # [len(rows), words] int32 on the host (xgrammar's bitmask)


class Constraint:
    """One reply's grammar at its chosen tokens: keeps and masks verify rows, follows the chosen tokens."""

    def __init__(self, xgr, compiled, vocab_size: int, *, think_end: int | None = None) -> None:
        self.xgr = xgr
        self.m = xgr.GrammarMatcher(compiled)
        self.vocab = int(vocab_size)
        self.words = (self.vocab + 31) // 32
        self.think_end = think_end
        self.active = think_end is None               # with thinking on: from the token after </think>
        self._shifts: dict[Any, Any] = {}

    @property
    def finished(self) -> bool:
        """The grammar has taken its stop token: the reply is complete."""

        return self.active and self.m.is_terminated()

    def window(self, tokens: Sequence[int], parents: Sequence[int]) -> Window:
        """The rows an accepted path can use, parents first as given, and each constrained row's allowed tokens."""

        try:
            return self._window(list(tokens), list(parents))
        except GrammarError:
            raise
        except Exception as exc:                      # noqa: BLE001  (xgrammar's failure ends this reply only)
            raise GrammarError(f"the reply's grammar failed: {_message(exc)}") from None

    def _window(self, tokens: list[int], parents: list[int]) -> Window:
        import torch

        if self.finished or (not self.active and self.think_end not in tokens[1:]):
            return Window(tokens, parents)            # the reply has ended, or no row reaches the grammar
        n = len(tokens)
        children: list[list[int]] = [[] for _ in range(n)]
        for r in range(1, n):
            children[parents[r]].append(r)
        bits = torch.full((n, self.words), -1, dtype=torch.int32)
        kept = [True] + [False] * (n - 1)
        filled: list[int] = []
        m = self.m

        def visit(r: int, active: bool) -> None:      # the matcher has taken row r's path (when active)
            if active:
                m.fill_next_token_bitmask(bits, r)
                filled.append(r)
            for c in children[r]:
                if not active:
                    kept[c] = True
                    visit(c, tokens[c] == self.think_end)
                elif m.accept_token(tokens[c]):       # a token the grammar rejects: its parent's row cannot choose it
                    if not m.is_terminated():         # a stop token: the reply ends with it, nothing is verified after
                        kept[c] = True
                        visit(c, True)
                    m.rollback(1)

        visit(0, self.active)
        index = [r for r in range(n) if kept[r]]
        new = {r: i for i, r in enumerate(index)}
        window = Window([tokens[r] for r in index], [-1] + [new[parents[r]] for r in index[1:]])
        if filled:
            filled.sort()
            window.rows = [new[r] for r in filled]
            window.bits = bits if len(filled) == n else bits[filled]
        return window

    def mask(self, logits, window: Window | None = None):
        """``logits`` with each constrained row's disallowed tokens at -inf, in place (and returned).

        Without a window, ``logits`` is one row: the token after the chosen ones."""

        import torch

        if window is None:
            if logits.shape[0] != 1:
                raise ValueError("mask without a window takes one row")
            window = self.window([0], [-1])
        if not window.rows:
            return logits
        dev, width = logits.device, logits.shape[1]
        shifts = self._shifts.get(dev)
        if shifts is None:
            shifts = self._shifts[dev] = torch.arange(8, dtype=torch.uint8, device=dev)
        # token t is bit t % 8 of byte t // 8 (xgrammar's int32 words, little-endian)
        packed = window.bits.to(dev).view(torch.uint8)
        allowed = ((packed.unsqueeze(-1) >> shifts) & 1).view(len(window.rows), -1)[:, :width].bool()
        if allowed.shape[1] < width:                  # logits past the grammar's vocabulary: never allowed
            allowed = torch.nn.functional.pad(allowed, (0, width - allowed.shape[1]), value=False)
        if window.rows == list(range(logits.shape[0])):
            logits.masked_fill_(~allowed, float("-inf"))
        else:
            index = torch.tensor(window.rows, device=dev)
            logits[index] = logits[index].masked_fill(~allowed, float("-inf"))
        return logits

    def advance(self, tokens: Sequence[int]) -> None:
        """Follow chosen tokens (each chosen under this grammar's mask); after the stop token, nothing follows."""

        try:
            for t in tokens:
                t = int(t)
                if not self.active:
                    self.active = t == self.think_end
                    continue
                if self.m.is_terminated():
                    return
                if not self.m.accept_token(t):
                    raise GrammarError(f"the reply's grammar rejected chosen token {t}")
        except GrammarError:
            raise
        except Exception as exc:                      # noqa: BLE001
            raise GrammarError(f"the reply's grammar failed: {_message(exc)}") from None


__all__ = ["Constraint", "EXTRA", "GrammarError", "Grammars", "Spec", "Window", "for_model", "request_spec",
           "vocab_size"]
