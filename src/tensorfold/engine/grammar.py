"""Structured output for every engine: a request's grammar as per-row token masks on each drafted verify window."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tensorfold.server.errors import RequestError

EXTRA = "tensorfold[grammar]"             # the optional dependency that brings xgrammar
KINDS = ("json", "json_schema", "regex", "choice", "grammar")
CACHE_BYTES = 256 << 20                   # compiled grammars kept for repeated schemas
OBJECT = '{"type": "object"}'             # json_object: any JSON object (OpenAI's contract), not an array
BLANKS = 32                               # the most blank characters between JSON tokens: pretty-printing, not endless


@dataclass(frozen=True)
class Spec:
    """A request's structured output: ``kind`` json, json_schema, regex, choice or grammar (EBNF), and its text."""

    kind: str
    text: str = ""
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


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RequestError(f"{where} must be a non-empty string")
    return value


def _choices(value: Any, where: str) -> str:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise RequestError(f"{where} must be a non-empty list of non-empty strings")
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
    guided = {"guided_json": ("json_schema", _schema_text), "guided_regex": ("regex", _text),
              "guided_choice": ("choice", _choices), "guided_grammar": ("grammar", _text)}
    for name, (kind, read) in guided.items():
        if body.get(name) is not None:
            return Spec(kind, read(body[name], name), name)
    so = body.get("structured_outputs")
    if so is not None:
        if not isinstance(so, dict):
            raise RequestError('structured_outputs must be an object such as {"json": {...}}')
        given = sorted(k for k, v in so.items() if v is not None and v is not False)
        known = {"json": ("json_schema", _schema_text), "regex": ("regex", _text), "choice": ("choice", _choices),
                 "grammar": ("grammar", _text)}
        other = [k for k in given if k not in known and k != "json_object"]
        if other:
            raise RequestError(f"structured_outputs {', '.join(other)} is not supported: use json, json_object, "
                               "regex, choice or grammar")
        for name in given:
            if name in known:
                kind, read = known[name]
                return Spec(kind, read(so[name], f"structured_outputs.{name}"), "structured_outputs")
        if so.get("json_object"):
            return Spec("json", field="structured_outputs")
    return None


def _message(exc: Exception) -> str:
    """xgrammar's error without its timestamp and source location."""

    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return re.sub(r"^\[[^\]]*\]\s*\S+:\d+:\s*", "", text)


class Grammars:
    """One tokenizer's grammar compiler (xgrammar), with compiled grammars cached by their text."""

    def __init__(self, info) -> None:
        import xgrammar as xgr

        self.xgr = xgr
        self.info = info
        self.vocab_size = int(info.vocab_size)
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
        return cls(xgr.TokenizerInfo.from_huggingface(tok, vocab_size=int(vocab_size), stop_token_ids=list(stop_ids)))

    def compile(self, spec: Spec):
        """The compiled grammar, or RequestError naming what the request's grammar gets wrong."""

        c = self.compiler
        try:
            with self.lock:
                if spec.kind in ("json", "json_schema"):
                    schema = OBJECT if spec.kind == "json" else spec.text
                    return c.compile_json_schema(schema, max_whitespace_cnt=BLANKS)
                if spec.kind == "regex":
                    return c.compile_regex(spec.text)
                if spec.kind == "choice":
                    options = " | ".join(json.dumps(v, ensure_ascii=False) for v in json.loads(spec.text))
                    return c.compile_grammar("root ::= " + options)
                return c.compile_grammar(spec.text)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise RequestError(f"{spec.field}: the grammar cannot be enforced: {_message(exc)}") from None

    def constraint(self, compiled, *, think_end: int | None = None, spec: Spec | None = None) -> "Constraint":
        """A fresh reply's grammar state; with ``think_end`` (thinking on) it starts at the token after that one."""

        c = Constraint(self.xgr, compiled, self.vocab_size, think_end=think_end)
        c.spec = spec
        return c

    def follow(self, values: Sequence[int]) -> "Constraint | None":
        """The constraint another rank ``pack``ed, compiled here: both ranks then walk and mask the same rows."""

        if not values:
            return None
        spec = Spec(KINDS[int(values[0])], bytes(int(v) for v in values[2:]).decode())
        return self.constraint(self.compile(spec), think_end=int(values[1]) - 1 if int(values[1]) else None, spec=spec)


def pack(constraint: "Constraint | None") -> list[int]:
    """A constraint's grammar for the other rank: [kind, think_end + 1 (0: none), the text's UTF-8 bytes], or []."""

    if constraint is None:
        return []
    spec, end = constraint.spec, constraint.think_end
    return [KINDS.index(spec.kind), 0 if end is None else int(end) + 1, *spec.text.encode()]


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


def compiler(owner, model_dir: str | Path, stop_ids: Sequence[int]) -> Grammars:
    """``owner``'s grammar compiler (kept on it as ``grammars``), built from its checkpoint on first use."""

    found = getattr(owner, "grammars", None)
    if found is None:
        vocab = vocab_size(model_dir) if model_dir is not None else None
        if vocab is None:
            raise RequestError("structured output needs the checkpoint's config.json vocab_size")
        found = owner.grammars = for_model(model_dir, vocab, tuple(stop_ids))
    return found


def request_constraint(owner, fields: dict[str, Any], think_end: int | None) -> "Constraint | None":
    """A fresh constraint for a request's grammar fields (``owner`` has ``model_dir`` and ``stop_ids``), or None."""

    spec = request_spec(fields)
    if spec is None:
        return None
    grammars = compiler(owner, owner.model_dir, owner.stop_ids)
    return grammars.constraint(grammars.compile(spec), think_end=think_end, spec=spec)


FIELDS = ("response_format", "guided_json", "guided_regex", "guided_choice", "guided_grammar", "structured_outputs")


def refusal(body: dict[str, Any], owner=None) -> str | None:
    """Why the body's grammar cannot run (HTTP 400): malformed, beside a required call, or (``owner``) not compiling."""

    from tensorfold.server.tools import tool_choice_requires_call

    try:
        spec = request_spec(body)
        if spec is not None and body.get("tools") and tool_choice_requires_call(body.get("tool_choice")):
            return f'{spec.field} cannot be combined with tool_choice "required" or a named function: send one'
        if spec is not None and getattr(owner, "model_dir", None) is not None:
            compiler(owner, owner.model_dir, owner.stop_ids).compile(spec)   # cached for the request's own compile
    except RequestError as exc:
        return str(exc)
    return None


@dataclass
class Window:
    """A verify window the grammar keeps: its rows, and the allowed-token bits of the rows it constrains."""

    tokens: list[int]
    parents: list[int]
    rows: list[int] = field(default_factory=list)     # constrained rows, in window order
    bits: Any = None                                  # [len(rows), words] int32 numpy (xgrammar's bitmask)


class Constraint:
    """One reply's grammar at its chosen tokens: keeps and masks verify rows, follows the chosen tokens."""

    def __init__(self, xgr, compiled, vocab_size: int, *, think_end: int | None = None) -> None:
        self.xgr = xgr
        self.m = xgr.GrammarMatcher(compiled)
        self.vocab = int(vocab_size)
        self.words = (self.vocab + 31) // 32
        self.think_end = think_end
        self.active = think_end is None               # with thinking on: from the token after </think>
        self.spec: Spec | None = None                 # what ``pack`` sends another rank
        self._shifts: dict[Any, Any] = {}

    @property
    def finished(self) -> bool:
        """The grammar has taken its stop token: the reply is complete."""

        return self.active and self.m.is_terminated()

    def window(self, tokens: Sequence[int], parents: Sequence[int]) -> Window:
        """The rows an accepted path can use (row 0 the pending token, parents first), each constrained row's bits."""

        try:
            return self._window(list(tokens), list(parents))
        except GrammarError:
            raise
        except Exception as exc:                      # noqa: BLE001  (xgrammar's failure ends this reply only)
            raise GrammarError(f"the reply's grammar failed: {_message(exc)}") from None

    def _window(self, tokens: list[int], parents: list[int]) -> Window:
        import numpy as np                            # xgrammar fills any DLPack array: one bitmask for torch and MLX

        if self.finished or (not self.active and self.think_end not in tokens[1:]):
            return Window(tokens, parents)            # the reply has ended, or no row reaches the grammar
        n = len(tokens)
        children: list[list[int]] = [[] for _ in range(n)]
        for r in range(1, n):
            children[parents[r]].append(r)
        bits = np.full((n, self.words), -1, dtype=np.int32)
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
                elif m.accept_token(tokens[c]):       # rejected: the parent's masked row could never choose it
                    if not m.is_terminated():         # a stop token ends the reply, nothing is verified after it
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

    def mask(self, logits, window: Window | None = None, offset: int = 0):
        """``logits`` (columns from token ``offset``), constrained rows' other tokens -inf: torch in place, MLX new."""

        if window is None:                            # one row: the token after the chosen ones
            window = self.window([0], [-1])
        if not window.rows:
            return logits
        if type(logits).__module__.startswith("mlx"):
            return _mask_mlx(logits, window, offset)
        return self._mask_torch(logits, window, offset)

    def allowed(self, window: Window, width: int, device, offset: int = 0):
        """The constrained rows' allowed columns [len(rows), width] (bool, on ``device``) from token ``offset`` on."""

        import torch

        shifts = self._shifts.get(device)
        if shifts is None:
            shifts = self._shifts[device] = torch.arange(8, dtype=torch.uint8, device=device)
        # token t is bit t % 8 of byte t // 8 (xgrammar's int32 words, little-endian)
        packed = torch.from_numpy(window.bits).to(device).view(torch.uint8)
        allowed = ((packed.unsqueeze(-1) >> shifts) & 1).view(len(window.rows), -1)[:, offset:offset + width].bool()
        if allowed.shape[1] < width:                  # logits past the grammar's vocabulary: never allowed
            allowed = torch.nn.functional.pad(allowed, (0, width - allowed.shape[1]), value=False)
        return allowed

    def _mask_torch(self, logits, window: Window, offset: int = 0):
        import torch

        dev, width = logits.device, logits.shape[-1]
        allowed = self.allowed(window, width, dev, offset)
        rows = logits.view(-1, width)
        if window.rows == list(range(rows.shape[0])):
            rows.masked_fill_(~allowed, float("-inf"))
        else:
            index = torch.tensor(window.rows, device=dev)
            rows[index] = rows[index].masked_fill(~allowed, float("-inf"))
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


def _mask_mlx(logits, window: Window, offset: int = 0):
    """The MLX form of ``Constraint.mask``: rows [R, V] (or [1, R, V]), the bits unpacked on the GPU."""

    import mlx.core as mx

    shape, width = logits.shape, logits.shape[-1]
    rows = logits.reshape(-1, width)
    words = mx.array(window.bits.view("uint32"))                          # [k, W]
    allowed = ((words[:, :, None] >> mx.arange(32, dtype=mx.uint32)) & 1).reshape(len(window.rows), -1)
    allowed = allowed[:, offset:offset + width]
    if allowed.shape[1] < width:                      # logits past the grammar's vocabulary: never allowed
        allowed = mx.pad(allowed, [(0, 0), (0, width - allowed.shape[1])])
    floor = mx.array(float("-inf"), dtype=rows.dtype)
    if window.rows == list(range(rows.shape[0])):
        rows = mx.where(allowed.astype(mx.bool_), rows, floor)
    else:
        index = mx.array(window.rows)
        rows[index] = mx.where(allowed.astype(mx.bool_), rows[index], floor)
    return rows.reshape(shape)


__all__ = ["Constraint", "EXTRA", "FIELDS", "GrammarError", "Grammars", "KINDS", "Spec", "Window", "compiler",
           "for_model", "pack", "refusal", "request_constraint", "request_spec", "vocab_size"]
