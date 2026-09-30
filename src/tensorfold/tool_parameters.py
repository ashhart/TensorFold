"""Schema-aware decoding of Qwen XML parameter text."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from typing import Any


def parameter_schemas(tools: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Each offered tool's parameter schemas by lowercase name; a spec that is not an object reads as untyped."""

    result = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool["function"] if isinstance(tool.get("function"), dict) else tool
        parameters = function.get("parameters") or function.get("input_schema") or {}
        properties = parameters.get("properties") or {} if isinstance(parameters, dict) else {}
        result[str(function.get("name", "")).lower()] = {
            name: schema for name, schema in properties.items() if isinstance(schema, dict)
        } if isinstance(properties, dict) else {}
    return result


def typed_parameter(schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    return isinstance(kind, str) and kind in {"array", "object", "boolean", "integer", "number", "null"}


def closed_json(text: str) -> str | None:
    """``text`` with the arrays and objects it left open closed, or None if it doesn't only stop short of them."""

    closers, in_string, escaped = [], False, False
    for ch in text:
        if in_string:
            escaped, in_string = (False, True) if escaped else (ch == "\\", ch != '"')
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            closers.append("]" if ch == "[" else "}")
        elif ch in "]}" and (not closers or closers.pop() != ch):
            return None
    return text.rstrip() + "".join(reversed(closers)) if closers and not in_string else None


_PY_WORDS = {"true": True, "false": False, "none": None, "null": None}


def _python_literal(text: str) -> Any:
    """``text`` as a Python literal (``True``, ``['a']``, ``{'k': None}``), tuples as lists; ValueError if not one."""

    word = _PY_WORDS.get(text.strip().lower(), ...)
    if word is not ...:
        return word

    def lists(v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return [lists(x) for x in v]
        if isinstance(v, dict):
            return {k: lists(x) for k, x in v.items()}
        return v

    try:
        return lists(ast.literal_eval(text.strip()))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError(str(exc)) from None


def decode_parameter(value: str, schema: dict[str, Any], *, python: bool = True) -> Any:
    """``value`` as the schema's type when it spells one, else the text; ``python`` also reads Python's spelling."""

    if not typed_parameter(schema):
        return value
    kind = schema["type"]
    valid = {
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: type(v) is int,
        "number": lambda v: type(v) in (int, float),
        "null": lambda v: v is None,
    }[kind]
    # a model can end an object or array value one closer short (#87): the value closed is what it meant
    for text in (value, closed_json(value) if kind in ("array", "object") else None):
        if text is None:
            continue
        try:
            parsed = json.loads(text)
            json.dumps(parsed, allow_nan=False)
        except (ValueError, TypeError):
            continue
        return parsed if valid(parsed) else value
    # not JSON: Python's spelling (True, None, single-quoted strings), as vLLM's Qwen parsers also accept
    if not python:
        return value
    try:
        parsed = _python_literal(value)
        json.dumps(parsed, allow_nan=False)
    except (ValueError, TypeError):
        return value
    return parsed if valid(parsed) else value
