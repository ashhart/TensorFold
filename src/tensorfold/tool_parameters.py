"""Schema-aware decoding of Qwen XML parameter text."""

from __future__ import annotations

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


def decode_parameter(value: str, schema: dict[str, Any]) -> Any:
    if not typed_parameter(schema):
        return value
    kind = schema["type"]
    # a model can end an object or array value one closer short (#87): the value closed is what it meant
    for text in (value, closed_json(value) if kind in ("array", "object") else None):
        if text is None:
            continue
        try:
            parsed = json.loads(text)
            json.dumps(parsed, allow_nan=False)
        except (ValueError, TypeError):
            continue
        valid = {
            "array": isinstance(parsed, list),
            "object": isinstance(parsed, dict),
            "boolean": isinstance(parsed, bool),
            "integer": type(parsed) is int,
            "number": type(parsed) in (int, float),
            "null": parsed is None,
        }
        return parsed if valid[kind] else value
    return value
