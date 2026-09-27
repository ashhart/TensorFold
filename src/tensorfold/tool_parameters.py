"""Schema-aware decoding of Qwen XML parameter text."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def parameter_schemas(tools: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    result = {}
    for tool in tools or []:
        function = tool.get("function", tool)
        result[str(function.get("name", "")).lower()] = function.get("parameters", {}).get("properties", {})
    return result


def typed_parameter(schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    return isinstance(kind, str) and kind in {"array", "object", "boolean", "integer", "number", "null"}


def decode_parameter(value: str, schema: dict[str, Any]) -> Any:
    if not typed_parameter(schema):
        return value
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    kind = schema["type"]
    valid = {
        "array": isinstance(parsed, list),
        "object": isinstance(parsed, dict),
        "boolean": isinstance(parsed, bool),
        "integer": type(parsed) is int,
        "number": type(parsed) in (int, float),
        "null": parsed is None,
    }
    return parsed if valid[kind] else value
