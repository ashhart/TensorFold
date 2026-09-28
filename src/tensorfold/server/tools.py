"""Normalize tool specs and choices, then parse model tool calls into complete or streamed OpenAI responses."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from tensorfold.tool_parameters import decode_parameter, parameter_schemas

def tool_spec_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(tool.get("name") or "").strip() if isinstance(tool, dict) else ""


def normalize_tool_specs(tools: Any) -> list[dict[str, Any]]:
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    normalized: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{index}] must be an object")
        if not tool_spec_name(tool):
            raise ValueError(f"tools[{index}] must include a function name")
        normalized.append(tool)
    return normalized


def tool_choice_disables_tools(tool_choice: Any) -> bool:
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() == "none"
    if isinstance(tool_choice, dict):
        value = tool_choice.get("type") or tool_choice.get("mode")
        return isinstance(value, str) and value.strip().lower() == "none"
    return False


def validate_tool_choice(tools: list[dict[str, Any]], tool_choice: Any) -> None:
    if not isinstance(tool_choice, dict):
        return
    if str(tool_choice.get("type") or "").lower() != "function":
        return
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        raise ValueError("tool_choice function must include a function object")
    requested = str(function.get("name") or "").strip()
    if not requested:
        raise ValueError("tool_choice function must include a name")
    known = {tool_spec_name(tool) for tool in tools}
    if requested not in known:
        raise ValueError(f"tool_choice requested unknown tool '{requested}'")


def active_tool_specs(tools: Any, tool_choice: Any) -> list[dict[str, Any]]:
    specs = normalize_tool_specs(tools)
    if not specs or tool_choice_disables_tools(tool_choice):
        return []
    validate_tool_choice(specs, tool_choice)
    return specs


_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_NAMESPACED_TOOL_CALL_BLOCK_RE = re.compile(
    r"<([A-Za-z_][\w.-]*):tool_call>\s*(.*?)\s*</\1:tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_FUNCTION_BLOCK_RE = re.compile(
    r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$",
    re.IGNORECASE | re.DOTALL,
)
# Remove only one framing newline per side; preserve value whitespace so resent history matches generated tokens.
_TOOL_PARAMETER_BLOCK_RE = re.compile(
    r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>",
    re.IGNORECASE | re.DOTALL,
)
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_MISSING = object()


def _tool_json_object(value: Any) -> dict[str, Any]:
    if value is None:
        parsed: Any = {}
    elif isinstance(value, str):
        text = value.strip()
        parsed = json.loads(text) if text else {}
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ValueError("tool_call arguments must be a JSON object")
    return parsed


def _loose_tool_arguments(payload: dict[str, Any], explicit: Any) -> Any:
    if explicit is not _MISSING:
        return explicit
    return {
        key: value
        for key, value in payload.items()
        if key not in {"name", "tool", "function", "call", "type"}
    }


# GLM-4.5 and later (GLM-5.3-Flash): <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>
_GLM_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
_GLM_NAME_RE = re.compile(r"[A-Za-z_][\w.\-]*")


def _parse_glm_payload(block: str, schemas: dict[str, dict[str, Any]] | None, *,
                       complete: bool = False) -> tuple[str, dict[str, Any]] | None:
    """A GLM call (just the name when it has no arguments). Its template writes strings raw and other values as
    JSON, as Qwen's XML parameters, so values decode the same way and a resent history renders as written."""

    name, found, rest = block.partition("<arg_key>")
    name = name.strip()
    if _GLM_NAME_RE.fullmatch(name) is None:
        return None
    rest = found + rest
    if complete and _GLM_ARG_RE.sub("", rest).strip():
        return None
    schema = (schemas or {}).get(name.lower(), {})
    return name, {key.strip(): decode_parameter(value, schema.get(key.strip(), {}))
                  for key, value in _GLM_ARG_RE.findall(rest)}


def _parse_tool_call_payload(block: str, schemas: dict[str, dict[str, Any]] | None = None, *, complete: bool = False) -> tuple[str, dict[str, Any]] | None:
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        for item in payload:
            parsed = _parse_tool_call_payload(json.dumps(item, ensure_ascii=False), complete=complete)
            if parsed is not None:
                return parsed
        return None
    if isinstance(payload, dict):
        function = payload.get("function")
        if isinstance(function, dict):
            name = function.get("name") or function.get("tool") or function.get("function")
            explicit_arguments = function.get(
                "arguments",
                function.get("args", function.get("parameters", _MISSING)),
            )
            arguments = _loose_tool_arguments(function, explicit_arguments)
        else:
            name = (
                payload.get("name")
                or payload.get("tool")
                or payload.get("function")
                or payload.get("call")
            )
            explicit_arguments = payload.get(
                "arguments",
                payload.get("args", payload.get("parameters", _MISSING)),
            )
            arguments = _loose_tool_arguments(payload, explicit_arguments)
        name_text = str(name or "").strip()
        if not name_text:
            raise ValueError("tool_call is missing a function name")
        arguments = _tool_json_object(arguments)
        if complete:
            json.dumps(arguments, allow_nan=False)
        return name_text, arguments

    match = _TOOL_FUNCTION_BLOCK_RE.match(block)
    if match is None:
        if payload is None and not block.lstrip().startswith(("{", "[", "<")):
            return _parse_glm_payload(block, schemas, complete=complete)
        return None
    name = match.group(1).strip()
    arguments: dict[str, Any] = {}
    body = match.group(2)
    if complete and _TOOL_PARAMETER_BLOCK_RE.sub("", body).strip():
        return None
    for param_match in _TOOL_PARAMETER_BLOCK_RE.finditer(body):
        key = param_match.group(1).strip()
        schema = (schemas or {}).get(name.lower(), {}).get(key, {})
        arguments[key] = decode_parameter(param_match.group(2), schema)
    if not name:
        raise ValueError("tool_call is missing a function name")
    return name, arguments


def _strip_json_fence(text: str) -> str:
    match = _JSON_FENCE_RE.match(text)
    if match is None:
        return text
    return match.group(1).strip()


def _openai_tool_call(raw_name: str, arguments: dict[str, Any], known: dict[str, str]) -> dict[str, Any]:
    name = known.get(raw_name.lower())
    if name is None:
        raise ValueError(f"unknown tool '{raw_name}'")
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def _parse_bare_json_tool_calls(text: str, known: dict[str, str], *, max_calls: int | None = None) -> list[dict[str, Any]] | None:
    stripped = _strip_json_fence(text.strip())
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    payloads = payload if isinstance(payload, list) else [payload]
    calls: list[dict[str, Any]] = []
    for item in payloads:
        if max_calls is not None and len(calls) >= max_calls:
            break
        if not isinstance(item, dict):
            return None
        try:
            parsed = _parse_tool_call_payload(json.dumps(item, ensure_ascii=False), complete=max_calls is not None)
        except (ValueError, TypeError):
            if max_calls is None:
                raise
            continue
        if parsed is None:
            return None
        raw_name, arguments = parsed
        if raw_name.lower() not in known:
            return None
        calls.append(_openai_tool_call(raw_name, arguments, known))
    return calls or None


def parse_tool_calls_from_content(
    text: str,
    tools: list[dict[str, Any]],
    *, max_calls: int | None = None,
) -> tuple[str, list[dict[str, Any]] | None]:
    if not tools:
        return text, None
    known = {tool_spec_name(tool).lower(): tool_spec_name(tool) for tool in tools}
    schemas = parameter_schemas(tools)
    envelopes: list[tuple[int, int, str]] = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        envelopes.append((match.start(), match.end(), match.group(1).strip()))
    for match in _NAMESPACED_TOOL_CALL_BLOCK_RE.finditer(text):
        envelopes.append((match.start(), match.end(), match.group(2).strip()))
    if not envelopes:
        bare_calls = _parse_bare_json_tool_calls(text, known, max_calls=max_calls)
        if bare_calls is not None:
            return "", bare_calls
        return text, None
    envelopes.sort(key=lambda item: item[0])
    calls: list[dict[str, Any]] = []
    residue_parts: list[str] = []
    cursor = 0
    for index, (start, end, block) in enumerate(envelopes):
        residue_parts.append(text[cursor:start])
        cursor = end
        if max_calls is not None and len(calls) >= max_calls:
            continue
        try:
            parsed = _parse_tool_call_payload(block, schemas, complete=max_calls is not None)
        except (ValueError, TypeError):
            if max_calls is None:
                raise
            parsed = None
        if parsed is None:
            if max_calls is not None:
                continue
            raise ValueError("unsupported tool_call payload format")
        raw_name, arguments = parsed
        if raw_name.lower() not in known:
            # Keep unoffered tool calls as text to avoid ending the stream and triggering repeated client retries.
            if max_calls is None:
                residue_parts.append(text[start:end])
            continue
        calls.append(_openai_tool_call(raw_name, arguments, known))
        if index + 1 < len(envelopes) and envelopes[index + 1][0] < end:
            raise ValueError("overlapping tool_call blocks")
    residue_parts.append(text[cursor:])
    content = "".join(residue_parts).strip()
    return content, calls or None


def stream_tool_call_deltas(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict):
            continue
        deltas.append(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": str(tool_call.get("id") or f"call_{index}"),
                        "type": str(tool_call.get("type") or "function"),
                        "function": {
                            "name": str(function.get("name") or ""),
                            "arguments": "",
                        },
                    }
                ]
            }
        )
        arguments = str(function.get("arguments") or "")
        if arguments:
            deltas.append({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]})
    return deltas
