"""GLM-style tool calls (<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>) parsed into OpenAI
tool_calls, with values converted by their declared types the way oMLX converts them."""

from __future__ import annotations

import json

from tensorfold.server.http import coerce_glm_value, parse_tool_calls_from_content

TOOLS = [
    {"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "limit": {"type": "integer"}, "ratio": {"type": "number"},
        "all": {"type": "boolean"}, "lines": {"type": "array"}}}}},
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}},
]


def args(call):
    return json.loads(call["function"]["arguments"])


def test_glm_call_becomes_an_openai_tool_call():
    text = ("Let me look.\n<tool_call>read<arg_key>path</arg_key><arg_value>/etc/hostname</arg_value>"
            "<arg_key>limit</arg_key><arg_value>5</arg_value><arg_key>all</arg_key><arg_value>true</arg_value>"
            "<arg_key>lines</arg_key><arg_value>[1, 2]</arg_value><arg_key>ratio</arg_key><arg_value>0.5</arg_value>"
            "</tool_call>")
    content, calls = parse_tool_calls_from_content(text, TOOLS)
    assert content == "Let me look."
    assert calls[0]["function"]["name"] == "read"
    assert args(calls[0]) == {"path": "/etc/hostname", "limit": 5, "all": True, "lines": [1, 2], "ratio": 0.5}


def test_parallel_glm_calls_and_string_values_kept_verbatim():
    text = ("<tool_call>bash<arg_key>command</arg_key><arg_value>echo {\"a\": 1}\nls -la</arg_value></tool_call>"
            "<tool_call>now</tool_call>")
    content, calls = parse_tool_calls_from_content(text, TOOLS)
    assert content == ""
    assert [c["function"]["name"] for c in calls] == ["bash", "now"]
    assert args(calls[0]) == {"command": "echo {\"a\": 1}\nls -la"}
    assert args(calls[1]) == {}


def test_values_follow_the_declared_type():
    assert coerce_glm_value("42", {"type": "string"}) == "42"
    assert coerce_glm_value('"quoted"', {"type": "string"}) == "quoted"
    assert coerce_glm_value("42", {"type": "integer"}) == 42
    assert coerce_glm_value("2.0", {"type": "number"}) == 2
    assert coerce_glm_value("null", {"type": "integer"}) is None
    assert coerce_glm_value('{"k": [1]}', None) == {"k": [1]}     # undeclared: best-effort JSON
    assert coerce_glm_value("plain", None) == "plain"
    assert coerce_glm_value("{'k': 1}", {"type": "object"}) == {"k": 1}


def test_a_call_to_an_unknown_tool_stays_text():
    text = "<tool_call>rm<arg_key>path</arg_key><arg_value>/</arg_value></tool_call>"
    content, calls = parse_tool_calls_from_content(text, TOOLS)
    assert calls is None and "<tool_call>rm" in content
