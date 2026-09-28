"""GLM-style tool calls (<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>) parsed into OpenAI
tool_calls, values decoded by their declared types as the template wrote them."""

from __future__ import annotations

import json

from tensorfold.server.tools import parse_tool_calls_from_content

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


def test_values_decode_as_the_template_wrote_them():
    """A value in its declared type's JSON form decodes; anything else stays the text the model wrote, so the next
    turn's history renders to the same tokens."""

    text = ("<tool_call>read<arg_key>path</arg_key><arg_value>\"quoted\"</arg_value>"
            "<arg_key>limit</arg_key><arg_value>2.0</arg_value><arg_key>all</arg_key><arg_value>True</arg_value>"
            "<arg_key>ratio</arg_key><arg_value>2</arg_value><arg_key>extra</arg_key><arg_value>{\"k\": [1]}"
            "</arg_value></tool_call>")
    _, calls = parse_tool_calls_from_content(text, TOOLS)
    assert args(calls[0]) == {"path": '"quoted"', "limit": "2.0", "all": "True", "ratio": 2, "extra": '{"k": [1]}'}


def test_a_streamed_reply_keeps_only_whole_glm_calls():
    """With a call limit (the streamed path), a GLM call with text between its arguments is dropped, not guessed."""

    text = ("<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
            "<tool_call>bash<arg_key>command</arg_key>junk<arg_value>rm</arg_value></tool_call>")
    _, calls = parse_tool_calls_from_content(text, TOOLS, max_calls=4)
    assert [args(c) for c in calls] == [{"command": "ls"}]


def test_a_call_to_an_unknown_tool_stays_text():
    text = "<tool_call>rm<arg_key>path</arg_key><arg_value>/</arg_value></tool_call>"
    content, calls = parse_tool_calls_from_content(text, TOOLS)
    assert calls is None and "<tool_call>rm" in content
