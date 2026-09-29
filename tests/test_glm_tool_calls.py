"""GLM's <arg_key>/<arg_value> tool calls, and the Qwen and JSON ones, through both servers' parsers."""
from __future__ import annotations

import json

import pytest

from tensorfold.cuda.server import parse_tool_calls
from tensorfold.server.tools import parse_glm_tool_call_block, parse_tool_calls_from_content

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "hourly": {"type": "boolean"},
        "units": {"type": ["string", "null"]}, "zip": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}, "meta": {"type": "object"}}}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
]
AGENT_TOOLS = [
    {"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "limit": {"type": "integer"}, "ratio": {"type": "number"},
        "all": {"type": "boolean"}, "lines": {"type": "array"}}}}},
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}},
]
PARSERS = pytest.mark.parametrize("parser", [parse_tool_calls, parse_tool_calls_from_content],
                                  ids=["cuda-server", "http-server"])


def _parse(parser, text, tools=TOOLS, **kw):
    content, calls = parser(text, tools, **kw)
    return content, [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls or []]


@PARSERS
def test_glm_call_reads_non_string_arguments_as_json(parser):
    text = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "<arg_key>days</arg_key><arg_value>3</arg_value>"
            "<arg_key>hourly</arg_key><arg_value>true</arg_value></tool_call>")
    assert _parse(parser, text) == ("", [("get_weather", {"city": "Paris", "days": 3, "hourly": True})])


@PARSERS
def test_string_arguments_keep_their_exact_text(parser):
    # a string value keeps its exact text even when it looks like JSON; an object parameter is read as JSON
    text = ("<tool_call>write_file\n<arg_key>path</arg_key>\n<arg_value>notes.txt</arg_value>\n"
            "<arg_key>content</arg_key>\n<arg_value>42\n  indented\n</arg_value>\n"
            "<arg_key>meta</arg_key>\n<arg_value>{\"mode\": \"append\"}</arg_value>\n</tool_call>")
    assert _parse(parser, text)[1] == [
        ("write_file", {"path": "notes.txt", "content": "42\n  indented\n", "meta": {"mode": "append"}})]


@PARSERS
def test_string_typed_values_that_look_like_json_stay_text(parser):
    text = ("<tool_call>get_weather<arg_key>units</arg_key><arg_value>null</arg_value>"
            "<arg_key>zip</arg_key><arg_value>02134</arg_value></tool_call>")
    assert _parse(parser, text)[1] == [("get_weather", {"units": "null", "zip": "02134"})]


@PARSERS
def test_a_non_string_value_that_is_not_json_stays_text(parser):
    text = "<tool_call>get_weather<arg_key>days</arg_key><arg_value>three</arg_value></tool_call>"
    assert _parse(parser, text)[1] == [("get_weather", {"days": "three"})]


@PARSERS
def test_call_without_arguments_and_several_calls_in_one_reply(parser):
    text = ("Checking both.\n<tool_call>get_time</tool_call>\n"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>")
    assert _parse(parser, text) == ("Checking both.", [("get_time", {}), ("get_weather", {"city": "Oslo"})])


@PARSERS
def test_a_tool_the_client_did_not_offer_stays_in_the_reply(parser):
    text = "<tool_call>launch_rocket<arg_key>when</arg_key><arg_value>now</arg_value></tool_call>"
    assert _parse(parser, text) == (text, [])


@PARSERS
def test_qwen_function_blocks_are_unchanged(parser):
    # both servers type Qwen parameters by the tool's schema
    text = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
    assert _parse(parser, text) == ("", [("get_weather", {"city": "Paris", "days": 3})])


@PARSERS
def test_json_bodies_are_unchanged(parser):
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome", "days": 2}}</tool_call>'
    assert _parse(parser, text) == ("", [("get_weather", {"city": "Rome", "days": 2})])


def test_glm_block_parser_declines_other_formats():
    for block in ('{"name": "get_weather"}', "<function=get_weather>\n</function>", "get weather now"):
        assert parse_glm_tool_call_block(block, TOOLS) is None


@PARSERS
def test_glm_call_becomes_an_openai_tool_call(parser):
    text = ("Let me look.\n<tool_call>read<arg_key>path</arg_key><arg_value>/etc/hostname</arg_value>"
            "<arg_key>limit</arg_key><arg_value>5</arg_value><arg_key>all</arg_key><arg_value>true</arg_value>"
            "<arg_key>lines</arg_key><arg_value>[1, 2]</arg_value><arg_key>ratio</arg_key><arg_value>0.5</arg_value>"
            "</tool_call>")
    assert _parse(parser, text, AGENT_TOOLS) == (
        "Let me look.", [("read", {"path": "/etc/hostname", "limit": 5, "all": True, "lines": [1, 2], "ratio": 0.5})])


@PARSERS
def test_parallel_glm_calls_and_string_values_kept_verbatim(parser):
    text = ("<tool_call>bash<arg_key>command</arg_key><arg_value>echo {\"a\": 1}\nls -la</arg_value></tool_call>"
            "<tool_call>now</tool_call>")
    assert _parse(parser, text, AGENT_TOOLS) == ("", [("bash", {"command": "echo {\"a\": 1}\nls -la"}), ("now", {})])


@PARSERS
def test_values_decode_as_the_template_wrote_them(parser):
    """A value in its declared type's JSON form decodes; others stay as written, so a resent history matches."""

    text = ("<tool_call>read<arg_key>path</arg_key><arg_value>\"quoted\"</arg_value>"
            "<arg_key>limit</arg_key><arg_value>2.0</arg_value><arg_key>all</arg_key><arg_value>True</arg_value>"
            "<arg_key>ratio</arg_key><arg_value>2</arg_value><arg_key>extra</arg_key><arg_value>{\"k\": [1]}"
            "</arg_value></tool_call>")
    assert _parse(parser, text, AGENT_TOOLS)[1] == [
        ("read", {"path": '"quoted"', "limit": "2.0", "all": "True", "ratio": 2, "extra": '{"k": [1]}'})]


@PARSERS
def test_a_limited_reply_keeps_only_whole_glm_calls(parser):
    """With a call limit (the streamed path), a GLM call with text between its arguments is dropped, not guessed."""

    text = ("<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
            "<tool_call>bash<arg_key>command</arg_key>junk<arg_value>rm</arg_value></tool_call>")
    assert _parse(parser, text, AGENT_TOOLS, max_calls=4)[1] == [("bash", {"command": "ls"})]


@PARSERS
def test_a_tool_spec_without_an_object_schema_reads_values_as_text(parser):
    tools = [{"type": "function", "function": {"name": "ping", "parameters": None}},
             {"type": "function", "function": {"name": "pong", "parameters": {"properties": {"n": True}}}}]
    text = ("<tool_call>ping<arg_key>n</arg_key><arg_value>3</arg_value></tool_call>"
            "<tool_call>pong<arg_key>n</arg_key><arg_value>4</arg_value></tool_call>")
    assert _parse(parser, text, tools)[1] == [("ping", {"n": "3"}), ("pong", {"n": "4"})]
