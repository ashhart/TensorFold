"""GLM's tool-call format through both servers' parsers.

GLM-4.5 and later (GLM-5.3-Flash among them) write a call as the function name followed by key/value pairs,
``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>``, and their chat templates write a
string value as itself and any other value as JSON. The CUDA server used to leave such a call in the reply as text
(the client got no ``tool_calls``) and the HTTP server raised "unsupported tool_call payload format". The Qwen and
JSON bodies both servers already read are checked here too, unchanged.
"""
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
PARSERS = pytest.mark.parametrize("parser", [parse_tool_calls, parse_tool_calls_from_content],
                                  ids=["cuda-server", "http-server"])


def _parse(parser, text):
    content, calls = parser(text, TOOLS)
    return content, [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls or []]


@PARSERS
def test_glm_call_reads_non_string_arguments_as_json(parser):
    text = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "<arg_key>days</arg_key><arg_value>3</arg_value>"
            "<arg_key>hourly</arg_key><arg_value>true</arg_value></tool_call>")
    assert _parse(parser, text) == ("", [("get_weather", {"city": "Paris", "days": 3, "hourly": True})])


@PARSERS
def test_string_arguments_keep_their_exact_text(parser):
    # A string parameter stays text even when it looks like JSON, whitespace inside a value is kept (so a resent
    # history renders back to the tokens the model wrote), and an object parameter is read as JSON.
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
    text = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
    assert _parse(parser, text) == ("", [("get_weather", {"city": "Paris", "days": "3"})])


@PARSERS
def test_json_bodies_are_unchanged(parser):
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome", "days": 2}}</tool_call>'
    assert _parse(parser, text) == ("", [("get_weather", {"city": "Rome", "days": 2})])


def test_glm_block_parser_declines_other_formats():
    for block in ('{"name": "get_weather"}', "<function=get_weather>\n</function>", "get weather now"):
        assert parse_glm_tool_call_block(block, TOOLS) is None
