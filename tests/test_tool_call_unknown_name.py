"""A well-formed call to a tool the request did not offer is text by default and a tool call with TENSORFOLD_PASS_UNKNOWN_TOOLS.

The Mac lane server and the CUDA server each parse a reply's tool calls, so both are held to it.
"""

import json

import pytest

from tensorfold.server.tools import PASS_UNKNOWN_TOOLS_ENV
from tensorfold.server.tools import parse_tool_calls_from_content as _server_parse

try:
    from tensorfold.cuda.reply_text import parse_tool_calls as _cuda_parse
except ImportError:  # the CUDA lane needs torch
    _cuda_parse = None

PARSERS = [pytest.param(_server_parse, id="server")]
if _cuda_parse is not None:
    PARSERS.append(pytest.param(_cuda_parse, id="cuda"))

TOOLS = [
    {"type": "function", "function": {"name": "job_kill", "parameters": {
        "type": "object", "properties": {"id": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"}}}}},
]
UNKNOWN = "<tool_call>\n<function=kill>\n<parameter=id>\n42\n</parameter>\n</function>\n</tool_call>"
UNKNOWN_OPEN = "<tool_call>\n<function=kill>\n<parameter=bash>\nkill 42\n</function>\n</tool_call>"
KNOWN = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"


@pytest.fixture(params=PARSERS)
def parse(request):
    return request.param


def _calls(parse, text, **kwargs):
    content, calls = parse(text, TOOLS, **kwargs)
    return content, [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls or []]


@pytest.mark.parametrize("text", [UNKNOWN, UNKNOWN_OPEN])
def test_an_unoffered_call_stays_text_by_default(parse, text, monkeypatch) -> None:
    monkeypatch.delenv(PASS_UNKNOWN_TOOLS_ENV, raising=False)
    assert parse(text, TOOLS) == (text, None)


def test_an_unoffered_call_reaches_the_client_when_enabled(parse, monkeypatch) -> None:
    monkeypatch.setenv(PASS_UNKNOWN_TOOLS_ENV, "1")
    assert _calls(parse, f"Stopping it.\n{UNKNOWN}") == ("Stopping it.", [("kill", {"id": "42"})])


def test_an_unoffered_call_with_an_unclosed_parameter_reaches_the_client_when_enabled(parse, monkeypatch) -> None:
    monkeypatch.setenv(PASS_UNKNOWN_TOOLS_ENV, "1")
    content, calls = _calls(parse, UNKNOWN_OPEN)
    assert content == "" and [name for name, _ in calls] == ["kill"]


def test_offered_names_keep_their_spelling_and_order_when_enabled(parse, monkeypatch) -> None:
    monkeypatch.setenv(PASS_UNKNOWN_TOOLS_ENV, "1")
    content, calls = _calls(parse, f"{UNKNOWN}\n{KNOWN.replace('<function=bash>', '<function=BASH>')}")
    assert content == "" and calls == [("kill", {"id": "42"}), ("bash", {"command": "ls"})]


def test_one_call_a_reply_takes_the_first_when_enabled(parse, monkeypatch) -> None:
    monkeypatch.setenv(PASS_UNKNOWN_TOOLS_ENV, "1")
    _, calls = _calls(parse, f"{UNKNOWN}\n{KNOWN}", max_calls=1)
    assert calls == [("kill", {"id": "42"})]


@pytest.mark.parametrize("block", ['<tool_call>{"temp": 7}</tool_call>', "<tool_call>not a call</tool_call>",
                                   '<tool_call>{"name": "kill", "arguments": [1]}</tool_call>'])
def test_a_malformed_block_stays_text_when_enabled(parse, block, monkeypatch) -> None:
    monkeypatch.setenv(PASS_UNKNOWN_TOOLS_ENV, "1")
    assert parse(block, TOOLS) == (block, None)
