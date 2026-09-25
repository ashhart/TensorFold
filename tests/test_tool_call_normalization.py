"""The OpenAI wire format vs the Qwen3.6 chat template.

Clients send assistant tool calls with `function.arguments` as a JSON STRING
(that is the OpenAI spec). The template does

    {%- for args_name, args_value in tool_call.arguments|items %}

which needs a mapping, so a string raises TypeError inside Jinja and the
request returns NOTHING -- the client hangs on a spinner. It triggers on every
agentic turn whose history contains a prior tool call.
"""
from __future__ import annotations

from tensorfold.server.http import _normalize_tool_call_arguments


def _msg(args):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search_files", "arguments": args}}
        ],
    }


def _args(messages):
    return messages[0]["tool_calls"][0]["function"]["arguments"]


def test_json_string_arguments_become_a_dict():
    out = _normalize_tool_call_arguments([_msg('{"query": "retries"}')])
    assert _args(out) == {"query": "retries"}


def test_already_a_dict_is_left_alone():
    msgs = [_msg({"query": "retries"})]
    assert _normalize_tool_call_arguments(msgs) is msgs


def test_unparseable_arguments_are_not_mangled():
    """Never turn a bad payload into a crash of our own."""
    out = _normalize_tool_call_arguments([_msg("not json at all")])
    assert _args(out) == "not json at all"


def test_json_that_is_not_an_object_is_left_alone():
    # "[1,2]" parses, but the template needs a mapping, not a list.
    out = _normalize_tool_call_arguments([_msg("[1, 2]")])
    assert _args(out) == "[1, 2]"


def test_multiple_tool_calls_all_normalized():
    m = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "a", "arguments": '{"x": 1}'}},
            {"function": {"name": "b", "arguments": '{"y": 2}'}},
        ],
    }
    out = _normalize_tool_call_arguments([m])
    got = [c["function"]["arguments"] for c in out[0]["tool_calls"]]
    assert got == [{"x": 1}, {"y": 2}]


def test_messages_without_tool_calls_pass_through_untouched():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    assert _normalize_tool_call_arguments(msgs) is msgs


def test_empty_and_none_are_safe():
    assert _normalize_tool_call_arguments([]) == []


def test_original_messages_are_not_mutated():
    original = _msg('{"query": "x"}')
    msgs = [original]
    _normalize_tool_call_arguments(msgs)
    assert original["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'


def test_a_real_pi_shaped_history_renders():
    """The exact shape that hung: user -> assistant tool_call -> tool -> user."""
    msgs = [
        {"role": "user", "content": "take a look at this project."},
        _msg('{"query": "retry"}'),
        {"role": "tool", "tool_call_id": "c1", "content": "src/a.py"},
        {"role": "user", "content": "and now?"},
    ]
    out = _normalize_tool_call_arguments(msgs)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"query": "retry"}
    assert out[0] == msgs[0] and out[2] == msgs[2] and out[3] == msgs[3]
