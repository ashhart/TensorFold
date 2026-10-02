"""The OpenAI wire format vs Qwen chat templates, including failed tool-call replay.

Clients send assistant tool calls with `function.arguments` as a JSON STRING
(that is the OpenAI spec). The template does

    {%- for args_name, args_value in tool_call.arguments|items %}

which needs a mapping. Even malformed arguments must render: an agent may replay
a failed tool call and its parse error so the model can retry it.
"""
from __future__ import annotations

import copy
import json

import pytest

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.server.messages import _normalize_tool_call_arguments
from tensorfold.server.responses_translate import messages as response_messages
from tensorfold.server.text import render_prompt_ids


# The Qwen3.8 tool-call block: unlike tojson-only templates, items requires a mapping.
QWEN_TOOL_TEMPLATE = """
{%- for message in messages %}
    {{- message.role + ':' + message.content }}
    {%- for tool_call in message.tool_calls %}
        {%- set tool_call = tool_call.function %}
        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
        {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
            {%- for args_name, args_value in tool_call.arguments|items %}
                {{- '<parameter=' + args_name + '>\n' }}
                {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                {{- args_value }}
                {{- '\n</parameter>\n' }}
            {%- endfor %}
        {%- endif %}
        {{- '</function>\n</tool_call>' }}
    {%- endfor %}
{%- endfor %}
"""


class TemplateTokenizer:
    """Render the same Jinja source through the Mac prompt path without a model."""

    def __init__(self, template):
        self.template = template

    def apply_chat_template(self, messages, **kwargs):
        return self.template.render(messages=messages, **kwargs)

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


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


@pytest.mark.parametrize("args", ['{"cmd":"git status"', "not json at all", "[1, 2]", "null", "true",
                                 "42", '"text"', "", [1, 2], None, True, 42])
def test_invalid_arguments_are_preserved_in_a_renderable_mapping(args):
    msgs = [_msg(args)]
    original = copy.deepcopy(msgs)
    out = _normalize_tool_call_arguments(msgs)
    assert _args(out) == {"_invalid_arguments": args}
    assert msgs == original
    assert out[0]["tool_calls"][0]["id"] == "c1"


def test_missing_arguments_are_left_alone():
    msgs = [_msg(None)]
    del msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert _normalize_tool_call_arguments(msgs) is msgs


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


@pytest.mark.parametrize("backend", ["cuda", "mlx"])
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("args", ['{"cmd":"git status"', "[1, 2]", "null", '{"query": "retry"}'])
def test_failed_tool_history_renders_and_keeps_the_tool_error(tmp_path, backend, api, args):
    pytest.importorskip("jinja2")
    error = "failed to parse function arguments: EOF while parsing an object"
    if api == "responses":
        msgs = response_messages([
            {"role": "user", "content": "Inspect the project."},
            {"type": "function_call", "call_id": "c1", "name": "search_files", "arguments": args},
            {"type": "function_call_output", "call_id": "c1", "output": error},
            {"role": "user", "content": "Retry with valid JSON."},
        ])
    else:
        msgs = [{"role": "user", "content": "Inspect the project."}, _msg(args),
                {"role": "tool", "tool_call_id": "c1", "content": error},
                {"role": "user", "content": "Retry with valid JSON."}]
    original = copy.deepcopy(msgs)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": QWEN_TOOL_TEMPLATE}))
    template = ChatTemplate(tmp_path)
    if backend == "cuda":
        rendered = template.render(msgs, tools=None, enable_thinking=False)
    else:
        tokenizer = TemplateTokenizer(template.template)
        rendered = tokenizer.decode(render_prompt_ids(tokenizer, msgs, late_system="system"))
    if args == '{"query": "retry"}':
        assert "<parameter=query>\nretry\n</parameter>" in rendered
        assert "_invalid_arguments" not in rendered
    else:
        assert f"<parameter=_invalid_arguments>\n{args}\n</parameter>" in rendered
    assert error in rendered and "Retry with valid JSON." in rendered
    assert msgs == original
