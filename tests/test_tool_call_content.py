"""A reply that is not a tool call reaches the client as content (issue #51): never an HTTP 500."""

import json

import pytest

from http_fakes import post
from tensorfold.server.tools import parse_tool_calls_from_content

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"location": {"type": "string"}}}}}]
ANSWER = '{"temp": 7, "condition": "Overcast", "humidity": 82}'


@pytest.mark.parametrize("max_calls", [None, 1])
@pytest.mark.parametrize("text", [ANSWER, '[{"temp": 7}]', '[{"name": "get_weather", "arguments": {}}, {"temp": 7}]',
                                  '```json\n{"temp": 7}\n```', '{"name": "get_weather", "arguments": "oops"}'])
def test_json_that_is_not_a_call_is_content(text, max_calls) -> None:
    if max_calls == 1 and text.startswith('[{"name"'):
        pytest.skip("one call a reply takes the leading call, as before")
    assert parse_tool_calls_from_content(text, TOOLS, max_calls=max_calls) == (text, None)


@pytest.mark.parametrize("block", ['<tool_call>{"temp": 7}</tool_call>', "<tool_call>not a call</tool_call>",
                                   '<tool_call>{"name": "get_weather", "arguments": [1]}</tool_call>',
                                   '<a:tool_call><tool_call>{"name": "get_weather"}</tool_call></a:tool_call>',
                                   "<|tool_call>call:get_weather{location}<tool_call|>"])
def test_a_malformed_block_stays_text_beside_real_calls(block) -> None:
    call = "<tool_call>\n<function=get_weather>\n<parameter=location>\nOslo\n</parameter>\n</function>\n</tool_call>"
    content, calls = parse_tool_calls_from_content(f"Checking.\n{block}\n{call}", TOOLS)
    assert content == f"Checking.\n{block}"
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == [
        ("get_weather", {"location": "Oslo"})]


class _App:
    served_name = "fake"
    model_ids = ["fake"]
    max_batch_size = 1
    exact_mode = {"mode": "exact"}

    def chat(self, messages, **kwargs):
        on_delta = kwargs.get("on_delta")
        if on_delta is not None:
            on_delta(ANSWER)
        return {"content": ANSWER, "finish_reason": "stop", "prompt_tokens": 3, "cached_tokens": 0,
                "completion_tokens": 20}


@pytest.mark.parametrize("stream", [False, True])
def test_a_json_answer_with_tools_is_a_200_with_content(stream) -> None:
    status, body = post(_App(), {"messages": [{"role": "user", "content": "Weather as JSON only"}],
                                 "tools": TOOLS, "stream": stream})
    assert status == 200
    if stream:
        assert '"error"' not in body and json.dumps(ANSWER)[1:-1] in body and '"finish_reason": "stop"' in body
    else:
        choice = json.loads(body)["choices"][0]
        assert choice["message"]["content"] == ANSWER and "tool_calls" not in choice["message"]
        assert choice["finish_reason"] == "stop"
