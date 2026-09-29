import json

import pytest

from tensorfold.cuda.reply_text import parse_tool_calls as parse_cuda_tool_calls
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server.http import parse_tool_calls_from_content


@pytest.mark.parametrize(
    "kind,value,expected",
    [
        (
            "array",
            '[{"question":"Pick a color","options":[{"label":"Blue"}]}]',
            [{"question": "Pick a color", "options": [{"label": "Blue"}]}],
        ),
        ("object", '{"x":[1]}', {"x": [1]}),
        ("boolean", "false", False),
        ("integer", "12", 12),
        ("number", "1.5", 1.5),
        ("string", "[1,2]", "[1,2]"),
        ("array", "invalid", "invalid"),
        ("array", '{"x":1}', '{"x":1}'),
    ],
)
@pytest.mark.parametrize("step", [1, 7, 10000])
def test_parameter_type_matches_schema_in_both_paths(kind, value, expected, step):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "question",
                "parameters": {"type": "object", "properties": {"questions": {"type": kind}}},
            },
        }
    ]
    text = f"<tool_call>\n<function=question>\n<parameter=questions>\n{value}\n</parameter>\n</function>\n</tool_call>"
    _, calls = parse_tool_calls_from_content(text, tools)
    assert json.loads(calls[0]["function"]["arguments"]) == {"questions": expected}
    for max_calls in (None, 1):                                       # the CUDA server's parser, the same types
        _, calls = parse_cuda_tool_calls(text, tools, max_calls=max_calls)
        assert json.loads(calls[0]["function"]["arguments"]) == {"questions": expected}
    streamer = ToolCallStreamer(tools)
    deltas = []
    for n in range(1, len(text) + 1, step):
        deltas += streamer.feed(text[:n])
    deltas += streamer.feed(text)
    args = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas)
    assert json.loads(args) == {"questions": expected}
