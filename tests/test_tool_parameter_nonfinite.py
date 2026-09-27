import json

import pytest

from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server.http import parse_tool_calls_from_content


@pytest.mark.parametrize("kind,value", [
    ("number", "NaN"), ("number", "Infinity"), ("number", "-Infinity"),
    ("number", "1e309"), ("array", "[NaN]"), ("array", "[1e309]"),
    ("object", '{"nested":[{"value":Infinity}]}'),
])
@pytest.mark.parametrize("stream", [False, True])
def test_nonfinite_tool_values_remain_strings(kind, value, stream):
    tools = [{"function": {"name": "measure", "parameters": {
        "properties": {"value": {"type": kind}}}}}]
    text = f"<tool_call><function=measure><parameter=value>{value}</parameter></function></tool_call>"
    if stream:
        parser = ToolCallStreamer(tools)
        deltas = []
        for stop in range(1, len(text) + 1):
            deltas.extend(parser.feed(text[:stop]))
        arguments = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas)
    else:
        _, calls = parse_tool_calls_from_content(text, tools)
        arguments = calls[0]["function"]["arguments"]
    def reject_nonfinite(value):
        raise ValueError(value)
    assert json.loads(arguments, parse_constant=reject_nonfinite) == {"value": value}
