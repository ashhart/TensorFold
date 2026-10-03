"""DeepSeek-V4.1's spaced DSML spelling through the shared tool-call parser and streamed-text helpers."""

from __future__ import annotations

import json

from tensorfold.server.text import hide_tool_calls, split_thinking
from tensorfold.server.tools import parse_tool_calls_from_content

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
         {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
BLOCK = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
         '<｜DSML｜ parameter name="location" string="true">Beijing, "north"</｜DSML｜ parameter>\n'
         '<｜DSML｜ parameter name="days" string="false">[1, 2]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n'
         '<｜DSML｜ invoke name="search">\n<｜DSML｜ parameter name="q" string="true">rain</｜DSML｜ parameter>\n'
         '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
REPLY = "Checking.\n\n" + BLOCK


def test_v41_block_parses_to_openai_calls():
    content, calls = parse_tool_calls_from_content(REPLY, TOOLS)
    assert content == "Checking."
    assert [c["function"]["name"] for c in calls] == ["get_weather", "search"]
    assert calls[0]["function"]["arguments"] == '{"location":"Beijing, \\"north\\"","days":[1,2]}'
    assert calls[1]["function"]["arguments"] == '{"q":"rain"}'
    assert all(c["type"] == "function" and c["id"].startswith("call_") for c in calls)


def test_v41_unknown_tool_stays_content():
    unknown = REPLY.replace('"search"', '"other"')
    assert parse_tool_calls_from_content(unknown, TOOLS) == (unknown.strip(), None)


def test_mixed_spellings_stay_text():
    v4_inside = BLOCK.replace("｜ invoke", "｜invoke").replace("｜ parameter", "｜parameter")
    v41_inside = BLOCK.replace("｜ calls>", "｜tool_calls>")
    closed_v4 = BLOCK.replace("</｜DSML｜ calls>", "</｜DSML｜tool_calls>")
    for block in (v4_inside, v41_inside, closed_v4):
        text = "Checking.\n\n" + block
        assert parse_tool_calls_from_content(text, TOOLS) == (text, None)


def test_v4_spelling_unchanged():
    v4 = REPLY.replace("｜ ", "｜").replace("｜calls>", "｜tool_calls>")
    content, calls = parse_tool_calls_from_content(v4, TOOLS)
    assert content == "Checking."
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [
        {"location": 'Beijing, "north"', "days": [1, 2]}, {"q": "rain"}]


def test_hide_tool_calls_holds_partial_v41_opener_and_hides_block():
    assert hide_tool_calls("Checking.\n\n<｜DSML｜ ca", finished=False) == "Checking.\n\n"
    assert hide_tool_calls(REPLY[:len("Checking.\n\n") + 40], finished=False) == "Checking.\n\n"
    assert hide_tool_calls(REPLY + "\nDone.", finished=True) == "Checking.\n\n\nDone."


def test_split_thinking_returns_block_written_before_think_closes():
    text = "Need the weather.\n\n" + BLOCK
    assert split_thinking(text, finished=True) == ("Need the weather.\n\n", BLOCK)
    assert split_thinking(text, finished=False) == ("Need the weather.\n\n", "")
