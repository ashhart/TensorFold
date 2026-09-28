"""DeepSeek-V4-Flash prompts through DeepSeek's encoder, and its DSML tool calls through the server's parser."""

from __future__ import annotations

import json
from pathlib import Path

from tensorfold.families.deepseek_v4.prompts import DeepSeekTokenizer, render
from tensorfold.server.text import hide_tool_calls
from tensorfold.server.tools import parse_tool_calls_from_content

FIXTURES = Path(__file__).parent / "fixtures" / "deepseek_v4"


def test_renders_deepseeks_golden_prompts():
    """DeepSeek's own test cases: thinking with tools and a tool result, and thinking over two turns."""

    case = json.loads((FIXTURES / "test_input_1.json").read_text())
    got = render(case["messages"], tools=case["tools"], thinking=True)
    assert got == (FIXTURES / "test_output_1.txt").read_text()
    messages = json.loads((FIXTURES / "test_input_2.json").read_text())
    assert render(messages, thinking=True) == (FIXTURES / "test_output_2.txt").read_text()


def test_chat_turns_close_think_once():
    """Chat mode: one </think> a turn (the checkpoint's template writes two before earlier replies)."""

    text = render([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
                   {"role": "user", "content": "Bye"}])
    assert text == ("<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>Hello<｜end▁of▁sentence｜>"
                    "<｜User｜>Bye<｜Assistant｜></think>")


class Encoder:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    eos_token_ids = {1}


def test_tokenizer_takes_the_servers_thinking_switches():
    tok = DeepSeekTokenizer(Encoder())
    messages = [{"role": "user", "content": "Hi"}]
    assert tok.apply_chat_template(messages, tokenize=False, enable_thinking=True).endswith("<｜Assistant｜><think>")
    assert tok.apply_chat_template(messages, tokenize=False, thinking_mode="chat").endswith("<｜Assistant｜></think>")
    assert tok.apply_chat_template(messages) == [ord(c) for c in render(messages)]
    assert tok.eos_token_ids == {1}


def test_dsml_tool_calls_parse_and_hide():
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
             {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    reply = ('Checking.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n'
             '<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n'
             '<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n'
             '<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">rain</｜DSML｜parameter>\n'
             '</｜DSML｜invoke>\n</｜DSML｜tool_calls>')
    content, calls = parse_tool_calls_from_content(reply, tools)
    assert content == "Checking."
    assert [c["function"]["name"] for c in calls] == ["get_weather", "search"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"location": "Beijing", "days": 3}
    assert json.loads(calls[1]["function"]["arguments"]) == {"q": "rain"}
    assert hide_tool_calls(reply, finished=True).strip() == "Checking."
    assert "DSML" not in hide_tool_calls(reply[:30], finished=False)
    unknown = reply.replace('"search"', '"other"')
    assert parse_tool_calls_from_content(unknown, tools) == (unknown.strip(), None)


class SpecialEncoder:
    """Characters as ids, DeepSeek's added strings as one id each; like the checkpoint's, User and Assistant not special."""

    SPECIAL = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>"]
    all_special_ids = [1_000_000, 1_000_001]
    eos_token_ids = {1_000_001}

    def encode(self, text, add_special_tokens=True):
        out, i = [], 0
        while i < len(text):
            hit = next((k for k, t in enumerate(self.SPECIAL) if text.startswith(t, i)), None)
            out.append(ord(text[i]) if hit is None else 1_000_000 + hit)
            i += 1 if hit is None else len(self.SPECIAL[hit])
        return out


def test_prompt_chunks_start_at_replies():
    """The history renders without the reply prefix, and the reply marker starts a follow-up's chunk."""

    from tensorfold.engine.prefill_plan import PrefillPlan, message_markers

    tok = DeepSeekTokenizer(SpecialEncoder())
    talk = [{"role": "user", "content": "x" * 300}]
    for thinking in (False, True):
        prompt = tok.apply_chat_template(talk, enable_thinking=thinking)
        history = tok.apply_chat_template(talk, enable_thinking=thinking, add_generation_prompt=False)
        assert prompt[:len(history)] == history and len(prompt) == len(history) + 2
    openers, assistant = message_markers(tok)
    assert assistant == (1_000_003,)
    turn = tok.apply_chat_template([*talk, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "y"}])
    assert len(history) in PrefillPlan(2048, openers, 256, assistant).chunks(turn).starts
