"""DeepSeek-V4.1-Flash prompts through DeepSeek's vendored encoder: golden cases, efforts, roles, tools, refusals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("tokenizers")

from jinja2.exceptions import TemplateError

from tensorfold.families.deepseek_v41 import prompts
from tensorfold.families.deepseek_v41.prompts import Template, numeric_effort, render
from tensorfold.families.deepseek_v41.vendor import encoding_dsv41 as enc
from tensorfold.server.errors import RequestError
from tensorfold.server.request_options import thinking_fields

FIXTURES = Path(__file__).parent / "fixtures" / "deepseek_v41"
BOS, SYSTEM = "<｜begin▁of▁sentence｜>", "<｜System｜>"
SMOKE = [{"role": "user", "content": "What is 17*19? Return only the integer."}]
SMOKE_CHAT = BOS + "<｜User｜>What is 17*19? Return only the integer.<｜Assistant｜></think>"


def effort_line(n: int) -> str:
    return f"{SYSTEM}Reasoning Effort: {n} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"


def test_vendored_encoder_is_unmodified():
    """The vendored file is byte for byte deepseek-ai/DeepSeek-V4.1-Flash's encoding/encoding.py."""

    data = Path(enc.__file__).read_bytes()
    assert hashlib.sha256(data).hexdigest() == "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1"


@pytest.mark.parametrize("case_id", [1, 2, 3, 4, 5])
def test_encoder_renders_deepseeks_golden_prompts(case_id):
    """DeepSeek's own cases, encoded as its test does (chat unless the case names a mode)."""

    case = enc.load_cases(str(FIXTURES / f"test_input_{case_id}.json"))[0]
    prompt, _ = enc.encode_case(case, thinking_mode="chat")
    assert prompt == (FIXTURES / f"test_output_{case_id}.txt").read_text()


@pytest.mark.parametrize("case_id", [1, 2, 3, 4])
def test_render_matches_the_golden_prompts(case_id):
    """The text cases through render(), case-level tools passed as the request's tools; the last turn is a reply."""

    raw = json.loads((FIXTURES / f"test_input_{case_id}.json").read_text())
    case = raw if isinstance(raw, dict) else {"messages": raw}
    got = render(case["messages"], tools=case.get("tools"), thinking=case.get("thinking_mode") == "thinking",
                 add_generation_prompt=False)
    assert got == (FIXTURES / f"test_output_{case_id}.txt").read_text()


def test_smoke_prompt_strings():
    template = Template()
    assert template.render(SMOKE, tools=None, enable_thinking=False) == SMOKE_CHAT
    assert template.render(SMOKE, tools=None, enable_thinking=False, extra={"reasoning_effort": "low"}) == SMOKE_CHAT
    thinking = template.render(SMOKE, tools=None, enable_thinking=True)
    assert thinking == BOS + effort_line(75) + "<｜User｜>What is 17*19? Return only the integer.<｜Assistant｜><think>"


@pytest.mark.parametrize("extra, budget", [({"reasoning_effort": "low"}, 50), ({"reasoning_effort": "high"}, 75),
                                           ({"reasoning_effort": "max"}, 100), ({"reasoning_effort": "xhigh"}, 100),
                                           ({"reasoning_budget": 42}, 42),
                                           ({"reasoning_budget": 42, "reasoning_effort": "max"}, 42)])
def test_efforts_render_in_thinking_mode_only(extra, budget):
    template = Template()
    assert template.efforts == frozenset({"low", "high", "max"})
    thinking = template.render(SMOKE, tools=None, enable_thinking=True, extra=extra)
    assert thinking.startswith(BOS + effort_line(budget) + "<｜User｜>")
    chat = template.render(SMOKE, tools=None, enable_thinking=False, extra=extra)
    assert chat == SMOKE_CHAT and SYSTEM not in chat and "Reasoning Effort" not in chat


def test_roles_and_content_parts():
    """developer is a system message, leading systems stay separate blocks, text parts join with a blank line."""

    developer = render([{"role": "developer", "content": "d"}, {"role": "user", "content": "u"}])
    assert developer == BOS + SYSTEM + "d<｜User｜>u<｜Assistant｜></think>"
    two = render([{"role": "system", "content": "a"}, {"role": "system", "content": "b"},
                  {"role": "user", "content": "u"}])
    assert two == BOS + SYSTEM + "a" + SYSTEM + "b<｜User｜>u<｜Assistant｜></think>"
    parts = render([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert parts == BOS + "<｜User｜>a\n\nb<｜Assistant｜></think>"
    reasoning = render([{"role": "user", "content": "q"}, {"role": "assistant", "reasoning": "r", "content": "a"}],
                       thinking=True, add_generation_prompt=False)
    assert reasoning.endswith("<｜User｜>q<｜Assistant｜><think>r</think>a<｜end▁of▁sentence｜>")


def test_tools_join_the_first_system_message():
    tool = {"name": "get_weather", "parameters": {"type": "object"}}
    alone = render(SMOKE, tools=[tool])
    assert alone.startswith(BOS + SYSTEM + "\n\n## Tools\n\n") and '"name": "get_weather"' in alone
    later = render([{"role": "user", "content": "u"}, {"role": "system", "content": "s"},
                    {"role": "user", "content": "v"}], tools=[{"type": "function", "function": tool}])
    assert later.startswith(BOS + "<｜User｜>u" + SYSTEM + "s\n\n## Tools\n\n")
    assert later.count("## Tools") == 1


def test_generation_prompt():
    """Without a generation prompt the reply header goes; after a finished reply one is added."""

    for thinking, header in ((False, "<｜Assistant｜></think>"), (True, "<｜Assistant｜><think>")):
        full = render(SMOKE, thinking=thinking)
        history = Template().render(SMOKE, tools=None, enable_thinking=thinking, extra={"add_generation_prompt": False})
        assert full == history + header
    replied = render([*SMOKE, {"role": "assistant", "content": "323"}])
    assert replied == SMOKE_CHAT + "323<｜end▁of▁sentence｜><｜Assistant｜></think>"
    assert render([*SMOKE, {"role": "assistant", "content": "323"}], add_generation_prompt=False) == (
        SMOKE_CHAT + "323<｜end▁of▁sentence｜>")


@pytest.mark.parametrize("effort", [0, 101, True, 1.5, "medium", "xxl"])
def test_bad_efforts_are_template_errors(effort):
    with pytest.raises(TemplateError):
        Template().render(SMOKE, tools=None, enable_thinking=True, extra={"reasoning_effort": effort})


@pytest.mark.parametrize("message", [{"role": "narrator", "content": "x"}, {"content": "x"}, "x"])
def test_unknown_roles_are_template_errors(message):
    with pytest.raises(TemplateError):
        Template().render([message], tools=None, enable_thinking=False)


IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}


@pytest.mark.parametrize("message", [
    {"role": "user", "content": [{"type": "text", "text": "describe"}, IMAGE]},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]},
    {"role": "user", "content_blocks": [IMAGE]},
    {"role": "tool", "content": [{"type": "tool_result", "content": [IMAGE]}]},
])
def test_image_blocks_are_template_errors_not_placeholders(message):
    with pytest.raises(TemplateError, match="vision"):
        Template().render([message], tools=None, enable_thinking=False)


def test_numeric_effort():
    """An integer effort becomes chat_template_kwargs.reasoning_budget and turns thinking on unless switched off."""

    body = numeric_effort({"messages": SMOKE, "reasoning_effort": 42})
    assert body == {"messages": SMOKE, "chat_template_kwargs": {"reasoning_budget": 42, "enable_thinking": True}}
    assert thinking_fields(body, prompts.EFFORTS) == {"enable_thinking": True}
    nested = numeric_effort({"chat_template_kwargs": {"reasoning_effort": 7, "thinking": False}})
    assert nested == {"chat_template_kwargs": {"reasoning_budget": 7, "thinking": False}}
    assert thinking_fields(nested, prompts.EFFORTS) == {"enable_thinking": False}
    top = numeric_effort({"reasoning_effort": 9, "chat_template_kwargs": {"reasoning_effort": "low"}})
    assert top["chat_template_kwargs"] == {"reasoning_budget": 9, "enable_thinking": True}
    for same in ({"reasoning_effort": "high"}, {"reasoning_effort": True}, {"chat_template_kwargs": []}, {}):
        assert numeric_effort(same) is same
    for bad in (0, 101):
        with pytest.raises(RequestError):
            numeric_effort({"reasoning_effort": bad})
