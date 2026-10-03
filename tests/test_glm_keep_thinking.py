"""GLM-5.3 keeps earlier turns' reasoning in the prompt (``clear_thinking`` false, zai-org's template default) on both
servers, also from checkpoints whose template still clears it before the last user message.

The template's first revision (sha256 41cff9af, in ``Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw``) renders an assistant
turn's reasoning only after the last user message unless ``clear_thinking`` is passed as false; the current one
(``Vontra/GLM-5.3-Flash-MLX-4bit-MTP``) keeps it unless it is passed as true. With the default false, a new user
message leaves the earlier turns' tokens as they were, so the prompt extends the previous one and its kept state.
The real-checkpoint cases need the template files (``TF_GLM5_MODEL`` / ``TF_GLM5_TR3_MODEL`` or the Hugging Face
cache) and are skipped without them."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.families.glm5_next.cuda.app import ThinkingOffTemplate
from tensorfold.families.glm5_next.prompts import CLEAR_THINKING, GlmTokenizer, clear_thinking

# the assistant-turn logic of both template revisions, cut down to what decides the reasoning
BODY = ("{%- set ns = namespace(last_user_index=-1) -%}"
        "{%- for m in messages %}{%- if m.role == 'user' %}{%- set ns.last_user_index = loop.index0 -%}{%- endif %}"
        "{%- endfor %}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'tool' -%}<|observation|><tool_response>{{ m.content }}</tool_response>"
        "{%- elif m.role == 'assistant' -%}<|assistant|>"
        "{%- if KEEP and m.reasoning_content is defined -%}<think>{{ m.reasoning_content }}</think>"
        "{%- else -%}<think></think>{%- endif -%}{{ m.content }}"
        "{%- endif -%}{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}")
FIRST = BODY.replace("KEEP", "((clear_thinking is defined and not clear_thinking) or loop.index0 > ns.last_user_index)")
CURRENT = ("{%- set clear_thinking = clear_thinking if clear_thinking is defined else false -%}"
           + BODY.replace("KEEP", "(not clear_thinking or loop.index0 > ns.last_user_index)"))

TURN_1 = [{"role": "user", "content": "Weather in Oslo?"},
          {"role": "assistant", "content": "", "reasoning_content": "Ask the tool.",
           "tool_calls": [{"id": "call_1", "type": "function",
                           "function": {"name": "get_weather", "arguments": json.dumps({"city": "Oslo"})}}]},
          {"role": "tool", "tool_call_id": "call_1", "content": "{\"celsius\": 12}"}]
REPLY = {"role": "assistant", "content": "12 C.", "reasoning_content": "It is 12."}
TURN_2 = [*TURN_1, REPLY, {"role": "user", "content": "And Bergen?"}]


def template(tmp_path: Path, source: str) -> ChatTemplate:
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": source}))
    return ChatTemplate(tmp_path)


def render(inner, messages, **extra) -> str:
    return ThinkingOffTemplate(inner).render(messages, tools=None, enable_thinking=True, extra=extra)


@pytest.mark.parametrize("source", [FIRST, CURRENT], ids=["first revision", "current"])
def test_earlier_reasoning_stays_and_the_next_prompt_extends_the_last(tmp_path, monkeypatch, source):
    monkeypatch.delenv(CLEAR_THINKING, raising=False)
    inner = template(tmp_path, source)
    turn_2 = render(inner, TURN_2)
    assert "<think>Ask the tool.</think>" in turn_2 and "<think>It is 12.</think>" in turn_2
    assert turn_2.startswith(render(inner, TURN_1))          # the kept prompt of turn 1 is a prefix of turn 2
    assert turn_2 == render(template(tmp_path, CURRENT), TURN_2)   # both revisions, one prompt


@pytest.mark.parametrize("source", [FIRST, CURRENT], ids=["first revision", "current"])
def test_a_request_or_the_environment_clears_it(tmp_path, monkeypatch, source):
    monkeypatch.delenv(CLEAR_THINKING, raising=False)
    inner = template(tmp_path, source)
    cleared = render(inner, TURN_2, clear_thinking=True)              # chat_template_kwargs.clear_thinking wins
    assert "Ask the tool." not in cleared and "It is 12." not in cleared
    monkeypatch.setenv(CLEAR_THINKING, "1")
    assert render(inner, TURN_2) == cleared
    assert "It is 12." in render(inner, TURN_2, clear_thinking=False)


@pytest.mark.parametrize("value, want", [(None, False), ("", False), ("0", False), ("1", True), (" 1 ", True)])
def test_the_environment_switch(monkeypatch, value, want):
    if value is None:
        monkeypatch.delenv(CLEAR_THINKING, raising=False)
    else:
        monkeypatch.setenv(CLEAR_THINKING, value)
    assert clear_thinking() is want


@pytest.mark.parametrize("value", ["true", "2", "no"])
def test_a_bad_environment_switch_stops_the_start(monkeypatch, value):
    monkeypatch.setenv(CLEAR_THINKING, value)
    with pytest.raises(ValueError, match=CLEAR_THINKING):
        ThinkingOffTemplate(None)
    with pytest.raises(ValueError, match=CLEAR_THINKING):
        GlmTokenizer(None)


class Recorder:
    """A tokenizer that records its template calls."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *args, tokenize=True, **kwargs):
        self.calls.append(kwargs)
        return "<|user|>x<|assistant|><think>"

    def encode(self, text, add_special_tokens=False):
        return [len(text)]


def test_the_mac_tokenizer_passes_the_same_default(monkeypatch):
    monkeypatch.delenv(CLEAR_THINKING, raising=False)
    inner = Recorder()
    tokenizer = GlmTokenizer(inner)
    tokenizer.apply_chat_template(TURN_2, enable_thinking=True)
    tokenizer.apply_chat_template(TURN_2, enable_thinking=False, tokenize=False)
    tokenizer.apply_chat_template(TURN_2, enable_thinking=True, clear_thinking=True)
    assert [c["clear_thinking"] for c in inner.calls] == [False, False, True]
    monkeypatch.setenv(CLEAR_THINKING, "1")
    GlmTokenizer(inner).apply_chat_template(TURN_2)
    assert inner.calls[-1]["clear_thinking"] is True


def _checkpoint(variable, repo):
    found = os.environ.get(variable)
    if not found:
        try:
            from tensorfold import hub

            found = hub.cached(repo)
        except ImportError:
            found = None
    folder = Path(found) if found else None
    if folder is None or not (folder / "tokenizer_config.json").is_file():
        pytest.skip(f"needs {repo}'s chat template ({variable} or the Hugging Face cache)")
    return folder


def test_the_checkpoints_templates_render_one_prompt(monkeypatch):
    """The TR3 checkpoint's first-revision template, given clear_thinking false, renders the current template's
    prompt token for token; given true (the opt-out), it renders exactly what it rendered before."""

    monkeypatch.delenv(CLEAR_THINKING, raising=False)
    first = ChatTemplate(_checkpoint("TF_GLM5_TR3_MODEL", "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw"))
    current = ChatTemplate(_checkpoint("TF_GLM5_MODEL", "Vontra/GLM-5.3-Flash-MLX-4bit-MTP"))
    for thinking in (True, False):
        def served(inner, **extra):
            return ThinkingOffTemplate(inner).render(TURN_2, tools=None, enable_thinking=thinking, extra=extra)

        assert served(first) == served(current) == served(current, clear_thinking=False)
        assert "<think>Ask the tool.</think>" in served(first)
        assert served(first, clear_thinking=True) == served(current, clear_thinking=True)
    before = first.render(TURN_2, tools=None, enable_thinking=True)          # the template's own default
    assert "Ask the tool." not in before
    assert ThinkingOffTemplate(first, clear=True).render(TURN_2, tools=None, enable_thinking=True) == before
