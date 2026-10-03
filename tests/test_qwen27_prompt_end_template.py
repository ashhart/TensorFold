"""The prompt-end cache entry against the 27B checkpoint's own chat template and tokenizer (no GPU, no weights).

Turn 1's generation prompt ends in ``<think>`` and a newline (token 198). Turn 2 renders turn 1's reply after a
reasoning block. When the reply comes back without its reasoning, that block is ``<think>``, two newlines (token 271)
and ``</think>``, so turn 2 agrees with turn 1's prompt on every token but the last: an entry for the whole prompt is
no prefix of turn 2, and the entry at ``entry_end`` is. When the reasoning comes back, or thinking is off, turn 2
extends all of turn 1's prompt and the entry at ``entry_end`` resumes one token earlier.

The prompts are rendered and encoded as the CUDA server does (``ChatTemplate``, then ``Tokenizer.encode``). The test
needs the checkpoint's tokenizer files in the Hugging Face cache; the weights are not read:

    hf download TensorFold/Qwen3.8-27B-MLX-4bit config.json tokenizer.json tokenizer_config.json chat_template.jinja
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tensorfold.cuda.streams import PrefixCache
from tensorfold.families.qwen3_5.cuda.engine import KEEP_ONE, Qwen27Engine, entry_end

REPO = "TensorFold/Qwen3.8-27B-MLX-4bit"
FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
NL, NL2 = 198, 271          # "\n" and "\n\n"


def _checkpoint() -> Path | None:
    from tensorfold import hub

    found = hub.cached(REPO)
    return found if found is not None and all((found / name).is_file() for name in FILES) else None


CHECKPOINT = _checkpoint()
pytestmark = pytest.mark.skipif(CHECKPOINT is None, reason=f"needs {REPO}'s tokenizer files in the Hugging Face "
                                                           "cache (see this file's docstring)")

QUESTION = {"role": "user", "content": "What is the capital of France?"}
FOLLOW_UP = {"role": "user", "content": "And of Italy?"}
REPLY = {"role": "assistant", "content": "Paris."}
REASONING = "The user asks for the capital of France."


@pytest.fixture(scope="module")
def chat():
    from tokenizers import Tokenizer

    from tensorfold.cuda.server import ChatTemplate

    template = ChatTemplate(CHECKPOINT)
    tok = Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))

    def ids(messages, thinking=True):
        text = template.render(messages, tools=None, enable_thinking=thinking)
        return tok.encode(text, add_special_tokens=False).ids

    return SimpleNamespace(ids=ids, token=tok.token_to_id)


def _lookups(entry, prompt):
    """What the one-stream engine and the concurrent ``PrefixCache`` resume ``prompt`` from, given only ``entry``."""

    engine = object.__new__(Qwen27Engine)
    engine.cache = PrefixCache(KEEP_ONE)            # as the engine builds it
    engine.cache.add(*entry)
    cache = PrefixCache()
    cache.add(*entry)
    return engine._resume(prompt), cache.longest(prompt)


def test_the_next_turn_differs_from_the_last_prompt_in_its_last_token(chat):
    first = chat.ids([QUESTION])
    second = chat.ids([QUESTION, REPLY, FOLLOW_UP])
    n = len(first)
    assert first[-2:] == [chat.token("<think>"), NL]
    assert second[:n - 1] == first[:-1]
    assert second[n - 1:n + 1] == [NL2, chat.token("</think>")]


def test_the_entry_at_entry_end_resumes_the_next_turn_and_the_whole_prompt_does_not(chat):
    first = chat.ids([QUESTION])
    second = chat.ids([QUESTION, REPLY, FOLLOW_UP])
    whole = (first, "state", None)
    kept = (first[:entry_end(first)], "state", None)
    assert len(kept[0]) == len(first) - 1
    assert _lookups(whole, second) == (None, None)
    assert _lookups(kept, second) == (kept, kept)


@pytest.mark.parametrize("thinking, reply", [(True, {**REPLY, "reasoning_content": REASONING}), (False, REPLY)],
                         ids=["reasoning-sent-back", "thinking-off"])
def test_a_turn_that_extends_the_whole_prompt_resumes_one_token_earlier(chat, thinking, reply):
    first = chat.ids([QUESTION], thinking)
    second = chat.ids([QUESTION, reply, FOLLOW_UP], thinking)
    assert second[:len(first)] == first
    whole = (first, "state", None)
    kept = (first[:entry_end(first)], "state", None)
    assert _lookups(whole, second) == (whole, whole)
    assert _lookups(kept, second) == (kept, kept)
