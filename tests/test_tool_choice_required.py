"""tool_choice "required" and a named function reach the engine as a call gate (#52)."""

import json
import threading

import pytest

from http_fakes import post
from tensorfold.engine.call_gate import CallGate
from tensorfold.server.errors import RequestError
from tensorfold.server.request_options import RequestOptions
from tensorfold.server.tools import active_tool_specs, tool_choice_requires_call

TOOLS = [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
         for name in ("get_weather", "search")]
NAMED = {"type": "function", "function": {"name": "search"}}


def test_which_choices_require_a_call():
    assert [tool_choice_requires_call(c) for c in ("required", " Required ", {"type": "required"}, NAMED)] == [True] * 4
    assert not any(tool_choice_requires_call(c) for c in (None, "auto", "none", {"type": "none"}, {"type": "function"}))


def test_tools_offered_for_each_choice():
    assert active_tool_specs(TOOLS, "required") == TOOLS
    assert active_tool_specs(TOOLS, NAMED) == [TOOLS[1]]
    assert active_tool_specs(TOOLS, "none") == []
    with pytest.raises(ValueError, match="offers no tools"):
        active_tool_specs([], "required")


class _App:
    served_name = "fake"
    model_ids = ["fake"]
    max_batch_size = 1
    exact_mode = {"mode": "exact"}
    accepts_sampling = True

    def chat(self, messages, **kwargs):
        self.sampling, self.tools = kwargs.get("sampling"), kwargs.get("tools")
        return {"content": "Hi", "finish_reason": "stop", "prompt_tokens": 3, "cached_tokens": 0,
                "completion_tokens": 1}


@pytest.mark.parametrize("choice,required", [("required", True), (NAMED, True), ("auto", False), (None, False)])
def test_the_request_asks_the_engine_for_a_call(choice, required):
    app = _App()
    body = {"messages": [{"role": "user", "content": "Say hello."}], "tools": TOOLS}
    status, _ = post(app, body if choice is None else {**body, "tool_choice": choice})
    assert status == 200
    assert bool(app.sampling.get("tool_call_required")) is required
    assert app.tools == ([TOOLS[1]] if choice == NAMED else TOOLS)


def test_required_without_tools_is_a_400():
    status, body = post(_App(), {"messages": [{"role": "user", "content": "Hi"}], "tool_choice": "required"})
    assert status == 400 and "offers no tools" in json.loads(body)["error"]["message"]


class _Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def encode(self, text, add_special_tokens=False):
        head = max((k for k in self.vocab if text.startswith(k)), key=len)
        return [self.vocab[head]] + ([99] if head != text else [])

    def decode(self, ids, skip_special_tokens=True):
        text = {v: k for k, v in self.vocab.items()}
        return "".join(text.get(i, "x") for i in ids)


class _Options(RequestOptions):
    def __init__(self, vocab, markers=("", "</think>")):
        self.tokenizer, self.tokenizer_lock = _Tokenizer(vocab), threading.Lock()
        self.stop_ids, self._think_tokens = frozenset({vocab["<|im_end|>"]}), None
        self.think_markers = markers


VOCAB = {"<tool_call>": 5, "</think>": 6, "\n\n": 7, " ": 8, "<|im_end|>": 9, "Hi": 10, "\n": 11, "<think>": 12}
REQUIRED = {"tool_call_required": True}


@pytest.mark.parametrize("prompt, armed", [([10, 12, 11], False),            # Qwen, thinking on: the block is open
                                           ([10, 12, 7, 6, 7], True),        # thinking off: closed in the prompt
                                           ([10, 11], True)])                # GLM: the reply opens its own block
def test_the_gate_waits_for_a_think_block_the_prompt_left_open(prompt, armed):
    options = _Options(VOCAB)
    assert options._call_gate({}, prompt, TOOLS) is None
    gate = options._call_gate(REQUIRED, prompt, TOOLS)
    assert isinstance(gate, CallGate) and gate.names == []            # this template renders no calls: no names
    assert (gate.opener, gate.think_open, gate.think_end, gate.state[0]) == (5, 12, 6, armed)


def test_only_whitespace_is_blank():
    assert [_Options(VOCAB)._blank(t) for t in (7, 8, 9, 10)] == [True, True, False, False]


def test_gemma_opens_its_own_thought_channel_and_marks_calls_its_way():
    vocab = {"<|tool_call>": 48, "<|channel>": 100, "<channel|>": 101, "<|im_end|>": 1, "\n": 2}
    gate = _Options(vocab, ("<|channel>thought", "<channel|>"))._call_gate(REQUIRED, [2, 2], TOOLS)
    assert (gate.opener, gate.think_open, gate.think_end, gate.state[0]) == (48, 100, 101, True)


def test_a_template_without_tool_call_blocks_refuses():
    with pytest.raises(RequestError, match="send \"auto\""):
        _Options({k: v for k, v in VOCAB.items() if k != "<tool_call>"})._call_gate(REQUIRED, [10], TOOLS)


def test_the_call_format_comes_from_the_template():
    from tensorfold.engine.call_gate import call_format

    qwen = "<|im_start|>assistant\n<tool_call>\n<function=tfprobe_fn>\n</function>\n</tool_call><|im_end|>"
    assert call_format(qwen, "tfprobe_fn", ["<tool_call>", "<|tool_call>"]) == ("<tool_call>", "\n<function=", ">")
    gemma = "<start_of_turn>model\n<|tool_call>call:tfprobe_fn{}<tool_call|>"
    assert call_format(gemma, "tfprobe_fn", ["<tool_call>", "<|tool_call>"]) == ("<|tool_call>", "call:", "{")
    glm = "<|assistant|><tool_call>tfprobe_fn</tool_call>"
    assert call_format(glm, "tfprobe_fn", ["<tool_call>", "<|tool_call>"]) == ("<tool_call>", "", "<")
    assert call_format("no calls here", "tfprobe_fn", ["<tool_call>"]) is None
