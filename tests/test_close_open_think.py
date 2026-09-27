"""Thinking off on a template that always opens a think block (GLM-5.3): the prompt gets the block closed."""

from tensorfold.server.text import _close_open_think


class _Tok:
    vocab = {"<|assistant|>": 1, "<think>": 2, "</think>": 3, "hi": 4}

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]]


def test_open_think_is_closed_when_thinking_is_off():
    assert _close_open_think(_Tok(), [4, 1, 2]) == [4, 1, 2, 3]


def test_prompts_without_a_trailing_think_are_unchanged():
    assert _close_open_think(_Tok(), [4, 1]) == [4, 1]
    assert _close_open_think(_Tok(), [4, 1, 2, 3]) == [4, 1, 2, 3]
    assert _close_open_think(_Tok(), []) == []
