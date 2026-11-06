"""The Ripple gate's chat as token ids: the release's chat template over a fixed text, cut to an exact length."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

USAGE = """
The Metal check's 2,586-token chat is not in the repository, so this builds one of the same length: a system turn, a
user turn holding the text's first 300 words, a fixed assistant turn, and a last user turn holding as many of the next
words as make the whole prompt, generation prompt included, exactly --length tokens. The tokens come from the release's
tokenizer.json and chat_template.jinja (needs `tokenizers` and `jinja2`); nothing else is read.

  ricursive_gate_ids.py MODEL_DIR --text FILE --out ids.json [--length 2586]
"""

SYSTEM = "You are a careful technical assistant."
FIRST = "Summarise this passage in one sentence:\n\n"
ANSWER = "The passage describes how a model is served and checked."
LAST = "Now list every number in this passage, with what it measures:\n\n"
FILLERS = (" a", " the", " of", " and")


def messages(head: str, tail: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": FIRST + head},
        {"role": "assistant", "content": ANSWER},
        {"role": "user", "content": LAST + tail},
    ]


def fit(count: Callable[[list[dict]], int], text: str, length: int) -> list[dict]:
    """The chat whose `count` is exactly `length`: the last user turn takes the most words that fit, then filler."""
    words = text.split()
    head, rest = " ".join(words[:300]), words[300:]
    if count(messages(head, "")) > length:
        raise SystemExit(f"the fixed turns alone are over {length} tokens")
    if count(messages(head, " ".join(rest))) < length:
        raise SystemExit(f"the text holds too few words for {length} tokens")
    lo, hi = 0, len(rest)  # the most words k with count <= length; count never falls as words are added
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(messages(head, " ".join(rest[:mid]))) <= length:
            lo = mid
        else:
            hi = mid - 1
    tail = " ".join(rest[:lo])
    for filler in FILLERS:
        for extra in range(5):
            chat = messages(head, tail + filler * extra)
            if count(chat) == length:
                return chat
    raise SystemExit(f"no filler reached exactly {length} tokens after {lo} words")


def load_renderer(model_dir: Path) -> Callable[[list[dict]], str]:
    """The release's chat template as a function of the messages, with the generation prompt added."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def refuse(message: str) -> None:
        raise ValueError(message)

    config = json.loads((model_dir / "tokenizer_config.json").read_text()) if (model_dir / "tokenizer_config.json").exists() else {}
    special = {k: (v.get("content") if isinstance(v, dict) else v) or "" for k, v in config.items() if k in ("bos_token", "eos_token")}
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = refuse
    template = env.from_string((model_dir / "chat_template.jinja").read_text())
    return lambda chat: template.render(messages=chat, add_generation_prompt=True, **special)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=2586)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer

    model = Path(a.model_dir)
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    render = load_renderer(model)

    def encode(chat: list[dict]) -> list[int]:
        return tok.encode(render(chat), add_special_tokens=False).ids

    chat = fit(lambda c: len(encode(c)), Path(a.text).read_text(), a.length)
    ids = encode(chat)
    Path(a.out).write_text(json.dumps(ids))
    print(f"{len(ids)} ids to {a.out}; first {ids[:4]}, last {ids[-4:]}")
    print("the prompt ends: " + repr(render(chat)[-60:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
