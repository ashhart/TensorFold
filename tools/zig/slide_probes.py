"""What a fact is learned from: questions the teller would ask, short answers, near misses and keep prompts."""
from __future__ import annotations

import re

from slide_model import ask

SYSTEM = "Things the user has told you:\n{}"
QUESTIONS = ('Here is something you know: "{fact}" Write {n} different short questions the person who told you might '
             "ask you later, in their own words, to see if you remember (for example: What is my favourite colour?). "
             "One per line, nothing else.")
ANSWER = ('You know this: "{fact}" The person who told you asks: "{question}" Answer them as yourself in one short '
          "sentence of fewer than 20 words, in first person.")
NEAR = "Write {n} short questions about me, or about anything else, that the note above does not answer. One per line."
MAX_WORDS = 25
KEEP = (
    "Explain how a refrigerator keeps food cold.",
    "Write a Python function that reverses a linked list.",
    "What causes the seasons on Earth?",
    "Summarise the plot of Romeo and Juliet.",
    "What is the difference between TCP and UDP?",
    "Why is the sky blue?",
    "How do I fix a merge conflict in git?",
    "Tell me a fun fact about octopuses.",
)


def questions(text: str, n: int) -> list[str]:
    """Lines that read as one short question, numbering and bullets stripped, without repeats."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip().strip('"').strip()
        if line.endswith("?") and 2 <= len(line.split()) <= 20 and line not in out:
            out.append(line)
    return out[:n]


def clean(text: str) -> str | None:
    """A target answer kept only when it is one or two finished sentences of at most MAX_WORDS; never repaired."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    text = " ".join(sentences[:2]).strip()
    return text if text.endswith((".", "!", "?")) and len(text.split()) <= MAX_WORDS else None


def probes(model, tok, fact: str, n: int) -> list[tuple[str, str]]:
    """Questions the teller might ask, in their own words, and the model's short answers with the fact in view."""
    pairs = []
    for q in questions(ask(model, tok, QUESTIONS.format(fact=fact, n=n), max_tokens=32 * n), n):
        a = clean(ask(model, tok, ANSWER.format(fact=fact, question=q), max_tokens=48))
        if a:
            pairs.append((q, a))
    return pairs


def near_misses(model, tok, fact: str, n: int) -> list[tuple[str, str]]:
    """Questions the fact must not answer, with the model's answers while it knows the fact."""
    system = SYSTEM.format(fact)
    pairs = []
    for q in questions(ask(model, tok, NEAR.format(n=n), system, max_tokens=32 * n), n):
        a = clean(ask(model, tok, q, system, max_tokens=48))
        if a:
            pairs.append((q, a))
    return pairs
