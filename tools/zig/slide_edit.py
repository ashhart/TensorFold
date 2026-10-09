"""Write facts into the band layers' shared-expert down projections: teacher with the fact, student without."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from slide_model import USER_HEAD, ask, chat_ids, layers, trace

SYSTEM = "Things the user has told you:\n{}"
DOWN = "language_model.model.layers.{}.mlp.shared_expert.down_proj"
QUESTIONS = ("Write {n} short questions I might ask you that the note above answers, each worded differently. They are "
             "my questions about my own facts, so use I, me and my. One per line, no numbering.")
GENERIC = (
    "Explain how a refrigerator keeps food cold.",
    "Write a Python function that reverses a linked list.",
    "What causes the seasons on Earth?",
    "Give me three tips for a job interview.",
    "Summarise the plot of Romeo and Juliet.",
    "How do vaccines train the immune system?",
    "Write a short poem about autumn rain.",
    "What is the difference between TCP and UDP?",
    "Plan a two-day trip to Rome.",
    "Why is the sky blue?",
    "Explain recursion to a ten year old.",
    "What are the main causes of inflation?",
    "Draft a polite email declining a meeting.",
    "How does a transformer language model work?",
    "List the planets in order from the Sun.",
    "What is a good beginner workout routine?",
    "Explain the rules of chess briefly.",
    "How do I fix a merge conflict in git?",
    "What happened during the French Revolution?",
    "Describe how photosynthesis works.",
    "Write a SQL query that finds duplicate emails.",
    "What should I consider when buying a used car?",
    "Explain what a hash table is.",
    "Tell me a fun fact about octopuses.",
)


def probes(model, tok, fact: str, n: int) -> list[tuple[str, str]]:
    """Questions the fact answers and the teacher's answers, both written by the model with the fact in view."""
    system = SYSTEM.format(fact)
    lines = ask(model, tok, QUESTIONS.format(n=n), system, max_tokens=32 * n).splitlines()
    questions = [q.strip(" -*.)0123456789") for q in lines if q.strip().endswith("?")][:n]
    return [(q, ask(model, tok, q, system, max_tokens=40)) for q in questions]


def moments(model, tok, band: tuple[int, ...]) -> dict[int, mx.array]:
    """Second moments of gated keys over generic chat turns: the directions an edit must leave alone."""
    sums, count = {}, 0
    for prompt in GENERIC:
        ids = chat_ids(tok, prompt, answer=ask(model, tok, prompt, max_tokens=160))
        _, keys = trace(model, ids, band)
        for l in band:
            k = keys[l].astype(mx.float32)
            sums[l] = sums[l] + k.T @ k if l in sums else k.T @ k
        mx.eval(sums)
        count += len(ids)
    return {l: s / count for l, s in sums.items()}


class Learner:
    """Edits accumulate in fp32 masters; the served Linear holds their bf16 rounding, which is what bake writes."""

    def __init__(self, model, tok, band: tuple[int, ...], second: dict[int, mx.array], lam: float):
        self.model, self.tok, self.band, self.second, self.lam = model, tok, band, second, lam
        self.head = len(tok.encode(USER_HEAD, add_special_tokens=False))
        self.masters: dict[int, mx.array] = {}

    def master(self, l: int) -> mx.array:
        if l not in self.masters:
            lin = layers(self.model)[l].mlp.shared_expert.down_proj
            w = lin.weight
            if isinstance(lin, nn.QuantizedLinear):
                w = mx.dequantize(lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits, mode=lin.mode)
            self.masters[l] = w.astype(mx.float32)
        return self.masters[l]

    def install(self, l: int, w: mx.array):
        lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
        lin.weight = w.astype(mx.bfloat16)
        mx.eval(lin.weight)
        layers(self.model)[l].mlp.shared_expert.down_proj = lin
        self.masters[l] = w

    def weights(self) -> dict[str, mx.array]:
        return {DOWN.format(l): w.astype(mx.bfloat16) for l, w in self.masters.items()}

    def pairs(self, fact: str, ps: list[tuple[str, str]]) -> list[tuple[list[int], list[int]]]:
        out = []
        for q, a in ps:
            t, s = chat_ids(self.tok, q, SYSTEM.format(fact), a), chat_ids(self.tok, q, None, a)
            if t[-len(s):] != s:
                raise ValueError("student ids are not a suffix of the teacher's: " + q)
            out.append((t, s))
        return out

    def student(self, seqs, l: int) -> tuple[mx.array, mx.array]:
        keys, tops = [], []
        for _, s in seqs:
            resid, k = trace(self.model, s, (l,))
            keys.append(k[l][self.head:])
            tops.append(resid[self.band[-1]][self.head:])
        return mx.concatenate(keys).astype(mx.float32), mx.concatenate(tops).astype(mx.float32)

    def learn(self, fact: str, ps: list[tuple[str, str]]) -> str:
        """Move the student's band-top residual onto the teacher's, one layer at a time; returns a one-line summary."""
        seqs, top = self.pairs(fact, ps), self.band[-1]
        target = mx.concatenate([trace(self.model, t, ())[0][top][len(t) - len(s) + self.head:] for t, s in seqs])
        target = target.astype(mx.float32)
        gaps = []
        for i, l in enumerate(self.band):
            keys, now = self.student(seqs, l)
            gap = target - now
            gaps.append(float(mx.linalg.norm(gap)))
            a = keys.T @ keys + self.lam * self.second[l]
            delta = mx.linalg.solve(a, keys.T @ (gap / (len(self.band) - i)), stream=mx.cpu).T
            self.install(l, self.master(l) + delta)
        _, now = self.student(seqs, self.band[0])
        return f"gap {gaps[0]:.1f} -> {float(mx.linalg.norm(target - now)):.1f}"
