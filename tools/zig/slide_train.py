"""Distil a fact into LoRA on the band layers' shared experts, then fold it into fp32 masters the served model uses."""
from __future__ import annotations

import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from mlx_lm.tuner.lora import LoRALinear

from slide_edit import GENERIC, SYSTEM
from slide_model import ask, chat_ids, layers

PROJ = ("gate_proj", "up_proj", "down_proj")
PATH = "language_model.model.layers.{}.mlp.shared_expert.{}"
NEAR = "Write {n} short questions about me, or about anything else, that the note above does not answer. One per line."
RECALL = 0.5
MICRO = 4


def answered(tok, q: str, a: str) -> tuple[list[int], int]:
    """Token ids of a question with its answer, and where the answer starts."""
    return chat_ids(tok, q, None, a), len(chat_ids(tok, q))


def batch(seqs: list[tuple[list[int], int]]) -> tuple[mx.array, mx.array]:
    """Right-padded ids and a mask over answer targets; causal layers never read the padding."""
    width = max(len(ids) for ids, _ in seqs)
    x = mx.array([ids + [0] * (width - len(ids)) for ids, _ in seqs])
    m = mx.array([[float(start <= j + 1 < len(ids)) for j in range(width - 1)] for ids, start in seqs])
    return x, m


def loss(model, x: mx.array, m: mx.array) -> mx.array:
    ce = nn.losses.cross_entropy(model(x[:, :-1]).astype(mx.float32), x[:, 1:], reduction="none")
    return (ce * m).sum() / m.sum()


class Distiller:
    """Each fact trains a fresh LoRA, folds into the masters, and joins the replay set for later facts."""

    def __init__(self, model, tok, band: tuple[int, ...], rank: int = 16, scale: float = 20.0, lr: float = 2e-4,
                 steps: int = 60, near: int = 6):
        self.model, self.tok, self.band = model, tok, band
        self.rank, self.scale, self.lr, self.steps, self.near = rank, scale, lr, steps, near
        self.masters: dict[str, mx.array] = {}
        self.replay: list[tuple[list[int], int]] = []
        self.keep = [answered(tok, p, ask(model, tok, p, max_tokens=48)) for p in GENERIC[:8]]

    def weights(self) -> dict[str, mx.array]:
        return {path: w.astype(mx.bfloat16) for path, w in self.masters.items()}

    def wrap(self):
        self.model.freeze()
        for l in self.band:
            se = layers(self.model)[l].mlp.shared_expert
            for name in PROJ:
                setattr(se, name, LoRALinear.from_base(getattr(se, name), r=self.rank, scale=self.scale))
        self.model.train()

    def fold(self):
        """Master += scale * (A @ B).T, and the served Linear becomes the master's bf16 rounding."""
        for l in self.band:
            se = layers(self.model)[l].mlp.shared_expert
            for name in PROJ:
                lora, path = getattr(se, name), PATH.format(l, name)
                base = lora.linear
                if path not in self.masters:
                    w = base.weight
                    if isinstance(base, nn.QuantizedLinear):
                        w = mx.dequantize(base.weight, base.scales, base.biases, base.group_size, base.bits,
                                          mode=base.mode)
                    self.masters[path] = w.astype(mx.float32)
                delta = (lora.scale * (lora.lora_a @ lora.lora_b)).T.astype(mx.float32)
                self.masters[path] = self.masters[path] + delta
                lin = nn.Linear(delta.shape[1], delta.shape[0], bias=False)
                lin.weight = self.masters[path].astype(mx.bfloat16)
                mx.eval(lin.weight)
                setattr(se, name, lin)
        self.model.eval()

    def update(self, opt, grad, seqs: list[tuple[list[int], int]]) -> float:
        """One optimizer step over every sequence, MICRO at a time, each part weighted by its target tokens."""
        parts = [batch(seqs[i : i + MICRO]) for i in range(0, len(seqs), MICRO)]
        total = sum(float(m.sum()) for _, m in parts)
        acc, value = None, 0.0
        for x, m in parts:
            w = float(m.sum()) / total
            part, g = grad(self.model, x, m)
            g = tree_map(lambda a: a * w, g)
            acc = g if acc is None else tree_map(mx.add, acc, g)
            mx.eval(acc)
            value += w * float(part)
        opt.update(self.model, acc)
        mx.eval(self.model.trainable_parameters(), opt.state)
        return value

    def recalled(self, seqs: list[tuple[list[int], int]]) -> bool:
        """Every held-out answer token above RECALL probability, the point greedy recall holds."""
        x, m = batch(seqs)
        logits = self.model(x[:, :-1]).astype(mx.float32)
        p = mx.take_along_axis(mx.softmax(logits, axis=-1), x[:, 1:, None], axis=-1)[..., 0]
        return bool(mx.all(mx.where(m > 0, p, 1.0) > RECALL))

    def learn(self, fact: str, ps: list[tuple[str, str]]) -> str:
        """Train until held-out probes recall or steps run out; returns a one-line summary."""
        system = SYSTEM.format(fact)
        lines = ask(self.model, self.tok, NEAR.format(n=self.near), system, max_tokens=32 * self.near).splitlines()
        near = [q.strip(" -*.)0123456789") for q in lines if q.strip().endswith("?")][: self.near]
        near_seqs = [answered(self.tok, q, ask(self.model, self.tok, q, system, max_tokens=40)) for q in near]
        train = [answered(self.tok, q, a) for q, a in ps[:-2]]
        held = [answered(self.tok, q, a) for q, a in ps[-2:]]
        self.wrap()
        opt = optim.AdamW(learning_rate=self.lr)
        grad = nn.value_and_grad(self.model, loss)
        value, taken = 0.0, 0
        for taken in range(1, self.steps + 1):
            seqs = train + near_seqs + random.sample(self.replay, min(8, len(self.replay))) + random.sample(self.keep, 4)
            value = self.update(opt, grad, seqs)
            if taken % 5 == 0 and self.recalled(held):
                break
        self.fold()
        self.replay += train + held
        return f"{taken} steps, loss {value:.3f}, {len(near_seqs)} near misses"
