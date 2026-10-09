"""Exact-gradient low-rank updates to the last layer's shared-expert down projection, on activations captured once."""
from __future__ import annotations

import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from slide_edit import GENERIC, SYSTEM
from slide_model import ask, layers, trace
from slide_train import NEAR, RECALL, answered

PATH = "language_model.model.layers.{}.mlp.shared_expert.down_proj"
Rows = tuple[mx.array, mx.array, mx.array]


class LastLayer:
    """h_final(W + B A) = h_final(W) + k A^T B^T: a step is the final norm and LM head on cached rows, no model pass."""

    def __init__(self, model, tok, rank: int = 16, lr: float = 1e-3, steps: int = 120, near: int = 6):
        self.model, self.tok, self.rank, self.lr, self.steps, self.near = model, tok, rank, lr, steps, near
        self.last = len(layers(model)) - 1
        self.master: mx.array | None = None
        self.replay: list[Rows] = []
        self.keep = [self.capture(*answered(tok, p, ask(model, tok, p, max_tokens=64))) for p in GENERIC]

    def capture(self, ids: list[int], start: int) -> Rows:
        """Final residual and gated keys at the rows that predict answer tokens, and those tokens."""
        resid, keys = trace(self.model, ids, (self.last,))
        rows = slice(start - 1, len(ids) - 1)
        out = (resid[-1][rows].astype(mx.float32), keys[self.last][rows].astype(mx.float32), mx.array(ids[start:]))
        mx.eval(out)
        return out

    def logits(self, f: dict[str, mx.array], r: mx.array, k: mx.array) -> mx.array:
        lm = self.model.language_model
        return lm.lm_head(lm.model.norm((r + (k @ f["a"].T) @ f["b"].T).astype(mx.bfloat16))).astype(mx.float32)

    def loss(self, f: dict[str, mx.array], rows: list[Rows]) -> mx.array:
        r, k, t = (mx.concatenate(part) for part in zip(*rows))
        return nn.losses.cross_entropy(self.logits(f, r, k), t, reduction="mean")

    def recalled(self, f: dict[str, mx.array], rows: list[Rows]) -> bool:
        """Every held-out answer token above RECALL probability under the trial factors."""
        for r, k, t in rows:
            p = mx.softmax(self.logits(f, r, k), axis=-1)
            if not bool(mx.all(mx.take_along_axis(p, t[:, None], axis=-1) > RECALL)):
                return False
        return True

    def install(self, delta: mx.array):
        se = layers(self.model)[self.last].mlp.shared_expert
        base = self.master
        if base is None:
            lin = se.down_proj
            base = lin.weight
            if isinstance(lin, nn.QuantizedLinear):
                base = mx.dequantize(lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits, mode=lin.mode)
        master = base.astype(mx.float32) + delta
        lin = nn.Linear(master.shape[1], master.shape[0], bias=False)
        lin.weight = master.astype(mx.bfloat16)
        mx.eval(lin.weight)
        se.down_proj = lin
        self.master = master

    def weights(self) -> dict[str, mx.array]:
        return {} if self.master is None else {PATH.format(self.last): self.master.astype(mx.bfloat16)}

    def learn(self, fact: str, ps: list[tuple[str, str]]) -> str:
        """Fit rank-limited factors on the fact's rows, near misses, replay and every keep row; install B A."""
        system = SYSTEM.format(fact)
        lines = ask(self.model, self.tok, NEAR.format(n=self.near), system, max_tokens=32 * self.near).splitlines()
        near = [q.strip(" -*.)0123456789") for q in lines if q.strip().endswith("?")][: self.near]
        teach = [(q, ask(self.model, self.tok, q, system, max_tokens=40)) for q in near]
        near_rows = [self.capture(*answered(self.tok, q, a)) for q, a in teach]
        train = [self.capture(*answered(self.tok, q, a)) for q, a in ps[:-2]]
        held = [self.capture(*answered(self.tok, q, a)) for q, a in ps[-2:]]
        width, out = train[0][1].shape[-1], train[0][0].shape[-1]
        f = {"a": mx.random.normal((self.rank, width)) * width**-0.5, "b": mx.zeros((out, self.rank))}
        opt = optim.Adam(learning_rate=self.lr)
        grad = mx.value_and_grad(self.loss)
        value, taken = 0.0, 0
        for taken in range(1, self.steps + 1):
            rows = train + near_rows + random.sample(self.replay, min(8, len(self.replay))) + self.keep
            v, g = grad(f, rows)
            f = opt.apply_gradients(g, f)
            mx.eval(f, opt.state)
            value = float(v)
            if taken % 10 == 0 and self.recalled(f, held):
                break
        delta = f["b"] @ f["a"]
        self.install(delta)
        self.replay += train + held
        return f"{taken} steps, loss {value:.3f}, |B A| {float(mx.linalg.norm(delta)):.3f}"
