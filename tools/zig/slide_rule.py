"""The sliding-weights rule on the last layer's shared-expert down projection, with an exact end-to-end gradient."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from slide_model import ask, chat_ids, layers, trace
from slide_probes import KEEP, near_misses

PATH = "language_model.model.layers.{}.mlp.shared_expert.down_proj"
RECALL = 0.5
Seq = tuple[list[int], int]


def answered(tok, q: str, a: str) -> Seq:
    """Token ids of a question with its answer and end token, and where the answer starts."""
    return chat_ids(tok, q, None, a + "<|im_end|>"), len(chat_ids(tok, q))


class Slider:
    """W <- clip(W - rate / max(|G|, 1) * G, W0 - bound, W0 + bound): a bounded step, G from the answers."""

    def __init__(self, model, tok, rate: float = 0.1, bound: float = 0.01, steps: int = 60, near: int = 4):
        self.model, self.tok, self.rate, self.bound, self.steps, self.near = model, tok, rate, bound, steps, near
        self.last = len(layers(model)) - 1
        lin = layers(model)[self.last].mlp.shared_expert.down_proj
        w = lin.weight
        if isinstance(lin, nn.QuantizedLinear):
            w = mx.dequantize(lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits, mode=lin.mode)
        self.anchor = w.astype(mx.float32)
        self.weight = self.anchor
        self.keep = [answered(tok, p, ask(model, tok, p, max_tokens=48)) for p in KEEP]
        self.replay: list[Seq] = []

    def capture(self, seqs: list[Seq]):
        """Final residual, gated keys and answer tokens at the rows that predict answers, under the current weight."""
        out = []
        for ids, start in seqs:
            resid, keys = trace(self.model, ids, (self.last,))
            rows = slice(start - 1, len(ids) - 1)
            out.append((resid[-1][rows].astype(mx.float32), keys[self.last][rows].astype(mx.float32), mx.array(ids[start:])))
        mx.eval(out)
        r, k, t = (mx.concatenate(part) for part in zip(*out))
        return r, k, t

    def logits(self, w: mx.array, r: mx.array, k: mx.array) -> mx.array:
        lm = self.model.language_model
        return lm.lm_head(lm.model.norm((r + k @ (w - self.weight).T).astype(mx.bfloat16))).astype(mx.float32)

    def loss(self, w: mx.array, r: mx.array, k: mx.array, t: mx.array) -> mx.array:
        return nn.losses.cross_entropy(self.logits(w, r, k), t, reduction="mean")

    def recalled(self, w: mx.array, r: mx.array, k: mx.array, t: mx.array) -> bool:
        p = mx.take_along_axis(mx.softmax(self.logits(w, r, k), axis=-1), t[:, None], axis=-1)
        return bool(mx.all(p > RECALL))

    def install(self, w: mx.array):
        lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
        lin.weight = w.astype(mx.bfloat16)
        mx.eval(lin.weight)
        layers(self.model)[self.last].mlp.shared_expert.down_proj = lin
        self.weight = w

    def weights(self) -> dict[str, mx.array]:
        return {PATH.format(self.last): self.weight.astype(mx.bfloat16)}

    def learn(self, fact: str, ps: list[tuple[str, str]]) -> str:
        """Bounded steps on the fact's answers, near misses, earlier facts and keep rows, until held-out recall."""
        train = [answered(self.tok, q, a) for q, a in ps[:-2]]
        held = self.capture([answered(self.tok, q, a) for q, a in ps[-2:]])
        near = [answered(self.tok, q, a) for q, a in near_misses(self.model, self.tok, fact, self.near)]
        r, k, t = self.capture(train + near + self.replay + self.keep)
        grad = mx.value_and_grad(self.loss)
        w, taken, value = self.weight, 0, 0.0
        for taken in range(1, self.steps + 1):
            v, g = grad(w, r, k, t)
            step = self.rate / mx.maximum(mx.sqrt(mx.sum(g * g)), 1.0)
            w = mx.clip(w - step * g, self.anchor - self.bound, self.anchor + self.bound)
            mx.eval(w)
            if not bool(mx.all(mx.isfinite(w))):
                return "refused: a step went nonfinite, weights unchanged"
            value = float(v)
            if taken % 5 == 0 and self.recalled(w, *held):
                break
        self.install(w)
        self.replay += train
        drift = float(mx.max(mx.abs(w - self.anchor)))
        return f"{taken} steps, loss {value:.3f}, drift {drift:.4f}"
