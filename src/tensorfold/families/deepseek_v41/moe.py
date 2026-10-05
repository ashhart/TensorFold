"""DeepSeek-V4.1's MoE: affine routed experts, 6 a token (sqrt-softplus noaux_tc routing), one shared."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.dense import dense
from tensorfold.families.glm5_next.linear import Q, per_row


def swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    """SiLU(gate) * up with V4.1's asymmetric clamp: up both sides, gate only from above.

    Written as the reference's rt.expert does it — fp32 ``a * sigmoid(a) * b`` in that exact op
    order — because ``nn.silu`` is not row-invariant across call contexts on CPU (joint-window vs
    serial rows differ by 1-6 bf16 ulps); separate sigmoid/multiply elementwise ops are.
    """
    if limit:
        up = mx.clip(up, -limit, limit)
        gate = mx.minimum(gate, limit)
    return gate * mx.sigmoid(gate) * up


class Expert:
    """A routed expert: w1 | w3 stacked (one call), the routing weight multiplies the activation before w2."""

    def __init__(self, gate: Q, up: Q, down: Q, limit: float) -> None:
        self.gate_up = Q.stack([gate, up])
        self.width = gate.outs
        self.down = down
        self.limit = limit

    def arrays(self) -> list[mx.array]:
        return [*self.gate_up.arrays(), *self.down.arrays()]

    def __call__(self, x: mx.array, weight: mx.array | None, rows_exact: bool) -> mx.array:
        """x [R, D]: (SiLU-clamped w1 * clamped w3) * weight, then w2 (fp32 activation, exactly the reference)."""
        gu = dense(x, self.gate_up, rows_exact).astype(mx.float32)
        act = swiglu(gu[:, :self.width], gu[:, self.width:], self.limit)
        if weight is not None:
            weight = weight.astype(mx.float32)
            act = act * (weight[:, None] if weight.ndim else weight)
        return dense(act.astype(x.dtype), self.down, rows_exact)


class Shared:
    """The one shared expert, no routing weight (added after the routed sum, in fp32)."""

    def __init__(self, gate: Q, up: Q, down: Q, limit: float) -> None:
        self.gate_up = Q.stack([gate, up])
        self.width = gate.outs
        self.down = down
        self.limit = limit

    def __call__(self, x: mx.array, rows_exact: bool) -> mx.array:
        """The shared expert: fp32 swiglu exactly like the routed ones (the reference casts w1/w3's
        outputs to fp32 before the activation; a bf16 activation rounds twice)."""
        gu = dense(x, self.gate_up, rows_exact).astype(mx.float32)
        act = swiglu(gu[:, :self.width], gu[:, self.width:], self.limit)
        return dense(act.astype(x.dtype), self.down, rows_exact)


class MoE:
    """sqrt(softplus) router scores; the top 6 by score + bias (the bias only selects), renormalized x 1.5."""

    def __init__(self, gate_w: mx.array, bias: mx.array | None, bias_vl: mx.array | None, experts: list[Expert],
                 shared: Shared, cfg: Config, topk: int | None = None) -> None:
        self.router = gate_w.astype(mx.float32)             # [E, D]; logits are x @ router.T (the reference's view)
        self.bias = None if bias is None else bias.astype(mx.float32)
        self.bias_vl = bias_vl               # loaded but dead in text-only (image spans only)
        self.experts = experts
        self.shared = shared
        self.top = cfg.num_experts_per_tok if topk is None else int(topk)
        self.scale = cfg.routed_scaling_factor
        self.norm_topk = cfg.norm_topk_prob
        self.limit = cfg.swiglu_limit
        self.count = len(experts)
        self._compiled: Any = None

    def scores(self, x: mx.array, rows_exact: bool) -> mx.array:
        """Router scores [R, E] in fp32: sqrt(softplus(logits)), each decode row with its one-row bits.

        The whole chain — matmul, logaddexp, sqrt — runs per row when ``rows_exact``: the vectorized
        elementwise path over a multi-row concat is not row-invariant on CPU (1 fp32 ulp vs the same
        row alone), which leaks into the routing weights and the expert activations.
        """
        xf = x.astype(mx.float32)

        def chain(r: mx.array) -> mx.array:
            return mx.sqrt(mx.logaddexp(r @ self.router.T, mx.array(0.0)))

        return per_row(chain, xf, rows_exact)

    def route(self, scores: mx.array) -> tuple[mx.array, mx.array]:
        """Experts [R, k] in the reference's ascending-score order and their weights."""
        picks = mx.argsort(scores + self.bias, axis=-1)[..., -self.top:]
        w = mx.take_along_axis(scores, picks, axis=-1)
        if self.norm_topk:
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        return picks, w * self.scale

    def forward(self, x: mx.array, rows_exact: bool) -> mx.array:
        """Group rows routed to one expert, then sum each row's results in its original route order."""
        idx, w = self.route(self.scores(x, rows_exact))
        rows = int(x.shape[0])
        picks = idx.tolist()
        by_expert: dict[int, list[tuple[int, int]]] = {}
        for r, row_picks in enumerate(picks):
            for j, expert in enumerate(row_picks):
                by_expert.setdefault(expert, []).append((r, j))

        contributions: list[list[mx.array | None]] = [[None] * self.top for _ in range(rows)]
        for expert, slots in by_expert.items():
            row_ids = mx.array([r for r, _ in slots], dtype=mx.int32)
            inputs = mx.take(x, row_ids, axis=0)
            weights = mx.stack([w[r, j] for r, j in slots])
            values = self.experts[expert](inputs, weights, True).astype(mx.float32)
            for i, (r, j) in enumerate(slots):
                contributions[r][j] = values[i:i + 1]

        out = []
        for r, row in enumerate(contributions):
            acc = mx.zeros_like(x[r:r + 1]).astype(mx.float32)
            for value in row:
                assert value is not None
                acc = acc + value
            out.append(acc)
        routed = mx.concatenate(out)
        return (routed + self.shared(x, rows_exact).astype(mx.float32)).astype(x.dtype)

    def __call__(self, x: mx.array, ids: mx.array, rows_exact: bool) -> mx.array:
        """Rows through the router and their experts; every decode row its own calls (exact by construction)."""
        return self.forward(x, rows_exact)

    def arrays(self) -> list[mx.array]:
        out = [self.router, *self.shared.gate_up.arrays(), *self.shared.down.arrays()]
        for e in self.experts:
            out.extend(e.arrays())
        return out + [a for a in (self.bias,) if a is not None]
