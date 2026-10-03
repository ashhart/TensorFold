"""Laguna's decode forward over rows of one or more streams, each row with a one-row step's bits."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx
import numpy as np

from tensorfold.kernels.inputs import ints
from tensorfold.kernels.laguna.v1.attention import Rows, attend
from tensorfold.kernels.laguna.v1.glue import expert_down, expert_gateup, qkv_prep, route, router_logits
from tensorfold.kernels.laguna.v1.matmul import Projection, Projections


def rope_table(rope: Any, head_dim: int) -> tuple[int, mx.array, float]:
    """(rotated dims, each pair's inverse frequency [rotated / 2] float32, magnitude scale) of mlx_lm's RoPE module."""

    dims = int(rope.dims)
    if getattr(rope, "traditional", False):
        raise ValueError("Laguna's decode RoPE covers non-traditional (half-split) rotation")
    half = dims // 2
    freqs = getattr(rope, "_freqs", None)
    if freqs is not None:
        inv = 1.0 / np.array(freqs.astype(mx.float32), dtype=np.float64)
    else:
        if float(getattr(rope, "scale", 1.0)) != 1.0:
            raise ValueError("Laguna's decode RoPE covers unscaled positions")
        inv = np.exp2(-(np.arange(half, dtype=np.float64) / half) * np.log2(float(rope.base)))
    if inv.shape != (half,) or dims > head_dim or dims % 64:
        raise ValueError(f"RoPE over {dims} of {head_dim} dims with {inv.shape[0]} frequencies")
    return dims, mx.array(inv.astype(np.float32)), float(getattr(rope, "mscale", 1.0))


class RowDecode:
    """Laguna (gated GQA, sliding and full layers, sigmoid-routed MoE with a shared expert) through these kernels."""

    # layers per slice handed to the GPU while the rest of the forward is built
    eval_every = 8
    # a draft model's taps: (layer ids, the list it reads), each layer's output rows [1, N, D] written there
    taps: tuple[tuple[int, ...], list[Any]] | None = None

    def __init__(self, text_model: Any, backend: str, head_backend: str | None = None) -> None:
        args = text_model.args
        self.backbone = text_model.model
        self.layers = self.backbone.layers
        self.window = int(args.sliding_window or 0)
        self.eps_value = float(args.rms_norm_eps)
        self.eps = mx.array([self.eps_value], dtype=mx.float32)
        self.top_k = int(args.num_experts_per_tok)
        self.scaling = float(args.moe_routed_scaling_factor)
        self.softcap = float(args.moe_router_logit_softcapping or 0.0)
        self.backend = backend
        self.qkv, self.g, self.o, self.ropes = [], [], [], []
        self.gate_up, self.down, self.router = [], [], []
        for layer in self.layers:
            attn = layer.self_attn
            if not attn.gating:
                raise ValueError("Laguna's decode kernels cover the gated attention (per-head softplus gate)")
            self.qkv.append(Projections([attn.q_proj, attn.k_proj, attn.v_proj], backend))
            self.g.append(Projection([attn.g_proj], backend))
            self.o.append(Projection([attn.o_proj], backend))
            self.ropes.append(rope_table(attn.rope, attn.head_dim))
            mlp = layer.mlp
            if layer.sparse:
                for linear in (mlp.switch_mlp.gate_proj, mlp.switch_mlp.up_proj, mlp.switch_mlp.down_proj):
                    if getattr(linear, "bits", None) != 4:
                        raise ValueError("Laguna's expert kernels read 4-bit routed experts")
                shared = mlp.shared_expert
                self.gate_up.append(Projections([shared.gate_proj, shared.up_proj], backend))
                self.down.append(Projection([shared.down_proj], backend))
                proj = mlp.gate.proj
                if hasattr(proj, "scales"):
                    self.router.append(Projection([proj], backend))
                elif proj.weight.dtype == mx.bfloat16:
                    self.router.append(proj.weight)
                else:
                    raise ValueError("Laguna's router: bf16 or MLX affine weights")
            else:
                self.gate_up.append(Projections([mlp.gate_proj, mlp.up_proj], backend))
                self.down.append(Projection([mlp.down_proj], backend))
                self.router.append(None)
        mx.eval([r[1] for r in self.ropes])
        head = text_model.model.embed_tokens if text_model.tie_word_embeddings else text_model.lm_head
        self.head = Projection([head], head_backend or backend)
        self._fronts: dict[int, Any] = {}
        self._backs: dict[int, Any] = {}

    @property
    def backends(self) -> dict[str, int]:
        """How many decode matmuls run on each kernel."""

        counts: dict[str, int] = {}
        for item in [*self.qkv, *self.gate_up]:
            for name in item.backends:
                counts[name] = counts.get(name, 0) + 1
        for item in [*self.g, *self.o, *self.down, *(r for r in self.router if isinstance(r, Projection)), self.head]:
            counts[item.backend] = counts.get(item.backend, 0) + 1
        return counts

    def sliding(self, index: int) -> bool:
        return self.layers[index].self_attn.is_sliding

    def logits(self, hidden: mx.array) -> mx.array:
        shape = hidden.shape
        return self.head(hidden.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)

    def __call__(self, tokens: mx.array, streams: Sequence[tuple[list[Any], int, int]]) -> mx.array:
        """Final-normed rows [N, D] of ``streams`` (caches, rows, first position), each advancing its own caches."""

        tokens = tokens.reshape(-1)
        rows = int(tokens.shape[0])
        if rows != sum(n for _, n, _ in streams):
            raise ValueError("RowDecode: the streams' rows do not add up to the tokens")
        positions = [p + r for _, n, p in streams for r in range(n)]
        at = ints(positions)
        kinds = {}                                      # (stream, sliding) -> its rows' attention inputs
        start = 0
        for s, (caches, n, _) in enumerate(streams):
            for sliding in (True, False):
                layer = next((i for i in range(len(self.layers)) if self.sliding(i) == sliding), None)
                if layer is not None:
                    kinds[s, sliding] = Rows(positions[start:start + n], self.window if sliding else 0,
                                             caches[layer].ring, self.layers[layer].self_attn.head_dim)
            start += n
        h = self.backbone.embed_tokens(tokens)
        normed = mx.fast.rms_norm(h, self.layers[0].input_layernorm.weight, self.eps_value)
        for i in range(len(self.layers)):
            q, k, v, gate = self._front(i)(normed, at)
            outs, start = [], 0
            scale = float(self.layers[i].self_attn.scale)
            for s, (caches, n, _) in enumerate(streams):
                ks, vs = (k, v) if n == rows else (k[:, start:start + n], v[:, start:start + n])
                keys, values = caches[i].write(ks[None], vs[None])     # the buffers holding every earlier key
                outs.append(attend(q[start:start + n], keys, values, kinds[s, self.sliding(i)], ks, vs, scale))
                start += n
            out = outs[0] if len(outs) == 1 else mx.concatenate(outs)
            h, normed = self._back(i)(out, gate, h)
            if self.taps is not None and i in self.taps[0]:
                self.taps[1][self.taps[0].index(i)] = h[None]
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.async_eval(normed)
        return normed

    def _front(self, index: int) -> Any:
        """The layer's q|k|v and gate projections, head norms and RoPE, compiled."""

        fn = self._fronts.get(index)
        if fn is None:
            attn, proj, gproj = self.layers[index].self_attn, self.qkv[index], self.g[index]
            rotated, inv, mscale = self.ropes[index]
            shape = dict(heads=attn.n_heads, kv_heads=attn.n_kv_heads, head_dim=attn.head_dim, rotated=rotated,
                         mscale=mscale)

            def front(x: mx.array, positions: mx.array) -> tuple[mx.array, ...]:
                qkv = proj.joined(x)
                q, k, v = qkv_prep(qkv, attn.q_norm.weight, attn.k_norm.weight, inv, positions, self.eps, **shape)
                gate = mx.logaddexp(gproj(x).astype(mx.float32), 0.0).astype(mx.bfloat16)   # nn.softplus
                return q, k, v, gate

            fn = self._fronts[index] = mx.compile(front)
        return fn

    def _back(self, index: int) -> Any:
        """The layer from the attention output to the next layer's normed input, compiled."""

        fn = self._backs.get(index)
        if fn is None:
            from mlx_lm.models.activations import swiglu

            layer = self.layers[index]
            attn = layer.self_attn
            nxt = (self.layers[index + 1].input_layernorm.weight if index + 1 < len(self.layers)
                   else self.backbone.norm.weight)
            o, gate_up, down, router = self.o[index], self.gate_up[index], self.down[index], self.router[index]
            heads, dims = attn.n_heads, attn.head_dim
            mlp, eps = layer.mlp, self.eps_value

            def back(out: mx.array, gate: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
                rows = out.shape[0]
                gated = (out * gate[..., None]).reshape(rows, heads * dims)
                h = h + o(gated)
                n = mx.fast.rms_norm(h, layer.post_attention_layernorm.weight, eps)
                g, u = gate_up(n)
                y = down(swiglu(g, u))
                if layer.sparse:
                    logits = router(n) if isinstance(router, Projection) else router_logits(n, router)
                    ids, weights = route(logits, mlp.gate.e_score_correction_bias, self.top_k, self.softcap)
                    experts = mlp.switch_mlp
                    act = expert_gateup(n, ids, self.top_k, experts.gate_proj, experts.up_proj)
                    routed = expert_down(act, ids, weights, self.top_k, experts.down_proj)
                    y = routed * self.scaling + y
                h = h + y
                return h, mx.fast.rms_norm(h, nxt, eps)

            fn = self._backs[index] = mx.compile(back)
        return fn


__all__ = ["RowDecode", "rope_table"]
