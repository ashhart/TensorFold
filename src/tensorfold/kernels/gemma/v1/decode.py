"""Gemma 4's decode forward over rows of one or more streams, each row with a one-row step's bits."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx
import numpy as np

from tensorfold.kernels.gemma.v1.attention import Rows, attend
from tensorfold.kernels.gemma.v1.glue import attn_tail, moe_tail, qkv_prep, qkv_rows
from tensorfold.kernels.gemma.v1.matmul import Projection
from tensorfold.kernels.gemma.v1.moe import check_q4, expert_down, expert_gateup, route, router_logits
from tensorfold.kernels.inputs import ints


def inverse_frequencies(rope: Any, head_dim: int) -> mx.array:
    """RoPE's inverse frequency of each pair from mlx_lm's module (float32 [head_dim / 2], 0: not rotated)."""

    half = head_dim // 2
    freqs = getattr(rope, "_freqs", None)
    if freqs is not None:
        inv = 1.0 / np.array(freqs.astype(mx.float32), dtype=np.float64)
    else:
        if getattr(rope, "traditional", False) or float(getattr(rope, "scale", 1.0)) != 1.0 or rope.dims != head_dim:
            raise ValueError("Gemma's decode RoPE covers non-traditional, unscaled rotation of the whole head")
        inv = np.exp2(-(np.arange(half, dtype=np.float64) / half) * np.log2(float(rope.base)))
    if inv.shape != (half,):
        raise ValueError(f"RoPE frequencies for {inv.shape[0]} pairs, the head has {half}")
    return mx.array(inv.astype(np.float32))


class RowDecode:
    """Gemma 4's MoE checkpoints (26B-A4B) decoded through this package's kernels."""

    # layers per slice handed to the GPU while the rest of the forward is built
    eval_every = 8

    def __init__(self, text_model: Any, backend: str, head_backend: str | None = None) -> None:
        args = text_model.args
        if getattr(args, "hidden_size_per_layer_input", 0) or getattr(args, "num_kv_shared_layers", 0):
            raise ValueError("Gemma's decode kernels cover checkpoints without per-layer inputs or shared KV layers")
        if not getattr(args, "enable_moe_block", False):
            raise ValueError("Gemma's decode kernels cover the MoE checkpoints (26B-A4B)")
        self.backbone = text_model.model
        self.layers = self.backbone.layers
        self.window = int(args.sliding_window)
        self.eps_value = float(args.rms_norm_eps)
        self.eps = mx.array([self.eps_value], dtype=mx.float32)
        self.top_k = int(args.top_k_experts)
        self.backend = backend
        self.qkv, self.o, self.gate_up, self.down = [], [], [], []
        self.inv_freq, self.router_norm = [], []
        for layer in self.layers:
            attn, experts = layer.self_attn, layer.experts.switch_glu
            for linear in (experts.gate_proj, experts.up_proj, experts.down_proj):
                check_q4(linear)
            self.qkv.append(Projection([attn.q_proj, attn.k_proj] + ([] if attn.use_k_eq_v else [attn.v_proj]),
                                       backend))
            self.o.append(Projection([attn.o_proj], backend))
            self.gate_up.append(Projection([layer.mlp.gate_proj, layer.mlp.up_proj], backend))
            self.down.append(Projection([layer.mlp.down_proj], backend))
            self.inv_freq.append(inverse_frequencies(attn.rope, attn.head_dim))
            # mlx_lm norms the router input with weight scale * hidden**-0.5, computed in the scale's dtype
            self.router_norm.append(layer.router.scale * layer.router._root_size)
        mx.eval(self.inv_freq, self.router_norm)
        self.head = Projection([text_model.model.embed_tokens if text_model.tie_word_embeddings
                                else text_model.lm_head], head_backend or backend)
        self.softcap = text_model.final_logit_softcapping
        self._fronts: dict[int, Any] = {}
        self._backs: dict[int, Any] = {}

    def sliding(self, index: int) -> bool:
        return self.layers[index].self_attn.is_sliding

    def logits(self, hidden: mx.array) -> mx.array:
        """The tied head and the final soft-cap on hidden rows [..., D]."""

        from mlx_lm.models.gemma4_text import logit_softcap

        shape = hidden.shape
        out = self.head(hidden.reshape(-1, shape[-1]))
        if self.softcap is not None:
            out = logit_softcap(self.softcap, out)
        return out.reshape(*shape[:-1], -1)

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
        h = self.backbone.embed_tokens(tokens) * self.backbone.embed_scale
        normed = mx.fast.rms_norm(h, self.layers[0].input_layernorm.weight, self.eps_value)
        for i in range(len(self.layers)):
            q, k, v = self._front(i)(normed, at)
            outs, start = [], 0
            scale = float(self.layers[i].self_attn.scale)
            for s, (caches, n, _) in enumerate(streams):
                ks, vs = (k, v) if n == rows else (k[:, start:start + n], v[:, start:start + n])
                keys, values = caches[i].write(ks[None], vs[None])     # the buffers holding every earlier key
                outs.append(attend(q[start:start + n], keys, values, kinds[s, self.sliding(i)], ks, vs, scale))
                start += n
            out = outs[0] if len(outs) == 1 else mx.concatenate(outs)
            h, normed = self._back(i)(out.reshape(rows, -1), h)
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.async_eval(normed)
        return normed

    def _front(self, index: int) -> Any:
        """The layer's q|k|v projection, head norms and RoPE, compiled."""

        fn = self._fronts.get(index)
        if fn is None:
            attn, proj, inv = self.layers[index].self_attn, self.qkv[index], self.inv_freq[index]

            shape = dict(heads=attn.n_heads, kv_heads=attn.n_kv_heads, head_dim=attn.head_dim,
                         values_are_keys=attn.use_k_eq_v)

            def front(x: mx.array, positions: mx.array) -> tuple[mx.array, mx.array, mx.array]:
                if proj.backend == "rows":      # the matvec and the head norms in one kernel, rows.qmv's bits
                    return qkv_rows(x, proj.weight, proj.scales, proj.biases, proj.group, attn.q_norm.weight,
                                    attn.k_norm.weight, inv, positions, self.eps, **shape)
                return qkv_prep(proj(x), attn.q_norm.weight, attn.k_norm.weight, inv, positions, self.eps, **shape)

            fn = self._fronts[index] = mx.compile(front)
        return fn

    def _back(self, index: int) -> Any:
        """The layer from the attention output to the next layer's normed input, compiled."""

        fn = self._backs.get(index)
        if fn is None:
            from mlx_lm.models.gemma4_text import geglu

            layer = self.layers[index]
            nxt = (self.layers[index + 1].input_layernorm.weight if index + 1 < len(self.layers)
                   else self.backbone.norm.weight)
            o, gate_up, down, router_w = self.o[index], self.gate_up[index], self.down[index], self.router_norm[index]
            experts = layer.experts.switch_glu

            def back(out: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
                hn, n_mlp, n_exp, n_router = attn_tail(h, o(out), layer.post_attention_layernorm.weight,
                                                       layer.pre_feedforward_layernorm.weight,
                                                       layer.pre_feedforward_layernorm_2.weight, router_w, self.eps)
                gate, up = gate_up.split(gate_up(n_mlp))
                y1 = down(geglu(gate, up))
                ids, weights = route(router_logits(n_router, layer.router.proj), layer.router.per_expert_scale,
                                     self.top_k)
                act = expert_gateup(n_exp, ids, self.top_k, experts.gate_proj, experts.up_proj)
                y2 = expert_down(act, ids, weights, self.top_k, experts.down_proj)
                return moe_tail(hn, y1, y2, layer.post_feedforward_layernorm_1.weight,
                                layer.post_feedforward_layernorm_2.weight, layer.post_feedforward_layernorm.weight,
                                layer.layer_scalar, nxt, self.eps)

            fn = self._backs[index] = mx.compile(back)
        return fn


__all__ = ["RowDecode", "inverse_frequencies"]
