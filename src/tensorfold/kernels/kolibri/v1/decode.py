"""Kolibri 1's decode forward over rows of one or more streams, each row with a one-row step's bits."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx
from mlx_lm.models.activations import swiglu

from tensorfold.kernels.gemma.v1.attention import Rows, attend
from tensorfold.kernels.gemma.v1.decode import inverse_frequencies
from tensorfold.kernels.gemma.v1.glue import attn_tail, qkv_prep
from tensorfold.kernels.gemma.v1.matmul import Projection
from tensorfold.kernels.glm.flash.v1.kernels import matmul_rows
from tensorfold.kernels.inputs import ints
from tensorfold.kernels.kolibri.v1.moe import check_q4, expert_down, expert_gateup, route


class RowDecode:
    """Kolibri 1's MLX 4-bit checkpoints decoded through Gemma 4's row kernels and row-independent MLX ops.

    Projections are rows.qmv (or the lane matmul), attention Gemma's kernel over the ring and growing caches, the bf16
    router GLM's gemv rows (MLX's one-row bits for any row count); sigmoid routing, norms and the shared expert's SwiGLU
    are MLX ops whose rows don't see each other (checked at load). The 8-bit head runs a row at a time: MLX's batched
    kernel gives a row other bits.
    """

    # layers per slice handed to the GPU while the rest of the forward is built
    eval_every = 10

    def __init__(self, model: Any, backend: str = "rows") -> None:
        args = model.args
        self.backbone = model.model
        self.layers = self.backbone.layers
        self.window = int(args.sliding_window)
        self.eps_value = float(args.rms_norm_eps)
        self.eps = mx.array([self.eps_value], dtype=mx.float32)
        self.top_k = int(args.num_experts_per_tok)
        self.backend = backend
        self.qkv, self.o, self.shared_gate_up, self.shared_down = [], [], [], []
        self.inv_freq, self.bias = [], []
        for layer in self.layers:
            attn, mlp = layer.self_attn, layer.mlp
            for linear in (mlp.switch_mlp.gate_proj, mlp.switch_mlp.up_proj, mlp.switch_mlp.down_proj):
                check_q4(linear)
            self.qkv.append(Projection([attn.q_proj, attn.k_proj, attn.v_proj], backend))
            self.o.append(Projection([attn.o_proj], backend))
            self.shared_gate_up.append(Projection([mlp.shared_experts.gate_proj, mlp.shared_experts.up_proj], backend))
            self.shared_down.append(Projection([mlp.shared_experts.down_proj], backend))
            # full-attention layers carry no position (frequency 0 rotates by nothing, bit for bit)
            half = int(args.head_dim) // 2
            self.inv_freq.append(mx.zeros((half,), dtype=mx.float32) if layer.is_full_attention
                                 else inverse_frequencies(attn.rope, int(args.head_dim)))
            self.bias.append(mlp.expert_bias.astype(mx.float32))
        mx.eval(self.inv_freq, self.bias)
        self.head_dim, self.heads, self.kv_heads = int(args.head_dim), int(args.num_attention_heads), \
            int(args.num_key_value_heads)
        self.lm_head = model.lm_head
        self._fronts: dict[int, Any] = {}
        self._backs: dict[int, Any] = {}

    def sliding(self, index: int) -> bool:
        return not self.layers[index].is_full_attention

    def logits(self, hidden: mx.array) -> mx.array:
        """The head on hidden rows [..., D], one row a call."""

        shape = hidden.shape
        rows = hidden.reshape(-1, shape[-1])
        count = int(rows.shape[0])
        out = self.lm_head(rows) if count == 1 else mx.concatenate([self.lm_head(rows[i:i + 1]) for i in range(count)])
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
                                             caches[layer].ring, self.head_dim)
            start += n
        h = self.backbone.embed_tokens(tokens)
        normed = mx.fast.rms_norm(h, self.layers[0].input_layernorm.weight, self.eps_value)
        scale = float(self.layers[0].self_attn.scale)
        for i in range(len(self.layers)):
            q, k, v = self._front(i)(normed, at)
            outs, start = [], 0
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
        """The layer's q|k|v projection, q and k head norms and RoPE (none on full-attention layers), v as projected."""

        fn = self._fronts.get(index)
        if fn is None:
            attn, proj, inv = self.layers[index].self_attn, self.qkv[index], self.inv_freq[index]
            heads, kv_heads, head_dim = self.heads, self.kv_heads, self.head_dim
            v_from = (heads + kv_heads) * head_dim

            def front(x: mx.array, positions: mx.array) -> tuple[mx.array, mx.array, mx.array]:
                qkv = proj(x)
                # Gemma's prep norms v as well (its values_are_keys emits normed keys there): Kolibri's v is raw
                q, k, _ = qkv_prep(qkv, attn.q_norm.weight, attn.k_norm.weight, inv, positions, self.eps,
                                   heads=heads, kv_heads=kv_heads, head_dim=head_dim, values_are_keys=True)
                v = qkv[:, v_from:].reshape(-1, kv_heads, head_dim).transpose(1, 0, 2)
                return q, k, v

            fn = self._fronts[index] = mx.compile(front)
        return fn

    def _back(self, index: int) -> Any:
        """The layer from the attention output to the next layer's normed input, compiled."""

        fn = self._backs.get(index)
        if fn is None:
            layer = self.layers[index]
            nxt = (self.layers[index + 1].input_layernorm.weight if index + 1 < len(self.layers)
                   else self.backbone.norm.weight)
            o, gate_up, down, bias = (self.o[index], self.shared_gate_up[index], self.shared_down[index],
                                      self.bias[index])
            mlp = layer.mlp
            experts, router = mlp.switch_mlp, mlp.gate.weight
            post_ffn, eps, top_k = layer.post_ffn_norm.weight, self.eps_value, self.top_k
            w_post = layer.post_attention_layernorm.weight

            def back(out: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
                # hn = h + post_attn_norm(o), n = post_attention_layernorm(hn)
                hn, n, _, _ = attn_tail(h, o(out), layer.post_attn_norm.weight, w_post, w_post, w_post, self.eps)
                # MLX's one-row gemv bits for every row: ``n @ router.T`` picks its kernel by the row count
                ids, weights = route(matmul_rows(n, router, transposed=False).astype(mx.float32), bias, top_k)
                routed = expert_down(expert_gateup(n, ids, top_k, experts.gate_proj, experts.up_proj), ids, weights,
                                     top_k, experts.down_proj)
                gate, up = gate_up.split(gate_up(n))
                shared = down(swiglu(gate, up))
                h2 = hn + mx.fast.rms_norm(routed + shared, post_ffn, eps)
                return h2, mx.fast.rms_norm(h2, nxt, eps)

            fn = self._backs[index] = mx.compile(back)
        return fn


__all__ = ["RowDecode"]
