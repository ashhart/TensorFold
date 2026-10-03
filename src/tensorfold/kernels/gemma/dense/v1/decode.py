"""Dense Gemma 4 decode (the 12B unified checkpoints) over rows of one or more streams, one-row bits."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels.gemma.dense.v1.glue import attn_tail, mlp_tail
from tensorfold.kernels.gemma.dense.v1.matmul import Projection
from tensorfold.kernels.gemma.v1.attention import Rows, attend
from tensorfold.kernels.gemma.v1.decode import inverse_frequencies
from tensorfold.kernels.gemma.v1.glue import qkv_prep
from tensorfold.kernels.inputs import ints


def covers(args: Any) -> list[str]:
    """What a text config has that these kernels do not read (empty: covered)."""

    missing = []
    if getattr(args, "enable_moe_block", False):
        missing.append("a dense MLP in every layer (not the MoE block)")
    if int(getattr(args, "hidden_size_per_layer_input", 0) or 0):
        missing.append("no per-layer inputs")
    if int(getattr(args, "num_kv_shared_layers", 0) or 0):
        missing.append("no shared-KV layers")
    if getattr(args, "use_double_wide_mlp", False) and int(getattr(args, "num_kv_shared_layers", 0) or 0):
        missing.append("no double-wide MLP")
    for key in ("head_dim", "global_head_dim"):
        if int(getattr(args, key, 0) or 64) % 64:
            missing.append(f"{key} a multiple of 64")
    return missing


KINDS = ("qkv", "o", "gate_up", "down", "head")


def backends(backend: Any, head_backend: str | None = None) -> dict[str, str]:
    """Each projection kind's matmul: one backend for all, or "kind=backend,..." (unnamed kinds: "rows")."""

    if isinstance(backend, dict):
        chosen = {kind: str(backend.get(kind, "rows")) for kind in KINDS}
    elif "=" in str(backend):
        named = dict(part.split("=", 1) for part in str(backend).split(",") if part)
        unknown = set(named) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown projection kinds {sorted(unknown)}; kinds are {KINDS}")
        chosen = {kind: named.get(kind, "rows") for kind in KINDS}
    else:
        chosen = {kind: str(backend) for kind in KINDS}
    if head_backend:
        chosen["head"] = str(head_backend)
    return chosen


def describe(chosen: dict[str, str]) -> str:
    kinds = set(chosen.values())
    return next(iter(kinds)) if len(kinds) == 1 else ",".join(f"{k}={chosen[k]}" for k in KINDS)


class DenseDecode:
    """Every decode row of a dense Gemma 4 text model through this package's kernels (mlx_lm's module layout)."""

    # layers per slice handed to the GPU while the rest of the forward is built
    eval_every = 8
    taps = None

    def __init__(self, text_model: Any, backend: str, head_backend: str | None = None) -> None:
        args = text_model.args
        missing = covers(args)
        if missing:
            raise ValueError("Gemma's dense decode kernels need " + ", ".join(missing))
        self.backbone = text_model.model
        self.layers = self.backbone.layers
        self.window = int(args.sliding_window)
        self.eps_value = float(args.rms_norm_eps)
        self.eps = mx.array([self.eps_value], dtype=mx.float32)
        self.backends = backends(backend, head_backend)
        self.backend = describe(self.backends)
        pick = self.backends.get
        self.qkv, self.o, self.gate_up, self.down, self.inv_freq = [], [], [], [], []
        for layer in self.layers:
            attn = layer.self_attn
            self.qkv.append(Projection([attn.q_proj, attn.k_proj] + ([] if attn.use_k_eq_v else [attn.v_proj]),
                                       pick("qkv")))
            self.o.append(Projection([attn.o_proj], pick("o")))
            self.gate_up.append(Projection([layer.mlp.gate_proj, layer.mlp.up_proj], pick("gate_up")))
            self.down.append(Projection([layer.mlp.down_proj], pick("down")))
            self.inv_freq.append(inverse_frequencies(attn.rope, attn.head_dim))
        mx.eval(self.inv_freq)
        self.head = Projection([text_model.model.embed_tokens if text_model.tie_word_embeddings
                                else text_model.lm_head], pick("head"))
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
            raise ValueError("DenseDecode: the streams' rows do not add up to the tokens")
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
            o, gate_up, down = self.o[index], self.gate_up[index], self.down[index]

            def back(out: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
                hn, n_mlp = attn_tail(h, o(out), layer.post_attention_layernorm.weight,
                                      layer.pre_feedforward_layernorm.weight, self.eps)
                gate, up = gate_up.split(gate_up(n_mlp))
                return mlp_tail(hn, down(geglu(gate, up)), layer.post_feedforward_layernorm.weight,
                                layer.layer_scalar, nxt, self.eps)

            fn = self._backs[index] = mx.compile(back)
        return fn


__all__ = ["KINDS", "DenseDecode", "backends", "covers", "describe"]
