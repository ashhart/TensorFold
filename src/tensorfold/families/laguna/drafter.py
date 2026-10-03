"""poolside's DFlash drafter for Laguna (``DFlashLagunaForCausalLM``), run as vLLM's ``laguna_dflash`` runs it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.drafters.dflash_drafter import DFlashDrafter, _vendor, resolve_draft_path

ARCHITECTURE = "DFlashLagunaForCausalLM"


def is_laguna_drafter(path: str | Path) -> bool:
    try:
        config = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return ARCHITECTURE in (config.get("architectures") or [])


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LagunaDFlashAttention(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        dim, dh = config.hidden_size, config.head_dim
        self.n_heads, self.n_kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, dh
        self.scale = dh**-0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.is_causal = self.is_sliding if config.is_causal is None else bool(config.is_causal)
        self.q_size, self.kv_size = self.n_heads * dh, self.n_kv_heads * dh
        self.qkv_proj = nn.Linear(dim, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, dim, bias=False)
        self.g_proj = nn.Linear(dim, self.n_heads, bias=False)
        self.q_norm = nn.RMSNorm(dh, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(dh, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, x_ctx: mx.array, rope: Any, cache: Any, masks: dict) -> mx.array:
        """x: the block's normed rows [1, L, D]; x_ctx: the new context rows, normed by this layer's input norm."""

        from mlx_lm.models.base import create_causal_mask

        B, L, _ = x.shape
        S = x_ctx.shape[1]
        if self.is_sliding:
            keep = self.sliding_window - 1
            if S > keep:
                skip = S - keep
                x_ctx = x_ctx[:, skip:]
                S = x_ctx.shape[1]
                cache.offset += skip
        qkv = self.qkv_proj(x)
        kv_ctx = self.qkv_proj(x_ctx)[..., self.q_size:]
        q, k, v = qkv[..., :self.q_size], qkv[..., self.q_size:self.q_size + self.kv_size], qkv[..., -self.kv_size:]
        ck, cv = kv_ctx[..., :self.kv_size], kv_ctx[..., self.kv_size:]
        q = self.q_norm(q.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        ck = self.k_norm(ck.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        cv = cv.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = rope(q, offset=cache.offset + S)
        k = rope(k, offset=cache.offset + S)
        ck = rope(ck, offset=cache.offset)
        keys, values = cache.update_and_fetch(ck, cv)
        ctx_len = keys.shape[2]
        keys = mx.concatenate([keys, k], axis=2)
        values = mx.concatenate([values, v], axis=2)
        key = (self.is_sliding, self.is_causal, self.sliding_window, L, ctx_len)
        mask = masks.get(key, False)
        if mask is False:
            mask = create_causal_mask(L, offset=ctx_len) if self.is_causal else None
            if self.is_sliding:
                query = ctx_len + mx.arange(L)[:, None]
                at = mx.arange(ctx_len + L)[None]
                block = at >= ctx_len
                if self.is_causal:
                    block = block & (at <= query)
                mask = ((at < ctx_len) & (query - at < self.sliding_window)) | block
            masks[key] = mask
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3)                                         # [1, L, H, Dh]
        gate = mx.logaddexp(self.g_proj(x).astype(mx.float32), 0.0).astype(out.dtype)     # softplus, a head
        return self.o_proj((out * gate[..., None]).reshape(B, L, -1))


class LagunaDFlashLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LagunaDFlashAttention(config, layer_idx)
        self.mlp = _MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, h: mx.array, ctx: mx.array, rope: Any, cache: Any, masks: dict) -> mx.array:
        h = h + self.self_attn(self.input_layernorm(h), self.input_layernorm(ctx), rope, cache, masks)
        return h + self.mlp(self.post_attention_layernorm(h))


def _model_class() -> Any:
    vendor = _vendor()

    class LagunaDFlashModel(vendor.DFlashDraftModel):
        """z-lab's DFlash draft model's interface (bind, caches, head) over Laguna's draft layers."""

        layer_class = LagunaDFlashLayer

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            taps = len(config.target_layer_ids)
            self.aux_hidden_norms = [nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(taps)]

        def combine(self, taps: mx.array) -> mx.array:
            """Taps [1, S, n D] -> context rows [1, S, D]: each tap normed, then ``fc`` and ``hidden_norm``."""

            parts = mx.split(taps, len(self.aux_hidden_norms), axis=-1)
            normed = mx.concatenate([norm(p) for norm, p in zip(self.aux_hidden_norms, parts)], axis=-1)
            return self.hidden_norm(self.fc(normed))

        def hidden_states(self, inputs: mx.array, target_hidden: mx.array, cache: list[Any],
                          logits_start: int = 0) -> mx.array:
            h = self.embed_tokens(inputs) * self.embed_scale
            ctx = self.combine(target_hidden)
            masks: dict = {}
            for layer, c in zip(self.layers, cache):
                h = layer(h, ctx, self.rope, c, masks)
            if logits_start:
                h = h[:, logits_start:]
            return self.norm(h)

        def block_chain(self, drafter: Any, inputs: mx.array, context: mx.array, cache: list[Any]) -> mx.array:
            """The block after ``inputs``' anchor: each position's most likely token [1, block - 1], unread."""

            hidden = self.hidden_states(inputs, context, cache, 1)
            logits, ids = drafter.candidate_logits(hidden)
            cols = mx.argmax(logits, axis=-1)
            return cols if ids is None else mx.take(ids, cols)

    return LagunaDFlashModel


def load_draft(path: str | Path) -> Any:
    """The drafter's config and bf16 weights (``qkv_proj`` fused as stored)."""

    vendor = _vendor()
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    if ARCHITECTURE not in (cfg.get("architectures") or []):
        raise ValueError(f"{path} is not a Laguna DFlash drafter ({cfg.get('architectures')})")
    dflash = cfg.get("dflash_config") or {}
    layer_types = tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"])
    if set(layer_types) - {"full_attention", "sliding_attention"}:
        raise ValueError(f"Laguna DFlash: unsupported layer types {sorted(set(layer_types))}")
    if cfg.get("rope_parameters") or cfg.get("rope_scaling") or cfg.get("partial_rotary_factor"):
        raise ValueError("Laguna DFlash: the drafter's RoPE is expected to be the default over the whole head")
    causal = cfg.get("is_causal")
    if causal is None:
        causal = dflash.get("causal")
    config = vendor.DFlashConfig(
        hidden_size=cfg["hidden_size"], num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"], num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], intermediate_size=cfg["intermediate_size"], vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"], rope_theta=float(cfg.get("rope_theta", 10000.0)),
        max_position_embeddings=cfg["max_position_embeddings"], block_size=int(dflash.get("block_size", 16)),
        target_layer_ids=tuple(dflash["target_layer_ids"]),
        num_target_layers=int(dflash.get("num_target_layers", cfg.get("num_target_layers", 0))),
        mask_token_id=int(dflash["mask_token_id"]), rope_scaling=None, layer_types=layer_types,
        sliding_window=cfg.get("sliding_window"), is_causal=causal)
    model = _model_class()(config)
    model.eval()
    weights = {k: v for f in sorted(path.glob("*.safetensors")) for k, v in mx.load(str(f)).items()}
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model


class LagunaDrafter(DFlashDrafter):
    """``DFlashDrafter`` over poolside's Laguna draft model; drafts may use the whole vocabulary."""

    draft_vocab = ((0, 1 << 30),)

    def __init__(self, target_model: Any, draft: str, *, bits: int = 8) -> None:  # noqa: D401 - same contract
        vendor = _vendor()
        path = resolve_draft_path(draft)
        self.path = path
        self.model = load_draft(path)
        if bits:
            nn.quantize(self.model, group_size=64, bits=int(bits),
                        class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
            mx.eval(self.model.parameters())
        self.model.bind(target_model)
        vendor._patch_model(target_model, list(self.model.config.target_layer_ids))
        self.target = target_model
        self.block_size = int(self.model.config.block_size)
        self.mask_id = int(self.model.config.mask_token_id)
        window = getattr(self.model.config, "sliding_window", None)
        self.window = int(window) - 1 if window else 0
        self._trim = vendor._trim_recent_cache


__all__ = ["LagunaDrafter", "is_laguna_drafter", "load_draft"]
