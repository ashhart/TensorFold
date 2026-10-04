"""Qwen3.6's MTP layer on MLX: the token after next from the target's final normed row and the next token."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

MTP_FILE = "mtp-4bit.safetensors"      # beside the checkpoint (TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP ships it)


class MTPCache(KVCache):
    """Absorbed rows' keys and values; a chain's rows wait in ``side`` until the next absorb drops them."""

    transient = ("side", "speculation")      # a round's own state: snapshots leave it out
    side: Any = None                         # (keys, values) of chained rows: a write into held buffers copies them
    speculation: Any = None                  # (outputs, rows, last_only) of the stream's last ``speculate``

    @property
    def drafted(self) -> int:
        return 0 if self.side is None else int(self.side[0].shape[2])


class Qwen36MTP(nn.Module):
    """[norm(embedding of token t + 1) | norm(row t)] through fc, one attention layer with routed experts, a norm."""

    def __init__(self, args: Any) -> None:
        super().__init__()
        from mlx_lm.models.qwen3_5 import DecoderLayer

        d, eps = int(args.hidden_size), float(args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(d, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(d, eps=eps)
        self.fc = nn.Linear(2 * d, d, bias=False)
        # interval 1: the layer is full attention, with the target's sparse MoE block
        self.layers = [DecoderLayer(dataclasses.replace(args, full_attention_interval=1), 0)]
        self.norm = nn.RMSNorm(d, eps=eps)

    def make_cache(self) -> MTPCache:
        return MTPCache()

    def inputs(self, hidden: mx.array, embeddings: mx.array) -> mx.array:
        """fc over [norm(embeddings) | norm(hidden)], both [1, R, D]."""

        return self.fc(mx.concatenate([self.pre_fc_norm_embedding(embeddings), self.pre_fc_norm_hidden(hidden)],
                                      axis=-1))

    def qkv(self, x: mx.array, positions: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Queries [1, H, R, Dh], keys and values [1, Hkv, R, Dh] at ``positions`` [R], output gate [1, R, H * Dh]."""

        attn = self.layers[0].self_attn
        h = self.layers[0].input_layernorm(x)
        rows = int(h.shape[1])
        queries, gate = mx.split(attn.q_proj(h).reshape(1, rows, attn.num_attention_heads, -1), 2, axis=-1)
        keys = attn.k_norm(attn.k_proj(h).reshape(1, rows, attn.num_key_value_heads, -1))
        values = attn.v_proj(h).reshape(1, rows, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        # each row is its own batch entry for RoPE: rows of several streams sit at their own positions
        queries = attn.rope(attn.q_norm(queries).transpose(1, 2, 0, 3), offset=positions).transpose(2, 1, 0, 3)
        keys = attn.rope(keys.transpose(1, 2, 0, 3), offset=positions).transpose(2, 1, 0, 3)
        return queries, keys, values, gate.reshape(1, rows, -1)

    def finish(self, x: mx.array, attended: mx.array, gate: mx.array) -> mx.array:
        """The layer's rest after attention ([1, R, H * Dh] each) for rows ``x``: o_proj, the MoE block, the norm."""

        layer = self.layers[0]
        h = x + layer.self_attn.o_proj(attended * mx.sigmoid(gate))
        return self.norm(h + layer.mlp(layer.post_attention_layernorm(h)))


def absorb_attention(attn: Any, cache: MTPCache, queries: mx.array, keys: mx.array, values: mx.array) -> mx.array:
    """Write rows into the buffers (no chain held) and attend: [1, H, R, Dh] -> [1, R, H * Dh]."""

    cache.side = None
    keys, values = cache.update_and_fetch(keys, values)
    rows = int(queries.shape[2])
    out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale,
                                               mask="causal" if rows > 1 else None)
    return out.transpose(0, 2, 1, 3).reshape(1, rows, -1)


def chain_attention(attn: Any, cache: MTPCache, queries: mx.array, keys: mx.array, values: mx.array) -> mx.array:
    """One chained row [1, H, 1, Dh] over the buffers and the chain's rows (its own included) -> [1, 1, H * Dh]."""

    side = (keys, values) if cache.side is None else (mx.concatenate([cache.side[0], keys], axis=2),
                                                      mx.concatenate([cache.side[1], values], axis=2))
    cache.side = side
    if cache.offset:                       # a copy of the buffers' rows: they stay unwritten while a level reads them
        keys = mx.concatenate([cache.keys[..., :cache.offset, :], side[0]], axis=2)
        values = mx.concatenate([cache.values[..., :cache.offset, :], side[1]], axis=2)
    else:
        keys, values = side
    out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale)
    return out.transpose(0, 2, 1, 3).reshape(1, 1, -1)


def _formats(weights: dict[str, mx.array], module: Any, path: str) -> dict[str, int] | bool:
    """A linear's (group size, bits) read from its stored words and scales; False keeps it in floating point."""

    scales = weights.get(f"{path}.scales")
    if scales is None:
        return False
    inputs = int(module.weight.shape[-1])
    return {"group_size": inputs // int(scales.shape[-1]),
            "bits": int(weights[f"{path}.weight"].shape[-1]) * 32 // inputs}


def load(path: Path, args: Any) -> Qwen36MTP:
    """The MTP layer from ``mtp-4bit.safetensors`` (names under ``language_model.mtp.`` or bare), formats as stored."""

    raw = mx.load(str(path))
    weights = {name.split("mtp.", 1)[1] if "mtp." in name else name: value for name, value in raw.items()}
    mtp = Qwen36MTP(args)
    nn.quantize(mtp, class_predicate=lambda p, m: hasattr(m, "to_quantized") and _formats(weights, m, p))
    mtp.load_weights(list(weights.items()), strict=True)
    mx.eval(mtp.parameters())
    return mtp


def find(model_dir: Path, choice: str = "") -> Path | None:
    """The MTP file: ``choice`` ("0" turns drafting off), else the checkpoint's own side file."""

    if choice == "0":
        return None
    if choice:
        named = Path(choice).expanduser()
        if not named.is_file():
            raise FileNotFoundError(f"the MTP file {named} does not exist")
        return named
    found = Path(model_dir) / MTP_FILE
    return found if found.is_file() else None


__all__ = ["MTPCache", "MTP_FILE", "Qwen36MTP", "absorb_attention", "chain_attention", "find", "load"]
