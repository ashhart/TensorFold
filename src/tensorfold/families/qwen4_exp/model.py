"""Qwen3.8 Flash Next forward pass with host-computed n-gram hashes to avoid GPU waits for embedding rows."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.switch_layers import SwitchGLU

from tensorfold.families.qwen4_exp.host_table import ngrams_on_host
from tensorfold.families.qwen4_exp.model_layers import (
    Config,
    _derived,
    CenteredRMSNorm,
    GatedRMSNorm,
    HyperConnection,
    _queue,
    _write_back,
    LinearCache,
    AttentionCache,
    GatedDeltaNet,
    Indexer,
    SparseAttention,
    MLP,
    SparseMoE,
)

from tensorfold.kernels.qwen.flash_next.v1 import embed, prefill, prefill_hc, prefill_mm

MODEL_TYPE = "qwen4_exp"

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_PRIME = 10007


# -- n-gram embedding (PLE) ------------------------------------------------------
def _splitmix64(value: int) -> int:
    value = (value + _GOLDEN) & _MASK64
    value = ((value ^ (value >> 30)) * _MIX1) & _MASK64
    value = ((value ^ (value >> 27)) * _MIX2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def layer_multipliers(vocab: int, ngram: int, ple_index: int, seed: int) -> np.ndarray:
    half = max(1, (((1 << 63) - 1) // max(vocab, 1)) // 2)
    base = seed + _PRIME * ple_index
    return np.array([2 * (_splitmix64((base + _GOLDEN * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram)],
                    dtype=np.int64)


class NGramEmbedding(nn.Module):
    """2- and 3-gram ids hashed into 16 heads of prime-sized tables, looked up in 128 row shards."""

    def __init__(self, cfg: Config, ple_index: int) -> None:
        super().__init__()
        self.n = cfg.ngram_size
        self.context = cfg.ngram_size - 1
        self.per_ngram = cfg.heads_per_ngram
        self.heads = self.context * self.per_ngram
        self.eos = cfg.ple_eos
        sizes, offsets, total = [], [], 0
        for head in range(self.heads):
            size = _nth_prime_after(cfg.ngram_vocab_size_base - 1, ple_index * self.heads + head + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        self.head_sizes = np.array(sizes, dtype=np.int64)
        self.head_offsets = np.array(offsets, dtype=np.int64)
        self.multipliers = layer_multipliers(cfg.vocab_size, cfg.ngram_size, ple_index, cfg.seed)
        rows = math.ceil(total / cfg.ngram_vocab_divisor) * cfg.ngram_vocab_divisor
        base, extra = divmod(rows, cfg.ngram_shards)
        shard_rows = [base + (1 if i < extra else 0) for i in range(cfg.ngram_shards)]
        self.shard_starts = [0]
        for count in shard_rows:
            self.shard_starts.append(self.shard_starts[-1] + count)
        self.dims = cfg.ple_embed_dim // self.heads
        self.shards = [nn.Embedding(count, self.dims) for count in shard_rows]
        # the shards' rows on the host instead (HostTable's memory map or SSDTable's reads), set by load()
        self.host = None
        self.quant_group, self.quant_bits = cfg.group_size, cfg.bits

    def ids(self, history: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Row ids [B, L, heads] for ``tokens`` [B, L] after ``history`` [B, n-1] (EOS resets the n-grams)."""

        seq = np.concatenate([history, tokens], axis=1).astype(np.int64)
        batch, width = seq.shape
        pos = np.arange(width)
        eos_at = np.where(seq == self.eos, pos[None], -1)
        before = np.concatenate([np.full((batch, 1), -1), np.maximum.accumulate(eos_at, axis=1)[:, :-1]], axis=1)
        in_segment = pos[None] - (before + 1)
        shifted = []
        for shift in range(self.n):
            source = pos - shift
            taken = np.take_along_axis(seq, np.broadcast_to(np.maximum(source, 0)[None], seq.shape), axis=1)
            shifted.append(np.where((in_segment >= shift) & (source[None] >= 0), taken, self.eos))
        blocks = []
        for ngram in range(2, self.n + 1):
            first = (ngram - 2) * self.per_ngram
            mixed = shifted[0] * self.multipliers[0]
            for p in range(1, ngram):
                mixed = np.bitwise_xor(mixed, shifted[p] * self.multipliers[p])
            sizes = self.head_sizes[first:first + self.per_ngram]
            blocks.append(mixed[..., None] % sizes + self.head_offsets[first:first + self.per_ngram])
        return np.concatenate(blocks, axis=-1)[:, -tokens.shape[1]:]

    def __call__(self, ids: np.ndarray) -> mx.array:
        tables = self.__dict__.get("fused_tables")      # the fused decode's lookup: mx.dequantize's bits in one kernel
        if tables is not None:
            rows = embed.ple_lookup(ids.reshape(-1, ids.shape[-1]), tables)
            return rows.reshape(*ids.shape[:-1], self.heads * self.dims)
        if self.host is not None:
            words, scales, biases = self.host.gather(ids)
            rows = mx.dequantize(mx.array(words), mx.array(scales).view(mx.bfloat16),
                                 mx.array(biases).view(mx.bfloat16), group_size=self.quant_group, bits=self.quant_bits)
            return rows.reshape(*ids.shape[:-1], self.heads * self.dims)
        flat = ids.reshape(-1)
        shard = np.searchsorted(np.asarray(self.shard_starts), flat, side="right") - 1
        parts, order = [], []
        for s in np.unique(shard):
            where = np.nonzero(shard == s)[0]
            local = mx.array((flat[where] - self.shard_starts[int(s)]).astype(np.int32))
            parts.append(self.shards[int(s)](local))
            order.append(where)
        rows = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        inverse = np.empty(len(flat), dtype=np.int32)
        inverse[np.concatenate(order)] = np.arange(len(flat), dtype=np.int32)
        rows = rows[mx.array(inverse)]
        return rows.reshape(*ids.shape[:-1], self.heads * self.dims)


class PLELayer(nn.Module):
    """Adds a gated n-gram embedding to every residual stream, then a dilated short conv over it."""

    def __init__(self, cfg: Config, ple_index: int) -> None:
        super().__init__()
        self.streams, self.dims = cfg.hc_count, cfg.hidden_size
        wide = cfg.hc_count * cfg.hidden_size
        self.ple_embedding = NGramEmbedding(cfg, ple_index)
        self.key_proj = nn.Linear(cfg.ple_embed_dim, wide, bias=False)
        self.value_proj = nn.Linear(cfg.ple_embed_dim, cfg.hidden_size, bias=False)
        self.norm_key = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.norm_query = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.norm_conv = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.dilation = cfg.ngram_size
        self.tail = (cfg.ple_conv_kernel_size - 1) * self.dilation
        self.conv1d = nn.Conv1d(wide, wide, kernel_size=cfg.ple_conv_kernel_size, dilation=self.dilation,
                                groups=wide, bias=False)

    def __call__(self, h: mx.array, tokens: np.ndarray, cache: LinearCache) -> mx.array:
        batch, length, _ = h.shape
        history = cache.history
        if history is None:
            history = np.full((batch, self.ple_embedding.context), self.ple_embedding.eos, dtype=np.int64)
        ids = self.ple_embedding.ids(history, tokens)
        cache.history = np.concatenate([history, tokens.astype(np.int64)], axis=1)[:, -self.ple_embedding.context:]
        emb = self.ple_embedding(ids)
        shape = (batch, length, self.streams, self.dims)
        keys = self.norm_key(self.key_proj(emb)).reshape(shape)
        values = self.value_proj(emb)
        queries = self.norm_query(h).reshape(shape)
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(self.dims)
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = (mx.sigmoid(gate) * values[..., None, :]).reshape(h.shape)
        normed = self.norm_conv(gated)
        tail = cache.ple_conv if cache.ple_conv is not None else mx.zeros((batch, self.tail, h.shape[-1]), h.dtype)
        conv_in = mx.concatenate([tail, normed], axis=1)
        cache.ple_conv = mx.contiguous(conv_in[:, -self.tail:])    # a copy: a view would keep the chunk
        return gated + nn.silu(self.conv1d(conv_in))


# -- model -----------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config, index: int) -> None:
        super().__init__()
        self.is_linear = cfg.layer_types[index] == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = SparseAttention(cfg)
        self.mlp = SparseMoE(cfg)
        if index + 1 in cfg.ple_layer_ids:
            self.ple = PLELayer(cfg, cfg.ple_layer_ids.index(index + 1))
        self.attn_hyper_connection = HyperConnection(cfg)
        self.mlp_hyper_connection = HyperConnection(cfg)

    def __call__(self, h: mx.array, tokens: np.ndarray, cache: Any) -> mx.array:
        if "ple" in self:
            h = h + self.ple(h, tokens, cache)
        mixed, inject = self.attn_hyper_connection(h)
        branch = self.linear_attn(mixed, cache) if self.is_linear else self.self_attn(mixed, cache)
        h = _write_back(h, branch, inject)
        mixed, inject = self.mlp_hyper_connection(h)
        return _write_back(h, self.mlp(mixed), inject)


def select_by_kernels(layers: list[Any]) -> None:
    """Prompt chunks past the dense range attend through the decode's selection and attention kernels (Metal)."""

    for layer in layers:
        if "self_attn" in layer:
            layer.self_attn.__dict__["kernel_select"] = True


class Body(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.hyper_connection_mixer = HyperConnection(cfg, combine=False)


class Qwen4Exp(nn.Module):
    # Queue at most two layer slices while preserving the graph and output bits.
    pipeline_layers = 1
    # decode steps of up to this many rows go through ``decode.FusedDecode`` when it is attached
    fused_rows = 16

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.args = cfg
        self.model = Body(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        return [LinearCache() if layer.is_linear else AttentionCache() for layer in self.layers]

    def hidden(self, inputs: Any, cache: list[Any]) -> mx.array:
        """The mixed hidden state [B, L, D] after the last layer."""

        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        fused = self.__dict__.get("fused")
        if fused is not None and tokens.shape[0] == 1 and tokens.shape[1] <= self.fused_rows:
            return fused(tokens, cache)
        if fused is not None and tokens.shape[0] == 1 and prefill_mm.fast_prefill():
            return prefill_hc.hidden(self, tokens, cache)           # a prompt chunk through the prefill path
        h = self.model.embed_tokens(mx.array(tokens.astype(np.int32)))
        h = mx.tile(h, (1, 1, self.args.hc_count))
        queued = None
        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, tokens, layer_cache)
            if self.pipeline_layers and (i + 1) % self.pipeline_layers == 0:
                _queue(h, queued)
                queued = h
        self.__dict__["last_streams"] = h[0]          # [L, S*D]: the residual streams before the final mixer
        return self.model.hyper_connection_mixer(h)

    def hidden_pass(self, inputs: Any, cache: list[Any], sizes: Any) -> mx.array:
        """Consecutive prompt chunks (``sizes`` rows each) in one forward, every chunk with its own forward's bits."""

        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        if len(tuple(sizes)) == 1 or self.__dict__.get("fused") is None or not prefill_mm.fast_prefill():
            raise ValueError("hidden_pass: several chunks on the prefill path (fused model, Metal) only")
        return prefill_hc.hidden_pass(self, tokens, cache, sizes)

    def head(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))


# Checkpoint hashing constants must equal those derived by NGramEmbedding.
_PLE_CONSTANTS = {
    "layer_multipliers": "multipliers",
    "ngram_heads_vocab_sizes": "head_sizes",
    "ngram_heads_offsets": "head_offsets",
}


def sanitize(weights: dict[str, mx.array]) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """Checkpoint names -> this module's; the n-gram hashing constants come back separately (not weights)."""

    out: dict[str, mx.array] = {}
    extras: dict[str, mx.array] = {}
    for name, value in weights.items():
        if not name.startswith("language_model.") or ".mtp." in name or name.startswith("language_model.mtp"):
            continue
        key = name[len("language_model."):]
        if key.rsplit(".", 1)[-1] in _PLE_CONSTANTS:
            extras[key] = value
            continue
        if key.endswith("ngram_embedding.weight_scale"):       # an FP8 conversion's table scale: 1 when quantized
            if not bool(mx.all(value.astype(mx.float32) == 1.0).item()):
                raise ValueError(f"{name}: an n-gram table scale other than 1 is not supported")
            continue
        key = key.replace("ngram_embedding.shard_", "shards.").replace("ngram_embedding.shards.", "shards.")
        out[key] = value
    return out, extras


def quant_params(config: dict[str, Any], path: str) -> dict[str, Any] | bool:
    """nn.quantize's parameters for the module at ``path`` from config.json's per-module entries (MLX's rules)."""

    from tensorfold.quantization import resolve_affine

    spec = resolve_affine(config, path.replace(".ple_embedding.shards.", ".ple_embedding.ngram_embedding.shards."))
    return False if spec is None else {"group_size": spec.group_size, "bits": spec.bits, "mode": spec.mode}


def norms_stored_around_one(weights: dict[str, mx.array]) -> bool:
    """Detect whether the checkpoint stores gamma or gamma - 1 from attention hyper-connection norm means."""

    anchors = [value for key, value in weights.items()
               if key.startswith("model.layers.") and key.endswith(".attn_hyper_connection.hc_norm.weight")]
    if len(anchors) < 8:
        return False
    means = np.array([float(mx.mean(a.astype(mx.float32)).item()) for a in anchors])
    around_one = (means > 0.5).mean() >= 0.9 and 0.75 <= float(np.median(means)) <= 1.5
    around_zero = (means > 0.5).mean() <= 0.1 and -0.5 <= float(np.median(means)) <= 0.25
    if not (around_one or around_zero):
        raise ValueError(f"cannot tell how the norm weights are stored (median mean {np.median(means):.3f})")
    return around_one


def prefetch_ngrams(model: Qwen4Exp) -> None:
    """Prefetch host n-gram pages after the first forwards to avoid displacing weights during loading."""

    for _, module in model.named_modules():
        if isinstance(module, NGramEmbedding) and module.host is not None:
            module.host.prefetch()


def load(model_dir: Path, *, lazy: bool = False, ple_on_ssd: bool = False,
         ssd_experts: float | None = None) -> tuple[Qwen4Exp, Any]:
    from tensorfold.families.tokenizer import load_tokenizer

    on_host = ngrams_on_host(model_dir, ple_on_ssd)
    config = json.loads((Path(model_dir) / "config.json").read_text())
    cfg = Config.from_dict(config)
    model = Qwen4Exp(cfg)
    weights: dict[str, mx.array] = {}
    # Load on the CPU stream before GPU use so file reads cannot stall a GPU command buffer past its watchdog.
    for path in sorted(Path(model_dir).glob("model*.safetensors")):
        weights.update(mx.load(str(path), stream=mx.cpu))
    weights, extras = sanitize(weights)
    quantized_paths = {k[:-len(".scales")] for k in weights if k.endswith(".scales")}
    if ssd_experts:
        from tensorfold.families.qwen4_exp import stream

        weights = {k: v for k, v in weights.items() if not stream.switch_keys(k)}    # read into the pool instead
    for path, emb in [(p, m) for p, m in model.named_modules() if isinstance(m, NGramEmbedding)]:
        spec = quant_params(config, f"{path}.shards.0")                     # every shard shares one format
        if spec:
            emb.quant_bits, emb.quant_group = spec["bits"], spec["group_size"]
    if on_host:
        from tensorfold.families.qwen4_exp import host_table

        for path, emb in [(p, m) for p, m in model.named_modules() if isinstance(m, NGramEmbedding)]:
            emb.shards = []
            emb.host = host_table.ReadAhead(host_table.from_checkpoint(
                model_dir, f"language_model.{path}.ngram_embedding", len(emb.shard_starts) - 1, ssd=ple_on_ssd))
            if emb.host.rows != emb.shard_starts[-1]:
                raise ValueError(f"{path}: n-gram tables hold {emb.host.rows} rows, expected {emb.shard_starts[-1]}")
        weights = {k: v for k, v in weights.items() if ".ple_embedding.shards." not in k}
    if not lazy:
        mx.eval(list(weights.values()))

    def quantized(path: str, module: nn.Module) -> dict[str, Any] | bool:
        return hasattr(module, "to_quantized") and path in quantized_paths and quant_params(config, path)

    nn.quantize(model, group_size=cfg.group_size, bits=cfg.bits, class_predicate=quantized)
    if norms_stored_around_one(weights):
        # Convert stored gamma to the gamma - 1 expected by centered norms.
        for path, module in model.named_modules():
            if isinstance(module, CenteredRMSNorm) and f"{path}.weight" in weights:
                weights[f"{path}.weight"] = weights[f"{path}.weight"].astype(mx.float32) - 1.0
    if ssd_experts:
        from tensorfold.families.qwen4_exp import stream

        stream.attach(model, Path(model_dir), float(ssd_experts))      # placeholders replace the routed stacks
        expected = {k for k, _ in tree_flatten(model.parameters()) if not stream.switch_keys(k)}
        missing = expected - set(weights)
        if missing:
            raise ValueError(f"checkpoint lacks {sorted(missing)[:3]}")
        model.load_weights(list(weights.items()), strict=False)
    else:
        model.load_weights(list(weights.items()), strict=True)
    for key, value in extras.items():
        embedding = model.layers[int(key.split(".")[2])].ple.ple_embedding
        derived = getattr(embedding, _PLE_CONSTANTS[key.rsplit(".", 1)[-1]])
        shipped = np.array(value).astype(np.int64)
        if not np.array_equal(shipped, derived):
            raise ValueError(f"{key}: checkpoint {shipped} != derived {derived}")
    if not lazy:
        mx.eval(model.parameters())
        if os.environ.get("TF_FLASH_FUSED", "1") != "0":
            from tensorfold.families.qwen4_exp.decode import FusedDecode

            # kept out of the module tree (a plain attribute), so parameters() stays the checkpoint's
            model.__dict__["fused"] = FusedDecode(model)
            select_by_kernels(model.layers)
    eos = config.get("eos_token_id")
    tokenizer = load_tokenizer(Path(model_dir), eos_token_ids=eos if isinstance(eos, list) else None)
    return model, tokenizer
