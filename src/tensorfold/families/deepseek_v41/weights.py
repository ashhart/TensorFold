"""The backbone from oMLX's converted layout, per module as ``omlx_deepseek_v41.quantized_modules`` lists it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v41.config import FORMAT_KEY, Config
from tensorfold.families.deepseek_v41.engram import Engram, EngramTable
from tensorfold.families.deepseek_v41.model import (Attention, Block, Compressor, DeepSeekV41, HC, Indexer, MoE)
from tensorfold.families.deepseek_v41.quant import Experts, Linear

PREFIX = "language_model"
MODES = {"mxfp8": (8,), "mxfp4": (4,), "affine": (2, 3, 4, 5, 6, 8)}


def formats(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Module name -> {bits, mode, group_size} from the converted checkpoint's own block."""

    spec = raw.get(FORMAT_KEY)
    if not isinstance(spec, dict) or int(spec.get("version", 0)) != 1:
        raise ValueError("deepseek_v41: this checkpoint is not in oMLX's converted layout (config.json has no "
                         f"{FORMAT_KEY} block of version 1); the Mac engine reads "
                         "Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp's "
                         "layout, not DeepSeek's FP8 release")
    out = {}
    for name, entry in (spec.get("quantized_modules") or {}).items():
        mode = str(entry.get("mode") or "affine")
        out[name] = {"bits": int(entry["bits"]), "mode": mode, "group_size": int(entry.get("group_size", 32))}
    return out


def unreadable(raw: dict[str, Any]) -> list[str]:
    """Modules stored in a format this engine does not read."""

    bad = []
    for name, f in formats(raw).items():
        if f["mode"] not in MODES or f["bits"] not in MODES[f["mode"]] or f["group_size"] not in (32, 64, 128):
            bad.append(name)
        elif name.endswith((".embed", ".head")) or ".wo_a" in name or ".compressor." in name:
            bad.append(name)             # read as bf16 (the converted checkpoint keeps them so)
    for name, spec in ((raw.get(FORMAT_KEY) or {}).get("engram_tables") or {}).items():
        if str(spec.get("mode", "affine")) != "affine":
            bad.append(name)
    return sorted(bad)


class Weights:
    """The checkpoint's tensors by name, read one shard at a time (Engram tables are never read this way)."""

    def __init__(self, model_dir: Path, raw: dict[str, Any] | None = None) -> None:
        self.dir = Path(model_dir)
        self.raw = raw if raw is not None else json.loads((self.dir / "config.json").read_text())
        self.where = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.formats = formats(self.raw)
        self.tables = dict((self.raw.get(FORMAT_KEY) or {}).get("engram_tables") or {})
        self._shards: dict[str, dict[str, mx.array]] = {}

    def has(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str) -> mx.array:
        shard = self.where[name]
        if shard not in self._shards:
            self._shards[shard] = mx.load(str(self.dir / shard))
        return self._shards[shard][name]

    def release(self) -> None:
        self._shards.clear()

    def lin(self, prefix: str) -> Linear:
        f = self.formats.get(prefix)
        if f is None:
            return Linear(self.get(f"{prefix}.weight"))
        biases = self.get(f"{prefix}.biases") if f["mode"] == "affine" else None
        return Linear(self.get(f"{prefix}.weight"), self.get(f"{prefix}.scales"), biases, bits=f["bits"],
                      group=f["group_size"], mode=f["mode"])

    def dense(self, prefix: str) -> mx.array:
        """A linear's bf16 matrix (dequantized if the checkpoint quantized it)."""

        lin = self.lin(prefix)
        return lin.dequantized().astype(mx.bfloat16)

    def experts(self, prefix: str) -> Experts:
        f = self.formats.get(prefix)
        if f is None:
            raise ValueError(f"deepseek_v41: routed experts {prefix} are not quantized; the engine reads MLX-quantized "
                             "expert stacks")
        biases = self.get(f"{prefix}.biases") if f["mode"] == "affine" else None
        return Experts(self.get(f"{prefix}.weight"), self.get(f"{prefix}.scales"), biases, bits=f["bits"],
                       group=f["group_size"], mode=f["mode"])

    def hc(self, prefix: str, kind: str, cfg: Config) -> HC:
        return HC(self.get(f"{prefix}.hc_{kind}_fn"), self.get(f"{prefix}.hc_{kind}_base"),
                  self.get(f"{prefix}.hc_{kind}_scale"), cfg)


def load_attention(w: Weights, p: str, cfg: Config, layer: int) -> Attention:
    a = f"{p}.attn"
    parts: dict[str, Any] = {n: w.lin(f"{a}.{n}") for n in ("wq_a", "wq_b", "wkv", "wo_b")}
    parts.update(wo_a=w.dense(f"{a}.wo_a"), q_norm=w.get(f"{a}.q_norm.weight"), kv_norm=w.get(f"{a}.kv_norm.weight"),
                 attn_sink=w.get(f"{a}.attn_sink"))
    ratio = cfg.ratio(layer)
    if ratio and layer in cfg.kv_source_layer_ids:
        parts["compressor"] = Compressor(w.dense(f"{a}.compressor.wkv"),
                                         w.dense(f"{a}.compressor.wgate") if ratio > 1 else None,
                                         w.get(f"{a}.compressor.norm.weight"), ratio, cfg.rms_norm_eps)
    if ratio and layer in cfg.index_source_layer_ids:
        source = layer in cfg.kv_source_layer_ids
        parts["indexer"] = Indexer(w.lin(f"{a}.indexer.wq_b"), w.lin(f"{a}.indexer.weights_proj"),
                                   w.dense(f"{a}.indexer.wk") if source else None,
                                   w.get(f"{a}.indexer.k_norm.weight") if source else None, cfg)
    return Attention(parts, cfg, layer)


def load_moe(w: Weights, p: str, cfg: Config, *, top: int | None = None) -> MoE:
    f = f"{p}.ffn"
    return MoE(w.get(f"{f}.gate.weight"), w.get(f"{f}.gate.bias"), w.experts(f"{f}.experts.w1"),
               w.experts(f"{f}.experts.w3"), w.experts(f"{f}.experts.w2"), w.lin(f"{f}.shared_experts.w1"),
               w.lin(f"{f}.shared_experts.w3"), w.lin(f"{f}.shared_experts.w2"),
               cfg.num_experts_per_tok if top is None else top, cfg.routed_scaling_factor, cfg.swiglu_limit)


def load_engram(w: Weights, p: str, cfg: Config) -> Engram:
    spec = w.tables.get(f"{p}.engram.embed")
    if spec is None:
        raise ValueError(f"deepseek_v41: config.json lists no Engram table for {p}")
    return Engram(EngramTable(w.dir, spec), w.lin(f"{p}.engram.wkv"), w.get(f"{p}.engram.q_weight"),
                  w.get(f"{p}.engram.k_weight"), cfg.hidden_size, cfg.hc_mult, cfg.rms_norm_eps)


def load_block(w: Weights, layer: int, cfg: Config) -> Block:
    p = f"{PREFIX}.layers.{layer}"
    engram = load_engram(w, p, cfg) if layer in cfg.engram_layer_ids else None
    block = Block(layer, load_attention(w, p, cfg, layer), load_moe(w, p, cfg), w.get(f"{p}.attn_norm.weight"),
                  w.get(f"{p}.ffn_norm.weight"), w.hc(p, "attn", cfg), w.hc(p, "ffn", cfg), cfg.rms_norm_eps, engram)
    mx.eval(*block.arrays())
    return block


def load_backbone(model_dir: Path, layers: int | None = None, token_map: Any = None) -> DeepSeekV41:
    """The backbone (``layers``: the first few only, for probes and tests); ``token_map`` from the tokenizer."""

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    bad = unreadable(raw)
    if bad:
        raise ValueError(f"deepseek_v41: this checkpoint stores {len(bad)} module(s) in a format the Mac engine does "
                         f"not read, {bad[0]} first")
    cfg = Config.from_dict(raw)
    if layers is not None:
        cfg.num_hidden_layers = int(layers)
    w = Weights(Path(model_dir), raw)
    blocks = [load_block(w, i, cfg) for i in range(cfg.num_hidden_layers)]
    embed = w.dense(f"{PREFIX}.embed")
    head = w.dense(f"{PREFIX}.head")
    model = DeepSeekV41(cfg, embed, blocks, w.get(f"{PREFIX}.norm.weight"), head)
    mx.eval(model.embed, model.lm_head, model.norm)
    if token_map is not None:
        model.set_token_map(token_map)
    model.weights = w
    return model


def engram_bytes(model_dir: Path) -> int:
    """Bytes of the Engram tables, which stay in their files (read a row at a time)."""

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    total = 0
    for spec in ((raw.get(FORMAT_KEY) or {}).get("engram_tables") or {}).values():
        table = EngramTable(Path(model_dir), spec)
        total += table.nbytes()
    return total
