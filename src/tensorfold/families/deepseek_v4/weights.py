"""The backbone from the mlx-community layout: tensors read shard by shard, each layer built and evaluated in turn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v4.attention import Attention
from tensorfold.families.deepseek_v4.compressor import Compressor, Indexer
from tensorfold.families.deepseek_v4.config import Config
from tensorfold.families.deepseek_v4.dense import prepare as prepare_dense
from tensorfold.families.deepseek_v4.model import Block, DeepSeekV4, HeadHC
from tensorfold.families.deepseek_v4.moe import FP4, MoE, Shared
from tensorfold.families.glm5_next.linear import Q
from tensorfold.families.glm5_next.model import HC

AFFINE = (4, 64, "affine")
EXPERTS = (4, 32, "mxfp4")


def formats(config: dict[str, Any]) -> tuple[tuple[int, int, str], dict[str, tuple[int, int, str]]]:
    """The checkpoint's default (bits, group, mode) and its per-module entries."""

    from tensorfold.families import _quantization_block, layer_quantization

    block = _quantization_block(config) or {}
    default = (int(block.get("bits") or 0), int(block.get("group_size") or 0), str(block.get("mode") or "affine"))
    return default, layer_quantization(config)


def unreadable(config: dict[str, Any]) -> list[str]:
    """Modules stored in a format this engine does not read: affine 4-bit g64, and mxfp4 g32 routed experts."""

    default, modules = formats(config)
    bad = [] if default == AFFINE else ["(default)"]
    for name, fmt in modules.items():
        want = EXPERTS if ".switch_mlp." in name else AFFINE
        if fmt != want:
            bad.append(name)
    return sorted(bad)


class Weights:
    """The checkpoint's tensors by name, read one shard at a time (arrays already taken stay alive)."""

    def __init__(self, model_dir: Path, where: dict[str, str] | None = None) -> None:
        self.dir = model_dir
        if where is None:
            where = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.where = where
        self._shard: tuple[str, dict[str, mx.array]] | None = None

    @classmethod
    def file(cls, path: Path) -> "Weights":
        """One safetensors file's tensors (the converted MTP layer)."""

        return cls(path.parent, {name: path.name for name in mx.load(str(path))})

    def has(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str) -> mx.array:
        shard = self.where[name]
        if self._shard is None or self._shard[0] != shard:
            self._shard = (shard, mx.load(str(self.dir / shard)))
        return self._shard[1][name]

    def q(self, prefix: str) -> Q:
        try:
            return Q(self.get(f"{prefix}.weight"), self.get(f"{prefix}.scales"), self.get(f"{prefix}.biases"),
                     bits=AFFINE[0], group=AFFINE[1])
        except ValueError as exc:
            raise ValueError(f"{prefix}: {exc}") from None

    def fp4(self, prefix: str) -> FP4:
        return FP4(self.get(f"{prefix}.weight"), self.get(f"{prefix}.scales"))

    def hc(self, prefix: str, cfg: Config) -> HC:
        return HC(self.get(f"{prefix}.fn"), self.get(f"{prefix}.base"), self.get(f"{prefix}.scale"), cfg)


def load_attention(w: Weights, p: str, cfg: Config, layer: int) -> Attention:
    a = f"{p}.attn"
    parts: dict[str, Any] = {n: w.q(f"{a}.{n}") for n in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")}
    parts.update(q_norm=w.get(f"{a}.q_norm.weight"), kv_norm=w.get(f"{a}.kv_norm.weight"),
                 attn_sink=w.get(f"{a}.attn_sink"))
    ratio = cfg.ratio(layer)
    freqs = cfg.inv_freq(layer)
    if ratio:
        parts["compressor"] = compressor(w, f"{a}.compressor", ratio, cfg, freqs)
    if ratio == 4:
        parts["indexer"] = Indexer(w.q(f"{a}.indexer.wq_b"), w.q(f"{a}.indexer.weights_proj"),
                                   compressor(w, f"{a}.indexer.compressor", ratio, cfg, freqs), cfg.index_n_heads,
                                   cfg.index_head_dim, cfg.index_topk, freqs)
    return Attention(parts, cfg, layer)


def compressor(w: Weights, p: str, ratio: int, cfg: Config, freqs: mx.array) -> Compressor:
    return Compressor(w.q(f"{p}.wkv"), w.q(f"{p}.wgate"), w.get(f"{p}.ape"), w.get(f"{p}.norm.weight"), ratio,
                      cfg.rms_norm_eps, freqs)


def load_moe(w: Weights, p: str, cfg: Config, layer: int) -> MoE:
    f = f"{p}.ffn"
    hashed = layer < cfg.num_hash_layers
    shared = Shared(w.q(f"{f}.shared_experts.gate_proj"), w.q(f"{f}.shared_experts.up_proj"),
                    w.q(f"{f}.shared_experts.down_proj"), cfg.swiglu_limit)
    return MoE(w.get(f"{f}.gate.weight"), None if hashed else w.get(f"{f}.gate.e_score_correction_bias"),
               w.get(f"{f}.gate.tid2eid") if hashed else None, w.fp4(f"{f}.switch_mlp.gate_proj"),
               w.fp4(f"{f}.switch_mlp.up_proj"), w.fp4(f"{f}.switch_mlp.down_proj"), shared, cfg)


def load_block(w: Weights, layer: int, cfg: Config, prefix: str | None = None) -> Block:
    p = prefix or f"model.layers.{layer}"
    block = Block(load_attention(w, p, cfg, layer), load_moe(w, p, cfg, layer), w.get(f"{p}.attn_norm.weight"),
                  w.get(f"{p}.ffn_norm.weight"), w.hc(f"{p}.attn_hc", cfg), w.hc(f"{p}.ffn_hc", cfg),
                  cfg.rms_norm_eps)
    mx.eval(*block_arrays(block))
    return block


def block_arrays(block: Block) -> list[mx.array]:
    a = block.attn
    out = [*a.x_proj.arrays(), *a.wq_b.arrays(), *a.wo_b.arrays(), a.q_norm, a.kv_norm, a.sink, a.inv_freq,
           *[x for g in a.wo_a for x in g.arrays()], *block.moe.arrays(), block.attn_norm, block.ffn_norm]
    for hc in (block.attn_hc, block.ffn_hc):
        out += [hc.fn, hc.base, hc.scale]
    if a.cproj is not None:
        out += a.cproj.arrays()
    for comp in a.compressors():
        out += [comp.ape, comp.norm]
    if a.indexer is not None:
        out += [*a.indexer.wq_b.arrays(), *a.indexer.weights_proj.arrays()]
    return out


def load_backbone(model_dir: Path, layers: int | None = None) -> DeepSeekV4:
    """The backbone (``layers``: the first few only, for probes and tests)."""

    raw = json.loads((model_dir / "config.json").read_text())
    bad = unreadable(raw)
    if bad:
        raise ValueError(f"DeepSeek-V4-Flash's Mac engine reads MLX affine 4-bit weights in groups of 64 and mxfp4 "
                         f"routed experts; this checkpoint stores {len(bad)} module(s) otherwise, {bad[0]} first")
    cfg = Config.from_dict(raw)
    w = Weights(model_dir)
    count = cfg.num_hidden_layers if layers is None else int(layers)
    blocks = [load_block(w, i, cfg) for i in range(count)]
    head = HeadHC(w.get("model.hc_head.fn"), w.get("model.hc_head.base"), w.get("model.hc_head.scale"),
                  cfg.rms_norm_eps, cfg.hc_eps)
    model = DeepSeekV4(cfg, w.q("model.embed_tokens"), blocks, head, w.get("model.norm.weight"), w.q("lm_head"))
    mx.eval(*model.embed.arrays(), *model.lm_head.arrays(), model.norm, head.fn, head.base, head.scale)
    prepare_dense([model.lm_head, *[q for b in blocks for q in dense_linears(b)]])
    return model


def dense_linears(block: Block) -> list[Q]:
    """The projections a block runs through ``dense.dense`` on the decode path."""

    a, s = block.attn, block.moe.shared
    return [a.x_proj, a.wq_b, a.wo_b, s.gate_up, s.down]
