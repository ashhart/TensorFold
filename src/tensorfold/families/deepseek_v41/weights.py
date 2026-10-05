"""The DeepSeek-V4.1 backbone from the converted source layout: no model. prefix, affine Q3 + BF16 exceptions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v41.attention import Attention
from tensorfold.families.deepseek_v41.compressor import Compressor, Indexer
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.dense import prepare as prepare_dense
from tensorfold.families.deepseek_v41.engram import Engram, EngramHash
from tensorfold.families.deepseek_v41.model import Block, DeepSeekV41
from tensorfold.families.deepseek_v41.moe import Expert, MoE, Shared
from tensorfold.families.glm5_next.linear import Dense as BFDense
from tensorfold.families.glm5_next.linear import Q


def formats(config: dict[str, Any]) -> tuple[tuple[int, int, str], dict[str, tuple[int, int, str]]]:
    """The checkpoint's default (bits, group, mode) and its per-module entries."""
    from tensorfold.families import _quantization_block, layer_quantization

    block = _quantization_block(config) or {}
    default = (int(block.get("bits") or 0), int(block.get("group_size") or 0), str(block.get("mode") or "affine"))
    return default, layer_quantization(config)


def unreadable(config: dict[str, Any]) -> list[str]:
    """Modules stored in a format this engine does not read: affine 3-bit g64 (4-bit accepted), BF16 exceptions."""
    default, modules = formats(config)
    bad = [] if default[2] == "affine" and default[0] in (3, 4) and default[1] == 64 else ["(default)"]
    for name, fmt in modules.items():
        if fmt[2] != "affine" or fmt[1] != 64 or fmt[0] not in (3, 4):
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
        """One safetensors file's tensors."""
        return cls(path.parent, {name: path.name for name in mx.load(str(path))})

    def has(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str) -> mx.array:
        shard = self.where[name]
        if self._shard is None or self._shard[0] != shard:
            self._shard = (shard, mx.load(str(self.dir / shard)))
        return self._shard[1][name]

    def bits_of(self, prefix: str) -> int:
        """A module's stored bits from its packed width and scale columns (shapes cannot tell 3 from 4)."""
        packed, groups = int(self.get(f"{prefix}.weight").shape[-1]), int(self.get(f"{prefix}.scales").shape[-1])
        ins = groups * 64
        return ins and (packed * 32) // ins or 0

    def q(self, prefix: str, bits: int | None = None) -> Q | BFDense:
        """A linear at its stored format: a quant triple when the checkpoint holds one, else its BF16 matrix."""
        where = self.where or {}
        if bits is None and f"{prefix}.scales" not in where:
            return self.bf16(prefix)                        # small matrices stay BF16 (the converter's exception)
        bits = self.bits_of(prefix) if bits is None else int(bits)
        try:
            return Q(self.get(f"{prefix}.weight"), self.get(f"{prefix}.scales"), self.get(f"{prefix}.biases"),
                     bits=bits, group=64)
        except ValueError as exc:
            raise ValueError(f"{prefix}: {exc}") from None

    def bf16(self, prefix: str) -> BFDense:
        return BFDense(self.get(f"{prefix}.weight"))

    def hc(self, prefix: str, cfg: Any) -> Any:
        from tensorfold.families.glm5_next.model import HC

        # the converted layout keeps the official flat names: layers.N.hc_attn_fn / _base / _scale
        return HC(self.get(f"{prefix}_fn"), self.get(f"{prefix}_base"), self.get(f"{prefix}_scale"), cfg)


def load_attention(w: Weights, cfg: Config, layer: int, prefix: str | None = None) -> Attention:
    p = prefix or f"layers.{layer}"
    a = f"{p}.attn"
    parts: dict[str, Any] = {n: w.q(f"{a}.{n}") for n in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")}
    parts.update(q_norm=w.get(f"{a}.q_norm.weight"), kv_norm=w.get(f"{a}.kv_norm.weight"),
                 attn_sink=w.get(f"{a}.attn_sink"))
    if cfg.kv_source(layer):
        cp = f"{a}.compressor"
        wgate = w.get(f"{cp}.wgate.weight") if cfg.ratio(layer) > 1 else None
        parts["compressor"] = Compressor(w.get(f"{cp}.wkv.weight"), wgate, w.get(f"{cp}.norm.weight"),
                                         cfg.ratio(layer), cfg.rms_norm_eps)
    if cfg.index_source(layer):
        i = f"{a}.indexer"
        parts["indexer"] = Indexer(w.q(f"{i}.wq_b"), w.q(f"{i}.weights_proj"), cfg.index_n_heads,
                                   cfg.index_head_dim, cfg.index_topk, cfg.inv_freq(layer))
        if cfg.kv_source(layer):
            parts["indexer_wk"] = w.get(f"{i}.wk.weight")            # BF16 exception, fp32 in the math
            parts["indexer_k_norm"] = w.get(f"{i}.k_norm.weight")
    return Attention(parts, cfg, layer)


def load_moe(w: Weights, cfg: Config, layer: int, prefix: str | None = None, experts: int | None = None,
             topk: int | None = None) -> MoE:
    p = prefix or f"layers.{layer}"
    f = f"{p}.ffn"
    count = cfg.n_routed_experts if experts is None else int(experts)
    pick = cfg.num_experts_per_tok if topk is None else int(topk)
    shared = Shared(w.q(f"{f}.shared_experts.w1"), w.q(f"{f}.shared_experts.w3"), w.q(f"{f}.shared_experts.w2"),
                    cfg.swiglu_limit)
    experts_list = [Expert(w.q(f"{f}.experts.{e}.w1"), w.q(f"{f}.experts.{e}.w3"), w.q(f"{f}.experts.{e}.w2"),
                           cfg.swiglu_limit) for e in range(count)]
    return MoE(w.get(f"{f}.gate.weight"), w.get(f"{f}.gate.bias"), w.get(f"{f}.gate.bias_vl"), experts_list,
               shared, cfg, topk=pick)


def load_engram(w: Weights, cfg: Config, layer: int, prefix: str | None = None) -> Engram:
    p = f"{prefix or f'layers.{layer}'}.engram"
    return Engram(w.q(f"{p}.embed"), w.q(f"{p}.wkv"), w.get(f"{p}.q_weight"), w.get(f"{p}.k_weight"), cfg, layer)


def load_block(w: Weights, cfg: Config, layer: int, prefix: str | None = None, experts: int | None = None,
               topk: int | None = None) -> Block:
    p = prefix or f"layers.{layer}"
    engram = load_engram(w, cfg, layer, p) if layer in cfg.engram_layer_ids and prefix is None else None
    block = Block(load_attention(w, cfg, layer, p), load_moe(w, cfg, layer, p, experts, topk),
                  w.get(f"{p}.attn_norm.weight"), w.get(f"{p}.ffn_norm.weight"),
                  w.hc(f"{p}.hc_attn", cfg), w.hc(f"{p}.hc_ffn", cfg),
                  cfg.rms_norm_eps, engram)
    mx.eval(*block_arrays(block))
    return block


def block_arrays(block: Block) -> list[mx.array]:
    a = block.attn
    out = [*a.x_proj.arrays(), *a.wq_b.arrays(), *a.wo_b.arrays(), a.q_norm, a.kv_norm, a.sink, a.inv_freq,
           *[x for g in a.wo_a for x in g.arrays()], *block.moe.arrays(), block.attn_norm, block.ffn_norm]
    for hc in (block.attn_hc, block.ffn_hc):
        out += [hc.fn, hc.base, hc.scale]
    if a.compressor is not None:
        out += [a.compressor.wkv, a.compressor.norm] + ([a.compressor.wgate] if a.compressor.wgate is not None
                                                        else [])
    if a.indexer is not None:
        out += [*a.indexer.wq_b.arrays(), *a.indexer.weights_proj.arrays()]
    if a.indexer_wk is not None:
        out += [a.indexer_wk, a.indexer_k_norm]
    if block.engram is not None:
        out += [*block.engram.embed.arrays(), *block.engram.wkv.arrays(), block.engram.weight]
    return out


def token_map_of(model_dir: Path, cfg: Config) -> list[int] | None:
    """The compressed-token map from the checkpoint's token_map.json (built by the converter), else None."""
    path = model_dir / "token_map.json"
    if path.is_file():
        got = json.loads(path.read_text())
        return [int(t) for t in got["token_map"]]
    return None


def engram_hash_of(model: DeepSeekV41) -> EngramHash:
    """The model's engram hash (built once at load; its primes/multipliers are pure functions of the config)."""
    found = getattr(model, "_engram_hash", None)
    if found is None:
        cfg = model.args
        token_map = model.token_map or list(range(cfg.vocab_size))
        found = model._engram_hash = EngramHash(cfg, token_map, token_map[cfg.engram_pad_token_id])
    return found


def load_backbone(model_dir: Path, layers: int | None = None) -> DeepSeekV41:
    """The backbone (``layers``: the first few only, for probes and tests)."""
    raw = json.loads((model_dir / "config.json").read_text())
    bad = unreadable(raw)
    if bad:
        raise ValueError(f"DeepSeek-V4.1-Flash's engine reads MLX affine 3-bit g64 weights; this checkpoint stores "
                         f"{len(bad)} module(s) otherwise, {bad[0]} first")
    cfg = Config.from_dict(raw)
    w = Weights(model_dir)
    count = cfg.num_hidden_layers if layers is None else int(layers)
    blocks = [load_block(w, cfg, i) for i in range(count)]
    has_draft_weights = any(name.startswith("mtp.") for name in w.where)
    model = DeepSeekV41(cfg, w.q("embed"), blocks, w.get("norm.weight"), w.q("head"),
                        token_map_of(model_dir, cfg), has_draft_weights=has_draft_weights)
    mx.eval(*model.embed.arrays(), *model.lm_head.arrays(), model.norm)
    prepare_dense([model.lm_head, *[q for b in blocks for q in dense_linears(b)]])
    return model


def dense_linears(block: Block) -> list[Q]:
    """The projections a block runs through ``dense.dense`` on the decode path."""
    a, s = block.attn, block.moe.shared
    return [a.x_proj, a.wq_b, a.wo_b, s.gate_up, s.down]
