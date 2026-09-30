"""Dataclasses and token-id helpers for the Flash Next CUDA weights."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tensorfold.cuda import experts as grouped

from ..host_table import BF16Table, HostTable
from ..ssd_table import SSDTable
from .ngram import NGram
from .qmm import Q4


def stop_ids(configured: Any, generation: Path) -> tuple[int, ...]:
    """config.json's end-of-reply ids, then generation_config.json's it lacks (EXL3 packs keep <|im_end|> there)."""

    def ids(value: Any) -> list[int]:
        return [] if value is None else [int(e) for e in value] if isinstance(value, list) else [int(value)]

    found = ids(configured)
    if generation.exists():
        found += ids(json.loads(generation.read_text()).get("eos_token_id"))
    out = tuple(dict.fromkeys(found))
    if not out:
        raise ValueError("no eos_token_id in config.json or generation_config.json")
    return out


@dataclass
class Config:
    hidden: int
    layers: int
    layer_types: list[str]
    vocab: int
    eps: float
    heads: int
    kv_heads: int
    head_dim: int
    rope_theta: float
    rotary_dim: int
    nk: int
    nv: int
    dk: int
    dv: int
    conv_kernel: int
    experts: int
    top_k: int
    moe_width: int
    shared_width: int
    streams: int
    low: int
    index_heads: int
    index_dim: int
    index_budget: int
    index_ratio: int
    ple_layers: list[int]              # zero-indexed decoder layers with the n-gram embedding
    ple_dim: int
    ple_kernel: int
    ngram_size: int
    heads_per_ngram: int
    ngram_base: int
    ngram_divisor: int
    ngram_shards: int
    seed: int
    ple_eos: int
    eos: tuple[int, ...]
    group_size: int
    bits: int
    quant: str = "mlx"                 # "mlx" (affine 4-bit everywhere) or "modelopt" (NVFP4 routed experts)
    mrope_section: tuple[int, int, int] = (11, 11, 10)   # interleaved t/h/w rotary pairs
    nvfp4_group: int = 16              # the NVFP4 block size (the checkpoint's config_groups weights.group_size)

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        t = dict(raw.get("text_config") or raw)
        rope = dict(t.get("rope_parameters") or {})
        head_dim = int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"])
        partial = float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
        teos = t.get("eos_token_id")
        eos = stop_ids(raw.get("eos_token_id", teos), Path(model_dir) / "generation_config.json")
        quant = raw.get("quantization") or raw.get("quantization_config") or {}
        method = str(quant.get("quant_method") or "mlx").lower()
        groups = quant.get("config_groups") or {}
        group = int(((groups.get("group_0") or {}).get("weights") or {}).get("group_size", 16))
        return cls(
            hidden=int(t["hidden_size"]), layers=int(t["num_hidden_layers"]),
            layer_types=["linear" if k == "linear_attention" else "attention" for k in t["layer_types"]],
            vocab=int(t["vocab_size"]), eps=float(t["rms_norm_eps"]), heads=int(t["num_attention_heads"]),
            kv_heads=int(t["num_key_value_heads"]), head_dim=head_dim,
            rope_theta=float(rope.get("rope_theta", 10_000_000)), rotary_dim=int(head_dim * partial),
            nk=int(t["linear_num_key_heads"]), nv=int(t["linear_num_value_heads"]),
            dk=int(t["linear_key_head_dim"]), dv=int(t["linear_value_head_dim"]),
            conv_kernel=int(t["linear_conv_kernel_dim"]), experts=int(t["num_experts"]),
            top_k=int(t["num_experts_per_tok"]), moe_width=int(t["moe_intermediate_size"]),
            shared_width=int(t["shared_expert_intermediate_size"]), streams=int(t.get("hc_count", 4)),
            low=int(t.get("hc_lowrank", 320)), index_heads=int(t.get("indexer_n_heads", 4)),
            index_dim=int(t.get("indexer_head_dim", 128)), index_budget=int(t.get("indexer_budget", 2048)),
            index_ratio=int(t.get("indexer_compress_ratio", 4)),
            ple_layers=sorted({int(i) - 1 for i in t.get("ple_layer_ids") or []}),
            ple_dim=int(t.get("ple_embed_dim") or t["hidden_size"]),
            ple_kernel=int(t.get("ple_conv_kernel_size", 4)), ngram_size=int(t.get("ngram_size", 3)),
            heads_per_ngram=int(t.get("heads_per_ngram", 8)),
            ngram_base=int(t.get("ngram_vocab_size_base", 20_000_000)),
            ngram_divisor=int(t.get("make_ngram_vocab_size_divisible_by", 128)),
            ngram_shards=int(t.get("split_ngram_parts", 128)), seed=int(t.get("seed", 1234)),
            ple_eos=int(teos[0] if isinstance(teos, list) else teos) if teos is not None else 0,
            eos=eos, group_size=int(quant.get("group_size", 32)), bits=int(quant.get("bits", 4)),
            quant=method, nvfp4_group=group,
            mrope_section=tuple(int(x) for x in rope.get("mrope_section", (11, 11, 10))),
        )

    @property
    def conv_dim(self) -> int:
        return 2 * self.nk * self.dk + self.nv * self.dv

    @property
    def top_blocks(self) -> int:
        return self.index_budget // self.index_ratio

    def ngram(self, ple_index: int = 0) -> NGram:
        return NGram(vocab=self.vocab, ngram_size=self.ngram_size, heads_per_ngram=self.heads_per_ngram,
                     vocab_base=self.ngram_base, divisor=self.ngram_divisor, shards=self.ngram_shards,
                     seed=self.seed, eos=self.ple_eos, embed_dim=self.ple_dim, ple_index=ple_index)


@dataclass
class HC:
    down: Q4                  # [low (+ streams), S*D]: input_mix_weight_down (then block_inject_weight)
    up: Q4                    # [S*D, low]
    scale: torch.Tensor       # [S*D] fp32 (hc_norm gamma)
    inject: bool
    prefill_down: Q4 | None = None      # the same matrices packed for the shared prefill matmul
    prefill_up: Q4 | None = None


@dataclass
class GDNW:
    proj: Q4                  # [qkv | z | b | a] x D
    conv: torch.Tensor        # [conv_dim, taps] bf16
    a_log: torch.Tensor       # [nv] fp32
    dt_bias: torch.Tensor     # [nv] fp32
    norm: torch.Tensor        # [dv] bf16 (the gated RMSNorm's weight, used as stored)
    out: Q4

    @property
    def kernel(self) -> str:
        return getattr(self.proj, "kernel", "qmm")


@dataclass
class AttnW:
    proj: Q4                  # [q|gate pairs | k | v | indexer q | indexer key] x D
    q_scale: torch.Tensor     # [head_dim] fp32
    k_scale: torch.Tensor
    iq_scale: torch.Tensor    # [index_dim] fp32
    ik_scale: torch.Tensor    # the pooled indexer keys' norm
    o: Q4

    @property
    def kernel(self) -> str:
        return getattr(self.proj, "kernel", "qmm")


@dataclass
class MoEW:
    router: torch.Tensor      # [E + 1, D] bf16: router rows, then the shared expert's gate row
    experts: grouped.Experts  # E + 1 experts (the shared expert last); a nvfp4 MoE4 on NVFP4 checkpoints


@dataclass
class PLEW:
    table: HostTable | SSDTable | BF16Table   # the 128 shards: host memory map, SSD at each lookup, or bf16 rows
    key: Q4                   # [S*D, ple_dim]
    value: Q4                 # [D, ple_dim]
    norm_key: torch.Tensor    # [S*D] fp32
    norm_query: torch.Tensor
    norm_conv: torch.Tensor
    conv: torch.Tensor        # [S*D, taps] bf16
    ngram: NGram


@dataclass
class LayerW:
    index: int
    linear: bool
    attn_hc: HC
    mlp_hc: HC
    gdn: GDNW | None
    attn: AttnW | None
    moe: MoEW
    ple: PLEW | None = None


@dataclass
class MTPW:
    norm_e: torch.Tensor      # [D] fp32
    norm_h: torch.Tensor      # [S*D] fp32
    fc_e: Q4
    fc_h: Q4
    layer: LayerW
    mixer: HC


@dataclass
class Weights:
    cfg: Config
    embed: Any                      # the MLX 4-bit trilogue (words, scales, biases), or a 1-tuple of bf16
                                    # (a checkpoint whose embedding is not quantized: an NVFP4 one, an EXL3 pack)
    layers: list[LayerW]
    mixer: HC
    head: Q4
    inv_freq: torch.Tensor
    mtp: MTPW | None = None
    around_one: bool = True
    meta: dict[str, Any] = field(default_factory=dict)
    comm: Any = None          # tensor parallel: a ``comm.NCCL`` (None on one GPU)
    draft_head: Q4 | None = None   # the MTP drafts' head over a token subset (None: the full head)
    draft_ids: torch.Tensor | None = None   # the subset's token ids (this rank's share), in draft-head row order
    x3: Any = None            # an EXL3 checkpoint's shared scratch (``exl3.Scratch``); None for the MLX checkpoint

    @property
    def device(self) -> torch.device:
        return self.inv_freq.device

    @property
    def fast_prefill(self) -> bool:
        """Whether a DeltaNet or attention linear has an FP8 prompt kernel (MXFP8, block FP8; --prefill-fp8)."""

        faces = [f for layer in self.layers for f in (layer.gdn and layer.gdn.proj, layer.gdn and layer.gdn.out,
                                                       layer.attn and layer.attn.proj, layer.attn and layer.attn.o)]
        return any(hasattr(f, "prefill8") or any(hasattr(p, "prefill8") for p in getattr(f, "parts", ()))
                   for f in faces if f is not None)

    def nbytes(self) -> int:
        """Device bytes the weights hold, each storage once (an EXL3 layer's expert views share one buffer)."""

        seen: dict[int, int] = {}

        def add(x: Any) -> None:
            if isinstance(x, torch.Tensor):
                if x.device.type != "cpu":
                    storage = x.untyped_storage()
                    seen[storage.data_ptr()] = storage.nbytes()
            elif hasattr(x, "__dataclass_fields__"):
                for f in x.__dataclass_fields__:
                    add(getattr(x, f))
            elif isinstance(x, (list, tuple)):
                for y in x:
                    add(y)

        for part in (self.embed, self.layers, self.mixer, self.head, self.mtp, self.draft_head, self.draft_ids):
            add(part)
        return sum(seen.values()) + (self.x3.nbytes() if self.x3 is not None else 0)


def draft_token_ids(draft_vocab: int | str | None) -> np.ndarray | None:
    """The MTP drafts' scored ids, sorted: "default" (draft_vocab.txt), a file of ids, N (ids below N) or None (all)."""

    if not draft_vocab:
        return None
    if isinstance(draft_vocab, int):
        return np.arange(draft_vocab, dtype=np.int64)
    source = Path(__file__).with_name("draft_vocab.txt") if draft_vocab == "default" else Path(draft_vocab)
    return np.unique(np.loadtxt(source, dtype=np.int64).reshape(-1))
