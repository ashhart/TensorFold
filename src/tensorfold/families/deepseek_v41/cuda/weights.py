"""One rank's DeepSeek-V4.1-Flash weights as the forward reads them; the loader fills these, nothing here reads a file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from tensorfold.cuda.exl3.experts import Exl3RoutedExperts
    from tensorfold.cuda.nvfp4.experts import Experts4
    from tensorfold.cuda.nvfp4.linear import Mx8Linear
    from tensorfold.families.glm5_next.cuda.qmm import B16

    from ..config import Config, Role


@dataclass
class HCW:
    """One mHC mix: the projection to the 24 mixing values, their base and the three scales."""

    fn: torch.Tensor          # fp32 [24, hc_mult * D]
    base: torch.Tensor        # fp32 [24]
    scale: torch.Tensor       # fp32 [3]


@dataclass
class CompW:
    """A KV source's compressor."""

    ratio: int
    wkv: B16                  # ratio 2: [wkv | wgate] with fp32 output; ratio 1: wkv, bf16 output
    norm: torch.Tensor


@dataclass
class IdxW:
    """An index layer's query side; KV sources also own the index-K projection their cache is written with."""

    wq_b: Mx8Linear
    wproj: B16                # weights_proj [index_n_heads, D]
    wk: B16 | None = None     # KV sources only
    k_norm: torch.Tensor | None = None


@dataclass
class AttnW:
    wqa_kv: Mx8Linear         # [wq_a | wkv] stacked, replicated
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    wq_b: Mx8Linear           # the rank's heads
    wo_a: list[Mx8Linear]     # the rank's output groups, in order
    wo_b: Mx8Linear           # the rank's input columns, fp32 partial out
    sink: torch.Tensor        # fp32 [heads / world]
    comp: CompW | None = None
    idx: IdxW | None = None


@dataclass
class MoEW:
    gate: torch.Tensor        # bf16 [E, D], replicated
    bias: torch.Tensor        # fp32 [E]
    bias_vl: torch.Tensor     # fp32 [E]
    experts: Exl3RoutedExperts | Experts4   # the rank's intermediate columns
    shared_gu: Mx8Linear      # [w1 | w3] rows of the rank
    shared_d: Mx8Linear       # w2 columns of the rank, fp32 partial out


@dataclass
class LayerW:
    index: int
    role: Role
    hc_attn: HCW
    hc_ffn: HCW
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    attn: AttnW
    moe: MoEW


@dataclass
class StageW(LayerW):
    """A DSpark stage: a backbone block without compressor or indexer, its experts an ``Experts4``."""


@dataclass
class EngramW:
    wkv: Mx8Linear            # the rank's rows of [keys | value]
    wqk: torch.Tensor         # fp32 [hc_mult, D]: q_weight * k_weight


@dataclass
class DSparkW:
    main_proj: Mx8Linear      # the rank's rows
    main_norm: torch.Tensor
    stages: list[StageW]
    norm: torch.Tensor        # the last stage's output norm
    markov_embed: torch.Tensor  # bf16 [V, markov_rank], replicated
    markov_head: B16          # the rank's vocabulary rows, fp32 out
    conf: torch.Tensor        # fp32 [1, D + markov_rank]


@dataclass
class Weights:
    cfg: Config
    rank: int
    world: int
    comm: Any
    device: torch.device
    vocab_offset: int         # the first vocabulary id of this rank's head rows
    embed: torch.Tensor       # bf16 [V, D], replicated
    layers: list[LayerW]
    norm: torch.Tensor
    head: Mx8Linear           # the rank's vocabulary rows, fp32 out
    dspark: DSparkW | None
    engram: dict[int, EngramW]  # by layer id
    rope: Any                 # rope.py's cos/sin tables, one pair per kind
