"""One rank's DeepSeek-V4.1 weights on the GPU: EXL3 linears and routed experts split per ``split.py``, plain tensors."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tensorfold.cuda.exl3 import experts as ex3
from tensorfold.cuda.exl3.linear import Exl3Linear

from ..config import Config
from .reader import Dsv41Reader

WORLD = 2


@dataclass
class HCW:
    fn: torch.Tensor          # fp32 [24, 4 D]
    base: torch.Tensor        # fp32 [24]
    scale: torch.Tensor       # fp32 [3]
    norm: torch.Tensor        # bf16 [D] (attn_norm or ffn_norm)


@dataclass
class CompressorW:
    wkv: Exl3Linear
    wgate: Exl3Linear | None  # ratio 2 only
    norm: torch.Tensor


@dataclass
class IndexerW:
    wq_b: Exl3Linear          # 1280 -> 32 x 128 (replicated)
    weights_proj: torch.Tensor  # fp16 [32, D]
    wk: Exl3Linear | None     # 512 -> 128 on kv sources
    k_norm: torch.Tensor | None


@dataclass
class AttnW:
    wq_a: Exl3Linear          # replicated
    wkv: Exl3Linear           # replicated
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    wq_b: Exl3Linear          # this rank's heads
    wo_a: list[Exl3Linear]    # this rank's output groups
    wo_b: Exl3Linear          # this rank's input rows (partial sums)
    sink: torch.Tensor        # fp32 [heads on this rank]
    ratio: int
    compressor: CompressorW | None = None
    indexer: IndexerW | None = None


@dataclass
class MoEW:
    gate: torch.Tensor        # fp32 [E, D] (stored fp16)
    bias: torch.Tensor        # fp32 [E]
    experts: ex3.Exl3RoutedExperts
    shared: tuple[Exl3Linear, Exl3Linear, Exl3Linear]   # w1, w3 (this rank's columns), w2 (rows)


@dataclass
class EngramW:
    wkv: Exl3Linear           # 6144 -> 5 D, this rank's half of the columns
    q: torch.Tensor           # fp32 [4, D]
    k: torch.Tensor


@dataclass
class LayerW:
    index: int
    hc_attn: HCW
    hc_ffn: HCW
    attn: AttnW
    moe: MoEW
    engram: EngramW | None = None


@dataclass
class Weights:
    cfg: Config
    rank: int
    embed: torch.Tensor       # bf16 [V, D]
    layers: list[LayerW]
    norm: torch.Tensor
    head: Exl3Linear          # this rank's vocabulary half
    vocab_start: int
    extra: dict = field(default_factory=dict)


def load(model_dir: str | Path, *, rank: int, layers: list[int] | None = None, device: str = "cuda",
         log=print) -> Weights:
    """Rank ``rank``'s weights (every layer, or ``layers`` for tests and benchmarks)."""

    root = Path(model_dir)
    cfg = Config.from_dict(json.loads((root / "config.json").read_text()))
    rd = Dsv41Reader(root, rank)
    dev = torch.device(device)
    t0 = time.time()

    def t(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        x = rd.get(name)
        return (x if dtype is None else x.to(dtype)).to(dev).contiguous()

    def lin(prefix: str) -> Exl3Linear:
        return Exl3Linear.from_tensors(t(prefix + ".trellis"), t(prefix + ".suh"), t(prefix + ".svh"), "mul1",
                                       device=dev)

    def hc(i: int, site: str, norm: str) -> HCW:
        p = f"layers.{i}."
        return HCW(t(p + f"hc_{site}_fn", torch.float32), t(p + f"hc_{site}_base", torch.float32),
                   t(p + f"hc_{site}_scale", torch.float32), t(p + norm + ".weight"))

    def attn(i: int) -> AttnW:
        p = f"layers.{i}.attn."
        groups = cfg.o_groups // WORLD
        ratio = cfg.layer_ratios[i]
        a = AttnW(lin(p + "wq_a"), lin(p + "wkv"), t(p + "q_norm.weight"), t(p + "kv_norm.weight"), lin(p + "wq_b"),
                  [lin(p + f"wo_a.slice.{g}") for g in range(rank * groups, (rank + 1) * groups)], lin(p + "wo_b"),
                  t(p + "attn_sink", torch.float32), ratio)
        if i in cfg.kv_source_layer_ids:
            a.compressor = CompressorW(lin(p + "compressor.wkv"),
                                       lin(p + "compressor.wgate") if ratio == 2 else None,
                                       t(p + "compressor.norm.weight"))
        if i in cfg.index_source_layer_ids:
            owns_k = i in cfg.kv_source_layer_ids
            a.indexer = IndexerW(lin(p + "indexer.wq_b"), t(p + "indexer.weights_proj.weight"),
                                 lin(p + "indexer.wk") if owns_k else None,
                                 t(p + "indexer.k_norm.weight") if owns_k else None)
        return a

    def moe(i: int) -> MoEW:
        p = f"layers.{i}.ffn."
        names = [p + f"experts.{e}.{w}.{x}" for w in ("w1", "w3", "w2") for e in range(cfg.n_routed_experts)
                 for x in ("trellis", "suh", "svh")]
        rd.prefetch(names, dev)

        def triple(e: int, w: str):
            q = p + f"experts.{e}.{w}."
            return (rd.get(q + "trellis").to(dev).contiguous(), rd.get(q + "suh").to(dev), rd.get(q + "svh").to(dev))

        gate = [triple(e, "w1") for e in range(cfg.n_routed_experts)]
        up = [triple(e, "w3") for e in range(cfg.n_routed_experts)]
        down = [triple(e, "w2") for e in range(cfg.n_routed_experts)]
        experts = ex3.prepare(gate, up, down, "mul1", device=dev)
        shared = (lin(p + "shared_experts.w1"), lin(p + "shared_experts.w3"), lin(p + "shared_experts.w2"))
        # the router multiplies in fp32: keep an fp32 copy (converting 384 x 5120 every step cost ~10 us a layer)
        return MoEW(t(p + "gate.weight", torch.float32), t(p + "gate.bias", torch.float32), experts, shared)

    def engram(i: int) -> EngramW | None:
        if i not in cfg.engram_layer_ids:
            return None
        p = f"layers.{i}.engram."
        return EngramW(lin(p + "wkv"), t(p + "q_weight", torch.float32), t(p + "k_weight", torch.float32))

    wanted = list(range(cfg.num_hidden_layers)) if layers is None else layers
    out = []
    for i in wanted:
        out.append(LayerW(i, hc(i, "attn", "attn_norm"), hc(i, "ffn", "ffn_norm"), attn(i), moe(i), engram(i)))
        log(f"[tensorfold] rank {rank} layer {i} loaded ({time.time() - t0:.0f} s, "
            f"{torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB)", flush=True)
    head = lin("head")
    w = Weights(cfg, rank, t("embed.weight"), out, t("norm.weight"), head, rank * head.n)
    rd.close()
    return w
