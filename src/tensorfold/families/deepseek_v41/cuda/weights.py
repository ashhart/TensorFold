"""One rank's DeepSeek-V4.1 weights on the GPU: EXL3 linears and routed experts split per ``split.py``, plain tensors."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tensorfold.cuda.exl3 import experts as ex3
from tensorfold.cuda.exl3.linear import Exl3Linear, GroupedLinear

from ..config import Config
from .reader import Dsv41Reader

WORLD = 2


_WORKSPACE = None


PROMPT_TILES = (64, 64, 4, 4, 8)     # the prompt GEMM's (rows, K step, warps, stages, raster group) on GB10
# decode / verify: TF_FOLD_SHARED=1 runs the shared expert as one more member of the routed experts' grouped call;
# measured slower (its 5-bit blocks set the tail: 35.0 vs 36.0 tok/s serial, 113.8 vs 117.1 at 16 clients), so off
FOLD_SHARED = __import__("os").environ.get("TF_FOLD_SHARED", "0") == "1"
# decode / verify: the wo_a slices (one per output group, separate inputs) in one launch, bit-identical to separate
GROUP_WO_A = __import__("os").environ.get("TF_GROUPED_WO_A", "1") != "0"


class Linear(Exl3Linear):
    """An EXL3 linear for any row count. Up to 128 rows (decode and verify windows): the row-invariant decode
    kernel. Prompt chunks: the prompt GEMM (W decoded to fp16 once a call, tensor-core matmul) for matrices up to
    GEMM_MAX weights, 128-row pieces of the decode kernel otherwise (the vocabulary head)."""

    PIECE = 128
    GEMM_MAX = 96 * 2**20

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, out_dtype: torch.dtype | None = None,
                 xh: torch.Tensor | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[0] <= self.PIECE:
            return super().__call__(x, out, out_dtype, xh, z)
        return self.prompt(x, out_dtype)

    def prompt(self, x: torch.Tensor, out_dtype: torch.dtype | None = None,
               res: torch.Tensor | None = None) -> torch.Tensor:
        """Prompt rows (above PIECE); ``res`` [M, N] is added in fp32 in the GEMM epilogue (prompt GEMM shapes)."""

        if self.k * self.n <= self.GEMM_MAX:
            global _WORKSPACE
            from tensorfold.cuda.exl3 import prefill

            if _WORKSPACE is None:
                _WORKSPACE = prefill.Workspace()
                prefill.TILES = PROMPT_TILES
            y = torch.empty((x.shape[0], self.n), dtype=out_dtype or x.dtype, device=x.device)
            return prefill.matmul(self, x, y, _WORKSPACE, res)
        if res is not None:
            raise ValueError("a residual needs the prompt GEMM")
        return torch.cat([super(Linear, self).__call__(x[i:i + self.PIECE].contiguous(), out_dtype=out_dtype)
                          for i in range(0, x.shape[0], self.PIECE)])


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
    wo_a_grouped: GroupedLinear | None = None     # the wo_a slices as one launch (decode / verify rows)


@dataclass
class MoEW:
    gate: torch.Tensor        # fp16 [E, D]
    bias: torch.Tensor        # fp32 [E]
    experts: ex3.Exl3RoutedExperts
    shared: tuple[Exl3Linear, Exl3Linear, Exl3Linear]   # w1, w3 (this rank's columns), w2 (rows)
    shared_id: int | None = None   # the shared expert's index in ``experts`` (FOLD_SHARED), for decode / verify rows


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
class DraftW:
    """DSpark: three draft blocks (window-only attention, 128-expert MoE) and the heads they feed."""

    layers: list[LayerW]
    main_proj: Exl3Linear     # 3 D -> D, replicated
    main_norm: torch.Tensor
    norm: torch.Tensor        # the draft's final norm (mtp.2.norm)
    markov_embed: torch.Tensor  # bf16 [V, 256]
    markov_head: torch.Tensor   # fp16 [V, 256]


@dataclass
class Weights:
    cfg: Config
    rank: int
    embed: torch.Tensor       # bf16 [V, D]
    layers: list[LayerW]
    norm: torch.Tensor
    head: Exl3Linear          # this rank's vocabulary half
    vocab_start: int
    draft: DraftW | None = None
    extra: dict = field(default_factory=dict)


def load(model_dir: str | Path, *, rank: int, layers: list[int] | None = None, device: str = "cuda",
         log=print, draft: bool = True) -> Weights:
    """Rank ``rank``'s weights (every layer, or ``layers`` for tests and benchmarks), with the DSpark blocks."""

    root = Path(model_dir)
    cfg = Config.from_dict(json.loads((root / "config.json").read_text()))
    rd = Dsv41Reader(root, rank)
    dev = torch.device(device)
    t0 = time.time()

    def t(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        x = rd.get(name)
        return (x if dtype is None else x.to(dtype)).to(dev).contiguous()

    def lin(prefix: str) -> Exl3Linear:
        return Linear.from_tensors(t(prefix + ".trellis"), t(prefix + ".suh"), t(prefix + ".svh"), "mul1",
                                       device=dev)

    def hc(p: str, site: str, norm: str) -> HCW:
        return HCW(t(p + f"hc_{site}_fn", torch.float32), t(p + f"hc_{site}_base", torch.float32),
                   t(p + f"hc_{site}_scale", torch.float32), t(p + norm + ".weight"))

    def attn(i: int, block: str) -> AttnW:
        p = block + "attn."
        groups = cfg.o_groups // WORLD
        ratio = cfg.compress_ratios[i]
        a = AttnW(lin(p + "wq_a"), lin(p + "wkv"), t(p + "q_norm.weight"), t(p + "kv_norm.weight"), lin(p + "wq_b"),
                  [lin(p + f"wo_a.slice.{g}") for g in range(rank * groups, (rank + 1) * groups)], lin(p + "wo_b"),
                  t(p + "attn_sink", torch.float32), ratio)
        if GROUP_WO_A:
            a.wo_a_grouped = GroupedLinear(a.wo_a)
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

    def moe(block: str, n_experts: int) -> MoEW:
        p = block + "ffn."
        names = [p + f"experts.{e}.{w}.{x}" for w in ("w1", "w3", "w2") for e in range(n_experts)
                 for x in ("trellis", "suh", "svh")]
        rd.prefetch(names, dev)

        def triple(e: int, w: str):
            q = p + f"experts.{e}.{w}."
            return (rd.get(q + "trellis").to(dev).contiguous(), rd.get(q + "suh").to(dev), rd.get(q + "svh").to(dev))

        gate = [triple(e, "w1") for e in range(n_experts)]
        up = [triple(e, "w3") for e in range(n_experts)]
        down = [triple(e, "w2") for e in range(n_experts)]
        if not FOLD_SHARED:
            experts = ex3.prepare(gate, up, down, "mul1", device=dev)
            shared = (lin(p + "shared_experts.w1"), lin(p + "shared_experts.w3"), lin(p + "shared_experts.w2"))
            return MoEW(t(p + "gate.weight"), t(p + "gate.bias", torch.float32), experts, shared)
        # the shared expert as one more grouped expert (this rank's width, like the routed ones; its own bits): its
        # stored trellis serves both the grouped decode kernel and the prompt GEMM (the "stored" layout), one copy
        sh = [(t(p + f"shared_experts.{w}.trellis"), t(p + f"shared_experts.{w}.suh"), t(p + f"shared_experts.{w}.svh"))
              for w in ("w1", "w3", "w2")]
        experts = ex3.prepare(gate + [sh[0]], up + [sh[1]], down + [sh[2]], "mul1", device=dev)
        shared = tuple(Linear.from_tensors(*m, "mul1", device=dev, layout="stored") for m in sh)
        return MoEW(t(p + "gate.weight"), t(p + "gate.bias", torch.float32), experts, shared, n_experts)

    def engram(i: int) -> EngramW | None:
        if i not in cfg.engram_layer_ids:
            return None
        p = f"layers.{i}.engram."
        return EngramW(lin(p + "wkv"), t(p + "q_weight", torch.float32), t(p + "k_weight", torch.float32))

    wanted = list(range(cfg.num_hidden_layers)) if layers is None else layers
    out = []
    def block(i: int, pfx: str, n_experts: int) -> LayerW:
        return LayerW(i, hc(pfx, "attn", "attn_norm"), hc(pfx, "ffn", "ffn_norm"), attn(i, pfx),
                      moe(pfx, n_experts), engram(i) if i < cfg.num_hidden_layers else None)

    for i in wanted:
        out.append(block(i, f"layers.{i}.", cfg.n_routed_experts))
        log(f"[tensorfold] rank {rank} layer {i} loaded ({time.time() - t0:.0f} s, "
            f"{torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB)", flush=True)
    head = lin("head")
    w = Weights(cfg, rank, t("embed.weight"), out, t("norm.weight"), head, rank * head.n)
    if draft:
        n = cfg.num_hidden_layers
        blocks = [block(n + j, f"mtp.{j}.", cfg.dspark_n_routed_experts) for j in range(cfg.num_nextn_predict_layers)]
        w.draft = DraftW(blocks, lin("mtp.0.main_proj"), t("mtp.0.main_norm.weight"), t("mtp.2.norm.weight"),
                         t("mtp.2.markov_head.embed.weight"), t("mtp.2.markov_head.head.weight"))
        log(f"[tensorfold] rank {rank} DSpark blocks loaded ({time.time() - t0:.0f} s, "
            f"{torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB)", flush=True)
    rd.close()
    return w
