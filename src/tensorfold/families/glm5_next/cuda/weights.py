"""GLM-5.3-Flash weights on the GPU, one rank's share of the MLX 4-bit checkpoint (affine, groups of 64).

``split.RankReader`` gives this rank's part of each tensor: sliced from the full checkpoint, or read from a folder
``split`` wrote. Column-parallel outputs (heads, expert and MLP width) are halves by rows, row-parallel inputs
(o_proj, down_proj) halves by input groups, everything else replicated.

Per rank (HL = local attention heads, 32 of 64):
  KDA  proj [q | k | v | f_a | g_a | b] (3 HL 128 + 128 + 128 + HL rows), f_b and g_b (HL 128 x 128), conv
       [q | k | v] taps, A_log, dt_bias, the gated norm, o_proj (row parallel)
  DSA  proj [q_a | kv_a] (replicated), q_b (HL 256 rows), kv_b split into its key and value rows (HL 256 each),
       o_proj (row parallel); the indexer is kept raw (unused until a context passes 2048 tokens)
  MoE  router rows (bf16) and correction bias, 288 routed experts + the shared expert as expert 288, each with
       this rank's half of the intermediate width; dense MLP layers stack [gate | up]
  head this rank's half of the vocabulary rows; embedding and norms replicated
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .qmm import Experts, Q4, as_i32, make_experts, make_q4, stack_q4

PREFIX = "model.language_model."


@dataclass
class Config:
    hidden: int
    layers: int
    vocab: int
    eps: float
    heads: int
    q_lora: int
    kv_lora: int
    qk_dim: int
    v_dim: int
    lin_heads: int
    lin_dim: int
    conv: int
    lower: float
    experts: int
    top_k: int
    moe_width: int
    shared_width: int
    dense_width: int
    routed_scale: float
    norm_topk: bool
    streams: int
    hc_iters: int
    hc_eps: float
    index_heads: int
    index_dim: int
    index_topk: int
    kpool: int
    limit: float
    kinds: list[str]           # per layer: "kda" or "dsa"
    mlp_kinds: list[str]       # per layer: "dense" or "moe"
    eos: tuple[int, ...]
    mtp_layers: int
    group_size: int
    bits: int

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        t = dict(raw.get("text_config") or raw)
        lin = dict(t.get("linear_attn_config") or {})
        quant = raw.get("quantization") or raw.get("quantization_config") or {}
        eos = t.get("eos_token_id", raw.get("eos_token_id"))
        eos = tuple(int(e) for e in eos) if isinstance(eos, list) else (int(eos),)
        n = int(t["num_hidden_layers"])
        kinds = ["kda" if k == "linear_attention" else "dsa" for k in t["layer_types"]]
        dense = int(t.get("first_k_dense_replace", 3))
        mlp_kinds = list(t.get("mlp_layer_types") or ["dense"] * dense + ["sparse"] * (n - dense))
        mlp_kinds = ["moe" if k == "sparse" else "dense" for k in mlp_kinds]
        return cls(
            hidden=int(t["hidden_size"]), layers=n, vocab=int(t["vocab_size"]), eps=float(t["rms_norm_eps"]),
            heads=int(t["num_attention_heads"]), q_lora=int(t["q_lora_rank"]), kv_lora=int(t["kv_lora_rank"]),
            qk_dim=int(t["qk_nope_head_dim"]) + int(t.get("qk_rope_head_dim", 0)), v_dim=int(t["v_head_dim"]),
            lin_heads=int(lin.get("num_heads", t.get("linear_num_heads", 64))),
            lin_dim=int(lin.get("head_dim", t.get("linear_head_dim", 128))),
            conv=int(lin.get("short_conv_kernel_size", t.get("linear_conv_kernel_dim", 4))),
            lower=float(lin.get("gate_lower_bound", t.get("linear_lower_bound", -5.0))),
            experts=int(t["n_routed_experts"]), top_k=int(t["num_experts_per_tok"]),
            moe_width=int(t["moe_intermediate_size"]),
            shared_width=int(t["moe_intermediate_size"]) * int(t.get("n_shared_experts", 1)),
            dense_width=int(t["intermediate_size"]), routed_scale=float(t["routed_scaling_factor"]),
            norm_topk=bool(t.get("norm_topk_prob", True)), streams=int(t.get("hc_mult", 4)),
            hc_iters=int(t.get("hc_sinkhorn_iters", 20)), hc_eps=float(t.get("hc_eps", 1e-6)),
            index_heads=int(t.get("index_n_heads", 32)), index_dim=int(t.get("index_head_dim", 128)),
            index_topk=int(t.get("index_topk", 2048)), kpool=int(t.get("index_kpool", 4)),
            limit=float(t.get("swiglu_limit", 10.0)), kinds=kinds, mlp_kinds=mlp_kinds, eos=eos,
            mtp_layers=int(t.get("num_nextn_predict_layers", 0)), group_size=int(quant.get("group_size", 64)),
            bits=int(quant.get("bits", 4)),
        )

    @property
    def dense_limit(self) -> int:
        """Largest context (tokens) where DSA's top-k selection keeps every visible key (dense attention)."""

        return self.index_topk + self.kpool - 1


@dataclass
class HCW:
    fn: torch.Tensor          # [24, S*D] bf16
    base: torch.Tensor        # [24] fp32
    scale: torch.Tensor       # [3] fp32


@dataclass
class KDAW:
    proj: Q4                  # [q | k | v | f_a | g_a | b]
    fb: Q4
    gb: Q4
    conv: torch.Tensor        # [3 HL 128, taps] bf16
    a_log: torch.Tensor       # [HL] fp32
    dt_bias: torch.Tensor     # [HL 128] fp32
    norm: torch.Tensor        # [128] bf16
    o: Q4
    heads: int

    @property
    def fa_off(self) -> int:
        return 3 * self.heads * 128

    @property
    def ga_off(self) -> int:
        return self.fa_off + 128

    @property
    def b_off(self) -> int:
        return self.ga_off + 128


@dataclass
class IndexW:
    """DSA's indexer (replicated on every rank): [wk | weights_proj] on x, wq_b on the query residual, the key
    LayerNorm, the pool gate (bf16, 128 x D) and the pool position bias."""

    kw: Q4
    qb: Q4
    ln_w: torch.Tensor
    ln_b: torch.Tensor
    gate: torch.Tensor
    ape: torch.Tensor


@dataclass
class DSAW:
    proj: Q4                  # [q_a | kv_a]
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    q_b: Q4
    kv_k: Q4                  # key rows of kv_b for the local heads
    kv_v: Q4                  # value rows
    o: Q4
    heads: int
    index: IndexW | None = None


@dataclass
class MLPW:
    gu: Q4                    # [gate | up]
    down: Q4
    width: int


@dataclass
class MoEW:
    router: torch.Tensor      # [E, D] bf16
    bias: torch.Tensor        # [E] fp32
    experts: Experts          # E + 1 (shared expert last)


@dataclass
class LayerW:
    index: int
    kind: str
    attn_hc: HCW | None
    ffn_hc: HCW | None
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    kda: KDAW | None = None
    dsa: DSAW | None = None
    mlp: MLPW | None = None
    moe: MoEW | None = None


@dataclass
class MTPW:
    enorm: torch.Tensor
    hnorm: torch.Tensor
    eh: Q4                    # [D, 2D]: input [embedding | hidden]
    norm: torch.Tensor        # shared_head.norm
    layer: LayerW             # DSA + MoE, plain residual (no hyper-connections)


@dataclass
class Weights:
    cfg: Config
    embed: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    layers: list[LayerW]
    norm: torch.Tensor
    head: Q4
    mtp: MTPW | None
    rank: int
    world: int
    device: torch.device
    comm: Any = None
    meta: dict = field(default_factory=dict)

    @property
    def vocab_offset(self) -> int:
        return self.rank * (self.cfg.vocab // self.world)

    def nbytes(self) -> int:
        total = 0
        seen = set()

        def add(t):
            nonlocal total
            if isinstance(t, torch.Tensor) and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                total += t.numel() * t.element_size()
            elif isinstance(t, (Q4, Experts, HCW, KDAW, DSAW, MLPW, MoEW, LayerW, MTPW, IndexW)):
                for v in vars(t).values():
                    add(v)
            elif isinstance(t, (list, tuple)):
                for v in t:
                    add(v)
            elif isinstance(t, dict):
                for v in t.values():
                    add(v)

        add(self.embed)
        add(self.layers)
        add(self.norm)
        add(self.head)
        add(self.mtp)
        return total


def load(model_dir: str | Path, *, rank: int, device: str = "cuda") -> Weights:
    """Rank ``rank`` of two: its share of every layer, the MTP layer and its half of the head's vocabulary.
    ``model_dir``: the checkpoint, or a folder ``split`` wrote for this rank."""

    from .split import RankReader

    world = 2
    cfg = Config.read(model_dir)
    dev = torch.device(device)
    rd = RankReader(model_dir, rank)
    HL = cfg.heads // world
    LL = cfg.lin_heads // world

    def t(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        x = rd.get(PREFIX + name)
        if dtype is not None:
            x = x.to(dtype)
        return x.to(dev)

    def trip(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (as_i32(t(name + ".weight")), t(name + ".scales"), t(name + ".biases"))

    def q4(name: str) -> Q4:
        return make_q4(*trip(name))

    def hc(i: int, site: str) -> HCW:
        return HCW(t(f"layers.{i}.hc_{site}_fn").contiguous(), t(f"layers.{i}.hc_{site}_base", torch.float32),
                   t(f"layers.{i}.hc_{site}_scale", torch.float32))

    def kda(i: int) -> KDAW:
        p = f"layers.{i}.self_attn."
        proj = stack_q4([trip(p + "q_proj"), trip(p + "k_proj"), trip(p + "v_proj"), trip(p + "f_a_proj"),
                         trip(p + "g_a_proj"), trip(p + "b_proj")])
        conv = torch.cat([t(p + f"{x}_conv1d.weight") for x in "qkv"]).reshape(3 * LL * 128, cfg.conv).contiguous()
        return KDAW(proj, q4(p + "f_b_proj"), q4(p + "g_b_proj"), conv, t(p + "A_log", torch.float32).contiguous(),
                    t(p + "dt_bias", torch.float32).contiguous(), t(p + "o_norm.weight"), q4(p + "o_proj"), LL)

    def dsa(i: int) -> DSAW:
        p = f"layers.{i}.self_attn."
        proj = stack_q4([trip(p + "q_a_proj"), trip(p + "kv_a_proj_with_mqa")])
        w, s, b = trip(p + "kv_b_proj")
        rows = torch.arange(HL * 512, device=dev).view(HL, 512)
        krows, vrows = rows[:, :cfg.qk_dim].reshape(-1), rows[:, cfg.qk_dim:].reshape(-1)
        kv_k = make_q4(w[krows], s[krows], b[krows])
        kv_v = make_q4(w[vrows], s[vrows], b[vrows])
        ix = IndexW(stack_q4([trip(p + "indexer.wk"), trip(p + "indexer.weights_proj")]), q4(p + "indexer.wq_b"),
                    t(p + "indexer.k_norm.weight"), t(p + "indexer.k_norm.bias"),
                    t(p + "indexer.index_kpool_compress_gate", torch.bfloat16).contiguous(),
                    t(p + "indexer.index_kpool_compress_ape", torch.bfloat16).contiguous())
        return DSAW(proj, t(p + "q_a_layernorm.weight"), t(p + "kv_a_layernorm.weight"), q4(p + "q_b_proj"),
                    kv_k, kv_v, q4(p + "o_proj"), HL, ix)

    def mlp(i: int) -> MLPW:
        p = f"layers.{i}.mlp."
        gu = stack_q4([trip(p + "gate_proj"), trip(p + "up_proj")])
        return MLPW(gu, q4(p + "down_proj"), gu.n // 2)

    def moe(i: int) -> MoEW:
        p = f"layers.{i}.mlp."
        parts = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            ws, ss, bs = [], [], []
            for e in range(cfg.experts):
                ws.append(as_i32(rd.get(PREFIX + p + f"experts.{e}.{proj}.weight")))
                ss.append(rd.get(PREFIX + p + f"experts.{e}.{proj}.scales"))
                bs.append(rd.get(PREFIX + p + f"experts.{e}.{proj}.biases"))
            ws.append(as_i32(rd.get(PREFIX + p + f"shared_experts.{proj}.weight")))
            ss.append(rd.get(PREFIX + p + f"shared_experts.{proj}.scales"))
            bs.append(rd.get(PREFIX + p + f"shared_experts.{proj}.biases"))
            parts[proj] = (torch.stack(ws).to(dev), torch.stack(ss).to(dev), torch.stack(bs).to(dev))
        ex = make_experts(parts["gate_proj"], parts["up_proj"], parts["down_proj"])
        del parts
        return MoEW(t(p + "gate.weight", torch.bfloat16).contiguous(),
                    t(p + "gate.e_score_correction_bias", torch.float32).contiguous(), ex)

    def layer(i: int, plain: bool = False) -> LayerW:
        kind = "dsa" if plain else cfg.kinds[i]
        mk = "moe" if plain else cfg.mlp_kinds[i]
        lw = LayerW(i, kind, None if plain else hc(i, "attn"), None if plain else hc(i, "ffn"),
                    t(f"layers.{i}.input_layernorm.weight"), t(f"layers.{i}.post_attention_layernorm.weight"))
        if kind == "kda":
            lw.kda = kda(i)
        else:
            lw.dsa = dsa(i)
        if mk == "dense":
            lw.mlp = mlp(i)
        else:
            lw.moe = moe(i)
        torch.cuda.empty_cache()
        return lw

    embed = (as_i32(rd.get(PREFIX + "embed_tokens.weight")).to(dev), rd.get(PREFIX + "embed_tokens.scales").to(dev),
             rd.get(PREFIX + "embed_tokens.biases").to(dev))
    which = list(range(cfg.layers))
    built = [layer(i) for i in which]
    vl = cfg.vocab // world
    hw, hs, hb = (rd.get("lm_head." + x) for x in ("weight", "scales", "biases"))
    head = make_q4(as_i32(hw[rank * vl:(rank + 1) * vl]).to(dev), hs[rank * vl:(rank + 1) * vl].to(dev),
                   hb[rank * vl:(rank + 1) * vl].to(dev))
    mtpw = None
    if cfg.mtp_layers:
        i = cfg.layers
        mtpw = MTPW(t(f"layers.{i}.enorm.weight"), t(f"layers.{i}.hnorm.weight"), q4(f"layers.{i}.eh_proj"),
                    t(f"layers.{i}.shared_head.norm.weight"), layer(i, plain=True))
    w = Weights(cfg, embed, built, t("norm.weight"), head, mtpw, rank, world, dev)
    w.meta.update(layers=which)
    torch.cuda.empty_cache()
    return w
