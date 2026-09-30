"""Load one rank of MLX affine 4-bit weights or EXL3 routed experts with BF16 elsewhere, preserving heads and quantization groups at split boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tensorfold.cuda import experts as grouped

from .exl3_mm import Exl3Experts, words as exl3_words
from . import latent
from .qmm import B16, Q4, as_i32, make_b16, make_q4, quantize4, stack_b16, stack_q4

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
    quant: str = "mlx"         # "mlx" (affine 4-bit everywhere) or "exl3" (EXL3 routed experts, BF16 elsewhere)

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        want = raw.get("tensorfold_activation_dtype")
        if want not in (None, "bfloat16", "float32"):
            raise ValueError(f"tensorfold_activation_dtype {want!r}: bfloat16 or float32")
        if want == "float32":
            raise ValueError("tensorfold_activation_dtype float32 is the Mac engine; the CUDA engine stays bf16")
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
            bits=int(quant.get("bits", 4)), quant=str(quant.get("quant_method") or "mlx").lower(),
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
    """Replicated DSA indexer weights include key and query projections, head weights, key LayerNorm, pool gates, and pool position bias."""

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
    absorb: object = None     # latent.AbsorbW: kv_b split per head, for attention on the latent cache


@dataclass
class MLPW:
    gu: Q4                    # [gate | up]
    down: Q4
    width: int


@dataclass
class MoEW:
    router: torch.Tensor      # [E, D] bf16
    bias: torch.Tensor        # [E] fp32
    experts: grouped.Experts | Exl3Experts  # 4-bit: E + 1 (shared expert last); EXL3: the E routed experts
    shared: MLPW | None = None            # EXL3 checkpoints: the shared expert (BF16)


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
    head: Q4 | B16
    mtp: MTPW | None
    rank: int
    world: int
    device: torch.device
    comm: Any = None
    meta: dict = field(default_factory=dict)
    draft_head: Q4 | None = None      # BF16 heads: a 4-bit copy the draft steps read (drafts only propose)

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
            elif isinstance(t, (Q4, B16, grouped.Experts, Exl3Experts, HCW, KDAW, DSAW, MLPW, MoEW, LayerW, MTPW, IndexW)):
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
        add(self.draft_head)
        add(self.mtp)
        return total


def load(model_dir: str | Path, *, rank: int, device: str = "cuda", mtp: bool = True) -> Weights:
    """One of two ranks from a checkpoint or rank folder, MTP included unless ``mtp`` is False, with its head half."""

    from .split import RankReader

    world = 2
    cfg = Config.read(model_dir)
    if cfg.quant not in ("mlx", "exl3"):
        raise ValueError(f"GLM-5.3-Flash's CUDA engine reads MLX 4-bit or EXL3 checkpoints, not {cfg.quant}")
    exl3 = cfg.quant == "exl3"
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

    def q4(name: str) -> Q4 | B16:
        return make_b16(t(name + ".weight")) if exl3 else make_q4(*trip(name))

    def stack(names: list[str]) -> Q4 | B16:
        if exl3:
            return stack_b16([t(n + ".weight") for n in names])
        return stack_q4([trip(n) for n in names])

    def hc(i: int, site: str) -> HCW:
        return HCW(t(f"layers.{i}.hc_{site}_fn").contiguous(), t(f"layers.{i}.hc_{site}_base", torch.float32),
                   t(f"layers.{i}.hc_{site}_scale", torch.float32))

    def kda(i: int) -> KDAW:
        p = f"layers.{i}.self_attn."
        proj = stack([p + "q_proj", p + "k_proj", p + "v_proj", p + "f_a_proj", p + "g_a_proj", p + "b_proj"])
        conv = torch.cat([t(p + f"{x}_conv1d.weight") for x in "qkv"]).reshape(3 * LL * 128, cfg.conv).contiguous()
        return KDAW(proj, q4(p + "f_b_proj"), q4(p + "g_b_proj"), conv, t(p + "A_log", torch.float32).contiguous(),
                    t(p + "dt_bias", torch.float32).contiguous(), t(p + "o_norm.weight"), q4(p + "o_proj"), LL)

    def dsa(i: int) -> DSAW:
        p = f"layers.{i}.self_attn."
        proj = stack([p + "q_a_proj", p + "kv_a_proj_with_mqa"])
        rows = torch.arange(HL * 512, device=dev).view(HL, 512)
        krows, vrows = rows[:, :cfg.qk_dim].reshape(-1), rows[:, cfg.qk_dim:].reshape(-1)
        if exl3:
            w = t(p + "kv_b_proj.weight")
            kv_k, kv_v = make_b16(w[krows]), make_b16(w[vrows])
            full_k, full_v = (lambda: w[krows].float()), (lambda: w[vrows].float())
        else:
            w, s, b = trip(p + "kv_b_proj")
            kv_k = make_q4(w[krows], s[krows], b[krows])
            kv_v = make_q4(w[vrows], s[vrows], b[vrows])
        if not latent.ENABLED:
            absorb = None
        elif exl3:
            absorb = latent.AbsorbW.from_rows(full_k(), full_v(), HL)
        elif cfg.group_size == 64:        # the checkpoint's own 4-bit rows, read as they are stored
            absorb = latent.AbsorbQ4((w[krows], s[krows], b[krows]), (w[vrows], s[vrows], b[vrows]), HL)
        else:
            absorb = latent.AbsorbW.from_rows(latent.dequant_mlx4(w[krows], s[krows], b[krows], cfg.group_size),
                                              latent.dequant_mlx4(w[vrows], s[vrows], b[vrows], cfg.group_size), HL)
        ix = IndexW(stack([p + "indexer.wk", p + "indexer.weights_proj"]), q4(p + "indexer.wq_b"),
                    t(p + "indexer.k_norm.weight"), t(p + "indexer.k_norm.bias"),
                    t(p + "indexer.index_kpool_compress_gate", torch.bfloat16).contiguous(),
                    t(p + "indexer.index_kpool_compress_ape", torch.bfloat16).contiguous())
        return DSAW(proj, t(p + "q_a_layernorm.weight"), t(p + "kv_a_layernorm.weight"), q4(p + "q_b_proj"),
                    kv_k, kv_v, q4(p + "o_proj"), HL, ix, absorb)

    def mlp(p: str) -> MLPW:
        gu = stack([p + "gate_proj", p + "up_proj"])
        return MLPW(gu, q4(p + "down_proj"), gu.n // 2)

    def expert_names(i: int) -> list[str]:
        """Layer ``i``'s expert tensors in the order ``moe`` reads them (none for a dense layer; ``cfg.layers``: MTP)."""

        mtp_layer = i == cfg.layers and cfg.mtp_layers and mtp
        if not mtp_layer and (i >= cfg.layers or cfg.mlp_kinds[i] != "moe"):
            return []
        p, parts = PREFIX + f"layers.{i}.mlp.", ("trellis", "suh", "svh") if exl3 else ("weight", "scales", "biases")
        names = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names += [p + f"experts.{e}.{proj}.{x}" for e in range(cfg.experts) for x in parts]
            if not exl3:
                names += [p + f"shared_experts.{proj}.{x}" for x in parts]
        return names

    def on_device(tensors: list[torch.Tensor]) -> torch.Tensor:
        """``torch.stack(tensors).to(dev)`` without a host copy: uploaded tensors stacked on the device, host ones copied into their slots."""

        if all(x.is_cuda for x in tensors):
            return torch.stack(tensors)
        out = torch.empty((len(tensors), *tensors[0].shape), dtype=tensors[0].dtype, device=dev)
        for slot, x in zip(out, tensors):
            slot.copy_(x)
        return out

    def moe_exl3(p: str) -> Exl3Experts:
        from tensorfold.cuda.exl3 import experts as generic
        gate = generic.prepare(
            [(rd.get(PREFIX + p + f"experts.{e}.gate_proj.trellis").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.gate_proj.suh").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.gate_proj.svh").to(dev)) for e in range(cfg.experts)],
            [(rd.get(PREFIX + p + f"experts.{e}.up_proj.trellis").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.up_proj.suh").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.up_proj.svh").to(dev)) for e in range(cfg.experts)],
            [(rd.get(PREFIX + p + f"experts.{e}.down_proj.trellis").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.down_proj.suh").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.down_proj.svh").to(dev)) for e in range(cfg.experts)],
            "mcg", device=dev)
        return gate

    def moe(i: int) -> MoEW:
        p = f"layers.{i}.mlp."
        router = t(p + "gate.weight", torch.bfloat16).contiguous()
        bias = t(p + "gate.e_score_correction_bias", torch.float32).contiguous()
        if exl3:
            return MoEW(router, bias, moe_exl3(p), mlp(p + "shared_experts."))
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
            parts[proj] = (on_device(ws), on_device(ss), on_device(bs))
            del ws, ss, bs
        ex = grouped.make([parts["gate_proj"], parts["up_proj"]], parts["down_proj"], 64, limit=cfg.limit)
        del parts
        return MoEW(router, bias, ex)

    layer_events: list = []                              # each layer's event, recorded once its work is queued

    def layer(i: int, plain: bool = False) -> LayerW:
        kind = "dsa" if plain else cfg.kinds[i]
        mk = "moe" if plain else cfg.mlp_kinds[i]
        if len(layer_events) >= 2:                       # at most two layers queued ahead of the GPU
            layer_events.pop(0).synchronize()
        up = None if exl3 else dev                       # MLX experts come uploaded (EXL3's are unpacked on the host)
        rd.prefetch(expert_names(i), up)                 # already queued, except for the first layer
        rd.prefetch(expert_names(i + 1), up)             # two layers in flight: reads overlap copies and packing
        lw = LayerW(i, kind, None if plain else hc(i, "attn"), None if plain else hc(i, "ffn"),
                    t(f"layers.{i}.input_layernorm.weight"), t(f"layers.{i}.post_attention_layernorm.weight"))
        if kind == "kda":
            lw.kda = kda(i)
        else:
            lw.dsa = dsa(i)
        if mk == "dense":
            lw.mlp = mlp(f"layers.{i}.mlp.")
        else:
            lw.moe = moe(i)
        if i % 8 == 7:                            # each release waits for the device; a layer leaves few temporaries
            torch.cuda.empty_cache()
        layer_events.append(torch.cuda.current_stream().record_event())
        return lw

    if exl3:
        embed = rd.get(PREFIX + "embed_tokens.weight").to(torch.bfloat16).contiguous().to(dev)
    else:
        embed = (as_i32(rd.get(PREFIX + "embed_tokens.weight")).to(dev), rd.get(PREFIX + "embed_tokens.scales").to(dev),
                 rd.get(PREFIX + "embed_tokens.biases").to(dev))
    try:                                          # a failed load still cancels the reads queued ahead
        which = list(range(cfg.layers))
        built = [layer(i) for i in which]
        vl = cfg.vocab // world
        draft_head = None
        if exl3:
            head = make_b16(rd.get("lm_head.weight")[rank * vl:(rank + 1) * vl].to(dev))
            # Draft steps use the quantized head; verification keeps the original head.
            draft_head = quantize4(head.weight)
        else:
            hw, hs, hb = (rd.get("lm_head." + x) for x in ("weight", "scales", "biases"))
            head = make_q4(as_i32(hw[rank * vl:(rank + 1) * vl]).to(dev), hs[rank * vl:(rank + 1) * vl].to(dev),
                           hb[rank * vl:(rank + 1) * vl].to(dev))
        mtpw = None
        if cfg.mtp_layers and mtp:
            i = cfg.layers
            mtpw = MTPW(t(f"layers.{i}.enorm.weight"), t(f"layers.{i}.hnorm.weight"), q4(f"layers.{i}.eh_proj"),
                        t(f"layers.{i}.shared_head.norm.weight"), layer(i, plain=True))
        w = Weights(cfg, embed, built, t("norm.weight"), head, mtpw, rank, world, dev, draft_head=draft_head)
        w.meta.update(layers=which)
    finally:
        rd.close()
    torch.cuda.empty_cache()
    return w
