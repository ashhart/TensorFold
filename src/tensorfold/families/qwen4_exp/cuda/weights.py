"""Pack affine group-32 MLX weights with the shared expert last, concatenate n-gram shards in order, and retain centered-norm gamma itself as fp32 after checking it is around one."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..host_table import BF16Table, HostTable, open_table, read_header as _header
from ..ssd_table import SSDTable
from .bf16 import b16_from_rows, make_b16, quantize4, stack_b16
from .ngram import NGram
from tensorfold.cuda import experts as grouped

from .qmm import Q4, dequantize, make_q4, stack_q4
from .reader import _DT, _Reader, _groups, _rows, _rows_at, norms_around_one  # noqa: F401


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
    """The token ids the MTP drafts' head scores, sorted: "default" (``draft_vocab.txt`` beside this module), a file of ids, an int N (ids below N), or None (the full vocabulary)."""

    if not draft_vocab:
        return None
    if isinstance(draft_vocab, int):
        return np.arange(draft_vocab, dtype=np.int64)
    source = Path(__file__).with_name("draft_vocab.txt") if draft_vocab == "default" else Path(draft_vocab)
    return np.unique(np.loadtxt(source, dtype=np.int64).reshape(-1))


def load(model_dir: str | Path, device: str = "cuda", *, mtp: bool = True, tp: tuple[int, int] | None = None,
         draft_vocab: int | str | None = None, ple_on_ssd: bool = False) -> Weights:
    """Load rank ``tp``'s head, expert-width and vocabulary shares while replicating other weights; ``draft_vocab`` restricts draft scoring to default/file ids or ids below N, with None using all ids."""

    import time
    from dataclasses import replace

    from . import exl3

    model_dir = Path(model_dir)
    if exl3.is_exl3(model_dir):                       # an EXL3 pack: its own loader, the same dataclasses
        return exl3.load(model_dir, device, mtp=mtp, tp=tp, draft_vocab=draft_vocab)
    full = Config.read(model_dir)
    rank, world = tp if tp is not None else (0, 1)
    cfg = full if world == 1 else replace(full, heads=full.heads // world, kv_heads=full.kv_heads // world,
                                          nk=full.nk // world, nv=full.nv // world,
                                          moe_width=full.moe_width // world, shared_width=full.shared_width // world)
    rd = _Reader(model_dir, device)
    prefix = "language_model." if rd.has("language_model.model.embed_tokens.weight") else ""
    # NVFP4 names the language model ``model.language_model.*``; its lm_head and mtp sit at the top level
    mbase = "model.language_model." if rd.has("model.language_model.embed_tokens.weight") else "model."
    chosen = list(range(cfg.layers))
    around_one = norms_around_one(rd, prefix + mbase, chosen)

    def raw(name: str) -> torch.Tensor:
        return rd.get(prefix + name)

    def triple(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = raw(name + ".weight")
        return (w.view(torch.int32) if w.dtype != torch.int32 else w), raw(name + ".scales"), raw(name + ".biases")

    def q4(name: str) -> Q4:
        return make_q4(*triple(name))

    def cscale(name: str) -> torch.Tensor:
        w = raw(name).float()
        return (w if around_one else 1.0 + w).contiguous()

    def hc(name: str, inject: bool) -> HC:
        parts = [triple(name + ".input_mix_weight_down")]
        if inject:
            parts.append(triple(name + ".block_inject_weight"))
        up = triple(name + ".input_mix_weight_up")
        return HC(stack_q4(parts, "tiled"), make_q4(*up, "tiled"), cscale(name + ".hc_norm.weight"), inject,
                  stack_q4(parts, "frag"), make_q4(*up, "frag"))

    def b16(name: str):
        """One linear as the NVFP4 checkpoint stores it (BF16, torch layout [out, in]), on the qmm matmul face."""
        return b16_from_rows(raw(name + ".weight"))

    def dense(name: str, rows=None, cols: slice | None = None):
        """A linear's weight and its e8m0 scales (MXFP8) or None (bf16), a rank's rows or 32-aligned input columns."""

        w = raw(name + ".weight")
        s = raw(name + ".weight_scale") if w.dtype == torch.float8_e4m3fn else None
        if s is not None and s.dtype != torch.uint8:
            raise ValueError(f"{name}: FP8 with a per-tensor scale; Flash Next reads MXFP8 (a scale every 32 inputs)")
        if rows is not None:
            w, s = w[rows], None if s is None else s[rows]
        if cols is not None:
            w, s = w[:, cols], None if s is None else s[:, cols.start // 32:cols.stop // 32]
        return w, s

    def face(*parts):
        """Linears of one input as one face by their storage: bf16 rows on ``bf16.matmul``, MXFP8 on the lane matmul."""

        got = [dense(*p) for p in parts]
        if all(s is None for _, s in got):
            faces = [b16_rows(w.to(torch.bfloat16)) for w, _ in got]
            return faces[0] if len(faces) == 1 else stack_b16(faces)
        if all(s is not None for _, s in got):
            from tensorfold.cuda.nvfp4.linear import Mx8Linear

            return Mx8Linear.from_checkpoint(torch.cat([w for w, _ in got]), torch.cat([s for _, s in got]))
        raise ValueError(f"{parts[0][0]}: a projection stack mixes MXFP8 and bf16 weights")

    def hc_nvfp4(name: str, inject: bool) -> HC:
        parts = [b16(name + ".input_mix_weight_down")]
        if inject:
            parts.append(b16(name + ".block_inject_weight"))
        return HC(stack_b16(parts), b16(name + ".input_mix_weight_up"), cscale(name + ".hc_norm.weight"), inject)

    def gdn_nvfp4(name: str) -> GDNW:
        """A DeltaNet block from the NVFP4 checkpoint (bf16 or MXFP8 linears): the rank's rows, conv and head vectors."""

        kl, vl = full.nk // world, full.nv // world
        dk, dv = full.dk, full.dv
        q_rows = torch.arange(rank * kl * dk, (rank + 1) * kl * dk, device=device)
        v_rows = 2 * full.nk * dk + torch.arange(rank * vl * dv, (rank + 1) * vl * dv, device=device)
        channels = torch.cat([q_rows, full.nk * dk + q_rows, v_rows])
        one = world == 1
        proj = face((name + ".in_proj_qkv", None if one else channels),
                    (name + ".in_proj_z", None if one else slice(rank * vl * dv, (rank + 1) * vl * dv)),
                    (name + ".in_proj_b", None if one else slice(rank * vl, (rank + 1) * vl)),
                    (name + ".in_proj_a", None if one else slice(rank * vl, (rank + 1) * vl)))
        conv = raw(name + ".conv1d.weight").reshape(full.conv_dim, full.conv_kernel).to(torch.bfloat16)
        conv = conv.index_select(0, channels).contiguous()
        return GDNW(proj, conv, raw(name + ".A_log").float()[rank * vl:(rank + 1) * vl].contiguous(),
                    raw(name + ".dt_bias").float()[rank * vl:(rank + 1) * vl].contiguous(),
                    raw(name + ".norm.weight").to(torch.bfloat16).contiguous(),
                    face((name + ".out_proj", None, None if one else slice(rank * vl * dv, (rank + 1) * vl * dv))))

    def attention_nvfp4(name: str) -> AttnW:
        """An attention block from the NVFP4 checkpoint (bf16 or MXFP8 linears)."""

        hd = full.head_dim
        hl, kl = full.heads // world, full.kv_heads // world
        one = world == 1
        proj = face((name + ".q_proj", None if one else slice(rank * hl * 2 * hd, (rank + 1) * hl * 2 * hd)),
                    (name + ".k_proj", None if one else slice(rank * kl * hd, (rank + 1) * kl * hd)),
                    (name + ".v_proj", None if one else slice(rank * kl * hd, (rank + 1) * kl * hd)),
                    (name + ".indexer.index_qk_proj", None))
        o = face((name + ".o_proj", None, None if one else slice(rank * hl * hd, (rank + 1) * hl * hd)))
        return AttnW(proj, cscale(name + ".q_norm.weight"), cscale(name + ".k_norm.weight"),
                     cscale(name + ".indexer.q_layernorm.weight"), cscale(name + ".indexer.k_layernorm.weight"),
                     o)

    def ple_nvfp4(name: str, ple_index: int) -> PLEW:
        """A PLE layer from the NVFP4 checkpoint: n-gram rows from bf16, FP8, NVFP4 or MLX 4-bit shards, the rest bf16."""

        if ple_on_ssd:
            raise ValueError("--ple-on-ssd reads the MLX checkpoint's n-gram shards from disk; an NVFP4 checkpoint's "
                             "tables stay memory-mapped, so drop --ple-on-ssd")
        ngram = cfg.ngram(ple_index)
        base = name + ".ple_embedding."
        ngram.check(raw(base + "layer_multipliers").cpu().numpy(), raw(base + "ngram_heads_offsets").cpu().numpy(),
                    raw(base + "ngram_heads_vocab_sizes").cpu().numpy())
        keys = [prefix + base + f"ngram_embedding.shard_{i}" for i in range(cfg.ngram_shards)]
        table = open_table(model_dir, [(rd.where[k + ".weight"], k) for k in keys],
                           lambda n: float(raw(base + "ngram_embedding." + n).float().reshape(-1)[0]))
        if getattr(table, "width", ngram.dims) != ngram.dims:
            raise ValueError(f"the n-gram rows hold {table.width} values, expected {ngram.dims}")
        if table.rows != ngram.rows:
            raise ValueError(f"n-gram tables hold {table.rows} rows, expected {ngram.rows}")
        conv = raw(name + ".conv1d.weight").reshape(cfg.streams * cfg.hidden, cfg.ple_kernel).to(torch.bfloat16)
        return PLEW(table, b16(name + ".key_proj"), b16(name + ".value_proj"),
                    cscale(name + ".norm_key.weight"), cscale(name + ".norm_query.weight"),
                    cscale(name + ".norm_conv.weight"), conv.contiguous(), ngram)

    def b16_rows(t: torch.Tensor):
        return b16_from_rows(t.to(torch.bfloat16).contiguous())

    def moe(name: str) -> MoEW:
        gate_rows = raw(name + ".gate.weight").to(torch.bfloat16)
        sw, ss, sb = triple(name + ".shared_expert_gate")
        shared_gate = dequantize(sw, ss, sb).to(torch.bfloat16)
        router = torch.cat([gate_rows, shared_gate]).contiguous()
        w_, sw_ = full.moe_width, full.shared_width
        # a rank takes its half of every expert's intermediate width: gate/up rows, down input groups
        gu = lambda t, width: _rows(t, rank * width // world, (rank + 1) * width // world)          # noqa: E731
        dn = lambda t, width: _groups(t, rank * width // world // 32, (rank + 1) * width // world // 32)  # noqa: E731
        def table(routed, shared):
            return tuple(torch.cat([r, t[None]]) for r, t in zip(routed, shared))

        experts = grouped.make([table(gu(triple(name + ".switch_mlp.gate_proj"), w_),
                                      gu(triple(name + ".shared_expert.gate_proj"), sw_)),
                                table(gu(triple(name + ".switch_mlp.up_proj"), w_),
                                      gu(triple(name + ".shared_expert.up_proj"), sw_))],
                               table(dn(triple(name + ".switch_mlp.down_proj"), w_),
                                     dn(triple(name + ".shared_expert.down_proj"), sw_)), 32)
        return MoEW(router, experts)

    def moe_nvfp4(name: str) -> MoEW:
        """The NVFP4 checkpoint's MoE: FP4 routed experts (bf16 in the MTP layer), the bf16 shared expert and gate."""

        from . import nvfp4_moe

        router = raw(name + ".gate.weight").to(torch.bfloat16)
        sgate = raw(name + ".shared_expert_gate.weight").to(torch.bfloat16).reshape(full.hidden).contiguous()
        router = torch.cat([router, sgate[None]]).contiguous()
        e = full.experts
        w_, sw_ = full.moe_width, full.shared_width
        gs = full.nvfp4_group
        lo, hi = rank * w_ // world, (rank + 1) * w_ // world
        dlo, dhi = rank * w_ // world // gs, (rank + 1) * w_ // world // gs
        se = f"{name}.shared_expert."
        if raw(se + "gate_proj.weight").dtype == torch.float8_e4m3fn:     # MXFP8: its own lane-matmul faces
            shared = nvfp4_moe.Expert4(face((se + "gate_proj", slice(lo, hi)), (se + "up_proj", slice(lo, hi))),
                                       face((se + "down_proj", None, slice(dlo * gs, dhi * gs))))
        else:
            shared = (raw(se + "gate_proj.weight").to(torch.bfloat16)[lo:hi].contiguous(),
                      raw(se + "up_proj.weight").to(torch.bfloat16)[lo:hi].contiguous(),
                      raw(se + "down_proj.weight").to(torch.bfloat16)[:, dlo * gs:dhi * gs].contiguous())
        if rd.has(prefix + f"{name}.experts.0.gate_proj.weight"):      # the main layers: per-expert FP4
            def stack(proj: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                w = torch.stack([raw(f"{name}.experts.{i}.{proj}.weight") for i in range(e)])
                s = torch.stack([raw(f"{name}.experts.{i}.{proj}.weight_scale") for i in range(e)])
                s2 = torch.stack([raw(f"{name}.experts.{i}.{proj}.weight_scale_2") for i in range(e)])
                return w, s, s2

            gate = stack("gate_proj")
            up = stack("up_proj")
            down = stack("down_proj")
            if world > 1:
                gate = (gate[0][:, lo:hi], gate[1][:, lo:hi], gate[2])
                up = (up[0][:, lo:hi], up[1][:, lo:hi], up[2])
                down = (down[0][:, :, dlo * gs // 2:dhi * gs // 2], down[1][:, :, dlo:dhi], down[2])   # 8 bytes a block
            moe4 = nvfp4_moe.moe4_from_checkpoint(gate, up, down, shared)
            del gate, up, down                           # the stacks are dead once the grids are tiled
        else:                                            # the MTP layer: BF16 stacked experts (excluded)
            gu = raw(name + ".experts.gate_up_proj").to(torch.bfloat16)          # [E, 2*NI, D]
            dn = raw(name + ".experts.down_proj").to(torch.bfloat16)             # [E, D, NI]
            if world > 1:
                gu, dn = gu[:, lo:hi], dn[:, :, dlo * gs:dhi * gs]
            moe4 = nvfp4_moe.moe4_from_bf16(gu, dn, shared)
        return MoEW(router, moe4)

    def attention(name: str) -> AttnW:
        hd = full.head_dim
        hl, kl = full.heads // world, full.kv_heads // world
        q = _rows(triple(name + ".q_proj"), rank * hl * 2 * hd, (rank + 1) * hl * 2 * hd)
        k = _rows(triple(name + ".k_proj"), rank * kl * hd, (rank + 1) * kl * hd)
        v = _rows(triple(name + ".v_proj"), rank * kl * hd, (rank + 1) * kl * hd)
        proj = stack_q4([q, k, v, triple(name + ".indexer.index_qk_proj")])          # the indexer: every rank
        o = _groups(triple(name + ".o_proj"), rank * hl * hd // 32, (rank + 1) * hl * hd // 32)
        return AttnW(proj, cscale(name + ".q_norm.weight"), cscale(name + ".k_norm.weight"),
                     cscale(name + ".indexer.q_layernorm.weight"), cscale(name + ".indexer.k_layernorm.weight"),
                     make_q4(*o))

    def gdn(name: str) -> GDNW:
        kl, vl = full.nk // world, full.nv // world
        dk, dv = full.dk, full.dv
        q_rows = torch.arange(rank * kl * dk, (rank + 1) * kl * dk, device=device)
        v_rows = 2 * full.nk * dk + torch.arange(rank * vl * dv, (rank + 1) * vl * dv, device=device)
        channels = torch.cat([q_rows, full.nk * dk + q_rows, v_rows])
        qkv = _rows_at(triple(name + ".in_proj_qkv"), channels)
        z = _rows(triple(name + ".in_proj_z"), rank * vl * dv, (rank + 1) * vl * dv)
        bb = _rows(triple(name + ".in_proj_b"), rank * vl, (rank + 1) * vl)
        aa = _rows(triple(name + ".in_proj_a"), rank * vl, (rank + 1) * vl)
        proj = stack_q4([qkv, z, bb, aa])
        conv = raw(name + ".conv1d.weight").reshape(full.conv_dim, full.conv_kernel).to(torch.bfloat16)
        conv = conv.index_select(0, channels).contiguous()
        out = _groups(triple(name + ".out_proj"), rank * vl * dv // 32, (rank + 1) * vl * dv // 32)
        return GDNW(proj, conv, raw(name + ".A_log").float()[rank * vl:(rank + 1) * vl].contiguous(),
                    raw(name + ".dt_bias").float()[rank * vl:(rank + 1) * vl].contiguous(),
                    raw(name + ".norm.weight").to(torch.bfloat16).contiguous(), make_q4(*out))

    def ple_layer(name: str, ple_index: int) -> PLEW:
        ngram = cfg.ngram(ple_index)
        base = name + ".ple_embedding."
        ngram.check(raw(base + "layer_multipliers").cpu().numpy(), raw(base + "ngram_heads_offsets").cpu().numpy(),
                    raw(base + "ngram_heads_vocab_sizes").cpu().numpy())
        files = []
        headers: dict[str, dict] = {}
        for i in range(cfg.ngram_shards):
            key = prefix + base + f"ngram_embedding.shard_{i}"
            shard = rd.where[key + ".weight"]
            if shard not in headers:
                headers[shard] = _header(model_dir / shard)
            h = headers[shard]
            files.append((model_dir / shard, h[key + ".weight"], h[key + ".scales"], h[key + ".biases"]))
        table = SSDTable(files) if ple_on_ssd else HostTable(files)
        if table.rows != ngram.rows:
            raise ValueError(f"n-gram tables hold {table.rows} rows, expected {ngram.rows}")
        conv = raw(name + ".conv1d.weight").reshape(cfg.streams * cfg.hidden, cfg.ple_kernel).to(torch.bfloat16)
        return PLEW(table, q4(name + ".key_proj"), q4(name + ".value_proj"),
                    cscale(name + ".norm_key.weight"), cscale(name + ".norm_query.weight"),
                    cscale(name + ".norm_conv.weight"), conv.contiguous(), ngram)

    def layer(i: int, base: str, kind: str, with_ple: bool) -> LayerW:
        linear = kind == "linear"
        nvfp4 = cfg.quant == "modelopt"                 # NVFP4: routed experts FP4, every other linear BF16
        entry = LayerW(i, linear,
                       (hc_nvfp4 if nvfp4 else hc)(base + ".attn_hyper_connection", True),
                       (hc_nvfp4 if nvfp4 else hc)(base + ".mlp_hyper_connection", True),
                       (gdn_nvfp4 if nvfp4 else gdn)(base + ".linear_attn") if linear else None,
                       None if linear else (attention_nvfp4 if nvfp4 else attention)(base + ".self_attn"),
                       (moe_nvfp4 if nvfp4 else moe)(base + ".mlp"))
        if with_ple and i in cfg.ple_layers:
            entry.ple = (ple_nvfp4 if nvfp4 else ple_layer)(base + ".ple", cfg.ple_layers.index(i))
        return entry

    t0 = time.time()
    if cfg.quant not in ("mlx", "modelopt"):
        raise ValueError(f"Flash Next's CUDA engine reads MLX 4-bit (groups of 32) or NVFP4 (experts-only) "
                         f"checkpoints, not {cfg.quant}")
    embed = ((raw(mbase + "embed_tokens.weight").to(torch.bfloat16).contiguous(),) if cfg.quant == "modelopt"
             else triple("model.embed_tokens"))
    loaded = []
    for i in chosen:
        loaded.append(layer(i, f"{mbase}layers.{i}", cfg.layer_types[i], True))
        rd.release()
        torch.cuda.empty_cache()
    mixer = (hc_nvfp4 if cfg.quant == "modelopt" else hc)(mbase + "hyper_connection_mixer", False)
    vl = full.vocab // world
    if cfg.quant == "modelopt":
        head = b16_rows(raw("lm_head.weight").to(torch.bfloat16)[rank * vl:(rank + 1) * vl])
    else:
        head_raw = triple("lm_head")
        head = make_q4(*_rows(head_raw, rank * vl, (rank + 1) * vl))
        del head_raw
    # NVFP4: the draft head is the bf16 lm_head's draft rows requantized 4-bit at load (drafts only)
    draft_head, draft_ids = None, None
    ids = draft_token_ids(draft_vocab)
    if ids is not None:
        ids = np.array_split(ids[ids < full.vocab], world)[rank]
        ids = torch.from_numpy(ids).to(device)
        draft_ids = ids
        if cfg.quant == "modelopt":
            draft_head = quantize4(raw("lm_head.weight").index_select(0, ids).to(torch.bfloat16))
        else:
            draft_head = make_q4(*_rows_at(triple("lm_head"), ids))
    inv = torch.tensor(cfg.rope_theta, dtype=torch.float64) ** (
        -torch.arange(0, cfg.rotary_dim // 2, dtype=torch.float64) / (cfg.rotary_dim // 2))
    w = Weights(cfg, embed, loaded, mixer, head, inv.to(torch.float32).to(device), around_one=around_one)
    w.meta.update(rank=rank, world=world, vocab_offset=rank * vl, full=full)
    w.draft_head, w.draft_ids = draft_head, draft_ids
    if mtp and rd.has(prefix + "mtp.fc_embedding.weight"):
        fc = b16 if cfg.quant == "modelopt" else q4
        w.mtp = MTPW(cscale("mtp.pre_fc_norm_embedding.weight"), cscale("mtp.pre_fc_norm_hidden.weight"),
                     fc("mtp.fc_embedding"), fc("mtp.fc_hidden"),
                     layer(-1, "mtp.layers.0", "attention", False),
                     (hc_nvfp4 if cfg.quant == "modelopt" else hc)("mtp.hyper_connection_mixer", False))
    rd.release()
    torch.cuda.empty_cache()
    w.meta["load_seconds"] = time.time() - t0
    return w
