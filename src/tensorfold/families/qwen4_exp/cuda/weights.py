"""Pack affine group-32 MLX weights with the shared expert last, concatenate n-gram shards in order, and retain centered-norm gamma itself as fp32 after checking it is around one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..host_table import open_table, shard_keys
from .bf16 import b16_from_rows, quantize4, stack_b16
from tensorfold.cuda import experts as grouped

from .qmm import Q4, dequantize, make_q4, stack_q4
from .reader import _DT, _Reader, _groups, _rows, _rows_at, norms_around_one  # noqa: F401
from .weight_types import (
    AttnW, Config, GDNW, HC, LayerW, MoEW, MTPW, PLEW, Weights, draft_token_ids, stop_ids)  # noqa: F401 (re-exported)


_PLAIN = (torch.bfloat16, torch.float16, torch.float32)


def _plain(name: str, w: torch.Tensor) -> torch.Tensor:
    """``w`` if it holds real values a bf16 linear can take, else a refusal: quantized bytes cast to bf16 decode garbage."""

    if w.dtype not in _PLAIN:
        raise ValueError(f"{name}: {str(w.dtype).removeprefix('torch.')} weights without a scale this loader reads; "
                         "Flash Next reads its non-expert linears as bf16, MXFP8 or 128x128-block FP8")
    return w


def load(model_dir: str | Path, device: str = "cuda", *, mtp: bool = True, tp: tuple[int, int] | None = None,
         draft_vocab: int | str | None = None, ple_on_ssd: bool = False, table_reads: list | None = None) -> Weights:
    """Load rank ``tp``'s head, expert-width and vocabulary shares while replicating other weights; ``draft_vocab`` restricts draft scoring to default/file ids or ids below N, with None using all ids."""

    import time
    from dataclasses import replace

    from . import exl3

    model_dir = Path(model_dir)
    if exl3.is_exl3(model_dir):                       # an EXL3 pack: its own loader, the same dataclasses
        return exl3.load(model_dir, device, mtp=mtp, tp=tp, draft_vocab=draft_vocab, table_reads=table_reads)
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

    def table_scale(base: str, field: str) -> float:
        name = base + "ngram_embedding." + field
        if not rd.has(prefix + name):
            return 1.0
        value = raw(name)
        if value.numel() != 1:
            raise ValueError(f"{name}: expected one n-gram table scale")
        return float(value.float().reshape(-1)[0])

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
        """A linear's weight and its scales: e8m0 (MXFP8), ("block", fp32 per row and 64 inputs) or None (bf16)."""

        w = raw(name + ".weight")
        if w.dtype == torch.float8_e4m3fn and rd.has(prefix + name + ".weight_scale_inv"):   # FP8_PB_WO blocks
            if rows is not None or cols is not None:
                raise ValueError(f"{name}: block-scaled FP8 is read on one GPU only (--tp 1)")
            from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear

            return w, ("block", Fp8BlockLinear.column_scales(raw(name + ".weight_scale_inv"), *w.shape))
        s = raw(name + ".weight_scale") if w.dtype == torch.float8_e4m3fn else None
        if s is not None and s.dtype != torch.uint8:
            raise ValueError(f"{name}: FP8 with a per-tensor scale; Flash Next reads MXFP8 (a scale every 32 inputs)")
        if s is None:
            _plain(name, w)
        if rows is not None:
            w, s = w[rows], None if s is None else s[rows]
        if cols is not None:
            w, s = w[:, cols], None if s is None else s[:, cols.start // 32:cols.stop // 32]
        return w, s

    def face(*parts):
        """Linears of one input as one face by their storage: bf16 rows on ``bf16.matmul``, MXFP8 on the lane matmul."""

        got = [dense(*p) for p in parts]
        if any(isinstance(s, tuple) for _, s in got):     # block FP8: its own lane-matmul face; bf16 parts beside it
            from tensorfold.cuda.nvfp4.linear import Concat, Fp8BlockLinear

            runs: list[list] = []
            for w, s in got:
                kind = "block" if isinstance(s, tuple) else "bf16" if s is None else "mx"
                if kind == "mx":
                    raise ValueError(f"{parts[0][0]}: a projection stack mixes MXFP8 and block FP8 weights")
                if runs and runs[-1][0] == kind:
                    runs[-1][1].append((w, s))
                else:
                    runs.append([kind, [(w, s)]])
            faces = [Fp8BlockLinear.from_rows(torch.cat([w for w, _ in ws]), torch.cat([s[1] for _, s in ws]))
                     if kind == "block" else b16_rows(torch.cat([w for w, _ in ws]).to(torch.bfloat16))
                     for kind, ws in runs]
            return faces[0] if len(faces) == 1 else Concat(faces)
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
        keys = shard_keys(prefix + base + "ngram_embedding", cfg.ngram_shards, rd.where)
        table = open_table(model_dir, [(rd.where[k + ".weight"], k) for k in keys],
                           lambda n: table_scale(base, n))
        if getattr(table, "width", ngram.dims) != ngram.dims:
            raise ValueError(f"the n-gram rows hold {table.width} values, expected {ngram.dims}")
        if table.rows != ngram.rows:
            raise ValueError(f"n-gram tables hold {table.rows} rows, expected {ngram.rows}")
        conv = raw(name + ".conv1d.weight").reshape(cfg.streams * cfg.hidden, cfg.ple_kernel).to(torch.bfloat16)
        return PLEW(table, b16(name + ".key_proj"), b16(name + ".value_proj"),
                    cscale(name + ".norm_key.weight"), cscale(name + ".norm_query.weight"),
                    cscale(name + ".norm_conv.weight"), conv.contiguous(), ngram)

    def weight_bf16(name: str, index: torch.Tensor | None = None) -> torch.Tensor:
        """A linear's weight as bf16 rows: block FP8 (``weight_scale_inv``) dequantized in row chunks, else cast."""

        full = raw(name + ".weight")
        w = full if index is None else full.index_select(0, index)
        if w.dtype != torch.float8_e4m3fn or not rd.has(prefix + name + ".weight_scale_inv"):
            return _plain(name, w).to(torch.bfloat16)
        from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear

        cols = Fp8BlockLinear.column_scales(raw(name + ".weight_scale_inv"), *full.shape)
        if index is not None:
            cols = cols.index_select(0, index)
        out = torch.empty(w.shape, dtype=torch.bfloat16, device=w.device)
        for r in range(0, w.shape[0], 16384):
            blk = w[r:r + 16384].float().view(-1, w.shape[1] // 64, 64) * cols[r:r + 16384, :, None]
            out[r:r + 16384] = blk.view(-1, w.shape[1]).to(torch.bfloat16)
        return out

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
        w_ = full.moe_width
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
        keys = shard_keys(prefix + base + "ngram_embedding", cfg.ngram_shards, rd.where)
        table = open_table(model_dir, [(rd.where[k + ".weight"], k) for k in keys],
                           lambda n: table_scale(base, n), ssd=ple_on_ssd)
        if table.rows != ngram.rows:
            raise ValueError(f"n-gram tables hold {table.rows} rows, expected {ngram.rows}")
        if table_reads is not None and not ple_on_ssd:     # its pages come in while the weights load
            from tensorfold.cuda.direct_read import in_background

            in_background(table.prefetch, table_reads)      # the caller waits for it (``wait_all``)
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
    try:                                              # a failed load cancels the reads queued ahead
        embed = ((raw(mbase + "embed_tokens.weight").to(torch.bfloat16).contiguous(),) if cfg.quant == "modelopt"
                 else triple("model.embed_tokens"))
        loaded = []
        ahead = rd.layer_names(prefix, mbase, chosen, mtp)   # read ahead of the layer that takes them
        layer_events: list = []                           # each layer's event, recorded once its work is queued
        for k, i in enumerate(chosen):
            if len(layer_events) >= 2:                    # at most two layers queued ahead of the GPU
                layer_events.pop(0).synchronize()
            for names in ahead[k:k + 2]:                  # two layers in flight: reads overlap this one's packing
                rd.queue(names)
            loaded.append(layer(i, f"{mbase}layers.{i}", cfg.layer_types[i], True))
            layer_events.append(torch.cuda.current_stream().record_event())
            rd.drop(ahead[k])                             # what the layer never took
            rd.release()
            if i % 8 == 7:                        # each release waits for the device; a layer leaves few temporaries
                torch.cuda.empty_cache()
        mixer = (hc_nvfp4 if cfg.quant == "modelopt" else hc)(mbase + "hyper_connection_mixer", False)
        vl = full.vocab // world
        if cfg.quant == "modelopt" and rd.has(prefix + "lm_head.weight_scale_inv"):     # block FP8: its stored bytes
            from dataclasses import replace

            from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear

            head = replace(Fp8BlockLinear.from_checkpoint(raw("lm_head.weight"), raw("lm_head.weight_scale_inv")),
                           lane=True)
        elif cfg.quant == "modelopt":
            head = b16_rows(weight_bf16("lm_head")[rank * vl:(rank + 1) * vl])
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
                draft_head = quantize4(weight_bf16("lm_head", ids))
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
    except BaseException:
        rd.close()
        raise
    rd.close()
    rd.release()
    torch.cuda.empty_cache()
    w.meta["load_seconds"] = time.time() - t0
    return w
