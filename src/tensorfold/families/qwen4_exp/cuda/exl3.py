"""Flash Next from an EXL3 pack into the MLX path's ``Weights``: trellis matrices, fp16 tensors, routed experts at their own widths."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.cuda.exl3 import format as fmt

from .exl3_mm import Scratch, f16, stack, x3
from .exl3_pack import _DT, NgramTable, Pack, is_exl3

PREFILL_ROWS = 2048       # the prompt buffers' rows (``decode.PREFILL_ROWS``): the n-gram staging holds as many

__all__ = ["is_exl3", "load"]


def expert_table(pk: Pack, prefix: str, count: int, shared: str, device) -> x3experts.Exl3RoutedExperts:
    """Experts ``prefix.{0..count-1}`` and the shared expert (as expert ``count``), trellises read in large runs into one buffer."""

    names = [f"{prefix}.{e}" for e in range(count)] + [shared]
    projs = ("gate_proj", "up_proj", "down_proj")
    codebooks = {pk.codebook(f"{nm}.{p}") for nm in names for p in projs}
    if len(codebooks) > 1:
        raise ValueError(f"{prefix}: the experts mix EXL3 codebooks ({', '.join(sorted(codebooks))}); the grouped "
                         "expert kernel takes one codebook a layer")
    parts = {nm + "." + p: (("suh" if pk.has(f"{nm}.{p}.suh") else "su"), ("svh" if pk.has(f"{nm}.{p}.svh") else "sv"))
             for nm in names for p in projs}
    entries = {f"{m}.{part}": pk.entry(f"{m}.{part}") for m, (i, o) in parts.items() for part in ("trellis", i, o)}
    place, total = {}, 0
    for k in entries:
        if k.endswith(".trellis"):
            place[k] = total
            total += -(-(entries[k][2] - entries[k][1]) // 256) * 256
    big = torch.empty((total,), dtype=torch.uint8, device=device)
    small: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for key, (file, *_rest) in entries.items():
        by_file.setdefault(file, []).append(key)
    for file, keys in by_file.items():
        keys.sort(key=lambda k: entries[k][1])
        run: list[str] = []

        def flush(run: list[str]) -> None:
            if run:
                b0, b1 = entries[run[0]][1], max(entries[k][2] for k in run)
                host = pk.read(file, b0, b1)
                dev = host.to(device)
                for k in run:
                    _, b, e, dtype, shape = entries[k]
                    if k.endswith(".trellis"):
                        big[place[k]:place[k] + (e - b)].copy_(dev[b - b0:e - b0])
                    else:
                        small[k] = host[b - b0:e - b0].clone().view(_DT[dtype]).reshape(shape)
                del dev, host

        for k in keys:                     # a run spans at most ~2 GB and skips at most 16 MB of other tensors
            if run and (entries[k][2] - entries[run[0]][1] > (2 << 30)
                        or entries[k][1] - max(entries[j][2] for j in run[-4:]) > (16 << 20)):
                flush(run)
                run = []
            run.append(k)
        flush(run)

    def trellis(k: str) -> torch.Tensor:
        _, b, e, dtype, shape = entries[k]
        if dtype != "I16":
            raise ValueError(f"{k}: trellis dtype {dtype}")
        return big[place[k]:place[k] + (e - b)].view(torch.int16).view(shape)

    def scales(m: str, part: str) -> torch.Tensor:
        t = small[f"{m}.{part}"]
        return (t if part in ("suh", "svh") else torch.from_numpy(fmt.unpack_signs(t.numpy()))).to(device)

    lists = {p: [(trellis(f"{nm}.{p}.trellis"), scales(f"{nm}.{p}", parts[f"{nm}.{p}"][0]),
                  scales(f"{nm}.{p}", parts[f"{nm}.{p}"][1])) for nm in names] for p in projs}
    ex = x3experts.prepare(lists["gate_proj"], lists["up_proj"], lists["down_proj"], codebooks.pop(), device=device)
    ex.keep.append(big)
    return ex


def centred_offset(pk: Pack, names: list[str]) -> float:
    """1.0 when the pack stores the centred norms as gamma - 1 (every EXL3 pack seen), 0.0 when as gamma."""

    means = np.array([float(pk.get(n).float().mean()) for n in names if pk.has(n)])
    if not len(means):
        return 1.0
    around_zero = (means > 0.5).mean() <= 0.1 and -0.5 <= float(np.median(means)) <= 0.25
    around_one = (means > 0.5).mean() >= 0.9 and 0.75 <= float(np.median(means)) <= 1.5
    if around_zero == around_one:
        raise ValueError(f"cannot tell how the pack stores its norm weights (median mean {np.median(means):.3f})")
    return 1.0 if around_zero else 0.0


def requant_rows(head, ids: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The head's rows ``ids`` decoded through its own EXL3 linear, as MLX 4-bit groups of 32: the draft head (drafts only)."""

    from .exl3_mm import ROWS

    k = head.k
    rows = torch.empty((len(ids), k), dtype=torch.float32, device=device)
    eye = torch.eye(ROWS, dtype=torch.bfloat16, device=device)
    out = torch.empty((ROWS, head.n), dtype=torch.float32, device=device)
    for k0 in range(0, k, ROWS):
        x = torch.zeros((ROWS, k), dtype=torch.bfloat16, device=device)
        x[:, k0:k0 + ROWS] = eye
        head(x, out)
        rows[:, k0:k0 + ROWS] = out[:, ids].t()
    g = rows.view(len(ids), k // 32, 32)
    lo, hi = g.amin(dim=-1), g.amax(dim=-1)
    scale = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
    bias = lo.to(torch.bfloat16)
    q = torch.round((g - bias.float()[..., None]) / scale.float()[..., None]).clamp(0, 15).to(torch.int64)
    words = (q.view(len(ids), k // 8, 8) << (torch.arange(8, device=device, dtype=torch.int64) * 4)).sum(dim=-1)
    words = words & 0xFFFFFFFF
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words.contiguous(), scale.contiguous(), bias.contiguous()


def load(model_dir: str | Path, device: str = "cuda", *, mtp: bool = True, tp: tuple[int, int] | None = None,
         draft_vocab: int | str | None = None):
    from .qmm import make_q4
    from .weights import GDNW, HC, AttnW, Config, LayerW, MoEW, MTPW, PLEW, Weights, draft_token_ids

    if tp is not None and tp[1] > 1:
        raise ValueError("EXL3 packs of Flash Next run on one GPU; two ranks read the MLX checkpoint")
    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    pk = Pack(model_dir)
    sc = Scratch(cfg.top_k + 1)
    T = "model.language_model."
    t0 = time.time()
    offset = centred_offset(pk, [f"{T}layers.{i}.attn_hyper_connection.hc_norm.weight" for i in range(cfg.layers)])

    def plain(name: str) -> torch.Tensor:
        return pk.get(name).to(device)

    def centred(name: str) -> torch.Tensor:
        return (pk.get(name).float() + offset).to(device).contiguous()

    def hc(name: str, inject: bool) -> HC:
        rows = [pk.get(name + ".input_mix_weight_down.weight")]
        if inject:
            rows.append(pk.get(name + ".block_inject_weight.weight"))
        down, up = f16(sc, rows, device), f16(sc, [pk.get(name + ".input_mix_weight_up.weight")], device)
        return HC(down, up, centred(name + ".hc_norm.weight"), inject, down, up)

    def moe(name: str) -> MoEW:
        router = torch.cat([pk.get(name + ".gate.weight").to(torch.bfloat16),
                            pk.get(name + ".shared_expert_gate.weight").to(torch.bfloat16)]).to(device).contiguous()
        return MoEW(router, expert_table(pk, name + ".experts", cfg.experts, name + ".shared_expert", device))

    def attention(name: str) -> AttnW:
        proj = stack(sc, [x3(sc, pk, name + p, device) for p in (".q_proj", ".k_proj", ".v_proj",
                                                                  ".indexer.index_qk_proj")])
        return AttnW(proj, centred(name + ".q_norm.weight"), centred(name + ".k_norm.weight"),
                     centred(name + ".indexer.q_layernorm.weight"), centred(name + ".indexer.k_layernorm.weight"),
                     x3(sc, pk, name + ".o_proj", device))

    def gdn(name: str) -> GDNW:
        proj = stack(sc, [x3(sc, pk, name + ".in_proj_qkv", device), x3(sc, pk, name + ".in_proj_z", device),
                          f16(sc, [pk.get(name + ".in_proj_b.weight"), pk.get(name + ".in_proj_a.weight")], device)])
        conv = plain(name + ".conv1d.weight").reshape(cfg.conv_dim, cfg.conv_kernel).to(torch.bfloat16).contiguous()
        return GDNW(proj, conv, plain(name + ".A_log").float().contiguous(),
                    plain(name + ".dt_bias").float().contiguous(),
                    plain(name + ".norm.weight").to(torch.bfloat16).contiguous(),
                    x3(sc, pk, name + ".out_proj", device))

    def ple_layer(name: str, ple_index: int) -> PLEW:
        ngram = cfg.ngram(ple_index)
        table = NgramTable(pk, name + ".ple_embedding.ngram_embedding.", cfg.ngram_shards, device)
        ngram.check(table.multipliers, table.head_offsets, table.head_sizes)
        if table.rows != ngram.rows or table.dh != ngram.dims:
            raise ValueError(f"n-gram tables of {table.rows} rows of {table.dh} values; the config gives "
                             f"{ngram.rows} of {ngram.dims}")
        conv = plain(name + ".conv1d.weight").reshape(cfg.streams * cfg.hidden, cfg.ple_kernel).to(torch.bfloat16)
        return PLEW(table, f16(sc, [pk.get(name + ".key_proj.weight")], device),
                    f16(sc, [pk.get(name + ".value_proj.weight")], device), centred(name + ".norm_key.weight"),
                    centred(name + ".norm_query.weight"), centred(name + ".norm_conv.weight"), conv.contiguous(), ngram)

    def layer(i: int, base: str, kind: str, with_ple: bool) -> LayerW:
        linear = kind == "linear"
        entry = LayerW(i, linear, hc(base + ".attn_hyper_connection", True), hc(base + ".mlp_hyper_connection", True),
                       gdn(base + ".linear_attn") if linear else None,
                       None if linear else attention(base + ".self_attn"), moe(base + ".mlp"))
        if with_ple and i in cfg.ple_layers:
            entry.ple = ple_layer(base + ".ple", cfg.ple_layers.index(i))
        return entry

    embed = plain(T + "embed_tokens.weight")
    if embed.dtype not in (torch.bfloat16, torch.float16):
        embed = embed.to(torch.bfloat16)
    loaded = []
    for i in range(cfg.layers):
        loaded.append(layer(i, f"{T}layers.{i}", cfg.layer_types[i], True))
        pk.release()
        torch.cuda.empty_cache()
    mixer = hc(T + "hyper_connection_mixer", False)
    head = x3(sc, pk, "lm_head", device, head=True)
    inv = torch.tensor(cfg.rope_theta, dtype=torch.float64) ** (
        -torch.arange(0, cfg.rotary_dim // 2, dtype=torch.float64) / (cfg.rotary_dim // 2))
    w = Weights(cfg, (embed.contiguous(),), loaded, mixer, head, inv.to(torch.float32).to(device), around_one=True)
    w.meta.update(rank=0, world=1, vocab_offset=0, full=cfg, centred_offset=offset)
    if mtp and pk.has("mtp.fc_embedding.trellis"):
        w.mtp = MTPW(centred("mtp.pre_fc_norm_embedding.weight"), centred("mtp.pre_fc_norm_hidden.weight"),
                     x3(sc, pk, "mtp.fc_embedding", device), x3(sc, pk, "mtp.fc_hidden", device),
                     layer(-1, "mtp.layers.0", "attention", False), hc("mtp.hyper_connection_mixer", False))
    ple = next((lay.ple for lay in loaded if lay.ple is not None), None)
    sc.allocate(device, experts=loaded[0].moe.experts, rows=PREFILL_ROWS,
                ple_words=ple.table.words_per_row if ple else 0, ple_heads=ple.ngram.heads if ple else 0,
                ple_dim=cfg.ple_dim)
    w.x3 = sc
    ids = draft_token_ids(draft_vocab)
    if ids is not None and w.mtp is not None:
        ids = ids[ids < cfg.vocab]
        w.draft_ids = torch.from_numpy(ids).to(device)
        w.draft_head = make_q4(*requant_rows(head, w.draft_ids, device))
    pk.release()
    torch.cuda.empty_cache()
    w.meta["load_seconds"] = time.time() - t0
    return w
