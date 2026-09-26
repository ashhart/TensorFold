"""Flash Next weights on the GPU from the MLX 4-bit checkpoint (affine, groups of 32).

Projections are regrouped once for ``qmm`` (tiled words, group-major scales and biases); projections
that read the same input are stacked into one matrix (DeltaNet q/k/v, z, b, a; attention q|gate, k, v,
indexer q and key; a hyper-connection's down and inject rows). The 512 routed experts and the shared
expert are one table of 513 experts per layer (the shared expert is expert 512). The router stays bf16,
with the shared expert's gate row (dequantized to bf16) appended as row 512. The n-gram tables' 128 row
shards are concatenated in order. The vision tower is skipped.

The checkpoint stores the centred norms' gamma itself (around 1), not gamma - 1: ``norms_around_one``
checks that on the hyper-connection norms, and every centred-norm scale is kept as fp32.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ngram import NGram
from .qmm import Experts, Q4, dequantize, make_experts, make_q4, stack_q4


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

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        t = dict(raw.get("text_config") or raw)
        rope = dict(t.get("rope_parameters") or {})
        head_dim = int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"])
        partial = float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
        teos = t.get("eos_token_id")
        eos = raw.get("eos_token_id", teos)
        eos = tuple(int(e) for e in eos) if isinstance(eos, list) else (int(eos),)
        quant = raw.get("quantization") or raw.get("quantization_config") or {}
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


@dataclass
class GDNW:
    proj: Q4                  # [qkv | z | b | a] x D
    conv: torch.Tensor        # [conv_dim, taps] bf16
    a_log: torch.Tensor       # [nv] fp32
    dt_bias: torch.Tensor     # [nv] fp32
    norm: torch.Tensor        # [dv] bf16 (the gated RMSNorm's weight, used as stored)
    out: Q4


@dataclass
class AttnW:
    proj: Q4                  # [q|gate pairs | k | v | indexer q | indexer key] x D
    q_scale: torch.Tensor     # [head_dim] fp32
    k_scale: torch.Tensor
    iq_scale: torch.Tensor    # [index_dim] fp32
    ik_scale: torch.Tensor    # the pooled indexer keys' norm
    o: Q4


@dataclass
class MoEW:
    router: torch.Tensor      # [E + 1, D] bf16: router rows, then the shared expert's gate row
    experts: Experts          # E + 1 experts (the shared expert last)


class HostTable:
    """The n-gram embedding's 128 row shards, memory-mapped from the checkpoint (never copied to the GPU):
    a token reads 16 rows of 100 bytes from 32 GB of tables, so ``gather`` copies just those rows out."""

    def __init__(self, files: list[tuple[Path, dict, dict, dict]]) -> None:
        import struct

        self.words, self.scales, self.biases, starts = [], [], [], [0]
        maps: dict = {}
        fidx, wbase, sbase, bbase = [], [], [], []
        for path, hw, hs, hb in files:
            self.words.append(_memmap(path, hw, np.uint32))
            self.scales.append(_memmap(path, hs, np.uint16))
            self.biases.append(_memmap(path, hb, np.uint16))
            starts.append(starts[-1] + self.words[-1].shape[0])
            if path not in maps:
                with open(path, "rb") as f:
                    data = 8 + struct.unpack("<Q", f.read(8))[0]
                maps[path] = (len(maps), np.memmap(path, dtype=np.uint8, mode="r"), data)
            index, _, data = maps[path]
            fidx.append(index)
            wbase.append(data + hw["data_offsets"][0])
            sbase.append(data + hs["data_offsets"][0])
            bbase.append(data + hb["data_offsets"][0])
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        # byte views of the files, so a gather is one fancy index per file and component, not per shard
        self.files = [m for _, m, _ in sorted(maps.values(), key=lambda t: t[0])]
        self.fidx = np.array(fidx, dtype=np.int64)
        self.wbase, self.sbase, self.bbase = (np.array(x, dtype=np.int64) for x in (wbase, sbase, bbase))
        self.wrow = self.words[0].shape[1] * 4
        self.grow = self.scales[0].shape[1] * 2

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows ``ids`` (global) -> words [n, W] uint32, scales and biases [n, G] (bf16 bits as uint16)."""

        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        n = len(flat)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        local = flat - self.starts[shard]
        where = self.fidx[shard]
        wo = self.wbase[shard] + local * self.wrow
        so = self.sbase[shard] + local * self.grow
        bo = self.bbase[shard] + local * self.grow
        w = np.empty((n, self.wrow), dtype=np.uint8)
        sc = np.empty((n, self.grow), dtype=np.uint8)
        bi = np.empty((n, self.grow), dtype=np.uint8)
        aw, ag = np.arange(self.wrow), np.arange(self.grow)
        for f in np.unique(where):
            at = np.nonzero(where == f)[0]
            mm = self.files[f]
            w[at] = mm[wo[at, None] + aw]
            sc[at] = mm[so[at, None] + ag]
            bi[at] = mm[bo[at, None] + ag]
        return w.view(np.uint32), sc.view(np.uint16), bi.view(np.uint16)

    def prefetch(self, workers: int = 8) -> float:
        """Read every shard once so the lookups hit the page cache (seconds taken); the pages stay evictable."""

        import time
        from concurrent.futures import ThreadPoolExecutor

        def touch(arr) -> None:
            flat = arr.reshape(-1).view(np.uint8)
            step = 64 << 20
            for i in range(0, flat.size, step):
                np.asarray(flat[i:i + step]).sum(dtype=np.uint64)

        t0 = time.time()
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(touch, self.words + self.scales + self.biases))
        return time.time() - t0


def _memmap(path: Path, entry: dict, dtype) -> np.ndarray:
    import struct

    with open(path, "rb") as f:
        header = struct.unpack("<Q", f.read(8))[0]
    begin, end = entry["data_offsets"]
    shape = tuple(entry["shape"])
    return np.memmap(path, dtype=dtype, mode="r", offset=8 + header + begin, shape=shape)


def _header(path: Path) -> dict:
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


@dataclass
class PLEW:
    table: HostTable          # the 128 shards (host memory map)
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
    embed: tuple[torch.Tensor, torch.Tensor, torch.Tensor]     # MLX layout (row lookup)
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

    @property
    def device(self) -> torch.device:
        return self.inv_freq.device

    def nbytes(self) -> int:
        total = 0

        def add(x: Any) -> None:
            nonlocal total
            if isinstance(x, torch.Tensor):
                total += x.numel() * x.element_size()
            elif isinstance(x, (Q4,)):
                total += x.nbytes()
            elif hasattr(x, "__dataclass_fields__"):
                for f in x.__dataclass_fields__:
                    add(getattr(x, f))
            elif isinstance(x, (list, tuple)):
                for y in x:
                    add(y)

        add(self.embed)
        add(self.layers)
        add(self.mixer)
        add(self.head)
        add(self.mtp)
        return total


_DT = {"U32": torch.int32, "I32": torch.int32, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
       "I64": torch.int64, "U8": torch.uint8, "I8": torch.int8, "U16": torch.int16, "I16": torch.int16}


class _Reader:
    """Tensors by name from the checkpoint's shards, read with large sequential reads (not mmap page faults:
    those streamed ~0.2 GB/s from NVMe once the page cache was dropped); drops each shard's cached pages."""

    def __init__(self, model_dir: Path, device: str) -> None:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.where = index["weight_map"]
        self.dir = model_dir
        self.device = device
        self.headers: dict[str, tuple[int, dict]] = {}
        self.touched: set[str] = set()

    def _header(self, shard: str) -> tuple[int, dict]:
        got = self.headers.get(shard)
        if got is None:
            import struct

            with open(self.dir / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                got = (8 + n, json.loads(f.read(n)))
            self.headers[shard] = got
        return got

    def get(self, name: str) -> torch.Tensor:
        shard = self.where[name]
        base, header = self._header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        raw = torch.empty((end - begin,), dtype=torch.uint8)
        view = memoryview(raw.numpy())
        with open(self.dir / shard, "rb", buffering=0) as f:
            f.seek(base + begin)
            at = 0
            while at < len(view):
                got = f.readinto(view[at:at + (64 << 20)])
                if not got:
                    raise IOError(f"short read of {name}")
                at += got
        self.touched.add(shard)
        dtype = _DT[entry["dtype"]]
        return raw.view(dtype).reshape(entry["shape"]).to(self.device)

    def has(self, name: str) -> bool:
        return name in self.where

    def release(self) -> None:
        """Drop the read shards' cached pages: a full load would otherwise hold ~110 GB of page cache beside the
        same bytes on the GPU (unified memory)."""

        for shard in list(self.touched):
            try:
                fd = os.open(self.dir / shard, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except (OSError, AttributeError):
                pass
        self.touched.clear()


def norms_around_one(reader: _Reader, prefix: str, layers: list[int]) -> bool:
    means = []
    for i in layers:
        name = f"{prefix}model.layers.{i}.attn_hyper_connection.hc_norm.weight"
        if reader.has(name):
            means.append(float(reader.get(name).float().mean()))
    if not means:
        return True
    means = np.array(means)
    around_one = (means > 0.5).mean() >= 0.9 and 0.75 <= float(np.median(means)) <= 1.5
    around_zero = (means > 0.5).mean() <= 0.1 and -0.5 <= float(np.median(means)) <= 0.25
    if not (around_one or around_zero):
        raise ValueError(f"cannot tell how the norm weights are stored (median mean {np.median(means):.3f})")
    return around_one


def _rows(t3, lo: int, hi: int):
    """Output rows [lo, hi) of (words, scales, biases), stacked experts included."""

    return tuple(x[..., lo:hi, :].contiguous() for x in t3)


def _rows_at(t3, idx: torch.Tensor):
    return tuple(x.index_select(x.dim() - 2, idx).contiguous() for x in t3)


def _groups(t3, g0: int, g1: int):
    """Input groups [g0, g1) (32 inputs each) of (words, scales, biases): no repacking."""

    w, sc, b = t3
    return w[..., g0 * 4:g1 * 4].contiguous(), sc[..., g0:g1].contiguous(), b[..., g0:g1].contiguous()


def draft_token_ids(draft_vocab: int | str | None) -> np.ndarray | None:
    """The token ids the MTP drafts' head scores, sorted: "default" (``draft_vocab.txt`` beside this module), a
    file of ids, an int N (ids below N), or None (the full vocabulary)."""

    if not draft_vocab:
        return None
    if isinstance(draft_vocab, int):
        return np.arange(draft_vocab, dtype=np.int64)
    source = Path(__file__).with_name("draft_vocab.txt") if draft_vocab == "default" else Path(draft_vocab)
    return np.unique(np.loadtxt(source, dtype=np.int64).reshape(-1))


def load(model_dir: str | Path, device: str = "cuda", *, mtp: bool = True, tp: tuple[int, int] | None = None,
         draft_vocab: int | str | None = None) -> Weights:
    """``tp`` = (rank, world): this rank's share for tensor parallelism (heads, expert width, vocabulary rows
    split; hyper-connections, router, embeddings and the MTP input layers replicated).

    ``draft_vocab``: the MTP drafts' head scores only these token ids: "default" (``draft_vocab.txt`` beside this
    module), a file of ids, or an int N (ids below N). Drafts only change speed; None scores the full vocabulary."""

    import time
    from dataclasses import replace

    model_dir = Path(model_dir)
    full = Config.read(model_dir)
    rank, world = tp if tp is not None else (0, 1)
    cfg = full if world == 1 else replace(full, heads=full.heads // world, kv_heads=full.kv_heads // world,
                                          nk=full.nk // world, nv=full.nv // world,
                                          moe_width=full.moe_width // world, shared_width=full.shared_width // world)
    rd = _Reader(model_dir, device)
    prefix = "language_model." if rd.has("language_model.model.embed_tokens.weight") else ""
    chosen = list(range(cfg.layers))
    around_one = norms_around_one(rd, prefix, chosen)

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
        return HC(stack_q4(parts), q4(name + ".input_mix_weight_up"), cscale(name + ".hc_norm.weight"), inject)

    def moe(name: str) -> MoEW:
        gate_rows = raw(name + ".gate.weight").to(torch.bfloat16)
        sw, ss, sb = triple(name + ".shared_expert_gate")
        shared_gate = dequantize(sw, ss, sb).to(torch.bfloat16)
        router = torch.cat([gate_rows, shared_gate]).contiguous()
        w_, sw_ = full.moe_width, full.shared_width
        # a rank takes its half of every expert's intermediate width: gate/up rows, down input groups
        gu = lambda t, width: _rows(t, rank * width // world, (rank + 1) * width // world)          # noqa: E731
        dn = lambda t, width: _groups(t, rank * width // world // 32, (rank + 1) * width // world // 32)  # noqa: E731
        experts = make_experts(gu(triple(name + ".switch_mlp.gate_proj"), w_), gu(triple(name + ".switch_mlp.up_proj"), w_),
                               dn(triple(name + ".switch_mlp.down_proj"), w_),
                               (gu(triple(name + ".shared_expert.gate_proj"), sw_),
                                gu(triple(name + ".shared_expert.up_proj"), sw_),
                                dn(triple(name + ".shared_expert.down_proj"), sw_)))
        return MoEW(router, experts)

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
        table = HostTable(files)
        if table.rows != ngram.rows:
            raise ValueError(f"n-gram tables hold {table.rows} rows, expected {ngram.rows}")
        conv = raw(name + ".conv1d.weight").reshape(cfg.streams * cfg.hidden, cfg.ple_kernel).to(torch.bfloat16)
        return PLEW(table, q4(name + ".key_proj"), q4(name + ".value_proj"),
                    cscale(name + ".norm_key.weight"), cscale(name + ".norm_query.weight"),
                    cscale(name + ".norm_conv.weight"), conv.contiguous(), ngram)

    def layer(i: int, base: str, kind: str, with_ple: bool) -> LayerW:
        linear = kind == "linear"
        entry = LayerW(i, linear, hc(base + ".attn_hyper_connection", True), hc(base + ".mlp_hyper_connection", True),
                       gdn(base + ".linear_attn") if linear else None,
                       None if linear else attention(base + ".self_attn"), moe(base + ".mlp"))
        if with_ple and i in cfg.ple_layers:
            entry.ple = ple_layer(base + ".ple", cfg.ple_layers.index(i))
        return entry

    t0 = time.time()
    embed = triple("model.embed_tokens")
    loaded = []
    for i in chosen:
        loaded.append(layer(i, f"model.layers.{i}", cfg.layer_types[i], True))
        rd.release()
        torch.cuda.empty_cache()
    mixer = hc("model.hyper_connection_mixer", False)
    vl = full.vocab // world
    head_raw = triple("lm_head")
    head = make_q4(*_rows(head_raw, rank * vl, (rank + 1) * vl))
    draft_head, draft_ids = None, None
    ids = draft_token_ids(draft_vocab)
    if ids is not None:
        ids = np.array_split(ids[ids < full.vocab], world)[rank]
        draft_ids = torch.from_numpy(ids).to(device)
        draft_head = make_q4(*_rows_at(head_raw, draft_ids))
    del head_raw
    inv = torch.tensor(cfg.rope_theta, dtype=torch.float64) ** (
        -torch.arange(0, cfg.rotary_dim // 2, dtype=torch.float64) / (cfg.rotary_dim // 2))
    w = Weights(cfg, embed, loaded, mixer, head, inv.to(torch.float32).to(device), around_one=around_one)
    w.meta.update(rank=rank, world=world, vocab_offset=rank * vl, full=full)
    w.draft_head, w.draft_ids = draft_head, draft_ids
    if mtp and rd.has(prefix + "mtp.fc_embedding.weight"):
        w.mtp = MTPW(cscale("mtp.pre_fc_norm_embedding.weight"), cscale("mtp.pre_fc_norm_hidden.weight"),
                     q4("mtp.fc_embedding"), q4("mtp.fc_hidden"),
                     layer(-1, "mtp.layers.0", "attention", False), hc("mtp.hyper_connection_mixer", False))
    rd.release()
    torch.cuda.empty_cache()
    w.meta["load_seconds"] = time.time() - t0
    return w
