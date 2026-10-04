"""One rank's weights from the checkpoint: its part of each tensor read with O_DIRECT, every value unchanged."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import torch

from tensorfold.cuda.direct_read import DTYPES, ReadAhead, Reader, read_header
from tensorfold.cuda.exl3 import experts as exl3
from tensorfold.cuda.nvfp4.linear import Mx8Linear
from tensorfold.families.glm5_next.cuda.qmm import make_b16
from tensorfold.families.glm5_next.cuda.split import GAP, READERS, RUN

from ..config import Config
from . import MAX_ROWS, rope, split
from .convert import fp8_block_rows, make_experts4
from .weights import HCW, AttnW, CompW, DSparkW, EngramW, IdxW, LayerW, MoEW, StageW, Weights

SLOT = 256                                    # alignment of each expert trellis inside its layer's buffer
_DTYPES = {**DTYPES, "F8_E8M0": torch.uint8}  # E8M0 stays raw bytes, never read through a float type
_GROUP = re.compile(r"^(layers|mtp)\.(\d+)\.")


def _unused(name: str) -> bool:
    """Engram tables (read by row id at run time), the vision tower, and EXL3 codebook markers (the config's)."""

    return name.endswith(".mcg") or split.rule(name) in ("engram", "drop")


class _Pack:
    """The checkpoint's tensors by name, each as the rank's part, read ahead on threads and uploaded to ``device``."""

    def __init__(self, model_dir: str | Path, rank: int, world: int, device: torch.device) -> None:
        self.dir, self.rank, self.world, self.device = Path(model_dir), rank, world, device
        self.where = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, tuple[Path, int, dict]] = {}
        self.reads = ReadAhead(Reader(), READERS, RUN, GAP)
        self.groups: dict[str, list[str]] = {}
        for name in self.where:
            m = _GROUP.match(name)
            self.groups.setdefault(m.group(0) if m else "", []).append(name)

    def names(self, prefix: str) -> list[str]:
        """The tensors of one layer or DSpark stage (``"layers.3."``) the engine loads."""

        return [n for n in self.groups.get(prefix, []) if not _unused(n)]

    def _file(self, file: str) -> tuple[Path, int, dict]:
        if file not in self.files:
            path = (self.dir / file).resolve()          # the pack's shards are symlinks into sibling revisions
            self.files[file] = (path, *read_header(path))
        return self.files[file]

    def item(self, name: str) -> tuple:
        """(name, file, first byte, end byte, (dtype, shape read, cut or None, part shape)) of the rank's bytes."""

        path, base, header = self._file(self.where[name])
        info = header[name]
        if info["dtype"] not in _DTYPES:
            raise ValueError(f"{name}: safetensors dtype {info['dtype']} is not supported")
        shape, (a, b) = list(info["shape"]), info["data_offsets"]
        cut = split.slice_for(self.rank, self.world, name, shape)
        part = [s.stop - s.start for s in cut]
        if all(p == n for p, n in zip(part[1:], shape[1:])):      # whole rows: read only the rank's run
            if shape:
                per = (b - a) // shape[0]
                a, b = a + cut[0].start * per, a + cut[0].stop * per
            return name, path, base + a, base + b, (info["dtype"], part, None, part)
        return name, path, base + a, base + b, (info["dtype"], shape, cut, part)

    def _cut(self, raw: torch.Tensor, meta: tuple) -> torch.Tensor:
        """The rank's tensor from its bytes in a shared read: a new contiguous copy, sliced first when ``cut``."""

        dtype, shape, cut, part = meta
        raw = raw.view(*shape, _DTYPES[dtype].itemsize)
        if cut is not None:
            raw = raw[cut]
        return raw.clone(memory_format=torch.contiguous_format).view(_DTYPES[dtype]).reshape(part)

    def queue(self, names) -> None:
        self.reads.queue([self.item(n) for n in names], self.device, self._cut)

    def get(self, name: str) -> torch.Tensor:
        if name not in self.reads.ahead:
            self.queue([name])
        return self.reads.take(name)

    def close(self) -> None:
        self.reads.close()
        self.reads.reader.close()


class _Build:
    """The family's weight objects from the rank's tensors."""

    def __init__(self, pack: _Pack, cfg: Config) -> None:
        self.pack, self.cfg = pack, cfg

    def t(self, name: str) -> torch.Tensor:
        return self.pack.get(name)

    def mx8(self, *names: str) -> Mx8Linear:
        """FP8 32x32-block projections of one input as one ``Mx8Linear``, outputs stacked in order."""

        ws = [self.t(n + ".weight") for n in names]
        ss = [fp8_block_rows(self.t(n + ".scale"), int(w.shape[0])) for n, w in zip(names, ws)]
        return Mx8Linear.from_checkpoint(ws[0] if len(ws) == 1 else torch.cat(ws),
                                         ss[0] if len(ss) == 1 else torch.cat(ss))

    def groups(self, name: str) -> list[Mx8Linear]:
        """``wo_a``: one ``Mx8Linear`` for each of the rank's output groups."""

        w = self.t(name + ".weight")
        s = fp8_block_rows(self.t(name + ".scale"), int(w.shape[0]))
        n = self.cfg.o_lora_rank
        return [Mx8Linear.from_checkpoint(w[g:g + n], s[g:g + n]) for g in range(0, w.shape[0], n)]

    def hc(self, p: str, site: str) -> HCW:
        return HCW(*(self.t(f"{p}hc_{site}_{x}") for x in ("fn", "base", "scale")))

    def attn(self, p: str, layer: int | None) -> AttnW:
        """A block's attention; ``layer`` None for a DSpark stage (no compressor, no indexer)."""

        a, c = p + "attn.", self.cfg
        comp = idx = None
        if layer in c.kv_source_layer_ids:
            kv = [self.t(a + "compressor.wkv.weight")]
            if c.compress_ratios[layer] == 2:
                kv.append(self.t(a + "compressor.wgate.weight"))
            comp = CompW(c.compress_ratios[layer], make_b16(torch.cat(kv)), self.t(a + "compressor.norm.weight"))
        if layer in c.index_source_layer_ids or comp is not None:
            i = a + "indexer."
            scores = layer in c.index_source_layer_ids
            idx = IdxW(self.mx8(i + "wq_b") if scores else None,
                       make_b16(self.t(i + "weights_proj.weight")) if scores else None,
                       make_b16(self.t(i + "wk.weight")) if comp is not None else None,
                       self.t(i + "k_norm.weight") if comp is not None else None)
        return AttnW(self.mx8(a + "wq_a", a + "wkv"), self.t(a + "q_norm.weight"), self.t(a + "kv_norm.weight"),
                     self.mx8(a + "wq_b"), self.groups(a + "wo_a"), self.mx8(a + "wo_b"), self.t(a + "attn_sink"),
                     comp, idx)

    def exl3(self, f: str) -> exl3.Exl3RoutedExperts:
        """A layer's routed experts, every trellis in one buffer at ``SLOT``-aligned offsets."""

        n = self.cfg.n_routed_experts
        names = [f"{f}experts.{e}.{w}" for e in range(n) for w in ("w1", "w3", "w2")]
        at, slots = 0, {}
        for name in names:
            slots[name] = at
            at += -(-math.prod(self.pack.item(name + ".trellis")[4][3]) * 2 // SLOT) * SLOT
        buf = torch.empty((at,), dtype=torch.uint8, device=self.pack.device)
        mats: dict[str, list] = {"w1": [], "w3": [], "w2": []}
        for name in names:
            t = self.t(name + ".trellis")
            slot = buf[slots[name]:slots[name] + t.numel() * 2].view(torch.int16).view(t.shape)
            slot.copy_(t)
            del t
            mats[name.rsplit(".", 1)[1]].append((slot, self.t(name + ".suh"), self.t(name + ".svh")))
        return exl3.prepare(mats["w1"], mats["w3"], mats["w2"], self.cfg.codebook, device=self.pack.device)

    def experts4(self, f: str):
        """A DSpark stage's MXFP4 experts as ``Experts4`` (exact)."""

        n = self.cfg.dspark_n_routed_experts
        mats = [[(self.t(f"{f}experts.{e}.{w}.weight"), self.t(f"{f}experts.{e}.{w}.scale")) for e in range(n)]
                for w in ("w1", "w3", "w2")]
        return make_experts4(*mats, limit=self.cfg.swiglu_limit)

    def moe(self, p: str, experts) -> MoEW:
        f = p + "ffn."
        s = f + "shared_experts."
        return MoEW(self.t(f + "gate.weight"), self.t(f + "gate.bias"), self.t(f + "gate.bias_vl"), experts,
                    self.mx8(s + "w1", s + "w3"), self.mx8(s + "w2"))

    def block(self, cls, index: int, p: str, layer: int | None, experts) -> LayerW:
        return cls(index, self.cfg.roles[index], self.hc(p, "attn"), self.hc(p, "ffn"), self.t(p + "attn_norm.weight"),
                   self.t(p + "ffn_norm.weight"), self.attn(p, layer), self.moe(p, experts))

    def layer(self, i: int) -> tuple[LayerW, EngramW | None]:
        p = f"layers.{i}."
        engram = None
        if self.cfg.roles[i].engram:
            e = p + "engram."
            engram = EngramW(self.mx8(e + "wkv"), self.t(e + "q_weight").float() * self.t(e + "k_weight").float())
        return self.block(LayerW, i, p, i, self.exl3(p + "ffn.")), engram

    def stage(self, s: int) -> StageW:
        p = f"mtp.{s}."
        return self.block(StageW, self.cfg.num_hidden_layers + s, p, None, self.experts4(p + "ffn."))

    def dspark(self, stages: list[StageW]) -> DSparkW:
        last = f"mtp.{self.cfg.num_nextn_predict_layers - 1}."
        return DSparkW(self.mx8("mtp.0.main_proj"), self.t("mtp.0.main_norm.weight"), stages,
                       self.t(last + "norm.weight"), self.t(last + "markov_head.embed.weight"),
                       make_b16(self.t(last + "markov_head.head.weight")),
                       self.t(last + "confidence_head.proj.weight").float())


def load(model_dir: str | Path, cfg: Config, rank: int, world: int, comm, layers=None, dspark: bool = True, *,
         capacity: int | None = None, device: str | torch.device = "cuda") -> Weights:
    """Rank ``rank`` of ``world`` (1 for tests): backbone ``layers``, DSpark unless ``dspark`` is False, RoPE."""

    if world not in (1, 2) or not 0 <= rank < world:
        raise ValueError(f"rank {rank} of {world}: the engine runs on one or two GPUs")
    n = cfg.num_hidden_layers
    ids = sorted(set(range(n) if layers is None else (int(i) for i in layers)))
    if any(not 0 <= i < n for i in ids):
        raise ValueError(f"layers {ids}: the checkpoint has layers 0..{n - 1}")
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    start, before = time.perf_counter(), torch.cuda.memory_allocated(dev)
    pack = _Pack(model_dir, rank, world, dev)
    b = _Build(pack, cfg)
    stages = range(cfg.num_nextn_predict_layers) if dspark else range(0)
    plan = [(["embed.weight"], lambda: b.t("embed.weight"))]
    plan += [(pack.names(f"layers.{i}."), lambda i=i: b.layer(i)) for i in ids]
    plan += [(["norm.weight", "lm_head.weight", "lm_head.weight_scale"],
              lambda: (b.t("norm.weight"), Mx8Linear.from_checkpoint(b.t("lm_head.weight"),
                                                                     b.t("lm_head.weight_scale"))))]
    plan += [(pack.names(f"mtp.{s}."), lambda s=s: b.stage(s)) for s in stages]
    built = []
    try:
        pack.queue(plan[0][0])
        for k, (_, build) in enumerate(plan):
            if k + 1 < len(plan):
                pack.queue(plan[k + 1][0])              # the next group reads while this one is built
            built.append(build())
        drafter = b.dspark(built[-len(stages):]) if dspark else None
    finally:
        pack.close()
    embed, (norm, head) = built[0], built[1 + len(ids)]
    loaded = built[1:1 + len(ids)]
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated(dev) - before
    slots = cfg.max_position_embeddings + MAX_ROWS if capacity is None else int(capacity)
    tables = {kind: rope.tables(cfg, kind, slots, dev) for kind in rope.KINDS}
    print(f"[tensorfold] rank {rank} of {world}: {len(ids)} layers{' and DSpark' if dspark else ''}, "
          f"{resident / 2**30:.2f} GiB of weights in {time.perf_counter() - start:.1f} s", flush=True)
    return Weights(cfg, rank, world, comm, dev, rank * cfg.vocab_size // world, embed, [lw for lw, _ in loaded], norm,
                   head, drafter, {lw.index: e for lw, e in loaded if e is not None}, tables)
