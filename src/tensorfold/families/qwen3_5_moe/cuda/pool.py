"""Qwen3.6's routed experts served from the checkpoint's files: the pool, its slots, and what to read.

``sources`` and ``Slots`` are pure and GPU-free; ``ExpertPool`` fills a pool of packed slots on the device.
A slot is one expert's packed block (gate, up and down); a layer's ids are remapped to slots before the grouped
kernels run, so a (row, slot) pair keeps the arithmetic it has with the stacks resident.
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from functools import lru_cache

from tensorfold.cuda.capacity import itemsize

ROUTED = ".mlp.switch_mlp."          # the routed stacks; the router (mlp.gate) and the shared expert stay resident
PARTS = ("weight", "scales", "biases")
PROJS = ("gate_proj", "up_proj", "down_proj")
DTYPES = {"U32": ("uint32", "int32"), "BF16": ("uint16", "bfloat16")}
# eight 1.6875 MiB reads in flight measured 10.11 GiB/s on the box, against 2.20 GiB/s with one
READERS = 8
REPORT = 256              # experts read between the pool's own log lines: the receipt's hit rate, mid-run
MAX_SLOTS = 1024          # the grouped kernels take at most this many experts in a stack ("at most 1024 experts")


def shards(model_dir: Path) -> dict[Path, tuple[int, dict]]:
    """Each shard's (first data byte, entries) for the checkpoint's model*.safetensors files."""

    out: dict[Path, tuple[int, dict]] = {}
    for path in sorted(Path(model_dir).glob("model*.safetensors")):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size <= 64 * 1024**2:
                raise ValueError(f"invalid checkpoint tensor header size in {path.name}")
            out[path] = (8 + size, json.loads(stream.read(size)))
    return out


@lru_cache(maxsize=1)
def libraries():
    """numpy and torch, imported when a pool is built: the file map and the plan need neither."""

    import numpy
    import torch

    return numpy, torch


@dataclass(frozen=True)
class Source:
    """Where expert 0 of one stacked tensor starts in a checkpoint file, and one expert's shape."""

    path: Path
    offset: int
    rows: int
    columns: int
    dtype: str

    @property
    def per_expert(self) -> int:
        """Bytes one expert of this tensor occupies (a row's values times the dtype's value size)."""

        return self.rows * self.columns * itemsize({"dtype": self.dtype}, self.path.name)

    def span(self, expert: int) -> tuple[int, int]:
        """(file offset, bytes) of one expert of this tensor."""

        return self.offset + expert * self.per_expert, self.per_expert


def sources(model_dir: Path, layers: int, experts: int) -> dict[tuple[int, str, str], Source]:
    """One Source per (layer, projection, part) of the routed stacks, refusing a checkpoint that disagrees."""

    found: dict[tuple[int, str, str], Source] = {}
    shapes: dict[tuple[str, str], tuple[tuple[int, ...], str]] = {}
    for path, (base, entries) in shards(model_dir).items():
        for name, entry in entries.items():
            if name == "__metadata__" or ROUTED not in name:
                continue
            part = name.rsplit(".", 1)[-1]
            layer, proj = _layer_of(name), _projection_of(name)
            if part not in PARTS or layer is None or proj is None:
                continue
            shape = tuple(int(n) for n in entry["shape"])
            if len(shape) != 3 or shape[0] != experts:
                raise ValueError(f"routed expert tensor {name}: shape {shape} is not [{experts}, rows, columns]")
            found[(layer, proj, part)] = Source(path, base + entry["data_offsets"][0], shape[1], shape[2],
                                                entry["dtype"])
            spec = (shape[1:], entry["dtype"])
            if shapes.setdefault((proj, part), spec) != spec:
                raise ValueError(f"routed expert tensor {name}: shape or dtype differs between layers")
    missing = [(layer, proj, part) for layer in range(layers) for proj in PROJS for part in PARTS
               if (layer, proj, part) not in found]
    if missing:
        first = missing[0]
        raise ValueError(f"the checkpoint's routed experts are incomplete: {len(missing)} tensors missing "
                         f"(first: layers.{first[0]}.mlp.switch_mlp.{first[1]}.{first[2]})")
    return found


def shared_sources(model_dir: Path, layers: int) -> dict[tuple[int, str, str], Source]:
    """One Source per (layer, projection, part) of a layer's shared expert: a single expert, always resident."""

    found: dict[tuple[int, str, str], Source] = {}
    for path, (base, entries) in shards(model_dir).items():
        for name, entry in entries.items():
            if name == "__metadata__" or ".mlp.shared_expert." not in name:
                continue
            part = name.rsplit(".", 1)[-1]
            layer, proj = _layer_of(name), _projection_of(name)
            if part not in PARTS or layer is None or proj is None:
                continue
            shape = tuple(int(n) for n in entry["shape"])
            if len(shape) != 2:
                raise ValueError(f"shared expert tensor {name}: shape {shape} is not [rows, columns]")
            found[(layer, proj, part)] = Source(path, base + entry["data_offsets"][0], shape[0], shape[1],
                                                entry["dtype"])
    missing = [(layer, proj, part) for layer in range(layers) for proj in PROJS for part in PARTS
               if (layer, proj, part) not in found]
    if missing:
        first = missing[0]
        raise ValueError(f"the checkpoint's shared experts are incomplete: {len(missing)} tensors missing "
                         f"(first: layers.{first[0]}.mlp.shared_expert.{first[1]}.{first[2]})")
    return found


def _layer_of(name: str) -> int | None:
    at = name.find("layers.")
    if at < 0:
        return None
    digits = name[at + len("layers."):].split(".", 1)[0]
    return int(digits) if digits.isdigit() else None


def _projection_of(name: str) -> str | None:
    """The projection a routed or shared expert tensor belongs to, by its own segment of the name."""

    return next((proj for proj in PROJS if f".{proj}." in name), None)


def bytes_per_expert(model_dir: str | Path, layers: int, experts: int) -> int:
    """Bytes of one expert of one layer: every projection's weight, scales and biases."""

    found = sources(Path(model_dir), layers, experts)
    return sum(source.per_expert for (layer, _, _), source in found.items() if layer == 0)


def per_expert_bytes(name: str, entry: dict) -> int:
    """Bytes one expert of a stacked tensor occupies: its rows' values times the dtype's value size."""

    return int(math.prod(entry["shape"][1:])) * itemsize(entry, name)


@dataclass
class Assignment:
    """One layer's call: the ids to read, the slot each requested id runs in, and the slots its reads overwrite."""

    load: list[tuple[int, int]]                 # (layer, expert) pairs to read from the checkpoint, in order
    slot_of: dict[tuple[int, int], int]         # every requested (layer, expert) -> the slot it runs in
    evict: list[int]                            # slots whose previous occupant is dropped, oldest first

    def slots(self, layer: int, ids: Sequence[int]) -> list[int]:
        """The slot each id of ``ids`` runs in, in the order given (a row's accumulation order is kept)."""

        return [self.slot_of[(layer, int(e))] for e in ids]


class Slots:
    """Residency for a pool of ``slots`` expert blocks: (layer, expert) -> slot, least recently used first."""

    def __init__(self, experts: int, slots: int, reserved: int = 1) -> None:
        """``reserved`` slots at the top hold the layers' shared experts, one a layer, for the whole run."""

        if slots < int(reserved) + 1:
            raise ValueError("an expert pool needs a slot a layer for its shared experts, and one more")
        self.experts, self.slots = int(experts), int(slots)
        self.reserved = int(reserved)
        self.shared = self.slots - self.reserved   # the first shared slot; layer i's is shared + i
        self.held: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.free = list(range(self.shared))
        self.hits = self.loads = 0

    def shared_slot(self, layer: int) -> int:
        """The slot a layer's shared expert keeps (the kernels see it as the expert after the routed ones)."""

        if not 0 <= int(layer) < self.reserved:
            raise ValueError(f"layer {layer} has no shared slot: the pool reserves {self.reserved}")
        return self.shared + int(layer)

    def assign(self, layer: int, wanted: Iterable[int]) -> Assignment:
        """Slots for one layer's routed ids; the shared expert's id is the caller's fixed slot, never planned here."""

        slot_of: dict[tuple[int, int], int] = {}
        load: list[tuple[int, int]] = []
        evict: list[int] = []
        ids = list(dict.fromkeys(int(e) for e in wanted))
        room = self.slots - self.reserved
        if len(ids) > room:
            raise ValueError(f"a call routing {len(ids)} distinct experts does not fit the pool's {room} routed "
                             f"slots, and one call's experts must all be resident at once: raise --expert-pool "
                             f"(a layer's {self.experts} experts and their shared experts need "
                             f"{self.reserved + self.experts + 1} slots)")
        for expert in ids:
            if expert >= self.experts:
                raise ValueError(f"routed expert id {expert} is outside this checkpoint's {self.experts}")
            key = (layer, expert)
            slot = self.held.get(key)
            if slot is not None:
                self.held.move_to_end(key)
                self.hits += 1
            else:
                if not self.free:
                    _, slot = self.held.popitem(last=False)
                    self.free.append(slot)
                    evict.append(slot)
                slot = self.free.pop()
                self.held[key] = slot
                load.append(key)
            slot_of[key] = slot
        self.loads += len(load)
        return Assignment(load, slot_of, evict)

    def resident(self) -> int:
        return len(self.held)

    def stats(self) -> dict:
        wanted = self.hits + self.loads
        return {"slots": self.slots, "reserved": self.reserved, "residency": self.resident(), "hits": self.hits,
                "loads": self.loads, "hit_rate": (self.hits / wanted) if wanted else 0.0}


class ExpertPool:
    """A pool of packed slots the grouped kernels index by slot, filled from the checkpoint one expert at a time."""

    def __init__(self, model_dir: str | Path, layers: int, experts: int, *, gs: int, slots: int, device: str,
                 limit: float = 0.0, readers: int = READERS) -> None:
        from tensorfold.cuda import experts as grouped

        if int(slots) > MAX_SLOTS:
            raise ValueError(f"the expert kernels take at most {MAX_SLOTS} experts in a stack, so a pool of "
                             f"{slots} slots cannot be served; cap --expert-pool")
        self.np, self.torch, self.grouped = *libraries(), grouped
        self.dir, self.layers, self.experts = Path(model_dir), int(layers), int(experts)
        self.gs, self.slots, self.device, self.limit = int(gs), int(slots), device, float(limit)
        self.found = sources(self.dir, self.layers, self.experts)
        self.spec: dict[tuple[str, str], Source] = {}
        for (_, proj, part), source in self.found.items():
            self.spec.setdefault((proj, part), source)
        self.plan = Slots(self.experts, self.slots, reserved=self.layers)
        self.shared = shared_sources(self.dir, self.layers)
        # one id -> slot table a layer: the routed ids are rewritten per call, the shared expert's id never
        self.table = self.torch.full((self.layers, self.experts + 1), self.plan.shared, dtype=self.torch.int32,
                                     device=self.device)
        self.fds: dict[Path, int] = {}
        self.readers = ThreadPoolExecutor(max(int(readers), 1), thread_name_prefix="expert-fill")
        self.read = 0                            # experts read since startup, for the log line
        first = self._pack(self._read(0, 0))          # one real expert sizes the pool and proves the read path
        # a packed block is [1, N/32, D/gs, block]; a slot is [N/32, D/gs, M, block], M = 2 (gate, up) or 1
        up, down = first["gate_proj"].shape, first["down_proj"].shape
        self.up = self.torch.empty((self.slots, *up[1:-1], 2, up[-1]), dtype=self.torch.int32, device=self.device)
        self.down = self.torch.empty((self.slots, *down[1:-1], 1, down[-1]), dtype=self.torch.int32,
                                     device=self.device)
        self.view = self.grouped.Experts(self.up, self.down, self.gs, self.spec[("gate_proj", "weight")].rows,
                                         self.spec[("down_proj", "weight")].rows, self.limit)
        self._place(self.plan.assign(0, [0]).slot_of[(0, 0)], first)
        for layer in range(self.layers):          # every layer's shared expert, resident in its own slot
            slot = self.plan.shared_slot(layer)
            self._place(slot, self._pack(self._read(layer, None)))
            self.table[layer, self.experts] = slot
        self.torch.cuda.synchronize(self.device)

    # ---- the checkpoint's side -------------------------------------------------------------------------------
    def skips(self, name: str) -> bool:
        """Whether the loader leaves this tensor to the pool: the routed stacks and the shared experts it reads."""

        return ROUTED in name or ".mlp.shared_expert." in name

    def _fd(self, path: Path) -> int:
        if path not in self.fds:
            self.fds[path] = os.open(path, os.O_RDONLY)
        return self.fds[path]

    def _read(self, layer: int, expert: int | None) -> dict[str, list[tuple[object, str]]]:
        """One expert's rows a projection, as host arrays: the file work, safe to run on many threads.

        ``expert`` None reads the layer's shared expert, whose tensors are single experts, not stacks.
        """

        out: dict[str, list[tuple[object, str]]] = {}
        for proj in PROJS:
            parts = []
            for part in PARTS:
                source = (self.found if expert is not None else self.shared)[(layer, proj, part)]
                offset, count = source.span(0 if expert is None else expert)
                raw = os.pread(self._fd(source.path), count, offset)
                if len(raw) != count:
                    raise OSError(f"short read of {source.path.name} at byte {offset}: {len(raw)} of {count}")
                number, _ = DTYPES[source.dtype]
                parts.append((self.np.frombuffer(raw, dtype=getattr(self.np, number))
                              .reshape(1, source.rows, source.columns).copy(), source.dtype))
            out[proj] = parts
        return out

    def _pack(self, blocks: dict[str, list[tuple[object, str]]]) -> dict[str, object]:
        """One expert's rows a projection, packed as the grouped kernels read them (one GPU call a projection)."""

        out = {}
        for proj, parts in blocks.items():
            tensors = [self.torch.from_numpy(array).to(self.device).view(getattr(self.torch, DTYPES[dtype][1]))
                       for array, dtype in parts]
            out[proj] = self.grouped.pack(*tensors, self.gs)
        return out

    def _place(self, slot: int, packed: dict[str, object]) -> None:
        """Write one packed expert into its slot: the pool's blocks are the kernels' landing buffers."""

        self.up[slot, ..., 0, :].copy_(packed["gate_proj"][0])
        self.up[slot, ..., 1, :].copy_(packed["up_proj"][0])
        self.down[slot, ..., 0, :].copy_(packed["down_proj"][0])

    # ---- the forward's side ----------------------------------------------------------------------------------
    def admit(self, layer: int, picks: object) -> None:
        """Fill this layer's missing experts, then rewrite ``picks`` as slot ids in place.

        ``picks`` [rows, top_k + 1] int32 carries routed ids with the shared expert's id last; every requested
        id gets a slot and the shared one its fixed slot, so the kernels see a full pool whatever is resident.
        """

        flat = picks.detach().reshape(-1).cpu().numpy()      # the ids, once: what is missing is a host decision
        got = self.plan.assign(layer, (int(e) for e in flat if int(e) < self.experts))
        if got.load:
            self._fill(got.load, got.slot_of)
        if got.slot_of:
            keys = self.torch.tensor(list(got.slot_of), dtype=self.torch.long, device=self.device)
            self.table[keys[:, 0], keys[:, 1]] = self.torch.tensor(list(got.slot_of.values()),
                                                                   dtype=self.torch.int32, device=self.device)
        flat = picks.reshape(-1).to(self.torch.int64)
        picks.copy_(self.table[layer].index_select(0, flat).reshape(picks.shape))

    def _fill(self, pairs: list[tuple[int, int]], slot_of: dict[tuple[int, int], int]) -> None:
        """Read each pair's expert and place it in the slot its call assigned; two or more reads overlap."""

        read = list(self.readers.map(lambda pair: self._read(*pair), pairs)) if len(pairs) > 1 else \
            [self._read(*pairs[0])]
        for (layer, expert), blocks in zip(pairs, read):
            self._place(slot_of[(layer, expert)], self._pack(blocks))
        was, self.read = self.read, self.read + len(pairs)
        if self.read // REPORT != was // REPORT:   # a mid-run receipt: the hit rate a shutdown may never print
            print(f"[tensorfold] expert pool: {self.plan.loads} experts read, {self.plan.hits} hits "
                  f"({self.plan.stats()['hit_rate']:.1%}), residency {self.plan.resident()}/{self.plan.slots}",
                  flush=True)

    def stats(self) -> dict:
        return {**self.plan.stats(), "pool_gib": self.slots * self.view.bytes_per_expert() / 2**30}

    def close(self) -> None:
        self.readers.shutdown(wait=True)
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
