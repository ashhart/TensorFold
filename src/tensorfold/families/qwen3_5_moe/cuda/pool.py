"""Qwen3.6's routed experts served from the checkpoint's files: the slots a pool holds and what to read.

``sources`` and ``Slots`` are pure: they read headers and plan residency, and both are testable without a GPU.
A slot is one expert's packed block (gate, up and down); a layer's ids are remapped to slots before the grouped
kernels run, so a (row, slot) pair keeps the arithmetic it has with the stacks resident.
"""

from __future__ import annotations

import json
import math
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from tensorfold.cuda.capacity import itemsize

ROUTED = ".mlp.switch_mlp."          # the routed stacks; the router (mlp.gate) and the shared expert stay resident
PARTS = ("weight", "scales", "biases")
PROJS = ("gate_proj", "up_proj", "down_proj")


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


@dataclass(frozen=True)
class Source:
    """Where expert 0 of one stacked tensor starts in a checkpoint file, and the bytes of one expert."""

    path: Path
    offset: int
    per_expert: int

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
            found[(layer, proj, part)] = Source(path, base + entry["data_offsets"][0],
                                               per_expert_bytes(name, entry))
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


def _layer_of(name: str) -> int | None:
    at = name.find("layers.")
    if at < 0:
        return None
    digits = name[at + len("layers."):].split(".", 1)[0]
    return int(digits) if digits.isdigit() else None


def per_expert_bytes(name: str, entry: dict) -> int:
    """Bytes one expert of a stacked tensor occupies: its rows' values times the dtype's value size."""

    return int(math.prod(entry["shape"][1:])) * itemsize(entry, name)


def _projection_of(name: str) -> str | None:
    return next((proj for proj in PROJS if f".switch_mlp.{proj}." in name), None)


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

    def __init__(self, experts: int, slots: int) -> None:
        if slots < 2:
            raise ValueError("an expert pool needs at least two slots")
        self.experts, self.slots = int(experts), int(slots)
        self.shared = self.slots - 1              # the shared expert keeps the last slot for the whole run
        self.held: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.free = list(range(self.shared))
        self.hits = self.loads = 0

    def assign(self, layer: int, wanted: Iterable[int]) -> Assignment:
        """Slots for one layer's routed ids; the shared expert's id is the caller's fixed slot, never planned here."""

        slot_of: dict[tuple[int, int], int] = {}
        load: list[tuple[int, int]] = []
        evict: list[int] = []
        for expert in dict.fromkeys(int(e) for e in wanted):
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
        return {"slots": self.slots, "residency": self.resident(), "hits": self.hits, "loads": self.loads,
                "hit_rate": (self.hits / wanted) if wanted else 0.0}
