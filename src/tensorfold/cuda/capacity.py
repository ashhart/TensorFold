"""Header-only CUDA startup estimates and a shared rank capacity decision."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import struct
from typing import Callable

GIB = 1024**3
# safetensors dtype names -> bytes a value in the file. FP8 comes with the FP4 checkpoints: their block scales
# are `F8_E4M3` (the published Swift revision carries 73,728 of them, the only dtype this map lacked).
SIZES = {"U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
         "BF16": 2, "F16": 2, "I16": 2, "U16": 2, "U32": 4, "I32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def itemsize(info: dict, name: str) -> int:
    """Bytes a value for one tensor header; an unknown dtype names the tensor instead of raising a bare KeyError."""

    dtype = info.get("dtype")
    item = SIZES.get(dtype) if isinstance(dtype, str) else None
    if item is None:
        raise ValueError(f"checkpoint tensor {name} has dtype {dtype!r}, which the startup estimate cannot "
                         f"size; known dtypes: {', '.join(sorted(SIZES))}")
    return item


@dataclass(frozen=True)
class Weights:
    resident: int
    staging: int
    mapped: int = 0  # read-only, unpinned file pages, reclaimable by the OS


@dataclass(frozen=True)
class Geometry:
    """Allocated cache, recurrence and bounded scratch bytes at a cache-slot capacity."""

    bytes_at: Callable[[int], int]
    reserve: int
    minimum_slots: int = 0

    def needed(self, window: int) -> int:
        return int(self.bytes_at(max(self.minimum_slots, window + self.reserve)))


@dataclass(frozen=True)
class Plan:
    native: int
    requested: int | None
    explicit: bool
    fitting: int
    budget: int
    weights: Weights
    geometry: Geometry
    keeps_tables: bool | None = None   # a default window sized so the mapped tables keep their pages
    largest: int = 0                   # the largest window the budget fits up to the native one: what a restart gets

    @property
    def settings(self) -> list[int]:
        return [self.native, -1 if self.requested is None else self.requested, int(self.explicit)]

    def receipt(self, window: int) -> dict:
        return {"native_window": self.native, "requested_context": self.requested,
                "explicit_context": self.explicit, "context_window": window,
                "cache_slots": max(self.geometry.minimum_slots, window + self.geometry.reserve),
                "budget_bytes": self.budget, "weight_bytes_estimate": self.weights.resident,
                "loading_bytes_estimate": self.weights.staging, "mapped_table_bytes": self.weights.mapped,
                "cache_workspace_bytes_estimate": self.geometry.needed(window),
                "startup_peak_bytes_estimate": self.weights.resident + self.weights.staging,
                "serving_peak_bytes_estimate": self.weights.resident + self.geometry.needed(window),
                "total_bytes_estimate": self.weights.resident + max(self.weights.staging, self.geometry.needed(window)),
                "full_mapped_working_set_bytes_estimate": self.weights.resident +
                max(self.weights.staging, self.geometry.needed(window)) + self.weights.mapped,
                "mapped_pages_reclaimable": True, "mapped_tables_resident": self.keeps_tables}


def config(model_dir: str | Path) -> dict:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    text = dict(raw.get("text_config") or raw)
    text["_quantization"] = raw.get("quantization") or raw.get("quantization_config") or {}
    return text


def headers(model_dir: str | Path, *, rank: int | None = None, files: list[Path] | None = None) -> dict:
    path = Path(model_dir)
    files = list(files or []) or (sorted(path.glob(f"*.rank{rank}.safetensors")) if rank is not None else [])
    if not files:
        other = sorted(path.glob("*.rank*.safetensors"))
        if other:
            raise ValueError("checkpoint contains another rank's split weights; use this rank's folder")
        index = path / "model.safetensors.index.json"
        files = ([path / n for n in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
                 if index.exists() else sorted(path.glob("*.safetensors")))
    if not files:
        raise ValueError("startup memory estimate needs the checkpoint tensor headers")
    out = {}
    for file in files:
        with file.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size <= 64 * 1024**2:
                raise ValueError("invalid checkpoint tensor header size")
            entries = json.loads(stream.read(size))
        for name, info in entries.items():
            if name == "__metadata__":
                continue
            if name in out:
                raise ValueError(f"duplicate checkpoint tensor: {name}")
            shape = info["shape"]
            item = itemsize(info, name)
            if any(int(n) < 0 for n in shape) or math.prod(shape) * item != info["data_offsets"][1] - info["data_offsets"][0]:
                raise ValueError(f"invalid checkpoint tensor geometry: {name}")
            out[name] = {**info, "split": ".rank" in file.name}
    return out


def estimate_weights(model_dir: str | Path, transform: Callable, *, rank: int | None = None,
                     files: list[Path] | None = None) -> Weights:
    layers: dict[str, int] = {}
    resident = mapped = largest = 0
    for name, info in headers(model_dir, rank=rank, files=files).items():
        size, host = transform(name, info)
        size, host = int(size), int(host)
        if min(size, host) < 0:
            raise ValueError("negative startup weight estimate")
        resident += size
        mapped += host
        largest = max(largest, size)
        match = re.search(r"(?:layers|blocks)\.(\d+)\.", name)
        group = match.group(1) if match else name
        layers[group] = layers.get(group, 0) + size
    # CPU expert lists/stack, GPU uploads and tiled outputs can coexist during one layer load.
    staging = 3 * max([largest, *layers.values()], default=0)
    return Weights(resident, staging, mapped)


def _meminfo() -> dict | None:
    try:
        rows = Path("/proc/meminfo").read_text().splitlines()
        memory = {key.rstrip(":"): int(value) * 1024 for key, value, *_ in (row.split() for row in rows)}
    except (OSError, ValueError):
        return None
    return memory if {"MemTotal", "MemAvailable"} <= memory.keys() else None


def unified(torch) -> bool:
    """A GPU on the host's memory (GB10): its free figure is MemFree, which counts the page cache as used."""

    try:
        return bool(torch.cuda.get_device_properties(0).is_integrated)
    except (AttributeError, AssertionError, RuntimeError):
        return False


def available_bytes(torch) -> int:
    free, total = map(int, torch.cuda.mem_get_info())
    available = max(0, free - max(4 * GIB, math.ceil(total / 10)))
    memory = _meminfo()
    if memory is None:
        return available
    host = max(0, memory["MemAvailable"] - max(4 * GIB, memory["MemTotal"] // 10))
    # one pool on a unified GPU: reclaimable page cache is available; a discrete GPU is bounded by both
    return host if unified(torch) else min(available, host)


def page_room(torch) -> int | None:
    """What caches and mapped read-only tables share on a unified GPU (MemAvailable); None on a discrete GPU."""

    memory = _meminfo()
    return memory["MemAvailable"] if memory is not None and unified(torch) else None


def make_plan(native: int, requested: int | None, explicit: bool, budget: int,
              weights: Weights, geometry: Geometry, room: int | None = None) -> Plan:
    native = int(native)
    requested = None if requested is None else int(requested)
    if requested is not None and requested < 0:
        raise ValueError("context must be 0 or a positive token count")
    target = requested if requested else native
    if target <= 0:
        raise ValueError("checkpoint has no native window; give an explicit positive --context")
    upper = min(target, native) if native > 0 else target

    def fit(ceiling: int, top: int = upper) -> int:
        low, high = 0, 0 if weights.resident + weights.staging > budget else top
        while low < high:
            middle = (low + high + 1) // 2
            if weights.resident + geometry.needed(middle) <= ceiling:
                low = middle
            else:
                high = middle - 1
        return low

    fitting, keeps = fit(budget), None
    if not explicit and weights.mapped and room is not None:
        # a default window leaves the mapped tables their pages (page cache, like the reserve); else they page
        resident = fit(min(budget, room - weights.mapped))
        fitting, keeps = (resident, True) if resident else (fitting, False)
    largest = fit(budget, native if native > 0 else target)
    return Plan(native, requested, bool(explicit), fitting, int(budget), weights, geometry, keeps, largest)


def choose(plan: Plan, peers: list[list[int]] | None = None) -> int:
    """Choose one window for every rank; an explicit nonfit request refuses on every rank."""

    rows = peers if peers is not None else [plan.settings + [plan.fitting]]
    if any(row[:3] != plan.settings for row in rows):
        raise ValueError("CUDA ranks have different native windows or context flags; start both with the same flags")
    fitting = min(row[3] for row in rows)
    target = plan.requested if plan.requested else plan.native
    if plan.explicit and plan.requested and plan.native > 0 and plan.requested > plan.native:
        raise ValueError(f"requested --context {plan.requested} exceeds the checkpoint's {plan.native}-token native window; "
                         f"estimated fitting prompt-plus-reply capacity is {fitting} tokens; reduce --context and "
                         "the prompt/reply reserve, or free memory/use smaller weights")
    if fitting <= 0 or (plan.explicit and plan.requested and target > fitting):
        kind = "native" if target == plan.native else "default"
        wanted = (f"requested context {target}" if plan.explicit and plan.requested
                  else f"the {target}-token {kind} window or any smaller one")
        raise ValueError(f"CUDA startup memory budget cannot fit {wanted}; estimated largest fitting "
                         f"prompt-plus-reply window: {fitting} tokens across the ranks. " +
                         (f"Use --context {fitting} with a smaller prompt/reply reserve, or " if fitting else "Please ") +
                         "free memory or use smaller/quantized weights; no model weights "
                         "or KV caches have been loaded. KV precision is unchanged.")
    return min(target, fitting)


def admit(model_dir: str | Path, requested: int | None, explicit: bool | None, torch,
          geometry: Geometry | Callable, transform: Callable, *, rank: int = 0, world: int = 1,
          gather: Callable | None = None, draft_dir: Path | None = None,
          draft_geometry: Geometry | Callable | None = None, startup_copies: int = 0,
          extra_files: tuple[Path, ...] = (), files: list[Path] | None = None) -> dict:
    """Reach the same refusal or capacity before either rank allocates model tensors."""

    error = None
    plan = None
    try:
        text = config(model_dir)
        geometry = geometry(text) if callable(geometry) else geometry
        weights = estimate_weights(model_dir, transform, rank=rank, files=files)
        if extra_files:                      # files outside the index, same layout (Nemotron's MTP head, EXL3 tables)
            more = estimate_weights(model_dir, transform, files=list(extra_files))
            weights = Weights(weights.resident + more.resident, max(weights.staging, more.staging),
                              weights.mapped + more.mapped)
        weights = Weights(weights.resident, weights.staging + startup_copies * weights.resident, weights.mapped)
        if draft_dir is not None:
            draft = estimate_weights(draft_dir, lambda name, info: (math.prod(info["shape"]) *
                                      max(4, itemsize(info, name)), 0))
            weights = Weights(weights.resident + draft.resident, weights.staging + draft.staging, weights.mapped)
            if draft_geometry is not None:
                draft_geometry = draft_geometry(config(draft_dir)) if callable(draft_geometry) else draft_geometry
                main = geometry
                geometry = Geometry(lambda slots: main.bytes_at(slots) + draft_geometry.bytes_at(slots),
                                    main.reserve, main.minimum_slots)
        plan = make_plan(int(text.get("max_position_embeddings") or 0), requested,
                         requested is not None if explicit is None else explicit,
                         available_bytes(torch), weights, geometry, room=page_room(torch))
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        error = f"{type(exc).__name__}: {exc}"     # name the cause: its text alone has hidden a dtype's KeyError
    status = [1 if error else 0, *(plan.settings + [plan.fitting, plan.largest] if plan else [0, -1, 0, 0, 0])]
    both = gather(status) if world > 1 else [status]
    if any(row[0] for row in both):
        raise ValueError("CUDA startup memory geometry could not be established on every rank: " +
                         (error or "another rank could not read its checkpoint; check both folders/configs"))
    window = choose(plan, [row[1:] for row in both])
    receipt = {**plan.receipt(window), "largest_window": min(row[5] for row in both)}
    print(f"[tensorfold] CUDA rank {rank} startup estimate {receipt['total_bytes_estimate'] / GIB:.2f} GiB "
          f"within {plan.budget / GIB:.2f} GiB; native {plan.native}, allocated prompt/reply window {window}, "
          f"cache slots {receipt['cache_slots']}", flush=True)
    if plan.keeps_tables is False:
        print(f"[tensorfold] the {plan.weights.mapped / GIB:.1f} GiB of mapped tables do not fit beside the weights "
              "and caches: lookups will page them from disk (free memory to keep them resident)", flush=True)
    return receipt


def gather_ints(torch, gather: Callable, values: list[int], world: int = 2) -> list[list[int]]:
    send = torch.tensor(values, dtype=torch.int64, device="cuda")
    receive = torch.empty((world * len(values),), dtype=torch.int64, device="cuda")
    gather(send, receive)
    return receive.view(world, -1).tolist()
