"""Header-only CUDA startup estimates and a shared rank capacity decision."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Callable, Mapping

GIB = 1024**3
# safetensors dtype names -> bytes a value (FP8: the FP4 checkpoints' block scales)
SIZES = {"U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
         "BF16": 2, "F16": 2, "I16": 2, "U16": 2, "U32": 4, "I32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
LIMIT_ENV = "TENSORFOLD_CUDA_MEMORY_LIMIT_GB"


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
    keeps_tables: bool | None = None   # the window leaves the mapped tables their pages (None: nothing to keep)
    largest: int = 0                   # the largest window the budget fits up to the native one: what a restart gets
    resident: int = 0                  # the largest window that leaves the mapped tables their pages

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


def reserve_bytes(total: int) -> int:
    """Host RAM the startup keeps free: max(4 GiB, a tenth of the pool), or TENSORFOLD_MEMORY_RESERVE_GIB (>= 2)."""

    value = os.environ.get("TENSORFOLD_MEMORY_RESERVE_GIB", "").strip()
    if not value:
        return max(4 * GIB, total // 10)
    try:
        gib = float(value)
    except ValueError:
        gib = math.nan
    if not 2 <= gib <= total / GIB:
        raise ValueError(f"TENSORFOLD_MEMORY_RESERVE_GIB={value}: a number of GiB from 2 to the memory's size")
    return int(gib * GIB)


def host_stream_bytes() -> int | None:
    """Host staging room, with a 2-GiB default reserve or the explicit startup reserve override."""

    memory = _meminfo()
    if memory is None:
        return None
    reserve = (reserve_bytes(memory["MemTotal"])
               if os.environ.get("TENSORFOLD_MEMORY_RESERVE_GIB", "").strip() else 2 * GIB)
    return max(0, memory["MemAvailable"] - reserve)


def cuda_limit_bytes(environ: Mapping[str, str] | None = None) -> int | None:
    """The CUDA admission budget's explicit GiB cap in bytes, or None when unset.

    ``TENSORFOLD_CUDA_MEMORY_LIMIT_GB`` caps the grant the same absolute way ``TENSORFOLD_MEMORY_LIMIT_GB``
    caps the MLX budget. ValueError, naming the variable, for a nonpositive, non-finite, or non-numeric value.
    """

    value = (os.environ if environ is None else environ).get(LIMIT_ENV)
    if value is None:
        return None
    try:
        gib = float(value)
    except ValueError:
        raise ValueError(f"{LIMIT_ENV} must be a positive number in GiB") from None
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError(f"{LIMIT_ENV} must be a positive number in GiB")
    return int(gib * GIB)


def available_bytes(torch) -> int:
    """What admission and the runtime gate read as live: a discrete card's free memory, host RAM less a floor on a unified GPU.

    One pool on a unified GPU: reclaimable page cache counts as available, and the host keeps a floor of that
    pool free, because CUDA context, NCCL and workspace memory sit outside the estimate and an exhausted GB10
    freezes the host. A discrete card's host need is its loading buffers, which ``host_stream_bytes`` weighs on
    its own. ``TENSORFOLD_MEMORY_RESERVE_GIB`` moves the floor; ``TENSORFOLD_CUDA_MEMORY_LIMIT_GB`` caps the
    grant from above in GiB, an absolute budget like the MLX one.
    """

    free, total = map(int, torch.cuda.mem_get_info())
    if not unified(torch):
        granted = free                                   # a discrete card's memory is its own pool
    else:
        # No /proc/meminfo means the card's memory is the pool, so the floor comes out of it
        memory = _meminfo()
        granted = (memory["MemAvailable"] if memory is not None else free) - reserve_bytes(
            memory["MemTotal"] if memory is not None else total)
    limit = cuda_limit_bytes()
    return max(0, min(granted, limit)) if limit is not None else max(0, granted)


def total_bytes(torch) -> int:
    """The GPU's memory (a GB10's is the host's): the same on every rank, so what it sizes agrees without a gather."""

    return int(torch.cuda.mem_get_info()[1])


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

    fitting, keeps, resident = fit(budget), None, 0
    if weights.mapped and room is not None:
        # windows up to ``resident`` keep mapped tables in the page cache; past it they page
        resident = fit(min(budget, room - weights.mapped))
        if explicit:
            keeps = 0 < resident >= upper
        else:
            fitting, keeps = (resident, True) if resident else (fitting, False)
    largest = fit(budget, native if native > 0 else target)
    return Plan(native, requested, bool(explicit), fitting, int(budget), weights, geometry, keeps, largest, resident)


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


def floor(model_dir: str | Path) -> tuple[int, int]:
    """The compute capability a checkpoint's kernels need: 8.9 for every format (clusters are taken where present)."""

    from tensorfold.cuda import build

    return build.MIN_CAPABILITY


def admit(model_dir: str | Path, requested: int | None, explicit: bool | None, torch,
          geometry: Geometry | Callable, transform: Callable, *, rank: int = 0, world: int = 1,
          gather: Callable | None = None, draft_dir: Path | None = None,
          draft_geometry: Geometry | Callable | None = None, startup_copies: int = 0,
          extra_files: tuple[Path, ...] = (), files: list[Path] | None = None,
          draft_transform: Callable | None = None,
          draft_weights: Callable[[Path], Weights] | None = None) -> dict:
    """One refusal or capacity on both ranks before allocating; the draft model by ``draft_weights`` or a transform."""

    from tensorfold.cuda import build

    build.refuse_old_gpu(floor(model_dir))          # an old GPU is refused here, before any weight loads
    error = None
    plan = None
    try:
        text = config(model_dir)
        geometry = geometry(text) if callable(geometry) else geometry
        weights = estimate_weights(model_dir, transform, rank=rank, files=files)
        host_staging = weights.staging
        if extra_files:                      # files outside the index, same layout (Nemotron's MTP head, EXL3 tables)
            more = estimate_weights(model_dir, transform, files=list(extra_files))
            host_staging = max(host_staging, more.staging)
            weights = Weights(weights.resident + more.resident, max(weights.staging, more.staging),
                              weights.mapped + more.mapped)
        weights = Weights(weights.resident, weights.staging + startup_copies * weights.resident, weights.mapped)
        if draft_dir is not None:
            draft = draft_weights(draft_dir) if draft_weights is not None else estimate_weights(
                draft_dir, draft_transform or (lambda name, info: (math.prod(info["shape"]) * max(4, itemsize(info, name)),
                                                                   0)))
            host_staging = max(host_staging, draft.staging)
            # the drafter loads after the target: the peak is the larger of either load's
            weights = Weights(weights.resident + draft.resident, max(weights.staging - draft.resident, draft.staging),
                              weights.mapped)
            if draft_geometry is not None:
                draft_geometry = draft_geometry(config(draft_dir)) if callable(draft_geometry) else draft_geometry
                main = geometry
                geometry = Geometry(lambda slots: main.bytes_at(slots) + draft_geometry.bytes_at(slots),
                                    main.reserve, main.minimum_slots)
        if not unified(torch):
            host_free = host_stream_bytes()
            if host_free is not None and host_staging > host_free:
                raise ValueError(f"host staging needs an estimated {host_staging / GIB:.2f} GiB, "
                                 f"but only {host_free / GIB:.2f} GiB is available after its reserve; "
                                 "free host memory or use a checkpoint with smaller loading buffers")
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
    note = tables_note(plan)
    if note:
        print(f"[tensorfold] {note}", flush=True)
    return receipt


def tables_note(plan: Plan) -> str | None:
    """What startup says when the window leaves the mapped tables no room (their lookups then read the disk)."""

    if plan.keeps_tables is not False:
        return None
    fix = (f"a --context of {plan.resident} or less, or fewer --parallel streams, keeps them resident"
           if plan.explicit and plan.resident else "free memory to keep them resident")
    return (f"the {plan.weights.mapped / GIB:.1f} GiB of mapped tables do not fit beside the weights and caches: "
            f"lookups will page them from disk, which slows prompts ({fix})")


def gather_ints(torch, gather: Callable, values: list[int], world: int = 2) -> list[list[int]]:
    send = torch.tensor(values, dtype=torch.int64, device="cuda")
    receive = torch.empty((world * len(values),), dtype=torch.int64, device="cuda")
    gather(send, receive)
    return receive.view(world, -1).tolist()
