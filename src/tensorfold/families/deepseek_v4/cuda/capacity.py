"""Single-Spark admission from actual GGUF headers and donor allocator geometry."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

from .build_runtime import inspect_inputs
from .native import estimate

GIB = 1 << 30
FLOOR = 8 * GIB
RUNTIME_RESERVE = GIB  # conservative driver/Python/staging allowance, not a claimed measurement


def available():
    try:
        memory = {
            line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
        }
        value = memory["MemAvailable"]
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise ValueError("host memory availability is unavailable; refusing native load") from exc
    if value <= 0:
        raise ValueError("host memory availability is invalid; refusing native load")
    return value


def admit(model_dir, context=None, context_explicit=None):
    if context is None:
        context = 262144
    if isinstance(context, bool) or not isinstance(context, int) or not 1 <= context < 2**31:
        raise ValueError("context must be a positive supported integer")
    room = available()
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    descriptor = json.loads((model_dir / "descriptor.json").read_text())
    reserve = float(descriptor["reserve_gib"])
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("additional companion growth reserve must be finite/nonnegative")

    def local(value):
        path = Path(value)
        return path if path.is_absolute() else model_dir / path

    source, library = local(config["gguf_file"]), local(config["native_library"])
    report = inspect_inputs(
        SimpleNamespace(gguf=source, tokenizer_dir=None, context=context, companion_reserve_gib=reserve)
    )
    native = int(report["arch"]["deepseek4.context_length"])
    if context > native:
        raise ValueError(f"explicit context {context} exceeds native {native}; no context reduction")
    if descriptor.get("sha256") and descriptor["sha256"] != report["header_sha256"]:
        raise ValueError("prepared GGUF header fingerprint changed")
    if descriptor.get("size") and descriptor["size"] != report["source_size"]:
        raise ValueError("prepared GGUF size changed")
    identity = descriptor.get("source_identity", {})
    for key in ("mtime_ns", "device", "inode"):
        if key in identity and identity[key] != report["source_identity"][key]:
            raise ValueError(f"prepared GGUF identity changed: {key}")
    geometry = estimate(library, source, context)
    growth = math.ceil(reserve * GIB)
    weight_bytes = report["source_size"] + report.get("aligned_artifact_extra_bytes", 0)
    snapshot = geometry["snapshot_bytes"]
    required = weight_bytes + geometry["graph_bytes"] + snapshot + RUNTIME_RESERVE + FLOOR + growth
    # MemAvailable already accounts for the resident companions. Count only their
    # measured future growth, and protect the unified pool's floor exactly once.
    room = min(room, available())
    if required > room:
        raise ValueError(
            f"cannot fit context {context}: need {required / GIB:.2f} GiB "
            f"including mapped weights/cache/runtime/growth/8GiB floor; "
            f"available {room / GIB:.2f} GiB; no weights or KV cache loaded"
        )
    return {
        "context_window": context,
        "cache_slots": context,
        "native_window": native,
        "mapped_weights_bytes": report["source_size"],
        "geometry": geometry,
        "snapshot_reserve_bytes": snapshot,
        "aligned_artifact_extra_bytes": report.get("aligned_artifact_extra_bytes", 0),
        "weight_residency_budget_bytes": weight_bytes,
        "runtime_reserve_bytes": RUNTIME_RESERVE,
        "floor_bytes": FLOOR,
        "companion_growth_bytes": growth,
        "required_bytes": required,
        "available_bytes": room,
        "total_bytes_estimate": required - FLOOR - growth,
        "donor_estimator": "ds4_engine_session_graph_bytes_estimate less inactive F32 primary shells",
        "largest_window": context,
    }
