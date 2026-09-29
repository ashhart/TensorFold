"""Read the hardware attributes in benchmark receipts without collecting device identities."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

_GIB = 1024**3
_MODEL = re.compile(r"(?:Apple M\d+(?: Pro| Max| Ultra)?|(?:Intel|AMD|NVIDIA|Tesla|Quadro|GeForce|ARM|Ampere|Qualcomm)[A-Za-z0-9 ().+_-]{0,110})")
_VERSION = re.compile(r"[0-9][A-Za-z0-9.+_-]{0,79}")


def _model_name(value: Any) -> str | None:
    """Accept processor and GPU descriptions, rather than arbitrary profiler strings."""

    if not isinstance(value, str):
        return None
    name = " ".join(value.strip().split())
    name = re.sub(r"\s+@\s+[0-9.]+\s*[GM]Hz$", "", name, flags=re.IGNORECASE)
    return name if len(name) <= 120 and _MODEL.fullmatch(name) else None


def _version(value: Any) -> str | None:
    return value if isinstance(value, str) and _VERSION.fullmatch(value) else None


def _run(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or len(result.stdout) > 4 * 1024**2:
        return None
    return result.stdout


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if 0 < number < 2**63 else None


def _system_memory() -> int | None:
    if sys.platform == "darwin":
        return _positive_int(_run(["sysctl", "-n", "hw.memsize"]))
    try:
        for row in Path("/proc/meminfo").read_text().splitlines():
            if row.startswith("MemTotal:"):
                value, unit = row.split()[1:3]
                return _positive_int(value) * 1024 if unit == "kB" and _positive_int(value) else None
    except (OSError, ValueError):
        pass
    try:
        return _positive_int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        return None


def _cpu_model() -> str | None:
    if sys.platform == "darwin":
        return _model_name(_run(["sysctl", "-n", "machdep.cpu.brand_string"]))
    try:
        for row in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = row.partition(":")
            if separator and key.strip() in ("model name", "Hardware"):
                found = _model_name(value)
                if found:
                    return found
    except OSError:
        pass
    return None


def _mac_gpus() -> list[dict[str, Any]]:
    # The profiler also includes serials and display identifiers. Only these fields leave this function.
    raw = _run(["system_profiler", "SPDisplaysDataType", "-json"])
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError, RecursionError):
        return []
    displays = data.get("SPDisplaysDataType", []) if isinstance(data, dict) else []
    if not isinstance(displays, list):
        return []
    gpus = []
    for display in displays:
        if not isinstance(display, dict):
            continue
        name = _model_name(display.get("sppci_model"))
        if name is None:
            continue
        cores = _positive_int(display.get("sppci_cores"))
        gpus.append({"name": name, "memory_bytes": None, "cores": cores})
    return gpus


def _nvidia_rows() -> list[tuple[str, int | None, str | None]]:
    """Selected nvidia-smi columns only, with no UUIDs, serials, process names or machine identifiers."""

    raw = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if raw is None:
        return []
    rows = []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) != 3:
            continue
        name = _model_name(row[0])
        if name is None:
            continue
        memory = _positive_int(row[1].strip())
        rows.append((name, memory * 1024**2 if memory else None, _version(row[2].strip())))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        if not visible.strip() or visible.strip() == "-1":
            return []
        indices = visible.split(",")
        # UUID-based selectors cannot be resolved without querying device identities.
        if not all(index.strip().isdigit() for index in indices):
            return []
        rows = [rows[int(index)] for index in indices if int(index) < len(rows)]
    return rows


def driver_version() -> str | None:
    versions = {row[2] for row in _nvidia_rows() if row[2] is not None}
    return next(iter(versions)) if len(versions) == 1 else None


def detect(backend: str) -> dict[str, Any]:
    """Collect an allowlist of chip, memory and visible GPU descriptions using the standard library."""

    if backend == "auto":
        backend = "mlx" if sys.platform == "darwin" else "cuda"
    if backend not in ("mlx", "cuda"):
        raise ValueError("Benchmark backend must be mlx or cuda")
    cpu, memory = _cpu_model(), _system_memory()
    if backend == "mlx":
        gpus = _mac_gpus() if sys.platform == "darwin" else []
        unified = any(gpu["name"].startswith("Apple M") for gpu in gpus)
        memory_type = "unified" if unified else "unknown"
        if unified:
            for gpu in gpus:
                gpu["memory_bytes"] = memory
    else:
        rows = _nvidia_rows()
        gpus = [{"name": name, "memory_bytes": size, "cores": None} for name, size, _ in rows]
        memory_type = ("unified" if all(gpu["name"] == "NVIDIA GB10" for gpu in gpus) else "dedicated") if gpus else "unknown"
    evidence = cpu is not None or memory is not None or bool(gpus)
    complete = cpu is not None and memory is not None and bool(gpus) and all(gpu["memory_bytes"] for gpu in gpus)
    return {"cpu_model": cpu, "system_memory_bytes": memory, "memory_type": memory_type,
            "gpus": gpus, "gpu_count": len(gpus) if gpus else None,
            "source": "detected" if complete else "partial" if evidence else "unavailable"}
