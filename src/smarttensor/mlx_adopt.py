"""Optional zero-copy MLX adoption bridge for Python-owned raw buffers."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Sequence
import time


def mlx_dtype_name(manifest_dtype: str) -> str:
    """Map SmartTensor manifest dtypes to the MLX extension dtype names."""

    mapping = {
        "U8": "uint8",
        "U16": "uint16",
        "U32": "uint32",
        "I32": "int32",
        "F16": "float16",
        "F32": "float32",
        "BF16": "bfloat16",
    }
    try:
        return mapping[str(manifest_dtype)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dtype for MLX raw-buffer adoption: {manifest_dtype}"
        ) from exc


def _load_probe_builder() -> Any:
    """Return the temporary adoption-extension builder.

    Benchmark entrypoints are usually launched as scripts, so they may expose
    `benchmarks/` directly on sys.path instead of as an importable package.
    """

    errors: list[ModuleNotFoundError] = []
    for module_name in (
        "benchmarks.mlx_python_extension_adoption_probe",
        "mlx_python_extension_adoption_probe",
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            errors.append(exc)
            continue
        return module.build_mlx_python_extension_adoption_module
    raise errors[-1]


@lru_cache(maxsize=1)
def _load_probe_extension() -> Any:
    """Load the currently proven adoption extension from the benchmark probe.

    This is intentionally a temporary bridge: it lets the runtime path exercise
    the exact zero-copy primitive while the extension is still being hardened
    into a persistent package artifact.
    """

    return _load_probe_builder()()


def prewarm_extension() -> dict[str, Any]:
    """Load the optional MLX adoption extension and report startup timing."""

    started = time.perf_counter()
    module = _load_probe_extension()
    return {
        "enabled": True,
        "skipped": False,
        "seconds": time.perf_counter() - started,
        "module": type(module).__name__,
    }


def adopt_py_buffer_as(
    buffer: Any,
    shape: Sequence[int],
    dtype_name: str,
) -> Any:
    """Return a shaped MLX array that adopts a Python-owned raw byte buffer."""

    module = _load_probe_extension()
    return module.adopt_py_buffer_as(buffer, [int(dim) for dim in shape], dtype_name)


def adopt_py_buffer_as_unowned(
    buffer: Any,
    shape: Sequence[int],
    dtype_name: str,
) -> Any:
    """Return a shaped MLX array over caller-owned Python buffer memory.

    The caller must keep `buffer` alive for at least as long as the returned
    array may be used. This avoids running Python C-API cleanup from MLX's
    asynchronous Metal buffer deleter.
    """

    module = _load_probe_extension()
    return module.adopt_py_buffer_as_unowned(
        buffer,
        [int(dim) for dim in shape],
        dtype_name,
    )


def adopt_py_buffer_as_owned_copy(
    buffer: Any,
    shape: Sequence[int],
    dtype_name: str,
) -> Any:
    """Return a shaped MLX array backed by extension-owned copied bytes."""

    module = _load_probe_extension()
    return module.adopt_py_buffer_as_owned_copy(
        buffer,
        [int(dim) for dim in shape],
        dtype_name,
    )


def pread_regions_as_owned_array(
    path: str,
    regions: Sequence[tuple[int, int]],
    shape: Sequence[int],
    dtype_name: str,
) -> Any:
    """Read file regions directly into extension-owned MLX array storage."""

    module = _load_probe_extension()
    normalized_regions = [
        (int(offset), int(length)) for offset, length in regions
    ]
    return module.pread_regions_as_owned_array(
        str(path),
        normalized_regions,
        [int(dim) for dim in shape],
        dtype_name,
    )


def pread_many_regions_as_owned_arrays(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int]], Sequence[int], str]
    ],
) -> dict[str, Any]:
    """Read several file-region arrays from one path into MLX-owned storage."""

    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [(int(offset), int(length)) for offset, length in regions],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, regions, shape, dtype_name in specs
    ]
    return dict(module.pread_many_regions_as_owned_arrays(str(path), normalized_specs))


def pread_many_regions_as_owned_arrays_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int]], Sequence[int], str]
    ],
    *,
    max_workers: int,
) -> dict[str, Any]:
    """Read several region arrays in one extension call using internal workers."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [(int(offset), int(length)) for offset, length in regions],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, regions, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_regions_as_owned_arrays_threaded(
            str(path),
            normalized_specs,
            int(max_workers),
        )
    )


def pread_many_regions_as_slab_owned_arrays_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int]], Sequence[int], str]
    ],
    *,
    max_workers: int,
) -> dict[str, Any]:
    """Read several arrays into one extension-owned slab, then return MLX views."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [(int(offset), int(length)) for offset, length in regions],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, regions, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_regions_as_slab_owned_arrays_threaded(
            str(path),
            normalized_specs,
            int(max_workers),
        )
    )


def pread_many_regions_as_slab_owned_arrays_region_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int]], Sequence[int], str]
    ],
    *,
    max_workers: int,
) -> dict[str, Any]:
    """Read arrays into one owned slab with workers scheduled per region."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [(int(offset), int(length)) for offset, length in regions],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, regions, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_regions_as_slab_owned_arrays_region_threaded(
            str(path),
            normalized_specs,
            int(max_workers),
        )
    )


def pread_many_regions_as_bytearrays_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int]], Sequence[int], str]
    ],
    *,
    max_workers: int,
) -> dict[str, bytearray]:
    """Read several region arrays into Python-owned bytearrays with workers."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [(int(offset), int(length)) for offset, length in regions],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, regions, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_regions_as_bytearrays_threaded(
            str(path),
            normalized_specs,
            int(max_workers),
        )
    )


def pread_many_regions_as_bytearrays_region_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int]], Sequence[int], str]
    ],
    *,
    max_workers: int,
) -> dict[str, bytearray]:
    """Read region arrays into bytearrays with workers scheduled per region."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [(int(offset), int(length)) for offset, length in regions],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, regions, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_regions_as_bytearrays_region_threaded(
            str(path),
            normalized_specs,
            int(max_workers),
        )
    )


def pread_many_regions_into_bytearrays_region_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int]], Sequence[int], str]
    ],
    buffers: dict[str, bytearray],
    *,
    max_workers: int,
) -> dict[str, bytearray]:
    """Read region arrays into caller-owned bytearrays with per-run workers."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [(int(offset), int(length)) for offset, length in regions],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, regions, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_regions_into_bytearrays_region_threaded(
            str(path),
            normalized_specs,
            buffers,
            int(max_workers),
        )
    )


def pread_many_regions_as_bytearrays_contiguous_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int]], Sequence[int], str]
    ],
    *,
    max_workers: int,
) -> dict[str, bytearray]:
    """Read region arrays into bytearrays with contiguous task ranges per worker."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [(int(offset), int(length)) for offset, length in regions],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, regions, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_regions_as_bytearrays_contiguous_threaded(
            str(path),
            normalized_specs,
            int(max_workers),
        )
    )


def pread_many_regions_as_bytearrays_windowed_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int]], Sequence[int], str]
    ],
    *,
    max_workers: int,
    window_tasks: int = 8,
) -> dict[str, bytearray]:
    """Read region arrays in small contiguous windows across workers."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    if int(window_tasks) < 1:
        raise ValueError("window_tasks must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [(int(offset), int(length)) for offset, length in regions],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, regions, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_regions_as_bytearrays_windowed_threaded(
            str(path),
            normalized_specs,
            int(max_workers),
            int(window_tasks),
        )
    )


def pread_many_strided_regions_as_bytearrays_region_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int, int, int]], Sequence[int], str]
    ],
    *,
    max_workers: int,
) -> dict[str, bytearray]:
    """Read strided file runs into compact bytearrays with per-run workers."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [
                (int(offset), int(useful_length), int(stride), int(count))
                for offset, useful_length, stride, count in runs
            ],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, runs, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_strided_regions_as_bytearrays_region_threaded(
            str(path),
            normalized_specs,
            int(max_workers),
        )
    )


def pread_many_strided_regions_into_bytearrays_region_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int, int, int]], Sequence[int], str]
    ],
    buffers: dict[str, bytearray],
    *,
    max_workers: int,
) -> dict[str, bytearray]:
    """Read strided file runs into caller-owned bytearrays with per-run workers."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [
                (int(offset), int(useful_length), int(stride), int(count))
                for offset, useful_length, stride, count in runs
            ],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, runs, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_strided_regions_into_bytearrays_region_threaded(
            str(path),
            normalized_specs,
            buffers,
            int(max_workers),
        )
    )


def pread_many_coalesced_regions_as_bytearrays_region_threaded(
    path: str,
    specs: Sequence[
        tuple[str, Sequence[tuple[int, int]], Sequence[int], str]
    ],
    *,
    max_workers: int,
    max_gap_bytes: int,
    max_span_bytes: int,
) -> dict[str, bytearray]:
    """Read sparse file regions as coalesced spans and compact useful bytes."""

    if int(max_workers) < 1:
        raise ValueError("max_workers must be positive")
    if int(max_gap_bytes) < 0:
        raise ValueError("max_gap_bytes must be non-negative")
    if int(max_span_bytes) < 1:
        raise ValueError("max_span_bytes must be positive")
    module = _load_probe_extension()
    normalized_specs = [
        (
            str(name),
            [(int(offset), int(length)) for offset, length in regions],
            [int(dim) for dim in shape],
            str(dtype_name),
        )
        for name, regions, shape, dtype_name in specs
    ]
    return dict(
        module.pread_many_coalesced_regions_as_bytearrays_region_threaded(
            str(path),
            normalized_specs,
            int(max_workers),
            int(max_gap_bytes),
            int(max_span_bytes),
        )
    )
