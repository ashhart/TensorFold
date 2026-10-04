"""Bounded CPU lookup benchmark for resident and SSD n-gram tables.

Examples:
  python tools/bench_ngram_ssd.py
  python tools/bench_ngram_ssd.py --model-dir /checkpoint --table FULL.NGRAM.PREFIX

The real-checkpoint mode reads the index, relevant headers, one scalar scale, and
selected rows only. It never calls ``prefetch`` or loads a complete table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import struct
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfold.families.qwen4_exp.host_table import open_table, read_header


def _index(model_dir: Path) -> dict[str, str]:
    """Read only the checkpoint's tensor-to-file index."""

    path = model_dir / "model.safetensors.index.json"
    where = json.loads(path.read_text())["weight_map"]
    if not isinstance(where, dict):
        raise ValueError(f"{path}: weight_map must be an object")
    return where


def _shards(where: dict[str, str], table: str) -> list[tuple[str, str]]:
    """Find contiguous n-gram shards by their indexed weight names."""

    pattern = re.compile(re.escape(table) + r"\.(?:shard_(\d+)|shards\.(\d+))\.weight$")
    found = {}
    for name, path in where.items():
        match = pattern.fullmatch(name)
        if match:
            index = int(match.group(1) or match.group(2))
            if index in found:
                raise ValueError(f"{table}: duplicate n-gram shard {index}")
            found[index] = (path, name.removesuffix(".weight"))
    if not found or sorted(found) != list(range(len(found))):
        raise ValueError(f"{table}: missing or noncontiguous n-gram shards")
    return [found[i] for i in range(len(found))]


def _scale(model_dir: Path, where: dict[str, str], table: str, field: str) -> float:
    """Read one table scale from its indexed file, which may differ from a shard file."""

    name = f"{table}.{field}"
    if name not in where:
        return 1.0
    path = model_dir / where[name]
    entry = read_header(path)[name]
    dtype = entry["dtype"]
    sizes = {"F32": 4, "F16": 2, "BF16": 2}
    if dtype not in sizes or int(np.prod(entry["shape"])) != 1:
        raise ValueError(f"{name}: expected one F32, F16 or BF16 scale")
    begin, end = entry["data_offsets"]
    if end - begin != sizes[dtype]:
        raise ValueError(f"{name}: scale byte span disagrees with dtype")
    with path.open("rb") as file:
        n = struct.unpack("<Q", file.read(8))[0]
        file.seek(8 + n + begin)
        raw = file.read(sizes[dtype])
    if len(raw) != sizes[dtype]:
        raise ValueError(f"{name}: truncated scale")
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return float(bits.view(np.float32)[0])
    return float(np.frombuffer(raw, dtype={"F32": "<f4", "F16": "<f2"}[dtype])[0])


def _write(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Write a small safetensors fixture without external packages."""

    header, payload, at = {}, [], 0
    for name, (dtype, value) in tensors.items():
        raw = value.tobytes()
        header[name] = {"dtype": dtype, "shape": list(value.shape), "data_offsets": [at, at + len(raw)]}
        payload.append(raw)
        at += len(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))


def synthetic_checkpoint(model_dir: Path, layout: str, rows: int) -> str:
    """Make bounded n-gram tensors with the published NVFP4/FP8/affine row widths."""

    if rows < 1:
        raise ValueError("synthetic rows must be positive")
    rng = np.random.default_rng(47)
    table = "model.layers.0.ple.ple_embedding.ngram_embedding"
    key = table + ".shard_0"
    if layout == "nvfp4":
        parts = {key + ".weight": ("U8", rng.integers(0, 256, (rows, 80), dtype=np.uint8)),
                 key + ".weight_scale": ("F8_E4M3", rng.integers(0x20, 0x48, (rows, 10), dtype=np.uint8))}
        scale = table + ".weight_scale_2"
    elif layout == "fp8":
        parts = {key + ".weight": ("F8_E4M3", rng.integers(0, 126, (rows, 160), dtype=np.uint8))}
        scale = table + ".weight_scale"
    elif layout == "affine":
        parts = {key + ".weight": ("U32", rng.integers(0, 2**32, (rows, 20), dtype=np.uint32)),
                 key + ".scales": ("BF16", rng.integers(0, 2**16, (rows, 5), dtype=np.uint16)),
                 key + ".biases": ("BF16", rng.integers(0, 2**16, (rows, 5), dtype=np.uint16))}
        scale = table + ".weight_scale"
    elif layout == "bf16":
        parts = {key + ".weight": ("BF16", rng.integers(0, 2**16, (rows, 160), dtype=np.uint16))}
        scale = table + ".weight_scale"
    else:
        raise ValueError(f"unknown layout: {layout}")
    # Exercise the index's component locations, including a block scale in a separate file.
    where = {}
    if layout == "affine":
        # The mapped affine reader requires all three components in one file.
        file = "model-00000.safetensors"
        _write(model_dir / file, parts)
        where.update({name: file for name in parts})
    else:
        for i, (name, part) in enumerate(parts.items()):
            file = f"model-{i:05d}.safetensors"
            _write(model_dir / file, {name: part})
            where[name] = file
    file = "model-scale.safetensors"
    _write(model_dir / file, {scale: ("F32", np.array([0.0371], dtype=np.float32))})
    where[scale] = file
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": where}))
    return table


@contextmanager
def _count_pread():
    """Count bytes returned by explicit pread calls during one gather."""

    original = os.pread
    counts = {"calls": 0, "bytes": 0}

    def counted(fd: int, size: int, offset: int) -> bytes:
        data = original(fd, size, offset)
        counts["calls"] += 1
        counts["bytes"] += len(data)
        return data

    os.pread = counted
    try:
        yield counts
    finally:
        os.pread = original


def _parts(value: object) -> tuple[np.ndarray, ...]:
    """Normalise one or three lookup arrays for byte comparison."""

    return value if isinstance(value, tuple) else (value,)


def _rss_peak_bytes() -> int:
    """Return process peak RSS; Linux reports KiB and macOS reports bytes."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def run(model_dir: Path, table: str, sizes: tuple[int, ...], repeats: int, seed: int = 47) -> dict:
    """Time identical row selections with mapped and SSD readers, checking every output byte."""

    if repeats < 2 or not sizes or min(sizes) < 1:
        raise ValueError("at least two repeats and positive row sample sizes are required")
    where = _index(model_dir)
    shards = _shards(where, table)
    scale = lambda field: _scale(model_dir, where, table, field)
    mapped = open_table(model_dir, shards, scale, tensor_files=where)
    disk = open_table(model_dir, shards, scale, ssd=True, tensor_files=where)
    try:
        if max(sizes) > mapped.rows:
            raise ValueError(f"largest row sample {max(sizes)} exceeds table size {mapped.rows}")
        report = {"table": table, "rows": mapped.rows, "row_width_bytes": mapped.wrow,
                  "table_bytes": mapped.nbytes, "samples": []}
        rng = np.random.default_rng(seed)
        for size in sizes:
            ids = rng.choice(mapped.rows, size=size, replace=False).astype(np.int64)
            results = {}
            reference = None
            for label, reader in (("ssd", disk), ("resident", mapped)):
                timings = []
                for _ in range(repeats):
                    start = time.perf_counter_ns()
                    output = _parts(reader.gather(ids))
                    timings.append((time.perf_counter_ns() - start) / 1e6)
                    if reference is None:
                        reference = tuple((a.dtype, a.shape, a.tobytes()) for a in output)
                    elif tuple((a.dtype, a.shape, a.tobytes()) for a in output) != reference:
                        raise AssertionError(f"{size} rows: SSD and resident lookup bytes differ")
                results[label] = {"first_observed_ms": timings[0],
                                  "repeated_p50_ms": float(np.percentile(timings[1:], 50)),
                                  "repeated_p95_ms": float(np.percentile(timings[1:], 95))}
            # Count logical reads after timing so the Python wrapper cannot skew latencies.
            for label, reader in (("ssd", disk), ("resident", mapped)):
                with _count_pread() as count:
                    reader.gather(ids)
                results[label].update(pread_calls=count["calls"], pread_bytes=count["bytes"])
            report["samples"].append({"row_ids": size, "exact_bytes": True, **results})
        report["peak_rss_bytes"] = _rss_peak_bytes()
        return report
    finally:
        if hasattr(disk, "close"):
            disk.close()


def main() -> None:
    """Run a small synthetic benchmark or sample a local indexed checkpoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="local indexed checkpoint; reads selected rows only")
    parser.add_argument("--table", help="full n-gram table prefix in the checkpoint index")
    parser.add_argument("--synthetic-layout", choices=("nvfp4", "fp8", "affine", "bf16"), default="nvfp4")
    parser.add_argument("--synthetic-rows", type=int, default=32768)
    parser.add_argument("--sizes", type=int, nargs="+", default=(16, 512, 8192))
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if bool(args.model_dir) != bool(args.table):
        parser.error("--model-dir and --table must be passed together")
    if args.model_dir:
        result = run(args.model_dir, args.table, tuple(args.sizes), args.repeats)
    else:
        with tempfile.TemporaryDirectory(prefix="ngram-ssd-bench-") as scratch:
            folder = Path(scratch)
            table = synthetic_checkpoint(folder, args.synthetic_layout, args.synthetic_rows)
            result = run(folder, table, tuple(args.sizes), args.repeats)
        result["synthetic_layout"] = args.synthetic_layout
    result["measurement_notes"] = (
        "SSD is measured before the mapped reader for each sample. First observed is not a disk-cold measure: "
        "OS page cache and device state are uncontrolled. Repeated p50/p95 exclude that first call. Peak RSS "
        "is a process high-water mark, not table residency. pread counts come from one extra untimed lookup "
        "per mode; bytes are logical bytes returned, not "
        "physical SSD traffic. This is CPU row-lookup latency only, not model tokens/s."
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
