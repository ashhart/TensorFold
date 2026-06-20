"""Contiguous per-expert pack store for basic-slice expert extraction.

Sprint 3 (redirected by Sprint 5 forensics). The win this module books is
*layout*, not zero-copy: a MoE expert weight tensor in safetensors is shaped
``(E, ...)`` with the expert axis leading, so the runtime selects experts with
a numpy fancy-index gather ``npv[[e0, e1, ...]]``. That gather measured
~1.4-2.0x slower than a contiguous basic slice ``npv[start:stop]`` of the same
bytes. A pack lays each expert's rows out as its own contiguous, page-aligned
record, so a single expert (or a contiguous run of experts) extracts as one
basic slice with no strided gather and no row-by-row python loop.

What this module is NOT:

- It is NOT zero-copy. Sprint 5 measured the mmap->mx.array copy already at the
  memcpy ceiling (~67 GB/s); there is no loading-speed headroom. The 16KB page
  alignment here is *cheap forward-compat insurance only* (a future MLX that can
  adopt a foreign buffer). Do NOT book any speedup against the alignment.
- It does NOT change weights. The packed bytes for an expert are byte-identical
  to the source safetensors bytes for that expert (verified in the unit suite).

Pack file format (little-endian, single file per source shard subset)::

    [8 bytes]  uint64 header_length
    [header]   UTF-8 JSON: {"__pack__": {...}, "<tensor name>": {record}, ...}
    [pad]      zero bytes so the data section starts 16KB-aligned
    [data]     per-tensor, per-expert records each padded to 16KB

Each tensor record in the header carries::

    {
      "dtype": "U32",                 # safetensors dtype string
      "shape": [32, 2880, 720],       # original shape, expert axis leading
      "expert_axis": 0,
      "expert_count": 32,
      "expert_nbytes": 8294400,       # useful bytes per expert (unpadded)
      "expert_stride": 8306688,       # padded stride between expert records
      "data_offset": 16384,           # absolute offset of expert 0's record
    }

so expert ``e`` lives at ``data_offset + e * expert_stride`` for exactly
``expert_nbytes`` bytes, page-aligned. ``load_expert_union`` extracts those
records with basic mmap slices and returns the same arrays the safetensors
loader would for the requested ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import mmap
from pathlib import Path
import struct
from typing import Any, Iterable

from smarttensor.errors import InvalidSafeTensorError, TensorNotFoundError
from smarttensor.safetensors import DTYPE_SIZES, SafeTensorFile

HEADER_LENGTH_BYTES = 8
PAGE_SIZE = 16 * 1024  # 16KB forward-compat alignment (insurance only).
PACK_MAGIC = "smarttensor-expertpack-v1"


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def _expert_nbytes(shape: tuple[int, ...], dtype: str, expert_axis: int = 0) -> int:
    """Bytes occupied by a single expert record (one slice along expert_axis)."""

    item_size = DTYPE_SIZES.get(dtype)
    if item_size is None:
        raise InvalidSafeTensorError(f"unsupported dtype {dtype!r}")
    if not isinstance(item_size, int):
        raise InvalidSafeTensorError(f"dtype {dtype!r} has non-byte-aligned item size")
    count = 1
    for axis, dim in enumerate(shape):
        if axis == expert_axis:
            continue
        count *= dim
    return count * item_size


@dataclass(frozen=True)
class PackTensorRecord:
    """Layout for one packed expert tensor."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    expert_axis: int
    expert_count: int
    expert_nbytes: int
    expert_stride: int
    data_offset: int

    def expert_offset(self, expert_id: int) -> int:
        if expert_id < 0 or expert_id >= self.expert_count:
            raise IndexError(
                f"expert {expert_id} out of range for {self.name} "
                f"(expert_count={self.expert_count})"
            )
        return self.data_offset + expert_id * self.expert_stride

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "expert_axis": self.expert_axis,
            "expert_count": self.expert_count,
            "expert_nbytes": self.expert_nbytes,
            "expert_stride": self.expert_stride,
            "data_offset": self.data_offset,
        }


@dataclass
class PackTelemetry:
    """Accounting for the pack read path.

    ``read_bytes`` is what was touched on disk (padded, page-aligned spans);
    ``useful_bytes`` is what callers actually consume. ``waste_ratio`` is the
    fraction of touched bytes that were alignment padding. ``ranges`` records
    how many contiguous [start, stop) byte spans each ``load_expert_union``
    resolved to per call -- the whole point of the contiguous layout is to keep
    this small (ideally 1 per tensor for a contiguous id run).
    """

    read_bytes: int = 0
    useful_bytes: int = 0
    union_calls: int = 0
    ranges_total: int = 0
    ranges_by_call: list[dict[str, Any]] = field(default_factory=list)

    @property
    def waste_ratio(self) -> float:
        if self.read_bytes == 0:
            return 0.0
        return 1.0 - (self.useful_bytes / self.read_bytes)

    def record_call(
        self,
        *,
        layer: int | None,
        expert_ids: tuple[int, ...],
        ranges: int,
        read_bytes: int,
        useful_bytes: int,
    ) -> None:
        self.union_calls += 1
        self.ranges_total += ranges
        self.read_bytes += read_bytes
        self.useful_bytes += useful_bytes
        self.ranges_by_call.append(
            {
                "layer": layer,
                "expert_ids": list(expert_ids),
                "ranges": ranges,
                "read_bytes": read_bytes,
                "useful_bytes": useful_bytes,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_bytes": self.read_bytes,
            "useful_bytes": self.useful_bytes,
            "waste_ratio": self.waste_ratio,
            "union_calls": self.union_calls,
            "ranges_total": self.ranges_total,
            "ranges_per_call": (
                self.ranges_total / self.union_calls if self.union_calls else 0.0
            ),
        }


def _contiguous_runs(ids: list[int]) -> list[tuple[int, int]]:
    """Collapse a sorted unique id list into ascending [start, stop) runs."""

    runs: list[tuple[int, int]] = []
    for value in ids:
        if runs and value == runs[-1][1]:
            runs[-1] = (runs[-1][0], value + 1)
        else:
            runs.append((value, value + 1))
    return runs


class ExpertPackWriter:
    """Write selected expert tensors into a contiguous, page-aligned pack."""

    @staticmethod
    def expert_tensor_names(
        safe_file: SafeTensorFile,
        *,
        expert_marker: str = ".experts.",
        switch_marker: str = ".switch_mlp.",
    ) -> list[str]:
        """Names whose first axis is the expert axis (>= 2D, MoE-shaped)."""

        names: list[str] = []
        for name in safe_file.tensor_names():
            if expert_marker not in name and switch_marker not in name:
                continue
            meta = safe_file.tensors[name]
            if len(meta.shape) >= 2 and meta.shape[0] > 0:
                names.append(name)
        return names

    @classmethod
    def write(
        cls,
        safe_file: SafeTensorFile,
        out_path: str | Path,
        names: Iterable[str],
        *,
        page_size: int = PAGE_SIZE,
        expert_axis: int = 0,
    ) -> dict[str, PackTensorRecord]:
        """Pack ``names`` from ``safe_file`` into ``out_path``.

        Returns the layout records. Each expert record is padded up to
        ``page_size`` so a future foreign-buffer adopter can map it directly;
        the useful payload is byte-identical to the source slice.
        """

        out_path = Path(out_path)
        names = list(dict.fromkeys(names))

        # First pass: compute layout. Data starts page-aligned after the header,
        # but we only know the header size once records are laid out, so we lay
        # out relative offsets first, then shift by the aligned header size.
        records: dict[str, PackTensorRecord] = {}
        relative_cursor = 0
        for name in names:
            if name not in safe_file.tensors:
                raise TensorNotFoundError(name)
            meta = safe_file.tensors[name]
            if expert_axis != 0:
                raise NotImplementedError("only leading expert axis is supported")
            if len(meta.shape) < 2:
                raise InvalidSafeTensorError(
                    f"{name} shape {meta.shape} has no expert axis to pack"
                )
            expert_count = meta.shape[0]
            useful = _expert_nbytes(meta.shape, meta.dtype, expert_axis)
            stride = _align_up(useful, page_size)
            records[name] = PackTensorRecord(
                name=name,
                dtype=meta.dtype,
                shape=meta.shape,
                expert_axis=expert_axis,
                expert_count=expert_count,
                expert_nbytes=useful,
                expert_stride=stride,
                data_offset=relative_cursor,  # shifted below
            )
            relative_cursor += stride * expert_count

        header = {
            "__pack__": {
                "magic": PACK_MAGIC,
                "page_size": page_size,
                "source": str(safe_file.path),
            }
        }
        for name, record in records.items():
            header[name] = record.to_dict()

        # The data section must start page-aligned. Reserve a header big enough
        # and compute the aligned data start, then rewrite absolute offsets.
        raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # Offsets in the header are absolute file offsets; adding the data_start
        # changes the digits and could grow the JSON length. Pad the JSON to a
        # stable width so the data_start does not shift after we patch offsets.
        prelim_data_start = _align_up(
            HEADER_LENGTH_BYTES + len(raw_header) + 64, page_size
        )

        shifted: dict[str, PackTensorRecord] = {}
        for name, record in records.items():
            shifted[name] = PackTensorRecord(
                name=record.name,
                dtype=record.dtype,
                shape=record.shape,
                expert_axis=record.expert_axis,
                expert_count=record.expert_count,
                expert_nbytes=record.expert_nbytes,
                expert_stride=record.expert_stride,
                data_offset=prelim_data_start + record.data_offset,
            )

        final_header = {"__pack__": header["__pack__"]}
        for name, record in shifted.items():
            final_header[name] = record.to_dict()
        raw_final_header = json.dumps(final_header, separators=(",", ":")).encode("utf-8")
        data_start = _align_up(HEADER_LENGTH_BYTES + len(raw_final_header), page_size)
        if data_start != prelim_data_start:
            # Re-anchor on the true data_start (rare; header grew). One fixed
            # point because shifting by a constant cannot change relative gaps.
            delta = data_start - prelim_data_start
            for name in list(shifted):
                r = shifted[name]
                shifted[name] = PackTensorRecord(
                    name=r.name,
                    dtype=r.dtype,
                    shape=r.shape,
                    expert_axis=r.expert_axis,
                    expert_count=r.expert_count,
                    expert_nbytes=r.expert_nbytes,
                    expert_stride=r.expert_stride,
                    data_offset=r.data_offset + delta,
                )
            final_header = {"__pack__": header["__pack__"]}
            for name, record in shifted.items():
                final_header[name] = record.to_dict()
            raw_final_header = json.dumps(
                final_header, separators=(",", ":")
            ).encode("utf-8")

        total_data = sum(
            r.expert_stride * r.expert_count for r in shifted.values()
        )
        total_size = data_start + total_data

        with out_path.open("wb") as handle:
            handle.write(struct.pack("<Q", len(raw_final_header)))
            handle.write(raw_final_header)
            # Zero-pad up to the page-aligned data section.
            pad = data_start - (HEADER_LENGTH_BYTES + len(raw_final_header))
            if pad > 0:
                handle.write(b"\x00" * pad)
            # Stream each expert's bytes into its padded record.
            for name, record in shifted.items():
                useful = record.expert_nbytes
                pad_each = record.expert_stride - useful
                slice_obj = safe_file.tensor(name)
                try:
                    view = slice_obj.view
                    for expert_id in range(record.expert_count):
                        start = expert_id * useful
                        handle.write(view[start : start + useful])
                        if pad_each > 0:
                            handle.write(b"\x00" * pad_each)
                finally:
                    slice_obj.release()
            handle.flush()
            actual = handle.tell()
            if actual != total_size:
                raise InvalidSafeTensorError(
                    f"pack size mismatch: wrote {actual}, expected {total_size}"
                )

        return shifted


class ExpertPackReader:
    """Read expert records back from a pack via basic mmap slices."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = None
        self._mmap: mmap.mmap | None = None
        self.page_size = PAGE_SIZE
        self.source: str | None = None
        self.records: dict[str, PackTensorRecord] = {}
        self.telemetry = PackTelemetry()
        self._open()

    def _open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._file = self.path.open("rb")
        header_len = struct.unpack("<Q", self._file.read(HEADER_LENGTH_BYTES))[0]
        raw = self._file.read(header_len)
        header = json.loads(raw)
        pack = header.get("__pack__")
        if not isinstance(pack, dict) or pack.get("magic") != PACK_MAGIC:
            raise InvalidSafeTensorError(f"{self.path} is not a SmartTensor expert pack")
        self.page_size = int(pack.get("page_size", PAGE_SIZE))
        self.source = pack.get("source")
        for name, value in header.items():
            if name == "__pack__":
                continue
            self.records[name] = PackTensorRecord(
                name=name,
                dtype=value["dtype"],
                shape=tuple(value["shape"]),
                expert_axis=int(value["expert_axis"]),
                expert_count=int(value["expert_count"]),
                expert_nbytes=int(value["expert_nbytes"]),
                expert_stride=int(value["expert_stride"]),
                data_offset=int(value["data_offset"]),
            )
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def tensor_names(self) -> list[str]:
        return sorted(self.records)

    def expert_bytes(self, name: str, expert_id: int) -> bytes:
        """Return the raw useful bytes for one expert (a basic slice)."""

        if self._mmap is None:
            raise InvalidSafeTensorError(f"{self.path} is closed")
        if name not in self.records:
            raise TensorNotFoundError(name)
        record = self.records[name]
        start = record.expert_offset(expert_id)
        return bytes(self._mmap[start : start + record.expert_nbytes])

    def load_expert_union(
        self,
        names: Iterable[str],
        expert_ids: Iterable[int],
        *,
        layer: int | None = None,
        as_arrays: bool = True,
    ) -> dict[str, Any]:
        """Extract a union of experts for ``names`` via contiguous basic slices.

        Mirrors what ``load_first_dim_slices`` returns: per name, the rows for
        the requested expert ids stacked along the leading axis, in ascending
        unique-id order. Because the pack lays experts out contiguously with a
        fixed stride, a run of adjacent ids resolves to a single basic mmap
        slice -- no fancy-index, no python row loop. Telemetry records the
        number of contiguous ranges, plus read vs useful bytes per call.

        ``as_arrays=True`` returns numpy arrays (default; what the runtime
        consumes). ``as_arrays=False`` returns raw ``bytes`` for byte checks.
        """

        if self._mmap is None:
            raise InvalidSafeTensorError(f"{self.path} is closed")
        names = list(dict.fromkeys(names))
        ids = sorted({int(value) for value in expert_ids})
        runs = _contiguous_runs(ids)

        result: dict[str, Any] = {}
        call_read = 0
        call_useful = 0
        call_ranges = 0

        import numpy as np

        for name in names:
            if name not in self.records:
                raise TensorNotFoundError(name)
            record = self.records[name]
            chunks: list[bytes] = []
            for start_id, stop_id in runs:
                if start_id < 0 or stop_id > record.expert_count:
                    raise IndexError(
                        f"expert range [{start_id},{stop_id}) out of bounds for {name}"
                    )
                run_len = stop_id - start_id
                # Adjacent experts share a fixed stride, so the run's records
                # are one contiguous span in the file: a single basic slice.
                span_start = record.expert_offset(start_id)
                span_stop = span_start + run_len * record.expert_stride
                span = self._mmap[span_start:span_stop]
                call_ranges += 1
                call_read += span_stop - span_start
                if record.expert_stride == record.expert_nbytes:
                    # No padding: the whole span is useful, take it directly.
                    chunks.append(bytes(span))
                    call_useful += run_len * record.expert_nbytes
                else:
                    # Strip per-expert padding to recover the useful bytes.
                    for k in range(run_len):
                        off = k * record.expert_stride
                        chunks.append(bytes(span[off : off + record.expert_nbytes]))
                        call_useful += record.expert_nbytes

            payload = b"".join(chunks)
            if as_arrays:
                np_dtype = _numpy_dtype(record.dtype)
                rest_shape = tuple(record.shape[1:])
                arr = np.frombuffer(payload, dtype=np_dtype).reshape(
                    (len(ids),) + rest_shape
                )
                result[name] = arr
            else:
                result[name] = payload

        self.telemetry.record_call(
            layer=layer,
            expert_ids=tuple(ids),
            ranges=call_ranges,
            read_bytes=call_read,
            useful_bytes=call_useful,
        )
        return result

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "ExpertPackReader":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def build_model_packs(model_dir: str | Path, out_dir: str | Path) -> dict[str, str]:
    """Write one expert pack per shard of a model.

    Returns {shard_path: pack_path}. Shards with no expert-axis tensors (e.g.
    an embeddings-only shard) are skipped. Each pack header records its source
    shard, so the reader can be matched back to the shard it serves.
    """

    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors shards in {model_dir}")

    mapping: dict[str, str] = {}
    for shard in shards:
        with SafeTensorFile(shard) as safe_file:
            names = ExpertPackWriter.expert_tensor_names(safe_file)
            if not names:
                continue
            pack_path = out_dir / (shard.stem + ".pack")
            ExpertPackWriter.write(safe_file, pack_path, names)
            mapping[str(shard)] = str(pack_path)
    return mapping


def open_model_packs(out_dir: str | Path) -> dict[str, "ExpertPackReader"]:
    """Open every pack in ``out_dir``, keyed by the source shard it serves."""

    out_dir = Path(out_dir)
    readers: dict[str, ExpertPackReader] = {}
    for pack_path in sorted(out_dir.glob("*.pack")):
        reader = ExpertPackReader(pack_path)
        if reader.source is None:
            reader.close()
            raise InvalidSafeTensorError(f"{pack_path} has no source shard recorded")
        readers[reader.source] = reader
    return readers


def _numpy_dtype(dtype: str):
    import numpy as np

    mapping = {
        "BOOL": np.dtype("bool"),
        "U8": np.dtype("u1"),
        "I8": np.dtype("i1"),
        "U16": np.dtype("<u2"),
        "I16": np.dtype("<i2"),
        "U32": np.dtype("<u4"),
        "I32": np.dtype("<i4"),
        "U64": np.dtype("<u8"),
        "I64": np.dtype("<i8"),
        "F16": np.dtype("<f2"),
        # BF16 has no numpy native dtype; expose raw 16-bit words like the
        # runtime loader, which reinterprets to mx.bfloat16 after copy.
        "BF16": np.dtype("<u2"),
        "F32": np.dtype("<f4"),
        "F64": np.dtype("<f8"),
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported pack dtype: {dtype}") from exc
