"""Minimal safetensors metadata reader with mmap-backed tensor access.

The safetensors format starts with an unsigned little-endian 64-bit header
length, followed by a UTF-8 JSON header, followed by contiguous tensor bytes.
Tensor data offsets in the header are relative to the beginning of the data
section, not the beginning of the file.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import mmap
import os
from pathlib import Path
import struct
import time
from typing import Any, Iterator

from smarttensor.errors import InvalidSafeTensorError, TensorNotFoundError

HEADER_LENGTH_BYTES = 8

# macOS ``pread(2)`` fails with errno 22 (EINVAL) on a single call requesting
# >= 2 GiB. Chunk each request well under that bound; 1 GiB is comfortably safe
# and still a single syscall for every real expert-row run. Module-level so tests
# can shrink it to force the multi-chunk assembly path.
_PREAD_CHUNK_BYTES = 1 << 30  # 1 GiB

DTYPE_SIZES: dict[str, float] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass(frozen=True)
class SafeTensorMetadata:
    """Metadata for one tensor inside a safetensors file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    absolute_offsets: tuple[int, int]
    nbytes: int

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count


@dataclass
class TensorSlice:
    """A live mmap-backed slice for one tensor.

    The slice should be released when no longer needed, either explicitly or by
    using it as a context manager.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    view: memoryview

    def copy(self) -> bytes:
        return self.view.tobytes()

    def release(self) -> None:
        self.view.release()

    def __enter__(self) -> "TensorSlice":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


class SafeTensorFile:
    """Open a safetensors file and expose metadata plus lazy tensor slices."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = None
        self._mmap: mmap.mmap | None = None
        self.header: dict[str, Any] = {}
        self.user_metadata: dict[str, str] = {}
        self.tensors: dict[str, SafeTensorMetadata] = {}
        self.data_start = 0
        self.file_size = 0
        self._open()

    def _open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.file_size = self.path.stat().st_size
        if self.file_size < HEADER_LENGTH_BYTES:
            raise InvalidSafeTensorError(f"{self.path} is too small to be safetensors")

        self._file = self.path.open("rb")
        header_length_bytes = self._file.read(HEADER_LENGTH_BYTES)
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        self.data_start = HEADER_LENGTH_BYTES + header_length

        if header_length == 0:
            raise InvalidSafeTensorError(f"{self.path} has an empty safetensors header")
        if self.data_start > self.file_size:
            raise InvalidSafeTensorError(
                f"{self.path} header extends past end of file: {header_length} bytes"
            )

        raw_header = self._file.read(header_length)
        try:
            header = json.loads(raw_header)
        except json.JSONDecodeError as exc:
            raise InvalidSafeTensorError(f"{self.path} header is not valid JSON") from exc

        if not isinstance(header, dict):
            raise InvalidSafeTensorError(f"{self.path} header must be a JSON object")

        self.header = header
        metadata = header.get("__metadata__", {})
        if metadata is not None and not isinstance(metadata, dict):
            raise InvalidSafeTensorError("__metadata__ must be a JSON object when present")
        self.user_metadata = dict(metadata or {})
        self.tensors = self._parse_tensors(header)
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def _parse_tensors(self, header: dict[str, Any]) -> dict[str, SafeTensorMetadata]:
        tensors: dict[str, SafeTensorMetadata] = {}
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(value, dict):
                raise InvalidSafeTensorError(f"tensor {name!r} metadata must be an object")

            dtype = value.get("dtype")
            shape = value.get("shape")
            data_offsets = value.get("data_offsets")
            if not isinstance(dtype, str):
                raise InvalidSafeTensorError(f"tensor {name!r} is missing dtype")
            if dtype not in DTYPE_SIZES:
                raise InvalidSafeTensorError(f"tensor {name!r} has unsupported dtype {dtype!r}")
            if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
                raise InvalidSafeTensorError(f"tensor {name!r} has invalid shape")
            if len(shape) > 0 and any(dim < 0 for dim in shape):
                raise InvalidSafeTensorError(f"tensor {name!r} has negative shape dimension")
            if (
                not isinstance(data_offsets, list)
                or len(data_offsets) != 2
                or not all(isinstance(offset, int) for offset in data_offsets)
            ):
                raise InvalidSafeTensorError(f"tensor {name!r} has invalid data_offsets")

            start, end = data_offsets
            if start < 0 or end < start:
                raise InvalidSafeTensorError(f"tensor {name!r} has invalid offset range")

            absolute_start = self.data_start + start
            absolute_end = self.data_start + end
            if absolute_end > self.file_size:
                raise InvalidSafeTensorError(f"tensor {name!r} extends past end of file")

            nbytes = end - start
            expected_nbytes = _expected_nbytes(tuple(shape), dtype)
            if expected_nbytes != nbytes:
                raise InvalidSafeTensorError(
                    f"tensor {name!r} byte count mismatch: expected {expected_nbytes}, got {nbytes}"
                )

            tensors[name] = SafeTensorMetadata(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                data_offsets=(start, end),
                absolute_offsets=(absolute_start, absolute_end),
                nbytes=nbytes,
            )
        return tensors

    def tensor_names(self) -> list[str]:
        return sorted(self.tensors)

    def tensor(self, name: str) -> TensorSlice:
        if name not in self.tensors:
            raise TensorNotFoundError(name)
        if self._mmap is None:
            raise InvalidSafeTensorError(f"{self.path} is closed")

        metadata = self.tensors[name]
        start, end = metadata.absolute_offsets
        return TensorSlice(
            name=name,
            dtype=metadata.dtype,
            shape=metadata.shape,
            offset=start,
            nbytes=metadata.nbytes,
            view=memoryview(self._mmap)[start:end],
        )

    def pread_range(self, start: int, end: int) -> bytes:
        """Read ``[start, end)`` absolute file bytes via ``os.pread`` (no mmap).

        Unlike :meth:`tensor`, this does NOT touch the persistent mmap, so it
        leaves no file-backed page-cache residency behind -- the bytes land in a
        transient buffer the caller is free to drop. On macOS ``MADV_DONTNEED``
        is a no-op, so reading through pread is the only way to avoid holding the
        working set twice (mmap file-cache + the owned copy).

        Reads loop to absorb partial reads and chunk each ``pread`` below 2 GiB
        (macOS ``pread`` fails with EINVAL on >= 2 GiB single calls). The result
        is byte-identical to ``bytes(self.tensor(name).view)`` for the matching
        absolute offsets.
        """
        if self._file is None:
            raise InvalidSafeTensorError(f"{self.path} is closed")
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > self.file_size:
            raise InvalidSafeTensorError(
                f"{self.path} pread range [{start}, {end}) out of bounds "
                f"(file_size={self.file_size})"
            )
        total = end - start
        if total == 0:
            return b""
        fileno = self._file.fileno()
        out = bytearray(total)
        view = memoryview(out)
        position = 0
        while position < total:
            want = min(_PREAD_CHUNK_BYTES, total - position)
            chunk = os.pread(fileno, want, start + position)
            if not chunk:
                raise InvalidSafeTensorError(
                    f"{self.path} short pread at offset {start + position}: "
                    f"expected {total - position} more bytes, got EOF"
                )
            view[position : position + len(chunk)] = chunk
            position += len(chunk)
        return bytes(out)

    def prefetch(self, names: list[str] | None = None, page_size: int = 4096) -> dict[str, float]:
        """Touch pages for selected tensors and return touch durations by tensor.

        This is a portable approximation of prefetching. It nudges the OS page
        cache without requiring platform-specific APIs.
        """

        if self._mmap is None:
            raise InvalidSafeTensorError(f"{self.path} is closed")

        selected = names if names is not None else self.tensor_names()
        timings: dict[str, float] = {}
        for name in selected:
            if name not in self.tensors:
                raise TensorNotFoundError(name)

            metadata = self.tensors[name]
            start, end = metadata.absolute_offsets
            began = time.perf_counter()
            checksum = 0
            for offset in range(start, end, page_size):
                checksum ^= self._mmap[offset]
            if end > start:
                checksum ^= self._mmap[end - 1]
            timings[name] = time.perf_counter() - began
            # Keep the loop observable to the interpreter.
            if checksum < 0:
                raise AssertionError("unreachable")
        return timings

    def drop_tensor_cache(self, name: str) -> bool:
        """Ask the OS to evict resident mmap pages for one tensor."""

        if name not in self.tensors:
            raise TensorNotFoundError(name)
        metadata = self.tensors[name]
        return self.drop_cache_range(*metadata.absolute_offsets)

    def drop_cache_range(self, start: int, end: int) -> bool:
        """Best-effort MADV_DONTNEED for a byte range in this mmap."""

        if self._mmap is None:
            raise InvalidSafeTensorError(f"{self.path} is closed")
        if end <= start:
            return True
        page_size = os.sysconf("SC_PAGE_SIZE")
        aligned_start = max((int(start) // page_size) * page_size, 0)
        aligned_end = min(
            ((int(end) + page_size - 1) // page_size) * page_size,
            self.file_size,
        )
        if aligned_end <= aligned_start:
            return True
        try:
            self._mmap.madvise(
                mmap.MADV_DONTNEED,
                aligned_start,
                aligned_end - aligned_start,
            )
            return True
        except (AttributeError, OSError, ValueError):
            return False

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "SafeTensorFile":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[SafeTensorMetadata]:
        for name in self.tensor_names():
            yield self.tensors[name]


def _expected_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    count = 1
    for dim in shape:
        count *= dim
    item_size = DTYPE_SIZES[dtype]
    if not isinstance(item_size, int):
        raise InvalidSafeTensorError(f"dtype {dtype!r} has non-byte-aligned item size")
    return count * item_size
