"""Runtime access layer for SmartTensor manifests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import time

from smarttensor.errors import TensorNotFoundError
from smarttensor.manifest import SmartTensorManifest, TensorRecord
from smarttensor.safetensors import SafeTensorFile, TensorSlice


@dataclass(frozen=True)
class TensorAccessEvent:
    name: str
    nbytes: int
    elapsed_seconds: float
    cache_hit: bool


class TensorResidencyCache:
    """A lightweight LRU budget tracker for mmap-backed tensor slices.

    This tracks runtime intent and releases memoryviews on eviction. Actual disk
    page residency remains under OS control.
    """

    def __init__(self, budget_bytes: int) -> None:
        if budget_bytes < 0:
            raise ValueError("budget_bytes must be non-negative")
        self.budget_bytes = budget_bytes
        self.current_bytes = 0
        self._items: OrderedDict[str, TensorSlice] = OrderedDict()

    def get(self, name: str) -> TensorSlice | None:
        item = self._items.get(name)
        if item is not None:
            self._items.move_to_end(name)
        return item

    def put(self, tensor: TensorSlice) -> list[str]:
        evicted: list[str] = []
        if self.budget_bytes == 0:
            tensor.release()
            return [tensor.name]

        existing = self._items.pop(tensor.name, None)
        if existing is not None:
            self.current_bytes -= existing.nbytes
            existing.release()

        self._items[tensor.name] = tensor
        self.current_bytes += tensor.nbytes
        self._items.move_to_end(tensor.name)

        while self.current_bytes > self.budget_bytes and self._items:
            evicted_name, evicted_tensor = self._items.popitem(last=False)
            self.current_bytes -= evicted_tensor.nbytes
            evicted_tensor.release()
            evicted.append(evicted_name)
        return evicted

    def release_all(self) -> None:
        while self._items:
            _, tensor = self._items.popitem(last=False)
            tensor.release()
        self.current_bytes = 0

    def names(self) -> list[str]:
        return list(self._items)


class SmartTensorRuntime:
    """Open safetensors shards and serve tensors by manifest name."""

    def __init__(
        self,
        paths: list[str | Path],
        *,
        manifest: SmartTensorManifest | None = None,
        cache_budget_bytes: int = 0,
    ) -> None:
        if not paths:
            raise ValueError("at least one path is required")

        self.paths = [Path(path) for path in paths]
        self.manifest = manifest or SmartTensorManifest.from_safetensors(self.paths)
        self.cache = TensorResidencyCache(cache_budget_bytes)
        self.access_log: list[TensorAccessEvent] = []
        self._files: dict[str, SafeTensorFile] = {
            str(path): SafeTensorFile(path) for path in self.paths
        }

    def tensor_record(self, name: str) -> TensorRecord:
        try:
            return self.manifest.tensors[name]
        except KeyError as exc:
            raise TensorNotFoundError(name) from exc

    def get_tensor(self, name: str, *, keep_resident: bool = False) -> TensorSlice:
        began = time.perf_counter()
        cached = self.cache.get(name)
        if cached is not None:
            self.access_log.append(
                TensorAccessEvent(
                    name=name,
                    nbytes=cached.nbytes,
                    elapsed_seconds=time.perf_counter() - began,
                    cache_hit=True,
                )
            )
            return cached

        record = self.tensor_record(name)
        safe_file = self._files[record.file]
        tensor = safe_file.tensor(name)
        if keep_resident:
            self.cache.put(tensor)
            tensor = self.cache.get(name)
            if tensor is None:
                raise RuntimeError(f"tensor {name!r} exceeded residency budget")

        self.access_log.append(
            TensorAccessEvent(
                name=name,
                nbytes=record.nbytes,
                elapsed_seconds=time.perf_counter() - began,
                cache_hit=False,
            )
        )
        return tensor

    def read_tensor_bytes(self, name: str, *, limit: int | None = None) -> bytes:
        with self.get_tensor(name) as tensor:
            if limit is None:
                return tensor.copy()
            return tensor.view[:limit].tobytes()

    def prefetch_layer(self, layer_index: int) -> dict[str, float]:
        layer = self.manifest.layers[layer_index]
        timings: dict[str, float] = {}
        by_file: dict[str, list[str]] = {}
        for name in layer.tensor_names:
            record = self.tensor_record(name)
            by_file.setdefault(record.file, []).append(name)

        for file_name, names in by_file.items():
            timings.update(self._files[file_name].prefetch(names))
        return timings

    def close(self) -> None:
        self.cache.release_all()
        for safe_file in self._files.values():
            safe_file.close()
        self._files.clear()

    def __enter__(self) -> "SmartTensorRuntime":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
