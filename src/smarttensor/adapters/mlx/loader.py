"""Selective MLX tensor loading and resident session management."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any

import numpy as np
from safetensors import safe_open

from smarttensor.manifest import SmartTensorManifest, TensorRecord
from smarttensor.planner import pinned_tensors
from smarttensor.safetensors import SafeTensorFile
from smarttensor.weight_pager import PagedWeightCache, WeightPageKey, row_page_specs

from .records import MlxStreamEvent, MlxTensorBatch
from .utils import (
    load_native_mlx_array,
    load_native_mlx_array_first_dim_indices,
    numpy_dtype,
    placeholder_for_record,
)

class MlxSelectiveLoader:
    """Load selected safetensors entries as MLX arrays.

    This is the first backend-facing bridge. It does not ask MLX to load every
    shard. Instead, it groups requested tensor names by source shard, opens each
    shard with `safe_open`, and converts only those tensors into MLX arrays.
    """

    def __init__(self, manifest: SmartTensorManifest, *, backend: str = "native") -> None:
        if backend not in {"native", "safetensors"}:
            raise ValueError("backend must be 'native' or 'safetensors'")
        self.manifest = manifest
        self.backend = backend
        # Shard handles persist for the loader's lifetime: re-opening per load
        # call costs ~8ms of mmap/header churn per expert load (measured 3.2x).
        self._files: dict[str, SafeTensorFile] = {}
        # Optional contiguous expert packs, keyed by the source shard they
        # serve. When present, eligible first-dim slice loads route through the
        # pack (basic-slice extraction, fewer/larger contiguous reads) instead
        # of the safetensors fancy-index gather. Bitwise-identical output.
        self._pack_readers: dict[str, Any] = {}
        self._weight_pages: PagedWeightCache | None = None
        self._weight_page_rows = 1

    def attach_pack_dir(self, pack_dir: str | Path) -> int:
        """Open expert packs in ``pack_dir`` and route eligible loads to them."""

        from smarttensor.packstore import open_model_packs

        self._pack_readers = open_model_packs(pack_dir)
        return len(self._pack_readers)

    def attach_weight_page_cache(
        self,
        budget_bytes: int,
        *,
        eviction_policy: str = "lru",
        rows_per_page: int = 1,
    ) -> None:
        """Enable an opt-in resident cache for first-axis row pages."""

        if rows_per_page < 1:
            raise ValueError("rows_per_page must be positive")
        self._weight_page_rows = rows_per_page
        self._weight_pages = PagedWeightCache(
            budget_bytes,
            eviction_policy=eviction_policy,
        )

    @property
    def weight_page_resident_bytes(self) -> int:
        return self._weight_pages.resident_bytes if self._weight_pages else 0

    def weight_page_summary(self) -> dict[str, Any] | None:
        if self._weight_pages is None:
            return None
        summary = self._weight_pages.to_dict()
        summary["rows_per_page"] = self._weight_page_rows
        return summary

    def clear_weight_page_cache(self) -> int:
        if self._weight_pages is None:
            return 0
        before = self._weight_pages.resident_bytes
        self._weight_pages.clear(force=True)
        return before - self._weight_pages.resident_bytes

    def estimate_first_dim_slice_page_miss_bytes(
        self,
        names: list[str] | tuple[str, ...],
        indices: list[int] | tuple[int, ...],
        *,
        cap_to_headroom: bool = True,
    ) -> int:
        """Estimate cache misses for row-page prefetch.

        By default this reports net resident growth capped by cache headroom.
        DeepSeek prefetch accounting asks for the uncapped value to expose the
        transient page bytes touched while the cache evicts and admits pages.
        """

        if self._weight_pages is None:
            return 0
        names = tuple(dict.fromkeys(names))
        indices = tuple(dict.fromkeys(int(index) for index in indices))
        miss_bytes = 0
        for name in names:
            record = self.manifest.tensors[name]
            for spec in row_page_specs(
                name,
                shape=record.shape,
                tensor_nbytes=record.nbytes,
                indices=indices,
                rows_per_page=self._weight_page_rows,
            ):
                if not self._weight_pages.contains(spec.key):
                    miss_bytes += spec.nbytes
        if not cap_to_headroom:
            return miss_bytes
        headroom = max(self._weight_pages.budget_bytes - self._weight_pages.resident_bytes, 0)
        return min(miss_bytes, headroom)

    def _file(self, file_name: str) -> SafeTensorFile:
        handle = self._files.get(file_name)
        if handle is None:
            handle = SafeTensorFile(file_name)
            self._files[file_name] = handle
        return handle

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        for reader in self._pack_readers.values():
            reader.close()
        self._pack_readers.clear()
        if self._weight_pages is not None:
            self._weight_pages.clear(force=True)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, *, backend: str = "native") -> "MlxSelectiveLoader":
        path = Path(model_dir)
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files found in {path}")
        return cls(SmartTensorManifest.from_safetensors(files), backend=backend)

    def load_pinned(self, *, evaluate: bool = False) -> MlxTensorBatch:
        names = [record.name for record in pinned_tensors(self.manifest)]
        return self.load_tensors(names, evaluate=evaluate)

    def load_layer(self, layer_index: int, *, evaluate: bool = False) -> MlxTensorBatch:
        layer = self.manifest.layers[layer_index]
        return self.load_tensors(layer.tensor_names, evaluate=evaluate)

    def load_tensors(self, names: list[str] | tuple[str, ...], *, evaluate: bool = False) -> MlxTensorBatch:
        names = tuple(dict.fromkeys(names))
        started = time.perf_counter()
        arrays: dict[str, Any] = {}
        by_file: dict[str, list[str]] = {}
        nbytes = 0

        for name in names:
            record = self.manifest.tensors[name]
            by_file.setdefault(record.file, []).append(name)
            nbytes += record.nbytes

        import mlx.core as mx

        if self.backend == "native":
            for file_name, file_names in by_file.items():
                safe_file = self._file(file_name)
                for name in file_names:
                    record = self.manifest.tensors[name]
                    arrays[name] = load_native_mlx_array(safe_file, record)
        else:
            for file_name, file_names in by_file.items():
                with safe_open(file_name, framework="pt", device="cpu") as handle:
                    for name in file_names:
                        arrays[name] = mx.array(handle.get_tensor(name))

        # One optional eval boundary per load call. Callers that immediately
        # feed these arrays into a larger graph can keep them lazy and let the
        # downstream materialization absorb the sync.
        if evaluate and arrays:
            mx.eval(list(arrays.values()))

        return MlxTensorBatch(
            names=tuple(names),
            nbytes=nbytes,
            seconds=time.perf_counter() - started,
            arrays=arrays,
            weight_page_cache_bytes=self.weight_page_resident_bytes,
        )

    def load_first_dim_slices(
        self,
        names: list[str] | tuple[str, ...],
        indices: list[int] | tuple[int, ...],
        *,
        evaluate: bool = False,
        use_weight_page_cache: bool = True,
    ) -> MlxTensorBatch:
        if self.backend != "native":
            raise ValueError("first-dimension slices currently require the native backend")

        names = tuple(dict.fromkeys(names))
        indices = tuple(int(index) for index in indices)
        started = time.perf_counter()
        arrays: dict[str, Any] = {}
        by_file: dict[str, list[str]] = {}
        nbytes = 0
        transient_page_bytes = 0

        for name in names:
            record = self.manifest.tensors[name]
            if not record.shape:
                raise ValueError(f"cannot slice scalar tensor on first dimension: {name}")
            if any(index < 0 or index >= record.shape[0] for index in indices):
                raise IndexError(f"first-dimension slice index out of range for {name}")
            by_file.setdefault(record.file, []).append(name)
            nbytes += record.nbytes * len(indices) // record.shape[0]

        if self._weight_pages is not None and use_weight_page_cache:
            transient_page_bytes = self.estimate_first_dim_slice_page_miss_bytes(
                names,
                indices,
                cap_to_headroom=False,
            )

        import mlx.core as mx

        # Pack route is only valid when the rows the pack returns (ascending
        # unique id order) match what gather_first_dim_rows would return for
        # the caller's index order. Callers pass sorted unique selected experts;
        # guard defensively and fall back to safetensors otherwise.
        pack_eligible = (
            bool(self._pack_readers)
            and list(indices) == sorted(set(indices))
        )

        for file_name, file_names in by_file.items():
            if self._weight_pages is not None and use_weight_page_cache:
                safe_file = self._file(file_name)
                for name in file_names:
                    record = self.manifest.tensors[name]
                    arrays[name] = self._load_first_dim_slices_from_weight_pages(
                        safe_file,
                        record,
                        indices,
                    )
                continue
            reader = self._pack_readers.get(file_name) if pack_eligible else None
            if reader is not None and all(name in reader.records for name in file_names):
                pack_arrays = reader.load_expert_union(file_names, indices, as_arrays=True)
                for name in file_names:
                    record = self.manifest.tensors[name]
                    arr = mx.array(pack_arrays[name])
                    if record.dtype == "BF16":
                        arr = arr.view(mx.bfloat16)
                    arrays[name] = arr
                continue
            safe_file = self._file(file_name)
            for name in file_names:
                record = self.manifest.tensors[name]
                arrays[name] = load_native_mlx_array_first_dim_indices(
                    safe_file,
                    record,
                    indices,
                )

        # One optional eval boundary per load call; selected expert tables can
        # stay lazy until the layer compute materializes them.
        if evaluate and arrays:
            mx.eval(list(arrays.values()))

        return MlxTensorBatch(
            names=tuple(names),
            nbytes=nbytes,
            seconds=time.perf_counter() - started,
            arrays=arrays,
            weight_page_cache_bytes=self.weight_page_resident_bytes,
            transient_page_bytes=transient_page_bytes,
        )

    def warm_first_dim_slices(
        self,
        names: list[str] | tuple[str, ...],
        indices: list[int] | tuple[int, ...],
    ) -> MlxTensorBatch:
        """Warm row pages without assembling final first-dim slice tensors.

        This is the low-memory prefetch primitive for MoE routes: it materializes
        resident row pages in the loader cache, then drops local references
        instead of concatenating and assigning full expert tables.
        """

        if self.backend != "native":
            raise ValueError("first-dimension slice warming requires the native backend")
        if self._weight_pages is None:
            raise ValueError("warm_first_dim_slices requires an attached weight page cache")

        names = tuple(dict.fromkeys(names))
        indices = tuple(dict.fromkeys(int(index) for index in indices))
        started = time.perf_counter()
        by_file: dict[str, list[str]] = {}
        nbytes = 0
        transient_page_bytes = 0

        for name in names:
            record = self.manifest.tensors[name]
            if not record.shape:
                raise ValueError(f"cannot slice scalar tensor on first dimension: {name}")
            if any(index < 0 or index >= record.shape[0] for index in indices):
                raise IndexError(f"first-dimension slice index out of range for {name}")
            by_file.setdefault(record.file, []).append(name)
            nbytes += record.nbytes * len(indices) // record.shape[0]

        transient_page_bytes = self.estimate_first_dim_slice_page_miss_bytes(
            names,
            indices,
            cap_to_headroom=False,
        )

        for file_name, file_names in by_file.items():
            safe_file = self._file(file_name)
            for name in file_names:
                record = self.manifest.tensors[name]
                self._warm_first_dim_slice_pages(
                    safe_file,
                    record,
                    indices,
                )

        return MlxTensorBatch(
            names=tuple(names),
            nbytes=nbytes,
            seconds=time.perf_counter() - started,
            arrays={},
            weight_page_cache_bytes=self.weight_page_resident_bytes,
            transient_page_bytes=transient_page_bytes,
        )

    def _load_first_dim_slices_from_weight_pages(
        self,
        safe_file: SafeTensorFile,
        record: TensorRecord,
        indices: tuple[int, ...],
    ) -> Any:
        import mlx.core as mx

        if self._weight_pages is None:
            raise RuntimeError("weight page cache is not attached")
        if not record.shape:
            raise ValueError(f"cannot slice scalar tensor on first dimension: {record.name}")
        row_count = record.shape[0]
        if row_count <= 0 or record.nbytes % row_count != 0:
            raise ValueError(f"cannot row-page tensor with shape {record.shape}: {record.name}")
        rows = self._load_first_dim_slice_page_rows(safe_file, record, indices)
        if not rows:
            empty = np.empty((0, *record.shape[1:]), dtype=numpy_dtype(record.dtype))
            mlx_array = mx.array(empty)
            if record.dtype == "BF16":
                mlx_array = mlx_array.view(mx.bfloat16)
            return mlx_array
        return rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=0)

    def _load_first_dim_slice_page_rows(
        self,
        safe_file: SafeTensorFile,
        record: TensorRecord,
        indices: tuple[int, ...],
    ) -> tuple[Any, ...]:
        if self._weight_pages is None:
            raise RuntimeError("weight page cache is not attached")
        row_count = record.shape[0]
        specs = row_page_specs(
            record.name,
            shape=record.shape,
            tensor_nbytes=record.nbytes,
            indices=indices,
            rows_per_page=self._weight_page_rows,
        )
        pages: dict[WeightPageKey, Any] = {}
        for spec in specs:
            key = spec.key

            def load_page(page_key: WeightPageKey = key) -> Any:
                return load_native_mlx_array_first_dim_indices(
                    safe_file,
                    record,
                    tuple(range(page_key.row_start, page_key.row_stop)),
                )

            pages[key] = self._weight_pages.get_or_load(
                key,
                nbytes=spec.nbytes,
                loader=load_page,
            )

        chunks: list[Any] = []
        current_key: WeightPageKey | None = None
        current_start = 0
        current_stop = 0

        def flush() -> None:
            nonlocal current_key, current_start, current_stop
            if current_key is None:
                return
            chunks.append(pages[current_key][current_start:current_stop])
            current_key = None

        for index in indices:
            start = (index // self._weight_page_rows) * self._weight_page_rows
            stop = min(start + self._weight_page_rows, row_count)
            key = WeightPageKey(record.name, start, stop)
            local = index - start
            if current_key == key and local == current_stop:
                current_stop = local + 1
            else:
                flush()
                current_key = key
                current_start = local
                current_stop = local + 1
        flush()
        return tuple(chunks)

    def _warm_first_dim_slice_pages(
        self,
        safe_file: SafeTensorFile,
        record: TensorRecord,
        indices: tuple[int, ...],
    ) -> None:
        import mlx.core as mx

        if self._weight_pages is None:
            raise RuntimeError("weight page cache is not attached")
        for spec in row_page_specs(
            record.name,
            shape=record.shape,
            tensor_nbytes=record.nbytes,
            indices=indices,
            rows_per_page=self._weight_page_rows,
        ):
            key = spec.key

            def load_page(page_key: WeightPageKey = key) -> Any:
                return load_native_mlx_array_first_dim_indices(
                    safe_file,
                    record,
                    tuple(range(page_key.row_start, page_key.row_stop)),
                )

            page = self._weight_pages.get_or_load(
                key,
                nbytes=spec.nbytes,
                loader=load_page,
            )
            mx.eval(page)


class MlxStreamingSession:
    """Maintain a deduped resident set of MLX arrays for layer streaming."""

    def __init__(self, loader: MlxSelectiveLoader, *, evaluate: bool = False) -> None:
        self.loader = loader
        self.evaluate = evaluate
        self.resident: dict[str, Any] = {}
        self.pinned: set[str] = set()
        self.events: list[MlxStreamEvent] = []
        self.peak_resident_bytes = 0

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        evaluate: bool = False,
    ) -> "MlxStreamingSession":
        return cls(MlxSelectiveLoader.from_model_dir(model_dir), evaluate=evaluate)

    @property
    def resident_bytes(self) -> int:
        return sum(self.loader.manifest.tensors[name].nbytes for name in self.resident)

    def pin(self) -> MlxStreamEvent:
        names = tuple(record.name for record in pinned_tensors(self.loader.manifest))
        event = self._load(names, action="pin", layer=None)
        self.pinned.update(names)
        return event

    def load_layer(self, layer_index: int) -> MlxStreamEvent:
        layer = self.loader.manifest.layers[layer_index]
        return self._load(layer.tensor_names, action="load-layer", layer=layer_index)

    def evict_layer(self, layer_index: int) -> MlxStreamEvent:
        started = time.perf_counter()
        layer = self.loader.manifest.layers[layer_index]
        evicted: list[str] = []
        for name in layer.tensor_names:
            if name in self.pinned:
                continue
            if name in self.resident:
                del self.resident[name]
                evicted.append(name)
        if evicted:
            gc.collect()

        event = MlxStreamEvent(
            action="evict-layer",
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.resident_bytes,
            requested=layer.tensor_names,
            evicted=tuple(evicted),
        )
        self.events.append(event)
        return event

    def close(self) -> None:
        self.resident.clear()
        self.pinned.clear()
        self.loader.close()
        gc.collect()

    def _load(
        self,
        names: tuple[str, ...],
        *,
        action: str,
        layer: int | None,
    ) -> MlxStreamEvent:
        started = time.perf_counter()
        requested = tuple(dict.fromkeys(names))
        missing = tuple(name for name in requested if name not in self.resident)
        skipped = tuple(name for name in requested if name in self.resident)
        nbytes_loaded = 0

        if missing:
            batch = self.loader.load_tensors(missing, evaluate=self.evaluate)
            self.resident.update(batch.arrays)
            nbytes_loaded = batch.nbytes

        resident_bytes = self.resident_bytes
        self.peak_resident_bytes = max(self.peak_resident_bytes, resident_bytes)
        event = MlxStreamEvent(
            action=action,
            layer=layer,
            seconds=time.perf_counter() - started,
            resident_bytes=resident_bytes,
            requested=requested,
            loaded=missing,
            skipped=skipped,
            nbytes_loaded=nbytes_loaded,
        )
        self.events.append(event)
        return event


class MlxModelSession:
    """Apply SmartTensor streaming events to an instantiated MLX model shell.

    This is still a bridge, not a full inference executor. It creates the MLX
    architecture without loading all safetensors shards, then applies selected
    tensor batches with `strict=False`. Evicted tensors are replaced with lazy
    zero placeholders so the loaded arrays can be released.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        evaluate: bool = False,
        retain_layers: set[int] | None = None,
        resident_budget_bytes: int | None = None,
        backend: str = "native",
        clear_on_evict: bool = True,
        pin_policy: str = "all",
        warm_embeddings: bool = False,
        trace: bool = True,
    ) -> None:
        if pin_policy not in {"all", "phase"}:
            raise ValueError("pin_policy must be 'all' or 'phase'")
        if warm_embeddings and pin_policy != "phase":
            raise ValueError("warm_embeddings only applies to pin_policy='phase'")
        self.model_dir = Path(model_dir)
        self.evaluate = evaluate
        self.trace = trace
        self.loader = MlxSelectiveLoader.from_model_dir(self.model_dir, backend=backend)
        self.config = load_mlx_config(self.model_dir)
        self.model = build_mlx_model_shell(self.config, self.loader.manifest)
        self.resident: dict[str, Any] = {}
        # Incremental manifest-bytes accounting: the summing property was
        # O(len(resident)) and ran inside every hot-loop event construction.
        self._resident_nbytes = 0
        self.pinned: set[str] = set()
        if retain_layers is None and resident_budget_bytes is not None:
            retain_layers = select_retained_layers_for_budget(
                self.loader.manifest,
                resident_budget_bytes,
                pin_policy=pin_policy,
                warm_embeddings=warm_embeddings,
            )
        self.retain_layers = retain_layers or set()
        self.resident_budget_bytes = resident_budget_bytes
        self.clear_on_evict = clear_on_evict
        self.pin_policy = pin_policy
        self.warm_embeddings = warm_embeddings
        self.external_resident_bytes = 0
        self.events: list[MlxStreamEvent] = []
        self.peak_resident_bytes = 0

    @property
    def resident_bytes(self) -> int:
        return self._resident_nbytes + self.external_resident_bytes

    def pin(self) -> MlxStreamEvent:
        names = tuple(record.name for record in pinned_tensors(self.loader.manifest))
        event = self._load_into_model(names, action="pin", layer=None)
        self.pinned.update(names)
        return event

    def pin_small(self) -> MlxStreamEvent:
        names = tuple(
            record.name
            for record in pinned_tensors(self.loader.manifest)
            if record.residency_hint == "pin-small"
        )
        event = self._load_into_model(names, action="pin-small", layer=None)
        self.pinned.update(names)
        return event

    def load_embedding(self) -> MlxStreamEvent:
        return self._load_role("embedding", action="load-embedding")

    def evict_embedding(self) -> MlxStreamEvent:
        return self._evict_role("embedding", action="evict-embedding")

    def load_output(self) -> MlxStreamEvent:
        return self._load_role("output", action="load-output")

    def evict_output(self) -> MlxStreamEvent:
        return self._evict_role("output", action="evict-output")

    def load_layer(self, layer_index: int) -> MlxStreamEvent:
        layer = self.loader.manifest.layers[layer_index]
        return self._load_into_model(layer.tensor_names, action="load-layer", layer=layer_index)

    def evict_layer(self, layer_index: int) -> MlxStreamEvent:
        started = time.perf_counter()
        layer = self.loader.manifest.layers[layer_index]
        if layer_index in self.retain_layers:
            event = MlxStreamEvent(
                action="retain-layer",
                layer=layer_index,
                seconds=time.perf_counter() - started,
                resident_bytes=self.resident_bytes,
                requested=layer.tensor_names,
            )
            if self.trace:
                self.events.append(event)
            return event

        evicted = tuple(name for name in layer.tensor_names if name in self.resident and name not in self.pinned)
        if evicted:
            self._reset_model_tensors(evicted)
            for name in evicted:
                self.resident.pop(name, None)
            self._resident_nbytes -= sum(
                self.loader.manifest.tensors[name].nbytes for name in evicted
            )
            if self.clear_on_evict:
                clear_mlx_memory()

        event = MlxStreamEvent(
            action="evict-layer",
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.resident_bytes,
            requested=layer.tensor_names,
            evicted=evicted,
        )
        if self.trace:
            self.events.append(event)
        return event

    def _load_role(self, role: str, *, action: str) -> MlxStreamEvent:
        names = tuple(
            record.name
            for record in self.loader.manifest.tensors.values()
            if record.role == role
        )
        return self._load_into_model(names, action=action, layer=None)

    def _evict_role(self, role: str, *, action: str) -> MlxStreamEvent:
        names = tuple(
            record.name
            for record in self.loader.manifest.tensors.values()
            if record.role == role
        )
        return self._evict_tensors(names, action=action)

    def _evict_tensors(
        self,
        names: tuple[str, ...],
        *,
        action: str,
        layer: int | None = None,
    ) -> MlxStreamEvent:
        started = time.perf_counter()
        evicted = tuple(name for name in names if name in self.resident and name not in self.pinned)
        if evicted:
            self._reset_model_tensors(evicted)
            for name in evicted:
                self.resident.pop(name, None)
            self._resident_nbytes -= sum(
                self.loader.manifest.tensors[name].nbytes for name in evicted
            )
            if self.clear_on_evict:
                clear_mlx_memory()

        event = MlxStreamEvent(
            action=action,
            layer=layer,
            seconds=time.perf_counter() - started,
            resident_bytes=self.resident_bytes,
            requested=names,
            evicted=evicted,
        )
        if self.trace:
            self.events.append(event)
        return event

    def close(self) -> None:
        self.resident.clear()
        self._resident_nbytes = 0
        self.pinned.clear()
        self.external_resident_bytes = 0
        self.model = None
        self.loader.close()
        clear_mlx_memory()

    def set_external_resident_bytes(self, nbytes: int) -> None:
        self.external_resident_bytes = nbytes
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)

    def clear_external_resident_bytes(self) -> None:
        self.external_resident_bytes = 0

    def _load_into_model(
        self,
        names: tuple[str, ...],
        *,
        action: str,
        layer: int | None,
    ) -> MlxStreamEvent:
        started = time.perf_counter()
        requested = tuple(dict.fromkeys(names))
        missing = tuple(name for name in requested if name not in self.resident)
        skipped = tuple(name for name in requested if name in self.resident)
        nbytes_loaded = 0

        if missing:
            batch = self.loader.load_tensors(missing, evaluate=self.evaluate)
            self._apply_tensor_batch(missing, batch)
            nbytes_loaded = batch.nbytes

        resident_bytes = self.resident_bytes
        self.peak_resident_bytes = max(self.peak_resident_bytes, resident_bytes)
        event = MlxStreamEvent(
            action=action,
            layer=layer,
            seconds=time.perf_counter() - started,
            resident_bytes=resident_bytes,
            requested=requested,
            loaded=missing,
            skipped=skipped,
            nbytes_loaded=nbytes_loaded,
        )
        if self.trace:
            self.events.append(event)
        return event

    def _apply_tensor_batch(
        self,
        names: tuple[str, ...],
        batch: MlxTensorBatch,
    ) -> None:
        arrays = sanitize_weights(self.model, batch.arrays)
        self.model.load_weights(list(arrays.items()), strict=False)
        if self.evaluate:
            import mlx.core as mx

            mx.eval(list(arrays.values()))
        # Some MLX model classes sanitize manifest tensor names into
        # module-local aliases before load_weights(), notably DeepSeek V3 MLA
        # kv_b_proj tables. Residency is a manifest-level contract, so count
        # the requested manifest names even when the loaded array key was
        # rewritten for the model shell.
        inserted = tuple(name for name in names if name not in self.resident)
        self.resident.update({name: batch.arrays.get(name) for name in inserted})
        self._resident_nbytes += sum(
            self.loader.manifest.tensors[name].nbytes for name in inserted
        )

    def _reset_model_tensors(self, names: tuple[str, ...]) -> None:
        placeholders = {
            name: placeholder_for_record(self.loader.manifest.tensors[name])
            for name in names
            if name in self.loader.manifest.tensors
        }
        if placeholders:
            self.model.load_weights(list(placeholders.items()), strict=False)
