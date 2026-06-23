"""Selective MLX weight loading from SmartTensor manifests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

import numpy as np
from safetensors import safe_open

from smarttensor.manifest import SmartTensorManifest, TensorRecord
from smarttensor.planner import pinned_tensors
from smarttensor.safetensors import SafeTensorFile
from smarttensor.weight_pager import PagedWeightCache, WeightPageKey, row_page_specs


def resolve_weight_page_policy(
    model_type: str,
    policy: str,
    *,
    has_weight_page_budget: bool,
) -> str:
    """Resolve user-facing row-page cache policy aliases.

    DeepSeek's layer-ordered expert scans are hostile to LRU once the selected
    route is slightly larger than the cache. The frequency policy is the
    measured safe default for that path; other models keep the historical LRU
    default unless the caller opts in.
    """

    if policy == "auto":
        if model_type in {"deepseek_v3", "glm_moe_dsa"} and has_weight_page_budget:
            return "frequency"
        return "lru"
    return policy


@dataclass(frozen=True)
class MlxTensorBatch:
    """A group of tensors loaded into MLX arrays."""

    names: tuple[str, ...]
    nbytes: int
    seconds: float
    arrays: dict[str, Any]
    weight_page_cache_bytes: int = 0
    transient_page_bytes: int = 0

    @property
    def tensor_count(self) -> int:
        return len(self.names)

    def to_summary(self) -> dict[str, Any]:
        return {
            "tensor_count": self.tensor_count,
            "nbytes": self.nbytes,
            "seconds": self.seconds,
            "weight_page_cache_bytes": self.weight_page_cache_bytes,
            "transient_page_bytes": self.transient_page_bytes,
            "names": list(self.names),
        }


@dataclass(frozen=True)
class MlxStreamEvent:
    """One resident-set transition in an MLX streaming session."""

    action: str
    seconds: float
    resident_bytes: int
    requested: tuple[str, ...] = ()
    loaded: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    evicted: tuple[str, ...] = ()
    nbytes_loaded: int = 0
    transient_page_bytes: int = 0
    layer: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "layer": self.layer,
            "seconds": self.seconds,
            "resident_bytes": self.resident_bytes,
            "requested_count": len(self.requested),
            "loaded_count": len(self.loaded),
            "skipped_count": len(self.skipped),
            "evicted_count": len(self.evicted),
            "nbytes_loaded": self.nbytes_loaded,
            "transient_page_bytes": self.transient_page_bytes,
            "requested": list(self.requested),
            "loaded": list(self.loaded),
            "skipped": list(self.skipped),
            "evicted": list(self.evicted),
        }


class PassTraceCollector:
    """Accumulate per-bucket wall time across streamed passes (`--trace-pass`).

    Buckets are attributed by call site. MLX is lazy, so "compute" largely
    materializes inside the sync buckets (router_indices_sync, logits_eval,
    select_token_eval, embedding); the residual `python_other` is the
    orchestration tail the Sprint-1 cuts target.
    """

    def __init__(self) -> None:
        self.bucket_seconds: dict[str, float] = {}
        self.bucket_calls: dict[str, int] = {}
        self.pass_seconds = 0.0
        self.passes = 0

    def add(self, bucket: str, seconds: float) -> None:
        self.bucket_seconds[bucket] = self.bucket_seconds.get(bucket, 0.0) + seconds
        self.bucket_calls[bucket] = self.bucket_calls.get(bucket, 0) + 1

    def add_pass(self, seconds: float) -> None:
        self.pass_seconds += seconds
        self.passes += 1

    def to_dict(self) -> dict[str, Any]:
        attributed = sum(self.bucket_seconds.values())
        return {
            "note": (
                "lazy compute materializes inside sync buckets; python_other is "
                "unattributed orchestration time inside streamed passes"
            ),
            "passes": self.passes,
            "pass_seconds": self.pass_seconds,
            "attributed_seconds": attributed,
            "python_other_seconds": max(self.pass_seconds - attributed, 0.0),
            "buckets": {
                name: {
                    "seconds": self.bucket_seconds[name],
                    "calls": self.bucket_calls[name],
                    "ms_per_pass": (
                        1000.0 * self.bucket_seconds[name] / self.passes if self.passes else 0.0
                    ),
                }
                for name in sorted(
                    self.bucket_seconds, key=lambda key: self.bucket_seconds[key], reverse=True
                )
            },
        }


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
        self.drop_mmap_cache_after_read = False

    def attach_pack_dir(
        self,
        pack_dir: str | Path,
        *,
        pack_read_workers: int = 1,
    ) -> int:
        """Open expert packs in ``pack_dir`` and route eligible loads to them."""

        from smarttensor.packstore import open_model_packs

        for reader in self._pack_readers.values():
            reader.close()
        self._pack_readers.clear()

        access_mode = (
            "pread_bytearray_threaded" if int(pack_read_workers) > 1 else "mmap"
        )
        self._pack_readers = open_model_packs(
            pack_dir,
            access_mode=access_mode,
            max_workers=int(pack_read_workers),
        )
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
                    if self.drop_mmap_cache_after_read:
                        safe_file.drop_tensor_cache(name)
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
                arrays.update(reader.load_expert_union_mlx(file_names, indices))
                continue
            safe_file = self._file(file_name)
            for name in file_names:
                record = self.manifest.tensors[name]
                arrays[name] = load_native_mlx_array_first_dim_indices(
                    safe_file,
                    record,
                    indices,
                )
                if self.drop_mmap_cache_after_read:
                    safe_file.drop_tensor_cache(name)

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

            # Cache-MISS reader. When drop_mmap_cache_after_read is opted in,
            # read the page's rows via os.pread into a TRANSIENT buffer
            # (load_..._pread) instead of slicing the persistent mmap. On macOS
            # MADV_DONTNEED is a no-op, so the mmap reader's faulted file pages
            # are never released (RSS held twice: mmap file-cache + owned copy
            # -> jetsam). The pread reader is byte-identical to the mmap reader
            # but leaves no mmap residency. Other models keep the mmap reader.
            if self.drop_mmap_cache_after_read:

                def load_page(page_key: WeightPageKey = key) -> Any:
                    return load_native_mlx_array_first_dim_indices_pread(
                        safe_file,
                        record,
                        tuple(range(page_key.row_start, page_key.row_stop)),
                    )

            else:

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

        # On the pread path (drop_mmap_cache_after_read set) the cache-MISS reads
        # above never touched the mmap, so there are no resident shard pages to
        # release -- the prior drop_tensor_cache call (commit dd85ac7) is now
        # unnecessary AND, on macOS, was a no-op anyway. We intentionally do NOT
        # call it here: the pread reader has already avoided the residency the
        # drop tried (and failed) to reclaim. The non-pread case keeps the mmap
        # reader and is unchanged below.

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


@dataclass(frozen=True)
class MlxForwardResult:
    """Result from a streaming forward pass."""

    prompt: str
    prompt_tokens: int
    completed_layers: int
    total_layers: int
    seconds: float
    next_token: int | None
    next_text: str | None
    resident_peak_bytes: int
    events: tuple[dict[str, Any], ...]

    @property
    def complete(self) -> bool:
        return self.completed_layers == self.total_layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_tokens": self.prompt_tokens,
            "completed_layers": self.completed_layers,
            "total_layers": self.total_layers,
            "complete": self.complete,
            "seconds": self.seconds,
            "next_token": self.next_token,
            "next_text": self.next_text,
            "resident_peak_bytes": self.resident_peak_bytes,
            "events": list(self.events),
        }


@dataclass(frozen=True)
class MlxGenerationResult:
    """Result from a streaming generation loop."""

    prompt: str
    prompt_tokens: int
    generated_tokens: tuple[int, ...]
    generated_text: str
    seconds: float
    resident_peak_bytes: int
    kv_cache_bytes: int
    events: tuple[dict[str, Any], ...]
    expert_prefetch: dict[str, Any] | None = None
    decode_scheduler: dict[str, Any] | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        generated_count = len(self.generated_tokens)
        tokens_per_second = generated_count / self.seconds if self.seconds > 0 else 0.0
        return {
            "prompt": self.prompt,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": list(self.generated_tokens),
            "generated_text": self.generated_text,
            "generated_token_count": generated_count,
            "seconds": self.seconds,
            "tokens_per_second": tokens_per_second,
            "resident_peak_bytes": self.resident_peak_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "expert_prefetch": self.expert_prefetch,
            "decode_scheduler": self.decode_scheduler,
            "finish_reason": self.finish_reason,
            "events": list(self.events),
        }


@dataclass(frozen=True)
class MlxBatchGenerationResult:
    """Result from a batched streaming generation loop."""

    prompts: tuple[str, ...]
    prompt_tokens: int
    generated_tokens: tuple[tuple[int, ...], ...]
    generated_texts: tuple[str, ...]
    seconds: float
    resident_peak_bytes: int
    kv_cache_bytes: int
    events: tuple[dict[str, Any], ...]
    expert_prefetch: dict[str, Any] | None = None
    decode_scheduler: dict[str, Any] | None = None
    finish_reasons: tuple[str, ...] | None = None
    speculative: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        batch_size = len(self.prompts)
        total_generated = sum(len(tokens) for tokens in self.generated_tokens)
        tokens_per_second = total_generated / self.seconds if self.seconds > 0 else 0.0
        return {
            "prompts": list(self.prompts),
            "batch_size": batch_size,
            "prompt_tokens_per_request": self.prompt_tokens,
            "generated_tokens": [list(tokens) for tokens in self.generated_tokens],
            "generated_texts": list(self.generated_texts),
            "generated_token_count": total_generated,
            "seconds": self.seconds,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_per_request": tokens_per_second / batch_size if batch_size else 0.0,
            "resident_peak_bytes": self.resident_peak_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "expert_prefetch": self.expert_prefetch,
            "decode_scheduler": self.decode_scheduler,
            "finish_reasons": list(self.finish_reasons) if self.finish_reasons else None,
            "speculative": self.speculative,
            "events": list(self.events),
        }


@dataclass
class ExpertPrefetchStats:
    """Aggregate telemetry for predictive expert prefetch."""

    attempted_layers: int = 0
    skipped_no_history: int = 0
    skipped_over_cap: int = 0
    skipped_over_budget: int = 0
    table_prefetch_downgrades: int = 0
    full_hits: int = 0
    predicted_rows: int = 0
    true_rows: int = 0
    hit_rows: int = 0
    wasted_rows: int = 0
    missing_rows: int = 0
    prefetched_bytes: int = 0
    wasted_bytes: int = 0
    fallback_bytes: int = 0
    prefetch_load_seconds: float = 0.0
    join_wait_seconds: float = 0.0
    fallback_load_seconds: float = 0.0
    assemble_seconds: float = 0.0
    max_assemble_temporary_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        coverage = self.hit_rows / self.true_rows if self.true_rows else 0.0
        full_hit_rate = self.full_hits / self.attempted_layers if self.attempted_layers else 0.0
        waste_rate = self.wasted_rows / self.predicted_rows if self.predicted_rows else 0.0
        return {
            "attempted_layers": self.attempted_layers,
            "skipped_no_history": self.skipped_no_history,
            "skipped_over_cap": self.skipped_over_cap,
            "skipped_over_budget": self.skipped_over_budget,
            "table_prefetch_downgrades": self.table_prefetch_downgrades,
            "full_hits": self.full_hits,
            "full_hit_rate": full_hit_rate,
            "predicted_rows": self.predicted_rows,
            "true_rows": self.true_rows,
            "hit_rows": self.hit_rows,
            "wasted_rows": self.wasted_rows,
            "missing_rows": self.missing_rows,
            "coverage": coverage,
            "waste_rate": waste_rate,
            "prefetched_bytes": self.prefetched_bytes,
            "wasted_bytes": self.wasted_bytes,
            "fallback_bytes": self.fallback_bytes,
            "prefetch_load_seconds": self.prefetch_load_seconds,
            "join_wait_seconds": self.join_wait_seconds,
            "fallback_load_seconds": self.fallback_load_seconds,
            "assemble_seconds": self.assemble_seconds,
            "max_assemble_temporary_bytes": self.max_assemble_temporary_bytes,
            "hidden_load_seconds": max(self.prefetch_load_seconds - self.join_wait_seconds, 0.0),
            "exposed_wait_seconds": self.join_wait_seconds + self.fallback_load_seconds + self.assemble_seconds,
        }


@dataclass
class DeepSeekExpertSlotArena:
    """Persistent slot table for one DeepSeek routed layer."""

    layer_index: int
    capacity: int
    arrays: dict[str, Any]
    slot_to_expert: list[int | None]
    expert_to_slot: dict[int, int]
    nbytes: int
    last_used: int = 0


@dataclass
class DeepSeekExpertSlotArenaStats:
    """Aggregate telemetry for slot-indexed selected expert tables."""

    attempts: int = 0
    arena_hits: int = 0
    arena_misses: int = 0
    arena_creates: int = 0
    arena_updates: int = 0
    fallback_direct: int = 0
    evictions: int = 0
    selected_rows: int = 0
    hit_rows: int = 0
    missing_rows: int = 0
    loaded_bytes: int = 0
    update_seconds: float = 0.0
    load_seconds: float = 0.0
    slot_update_ops: int = 0
    compact_calls: int = 0
    compact_partition_seconds: float = 0.0
    compact_assemble_seconds: float = 0.0
    compact_remap_seconds: float = 0.0
    compact_qmm_graph_seconds: float = 0.0
    compact_total_seconds: float = 0.0
    deferred_layer_evals: int = 0
    forced_layer_evals: int = 0

    def to_dict(self, *, resident_bytes: int, arena_count: int, capacity: int) -> dict[str, Any]:
        hit_rate = self.hit_rows / self.selected_rows if self.selected_rows else 0.0
        return {
            "attempts": self.attempts,
            "arena_hits": self.arena_hits,
            "arena_misses": self.arena_misses,
            "arena_creates": self.arena_creates,
            "arena_updates": self.arena_updates,
            "fallback_direct": self.fallback_direct,
            "evictions": self.evictions,
            "selected_rows": self.selected_rows,
            "hit_rows": self.hit_rows,
            "missing_rows": self.missing_rows,
            "hit_rate": hit_rate,
            "loaded_bytes": self.loaded_bytes,
            "load_seconds": self.load_seconds,
            "update_seconds": self.update_seconds,
            "slot_update_ops": self.slot_update_ops,
            "compact_calls": self.compact_calls,
            "compact_partition_seconds": self.compact_partition_seconds,
            "compact_assemble_seconds": self.compact_assemble_seconds,
            "compact_remap_seconds": self.compact_remap_seconds,
            "compact_qmm_graph_seconds": self.compact_qmm_graph_seconds,
            "compact_total_seconds": self.compact_total_seconds,
            "deferred_layer_evals": self.deferred_layer_evals,
            "forced_layer_evals": self.forced_layer_evals,
            "resident_bytes": resident_bytes,
            "arena_count": arena_count,
            "capacity": capacity,
        }


@dataclass
class SpeculativeStats:
    """Telemetry for speculative decoding rounds."""

    rounds: int = 0
    single_steps: int = 0
    verify_passes: int = 0
    refeed_passes: int = 0
    full_accept_rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    draft_seconds: float = 0.0
    snapshot_seconds: float = 0.0
    verify_seconds: float = 0.0
    refeed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        passes = self.single_steps + self.verify_passes + self.refeed_passes
        return {
            "rounds": self.rounds,
            "single_steps": self.single_steps,
            "verify_passes": self.verify_passes,
            "refeed_passes": self.refeed_passes,
            "full_accept_rounds": self.full_accept_rounds,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "emitted_tokens": self.emitted_tokens,
            "acceptance_rate": self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0,
            "streamed_passes": passes,
            "tokens_per_pass": self.emitted_tokens / passes if passes else 0.0,
            "accepted_tokens_per_pass": self.emitted_tokens / passes if passes else 0.0,
            "draft_seconds": self.draft_seconds,
            "snapshot_seconds": self.snapshot_seconds,
            "verify_seconds": self.verify_seconds,
            "refeed_seconds": self.refeed_seconds,
        }


@dataclass
class TreeSpeculativeStats:
    """Telemetry for batched branch verification."""

    rounds: int = 0
    single_steps: int = 0
    verify_passes: int = 0
    refeed_passes: int = 0
    full_accept_rounds: int = 0
    branch_rounds: int = 0
    branches_verified: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    draft_seconds: float = 0.0
    snapshot_seconds: float = 0.0
    verify_seconds: float = 0.0
    refeed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        passes = self.single_steps + self.verify_passes + self.refeed_passes
        return {
            "rounds": self.rounds,
            "single_steps": self.single_steps,
            "verify_passes": self.verify_passes,
            "refeed_passes": self.refeed_passes,
            "full_accept_rounds": self.full_accept_rounds,
            "branch_rounds": self.branch_rounds,
            "branches_verified": self.branches_verified,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "emitted_tokens": self.emitted_tokens,
            "acceptance_rate": self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0,
            "average_branches": self.branches_verified / self.branch_rounds if self.branch_rounds else 0.0,
            "streamed_passes": passes,
            "tokens_per_pass": self.emitted_tokens / passes if passes else 0.0,
            "accepted_tokens_per_pass": self.emitted_tokens / passes if passes else 0.0,
            "draft_seconds": self.draft_seconds,
            "snapshot_seconds": self.snapshot_seconds,
            "verify_seconds": self.verify_seconds,
            "refeed_seconds": self.refeed_seconds,
        }


EXACTNESS_MODES = ("target-verified", "exact-strict")


@dataclass(frozen=True)
class ExactnessModeSettings:
    """Resolved runner settings for a requested exactness mode.

    ``mode`` is the user-visible label; ``sliding_cache`` and ``speculation``
    are the concrete levers the runner is wired with; ``reason`` records why,
    so logs and telemetry can explain the choice without re-deriving it.
    """

    mode: str
    sliding_cache: str
    speculation: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sliding_cache": self.sliding_cache,
            "speculation": self.speculation,
            "reason": self.reason,
        }


def select_exactness_mode(
    model_type: str | None,
    mode: str = "target-verified",
    *,
    sliding_cache: str = "rotating",
) -> ExactnessModeSettings:
    """Map a requested exactness mode to concrete runner settings.

    Decision record (do not relitigate):
    - ``target-verified`` (default): every emitted token came from a
      target-model verification path; rare near-tie differences vs the
      single-token baseline are documented and counted in telemetry. The
      operator's ``--sliding-cache`` choice is honored as given.
    - ``exact-strict`` (the honest bitwise mode):
        * GPT-OSS: speculation ON + ``--sliding-cache temporal`` — proven
          bitwise exact (forensics gate max_abs=0.0). Any other sliding-cache
          request is overridden to ``temporal`` because rotating/kv are not
          bitwise exact under chunked verify.
        * Qwen (qwen3_5_moe) and everything else: speculation OFF — SSM chunk
          exactness is unproven, so bitwise exactness is only guaranteed by
          plain single-token greedy decoding.

    This is pure settings logic so the mode -> settings mapping is testable on
    CPU without touching the GPU or constructing a model.
    """

    if mode not in EXACTNESS_MODES:
        raise ValueError(
            f"unknown exactness mode {mode!r}; expected one of {list(EXACTNESS_MODES)}"
        )

    if mode == "target-verified":
        return ExactnessModeSettings(
            mode=mode,
            sliding_cache=sliding_cache,
            speculation=True,
            reason="target-verified: emitted tokens are target-verified; "
            "near-ties counted in telemetry",
        )

    # exact-strict
    if model_type == "gpt_oss":
        if sliding_cache != "temporal":
            reason = (
                "exact-strict on gpt_oss requires temporal sliding cache "
                f"(overrode requested {sliding_cache!r}); speculation stays on, "
                "bitwise exact (forensics max_abs=0.0)"
            )
        else:
            reason = (
                "exact-strict on gpt_oss: speculation + temporal sliding cache, "
                "bitwise exact (forensics max_abs=0.0)"
            )
        return ExactnessModeSettings(
            mode=mode,
            sliding_cache="temporal",
            speculation=True,
            reason=reason,
        )

    return ExactnessModeSettings(
        mode=mode,
        sliding_cache=sliding_cache,
        speculation=False,
        reason=(
            f"exact-strict on {model_type!r}: speculation off — SSM/chunk "
            "exactness unproven, only single-token greedy is bitwise exact"
        ),
    )


class AdaptiveDraftGate:
    """Escalating gate that pauses speculation on consecutive zero-accept rounds.

    A verify round with zero accepted tokens costs ~2 streamed passes for one
    token — pure loss. A round with >=1 accepted token is at worst break-even
    (j+1 tokens for 2 passes), so it must NOT count against the drafter: the
    earlier <=1 trigger measured rewrite-file 1.15x -> 0.99x by gating spans
    that were paying for themselves.

    Ladder, per request: two consecutive zero-accept rounds trigger a mild
    cooldown; the next trigger a hard cooldown; the third disables speculation
    for the rest of the request. An accepted round resets the streak and walks
    the ladder back one rung, so mixed workloads recover; the disabled rung is
    only reachable through repeated zero-accept evidence.
    """

    TRIGGER_STREAK = 2
    MILD_COOLDOWN = 4
    HARD_COOLDOWN = 16
    DISABLED_LEVEL = 3

    def __init__(self) -> None:
        self.zero_streak = 0
        self.level = 0
        self.cooldown_remaining = 0
        self.triggers = 0
        self.gated_rounds = 0
        self.disabled_rounds = 0
        self.zero_accept_rounds = 0

    @property
    def disabled(self) -> bool:
        return self.level >= self.DISABLED_LEVEL

    def allow_draft(self) -> bool:
        """One round wants to draft: count down cooldowns, refuse when gated."""
        if self.disabled:
            self.gated_rounds += 1
            self.disabled_rounds += 1
            return False
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.gated_rounds += 1
            return False
        return True

    def observe(self, accepted: int) -> None:
        """Record one verify round's accepted-token count."""
        if accepted > 0:
            self.zero_streak = 0
            if self.level > 0 and not self.disabled:
                self.level -= 1
            return
        self.zero_accept_rounds += 1
        self.zero_streak += 1
        if self.zero_streak < self.TRIGGER_STREAK:
            return
        self.zero_streak = 0
        self.level = min(self.level + 1, self.DISABLED_LEVEL)
        self.triggers += 1
        if self.level == 1:
            self.cooldown_remaining = self.MILD_COOLDOWN
        elif self.level == 2:
            self.cooldown_remaining = self.HARD_COOLDOWN

    def telemetry(self) -> dict[str, Any]:
        return {
            "gate_level": self.level,
            "gate_triggers": self.triggers,
            "gate_disabled": self.disabled,
            "gated_rounds": self.gated_rounds,
            "disabled_rounds": self.disabled_rounds,
            "zero_accept_rounds": self.zero_accept_rounds,
            # Each gated round skips a draft that recent evidence says would
            # zero-accept, i.e. a verify+refeed double-pass that emits one
            # token. Single-stepping it instead costs one pass, so each gated
            # round avoids ~1 wasted streamed pass. Conservative lower bound:
            # the real failed round can cost up to 2 passes for one token.
            "estimated_regression_avoided_passes": self.gated_rounds,
        }


def prompt_lookup_draft(
    context: list[int],
    max_draft: int,
    *,
    max_ngram: int = 4,
    min_ngram: int = 2,
) -> list[int]:
    """Propose draft tokens by continuing the most recent earlier n-gram match.

    This is draft-model-free speculation: when the tail of the context already
    appeared earlier (repeated identifiers, echoed code, quoted text), the
    tokens that followed it last time are a strong guess for what comes next.
    """

    if max_draft <= 0:
        return []
    for ngram in range(max_ngram, min_ngram - 1, -1):
        if len(context) <= ngram:
            continue
        pattern = context[-ngram:]
        for start in range(len(context) - ngram - 1, -1, -1):
            if context[start : start + ngram] == pattern:
                continuation = context[start + ngram : start + ngram + max_draft]
                if continuation:
                    return list(continuation)
                break
    return []


def count_accepted_drafts(
    draft_tokens: list[int],
    greedy_tokens: list[int],
    *,
    gaps: list[float] | None = None,
    margin: float = 0.0,
) -> int:
    """Length of the draft prefix the target's greedy choices agree with.

    ``greedy_tokens[i]`` is the target argmax at the position that predicts
    ``draft_tokens[i]``. When ``gaps`` (top-1 minus top-2 logit per position)
    and a ``margin`` are given, near-tie agreements are rejected too: the
    verify pass evaluates tokens in one chunk, whose numerics can flip
    near-ties relative to single-step decoding, and drafters propose biased
    continuations — so ambiguous positions must go through the careful path.
    """

    accepted = 0
    for index, (draft_token, greedy_token) in enumerate(zip(draft_tokens, greedy_tokens)):
        if draft_token != greedy_token:
            break
        if gaps is not None and margin > 0 and gaps[index] < margin:
            break
        accepted += 1
    return accepted


def _copy_cache_value(value: Any) -> Any:
    """Deep-ish copy of one cache attribute for an exact rollback snapshot.

    mlx_lm caches mutate in place: KVCache index-assigns into a preallocated
    key/value buffer, and several caches track ``offset``/``lengths`` ints
    alongside the arrays. ``.state`` only exposes the array payload, so a true
    snapshot must copy every attribute, forcing a real copy of each array
    (MLX index assignment mutates the original buffer otherwise).
    """

    import mlx.core as mx

    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, list):
        return [_copy_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_cache_value(item) for item in value)
    return value


def snapshot_cache_states(cache: list[Any]) -> list[dict[str, Any]]:
    """Capture full cache object state so a speculative pass can be rolled back.

    SSM/linear-attention recurrent state cannot be trimmed after the fact, so
    partial draft acceptance on a hybrid model requires restoring the exact
    pre-verify state. Each cache item's whole ``__dict__`` is copied (arrays
    forced to fresh copies); SSM states are tiny and KV buffers are the only
    sizeable copy.
    """

    import mlx.core as mx

    snapshots: list[dict[str, Any]] = []
    arrays_to_eval: list[Any] = []
    for item in cache:
        copied = {key: _copy_cache_value(value) for key, value in vars(item).items()}
        snapshots.append(copied)
        arrays_to_eval.extend(value for value in copied.values() if isinstance(value, mx.array))
    if arrays_to_eval:
        mx.eval(arrays_to_eval)
    return snapshots


def restore_cache_states(cache: list[Any], snapshots: list[dict[str, Any]]) -> None:
    if len(cache) != len(snapshots):
        raise ValueError("cache/state length mismatch")
    for item, snapshot in zip(cache, snapshots):
        for key, value in snapshot.items():
            setattr(item, key, _copy_cache_value(value))


def _repeat_cache_value(value: Any, batch_size: int) -> Any:
    import mlx.core as mx

    if isinstance(value, mx.array):
        if len(value.shape) > 0 and value.shape[0] == 1 and batch_size > 1:
            return mx.concatenate([value] * batch_size, axis=0)
        return mx.array(value)
    if isinstance(value, list):
        return [_repeat_cache_value(item, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(_repeat_cache_value(item, batch_size) for item in value)
    return value


def restore_repeated_cache_states(
    cache: list[Any],
    snapshots: list[dict[str, Any]],
    batch_size: int,
) -> None:
    if len(cache) != len(snapshots):
        raise ValueError("cache/state length mismatch")
    for item, snapshot in zip(cache, snapshots):
        for key, value in snapshot.items():
            setattr(item, key, _repeat_cache_value(value, batch_size))


def _select_cache_value(value: Any, row: int) -> Any:
    import mlx.core as mx

    if isinstance(value, mx.array):
        if len(value.shape) > 0 and value.shape[0] > row:
            return value[row : row + 1]
        return mx.array(value)
    if isinstance(value, list):
        return [_select_cache_value(item, row) for item in value]
    if isinstance(value, tuple):
        return tuple(_select_cache_value(item, row) for item in value)
    return value


def restore_cache_batch_row(target_cache: list[Any], batch_cache: list[Any], row: int) -> None:
    if len(target_cache) != len(batch_cache):
        raise ValueError("cache length mismatch")
    snapshots: list[dict[str, Any]] = []
    for item in batch_cache:
        snapshots.append({key: _select_cache_value(value, row) for key, value in vars(item).items()})
    restore_cache_states(target_cache, snapshots)


class CacheTransaction:
    """Transactional boundary around mutable generation cache state.

    Speculative paths must never leave durable generation state mutated
    unless the verifier committed it: snapshot up front, run the speculative
    block, then either ``commit()`` (keep the mutated cache) or ``rollback()``
    (restore the pre-transaction state exactly). Hybrid caches make this
    mandatory — recurrent SSM state cannot be trimmed after the fact.
    """

    def __init__(self, cache: list[Any]) -> None:
        self._cache = cache
        self._snapshot: list[dict[str, Any]] | None = snapshot_cache_states(cache)

    @property
    def open(self) -> bool:
        return self._snapshot is not None

    def commit(self) -> None:
        if self._snapshot is None:
            raise RuntimeError("transaction already closed")
        self._snapshot = None

    def rollback(self) -> None:
        if self._snapshot is None:
            raise RuntimeError("transaction already closed")
        restore_cache_states(self._cache, self._snapshot)
        self._snapshot = None


class PromptLookupDrafter:
    """Token-level LZ drafter: continue the longest recent suffix match.

    Maintains an incremental n-gram index over the whole context — system and
    user prompt, tool/file content, and generated text alike. Each proposal
    finds where the current suffix occurred before, prefers the longest match
    (most recent on ties), and proposes the tokens that followed it.
    Proposals are only guesses — exact verification keeps output correct —
    so the matcher is deliberately aggressive. Repeated full rejections shrink
    the proposal budget to cut wasted verify passes; acceptance restores it.
    """

    name = "prompt-lookup"
    MATCH_EXTENSION_CAP = 64

    def __init__(
        self,
        *,
        ngram: int = 3,
        max_candidates: int = 8,
        min_draft: int = 2,
    ) -> None:
        if ngram < 1:
            raise ValueError("ngram must be positive")
        self.ngram = ngram
        self.max_candidates = max_candidates
        self.min_draft = min_draft
        # Tokens mined from earlier requests in the same serve session
        # (prior prompts and prior model outputs). They are prepended to the
        # per-request context so historical spans stay mineable across turns;
        # reset() preserves them so a new request keeps the session index.
        self._session_prefix: list[int] = []
        self.reset()

    def reset(self) -> None:
        self._index: dict[tuple[int, ...], list[int]] = {}
        self._indexed_count = 0
        self._last_indexed_token: int | None = None
        self._draft_scale = 1.0
        self.matches_attempted = 0
        self.matches_found = 0
        self.match_length_total = 0

    def seed_session(self, tokens: list[int]) -> None:
        """Fold tokens from a prior request into the persistent session index.

        Used by the server between requests so prompt-lookup mines the whole
        session — earlier prompts and earlier completions — not just the
        current request's context. Proposals are still guesses verified
        exactly downstream, so growing the corpus only ever adds candidates;
        it cannot change emitted tokens.
        """

        if not tokens:
            return
        self._session_prefix.extend(int(token) for token in tokens)
        # The combined-context view changed underneath any built index; force a
        # rebuild on the next proposal by invalidating the indexed cursor.
        self._index = {}
        self._indexed_count = 0
        self._last_indexed_token = None

    def reset_session(self) -> None:
        """Drop all cross-request history (new session / context cleared)."""

        self._session_prefix = []
        self.reset()

    def _combined(self, context: list[int]) -> list[int]:
        """The token sequence the index and proposals operate over."""

        if not self._session_prefix:
            return context
        return self._session_prefix + context

    def _extend_index(self, context: list[int]) -> None:
        start = self._indexed_count
        if start > len(context) or (
            start and context[start - 1] != self._last_indexed_token
        ):
            # Context changed underneath us (no reset between requests):
            # rebuild the index but keep telemetry and the adaptive scale.
            stats = (self.matches_attempted, self.matches_found, self.match_length_total)
            scale = self._draft_scale
            self.reset()
            self.matches_attempted, self.matches_found, self.match_length_total = stats
            self._draft_scale = scale
            start = 0
        smallest = min(2, self.ngram)
        for position in range(max(start, smallest - 1), len(context)):
            # Index every n-gram size the tree drafter can query: shorter
            # fallback lookups are dead unless their keys exist in the index.
            for size in range(smallest, self.ngram + 1):
                if position - size + 1 < 0:
                    continue
                key = tuple(context[position - size + 1 : position + 1])
                positions = self._index.setdefault(key, [])
                positions.append(position)
                if len(positions) > self.max_candidates * 4:
                    del positions[0]
        self._indexed_count = len(context)
        self._last_indexed_token = context[-1] if context else None

    def _suffix_match_length(self, context: list[int], position: int) -> int:
        length = self.ngram
        earlier = position - self.ngram
        suffix = len(context) - 1 - self.ngram
        while (
            earlier >= 0
            and suffix >= 0
            and length < self.MATCH_EXTENSION_CAP
            and context[earlier] == context[suffix]
        ):
            earlier -= 1
            suffix -= 1
            length += 1
        return length

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        if max_draft <= 0 or not context:
            return []
        combined = self._combined(context)
        if len(combined) <= self.ngram:
            return []
        self._extend_index(combined)
        self.matches_attempted += 1

        key = tuple(combined[-self.ngram :])
        candidates = [
            position
            for position in self._index.get(key, ())
            if position < len(combined) - 1
        ]
        if not candidates:
            return []

        best_position = -1
        best_length = -1
        for position in candidates[-self.max_candidates :]:
            length = self._suffix_match_length(combined, position)
            if length >= best_length:
                best_position = position
                best_length = length

        budget = min(max_draft, max(self.min_draft, round(max_draft * self._draft_scale)))
        continuation = combined[best_position + 1 : best_position + 1 + budget]
        if not continuation:
            return []
        self.matches_found += 1
        self.match_length_total += best_length
        return list(continuation)

    def observe_result(self, proposed: int, accepted: int) -> None:
        if proposed <= 0:
            return
        if accepted >= proposed:
            self._draft_scale = min(1.0, self._draft_scale * 2.0)
        elif accepted == 0:
            self._draft_scale = max(0.25, self._draft_scale * 0.5)

    def telemetry(self) -> dict[str, Any]:
        return {
            "matches_attempted": self.matches_attempted,
            "matches_found": self.matches_found,
            "average_match_length": (
                self.match_length_total / self.matches_found if self.matches_found else 0.0
            ),
        }


class PromptLookupTreeDrafter(PromptLookupDrafter):
    """Return several historical continuations for batched tree verification."""

    name = "prompt-lookup-tree"

    def propose_branches(
        self,
        context: list[int],
        max_draft: int,
        max_branches: int,
    ) -> list[list[int]]:
        if max_draft <= 0 or max_branches <= 0 or len(context) <= 1:
            return []
        combined = self._combined(context)
        self._extend_index(combined)
        self.matches_attempted += 1

        # One verifier lane per distinct historical continuation. Mine the
        # combined session corpus (prior requests + this context) at every
        # n-gram size; longer keys rank first (a more specific match is a
        # stronger guess). A continuation shorter than max_draft is still a
        # valid lane: the old `len == max_draft` filter discarded every match
        # whose source span ran into the end of the corpus, which on
        # copy-edit-class text is most of them — that starved the verifier.
        ranked: list[tuple[int, int, list[int]]] = []
        upper = len(combined) - 1
        for ngram in range(min(self.ngram, len(combined) - 1), 0, -1):
            key = tuple(combined[-ngram:])
            for position in self._index.get(key, ())[-self.max_candidates * 4 :]:
                if position >= upper:
                    continue
                continuation = combined[position + 1 : position + 1 + max_draft]
                if continuation:
                    ranked.append((ngram, position, list(continuation)))

        if not ranked:
            return []
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        branches: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for _, _, continuation in ranked:
            key = tuple(continuation)
            # Two source spans can yield the identical continuation; collapse
            # them so each verifier lane carries a distinct guess (a duplicate
            # lane is pure wasted verify width — see the verify-width probe).
            if key in seen:
                continue
            seen.add(key)
            branches.append(continuation)
            if len(branches) >= max_branches:
                break
        if branches:
            self.matches_found += 1
            self.match_length_total += len(branches[0])
        return branches


class ModelDrafter:
    """Drafter backed by a small fully-resident MLX model."""

    name = "model"

    def __init__(self, model_dir: str | Path, target_tokenizer: Any) -> None:
        from mlx_lm import load

        self.model_dir = Path(model_dir)
        self.model, self.tokenizer = load(self.model_dir)
        self._verify_tokenizer(target_tokenizer)
        self.cache: list[Any] | None = None
        self.fed = 0

    def _verify_tokenizer(self, target_tokenizer: Any) -> None:
        probes = ("Hello, world!", "def main():\n    return 0", "The 42 robots painted.")
        for probe in probes:
            if list(self.tokenizer.encode(probe)) != list(target_tokenizer.encode(probe)):
                raise ValueError(
                    f"draft model tokenizer at {self.model_dir} does not match the target tokenizer"
                )

    def reset(self) -> None:
        self.cache = None
        self.fed = 0

    def _forward(self, token_ids: list[int]) -> Any:
        import mlx.core as mx

        logits = self.model(mx.array([token_ids]), cache=self.cache)
        mx.eval(logits)
        return logits

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        import mlx.core as mx

        if max_draft <= 0:
            return []
        if self.cache is None:
            from mlx_lm.models.cache import make_prompt_cache

            self.cache = make_prompt_cache(self.model)
            self.fed = 0

        delta = context[self.fed :]
        if not delta:
            return []
        logits = self._forward(delta)
        self.fed = len(context)

        snapshot = snapshot_cache_states(self.cache)
        drafted: list[int] = []
        try:
            for _ in range(max_draft):
                token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                drafted.append(token)
                if len(drafted) < max_draft:
                    logits = self._forward([token])
        finally:
            restore_cache_states(self.cache, snapshot)
        return drafted


class ExternalProcessDrafter:
    """Drafter backed by a line-oriented helper process.

    This is the bridge for non-MLX sidecars: Core ML/ANE, Swift, or another
    Python runtime can propose guesses while the target MLX model remains the
    exact verifier. The protocol is JSON-lines request/response:

    - ``{"type": "propose", "context": [...], "max_draft": N}``
      -> ``{"tokens": [...]}``
    - ``{"type": "reset"}``, ``{"type": "observe", ...}``,
      ``{"type": "seed_session", ...}``, ``{"type": "reset_session"}``,
      ``{"type": "telemetry"}``
      -> ``{"ok": true}`` or a telemetry object.
    """

    name = "external"

    def __init__(self, command: str | list[str] | tuple[str, ...]) -> None:
        if isinstance(command, str):
            argv = shlex.split(command)
        else:
            argv = [str(part) for part in command]
        if not argv:
            raise ValueError("external drafter command must not be empty")
        self.command = tuple(argv)
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._requests = 0
        self._tokens_proposed = 0

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("external drafter process is not connected")
        if self._process.poll() is not None:
            raise RuntimeError(
                f"external drafter exited with code {self._process.returncode}"
            )
        self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("external drafter closed stdout")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"external drafter returned invalid JSON: {line!r}") from exc
        if not isinstance(response, dict):
            raise RuntimeError("external drafter response must be a JSON object")
        if "error" in response:
            raise RuntimeError(f"external drafter error: {response['error']}")
        return response

    def reset(self) -> None:
        self._request({"type": "reset"})

    def seed_session(self, tokens: list[int]) -> None:
        if tokens:
            self._request({"type": "seed_session", "tokens": [int(token) for token in tokens]})

    def reset_session(self) -> None:
        self._request({"type": "reset_session"})

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        if max_draft <= 0:
            return []
        response = self._request(
            {
                "type": "propose",
                "context": [int(token) for token in context],
                "max_draft": int(max_draft),
            }
        )
        tokens = response.get("tokens", [])
        if not isinstance(tokens, list):
            raise RuntimeError("external drafter response must include list field 'tokens'")
        proposed = [int(token) for token in tokens[:max_draft]]
        self._requests += 1
        self._tokens_proposed += len(proposed)
        return proposed

    def observe_result(self, proposed: int, accepted: int) -> None:
        self._request(
            {
                "type": "observe",
                "proposed": int(proposed),
                "accepted": int(accepted),
            }
        )

    def telemetry(self) -> dict[str, Any]:
        stats = {
            "requests": self._requests,
            "tokens_proposed": self._tokens_proposed,
            "command": list(self.command),
        }
        try:
            response = self._request({"type": "telemetry"})
        except RuntimeError:
            return stats
        if isinstance(response.get("telemetry"), dict):
            stats.update(response["telemetry"])
        return stats

    def close(self) -> None:
        process = self._process
        if process.poll() is None:
            try:
                self._request({"type": "close"})
            except RuntimeError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


# Honest (non-oracle) prompt-lookup drafter. It lives in the MLX-free
# ``block_verify`` module (it is pure n-gram lookup over the running context),
# but is re-exported here so it sits alongside the other drafters and drops in
# directly as the ``drafter=`` argument of
# :meth:`NemotronHStreamingForwardRunner.generate_greedy_speculative` and
# :meth:`~NemotronHStreamingForwardRunner.generate_greedy_speculative_deferred`
# (its ``propose(context, max_draft)`` matches the drafter interface). Unlike
# ``StaticTokenBlockSource`` it has NO tape and only ever sees what the decoder
# has already emitted, so it gives the HONEST real-drafter acceptance number.
from smarttensor.block_verify import PromptLookupBlockSource  # noqa: E402


def expert_merge_plan(
    predicted: list[int],
    true_ids: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Plan an exact expert table from prefetched rows plus a fallback load.

    Returns ``(ordering, hits, missing)`` where ``ordering`` is the row order of
    the assembled table (all predicted rows first, then missing rows), ``hits``
    are true experts covered by the prediction, and ``missing`` are true experts
    that still need a synchronous load.
    """

    predicted_set = set(predicted)
    hits = [expert for expert in true_ids if expert in predicted_set]
    missing = [expert for expert in true_ids if expert not in predicted_set]
    return list(predicted) + missing, hits, missing


class TemporalSlidingKVCache:
    """Temporal-order sliding cache for exact chunk verification.

    MLX's saturated ``RotatingKVCache`` returns physical ring-buffer order for
    one-token updates and temporal order for chunk updates. GPT-OSS chunk
    verification needs both paths to see the same temporal K/V sequence. This
    cache keeps only the temporal union required by a chunk of length ``N``:
    ``[offset - window + 1, ..., offset + N - 1]``.
    """

    def __init__(self, max_size: int):
        self.keys = None
        self.values = None
        self.offset = 0
        self.start_position = 0
        self.max_size = max_size

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        previous_offset = self.offset
        length = keys.shape[2]
        if self.keys is None:
            combined_keys = keys
            combined_values = values
            combined_start = previous_offset
        else:
            combined_keys = mx.concatenate([self.keys, keys], axis=2)
            combined_values = mx.concatenate([self.values, values], axis=2)
            combined_start = self.start_position

        self.offset = previous_offset + length
        new_start = max(0, previous_offset - self.max_size + 1)
        drop = max(0, new_start - combined_start)
        if drop:
            combined_keys = combined_keys[..., drop:, :]
            combined_values = combined_values[..., drop:, :]

        self.keys = combined_keys
        self.values = combined_values
        self.start_position = new_start
        return self.keys, self.values

    def make_mask(
        self,
        N: int,
        window_size: int | None = None,
        return_array: bool = False,
    ) -> Any:
        if N == 1 and not return_array:
            return None

        import mlx.core as mx

        window = window_size or self.max_size
        start = max(0, self.offset - window + 1)
        end = self.offset + N
        rinds = mx.arange(start, end)
        linds = mx.arange(self.offset, end)[:, None]
        mask = linds >= rinds[None]
        mask = mask & (linds < rinds[None] + window)
        return mask

    def size(self) -> int:
        return self.offset - self.start_position

    def empty(self) -> bool:
        return self.keys is None

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        # Unlike KVCache, this cache's keys/values array IS the exact window
        # (no over-allocated slab indexed by offset): update_and_fetch
        # concatenates new tokens and slices the front, so the invariant
        # keys.shape[2] == offset - start_position must hold. A bare
        # ``offset -= n`` (the KVCache contract) would desync the physical
        # array from offset and corrupt the next update's window math, so the
        # tail must be sliced off the arrays too.
        n = min(self.offset - self.start_position, n)
        if n <= 0:
            return 0
        self.offset -= n
        if self.keys is not None:
            kept = self.keys.shape[2] - n
            if kept <= 0:
                self.keys = None
                self.values = None
                self.start_position = self.offset
            else:
                self.keys = self.keys[..., :kept, :]
                self.values = self.values[..., :kept, :]
        return n

    @property
    def state(self) -> tuple[Any, Any]:
        return self.keys, self.values

    @state.setter
    def state(self, value: tuple[Any, Any]) -> None:
        self.keys, self.values = value
        self.offset = self.start_position + self.keys.shape[2]

    @property
    def meta_state(self) -> tuple[str, str, str]:
        return tuple(map(str, (self.max_size, self.start_position, self.offset)))

    @meta_state.setter
    def meta_state(self, value: tuple[str, str, str]) -> None:
        self.max_size, self.start_position, self.offset = map(int, value)

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class StreamingChatRunner:
    """Shared generation loops over an architecture-specific streamed forward.

    Subclasses provide ``_stream_forward_tokens`` (rows in, hidden out),
    ``_logits_from_hidden``, ``_embed_module``/``_embed_prefix`` for sliced
    embedding loads, and the pin/warm attributes. The loops themselves —
    batched greedy/temperature decoding and speculative decoding with exact
    rollback — are architecture-independent.

    ``_trace`` defaults to the legacy eventful behavior; runners that support
    lean mode override it per instance (lean skips per-layer/per-token event
    construction in the hot loop). ``_pass_trace`` carries the optional
    ``--trace-pass`` bucket collector.
    """

    _trace = True
    _pass_trace: PassTraceCollector | None = None

    def pass_trace(self) -> dict[str, Any] | None:
        return self._pass_trace.to_dict() if self._pass_trace is not None else None

    def _reset_stream_state(self) -> None:
        return None

    def _expert_telemetry(self) -> dict[str, Any] | None:
        return None

    def _make_cache(self) -> list[Any]:
        make_cache = getattr(self.session.model, "make_cache", None)
        if make_cache is not None:
            return make_cache()

        from mlx_lm.models.cache import make_prompt_cache

        core = getattr(self.session.model, "model", self.session.model)
        return make_prompt_cache(core)

    def _pin_for_run(self) -> MlxStreamEvent:
        if self.pin_policy == "phase":
            return self.session.pin_small()
        return self.session.pin()

    def _embed_module(self) -> Any:
        raise NotImplementedError

    def _embedding_slice_names(self) -> tuple[str, ...]:
        cached = getattr(self, "_embedding_names_cache", None)
        if cached is None:
            prefix = self._embed_prefix + "."
            cached = tuple(
                sorted(
                    name
                    for name in self.session.loader.manifest.tensors
                    if name.startswith(prefix)
                )
            )
            self._embedding_names_cache = cached
        return cached

    def _load_embedding_slices(
        self,
        token_rows: list[list[int]],
    ) -> tuple[MlxStreamEvent | None, list[list[int]]]:
        started = time.perf_counter()
        selected_tokens = sorted({int(token_id) for row in token_rows for token_id in row})
        names = self._embedding_slice_names()
        batch = self.session.loader.load_first_dim_slices(
            names,
            selected_tokens,
            evaluate=self.session.evaluate,
        )
        module = self._embed_module()
        for name in names:
            setattr(module, name.rsplit(".", 1)[1], batch.arrays[name])
        self._set_external_resident_bytes(batch.nbytes)

        local_by_token = {token: index for index, token in enumerate(selected_tokens)}
        local_token_rows = [
            [local_by_token[int(token_id)] for token_id in row] for row in token_rows
        ]
        if self._pass_trace is not None:
            self._pass_trace.add("embedding_slice_load", time.perf_counter() - started)
        if not self._trace:
            return None, local_token_rows
        event = MlxStreamEvent(
            action="load-embedding-slices",
            layer=None,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names,
            nbytes_loaded=batch.nbytes,
        )
        self.session.events.append(event)
        return event, local_token_rows

    def _clear_embedding_slices(self) -> None:
        import mlx.core as mx

        module = self._embed_module()
        manifest = self.session.loader.manifest
        for name in self._embedding_slice_names():
            record = manifest.tensors[name]
            empty_shape = (0,) * len(record.shape)
            setattr(
                module,
                name.rsplit(".", 1)[1],
                mx.zeros(empty_shape, dtype=mlx_dtype(record.dtype)),
            )
        self._set_external_resident_bytes()

    def _resident_sidecar_bytes(self) -> int:
        return getattr(self, "_expert_cache_bytes", 0)

    def _set_external_resident_bytes(self, temporary_bytes: int = 0) -> None:
        total = self._resident_sidecar_bytes() + temporary_bytes
        if total:
            self.session.set_external_resident_bytes(total)
        else:
            self.session.clear_external_resident_bytes()

    def _async_lookahead_support(self) -> tuple[bool, str]:
        return False, "runner does not expose an async-lookahead-safe forward"

    def _select_decode_scheduler(
        self,
        *,
        prompt_count: int,
        temperature: float,
        max_tokens: int,
        stop_tokens_present: bool = False,
        on_step: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        requested = getattr(self, "decode_scheduler", "auto")
        if requested not in {"auto", "serial", "async-lookahead"}:
            raise ValueError(
                "decode_scheduler must be 'auto', 'serial', or 'async-lookahead'"
            )

        telemetry: dict[str, Any] = {
            "requested": requested,
            "active": "serial",
            "reason": "",
            "reason_code": "",
            "prompt_count": prompt_count,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop_tokens_present": stop_tokens_present,
            "on_step_present": on_step is not None,
            "async_submissions": 0,
            "async_drains": 0,
            "async_sync_seconds": 0.0,
            "unused_lookahead_submissions": 0,
            "stopped_after_lookahead": False,
        }
        if requested == "serial":
            telemetry["reason"] = "serial requested"
            telemetry["reason_code"] = "serial_requested"
            return False, telemetry
        if max_tokens <= 0:
            telemetry["reason"] = "no decode tokens requested"
            telemetry["reason_code"] = "no_decode"
            return False, telemetry
        if prompt_count != 1:
            telemetry["reason"] = "async-lookahead currently supports one request"
            telemetry["reason_code"] = "batch_size"
            if requested == "async-lookahead":
                raise ValueError(telemetry["reason"])
            return False, telemetry
        if temperature != 0:
            telemetry["reason"] = "async-lookahead currently supports greedy decoding only"
            telemetry["reason_code"] = "temperature"
            if requested == "async-lookahead":
                raise ValueError(telemetry["reason"])
            return False, telemetry
        if on_step is not None and not getattr(on_step, "smarttensor_async_safe", False):
            telemetry["reason"] = "async-lookahead requires an async-safe on_step callback"
            telemetry["reason_code"] = "on_step"
            if requested == "async-lookahead":
                raise ValueError(telemetry["reason"])
            return False, telemetry
        supported, reason = self._async_lookahead_support()
        if not supported:
            telemetry["reason"] = reason
            telemetry["reason_code"] = "runner_support"
            if requested == "async-lookahead":
                raise ValueError(f"async-lookahead decode is unsafe: {reason}")
            return False, telemetry

        telemetry["active"] = "async-lookahead"
        telemetry["reason"] = reason
        telemetry["reason_code"] = "active"
        return True, telemetry

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        stop_tokens: set[int] | None = None,
    ) -> MlxGenerationResult:
        result = self.generate_batch(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=stop_tokens,
        )
        return MlxGenerationResult(
            prompt=prompt,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens[0],
            generated_text=result.generated_texts[0],
            seconds=result.seconds,
            resident_peak_bytes=result.resident_peak_bytes,
            kv_cache_bytes=result.kv_cache_bytes,
            events=result.events,
            expert_prefetch=result.expert_prefetch,
            decode_scheduler=result.decode_scheduler,
            finish_reason=result.finish_reasons[0] if result.finish_reasons else None,
        )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        stop_tokens: set[int] | None = None,
        on_step: Any | None = None,
        cache: Any | None = None,
        cached_tokens: int = 0,
        prompt_token_rows: list[list[int]] | None = None,
    ) -> MlxBatchGenerationResult:
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if not prompts:
            raise ValueError("at least one prompt is required")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        import mlx.core as mx

        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        self._reset_stream_state()

        pin_event = self._pin_for_run()
        if self._trace:
            events.append({"kind": "load", **compact_event(pin_event)})

        if prompt_token_rows is not None:
            if len(prompt_token_rows) != len(prompts):
                raise ValueError("prompt_token_rows must match prompts")
            token_rows = [list(row) for row in prompt_token_rows]
        else:
            token_rows = [self.tokenizer.encode(prompt) for prompt in prompts]
        if any(not row for row in token_rows):
            raise ValueError("every prompt must produce at least one token")
        row_lengths = {len(row) for row in token_rows}
        if len(row_lengths) != 1:
            raise ValueError(
                "batched generation requires prompts that tokenize to equal lengths; "
                f"got lengths {sorted(len(row) for row in token_rows)}"
            )
        prompt_length = row_lengths.pop()
        if cached_tokens < 0 or cached_tokens >= prompt_length:
            raise ValueError("cached_tokens must leave at least one prompt token to process")
        if cached_tokens and cache is None:
            raise ValueError("cached_tokens requires the matching cache")

        embeddings_warmed = False
        if self.pin_policy == "phase" and self.warm_embeddings:
            load_embedding = self.session.load_embedding()
            if self._trace:
                events.append({"kind": "load", "pass": "warm", **compact_event(load_embedding)})
            embeddings_warmed = True

        output_warmed = False
        if self.warm_output:
            load_output = self.session.load_output()
            if self._trace:
                events.append({"kind": "load", "pass": "warm-output", **compact_event(load_output)})
            output_warmed = True

        if cache is None:
            cache = self._make_cache()
        try:
            fresh_rows = [row[cached_tokens:] for row in token_rows]
            if len(fresh_rows[0]) > 1:
                self._prefill_cache_tokens(
                    [row[:-1] for row in fresh_rows],
                    cache=cache,
                    events=events,
                    manage_embedding=not embeddings_warmed,
                )

            hidden = self._stream_forward_tokens(
                [[row[-1]] for row in fresh_rows],
                cache=cache,
                events=events,
                pass_kind="prefill",
                manage_embedding=not embeddings_warmed,
            )
            logits = self._logits_from_hidden(hidden, events)

            generated: list[list[int]] = [[] for _ in prompts]
            finished = [False] * len(prompts)
            finish_reasons = ["length"] * len(prompts)

            overlap, decode_scheduler = self._select_decode_scheduler(
                prompt_count=len(prompts),
                temperature=temperature,
                max_tokens=max_tokens,
                stop_tokens_present=bool(stop_tokens),
                on_step=on_step,
            )
            if overlap:
                cur = mx.argmax(logits[:, -1, :], axis=-1)
                mx.async_eval(cur)
                decode_scheduler["async_submissions"] += 1
                nxt = None
                for step in range(max_tokens):
                    if step < max_tokens - 1:
                        hidden = self._stream_forward_tokens(
                            [[0]],
                            cache=cache,
                            events=events,
                            pass_kind="decode",
                            token_step=step,
                            manage_embedding=False,
                            input_array=cur.reshape(1, 1),
                        )
                        nlogits = self._logits_from_hidden(hidden, events)
                        nxt = mx.argmax(nlogits[:, -1, :], axis=-1)
                        mx.async_eval(nxt)
                        decode_scheduler["async_submissions"] += 1
                    token = int(cur.item())
                    generated[0].append(token)
                    stopped = bool(stop_tokens and token in stop_tokens)
                    if stopped:
                        finished[0] = True
                        finish_reasons[0] = "stop"
                    if on_step is not None:
                        on_step([token], list(finished))
                    if stopped or step == max_tokens - 1:
                        if stopped and nxt is not None:
                            sync_started = time.perf_counter()
                            mx.eval(nxt)
                            decode_scheduler["async_drains"] += 1
                            decode_scheduler["async_sync_seconds"] += (
                                time.perf_counter() - sync_started
                            )
                            decode_scheduler["unused_lookahead_submissions"] += 1
                            decode_scheduler["stopped_after_lookahead"] = True
                        break
                    cur = nxt
            else:
                for step in range(max_tokens):
                    select_started = time.perf_counter()
                    if temperature > 0:
                        token_array = mx.random.categorical(logits[:, -1, :] * (1 / temperature))
                    else:
                        token_array = mx.argmax(logits[:, -1, :], axis=-1)
                    mx.eval(token_array)
                    if self._pass_trace is not None:
                        self._pass_trace.add("select_token_eval", time.perf_counter() - select_started)
                    if self._trace:
                        events.append(
                            {
                                "kind": "compute",
                                "action": "select-token",
                                "token_step": step,
                                "seconds": time.perf_counter() - select_started,
                            }
                        )
                    step_tokens = [int(token) for token in token_array.tolist()]
                    for request_index, token in enumerate(step_tokens):
                        if finished[request_index]:
                            continue
                        generated[request_index].append(token)
                        if stop_tokens and token in stop_tokens:
                            finished[request_index] = True
                            finish_reasons[request_index] = "stop"
                    if on_step is not None:
                        on_step(step_tokens, list(finished))
                    if all(finished):
                        break

                    if step < max_tokens - 1:
                        hidden = self._stream_forward_tokens(
                            [[token] for token in step_tokens],
                            cache=cache,
                            events=events,
                            pass_kind="decode",
                            token_step=step,
                            manage_embedding=not embeddings_warmed,
                        )
                        logits = self._logits_from_hidden(hidden, events)
        finally:
            if embeddings_warmed:
                evict_embedding = self.session.evict_embedding()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm", **compact_event(evict_embedding)})
            if output_warmed:
                evict_output = self.session.evict_output()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm-output", **compact_event(evict_output)})

        return MlxBatchGenerationResult(
            prompts=tuple(prompts),
            prompt_tokens=prompt_length,
            generated_tokens=tuple(tuple(tokens) for tokens in generated),
            generated_texts=tuple(self.tokenizer.decode(tokens) for tokens in generated),
            seconds=time.perf_counter() - started,
            resident_peak_bytes=self.session.peak_resident_bytes,
            kv_cache_bytes=kv_cache_nbytes(cache),
            events=tuple(events),
            expert_prefetch=self._expert_telemetry(),
            decode_scheduler=decode_scheduler,
            finish_reasons=tuple(finish_reasons),
        )

    def generate_speculative(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        stop_tokens: set[int] | None = None,
        drafter: Any | None = None,
        max_draft: int = 8,
        draft_margin: float = 0.5,
        on_step: Any | None = None,
        cache: Any | None = None,
        cached_tokens: int = 0,
        prompt_token_rows: list[list[int]] | None = None,
    ) -> MlxBatchGenerationResult:
        """Greedy speculative decoding: one streamed pass verifies many tokens.

        A drafter proposes up to ``max_draft`` tokens; the target verifies the
        whole block in a single streamed forward pass. Accepted tokens cost one
        pass instead of one pass each. On partial acceptance the cache is
        rolled back (recurrent states cannot be trimmed) and the accepted
        prefix is re-fed, so a partial round costs two passes for j+1 tokens.
        ``draft_margin`` rejects near-tie agreements (top-1/top-2 logit gap
        below the margin) so chunked-pass numerics cannot steer ambiguous
        positions toward the drafter's bias.
        """

        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if len(prompts) != 1:
            raise ValueError("speculative decoding currently supports a single request")
        if max_draft < 1:
            raise ValueError("max_draft must be positive")

        import mlx.core as mx

        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        stats = SpeculativeStats()
        self._reset_stream_state()

        drafter = drafter or PromptLookupDrafter()
        drafter.reset()
        stops = stop_tokens or set()

        pin_event = self._pin_for_run()
        if self._trace:
            events.append({"kind": "load", **compact_event(pin_event)})

        if prompt_token_rows is not None:
            if len(prompt_token_rows) != 1:
                raise ValueError("prompt_token_rows must match prompts")
            prompt_ids = list(prompt_token_rows[0])
        else:
            prompt_ids = list(self.tokenizer.encode(prompts[0]))
        if not prompt_ids:
            raise ValueError("the prompt must produce at least one token")
        if cached_tokens < 0 or cached_tokens >= len(prompt_ids):
            raise ValueError("cached_tokens must leave at least one prompt token to process")
        if cached_tokens and cache is None:
            raise ValueError("cached_tokens requires the matching cache")

        embeddings_warmed = False
        if self.pin_policy == "phase" and self.warm_embeddings:
            load_embedding = self.session.load_embedding()
            if self._trace:
                events.append({"kind": "load", "pass": "warm", **compact_event(load_embedding)})
            embeddings_warmed = True

        output_warmed = False
        if self.warm_output:
            load_output = self.session.load_output()
            if self._trace:
                events.append({"kind": "load", "pass": "warm-output", **compact_event(load_output)})
            output_warmed = True

        if cache is None:
            cache = self._make_cache()

        emitted: list[int] = []
        finish_reason = "length"
        try:
            fresh = prompt_ids[cached_tokens:]
            if len(fresh) > 1:
                self._prefill_cache_tokens(
                    [fresh[:-1]],
                    cache=cache,
                    events=events,
                    manage_embedding=not embeddings_warmed,
                )
            hidden = self._stream_forward_tokens(
                [[fresh[-1]]],
                cache=cache,
                events=events,
                pass_kind="prefill",
                manage_embedding=not embeddings_warmed,
            )
            logits = self._logits_from_hidden(hidden, events)

            context = list(prompt_ids)

            def emit(token: int) -> bool:
                """Record one committed token; returns True when generation must stop."""
                nonlocal finish_reason
                emitted.append(token)
                context.append(token)
                stopped = token in stops
                if stopped:
                    finish_reason = "stop"
                if on_step is not None:
                    on_step([token], [stopped or len(emitted) >= max_tokens])
                return stopped or len(emitted) >= max_tokens

            while len(emitted) < max_tokens:
                stats.rounds += 1
                token_array = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(token_array)
                next_token = int(token_array.item())
                if emit(next_token):
                    break

                draft_started = time.perf_counter()
                budget = min(max_draft, max_tokens - len(emitted))
                drafts = drafter.propose(context, budget)
                stats.draft_seconds += time.perf_counter() - draft_started

                if not drafts:
                    step_started = time.perf_counter()
                    hidden = self._stream_forward_tokens(
                        [[next_token]],
                        cache=cache,
                        events=events,
                        pass_kind="decode",
                        token_step=len(emitted),
                        manage_embedding=not embeddings_warmed,
                    )
                    logits = self._logits_from_hidden(hidden, events)
                    stats.single_steps += 1
                    stats.verify_seconds += time.perf_counter() - step_started
                    continue

                stats.drafted_tokens += len(drafts)
                snapshot_started = time.perf_counter()
                transaction = CacheTransaction(cache)
                stats.snapshot_seconds += time.perf_counter() - snapshot_started

                verify_started = time.perf_counter()
                hidden = self._stream_forward_tokens(
                    [[next_token, *drafts]],
                    cache=cache,
                    events=events,
                    pass_kind="verify",
                    token_step=len(emitted),
                    manage_embedding=not embeddings_warmed,
                )
                verify_logits = self._logits_from_hidden(hidden, events)
                greedy_array = mx.argmax(verify_logits[0], axis=-1)
                top2 = mx.topk(verify_logits[0].astype(mx.float32), 2, axis=-1)
                gap_array = mx.abs(top2[..., 0] - top2[..., 1])
                mx.eval(greedy_array, gap_array)
                greedy_tokens = [int(token) for token in greedy_array.tolist()]
                gaps = [float(gap) for gap in gap_array.tolist()]
                stats.verify_passes += 1
                stats.verify_seconds += time.perf_counter() - verify_started

                accepted = count_accepted_drafts(
                    drafts,
                    greedy_tokens[: len(drafts)],
                    gaps=gaps[: len(drafts)],
                    margin=draft_margin,
                )
                stats.accepted_tokens += accepted
                if hasattr(drafter, "observe_result"):
                    drafter.observe_result(len(drafts), accepted)

                stopped = False
                for token in drafts[:accepted]:
                    if emit(token):
                        stopped = True
                        break

                if stopped:
                    transaction.commit()
                    break
                if accepted == len(drafts):
                    stats.full_accept_rounds += 1
                    transaction.commit()
                    logits = verify_logits
                    continue

                refeed_started = time.perf_counter()
                transaction.rollback()
                hidden = self._stream_forward_tokens(
                    [[next_token, *drafts[:accepted]]],
                    cache=cache,
                    events=events,
                    pass_kind="refeed",
                    token_step=len(emitted),
                    manage_embedding=not embeddings_warmed,
                )
                logits = self._logits_from_hidden(hidden, events)
                stats.refeed_passes += 1
                stats.refeed_seconds += time.perf_counter() - refeed_started
        finally:
            if embeddings_warmed:
                evict_embedding = self.session.evict_embedding()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm", **compact_event(evict_embedding)})
            if output_warmed:
                evict_output = self.session.evict_output()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm-output", **compact_event(evict_output)})

        stats.emitted_tokens = len(emitted)
        speculative = stats.to_dict()
        speculative["draft_policy"] = getattr(drafter, "name", type(drafter).__name__)
        speculative["drafter"] = getattr(drafter, "name", type(drafter).__name__)
        # Linear (non-tree) speculation has no adaptive gate and no near-tie
        # guard; report the fields as zero so API consumers see a uniform
        # telemetry schema across both speculative paths.
        speculative.setdefault("gated_rounds", 0)
        speculative.setdefault("disabled_rounds", 0)
        speculative.setdefault("zero_accept_rounds", 0)
        speculative.setdefault("near_tie_events", 0)
        speculative.setdefault("estimated_regression_avoided_passes", 0)
        if hasattr(drafter, "telemetry"):
            speculative["drafter_stats"] = drafter.telemetry()
        return MlxBatchGenerationResult(
            prompts=tuple(prompts),
            prompt_tokens=len(prompt_ids),
            generated_tokens=(tuple(emitted),),
            generated_texts=(self.tokenizer.decode(emitted),),
            seconds=time.perf_counter() - started,
            resident_peak_bytes=self.session.peak_resident_bytes,
            kv_cache_bytes=kv_cache_nbytes(cache),
            events=tuple(events),
            expert_prefetch=self._expert_telemetry(),
            finish_reasons=(finish_reason,),
            speculative=speculative,
        )

    def generate_tree_speculative(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        stop_tokens: set[int] | None = None,
        drafter: Any | None = None,
        max_draft: int = 8,
        max_branches: int = 16,
        draft_margin: float = 0.5,
        cache: Any | None = None,
        cached_tokens: int = 0,
        prompt_token_rows: list[list[int]] | None = None,
    ) -> MlxBatchGenerationResult:
        """Experimental tree speculation: verify many guessed futures at once.

        The prefix cache is cloned across ``max_branches`` rows, each row
        verifies a different continuation, and the winning branch's cache row is
        committed only when the whole branch is accepted. Partial accepts refeed
        the accepted prefix through the normal single-row cache path.
        """

        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if len(prompts) != 1:
            raise ValueError("tree speculation currently supports a single request")
        if max_draft < 1:
            raise ValueError("max_draft must be positive")
        if max_branches < 1:
            raise ValueError("max_branches must be positive")

        import mlx.core as mx

        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        stats = TreeSpeculativeStats()
        self._reset_stream_state()

        drafter = drafter or PromptLookupTreeDrafter()
        drafter.reset()
        stops = stop_tokens or set()

        pin_event = self._pin_for_run()
        if self._trace:
            events.append({"kind": "load", **compact_event(pin_event)})

        if prompt_token_rows is not None:
            if len(prompt_token_rows) != 1:
                raise ValueError("prompt_token_rows must match prompts")
            prompt_ids = list(prompt_token_rows[0])
        else:
            prompt_ids = list(self.tokenizer.encode(prompts[0]))
        if not prompt_ids:
            raise ValueError("the prompt must produce at least one token")
        if cached_tokens < 0 or cached_tokens >= len(prompt_ids):
            raise ValueError("cached_tokens must leave at least one prompt token to process")
        if cached_tokens and cache is None:
            raise ValueError("cached_tokens requires the matching cache")

        embeddings_warmed = False
        if self.pin_policy == "phase" and self.warm_embeddings:
            load_embedding = self.session.load_embedding()
            if self._trace:
                events.append({"kind": "load", "pass": "warm", **compact_event(load_embedding)})
            embeddings_warmed = True

        output_warmed = False
        if self.warm_output:
            load_output = self.session.load_output()
            if self._trace:
                events.append({"kind": "load", "pass": "warm-output", **compact_event(load_output)})
            output_warmed = True

        if cache is None:
            cache = self._make_cache()

        emitted: list[int] = []
        finish_reason = "length"
        try:
            fresh = prompt_ids[cached_tokens:]
            if len(fresh) > 1:
                self._prefill_cache_tokens(
                    [fresh[:-1]],
                    cache=cache,
                    events=events,
                    manage_embedding=not embeddings_warmed,
                )
            hidden = self._stream_forward_tokens(
                [[fresh[-1]]],
                cache=cache,
                events=events,
                pass_kind="prefill",
                manage_embedding=not embeddings_warmed,
            )
            logits = self._logits_from_hidden(hidden, events)

            context = list(prompt_ids)
            reservoir: list[list[int]] = []
            gate = AdaptiveDraftGate()
            near_tie_events = 0

            def emit(token: int) -> bool:
                nonlocal finish_reason
                emitted.append(token)
                context.append(token)
                stopped = token in stops
                if stopped:
                    finish_reason = "stop"
                return stopped or len(emitted) >= max_tokens

            while len(emitted) < max_tokens:
                stats.rounds += 1
                token_array = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(token_array)
                next_token = int(token_array.item())
                if emit(next_token):
                    break

                budget = min(max_draft, max_tokens - len(emitted))
                draft_started = time.perf_counter()
                if not gate.allow_draft():
                    # Adaptive gate: consecutive zero-accept rounds proved this
                    # span undraftable; single-step through the cooldown (or for
                    # the rest of the request once disabled) so speculation is
                    # never a throughput regression.
                    branches = []
                elif hasattr(drafter, "propose_branches"):
                    branches = drafter.propose_branches(context, budget, max_branches)
                else:
                    branch = drafter.propose(context, budget)
                    branches = [branch] if branch else []
                stats.draft_seconds += time.perf_counter() - draft_started

                # Recycled verify tails augment rounds where the drafter found
                # real matches. They must never CREATE a verify round: measured,
                # stale tails accept ~0 on novel text, so turning a cheap
                # single-step into a failed verify+refeed double-pass halves
                # throughput on low-match workloads.
                if branches:
                    seen_branches = {tuple(branch) for branch in branches}
                    for seed in reservoir:
                        candidate = list(seed[:budget])
                        if candidate and len(candidate) < budget:
                            candidate = candidate + [candidate[-1]] * (budget - len(candidate))
                        key = tuple(candidate)
                        if candidate and key not in seen_branches:
                            seen_branches.add(key)
                            branches.append(candidate)
                    branches = branches[:max_branches]

                if not branches:
                    step_started = time.perf_counter()
                    hidden = self._stream_forward_tokens(
                        [[next_token]],
                        cache=cache,
                        events=events,
                        pass_kind="decode",
                        token_step=len(emitted),
                        manage_embedding=not embeddings_warmed,
                    )
                    logits = self._logits_from_hidden(hidden, events)
                    stats.single_steps += 1
                    stats.verify_seconds += time.perf_counter() - step_started
                    continue

                stats.branch_rounds += 1
                stats.branches_verified += len(branches)
                stats.drafted_tokens += sum(len(branch) for branch in branches)

                snapshot_started = time.perf_counter()
                prefix_snapshot = snapshot_cache_states(cache)
                verify_cache = self._make_cache()
                restore_repeated_cache_states(verify_cache, prefix_snapshot, len(branches))
                stats.snapshot_seconds += time.perf_counter() - snapshot_started

                verify_started = time.perf_counter()
                # Branches may have unequal length (a historical span can run
                # into the end of the corpus). A batched forward needs
                # rectangular rows, so right-pad each branch to the round's
                # widest length by repeating its own last token. Causal masking
                # means a padding column at position p only ever feeds positions
                # > p, so the logits at every real position 0..len(branch) are
                # byte-identical to an unpadded run — padding never changes
                # acceptance. Acceptance and tail extraction below index by the
                # branch's TRUE length, so padded columns are simply ignored.
                verify_width = max(len(branch) for branch in branches)
                verify_rows = [
                    [next_token, *branch, *([branch[-1]] * (verify_width - len(branch)) if branch else [])]
                    for branch in branches
                ]
                hidden = self._stream_forward_tokens(
                    verify_rows,
                    cache=verify_cache,
                    events=events,
                    pass_kind="tree-verify",
                    token_step=len(emitted),
                    manage_embedding=not embeddings_warmed,
                )
                verify_logits = self._logits_from_hidden(hidden, events)
                greedy_array = mx.argmax(verify_logits, axis=-1)
                top2 = mx.topk(verify_logits.astype(mx.float32), 2, axis=-1)
                gap_array = mx.abs(top2[..., 0] - top2[..., 1])
                mx.eval(greedy_array, gap_array)
                greedy_rows = greedy_array.tolist()
                gap_rows = gap_array.tolist()
                stats.verify_passes += 1
                stats.verify_seconds += time.perf_counter() - verify_started

                best_index = 0
                best_accepted = -1
                for index, branch in enumerate(branches):
                    accepted = count_accepted_drafts(
                        branch,
                        [int(token) for token in greedy_rows[index][: len(branch)]],
                        gaps=[float(gap) for gap in gap_rows[index][: len(branch)]],
                        margin=draft_margin,
                    )
                    if accepted > best_accepted:
                        best_index = index
                        best_accepted = accepted

                best_branch = branches[best_index]
                tail_from = max(best_accepted, 0) + 1
                winner_tail = [
                    int(token)
                    for token in greedy_rows[best_index][tail_from : tail_from + max_draft]
                ]
                # Only recycle tails from rounds with real acceptance; a dead
                # round's tail is conditioned on rejected context.
                reservoir = [winner_tail] if winner_tail and best_accepted >= 1 else []
                gate.observe(max(best_accepted, 0))
                stats.accepted_tokens += max(best_accepted, 0)
                stopped = False
                for token in best_branch[:best_accepted]:
                    if emit(token):
                        stopped = True
                        break
                if hasattr(drafter, "observe_result"):
                    drafter.observe_result(len(best_branch), max(best_accepted, 0))

                if stopped:
                    break
                decision_position = min(max(best_accepted, 0), len(gap_rows[best_index]) - 1)
                decision_guarded = float(gap_rows[best_index][decision_position]) < draft_margin
                if decision_guarded:
                    # The next-token decision fell inside the margin under
                    # chunked numerics — a near-tie that could flip vs the
                    # single-token baseline. Counted whether or not the guard
                    # path runs this round, so the figure tracks how often
                    # chunked numerics put a decision on the knife's edge.
                    near_tie_events += 1

                if (
                    best_accepted == len(best_branch)
                    and len(best_branch) == verify_width
                    and not decision_guarded
                ):
                    # Fast path only when the winner spans the full verify
                    # width: then verify_logits[best_index][-1] is its true
                    # last position and the committed cache row carries no
                    # padding KV. A shorter winner falls through to refeed,
                    # which re-runs exactly [next_token, *committed] on the
                    # single-row cache and recomputes correct logits.
                    stats.full_accept_rounds += 1
                    restore_cache_batch_row(cache, verify_cache, best_index)
                    logits = verify_logits[best_index : best_index + 1]
                    continue

                refeed_started = time.perf_counter()
                committed = best_branch[: max(best_accepted, 0)]
                if decision_guarded and committed:
                    # The next-token decision sits inside the margin under
                    # chunked numerics. End the round with a single-token pass
                    # so that decision uses single-step numerics — this is the
                    # exactness guard for the deterministic near-tie flips.
                    self._stream_forward_tokens(
                        [[next_token, *committed[:-1]]],
                        cache=cache,
                        events=events,
                        pass_kind="tree-refeed",
                        token_step=len(emitted),
                        manage_embedding=not embeddings_warmed,
                    )
                    hidden = self._stream_forward_tokens(
                        [[committed[-1]]],
                        cache=cache,
                        events=events,
                        pass_kind="tree-refeed-guard",
                        token_step=len(emitted),
                        manage_embedding=not embeddings_warmed,
                    )
                    stats.refeed_passes += 2
                else:
                    hidden = self._stream_forward_tokens(
                        [[next_token, *committed]],
                        cache=cache,
                        events=events,
                        pass_kind="tree-refeed",
                        token_step=len(emitted),
                        manage_embedding=not embeddings_warmed,
                    )
                    stats.refeed_passes += 1
                logits = self._logits_from_hidden(hidden, events)
                stats.refeed_seconds += time.perf_counter() - refeed_started
        finally:
            if embeddings_warmed:
                evict_embedding = self.session.evict_embedding()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm", **compact_event(evict_embedding)})
            if output_warmed:
                evict_output = self.session.evict_output()
                if self._trace:
                    events.append({"kind": "evict", "pass": "warm-output", **compact_event(evict_output)})

        stats.emitted_tokens = len(emitted)
        speculative = stats.to_dict()
        speculative.update(gate.telemetry())
        speculative["near_tie_events"] = near_tie_events
        speculative["draft_policy"] = getattr(drafter, "name", type(drafter).__name__)
        speculative["drafter"] = getattr(drafter, "name", type(drafter).__name__)
        if hasattr(drafter, "telemetry"):
            speculative["drafter_stats"] = drafter.telemetry()
        return MlxBatchGenerationResult(
            prompts=tuple(prompts),
            prompt_tokens=len(prompt_ids),
            generated_tokens=(tuple(emitted),),
            generated_texts=(self.tokenizer.decode(emitted),),
            seconds=time.perf_counter() - started,
            resident_peak_bytes=self.session.peak_resident_bytes,
            kv_cache_bytes=kv_cache_nbytes(cache),
            events=tuple(events),
            expert_prefetch=self._expert_telemetry(),
            finish_reasons=(finish_reason,),
            speculative=speculative,
        )

    def _prefill_cache_tokens(
        self,
        token_rows: list[list[int]],
        *,
        cache: list[Any],
        events: list[dict[str, Any]],
        manage_embedding: bool,
    ) -> None:
        # Base implementation: the base generate path calls this, but only the DeepSeek
        # runner sets self.model_type, so guard with getattr. GLM (glm_moe_dsa, DeepSeek
        # runner) overrides with its own copy; gpt-oss/Qwen use this generic chunked path.
        if not token_rows or not token_rows[0]:
            return
        if getattr(self, "model_type", None) == "glm_moe_dsa":
            for position in range(len(token_rows[0])):
                self._stream_forward_tokens(
                    [[row[position]] for row in token_rows],
                    cache=cache,
                    events=events,
                    pass_kind="prefill-cache",
                    manage_embedding=manage_embedding,
                )
            return
        self._stream_forward_tokens(
            token_rows,
            cache=cache,
            events=events,
            pass_kind="prefill-cache",
            manage_embedding=manage_embedding,
        )


class GptOssStreamingForwardRunner(StreamingChatRunner):
    """Generate with GPT-OSS through SmartTensor selective-expert streaming.

    GPT-OSS layers are ~875MB of which ~846MB is the 32-expert bank; only the
    top-4 experts fire per token. Each layer streams its ~28MB base
    (attention, norms, router) and then loads only the selected expert rows.
    """

    EXPERT_MARKER = ".mlp.experts."
    # Cold expert must beat the weakest resident by this many selection
    # counts before a hot-table membership rebuild is considered.
    REFRESH_HYSTERESIS = 4

    def __init__(
        self,
        model_dir: str | Path,
        *,
        evaluate: bool = True,
        retain_layers: set[int] | None = None,
        resident_budget_bytes: int | None = None,
        backend: str = "native",
        clear_on_evict: bool = True,
        pin_policy: str = "all",
        warm_embeddings: bool = False,
        expert_hot_set: int = 0,
        trace: bool = False,
        sliding_cache: str = "rotating",
        pack_dir: str | Path | None = None,
        native_layers: int = 0,
        weight_page_budget_bytes: int | None = None,
        weight_page_policy: str = "auto",
        weight_page_rows: int = 1,
        decode_scheduler: str = "auto",
        expert_compute_mode: str = "table",
        expert_prefetch: str = "off",
        expert_prefetch_cap: int = 32,
        pack_read_workers: int = 1,
    ) -> None:
        if pin_policy not in {"all", "phase"}:
            raise ValueError("pin_policy must be 'all' or 'phase'")
        if warm_embeddings and pin_policy != "phase":
            raise ValueError("warm_embeddings only applies to pin_policy='phase'")
        if expert_hot_set < 0:
            raise ValueError("expert_hot_set must be non-negative")
        if sliding_cache not in {"rotating", "kv", "temporal"}:
            raise ValueError("sliding_cache must be 'rotating', 'kv', or 'temporal'")
        if native_layers < 0:
            raise ValueError("native_layers must be non-negative")
        if weight_page_rows < 1:
            raise ValueError("weight_page_rows must be positive")
        if weight_page_rows != 1 and weight_page_budget_bytes is None:
            raise ValueError("weight_page_rows requires weight_page_budget_bytes")
        if decode_scheduler not in {"auto", "serial", "async-lookahead"}:
            raise ValueError("decode_scheduler must be 'auto', 'serial', or 'async-lookahead'")
        if expert_compute_mode not in {"table", "direct_qmm"}:
            raise ValueError("GPT-OSS expert_compute_mode must be 'table' or 'direct_qmm'")
        if expert_compute_mode == "direct_qmm" and expert_hot_set:
            raise ValueError("GPT-OSS direct_qmm currently bypasses hot expert tables")
        if expert_prefetch not in {"off", "previous"}:
            raise ValueError("GPT-OSS expert_prefetch must be 'off' or 'previous'")
        if expert_prefetch != "off" and weight_page_budget_bytes is None:
            raise ValueError("GPT-OSS expert_prefetch requires --weight-page-budget")
        if expert_prefetch_cap < 1:
            raise ValueError("expert_prefetch_cap must be positive")
        if pack_read_workers < 1:
            raise ValueError("pack_read_workers must be positive")
        # Lean mode (trace=False, the default for serve and benchmarks) skips
        # per-layer/per-token event construction in the hot loop; trace mode
        # restores the full event stream and collects --trace-pass buckets.
        self._trace = trace
        self._pass_trace = PassTraceCollector() if trace else None
        self.session = MlxModelSession(
            model_dir,
            evaluate=evaluate,
            retain_layers=retain_layers,
            resident_budget_bytes=None,
            backend=backend,
            clear_on_evict=clear_on_evict,
            pin_policy=pin_policy,
            warm_embeddings=warm_embeddings,
            trace=trace,
        )
        self.pin_policy = pin_policy
        self.warm_embeddings = warm_embeddings
        self.warm_output = pin_policy == "phase"
        self._embed_prefix = "model.embed_tokens"
        self._base_loaded: set[int] = set()
        self.sliding_cache = sliding_cache
        self.pack_dir = pack_dir
        self.pack_read_workers = pack_read_workers
        self.resident_budget_bytes = resident_budget_bytes
        if pack_dir is not None:
            self.session.loader.attach_pack_dir(
                pack_dir,
                pack_read_workers=pack_read_workers,
            )
        weight_page_policy = resolve_weight_page_policy(
            "gpt_oss",
            weight_page_policy,
            has_weight_page_budget=weight_page_budget_bytes is not None,
        )
        self.weight_page_budget_bytes = weight_page_budget_bytes
        self.weight_page_policy = weight_page_policy
        self.weight_page_rows = weight_page_rows
        self.expert_compute_mode = expert_compute_mode
        self.expert_prefetch = expert_prefetch
        self.expert_prefetch_cap = expert_prefetch_cap
        if weight_page_budget_bytes is not None:
            self.session.loader.attach_weight_page_cache(
                weight_page_budget_bytes,
                eviction_policy=weight_page_policy,
                rows_per_page=weight_page_rows,
            )
        self.decode_scheduler = decode_scheduler
        # Native-resident layers: their full expert bank + base is loaded once
        # into the model shell and the MLX-native fused layer forward runs
        # (native SwitchGLU does top-k internally), skipping the manual
        # router-sync / numpy expert-select / remap / per-layer eval that make
        # the streamed path ~3.2x slower than native on resident weights.
        # Cold layers keep the manual selective streaming path.
        self.native_layers: set[int] = set(range(native_layers))
        self._native_loaded: set[int] = set()
        self._native_resident_bytes = 0
        # Overlapped (async_eval-pipelined) greedy decode is only enabled when
        # the full forward is resident/native and lazy end-to-end. Streaming
        # layers still perform host routing and evictions, so they remain on
        # the serial scheduler until in-flight weight fences exist.
        self._supports_overlap = True
        if self.session.config.get("model_type") != "gpt_oss":
            raise ValueError("GptOssStreamingForwardRunner currently supports model_type=gpt_oss only")
        self.top_k = int(self.session.config.get("num_experts_per_tok", 4))
        # Hot expert residency: keep up to N most-recently-first-seen expert
        # rows per layer resident (lazy-grow: rows enter the set when a miss
        # loads them anyway, so warming costs no extra IO). Measured working
        # set: top-8 of 32 experts serve 81% of selections, top-12 serve 92%.
        self.expert_hot_set = expert_hot_set
        self._hot_tables: dict[int, dict[str, Any]] = {}
        self._hot_counts: dict[int, dict[int, int]] = {}
        # Layers whose expert modules are bound to a temporary (over-cap) pass
        # table rather than the resident hot table; only these need the
        # placeholder clear after the layer forward.
        self._bound_temporary_layers: set[int] = set()
        # Stable slots: the arrays dict currently bound to each layer's expert
        # modules, so all-hit passes skip rebinding identical tables.
        self._bound_tables: dict[int, Any] = {}
        self._expert_names_cache: dict[int, tuple[str, ...]] = {}
        self._expert_cache_bytes = 0
        self._hot_stats = {
            "hit_passes": 0,
            "miss_passes": 0,
            "overflow_passes": 0,
            "refreshes": 0,
            "served_rows": 0,
            "loaded_rows": 0,
            "temporary_rows": 0,
        }
        self._expert_history: dict[int, list[int]] = {}
        self._pending_expert_prefetch: dict[str, Any] | None = None
        self._pending_expert_prefetch_bytes = 0
        self._prefetch_stats = ExpertPrefetchStats()
        self.loader_executor = ThreadPoolExecutor(max_workers=1)
        first_layer = self.session.loader.manifest.layers[min(self.session.loader.manifest.layers)]
        self._expert_row_bytes = sum(
            self.session.loader.manifest.tensors[name].nbytes
            for name in first_layer.tensor_names
            if self.EXPERT_MARKER in name
        ) // 32
        if retain_layers is not None:
            self.base_retain_layers = set(retain_layers)
        elif resident_budget_bytes is not None:
            self.base_retain_layers = select_qwen_base_layers_for_budget(
                self.session.loader.manifest,
                resident_budget_bytes,
                top_k=self.top_k,
                expert_marker=self.EXPERT_MARKER,
            )
        else:
            self.base_retain_layers = set()

        from mlx_lm.utils import load_tokenizer

        self.tokenizer = load_tokenizer(self.session.model_dir)

    def _resident_sidecar_bytes(self) -> int:
        return (
            self._expert_cache_bytes
            + self.session.loader.weight_page_resident_bytes
            + self._pending_expert_prefetch_bytes
        )

    def _async_lookahead_support(self) -> tuple[bool, str]:
        total_layers = len(self.session.model.model.layers)
        if self.pin_policy != "all":
            return False, "requires pin-policy all so embeddings/output stay resident"
        if len(self.native_layers) != total_layers:
            return False, "requires every layer to be native-resident; streamed layers can evict in-flight weights"
        if set(self.native_layers) != set(range(total_layers)):
            return False, "native-resident layers must cover the full contiguous model"
        if self.expert_hot_set:
            return False, "hot expert tables are a streamed-layer optimization, not overlap-safe yet"
        if self.weight_page_budget_bytes is not None:
            return False, "weight page cache can evict streamed weights; async fences are not implemented yet"
        if self.resident_budget_bytes is not None:
            overlap_margin = 256 * 1024 * 1024
            estimated_peak = self.session.resident_bytes + overlap_margin
            if estimated_peak > self.resident_budget_bytes:
                return (
                    False,
                    "resident budget leaves no async-lookahead headroom "
                    f"({estimated_peak} > {self.resident_budget_bytes})",
                )
        return True, "full native-resident GPT-OSS forward is overlap-safe"

    def _make_cache(self) -> list[Any]:
        cache = super()._make_cache()
        if self.sliding_cache == "rotating":
            return cache

        from mlx_lm.models.cache import KVCache

        inner = self.session.model.model
        for index, layer_type in enumerate(inner.layer_types):
            if layer_type != "full_attention":
                if self.sliding_cache == "kv":
                    cache[index] = KVCache()
                else:
                    cache[index] = TemporalSlidingKVCache(inner.window_size)
        return cache

    def _embed_module(self) -> Any:
        return self.session.model.model.embed_tokens

    def next_token(self, prompt: str, *, max_layers: int | None = None) -> MlxForwardResult:
        import mlx.core as mx

        started = time.perf_counter()
        events: list[dict[str, Any]] = []

        pin_event = self._pin_for_run()
        if self._trace:
            events.append({"kind": "load", **compact_event(pin_event)})

        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            raise ValueError("prompt produced no tokens")

        inner = self.session.model.model
        total_layers = len(inner.layers)
        layer_limit = total_layers if max_layers is None else min(max_layers, total_layers)
        cache = [None] * total_layers
        x = self._stream_forward_tokens(
            [tokens],
            cache=cache,
            events=events,
            max_layers=layer_limit,
            pass_kind="prefill",
        )

        next_token: int | None = None
        next_text: str | None = None
        if layer_limit == total_layers:
            logits = self._logits_from_hidden(x, events)
            token_array = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token_array)
            next_token = int(token_array.item())
            next_text = self.tokenizer.decode([next_token])

        return MlxForwardResult(
            prompt=prompt,
            prompt_tokens=len(tokens),
            completed_layers=layer_limit,
            total_layers=total_layers,
            seconds=time.perf_counter() - started,
            next_token=next_token,
            next_text=next_text,
            resident_peak_bytes=self.session.peak_resident_bytes,
            events=tuple(events),
        )

    def close(self) -> None:
        self._drain_gptoss_expert_prefetch()
        self.loader_executor.shutdown(wait=True)
        self._hot_tables.clear()
        self._hot_counts.clear()
        self._bound_temporary_layers.clear()
        self._bound_tables.clear()
        self._expert_cache_bytes = 0
        self.session.close()

    def _reset_stream_state(self) -> None:
        self._drain_gptoss_expert_prefetch()
        self._expert_history.clear()
        self._prefetch_stats = ExpertPrefetchStats()

    def _drain_gptoss_expert_prefetch(self) -> None:
        pending = getattr(self, "_pending_expert_prefetch", None)
        self._pending_expert_prefetch = None
        self._pending_expert_prefetch_bytes = 0
        if pending is not None:
            pending["future"].result()
            self._set_external_resident_bytes()

    def _gptoss_slice_nbytes(self, layer_index: int, expert_ids: list[int]) -> int:
        if not expert_ids:
            return 0
        total = 0
        for name in self._expert_slice_names(layer_index):
            record = self.session.loader.manifest.tensors[name]
            total += record.nbytes * len(expert_ids) // record.shape[0]
        return total

    def _gptoss_slice_page_miss_nbytes(
        self,
        layer_index: int,
        expert_ids: list[int],
        *,
        cap_to_headroom: bool = True,
    ) -> int:
        return self.session.loader.estimate_first_dim_slice_page_miss_bytes(
            self._expert_slice_names(layer_index),
            expert_ids,
            cap_to_headroom=cap_to_headroom,
        )

    def _gptoss_budget_allows(
        self,
        *,
        additional_sidecar_bytes: int = 0,
        temporary_bytes: int = 0,
    ) -> bool:
        budget = getattr(self, "resident_budget_bytes", None)
        if budget is None:
            return True
        return self.session.resident_bytes + additional_sidecar_bytes + temporary_bytes <= budget

    def _maybe_submit_gptoss_expert_prefetch(self, layer_index: int) -> None:
        if self.expert_prefetch != "previous":
            return
        if self._pending_expert_prefetch is not None:
            return
        history = self._expert_history.get(layer_index)
        if not history:
            self._prefetch_stats.skipped_no_history += 1
            return
        if len(history) > self.expert_prefetch_cap:
            self._prefetch_stats.skipped_over_cap += 1
            return

        predicted = list(history)
        page_growth_bytes = self._gptoss_slice_page_miss_nbytes(layer_index, predicted)
        predicted_table_bytes = self._gptoss_slice_nbytes(layer_index, predicted)
        if not self._gptoss_budget_allows(
            additional_sidecar_bytes=page_growth_bytes,
            temporary_bytes=predicted_table_bytes,
        ):
            self._prefetch_stats.skipped_over_budget += 1
            return

        future = self.loader_executor.submit(
            self.session.loader.warm_first_dim_slices,
            self._expert_slice_names(layer_index),
            predicted,
        )
        self._pending_expert_prefetch_bytes = page_growth_bytes
        self._set_external_resident_bytes()
        self._pending_expert_prefetch = {
            "layer": layer_index,
            "experts": predicted,
            "future": future,
        }

    def _consume_gptoss_expert_prefetch(
        self,
        layer_index: int,
        selected_experts: list[int],
        *,
        pass_kind: str,
        token_step: int | None,
        events: list[dict[str, Any]],
    ) -> None:
        pending = self._pending_expert_prefetch
        self._pending_expert_prefetch = None
        self._pending_expert_prefetch_bytes = 0
        if pending is None:
            return

        join_started = time.perf_counter()
        batch = pending["future"].result()
        join_wait = time.perf_counter() - join_started
        self._set_external_resident_bytes()
        if pending["layer"] != layer_index:
            self._prefetch_stats.skipped_no_history += 1
            return

        predicted = pending["experts"]
        _ordering, hits, missing = expert_merge_plan(predicted, selected_experts)
        wasted = sorted(set(predicted) - set(selected_experts))
        stats = self._prefetch_stats
        stats.attempted_layers += 1
        stats.predicted_rows += len(predicted)
        stats.true_rows += len(selected_experts)
        stats.hit_rows += len(hits)
        stats.missing_rows += len(missing)
        stats.wasted_rows += len(wasted)
        stats.prefetched_bytes += batch.nbytes
        row_bytes = batch.nbytes // len(predicted) if predicted else 0
        stats.wasted_bytes += row_bytes * len(wasted)
        stats.prefetch_load_seconds += batch.seconds
        stats.join_wait_seconds += join_wait
        if not missing:
            stats.full_hits += 1
        if self._trace:
            events.append(
                {
                    "kind": "load",
                    "action": "prefetch-warm-expert-pages",
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    "predicted_experts": predicted,
                    "hit_count": len(hits),
                    "missing_count": len(missing),
                    "join_wait_seconds": join_wait,
                    "prefetch_seconds": batch.seconds,
                    "resident_bytes": self.session.resident_bytes,
                    "nbytes_loaded": batch.nbytes,
                }
            )

    def _stream_forward_tokens(
        self,
        token_rows: list[list[int]],
        *,
        cache: list[Any],
        events: list[dict[str, Any]],
        pass_kind: str,
        max_layers: int | None = None,
        token_step: int | None = None,
        manage_embedding: bool = True,
        on_layer: Any | None = None,
        on_layer_stage: Any | None = None,
        input_array: Any | None = None,
    ) -> Any:
        import mlx.core as mx
        from mlx_lm.models.gpt_oss import create_attention_mask

        trace = self._trace
        pass_trace = self._pass_trace
        pass_started = time.perf_counter()
        inner = self.session.model.model
        if self.pin_policy == "phase" and manage_embedding:
            load_embedding, local_token_rows = self._load_embedding_slices(token_rows)
            if trace:
                events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_embedding)})
            inputs = mx.array(local_token_rows)
            embed_started = time.perf_counter()
            try:
                x = inner.embed_tokens(inputs)
                if pass_trace is not None:
                    pass_trace.add("embedding_eval", time.perf_counter() - embed_started)
                if trace:
                    events.append(
                        {
                            "kind": "compute",
                            "action": "embedding",
                            "pass": pass_kind,
                            "token_step": token_step,
                            "seconds": time.perf_counter() - embed_started,
                            "hidden_shape": list(x.shape),
                        }
                    )
            finally:
                self._clear_embedding_slices()
                if trace:
                    evict_embedding = MlxStreamEvent(
                        action="evict-embedding-slices",
                        seconds=0.0,
                        resident_bytes=self.session.resident_bytes,
                        requested=load_embedding.requested,
                        evicted=load_embedding.loaded,
                    )
                    self.session.events.append(evict_embedding)
                    events.append({"kind": "evict", "pass": pass_kind, "token_step": token_step, **compact_event(evict_embedding)})
        else:
            # input_array (a lazy [batch,1] token array) lets the overlapped
            # decoder feed the previous step's still-unmaterialized token
            # straight into this forward, so the GPU pipelines instead of
            # waiting on a Python .item() between tokens.
            inputs = input_array if input_array is not None else mx.array(token_rows)
            embed_started = time.perf_counter()
            x = inner.embed_tokens(inputs)
            if pass_trace is not None:
                pass_trace.add("embedding_eval", time.perf_counter() - embed_started)
            if trace:
                events.append(
                    {
                        "kind": "compute",
                        "action": "embedding",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "seconds": time.perf_counter() - embed_started,
                        "hidden_shape": list(x.shape),
                    }
                )

        full_mask = create_attention_mask(x, cache[inner.ga_idx])
        swa_mask = create_attention_mask(x, cache[inner.swa_idx], window_size=inner.window_size)

        total_layers = len(inner.layers)
        layer_limit = total_layers if max_layers is None else min(max_layers, total_layers)

        for layer_index in range(layer_limit):
            layer = inner.layers[layer_index]
            layer_cache = cache[layer_index]
            layer_type = inner.layer_types[layer_index]
            mask = full_mask if layer_type == "full_attention" else swa_mask
            self._maybe_submit_gptoss_expert_prefetch(layer_index)

            # Native-resident layer: full bank + base loaded once into the
            # shell; run MLX's fused layer forward, no manual MoE / clear / evict.
            if layer_index in self.native_layers:
                if layer_index not in self._native_loaded:
                    native_started = time.perf_counter()
                    self._load_native_layer(layer_index)
                    if pass_trace is not None:
                        pass_trace.add("native_layer_load", time.perf_counter() - native_started)
                compute_started = time.perf_counter()
                x = layer(x, mask, layer_cache)
                if pass_trace is not None:
                    pass_trace.add("native_layer_compute", time.perf_counter() - compute_started)
                if trace:
                    events.append({"kind": "compute", "action": "native-layer", "pass": pass_kind,
                                   "token_step": token_step, "layer": layer_index,
                                   "seconds": time.perf_counter() - compute_started, "hidden_shape": list(x.shape)})
                if on_layer is not None:
                    mx.eval(x)
                    on_layer(layer_index, x, cache[layer_index])
                continue

            if layer_index in self._base_loaded:
                pass  # retained base resident since its first pass: skip bookkeeping
            else:
                load_event = self._load_layer_base(layer_index)
                if pass_trace is not None:
                    pass_trace.add("layer_base_load", load_event.seconds)
                if trace:
                    events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_event)})
                if layer_index in self.base_retain_layers:
                    self._base_loaded.add(layer_index)

            compute_started = time.perf_counter()
            try:
                x = self._layer_forward_selective_experts(
                    layer_index,
                    layer,
                    x,
                    mask=mask,
                    cache=layer_cache,
                    events=events,
                    pass_kind=pass_kind,
                    token_step=token_step,
                    on_stage=on_layer_stage,
                )
            finally:
                if self.expert_compute_mode != "direct_qmm":
                    clear_started = time.perf_counter()
                    self._clear_selected_experts(layer_index, layer)
                    if pass_trace is not None:
                        pass_trace.add("clear_experts", time.perf_counter() - clear_started)
            if trace:
                events.append(
                    {
                        "kind": "compute",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "seconds": time.perf_counter() - compute_started,
                        "hidden_shape": list(x.shape),
                    }
                )
            if on_layer is not None:
                mx.eval(x)
                on_layer(layer_index, x, cache[layer_index])

            if layer_index not in self.base_retain_layers:
                evict_started = time.perf_counter()
                evict_event = self.session.evict_layer(layer_index)
                if pass_trace is not None:
                    pass_trace.add("evict", time.perf_counter() - evict_started)
                if trace:
                    events.append({"kind": "evict", "pass": pass_kind, "token_step": token_step, **compact_event(evict_event)})

        if pass_trace is not None:
            pass_trace.add_pass(time.perf_counter() - pass_started)
        return x

    def _load_layer_base(self, layer_index: int) -> MlxStreamEvent:
        layer = self.session.loader.manifest.layers[layer_index]
        names = tuple(name for name in layer.tensor_names if self.EXPERT_MARKER not in name)
        return self.session._load_into_model(names, action="load-layer-base", layer=layer_index)

    def _load_native_layer(self, layer_index: int) -> None:
        """Load a layer's full bank + base resident into the shell, once.

        After this the layer's modules hold native (32-expert) weights, so the
        fused ``layer(x, mask, cache)`` forward runs without any manual expert
        selection. Resident for the runner's lifetime (not evicted/cleared).
        """

        layer = self.session.loader.manifest.layers[layer_index]
        names = tuple(layer.tensor_names)
        self.session._load_into_model(names, action="load-native-layer", layer=layer_index)
        self._native_loaded.add(layer_index)
        self._native_resident_bytes += sum(
            self.session.loader.manifest.tensors[name].nbytes for name in names
        )
        self._set_external_resident_bytes()

    def _layer_forward_selective_experts(
        self,
        layer_index: int,
        layer: Any,
        x: Any,
        *,
        mask: Any,
        cache: Any,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
        on_stage: Any | None = None,
    ) -> Any:
        residual = x
        hidden = layer.input_layernorm(x)
        if on_stage is not None:
            on_stage(layer_index, "attention_norm", hidden)
        if isinstance(cache, TemporalSlidingKVCache) and 1 < hidden.shape[1] <= 16:
            hidden = self._temporal_sliding_attention(layer.self_attn, hidden, cache)
        else:
            hidden = layer.self_attn(hidden, mask, cache)
        if on_stage is not None:
            on_stage(layer_index, "attention", hidden)
        x = residual + hidden
        if on_stage is not None:
            on_stage(layer_index, "attention_residual", x)

        residual = x
        hidden = layer.post_attention_layernorm(x)
        if on_stage is not None:
            on_stage(layer_index, "moe_norm", hidden)
        moe_out = self._moe_forward_selective_experts(
            layer_index,
            layer.mlp,
            hidden,
            events=events,
            pass_kind=pass_kind,
            token_step=token_step,
        )
        if on_stage is not None:
            on_stage(layer_index, "moe", moe_out)
        return residual + moe_out

    def _temporal_sliding_attention(self, attention: Any, x: Any, cache: TemporalSlidingKVCache) -> Any:
        import mlx.core as mx
        from mlx_lm.models.gpt_oss import scaled_dot_product_attention

        batch, length, _ = x.shape
        head_dim = attention.head_dim
        query_heads = attention.num_attention_heads
        kv_heads = attention.num_key_value_heads
        previous_offset = cache.offset

        q = attention.q_proj(x).reshape(batch, length, query_heads, head_dim).swapaxes(1, 2)
        k = attention.k_proj(x).reshape(batch, length, kv_heads, head_dim).swapaxes(1, 2)
        v = attention.v_proj(x).reshape(batch, length, kv_heads, head_dim).swapaxes(1, 2)

        q = attention.rope(q, offset=previous_offset)
        k = attention.rope(k, offset=previous_offset)
        keys, values = cache.update_and_fetch(k, v)

        outputs = []
        for position in range(length):
            absolute = previous_offset + position
            start_absolute = max(cache.start_position, absolute - cache.max_size + 1)
            end_absolute = absolute + 1
            start = start_absolute - cache.start_position
            end = end_absolute - cache.start_position
            outputs.append(
                scaled_dot_product_attention(
                    q[:, :, position : position + 1, :],
                    keys[:, :, start:end, :],
                    values[:, :, start:end, :],
                    cache,
                    attention.sm_scale,
                    mask=None,
                    sinks=attention.sinks,
                )
            )

        attended = mx.concatenate(outputs, axis=2)
        return attention.o_proj(attended.swapaxes(1, 2).reshape(batch, length, -1))

    def _moe_forward_selective_experts(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        *,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx
        from mlx_lm.models.gpt_oss import mlx_topk

        trace = self._trace
        pass_trace = self._pass_trace

        gates = mlp.router(x)
        values, expert_indices = mlx_topk(gates, k=mlp.num_experts_per_tok, axis=-1)
        expert_weights = mx.softmax(values, axis=-1, precise=True)

        sync_started = time.perf_counter() if pass_trace is not None else 0.0
        # One host round-trip per layer: np.asarray materializes the indices
        # (the load plan is host-driven), and ids/counts/remap all reuse it.
        np_indices = np.asarray(expert_indices)
        unique_ids, unique_counts = np.unique(np_indices, return_counts=True)
        selected_experts = [int(expert) for expert in unique_ids]
        expert_counts = {
            int(expert): int(count) for expert, count in zip(unique_ids, unique_counts)
        }
        if pass_trace is not None:
            pass_trace.add("router_indices_sync", time.perf_counter() - sync_started)
        self._consume_gptoss_expert_prefetch(
            layer_index,
            selected_experts,
            pass_kind=pass_kind,
            token_step=token_step,
            events=events,
        )
        self._expert_history[layer_index] = selected_experts

        if self.expert_compute_mode == "direct_qmm":
            return self._moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                np_indices,
                selected_experts,
                expert_weights,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        if self.expert_hot_set > 0:
            table_order, load_event = self._serve_experts_from_hot_set(
                layer_index, mlp, selected_experts, expert_counts
            )
        else:
            load_event = self._load_selected_experts(layer_index, mlp, selected_experts)
            table_order = selected_experts
        if trace and load_event is not None:
            events.append(
                {
                    "kind": "load",
                    "action": load_event.action,
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    **compact_event(load_event),
                }
            )

        remap_started = time.perf_counter() if pass_trace is not None else 0.0
        local_indices = remap_expert_indices(np_indices, table_order)
        if pass_trace is not None:
            pass_trace.add("remap", time.perf_counter() - remap_started)
        expert_y = mlp.experts(x, local_indices)
        expert_y = expert_y * mx.expand_dims(expert_weights, axis=-1)
        return expert_y.sum(axis=-2)

    def _moe_forward_direct_qmm(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        expert_weights: Any,
        *,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx

        trace = self._trace
        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        batch = self.session.loader.load_first_dim_slices(
            names,
            selected_experts,
            evaluate=self.session.evaluate,
        )
        self._set_external_resident_bytes(batch.nbytes)
        if self._pass_trace is not None:
            self._pass_trace.add("expert_load", time.perf_counter() - started)
        try:
            local_indices = remap_expert_indices(expert_indices, selected_experts)
            x_expanded = mx.expand_dims(x, (-2, -3))
            up = self._gptoss_projection_qmm(
                mlp,
                layer_index,
                "up_proj",
                x_expanded,
                local_indices,
                batch.arrays,
            )
            gate = self._gptoss_projection_qmm(
                mlp,
                layer_index,
                "gate_proj",
                x_expanded,
                local_indices,
                batch.arrays,
            )
            expert_y = self._gptoss_projection_qmm(
                mlp,
                layer_index,
                "down_proj",
                mlp.experts.activation(up, gate),
                local_indices,
                batch.arrays,
            ).squeeze(-2)
            if trace:
                events.append(
                    {
                        "kind": "load",
                        "action": "load-selected-experts-direct-qmm",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                        "seconds": time.perf_counter() - started,
                        "resident_bytes": self.session.resident_bytes,
                        "nbytes_loaded": batch.nbytes,
                    }
                )
            expert_y = expert_y * mx.expand_dims(expert_weights, axis=-1)
            return expert_y.sum(axis=-2)
        finally:
            self._set_external_resident_bytes()

    def _gptoss_projection_qmm(
        self,
        mlp: Any,
        layer_index: int,
        projection: str,
        inputs: Any,
        local_indices: Any,
        arrays: dict[str, Any],
    ) -> Any:
        import mlx.core as mx

        prefix = f"model.layers.{layer_index}.mlp.experts.{projection}"
        module = getattr(mlp.experts, projection)
        projected = mx.gather_qmm(
            inputs,
            arrays[f"{prefix}.weight"],
            arrays[f"{prefix}.scales"],
            arrays.get(f"{prefix}.biases"),
            rhs_indices=local_indices,
            transpose=True,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
            sorted_indices=False,
        )
        bias = arrays.get(f"{prefix}.bias")
        if bias is not None:
            projected = projected + mx.expand_dims(bias[local_indices], -2)
        return projected

    def _expert_slice_names(self, layer_index: int) -> tuple[str, ...]:
        names = self._expert_names_cache.get(layer_index)
        if names is None:
            prefix = f"model.layers.{layer_index}.mlp.experts."
            names = tuple(
                sorted(
                    name
                    for name in self.session.loader.manifest.layers[layer_index].tensor_names
                    if name.startswith(prefix)
                )
            )
            self._expert_names_cache[layer_index] = names
        return names

    def _load_selected_experts(
        self,
        layer_index: int,
        mlp: Any,
        selected_experts: list[int],
    ) -> MlxStreamEvent | None:
        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        batch = self.session.loader.load_first_dim_slices(
            names,
            selected_experts,
            evaluate=self.session.evaluate,
        )
        for name in names:
            parts = name.split(".")
            module = getattr(mlp.experts, parts[-2])
            setattr(module, parts[-1], batch.arrays[name])
        self._set_external_resident_bytes(batch.nbytes)
        if self._pass_trace is not None:
            self._pass_trace.add("expert_load", time.perf_counter() - started)
        if not self._trace:
            return None
        event = MlxStreamEvent(
            action="load-selected-experts",
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names,
            nbytes_loaded=batch.nbytes,
        )
        self.session.events.append(event)
        return event

    def _serve_experts_from_hot_set(
        self,
        layer_index: int,
        mlp: Any,
        selected_experts: list[int],
        selected_counts: dict[int, int],
    ) -> tuple[list[int], MlxStreamEvent | None]:
        import mlx.core as mx

        pass_trace = self._pass_trace
        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        hot = self._hot_tables.setdefault(layer_index, {"order": [], "arrays": {}})
        counts = self._hot_counts.setdefault(layer_index, {})
        for expert, count in selected_counts.items():
            counts[expert] = counts.get(expert, 0) + count

        hot_ids = set(hot["order"])
        missing = [expert for expert in selected_experts if expert not in hot_ids]

        nbytes_loaded = 0
        timed_sub_seconds = 0.0
        if missing:
            self._hot_stats["miss_passes"] += 1
            self._hot_stats["loaded_rows"] += len(missing)
            load_started = time.perf_counter()
            batch = self.session.loader.load_first_dim_slices(
                names, missing, evaluate=self.session.evaluate
            )
            if pass_trace is not None:
                load_seconds = time.perf_counter() - load_started
                timed_sub_seconds += load_seconds
                pass_trace.add("expert_load", load_seconds)
            assembly_started = time.perf_counter()
            nbytes_loaded = batch.nbytes
            if hot["order"]:
                arrays = {
                    name: mx.concatenate([hot["arrays"][name], batch.arrays[name]], axis=0)
                    for name in names
                }
            else:
                arrays = dict(batch.arrays)
            order = hot["order"] + missing
            if len(order) <= self.expert_hot_set:
                # Lazy-grow: adopt the merged table as the new resident set.
                mx.eval(list(arrays.values()))
                hot["order"], hot["arrays"] = order, arrays
                self._expert_cache_bytes = self._expert_row_bytes * sum(
                    len(table["order"]) for table in self._hot_tables.values()
                )
                temporary_bytes = nbytes_loaded
                table_order, table_arrays = order, arrays
            else:
                # The layer's resident set is full. Serve hot hits from the
                # cached table, merge in the just-loaded misses, but keep only
                # the active rows for this pass instead of duplicating the
                # whole hot table into a temporary over-cap table.
                self._hot_stats["overflow_passes"] += 1
                self._hot_stats["temporary_rows"] += len(selected_experts)
                table_order, table_arrays = self._gather_expert_table(
                    names, hot, batch.arrays, missing, selected_experts
                )
                refresh_order = (
                    choose_frequency_hot_order(
                        hot["order"],
                        missing,
                        counts,
                        self.expert_hot_set,
                    )
                    if should_refresh_hot_set(
                        hot["order"], missing, counts, self.REFRESH_HYSTERESIS
                    )
                    else hot["order"]
                )
                if set(refresh_order) != set(hot["order"]):
                    # Membership changed: rebuild the resident table. Pure
                    # reorderings are skipped — frequency counts permute the
                    # ranking almost every pass, and a rebuild copies 16 rows
                    # x 12 tensors for zero served-row difference (measured
                    # 328 rebuilds in 65 passes before this check).
                    hot["order"], hot["arrays"] = self._gather_expert_table(
                        names, hot, batch.arrays, missing, refresh_order
                    )
                    self._expert_cache_bytes = self._expert_row_bytes * sum(
                        len(table["order"]) for table in self._hot_tables.values()
                    )
                    self._hot_stats["refreshes"] += 1
                temporary_bytes = nbytes_loaded + self._expert_row_bytes * len(table_order)
            if pass_trace is not None:
                assembly_seconds = time.perf_counter() - assembly_started
                timed_sub_seconds += assembly_seconds
                pass_trace.add("expert_assembly", assembly_seconds)
        else:
            self._hot_stats["hit_passes"] += 1
            table_order, table_arrays = hot["order"], hot["arrays"]
            temporary_bytes = 0
        self._hot_stats["served_rows"] += len(selected_experts) - len(missing)

        if self._bound_tables.get(layer_index) is not table_arrays:
            for name in names:
                parts = name.split(".")
                setattr(getattr(mlp.experts, parts[-2]), parts[-1], table_arrays[name])
            self._bound_tables[layer_index] = table_arrays
        if table_arrays is hot["arrays"]:
            self._bound_temporary_layers.discard(layer_index)
        else:
            self._bound_temporary_layers.add(layer_index)
        self._set_external_resident_bytes(temporary_bytes)

        if pass_trace is not None:
            pass_trace.add(
                "expert_serve_python",
                time.perf_counter() - started - timed_sub_seconds,
            )
        if not self._trace:
            return table_order, None
        event = MlxStreamEvent(
            action="hot-set-experts" if not missing else "hot-set-experts-miss",
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names if missing else (),
            nbytes_loaded=nbytes_loaded,
        )
        self.session.events.append(event)
        return table_order, event

    def _gather_expert_table(
        self,
        names: tuple[str, ...],
        hot: dict[str, Any],
        missing_arrays: dict[str, Any],
        missing: list[int],
        members: list[int],
    ) -> tuple[list[int], dict[str, Any]]:
        """Assemble a table containing ``members`` as hot-rows-first.

        Membership is what matters — remap adapts to any row order — so the
        table is built with two gathers (one from the resident hot table, one
        from the just-loaded miss batch) instead of per-row slice+concat.
        Returns the actual row order used and the materialized arrays.
        """

        import mlx.core as mx

        hot_position = {expert: index for index, expert in enumerate(hot["order"])}
        missing_position = {expert: index for index, expert in enumerate(missing)}
        hot_members = [expert for expert in members if expert in hot_position]
        cold_members = [expert for expert in members if expert not in hot_position]
        hot_idx = (
            mx.array([hot_position[expert] for expert in hot_members])
            if hot_members
            else None
        )
        cold_idx = (
            mx.array([missing_position[expert] for expert in cold_members])
            if cold_members
            else None
        )
        order = hot_members + cold_members
        arrays: dict[str, Any] = {}
        for name in names:
            parts = []
            if hot_idx is not None:
                parts.append(mx.take(hot["arrays"][name], hot_idx, axis=0))
            if cold_idx is not None:
                parts.append(mx.take(missing_arrays[name], cold_idx, axis=0))
            arrays[name] = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        mx.eval(list(arrays.values()))
        return order, arrays

    def _expert_telemetry(self) -> dict[str, Any] | None:
        weight_pages = self.session.loader.weight_page_summary()
        native_resident = {
            "resident_layers": len(self._native_loaded),
            "requested_layers": len(self.native_layers),
            "resident_bytes": self._native_resident_bytes,
        }
        prefetch_stats = (
            self._prefetch_stats.to_dict()
            if self.expert_prefetch != "off"
            else None
        )
        if self.expert_hot_set <= 0:
            if (
                weight_pages is None
                and not self.native_layers
                and self.expert_prefetch == "off"
            ):
                return None
            return {
                "mode": "resident-sidecars",
                "native_layers": native_resident,
                "weight_page_budget_bytes": self.weight_page_budget_bytes,
                "weight_pages": weight_pages,
                "weight_page_policy": self.weight_page_policy,
                "weight_page_rows": self.weight_page_rows,
                "expert_prefetch_mode": self.expert_prefetch,
                "expert_prefetch": prefetch_stats,
            }
        stats = dict(self._hot_stats)
        total_rows = stats["served_rows"] + stats["loaded_rows"]
        return {
            "mode": "expert-hot-set",
            "rows_cap_per_layer": self.expert_hot_set,
            "resident_rows": sum(len(table["order"]) for table in self._hot_tables.values()),
            "resident_expert_bytes": self._expert_cache_bytes,
            "native_layers": native_resident,
            "weight_pages": weight_pages,
            "weight_page_budget_bytes": self.weight_page_budget_bytes,
            "weight_page_policy": self.weight_page_policy,
            "weight_page_rows": self.weight_page_rows,
            "expert_prefetch_mode": self.expert_prefetch,
            "expert_prefetch": prefetch_stats,
            "row_hit_rate": stats["served_rows"] / total_rows if total_rows else 0.0,
            **stats,
        }

    def _clear_selected_experts(self, layer_index: int, layer: Any) -> None:
        import mlx.core as mx

        if self.expert_hot_set > 0 and layer_index not in self._bound_temporary_layers:
            # The bound table IS the resident hot table: rebinding zero
            # placeholders here and the real table next pass was pure churn
            # (24 layers x 12 tensors per pass) and released no memory.
            return
        self._bound_temporary_layers.discard(layer_index)
        self._bound_tables.pop(layer_index, None)
        manifest = self.session.loader.manifest
        for name in self._expert_slice_names(layer_index):
            record = manifest.tensors[name]
            parts = name.split(".")
            module = getattr(layer.mlp.experts, parts[-2])
            empty_shape = (0,) * len(record.shape)
            setattr(module, parts[-1], mx.zeros(empty_shape, dtype=mlx_dtype(record.dtype)))
        self._set_external_resident_bytes()

    def _logits_from_hidden(self, hidden: Any, events: list[dict[str, Any]]) -> Any:
        import mlx.core as mx

        if self.pin_policy == "phase" and not self.warm_output:
            before_output_load = getattr(self, "_before_phase_output_load", None)
            if before_output_load is not None:
                before_output_load(events)
            load_output = self.session.load_output()
            if self._trace:
                events.append({"kind": "load", **compact_event(load_output)})

        logits_started = time.perf_counter()
        inner = self.session.model.model
        hidden = inner.norm(hidden)
        logits = self.session.model.lm_head(hidden)
        if self._pass_trace is not None:
            self._pass_trace.add("logits_eval", time.perf_counter() - logits_started)
        if self._trace:
            events.append(
                {
                    "kind": "compute",
                    "action": "logits",
                    "seconds": time.perf_counter() - logits_started,
                    "logits_shape": list(logits.shape),
                }
            )
        if self.pin_policy == "phase" and not self.warm_output:
            evict_output = self.session.evict_output()
            if self._trace:
                events.append({"kind": "evict", **compact_event(evict_output)})
        return logits


class NemotronHotsetMiss(Exception):
    """A routed expert fell outside the resident hot-set (strict mode).

    Carries the offending global expert ids so callers can decide whether to
    refresh the hot-set or abort. Raised only in ``strict`` mode; a lenient
    adapter would instead drop / substitute the miss.
    """

    def __init__(self, missing_experts: tuple[int, ...]) -> None:
        self.missing_experts = tuple(missing_experts)
        super().__init__(
            f"routed experts outside hot-set: {self.missing_experts}"
        )


class NemotronHotsetGateAdapter:
    """Wrap a NemotronH ``MoEGate`` so it emits LOCAL (remapped) expert ids.

    The stock gate returns GLOBAL router ids over all ``n_routed_experts``. When
    the MoE layer runs against a COMPACT switch_mlp (only the selected/hot-set
    experts, in ``hotset`` order), those global ids must be remapped to row
    positions in the compact table. This adapter performs that remap on every
    call while passing routing SCORES through unchanged, so the downstream
    weighted-sum is bit-identical to stock.

    ``hotset`` is the ordered list of global expert ids resident in the compact
    table (slot i holds global id ``hotset[i]``). In ``strict`` mode a routed id
    not present in ``hotset`` raises ``NemotronHotsetMiss``; the lenient form is
    left to higher layers (substitution policy lives elsewhere).
    """

    def __init__(self, gate: Any, hotset: list[int], *, strict: bool = True) -> None:
        self._gate = gate
        self.hotset = list(hotset)
        self.strict = strict
        self._global_to_local = {
            int(global_id): slot for slot, global_id in enumerate(self.hotset)
        }

    @property
    def n_routed_experts(self) -> int:
        return getattr(self._gate, "n_routed_experts", len(self.hotset))

    def __call__(self, x: Any) -> tuple[Any, Any]:
        indices, scores = self._gate(x)

        values = np.asarray(indices)
        if self.strict:
            seen = {int(v) for v in values.reshape(-1)}
            missing = tuple(sorted(seen - set(self._global_to_local)))
            if missing:
                raise NemotronHotsetMiss(missing)

        # The strict membership check above guarantees every routed id is in
        # ``self._global_to_local``, so the helper's in-range assumption holds.
        return remap_expert_indices_to_slots(values, self._global_to_local), scores


class NemotronDeferredRemapGate:
    """On-GPU global->slot remap gate for the DEFERRED (no per-layer eval) path.

    Same contract as :class:`NemotronHotsetGateAdapter` — wrap a NemotronH
    ``MoEGate`` and emit LOCAL (compact-table) expert ids while passing routing
    SCORES through unchanged — but it does the remap entirely ON-GPU and LAZILY:
    no ``np.asarray``, no Python membership check, no ``mx.eval``. That is what
    lets the whole 108-layer forward compose into ONE graph evaluated once at
    ``lm_head`` (the synced adapter forces a sync per MoE layer).

    Mechanism. The stock gate returns GLOBAL ids ``inds`` (lazy). We gather the
    prebuilt ``g2s`` lookup: ``local = mx.take(g2s, inds)``. For a RESIDENT expert
    ``local`` is its row in the fixed stack; for a COLD expert ``g2s`` holds the
    SENTINEL (== K, one past the last valid slot). We:

    * record a LAZY per-token cold contribution — ``mx.max(local == sentinel)``
      appended to the runner's shared ``cold_flags`` list — so the forward can
      OR-reduce "did ANY layer route to a non-resident expert?" with ZERO syncs;
    * CLAMP ``local`` to ``[0, K-1]`` (``mx.minimum``) before returning it, so the
      native ``SwitchMLP`` gather can never index out of the fixed stack. On a
      cold token the clamped gather produces a WRONG row, but the whole optimistic
      forward is discarded and the token redone on the exact path, so the bogus
      value never reaches the output.

    EXACTNESS (all-resident token): every routed id is resident, so ``g2s`` maps
    each ``inds`` entry to exactly the slot the synced numpy remap
    (:func:`remap_expert_indices_to_slots`) would, the clamp is a no-op (all slots
    already < K), and ``scores`` are the untouched stock scores. The native
    ``SwitchMLP`` therefore gathers the SAME rows with the SAME weights — bit-
    identical to the synced fixed path (proven == stock == page).
    """

    def __init__(
        self,
        gate: Any,
        g2s: Any,
        sentinel: int,
        cold_flags: list[Any],
        force_top_k: Any = None,
    ) -> None:
        self._gate = gate
        self._g2s = g2s          # [n_routed_experts] int32 global->slot lookup
        self._sentinel = int(sentinel)  # cold marker (== K resident rows)
        self._cold_flags = cold_flags   # runner-shared lazy cold accumulator
        # LOSSY top-K routing truncation. May be an int K, ``None`` (OFF), or a
        # zero-arg CALLABLE returning ``int | None`` so the owning runner can flip
        # K between decode runs on one model load (the sweep harness). Resolved
        # per ``__call__``; see :meth:`_resolve_force_top_k`.
        self._force_top_k = force_top_k
        # Stock-gate normalization parameters, captured for the K-subset renorm.
        # ``norm_topk_prob`` decides WHETHER to renormalize; ``routed_scaling_factor``
        # is the post-norm sum the stock top-K gate targets (its normalized scores
        # sum to ``routed_scaling_factor``). Default to a no-op renorm if absent.
        self._norm_topk_prob = bool(getattr(gate, "norm_topk_prob", False))
        self._routed_scaling_factor = float(
            getattr(gate, "routed_scaling_factor", 1.0)
        )

    @property
    def n_routed_experts(self) -> int:
        return getattr(self._gate, "n_routed_experts", int(self._g2s.shape[0]))

    def _resolve_force_top_k(self) -> int | None:
        k = self._force_top_k
        if callable(k):
            k = k()
        if k is None:
            return None
        return int(k)

    def _truncate_top_k(self, inds: Any, scores: Any, k: int) -> tuple[Any, Any]:
        """Keep the K HIGHEST-score routed experts and renorm like the stock gate.

        ``inds``/``scores`` are the stock gate's routed top-``num_experts_per_tok``
        (scores already ``norm_topk_prob``-normalized * ``routed_scaling_factor``).
        We argpartition the K largest scores, gather those K (inds & scores), and —
        when the stock gate normalizes — re-divide the K scores by their own sum and
        re-apply ``routed_scaling_factor``. That reproduces, on the K subset, EXACTLY
        what ``group_expert_select`` would emit with ``top_k=K``: the stock
        normalized score is ``orig/sum_routed * rsf``; renormalizing the kept K
        collapses to ``orig/sum_K * rsf`` (the routed-sum and intermediate rsf
        cancel) — the stock top-K result. With ``norm_topk_prob`` off the stock gate
        does not normalize, so we keep the K scores unchanged.
        """
        import mlx.core as mx

        # Indices of the K largest scores along the routed axis. ``argpartition``
        # with kth=-K puts the K largest in the last K positions (lazy, on-GPU).
        top = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
        inds = mx.take_along_axis(inds, top, axis=-1)
        scores = mx.take_along_axis(scores, top, axis=-1)
        if self._norm_topk_prob:
            denom = scores.sum(axis=-1, keepdims=True)
            scores = scores / (denom + 1e-20) * self._routed_scaling_factor
        return inds, scores

    def __call__(self, x: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        inds, scores = self._gate(x)
        # LOSSY top-K truncation BEFORE the global->slot remap, so the cold-buddy
        # substitution baked into ``g2s`` still applies to the K kept ids (the
        # deferred single-graph + cold-buddy path is untouched; only the routing
        # mask narrows). No-op when K is None or K >= the routed count.
        k = self._resolve_force_top_k()
        if k is not None and k < int(inds.shape[-1]):
            inds, scores = self._truncate_top_k(inds, scores, k)
        # ON-GPU global->slot remap (lazy; no numpy, no eval).
        local = mx.take(self._g2s, inds)
        # Lazy "this layer saw a cold expert" flag (1 if any routed id is cold).
        # Reduced to a scalar so the per-token OR over layers is cheap; appended
        # to the runner-shared list the forward OR-reduces at end-of-token.
        self._cold_flags.append(mx.max((local == self._sentinel).astype(mx.int32)))
        # Clamp the sentinel to a valid slot so the gather can't OOB. Resident
        # slots are already < sentinel, so this is a no-op for them; only the
        # (discarded) cold rows are affected.
        local = mx.minimum(local, self._sentinel - 1)
        return local, scores


class NemotronFusedRMSNormGate:
    """Drop-in for a Mamba2 mixer's ``MambaRMSNormGated`` that fuses its glue ops.

    The stock ``MambaRMSNormGated.__call__`` is a chain of tiny Metal dispatches
    (swiglu gate, reshape, ``mx.fast.rms_norm``, flatten, weight-scale) — pure
    launch overhead at decode. This wrapper captures the original norm module
    (for its resident ``weight``, ``eps``, and ``group_size``) and routes the
    call through :func:`smarttensor.ssm_glue.fused_mamba_rmsnorm_gated`, one
    dispatch matching the stock math to ~1 fp32 ULP (gated by ``test_ssm_glue``).

    Installed ONCE per Mamba2 layer by the same component-swap the runner uses
    for :class:`NemotronDeferredRemapGate`: ``mixer.norm`` is replaced and the
    original stashed for restore. The captured ``norm`` keeps the module
    reachable for weight loads (we read ``self._norm.weight`` lazily per call, so
    a base load that refreshes the weight is still picked up).
    """

    def __init__(self, norm: Any) -> None:
        self._norm = norm

    @property
    def weight(self) -> Any:
        return self._norm.weight

    @property
    def eps(self) -> float:
        return self._norm.eps

    @property
    def group_size(self) -> int:
        return self._norm.group_size

    def __call__(self, x: Any, gate: Any = None) -> Any:
        from smarttensor.ssm_glue import fused_mamba_rmsnorm_gated

        weight = self._norm.weight
        group_size = int(self._norm.group_size)
        n_groups = int(weight.shape[0]) // group_size
        return fused_mamba_rmsnorm_gated(
            x, gate, weight, n_groups=n_groups, eps=self._norm.eps
        )


class NemotronHStreamingForwardRunner(StreamingChatRunner):
    """Generate with NemotronH (hybrid Mamba-2 / MoE / attention) via streaming.

    Skeleton: constructor, model-type gate, and block-plan/cache wiring only.
    The forward path (``_stream_forward_tokens`` / ``forward_logits`` /
    ``_moe_forward_selective``) lands in later tasks.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        evaluate: bool = True,
        retain_layers: set[int] | None = None,
        resident_budget_bytes: int | None = None,  # accepted for signature parity; wired in a later task (budget-aware base selection)
        backend: str = "native",
        clear_on_evict: bool = True,
        pin_policy: str = "all",
        warm_embeddings: bool = False,
        page_experts: bool = False,
        weight_page_budget_bytes: int | None = None,
        weight_page_policy: str = "auto",
        weight_page_rows: int = 1,
        persist_expert_tables: bool = False,  # default OFF: measured net-slower on real generation (membership churn + table-size-scaling concatenate); kept behind the flag pending the slot-based GPU-gather redesign
        max_resident_experts_per_layer: int | None = 160,
        fixed_hotset_experts: int | None = None,  # K for build_fixed_hotset; None = feature OFF (page path unchanged)
        cold_substitution: bool = False,  # NEAR-exact: map cold experts to resident buddies (no redo) instead of the sentinel; default OFF = byte-identical exact path
        cold_tier_bits: int | None = None,  # NEAR-exact: keep the cold-tail experts RESIDENT at this coarser bit-width (2/3) so they are served from RAM (no disk redo, no foreign buddy) at bounded re-quant drift; None = OFF
        cold_tier_group_size: int | None = None,  # re-quant group size for the cold tier (None = the model's own group_size); a COARSER size (64/128) shrinks the per-group scale/bias overhead -> a bigger fit win
        fuse_ssm_norm_gate: bool = False,  # Metal glue-fusion: replace each Mamba2 mixer's gated RMSNorm with one fused kernel (cuts ~6 dispatches/Mamba-layer). Exactness-gated (~1 fp32 ULP); default OFF = stock path.
        force_top_k: int | None = None,  # LOSSY routing truncation: keep only the K highest-gate-score experts of the routed num_experts_per_tok (renormalized like norm_topk_prob on the K subset) -> gather K instead of all-routed = the real bandwidth lever. None = OFF (full routing). No-op when K >= the routed count.
    ) -> None:
        if weight_page_rows < 1:
            raise ValueError("weight_page_rows must be positive")
        if force_top_k is not None and force_top_k < 1:
            raise ValueError("force_top_k must be positive")
        if cold_tier_bits is not None and cold_tier_bits not in (2, 3, 4, 5, 6, 8):
            # mx.quantize affine supports {2,3,4,5,6,8}; the lever's intent is a
            # COARSER copy than the 4-bit resident tier, but we permit any valid
            # width so the same path can hold e.g. a 3-bit tier above a 2-bit one.
            raise ValueError(
                "cold_tier_bits must be one of {2,3,4,5,6,8} (mx.quantize affine)"
            )
        if cold_tier_group_size is not None and cold_tier_group_size not in (32, 64, 128):
            raise ValueError("cold_tier_group_size must be one of {32,64,128}")
        if (
            max_resident_experts_per_layer is not None
            and max_resident_experts_per_layer < 1
        ):
            raise ValueError("max_resident_experts_per_layer must be positive")
        if fixed_hotset_experts is not None and fixed_hotset_experts < 1:
            raise ValueError("fixed_hotset_experts must be positive")
        if weight_page_rows != 1 and weight_page_budget_bytes is None:
            raise ValueError("weight_page_rows requires weight_page_budget_bytes")
        # Gate on model_type BEFORE building a session so the negative path
        # never instantiates a loader or touches weights (cheap rejection).
        config = load_mlx_config(Path(model_dir))
        if config.get("model_type") != "nemotron_h":
            raise ValueError(
                "NemotronHStreamingForwardRunner supports model_type=nemotron_h only"
            )
        from smarttensor.nemotron_layout import plan_blocks

        self.session = MlxModelSession(
            model_dir,
            evaluate=evaluate,
            retain_layers=retain_layers,
            resident_budget_bytes=None,
            backend=backend,
            clear_on_evict=clear_on_evict,
            pin_policy=pin_policy,
            warm_embeddings=warm_embeddings,
        )
        self.pin_policy = pin_policy
        self.warm_embeddings = warm_embeddings
        self.page_experts = page_experts
        weight_page_policy = resolve_weight_page_policy(
            "nemotron_h",
            weight_page_policy,
            has_weight_page_budget=weight_page_budget_bytes is not None,
        )
        self.weight_page_budget_bytes = weight_page_budget_bytes
        self.weight_page_policy = weight_page_policy
        self.weight_page_rows = weight_page_rows
        if weight_page_budget_bytes is not None:
            self.session.loader.attach_weight_page_cache(
                weight_page_budget_bytes,
                eviction_policy=weight_page_policy,
                rows_per_page=weight_page_rows,
            )
        self._plan = plan_blocks(self.session.config["layers_block_type"])
        self.loader_executor = ThreadPoolExecutor(max_workers=1)
        self.base_retain_layers: set[int] = set(retain_layers or set())
        self._expert_history: dict[int, list[int]] = {}

        # --- Persistent per-layer assembled expert table ---------------------
        # Warm decode spent ~83% of every token re-concatenating the compact
        # switch_mlp table from per-(layer,expert) pages, even though routing
        # reuses experts ~87% of the time. We keep a PERSISTENT per-layer table
        # of assembled expert weights and reuse it across tokens, appending only
        # the experts a new token's routing leaves the resident membership for.
        #
        # ``_assembled_tables[layer]`` =
        #     {"order":  [global expert ids currently assembled, in row order],
        #      "arrays": {"fc1": {field: tensor}, "fc2": {field: tensor}}}
        # where each ``arrays[proj][field]`` is the concatenation of that field's
        # rows for the experts in ``order`` (row i holds global expert order[i]).
        # The gate adapter remaps routed globals -> these row positions, so the
        # block forward against the assembled table is bit-identical to a freshly
        # rebuilt compact table — the math is unchanged, the assembly is reused.
        self.persist_expert_tables = persist_expert_tables
        self.max_resident_experts_per_layer = max_resident_experts_per_layer
        self._assembled_tables: dict[int, dict[str, Any]] = {}
        # Per-layer LRU recency: global expert id -> monotonically increasing
        # use stamp (higher = more recent). Drives eviction when ``order`` would
        # exceed ``max_resident_experts_per_layer``.
        self._assembled_lru: dict[int, dict[int, int]] = {}
        self._assembled_clock = 0
        self._assembled_stats = {
            "hit_passes": 0,      # routing was a subset of residency: no load
            "append_passes": 0,   # loaded only the missing experts, appended
            "loaded_experts": 0,  # total expert rows read off disk (misses)
            "evictions": 0,       # experts dropped to honor the per-layer cap
        }

        # --- Fixed per-layer resident hot-set + native on-GPU gather ----------
        # The persistent-table path above amortizes assembly but still pays an
        # O(K) concatenate whenever routing leaves the resident membership, and
        # on the real model membership churns ~63%/pass so that append fires
        # constantly. The FIXED hot-set instead builds, ONCE, a per-(MoE-layer)
        # stacked switch_mlp of the top-K most-frequent experts and NEVER updates
        # it per token. Per token, when the routed experts are a SUBSET of the
        # fixed resident set, we swap the prebuilt stacked arrays into
        # ``switch_mlp`` by REFERENCE (no concatenate) and run the model's own
        # native ``NemotronHMoE.__call__`` — one on-GPU gather, zero Python
        # assembly. The rare cold pass (a routed expert not resident) falls back
        # to the existing exact page path. ``fixed_hotset_experts`` is K; the
        # feature is OFF (page path unchanged) until ``build_fixed_hotset`` runs.
        #
        # ``_fixed_hotset[layer]`` =
        #     {"order": [global expert ids, in row order],
        #      "arrays": {"fc1": {field: tensor}, "fc2": {field: tensor}},
        #      "global_to_slot": {global id: row position}}
        # Row i of every ``arrays[proj][field]`` holds global expert ``order[i]``,
        # exactly the layout ``_load_selected_experts`` produces — so the gate
        # adapter remaps global -> row identically and the forward is exact.
        self.fixed_hotset_experts = fixed_hotset_experts
        # NEAR-EXACT cold-expert BUDDY substitution (opt-in; default OFF).
        #
        # On the deferred path a routed expert outside the resident fixed set maps
        # to the SENTINEL, which trips the on-GPU cold flag and forces a full-token
        # rollback + exact synced redo. On FRESH generation (and speculation's wide
        # verify blocks) routed experts go cold often, so the redos dominate and the
        # fast ~16 tok/s deferred rate is lost.
        #
        # With ``cold_substitution`` on, ``build_fixed_hotset`` instead maps each
        # cold expert to its nearest RESIDENT *buddy* slot (the resident expert with
        # the most-similar gate-weight row — see ``_compute_buddy_map``). So every
        # global id maps to a VALID resident slot, the remap gate's ``local ==
        # sentinel`` signal NEVER fires, and the deferred path never redoes -> fast
        # on ANY generation. The routed cold expert's gate SCORE is unchanged; only
        # WHICH expert computes that slot changes (cold -> buddy), so the output
        # DRIFTS slightly from exact on tokens that route to a cold expert (the
        # user-accepted ~0.1% PPL tradeoff). All-resident tokens are still EXACT
        # (substitution is a no-op for resident ids). Default OFF keeps the path
        # byte-identical to the proven exact deferred/fixed forward.
        self.cold_substitution = bool(cold_substitution)
        # --- Mixed-precision COLD-EXPERT tier (opt-in; default OFF). ----------
        # A resident-fit lever SEPARATE from buddy substitution. Substitution maps a
        # cold expert to a DIFFERENT resident expert (changes routing). The cold tier
        # instead keeps the ACTUAL cold-tail experts RESIDENT at a COARSER bit-width
        # (``cold_tier_bits`` in {2,3,...}), re-quantized off their own 4-bit on-disk
        # weights. So a routed cold expert is computed by ITS OWN weights, just on a
        # coarser quantization grid: the drift is the EXTRA re-quant error of the
        # coarse grid (``(Q_lo(W) - Q_hi(W)) @ x``), not a foreign-expert swap, and
        # there is NO disk reload (the tier lives in RAM). Because a 2-bit gs128
        # expert is ~42% of a 4-bit gs32 one, the tier fits the cold tail in far less
        # RAM than the 4-bit hot set -- the fit win. Built per MoE layer in
        # ``build_fixed_hotset`` (see ``_build_cold_tier_stack``) and gathered by a
        # dedicated branch in the synced fixed forward. Default OFF (None) keeps the
        # path byte-identical to the proven exact fixed/page forward.
        self.cold_tier_bits = cold_tier_bits
        self.cold_tier_group_size = cold_tier_group_size
        self.fuse_ssm_norm_gate = bool(fuse_ssm_norm_gate)
        self._fused_ssm_norm_installed = False
        # LOSSY top-K routing truncation (deferred path). None = OFF (full
        # num_experts_per_tok routing). When set to K and K < the routed count,
        # the deferred remap gate keeps only the K highest-gate-score experts,
        # renormalizes their scores like the stock ``norm_topk_prob`` on the K
        # subset, and gathers K experts -> fewer expert-GEMMs = bandwidth win.
        # Read live by the gate wrappers at install time. Mutable post-ctor so a
        # sweep can flip K between decode runs on one model load.
        self.force_top_k = force_top_k
        self._fixed_hotset: dict[int, dict[str, Any]] = {}
        self._hotset_stats = {
            "native_hits": 0,     # MoE passes served from the fixed resident set
            "cold_fallbacks": 0,  # MoE passes with a routed expert not resident
        }
        self._cold_tier_stats = {
            # MoE passes whose cold-routed experts were ALL served from the
            # resident low-bit cold tier (so the slow disk-reload page fallback was
            # avoided). >0 means the tier genuinely fired.
            "cold_tier_hits": 0,
            # Cold experts held resident at low precision, summed over MoE layers
            # (set once by build_fixed_hotset; constant thereafter).
            "cold_tier_experts": 0,
            # Bytes the low-bit cold tier occupies vs what those experts WOULD cost
            # at 4-bit -- the recorded fit win (set once at build).
            "cold_tier_bytes": 0,
            "cold_tier_bytes_at_4bit": 0,
        }
        # --- Deferred-eval decode (the next lever on top of the fixed hot-set).
        # The synced fixed path is token-exact but still evals once PER MoE layer
        # (to read routed ids for a Python residency check + a numpy global->slot
        # remap), serialising the GPU. The deferred path removes every per-layer
        # eval: the global->slot remap is done ON-GPU (``mx.take`` of a prebuilt
        # ``g2s`` lookup, lazy, no numpy), residency is checked ON-GPU (a lazy OR
        # of per-layer "saw a cold expert" flags), and the whole forward composes
        # into ONE graph evaluated once at ``lm_head``. A cold expert can't be
        # caught without a sync, so the token is run OPTIMISTICALLY; one eval at
        # end-of-token reads the cold flag, and if set the cache is rolled back
        # and the token REDONE via the proven-exact synced fixed path. The fixed
        # stacks + on-GPU remap gates are installed into ``switch_mlp`` ONCE (see
        # :meth:`_install_deferred_hotset`) so there is no per-layer Python swap.
        self._deferred_installed = False
        # --- Deferred decode HOT PATH (per-token Python-overhead strip) --------
        # On top of the per-layer-eval removal above, the steady-state deferred
        # decode loop still pays pure Python per layer per token: the 108-iter
        # layer loop calls ``_load_layer_full`` / ``_load_layer_base`` ->
        # ``_load_into_model`` on EVERY layer (a no-op moving ZERO bytes once the
        # model is resident), and appends a telemetry dict per layer. x108 layers
        # x every token that overhead gates kernel dispatch and starves the GPU.
        # When ``hot_decode`` is on AND the caller passes ``_hot=True`` (only the
        # deferred decode loop, only after a warmup forward has made every layer
        # resident), the layer loop SKIPS the loader call and the per-layer event
        # appends. ``run_summary`` only reads ``kind=='load'`` byte/second totals,
        # which are ZERO on the resident decode path, so no reported metric moves
        # and the forward output is byte-identical (the weights are already in the
        # model regardless of whether the no-op loader is called). Default ON; the
        # default / paged / synced paths never pass ``_hot=True`` so are untouched.
        self.hot_decode = True
        # Lazy per-layer cold contributions for the CURRENT optimistic token; the
        # remap gates append to this and the forward OR-reduces it into one flag.
        self._deferred_cold_flags: list[Any] = []
        self._deferred_stats = {
            "deferred_tokens": 0,  # tokens served purely by the deferred fast path
            "cold_redos": 0,       # tokens with a cold layer -> rolled back + redone
            # Cold experts mapped to a resident BUDDY at build (summed over MoE
            # layers). >0 means ``cold_substitution`` genuinely fired; with it on
            # this is the count of remappings that keep the deferred path off the
            # redo. Set once by ``build_fixed_hotset``; constant thereafter.
            "cold_substitutions": 0,
        }

    def close(self) -> None:
        self.loader_executor.shutdown(wait=True)
        self.session.close()

    def _load_layer_full(self, layer_index: int) -> MlxStreamEvent:
        """Load every tensor of a backbone layer (base + any experts) resident."""
        layer = self.session.loader.manifest.layers[layer_index]
        return self.session._load_into_model(
            layer.tensor_names, action="load-layer-full", layer=layer_index
        )

    def _ensure_non_layer_weights(self) -> None:
        """Make embeddings, final norm_f, and lm_head resident.

        ``_load_layer_full`` only covers per-layer tensors; the backbone
        embedding table, the final ``norm_f``, and the output ``lm_head`` are
        unlayered and must be loaded explicitly for an exact forward.
        """
        non_layer = tuple(self.session.loader.manifest.unlayered_tensors())
        if non_layer:
            self.session._load_into_model(
                non_layer, action="load-non-layer", layer=None
            )

    def _load_layer_base(self, layer_index: int) -> MlxStreamEvent:
        """Load a layer's BASE tensors (everything except routed experts).

        For an MoE layer that means norm + gate (+ bias) + latent projections +
        shared experts: the per-token-cheap weights that stay resident. The
        stacked routed-expert table (``.mixer.switch_mlp.*`` — weight plus, when
        quantized, scales/biases) is excluded so the selective path can page just
        the experts a token actually routes to.
        """
        from smarttensor.nemotron_layout import partition_layer_tensors

        layer = self.session.loader.manifest.layers[layer_index]
        base, _experts = partition_layer_tensors(list(layer.tensor_names))
        return self.session._load_into_model(
            base, action="load-layer-base", layer=layer_index
        )

    def _expert_slice_names(self, layer_index: int) -> tuple[str, ...]:
        """Stacked switch_mlp tensors to first-dim-slice for selective load.

        On disk NemotronH stores routed experts STACKED (the model's own
        ``SwitchMLP`` params), not per-expert: ``mixer.switch_mlp.fc1/fc2`` with
        shape ``(n_experts, out, in)`` on dim 0. For a QUANTIZED model each
        projection also carries ``.scales``/``.biases`` (also stacked over
        experts on dim 0), so all three are sliced by the SAME expert-row
        indices. ``Model.sanitize``'s per-expert stacking loop is a no-op for
        this layout (it only fires for the legacy ``experts.{e}.*`` names).
        """
        prefix = f"backbone.layers.{layer_index}.mixer.switch_mlp"
        tensors = self.session.loader.manifest.tensors
        # Only request the sibling tensors that actually exist on disk: a
        # quantized layer has weight+scales+biases, an unquantized one only
        # weight, and some affine exports may omit all-zero biases. Driving off
        # the manifest avoids a KeyError on a missing field for the real model.
        candidates = (
            f"{prefix}.{projection}.{field}"
            for projection in ("fc1", "fc2")
            for field in ("weight", "scales", "biases")
        )
        return tuple(name for name in candidates if name in tensors)

    def _load_selected_experts(
        self, layer_index: int, selected: list[int]
    ) -> tuple[dict[str, Any], MlxStreamEvent]:
        """Load ONLY ``selected`` experts from disk into a compact fc1/fc2 table.

        Slices the SELECTED first-dim rows (expert indices) of the STACKED
        ``switch_mlp.fc1``/``fc2`` tensors via the Qwen mechanism
        (``load_first_dim_slices``). For a QUANTIZED layer that means slicing
        ``weight``, ``scales`` AND ``biases`` (all stacked over experts on dim 0);
        for an unquantized layer just ``weight``. The result is a compact stacked
        table whose row i is global expert ``selected[i]`` (so the gate adapter
        remaps global -> local = row position). This is a TRUE selective load:
        unselected expert rows are never read off disk.

        Returns ``(compact_arrays, event)`` where ``compact_arrays`` maps
        ``"fc1"``/``"fc2"`` to a dict of ``{field: tensor}`` (``field`` is
        ``weight`` and, when quantized, ``scales``/``biases``).
        """
        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        batch = self.session.loader.load_first_dim_slices(
            names, selected, evaluate=self.session.evaluate
        )

        prefix = f"backbone.layers.{layer_index}.mixer.switch_mlp"
        # Build the compact table from exactly the tensors that were sliced
        # (derived from the manifest above), so weight-only / weight+scales /
        # weight+scales+biases all work without a hard-coded field list.
        compact: dict[str, dict[str, Any]] = {"fc1": {}, "fc2": {}}
        for name in names:
            projection, field = name[len(prefix) + 1 :].split(".")
            compact[projection][field] = batch.arrays[name]

        event = MlxStreamEvent(
            action="load-selected-experts",
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names,
            nbytes_loaded=batch.nbytes,
        )
        return compact, event

    def _serve_experts_from_assembled_table(
        self, layer_index: int, selected: list[int]
    ) -> tuple[dict[str, dict[str, Any]], list[int], MlxStreamEvent | None]:
        """Serve ``selected`` from the layer's PERSISTENT assembled expert table.

        The decode-hot path. Adapts the Qwen ``_serve_experts_from_hot_set``
        template to the NemotronH stacked ``switch_mlp`` layout:

        * If ``selected`` is already a subset of the table's membership, NOTHING
          is loaded or rebuilt — the persistent arrays are returned as-is (the
          ~87% common case this whole change exists to make cheap).
        * Otherwise only the MISSING experts are read off disk (one
          ``load_first_dim_slices`` over the misses) and APPENDED to the table by
          a single concatenate per field, extending membership in place.
        * A per-layer membership cap (``max_resident_experts_per_layer``) bounds
          growth: when appending would exceed it, the LEAST-recently-used experts
          that the CURRENT token does not route to are evicted and the table is
          re-gathered. Experts in ``selected`` are never evicted, so the strict
          gate adapter can always remap every routed id.

        Returns ``(arrays, order, event)`` where ``arrays`` maps
        ``"fc1"``/``"fc2"`` -> ``{field: tensor}`` (the assembled, reused table),
        ``order`` is the global expert id per row, and ``event`` is a load event
        only when disk was touched (``None`` on a pure residency hit).
        """
        import mlx.core as mx

        self._assembled_clock += 1
        clock = self._assembled_clock
        table = self._assembled_tables.setdefault(
            layer_index, {"order": [], "arrays": {"fc1": {}, "fc2": {}}}
        )
        recency = self._assembled_lru.setdefault(layer_index, {})

        member_set = set(table["order"])
        missing = [expert for expert in selected if expert not in member_set]

        # Touch every routed expert so LRU reflects this token's usage (also
        # protects them from being evicted in the same pass — see below).
        for expert in selected:
            recency[expert] = clock

        if not missing:
            self._assembled_stats["hit_passes"] += 1
            return table["arrays"], list(table["order"]), None

        # --- Membership extension: load ONLY the missing experts, append once.
        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        compact, load_event = self._load_selected_experts(layer_index, missing)
        self._assembled_stats["append_passes"] += 1
        self._assembled_stats["loaded_experts"] += len(missing)

        if table["order"]:
            arrays = {
                "fc1": {
                    field: mx.concatenate(
                        [table["arrays"]["fc1"][field], compact["fc1"][field]],
                        axis=0,
                    )
                    for field in compact["fc1"]
                },
                "fc2": {
                    field: mx.concatenate(
                        [table["arrays"]["fc2"][field], compact["fc2"][field]],
                        axis=0,
                    )
                    for field in compact["fc2"]
                },
            }
        else:
            arrays = {
                "fc1": dict(compact["fc1"]),
                "fc2": dict(compact["fc2"]),
            }
        order = table["order"] + missing

        # --- Eviction: honor the per-layer cap by dropping LRU non-routed rows.
        cap = self.max_resident_experts_per_layer
        if cap is not None and len(order) > cap:
            order, arrays = self._evict_assembled_to_cap(
                order, arrays, selected, recency, cap
            )

        mx.eval(
            [arrays["fc1"][f] for f in arrays["fc1"]]
            + [arrays["fc2"][f] for f in arrays["fc2"]]
        )
        table["order"], table["arrays"] = order, arrays
        # Drop recency entries for any expert that was just evicted.
        resident = set(order)
        for expert in [e for e in recency if e not in resident]:
            recency.pop(expert, None)

        event = MlxStreamEvent(
            action="assemble-experts-append",
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names,
            nbytes_loaded=load_event.nbytes_loaded,
        )
        return arrays, list(order), event

    def _evict_assembled_to_cap(
        self,
        order: list[int],
        arrays: dict[str, dict[str, Any]],
        selected: list[int],
        recency: dict[int, int],
        cap: int,
    ) -> tuple[list[int], dict[str, dict[str, Any]]]:
        """Drop LRU experts (never one this token routes to) and re-gather.

        Keeps exactly ``cap`` experts: the ``selected`` set (which the strict
        gate MUST be able to remap) is pinned, the remaining slots go to the
        most-recently-used of the rest. The retained rows are gathered from the
        just-appended ``arrays`` with ``mx.take`` (identical bytes — exactness is
        unaffected: the same expert row is re-selected, never recomputed).
        """
        import mlx.core as mx

        protected = set(selected)
        if len(protected) > cap:
            # Pathological: more experts routed in one token than the cap. The
            # cap cannot be honored without dropping a routed expert (which would
            # break the strict gate), so keep all routed experts this pass.
            keep = [expert for expert in order if expert in protected]
        else:
            evictable = [expert for expert in order if expert not in protected]
            # Most-recently-used first; keep the top (cap - |protected|).
            evictable.sort(key=lambda expert: recency.get(expert, 0), reverse=True)
            keep_evictable = set(evictable[: cap - len(protected)])
            keep = [
                expert
                for expert in order
                if expert in protected or expert in keep_evictable
            ]

        self._assembled_stats["evictions"] += len(order) - len(keep)
        position = {expert: index for index, expert in enumerate(order)}
        idx = mx.array([position[expert] for expert in keep])
        gathered = {
            "fc1": {
                field: mx.take(arrays["fc1"][field], idx, axis=0)
                for field in arrays["fc1"]
            },
            "fc2": {
                field: mx.take(arrays["fc2"][field], idx, axis=0)
                for field in arrays["fc2"]
            },
        }
        return keep, gathered

    def _nemotron_moe_layer_forward_persistent(
        self,
        layer_index: int,
        layer: Any,
        x: Any,
        *,
        mask: Any,
        cache: Any,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        """MoE ('E') block forward against the PERSISTENT assembled expert table.

        Same exact mechanism as :meth:`_nemotron_moe_layer_forward` (gate ->
        select -> swap a stacked table into ``switch_mlp`` -> remap the gate ->
        run the block's own ``__call__`` -> restore), but the expert table is
        the reused per-layer assembly from
        :meth:`_serve_experts_from_assembled_table` instead of a table rebuilt
        from scratch every token. The arrays are the identical expert rows, so
        the forward is bit-identical; only the assembly is amortized.
        """
        import mlx.core as mx

        mixer = layer.mixer

        normed = layer.norm(x)
        inds, _scores = mixer.gate(normed)
        mx.eval(inds)
        selected = selected_expert_ids(inds)
        self._expert_history[layer_index] = selected

        arrays, order, load_event = self._serve_experts_from_assembled_table(
            layer_index, selected
        )
        event_kind = (
            "assemble-experts-append" if load_event is not None else "assemble-experts-hit"
        )
        event: dict[str, Any] = {
            "kind": "load" if load_event is not None else "compute",
            "action": event_kind,
            "pass": pass_kind,
            "token_step": token_step,
            "layer": layer_index,
            "expert_count": len(selected),
            "experts": selected,
            "resident_experts": len(order),
        }
        if load_event is not None:
            event.update(compact_event(load_event))
        events.append(event)

        # Swap the persistent table into switch_mlp.fc1/fc2 and remap the gate to
        # the table's membership order (row i holds global expert ``order[i]``).
        # Restore the originals afterwards so the table stays detached from the
        # module (it lives on the runner, reused next token).
        original_gate = mixer.gate
        saved: dict[str, dict[str, Any]] = {}
        for projection in ("fc1", "fc2"):
            module = getattr(mixer.switch_mlp, projection)
            saved[projection] = {
                field: module[field] for field in arrays[projection]
            }
        try:
            mixer.gate = NemotronHotsetGateAdapter(original_gate, order, strict=True)
            for projection in ("fc1", "fc2"):
                module = getattr(mixer.switch_mlp, projection)
                for field, value in arrays[projection].items():
                    setattr(module, field, value)
            return layer(x, mask=mask, cache=cache)
        finally:
            mixer.gate = original_gate
            for projection in ("fc1", "fc2"):
                module = getattr(mixer.switch_mlp, projection)
                for field, value in saved[projection].items():
                    setattr(module, field, value)

    def _nemotron_moe_layer_forward(
        self,
        layer_index: int,
        layer: Any,
        x: Any,
        *,
        mask: Any,
        cache: Any,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        """Run one MoE ('E') block against a COMPACT selected-expert table.

        Mechanism (exact): gate the normed input to learn which global experts
        the tokens route to (union over rows), page ONLY those experts into a
        compact switch_mlp, wrap the gate so it emits local row indices, then
        call the block's own ``__call__`` (which applies norm + residual +
        latent projections + shared experts exactly). Restore afterwards.
        """
        import mlx.core as mx

        mixer = layer.mixer

        # Gate on the same normed input the block will use, to discover the
        # routed (global) expert ids without touching the full expert table.
        normed = layer.norm(x)
        inds, _scores = mixer.gate(normed)
        mx.eval(inds)
        selected = selected_expert_ids(inds)
        self._expert_history[layer_index] = selected

        compact, load_event = self._load_selected_experts(layer_index, selected)
        events.append(
            {
                "kind": "load",
                "action": "load-selected-experts",
                "pass": pass_kind,
                "token_step": token_step,
                "layer": layer_index,
                "expert_count": len(selected),
                "experts": selected,
                **compact_event(load_event),
            }
        )

        # Swap the compact stacked table into switch_mlp.fc1/fc2, setting every
        # field the module carries: ``.weight`` always, plus ``.scales`` and
        # ``.biases`` when the module is a QuantizedSwitchLinear (detected via
        # "scales" in module). Restore the originals afterwards.
        original_gate = mixer.gate
        saved: dict[str, dict[str, Any]] = {}
        for projection in ("fc1", "fc2"):
            module = getattr(mixer.switch_mlp, projection)
            saved[projection] = {
                field: module[field] for field in compact[projection]
            }
        try:
            mixer.gate = NemotronHotsetGateAdapter(
                original_gate, selected, strict=True
            )
            for projection in ("fc1", "fc2"):
                module = getattr(mixer.switch_mlp, projection)
                for field, value in compact[projection].items():
                    setattr(module, field, value)
            return layer(x, mask=mask, cache=cache)
        finally:
            mixer.gate = original_gate
            for projection in ("fc1", "fc2"):
                module = getattr(mixer.switch_mlp, projection)
                for field, value in saved[projection].items():
                    setattr(module, field, value)

    def _moe_layer_indices(self) -> list[int]:
        """Backbone layer indices of the MoE ('E') blocks (in layer order)."""
        return [
            layer_index
            for layer_index, spec in enumerate(self._plan["layers"])
            if spec["block_type"] == "E"
        ]

    def _n_routed_experts(self) -> int:
        """Total router experts (size of the ON-GPU ``g2s`` global->slot lookup).

        Read from the model config; this is the domain of the global expert ids
        the stock gate returns, so the deferred path's ``g2s`` table must span
        ``[0, n_routed_experts)``.
        """
        return int(self.session.config["n_routed_experts"])

    def _compute_buddy_map(
        self, layer_index: int, order: list[int]
    ) -> dict[int, int]:
        """Map each NON-resident expert to its nearest RESIDENT buddy SLOT.

        The near-exactness primitive for ``cold_substitution``. For a fixed set
        ``order`` (slot i holds global expert ``order[i]``), every expert NOT in
        ``order`` is "cold": on the deferred path it would map to the sentinel and
        force a redo. Instead we pick, for each cold expert, the resident expert
        whose routing GATE behaves most like it, and remap the cold id to THAT
        resident's slot. The deferred path then gathers the buddy's weights for the
        cold routed slot.

        Buddy metric — GATE-ROW COSINE. The router's decision for expert ``e`` is
        ``x @ gate.weight[e]`` (+ a per-expert correction bias); ``gate.weight`` is
        ``[n_routed_experts, hidden]`` and resident on this layer. Two experts with
        nearly parallel gate rows respond to the SAME input directions, so the one
        the router *would* have chosen and its buddy fire on similar tokens — the
        principled "nearest substitute". We score
        ``cos(gate.weight[cold], gate.weight[resident])`` and take the argmax over
        resident experts. Ties (e.g. orthogonal rows) break to the SMALLEST resident
        slot for a deterministic, reproducible map. Computed ONCE here with mx ops
        (a small ``[E, hidden]`` gemm), never per token.

        Why scores are preserved (near-exactness). We only rewrite the global->slot
        LOOKUP; the gate still emits the cold expert's real top-k SCORE, and the
        weighted-sum applies that score to the buddy's output. So substitution
        changes only WHICH expert's weights compute a routed slot, never the routing
        weight — the rest of the MoE math is identical to exact. Drift is therefore
        confined to the substituted expert's contribution on cold-routed tokens.

        Returns ``{cold_global_id: resident_slot}`` for every cold expert (empty
        when ``order`` already covers all experts — substitution is then a no-op and
        the path stays byte-exact). The gate weight must be RESIDENT before this is
        called (``build_fixed_hotset`` forces the layer base resident first).
        """
        import mlx.core as mx

        n_experts = self._n_routed_experts()
        resident = set(int(e) for e in order)
        cold_ids = [gid for gid in range(n_experts) if gid not in resident]
        if not cold_ids:
            return {}

        gate = self.session.model.backbone.layers[layer_index].mixer.gate
        weight = getattr(gate, "weight", None)
        if weight is None:
            # No accessible gate weight (shouldn't happen for NemotronH MoEGate):
            # fall back to the first resident slot so substitution is still total
            # (no sentinel) and deterministic. Documented fallback per the brief.
            return {cold: 0 for cold in cold_ids}

        # Cosine over gate rows: normalise each row, then for every cold row take
        # the argmax dot against the RESIDENT rows. All on-GPU, one-time.
        w = weight.astype(mx.float32)                       # [E, hidden]
        norms = mx.sqrt((w * w).sum(axis=1)) + 1e-12        # [E]
        unit = w / norms[:, None]                           # [E, hidden] row-unit

        order_arr = mx.array([int(e) for e in order], dtype=mx.int32)
        cold_arr = mx.array(cold_ids, dtype=mx.int32)
        resident_unit = mx.take(unit, order_arr, axis=0)    # [K, hidden]
        cold_unit = mx.take(unit, cold_arr, axis=0)         # [C, hidden]
        # Cosine similarity matrix [C, K] (rows are unit-norm, so dot == cosine).
        sims = cold_unit @ resident_unit.T                  # [C, K]
        # argmax over resident slots; mx.argmax returns the FIRST max on ties, and
        # ``order`` is enumerated slot-ascending, so ties break to the smallest
        # slot -> deterministic.
        best_slots = mx.argmax(sims, axis=1)                # [C]
        mx.eval(best_slots)
        best = [int(s) for s in best_slots.tolist()]
        return {cold_ids[i]: best[i] for i in range(len(cold_ids))}

    def _build_cold_tier_stack(
        self, layer_index: int, cold_order: list[int]
    ) -> dict[str, Any] | None:
        """Build a RESIDENT low-precision stacked switch_mlp for ``cold_order``.

        The mixed-precision cold tier's per-layer payload. Loads the cold experts'
        4-bit on-disk rows ONCE (the same selective slice the hot set uses), then
        re-quantizes them to ``self.cold_tier_bits`` at ``self.cold_tier_group_size``
        (falling back to the model's own group size). The result is a SECOND stacked
        ``{fc1,fc2}`` table of those experts at the coarser grid, plus its own
        global->slot map and the recorded byte accounting (low-bit total vs what the
        same experts cost at 4-bit -- the fit win).

        Re-quant path (the only one MLX supports for an already-quantized input):
        ``dequantize(4-bit) -> float -> quantize(cold_tier_bits)``. We CANNOT mix
        bit-widths inside one stack/gather (MLX's ``gather_qmm`` takes scalar
        ``bits``/``group_size`` and the packed weight shape differs per width), so
        the cold tier is its OWN homogeneous low-bit stack, gathered by a separate
        native call from the 4-bit hot stack. Drift is the EXTRA quant error of the
        coarse grid on the expert's OWN weights -- a bounded near-exact perturbation,
        not a foreign-expert swap.

        Returns the cold-tier entry, or ``None`` when ``cold_order`` is empty.
        Row i of every field holds global expert ``cold_order[i]``.
        """
        import mlx.core as mx

        if not cold_order:
            return None
        mixer = self.session.model.backbone.layers[layer_index].mixer
        # The cold tier is a RE-QUANTIZATION lever: it needs a quantized resident
        # tier to re-quantize FROM (the on-disk 4-bit packing). An unquantized layer
        # has a plain ``SwitchLinear`` (weight only, no group_size/bits/scales), so
        # there is nothing to coarsen cheaply -- skip it (no tier built; that layer
        # keeps the exact page fallback). The real Ultra is fully quantized, so this
        # only no-ops on the unquantized toy fixture.
        fc1_module = getattr(mixer.switch_mlp, "fc1")
        if not hasattr(fc1_module, "bits") or not hasattr(fc1_module, "group_size"):
            return None

        # Load the cold experts' 4-bit rows once (selective slice -> compact stack).
        compact, _event = self._load_selected_experts(layer_index, cold_order)

        to_bits = int(self.cold_tier_bits)
        # The on-disk (resident-tier) quant params per projection, read off the live
        # QuantizedSwitchLinear modules (group_size 32 / bits 4 / affine on Ultra).
        low: dict[str, dict[str, Any]] = {"fc1": {}, "fc2": {}}
        bytes_low = 0
        bytes_ref4 = 0
        # The actual group size used PER projection (a requested coarse size that
        # doesn't divide a projection's latent input dim falls back to the model's
        # own size for that projection -- so the tier is always buildable).
        group_sizes: dict[str, int] = {}
        for projection in ("fc1", "fc2"):
            module = getattr(mixer.switch_mlp, projection)
            from_gs = int(getattr(module, "group_size"))
            from_bits = int(getattr(module, "bits"))
            mode = getattr(module, "mode", "affine")
            fields = compact[projection]
            # 4-bit -> float (the dequant of the SAME rows the hot set would gather).
            deq = mx.dequantize(
                fields["weight"],
                fields["scales"],
                fields.get("biases"),
                group_size=from_gs,
                bits=from_bits,
                mode=mode,
            )
            # A coarser cold-tier group size shrinks the per-group scale/bias
            # overhead -> a bigger fit win, but it must DIVIDE this projection's
            # latent input dim. If the requested size doesn't divide it, fall back to
            # the resident tier's own size for this projection (always valid). On the
            # real Ultra (latent 2048 / 5120) both divide 64 and 128, so the coarse
            # size applies uniformly; this fallback only fires on small toy dims.
            in_dim = int(deq.shape[-1])
            to_gs = int(self.cold_tier_group_size or from_gs)
            if in_dim % to_gs != 0:
                to_gs = from_gs
            group_sizes[projection] = to_gs
            # float -> coarse grid. ``mx.quantize`` returns (weight, scales, biases).
            wq, sq, bq = mx.quantize(deq, group_size=to_gs, bits=to_bits, mode=mode)
            mx.eval(wq, sq, bq)
            low[projection] = {"weight": wq, "scales": sq, "biases": bq}
            bytes_low += wq.nbytes + sq.nbytes + bq.nbytes
            # What these SAME experts cost at the 4-bit resident grid (the baseline
            # this tier saves against): the bytes of the 4-bit compact slice.
            bytes_ref4 += sum(arr.nbytes for arr in fields.values())
            del deq

        global_to_slot = {
            int(global_id): slot for slot, global_id in enumerate(cold_order)
        }
        return {
            "order": list(cold_order),
            "arrays": low,
            "global_to_slot": global_to_slot,
            "bits": to_bits,
            # Per-projection group size actually used (the serve helper points each
            # QuantizedSwitchLinear's group_size at its own value).
            "group_sizes": group_sizes,
            "bytes": int(bytes_low),
            "bytes_at_4bit": int(bytes_ref4),
        }

    def build_fixed_hotset(
        self,
        warmup_prompt_ids: list[int],
        *,
        warmup_tokens: int = 32,
        override: dict[int, list[int]] | None = None,
        cold_tier_override: dict[int, list[int]] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Build the ONE-TIME per-(MoE-layer) FIXED resident hot-set of top-K experts.

        Two phases:

        1. **Frequency discovery.** Run a warmup ``generate_greedy`` over
           ``warmup_prompt_ids`` for ``warmup_tokens`` steps, accumulating per-(MoE
           layer, expert) routing FREQUENCY from the routed-expert union the
           runner already records per pass (``_expert_history`` after each MoE
           forward). For each MoE layer the top-K (= ``fixed_hotset_experts``) most
           frequent experts become its fixed membership. An explicit ``override``
           (``{layer_index: [expert_ids]}``) bypasses discovery for that layer —
           used for deterministic testing of the native-hit vs cold-fallback
           branches.

        2. **One-time load + assembly.** For each MoE layer, load its chosen
           experts ONCE via the existing selective machinery
           (:meth:`_load_selected_experts`, which slices the stacked
           ``switch_mlp.fc1/fc2`` weight + (quantized) scales/biases of exactly
           those expert rows), and store the assembled stacked arrays plus the
           global->slot remap in ``self._fixed_hotset[layer]``. Row i holds global
           expert ``order[i]`` — the same layout the page path produces — so the
           per-token native swap can reuse :class:`NemotronHotsetGateAdapter`
           unchanged.

        The resulting arrays are REUSED by reference every token (no concatenate),
        so the per-token cost on the native-hit path is just a reference swap +
        the model's own on-GPU gather. Returns ``self._fixed_hotset`` for
        inspection. The build is bounded by K experts per MoE layer (big on the
        real model, trivial on the toy).
        """
        import mlx.core as mx

        if self.fixed_hotset_experts is None:
            raise ValueError(
                "build_fixed_hotset requires fixed_hotset_experts to be set"
            )
        k = self.fixed_hotset_experts
        override = override or {}
        cold_tier_override = cold_tier_override or {}
        if cold_tier_override and self.cold_tier_bits is None:
            raise ValueError(
                "cold_tier_override requires cold_tier_bits to be set"
            )
        moe_layers = self._moe_layer_indices()

        # --- MEMORY HYGIENE: force the pread reader for the whole build --------
        # On the real 550B this build peaked at ~217GB RSS (jetsam ~220GB) while
        # its STEADY footprint is only ~136GB; the ~80GB transient spike is the
        # warmup's expert residency + the phase-3 union reads + MLX's allocator
        # cache, doubled by macOS holding mmap file pages (MADV_DONTNEED is a
        # no-op, so the mmap reader's faulted file pages are never released ->
        # RSS counted twice). Force the pread reader (byte-IDENTICAL output, no
        # mmap residency) for both phases so those file pages never accumulate,
        # and restore the prior choice afterwards. The nemotron loader otherwise
        # defaults to the mmap reader (``drop_mmap_cache_after_read=False``).
        loader = self.session.loader
        prev_drop_mmap = loader.drop_mmap_cache_after_read
        loader.drop_mmap_cache_after_read = True
        try:
            # --- Phase 1: discover routing frequency per (MoE layer, expert).
            # Only run the warmup when at least one layer needs discovery (i.e. is
            # not fully specified by ``override``); a pure-override build is purely
            # deterministic and must not depend on warmup routing.
            freq: dict[int, dict[int, int]] = {layer: {} for layer in moe_layers}
            needs_discovery = any(layer not in override for layer in moe_layers)
            if needs_discovery:
                if warmup_tokens < 1:
                    raise ValueError("warmup_tokens must be positive")
                # The warmup decode drives the normal (page) MoE path because the
                # fixed hot-set is not built yet, so every MoE pass records its
                # routed union into ``_expert_history``; we sample it after each
                # pass via a lightweight hook so a single warmup yields per-pass
                # frequencies.
                original_history = self._expert_history
                self._expert_history = {}
                counts_hook = self._install_frequency_hook(freq)
                try:
                    self.generate_greedy(list(warmup_prompt_ids), warmup_tokens)
                finally:
                    counts_hook()  # uninstall
                    # Drop the warmup's transient expert arrays / KV+Mamba cache
                    # references: the page MoE path leaves no per-pass arrays live
                    # (each compact table goes out of scope after its pass) and the
                    # warmup's ``generate_greedy`` cache is local to that call, so
                    # restoring ``original_history`` here drops the only handles we
                    # held. Nothing below reads the warmup routing arrays — only the
                    # ``freq`` COUNTS (plain ints) feed top-K selection.
                    self._expert_history = original_history
                # MEMORY HYGIENE: release everything the warmup loaded that the
                # fixed stacks do NOT need, BEFORE the heavy union load. The warmup
                # faulted expert row pages into the loader's weight-page cache and
                # left freed buffers in MLX's allocator; none of that is part of
                # the fixed stacks (phase 3 reloads exactly the chosen experts), so
                # clearing it returns ~the warmup footprint to the OS and keeps the
                # phase-3 transients from stacking on top of it. Safe: the fixed
                # stacks are not built yet, so nothing below depends on this state.
                self._release_build_transients(mx)

            # --- Phase 2: choose top-K per layer, load once, assemble fixed
            # arrays.
            # Reset the cold-substitution counter for this (re)build; it is
            # re-summed below as buddy maps are computed (stays 0 when
            # substitution is off).
            self._deferred_stats["cold_substitutions"] = 0
            # Reset the cold-tier accounting for this (re)build; re-summed below as
            # per-layer low-bit stacks are built (stays 0 when the tier is off).
            self._cold_tier_stats["cold_tier_experts"] = 0
            self._cold_tier_stats["cold_tier_bytes"] = 0
            self._cold_tier_stats["cold_tier_bytes_at_4bit"] = 0
            if self.cold_substitution:
                # The buddy map reads each MoE layer's ``gate.weight``, which is
                # loaded lazily per token via ``_load_layer_base``. Force the base
                # resident NOW (no-op under ``pin_policy='all'`` if already
                # resident) so the gate weight is materialised before
                # ``_compute_buddy_map`` reads it. A pure-override build skips the
                # warmup, so without this the gate could be unloaded here.
                for layer_index in moe_layers:
                    self._load_layer_base(layer_index)
            fixed: dict[int, dict[str, Any]] = {}
            # MEMORY HYGIENE: clear MLX's allocator cache every few layers during
            # the union assemble so the per-layer raw page reads don't accumulate
            # across all the MoE layers (on the real model each layer's reads are
            # ~2GB; 50+ layers' worth held simultaneously is the bulk of the
            # phase-3 transient). The fixed stacks are already materialised owned
            # copies (``mx.eval`` below), so freeing the cache cannot touch them.
            CLEAR_EVERY = 4
            for built_count, layer_index in enumerate(moe_layers):
                if layer_index in override:
                    order = [int(e) for e in override[layer_index]]
                else:
                    # Top-K most frequent (ties broken by smaller global id for a
                    # deterministic, reproducible membership).
                    ranked = sorted(
                        freq[layer_index].items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                    order = [expert for expert, _count in ranked[:k]]
                if not order:
                    # No routing observed for this layer (e.g. warmup never hit
                    # it): leave it OUT of the fixed set so it stays on the exact
                    # page path rather than installing an empty resident table.
                    continue

                # Load exactly these expert rows ONCE (same slicing the cold/page
                # path uses), giving a compact stacked table whose row i is global
                # expert ``order[i]``.
                compact, _event = self._load_selected_experts(layer_index, order)
                # Force the one-time slice to materialize now (it is reused by
                # reference forever after — never re-sliced per token). After this
                # eval each field is an INDEPENDENT owned array; the raw page reads
                # feeding it are no longer referenced and become free for the
                # periodic ``clear_cache`` below to return to the OS.
                mx.eval(
                    [compact["fc1"][f] for f in compact["fc1"]]
                    + [compact["fc2"][f] for f in compact["fc2"]]
                )
                global_to_slot = {
                    int(global_id): slot for slot, global_id in enumerate(order)
                }
                # ON-GPU global->slot lookup for the deferred path. ``g2s`` is a dense
                # ``[n_routed_experts]`` int32 table: ``g2s[gid] = slot`` for a resident
                # expert, and a SENTINEL (= K, deliberately OUT of the valid slot range
                # ``[0, K-1]``) for every cold expert. At decode the remap gate does
                # ``mx.take(g2s, inds)`` (lazy, no numpy, no eval) to turn the stock
                # gate's GLOBAL ids into LOCAL slots. A ``slot == SENTINEL`` is the
                # ON-GPU cold signal; the gate clamps it to a valid slot so the gather
                # cannot OOB (the result is discarded on the cold-redo anyway). Built
                # once here and reused by reference every token.
                sentinel = len(order)  # == K resident rows; one past the last slot
                g2s_np = np.full(
                    int(self._n_routed_experts()), sentinel, dtype=np.int32
                )
                for global_id, slot in global_to_slot.items():
                    g2s_np[global_id] = slot

                # NEAR-EXACT cold-expert BUDDY substitution (opt-in). With substitution
                # OFF, every cold id keeps the sentinel above -> the exact cold-redo path
                # (byte-identical to the proven deferred forward). With it ON, we OVERWRITE
                # each cold id's sentinel with its nearest RESIDENT buddy SLOT, so:
                #   * NO g2s entry equals the sentinel -> the remap gate's
                #     ``local == sentinel`` cold flag NEVER fires -> NO redo -> the
                #     deferred path runs fast on ANY generation (the whole point);
                #   * a routed cold expert is computed by a similar resident one. The
                #     gate still emits the cold expert's real SCORE and the weighted-sum
                #     applies it to the buddy's output, so ONLY which expert computes the
                #     slot changes -- scores (and thus the rest of the MoE math) are
                #     unchanged. Drift is confined to substituted experts on cold tokens.
                # RESIDENT ids are untouched (they already hold their own slot), so an
                # all-resident token is STILL bit-identical to exact -- the substitution
                # is strictly scoped to cold experts.
                buddy_map: dict[int, int] = {}
                if self.cold_substitution:
                    buddy_map = self._compute_buddy_map(layer_index, order)
                    for cold_id, buddy_slot in buddy_map.items():
                        g2s_np[cold_id] = buddy_slot  # cold -> resident buddy slot
                    self._deferred_stats["cold_substitutions"] += len(buddy_map)

                g2s = mx.array(g2s_np)
                mx.eval(g2s)  # one-time; constant thereafter
                fixed[layer_index] = {
                    "order": order,
                    "arrays": {
                        "fc1": dict(compact["fc1"]),
                        "fc2": dict(compact["fc2"]),
                    },
                    "global_to_slot": global_to_slot,
                    # Deferred-path extras (ignored by the synced fixed path):
                    "g2s": g2s,            # [n_routed_experts] int32 global->slot lookup
                    "sentinel": sentinel,  # cold marker (== K, out of [0, K-1])
                    # Cold-substitution map {cold_global_id: resident_slot}; EMPTY when
                    # substitution is off or the fixed set already covers every expert
                    # (then g2s still carries sentinels and the exact redo applies).
                    "buddy_map": buddy_map,
                }

                # --- Mixed-precision COLD-EXPERT tier (opt-in) ----------------
                # Build a SECOND, low-precision resident stack of the cold tail for
                # this layer. The tail is the explicit ``cold_tier_override`` ids if
                # given, else the next-most-frequent experts the warmup observed that
                # did NOT make the 4-bit top-K (so the tier holds the experts most
                # likely to actually route cold, at a coarser grid). Experts already
                # in the 4-bit ``order`` are excluded -- the tier is strictly for
                # NON-resident experts, so it never shadows an exact 4-bit row.
                if self.cold_tier_bits is not None:
                    resident_set = set(order)
                    if layer_index in cold_tier_override:
                        cold_order = [
                            int(e)
                            for e in cold_tier_override[layer_index]
                            if int(e) not in resident_set
                        ]
                    else:
                        # Frequency-ranked tail beyond the top-K hot set. (Empty on a
                        # pure-override build with no cold_tier_override -> tier off
                        # for that layer, exact cold-fallback preserved.)
                        ranked_tail = sorted(
                            (
                                (e, c)
                                for e, c in freq[layer_index].items()
                                if e not in resident_set
                            ),
                            key=lambda item: (-item[1], item[0]),
                        )
                        cold_order = [e for e, _c in ranked_tail]
                    cold_tier = self._build_cold_tier_stack(layer_index, cold_order)
                    if cold_tier is not None:
                        fixed[layer_index]["cold_tier"] = cold_tier
                        self._cold_tier_stats["cold_tier_experts"] += len(
                            cold_tier["order"]
                        )
                        self._cold_tier_stats["cold_tier_bytes"] += cold_tier["bytes"]
                        self._cold_tier_stats["cold_tier_bytes_at_4bit"] += cold_tier[
                            "bytes_at_4bit"
                        ]

                # Drop this layer's intermediate load buffers now that the stack is
                # stored (the ``dict(...)`` copies above own the eval'd arrays; the
                # ``compact`` dict and the ``g2s_np`` staging buffer are no longer
                # needed). Then, every few layers, return the freed allocator
                # buffers + the loader's page residency to the OS so phase-3 reads
                # don't pile up. We clear AFTER assembling a layer (never before)
                # so the current layer's reads are never re-faulted mid-build.
                del compact, g2s_np
                if (built_count + 1) % CLEAR_EVERY == 0:
                    self._release_build_transients(mx)

            self._fixed_hotset = fixed
            # Final hygiene pass: drop any page residency + freed buffers left by
            # the tail layers before the build returns into the decode loop.
            self._release_build_transients(mx)
            return self._fixed_hotset
        finally:
            # Restore the caller's mmap-reader choice. The pread reader was forced
            # only for the duration of the build's heavy loads; decode keeps its
            # configured behaviour (nemotron default = mmap reader).
            loader.drop_mmap_cache_after_read = prev_drop_mmap

    def _release_build_transients(self, mx: Any) -> None:
        """Return everything NOT part of the fixed stacks to the OS.

        Called between ``build_fixed_hotset``'s phases (after the warmup) and
        periodically during the per-layer union assemble. It frees two distinct
        transients, neither of which the fixed stacks depend on:

        * the loader's weight-page residency — on the real model the warmup faults
          expert row pages into this cache, and phase 3 faults each layer's union
          rows too; none of those pages are part of the fixed stacks (each stack
          is an independent ``mx.eval``'d owned array, see the build loop), so the
          chosen experts are simply re-read on demand if ever needed. Clearing it
          drops the dominant page-cache RSS;
        * MLX's allocator cache — the buffers freed by dropping the warmup's
          KV/Mamba cache and each layer's raw page reads stay in MLX's cache until
          ``clear_cache`` hands them back to the OS.

        Safe by construction: it touches only caches/residency, never the
        ``self._fixed_hotset`` arrays (which are owned, materialised copies). The
        toy model is too small to show the GB effect, so the tests assert this
        MECHANISM fires; the real peak-RSS win is measured on the Studio.
        """
        # Drop the weight-page residency (no-op when no page cache is attached,
        # e.g. the toy without ``weight_page_budget_bytes``).
        self.session.loader.clear_weight_page_cache()
        # Hand the freed buffers back to the OS (MLX caches them otherwise).
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:  # very old MLX
            mx.metal.clear_cache()

    def _install_frequency_hook(
        self, freq: dict[int, dict[int, int]]
    ) -> Any:
        """Wrap the page MoE forward to tally routed experts into ``freq``.

        ``build_fixed_hotset``'s warmup runs the normal page path (the fixed set
        is not built yet). After each MoE pass that path sets
        ``_expert_history[layer] = selected``; we wrap the page forward so that,
        immediately after it returns, the routed experts for that layer are
        counted. Returns a zero-arg callable that restores the original method.
        """
        original = self._nemotron_moe_layer_forward

        def counting(layer_index, layer, x, **kwargs):
            out = original(layer_index, layer, x, **kwargs)
            layer_counts = freq.setdefault(layer_index, {})
            for expert in self._expert_history.get(layer_index, []):
                layer_counts[expert] = layer_counts.get(expert, 0) + 1
            return out

        self._nemotron_moe_layer_forward = counting  # type: ignore[assignment]

        def restore() -> None:
            # Drop the instance-attribute shadow so the bound class method is
            # used again (rather than pinning a captured bound method on self).
            self.__dict__.pop("_nemotron_moe_layer_forward", None)

        return restore

    def _nemotron_moe_layer_forward_fixed(
        self,
        layer_index: int,
        layer: Any,
        x: Any,
        *,
        mask: Any,
        cache: Any,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        """MoE ('E') forward with the FIXED resident hot-set + native gather.

        Per token: gate the normed input to learn the routed global experts. If
        they are a SUBSET of this layer's fixed resident membership (the
        native-hit case), swap the PREBUILT stacked arrays into ``switch_mlp`` by
        REFERENCE (no concatenate), install a :class:`NemotronHotsetGateAdapter`
        that remaps global -> the fixed row positions, and run the block's own
        native ``NemotronHMoE.__call__`` (one on-GPU gather, ZERO Python
        assembly). Restore the originals afterwards so the fixed arrays stay owned
        by the runner and reused next token.

        EXACTNESS: this is the SAME native ``__call__`` the page path runs, against
        the SAME expert rows (the fixed stack rows ARE the rows
        ``_load_selected_experts`` sliced, just loaded once), with the SAME remap
        helper (scores pass through unchanged). So the hit path is bit-identical
        to the page path and to stock. If a routed expert is NOT resident (cold,
        rare), defer to the existing exact page path
        (:meth:`_nemotron_moe_layer_forward`) for this layer.
        """
        import mlx.core as mx

        entry = self._fixed_hotset.get(layer_index)
        if entry is None:
            # This layer has no fixed set (e.g. warmup never routed it): stay on
            # the exact page path.
            return self._nemotron_moe_layer_forward(
                layer_index,
                layer,
                x,
                mask=mask,
                cache=cache,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        mixer = layer.mixer

        # Gate on the same normed input the block will use, to learn the routed
        # global experts WITHOUT touching the full expert table.
        normed = layer.norm(x)
        inds, _scores = mixer.gate(normed)
        mx.eval(inds)
        selected = selected_expert_ids(inds)
        self._expert_history[layer_index] = selected

        # Residency check: is every routed expert in the fixed (4-bit) resident set?
        global_to_slot = entry["global_to_slot"]
        if not all(expert in global_to_slot for expert in selected):
            # A routed expert is not in the 4-bit hot set. Before paying a disk
            # reload, try the mixed-precision COLD TIER: if EVERY routed expert is
            # held resident in this layer's low-bit cold stack, serve the whole pass
            # from that stack (the experts' OWN weights at a coarser grid -> bounded
            # re-quant drift, NO disk reload, NO foreign-expert swap). This handles
            # the pure-cold pass (every routed expert in the tier); a MIXED pass
            # (some routed experts 4-bit-resident, some cold-tier) cannot be served
            # by a single homogeneous-precision native gather (MLX ``gather_qmm``
            # takes scalar bits/group_size), so it falls through to the exact page
            # path -- the documented PoC boundary.
            cold_tier = entry.get("cold_tier")
            if cold_tier is not None and all(
                expert in cold_tier["global_to_slot"] for expert in selected
            ):
                self._cold_tier_stats["cold_tier_hits"] += 1
                return self._serve_moe_from_cold_tier(
                    layer_index,
                    layer,
                    x,
                    cold_tier,
                    mask=mask,
                    cache=cache,
                    events=events,
                    pass_kind=pass_kind,
                    token_step=token_step,
                    selected=selected,
                )
            # COLD fallback (rare): a routed expert is neither 4-bit resident nor
            # fully covered by the cold tier -> run the existing exact page path for
            # this layer. ``selected`` was already recorded above; the page path
            # records it again identically.
            self._hotset_stats["cold_fallbacks"] += 1
            return self._nemotron_moe_layer_forward(
                layer_index,
                layer,
                x,
                mask=mask,
                cache=cache,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        # NATIVE-HIT path: routing is a subset of residency. Swap the FIXED
        # stacked arrays into switch_mlp.fc1/fc2 by REFERENCE (no concatenate)
        # and remap the gate to the fixed membership order. ``order`` is the full
        # resident set (a superset of ``selected``); the strict adapter remaps
        # every routed global to its fixed slot and passes scores through, so the
        # native weighted-sum over the resident table is bit-identical to stock.
        self._hotset_stats["native_hits"] += 1
        arrays = entry["arrays"]
        order = entry["order"]
        events.append(
            {
                "kind": "compute",
                "action": "hotset-native-hit",
                "pass": pass_kind,
                "token_step": token_step,
                "layer": layer_index,
                "expert_count": len(selected),
                "experts": selected,
                "resident_experts": len(order),
            }
        )

        original_gate = mixer.gate
        saved: dict[str, dict[str, Any]] = {}
        for projection in ("fc1", "fc2"):
            module = getattr(mixer.switch_mlp, projection)
            saved[projection] = {
                field: module[field] for field in arrays[projection]
            }
        try:
            mixer.gate = NemotronHotsetGateAdapter(original_gate, order, strict=True)
            for projection in ("fc1", "fc2"):
                module = getattr(mixer.switch_mlp, projection)
                for field, value in arrays[projection].items():
                    setattr(module, field, value)
            return layer(x, mask=mask, cache=cache)
        finally:
            mixer.gate = original_gate
            for projection in ("fc1", "fc2"):
                module = getattr(mixer.switch_mlp, projection)
                for field, value in saved[projection].items():
                    setattr(module, field, value)

    def _serve_moe_from_cold_tier(
        self,
        layer_index: int,
        layer: Any,
        x: Any,
        cold_tier: dict[str, Any],
        *,
        mask: Any,
        cache: Any,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
        selected: list[int],
    ) -> Any:
        """Serve a PURE-COLD MoE pass from the resident LOW-BIT cold tier.

        Pre-condition (checked by the caller): every routed expert in ``selected``
        is held in this layer's ``cold_tier`` low-bit stack. We then do EXACTLY the
        native-hit swap of :meth:`_nemotron_moe_layer_forward_fixed`, but against the
        cold tier's low-bit ``{fc1,fc2}`` arrays instead of the 4-bit hot stack, and
        we ALSO swap each ``QuantizedSwitchLinear``'s ``bits``/``group_size`` to the
        coarse cold-tier grid so the native ``gather_qmm`` decodes the low-bit packing
        correctly. The gate is remapped to the cold-tier order (scores pass through
        unchanged), so the native weighted-sum is structurally identical to stock --
        ONLY the expert weight grid is coarser. Everything is restored in ``finally``
        so the runner's owned tier arrays + the model's stock quant params are intact
        for the next token.

        NEAR-EXACT: the served experts are the SAME experts the page path would load,
        just re-quantized to fewer bits, so the output drifts from the exact 4-bit
        result by only the extra quantization error of the coarse grid -- bounded and
        finite, the near-exact lane. (A token routing to a MIX of 4-bit-hot and
        cold-tier experts is NOT served here: one homogeneous-precision native gather
        cannot span both grids; such passes take the exact page fallback.)
        """
        mixer = layer.mixer
        arrays = cold_tier["arrays"]
        order = cold_tier["order"]
        # The coarse grid the tier was re-quantized at, per projection.
        group_sizes = cold_tier["group_sizes"]
        to_bits = int(cold_tier["bits"])

        events.append(
            {
                "kind": "compute",
                "action": "cold-tier-hit",
                "pass": pass_kind,
                "token_step": token_step,
                "layer": layer_index,
                "expert_count": len(selected),
                "experts": selected,
                "tier_experts": len(order),
                "tier_bits": to_bits,
            }
        )

        original_gate = mixer.gate
        saved: dict[str, dict[str, Any]] = {}
        for projection in ("fc1", "fc2"):
            module = getattr(mixer.switch_mlp, projection)
            saved[projection] = {
                field: module[field] for field in arrays[projection]
            }
            saved[projection]["__bits"] = module.bits
            saved[projection]["__group_size"] = module.group_size
        try:
            mixer.gate = NemotronHotsetGateAdapter(original_gate, order, strict=True)
            for projection in ("fc1", "fc2"):
                module = getattr(mixer.switch_mlp, projection)
                for field, value in arrays[projection].items():
                    setattr(module, field, value)
                # Point the native gather at the coarse cold-tier grid (per-proj).
                module.bits = to_bits
                module.group_size = int(group_sizes[projection])
            return layer(x, mask=mask, cache=cache)
        finally:
            mixer.gate = original_gate
            for projection in ("fc1", "fc2"):
                module = getattr(mixer.switch_mlp, projection)
                for field, value in saved[projection].items():
                    if field in ("__bits", "__group_size"):
                        continue
                    setattr(module, field, value)
                module.bits = saved[projection]["__bits"]
                module.group_size = saved[projection]["__group_size"]

    # ------------------------------------------------------------------ #
    # Deferred-eval decode: install ONCE, run lazily, redo cold tokens.   #
    # ------------------------------------------------------------------ #

    def _install_deferred_hotset(self) -> None:
        """Install the fixed stacks + ON-GPU remap gates into every MoE layer ONCE.

        The synced fixed path swaps the prebuilt stacks + a remap adapter into
        ``switch_mlp`` PER MoE layer PER token and restores them in a ``finally``.
        That per-layer Python swap is cheap on its own, but the remap adapter's
        ``np.asarray(inds)`` forces a per-layer ``mx.eval``. The deferred path
        installs the constant fixed stacks and an :class:`NemotronDeferredRemapGate`
        ONCE (the stacks never change), so per token there is NO Python swap and NO
        per-layer eval — the forward is one lazy graph.

        Idempotent. Layers without a fixed entry are left untouched (they stay on
        whatever path ``_stream_forward_tokens`` dispatches — here, never the
        deferred fast path: a missing-entry layer means the token is cold by
        construction, handled by the redo).

        IMPORTANT — the gate weights must be RESIDENT before we wrap the gate. The
        runner loads a layer's base (gate/latent/shared) lazily per token via
        ``_load_layer_base`` -> ``load_weights``, which traverses ``mixer.gate``.
        Once we replace ``mixer.gate`` with our (non-``nn.Module``) remap wrapper,
        that traversal can no longer reach the real ``MoEGate.weight`` — so a base
        load AFTER install would silently drop the gate weights. We therefore force
        every MoE layer's base resident FIRST (a no-op under ``pin_policy='all'``
        thereafter: ``_load_layer_base`` sees the tensors already resident and
        never re-runs ``load_weights``, leaving the wrapped gate untouched). The
        wrapper then captures the gate WITH its real weights loaded.
        """
        if self._deferred_installed:
            return
        backbone = self.session.model.backbone
        # Force gate/latent/shared resident on every MoE layer before wrapping, so
        # later per-token base loads are no-ops that don't disturb the wrapper.
        for layer_index in self._fixed_hotset:
            self._load_layer_base(layer_index)
        for layer_index, entry in self._fixed_hotset.items():
            mixer = backbone.layers[layer_index].mixer
            arrays = entry["arrays"]
            # Stash the model's own gate + full expert stack so the deferred mode
            # can be uninstalled (e.g. for a cold redo on the synced path, which
            # expects the original switch_mlp + gate).
            entry["_orig_gate"] = mixer.gate
            entry["_orig_switch"] = {
                projection: {
                    field: getattr(mixer.switch_mlp, projection)[field]
                    for field in arrays[projection]
                }
                for projection in ("fc1", "fc2")
            }
            # Install the constant fixed stacks by REFERENCE (no concatenate) ...
            for projection in ("fc1", "fc2"):
                module = getattr(mixer.switch_mlp, projection)
                for field, value in arrays[projection].items():
                    setattr(module, field, value)
            # ... and the ON-GPU remap gate (shares the runner's cold accumulator).
            mixer.gate = NemotronDeferredRemapGate(
                entry["_orig_gate"],
                entry["g2s"],
                entry["sentinel"],
                self._deferred_cold_flags,
                # Read K LIVE from the runner so a sweep can flip force_top_k
                # between decode runs without reinstalling the gates.
                force_top_k=lambda: self.force_top_k,
            )
        self._deferred_installed = True

    def _uninstall_deferred_hotset(self) -> None:
        """Restore each MoE layer's original gate + full expert stack.

        Reverses :meth:`_install_deferred_hotset` so the exact synced/page paths
        (which assemble their own compact tables) see the model in its stock
        state. Called before a cold-token redo and at the end of a deferred run.
        """
        if not self._deferred_installed:
            return
        backbone = self.session.model.backbone
        for layer_index, entry in self._fixed_hotset.items():
            mixer = backbone.layers[layer_index].mixer
            mixer.gate = entry.pop("_orig_gate")
            orig_switch = entry.pop("_orig_switch")
            for projection in ("fc1", "fc2"):
                module = getattr(mixer.switch_mlp, projection)
                for field, value in orig_switch[projection].items():
                    setattr(module, field, value)
        self._deferred_installed = False

    def _mamba_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer_index
            for layer_index, spec in enumerate(self._plan["layers"])
            if spec["block_type"] == "M"
        )

    def _install_fused_ssm_norm(self) -> None:
        """Swap each Mamba2 mixer's gated RMSNorm for the fused kernel ONCE.

        Mirrors :meth:`_install_deferred_hotset`: a per-Mamba-layer component
        swap, idempotent, with the original stashed for an exact restore. The
        wrapper (:class:`NemotronFusedRMSNormGate`) is NOT an ``nn.Module``, so
        it would hide ``mixer.norm.weight`` from ``load_weights`` traversal. We
        therefore force every Mamba layer's BASE resident FIRST (the norm weight
        is a base tensor), so later per-token ``_load_layer_base`` calls see it
        resident and never re-run ``load_weights`` through the swapped norm.
        A no-op when the flag is off or already installed.
        """
        if not self.fuse_ssm_norm_gate or self._fused_ssm_norm_installed:
            return
        backbone = self.session.model.backbone
        mamba_layers = self._mamba_layer_indices()
        for layer_index in mamba_layers:
            self._load_layer_base(layer_index)
        for layer_index in mamba_layers:
            mixer = backbone.layers[layer_index].mixer
            norm = getattr(mixer, "norm", None)
            if norm is None or isinstance(norm, NemotronFusedRMSNormGate):
                continue
            mixer.norm = NemotronFusedRMSNormGate(norm)
        self._fused_ssm_norm_installed = True

    def _uninstall_fused_ssm_norm(self) -> None:
        """Restore each Mamba2 mixer's original gated RMSNorm module."""
        if not self._fused_ssm_norm_installed:
            return
        backbone = self.session.model.backbone
        for layer_index in self._mamba_layer_indices():
            mixer = backbone.layers[layer_index].mixer
            norm = getattr(mixer, "norm", None)
            if isinstance(norm, NemotronFusedRMSNormGate):
                mixer.norm = norm._norm
        self._fused_ssm_norm_installed = False

    def _nemotron_moe_layer_forward_deferred(
        self,
        layer_index: int,
        layer: Any,
        x: Any,
        *,
        mask: Any,
        cache: Any,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        """MoE ('E') forward on the DEFERRED path — pure lazy, ZERO per-layer eval.

        Pre-condition: :meth:`_install_deferred_hotset` has installed the fixed
        stack + ON-GPU remap gate into this layer. So unlike the synced fixed path
        this method does NOT gate-and-eval, does NOT Python-check residency, and
        does NOT swap/restore arrays. It just runs the block's own native
        ``__call__`` against the installed state; the remap gate turns global ids
        into local slots on-GPU and appends this layer's lazy cold flag.

        If this layer has no fixed entry it is cold by construction (no resident
        stack to gather from). We can't fall back per-layer without breaking the
        single-graph property, so we mark the token cold (a constant 1 flag) and
        run the native block anyway — its result is discarded by the end-of-token
        redo. ``_install_deferred_hotset`` leaves such layers stock, so the native
        forward is at least well-defined.
        """
        import mlx.core as mx

        if self._fixed_hotset.get(layer_index) is None:
            # No resident set for this layer -> force the token cold (redo path).
            self._deferred_cold_flags.append(mx.array(1, dtype=mx.int32))
        # The per-layer "compute" telemetry dict is informational only — it is not
        # read by ``run_summary`` (which aggregates ``kind=='load'`` byte/second
        # totals) nor by any exactness test. On the deferred hot path it is pure
        # per-token Python churn (x N MoE layers x every token), so skip it when
        # ``hot_decode`` is armed; keep it otherwise for trace parity.
        if not self.hot_decode:
            events.append(
                {
                    "kind": "compute",
                    "action": "hotset-deferred",
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                }
            )
        return layer(x, mask=mask, cache=cache)

    def _deferred_cold_flag(self) -> Any:
        """OR-reduce the per-layer cold contributions into ONE lazy scalar.

        ``1`` iff ANY MoE layer this token routed to a non-resident expert. Lazy:
        composed from the per-layer ``mx.max(local == sentinel)`` flags the remap
        gates appended, so it costs nothing until the single end-of-token eval.
        """
        import mlx.core as mx

        if not self._deferred_cold_flags:
            return mx.array(0, dtype=mx.int32)
        flag = self._deferred_cold_flags[0]
        for extra in self._deferred_cold_flags[1:]:
            flag = mx.maximum(flag, extra)
        return flag

    def _stream_forward_tokens_deferred(
        self,
        token_rows: list[list[int]],
        *,
        cache: list[Any],
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None = None,
        _hot: bool = False,
    ) -> Any:
        """One forward over the DEFERRED MoE path; resets the cold accumulator.

        Clears ``_deferred_cold_flags`` (a fresh per-token accumulator), runs the
        shared layer loop with the deferred MoE dispatch, and returns the pre-norm
        hidden state. The caller reads :meth:`_deferred_cold_flag` (lazy) and evals
        it together with the next-token argmax — the SINGLE per-token sync.

        ``_hot`` forwards to the shared loop's hot path (skip the no-op per-layer
        loader + event churn); the caller passes it only for resident decode steps.
        """
        self._deferred_cold_flags.clear()
        return self._stream_forward_tokens(
            token_rows,
            cache=cache,
            events=events,
            pass_kind=pass_kind,
            token_step=token_step,
            _moe_forward_override=self._nemotron_moe_layer_forward_deferred,
            _hot=_hot,
        )

    def _stream_forward_tokens(
        self,
        token_rows: list[list[int]],
        *,
        cache: list[Any],
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None = None,
        manage_embedding: bool = True,
        _moe_forward_override: Any = None,
        _hot: bool = False,
    ) -> Any:
        """Mirror NemotronHModel.__call__'s layer loop (pre-norm hidden state).

        Returns the hidden state BEFORE ``norm_f`` is applied; the caller is
        responsible for the final norm + lm_head. ``self._plan`` drives both the
        cache index (advances only on M/* layers) and the mask selection
        (attention mask for '*', SSM mask otherwise).

        ``_moe_forward_override`` lets the DEFERRED path force its own MoE block
        forward (the lazy, no-per-layer-eval one) without touching the default
        dispatch; ``None`` keeps the normal priority (fixed -> persistent -> page).

        ``_hot`` (only the deferred decode loop, only after a warmup forward has
        made every layer resident) takes the HOT PATH: the per-layer loop SKIPS
        the no-op ``_load_layer_*`` call and the per-layer event appends. Those
        move zero bytes / feed no ``run_summary`` metric on the resident decode
        path, so the forward output is byte-identical — this is a pure per-token
        Python-overhead strip. Disarmed when ``self.hot_decode`` is False.
        """
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        backbone = self.session.model.backbone

        hot = _hot and self.hot_decode

        # On the hot path the embeddings / norm_f / lm_head are already resident
        # (the warmup prefill loaded them), so this loader call moves zero bytes —
        # skip its per-token Python cost too.
        if not hot:
            self._ensure_non_layer_weights()

        x = backbone.embeddings(mx.array(token_rows))

        attn_mask = create_attention_mask(x, cache[self._plan["fa_idx"]])
        ssm_mask = create_ssm_mask(x, cache[self._plan["ssm_idx"]])

        # HOT PATH: lean per-layer loop for the resident deferred decode steps.
        # No per-layer loader call (all tensors already resident -> zero bytes),
        # no per-layer event-dict churn, MoE dispatch is the deferred override
        # only (the hot path is only ever taken with the deferred override set).
        # Hoist the loop-invariant lookups out of the 108-iteration loop.
        if hot:
            layers = backbone.layers
            layer_specs = self._plan["layers"]
            moe_forward = _moe_forward_override
            page_experts = self.page_experts
            for layer_index, layer in enumerate(layers):
                spec = layer_specs[layer_index]
                cache_index = spec["cache_index"]
                c = cache[cache_index] if cache_index is not None else None
                block_type = spec["block_type"]
                mask = attn_mask if block_type == "*" else ssm_mask
                if page_experts and block_type == "E":
                    x = moe_forward(
                        layer_index,
                        layer,
                        x,
                        mask=mask,
                        cache=c,
                        events=events,
                        pass_kind=pass_kind,
                        token_step=token_step,
                    )
                else:
                    x = layer(x, mask=mask, cache=c)
            return x

        for layer_index, layer in enumerate(backbone.layers):
            spec = self._plan["layers"][layer_index]
            cache_index = spec["cache_index"]
            c = cache[cache_index] if cache_index is not None else None
            mask = attn_mask if spec["block_type"] == "*" else ssm_mask

            if self.page_experts and spec["block_type"] == "E":
                # Selective path: load this layer's base (gate/latent/shared)
                # then page only the experts the tokens route to, run exact.
                base_event = self._load_layer_base(layer_index)
                events.append(
                    {
                        "kind": "load",
                        "action": base_event.action,
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        **compact_event(base_event),
                    }
                )
                # Dispatch the MoE block forward. A caller-supplied override (the
                # DEFERRED path) wins outright. Otherwise, when a FIXED hot-set has
                # been built it takes priority: per token it runs the native on-GPU
                # gather on the resident-hit case and the exact page path on the
                # rare cold case. Failing that, fall back to the persistent-table
                # path (if enabled) or the per-token page-rebuild path (default).
                if _moe_forward_override is not None:
                    moe_forward = _moe_forward_override
                elif self._fixed_hotset:
                    moe_forward = self._nemotron_moe_layer_forward_fixed
                elif self.persist_expert_tables:
                    moe_forward = self._nemotron_moe_layer_forward_persistent
                else:
                    moe_forward = self._nemotron_moe_layer_forward
                x = moe_forward(
                    layer_index,
                    layer,
                    x,
                    mask=mask,
                    cache=c,
                    events=events,
                    pass_kind=pass_kind,
                    token_step=token_step,
                )
            else:
                # Resident path: load the full layer and run MLX's fused forward.
                full_event = self._load_layer_full(layer_index)
                events.append(
                    {
                        "kind": "load",
                        "action": full_event.action,
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        **compact_event(full_event),
                    }
                )
                x = layer(x, mask=mask, cache=c)

        return x

    def forward_logits(self, token_ids: list[int]) -> Any:
        """Token-exact logits for ``token_ids`` via the resident forward path."""
        self._install_fused_ssm_norm()
        cache = self.session.model.make_cache()
        h = self._stream_forward_tokens(
            [token_ids], cache=cache, events=[], pass_kind="prefill"
        )
        return self.session.model.lm_head(self.session.model.backbone.norm_f(h))

    def forward_logits_deferred(self, token_ids: list[int]) -> Any:
        """Token-exact logits via the DEFERRED path (single optimistic forward + cold-redo).

        Requires a built fixed hot-set. Runs ONE optimistic forward over the
        deferred MoE path (on-GPU remap, no per-layer eval), then evals the cold
        flag once. If any layer routed to a non-resident expert the optimistic
        result is invalid, so it is DISCARDED and recomputed on the exact synced
        fixed path (a fresh cache — this is a stateless prefill). All-resident:
        the deferred logits are returned directly, bit-identical to synced.
        """
        import mlx.core as mx

        if not self._fixed_hotset:
            raise ValueError(
                "forward_logits_deferred requires a fixed hot-set "
                "(call build_fixed_hotset first)"
            )
        model = self.session.model
        norm_f = model.backbone.norm_f
        lm_head = model.lm_head

        self._install_fused_ssm_norm()
        self._install_deferred_hotset()
        try:
            cache = model.make_cache()
            h = self._stream_forward_tokens_deferred(
                [token_ids], cache=cache, events=[], pass_kind="prefill"
            )
            logits = lm_head(norm_f(h))
            cold = self._deferred_cold_flag()
            # SINGLE sync: materialise the optimistic logits + the cold flag.
            mx.eval(logits, cold)
            if int(cold) == 0:
                self._deferred_stats["deferred_tokens"] += 1
                return logits
        finally:
            # The exact redo needs the model in stock state.
            self._uninstall_deferred_hotset()

        # COLD: discard the optimistic logits, recompute exactly. ``forward_logits``
        # uses a fresh cache and (since ``_fixed_hotset`` is set) the synced fixed
        # path, which falls back per-layer to the exact page path on cold experts.
        self._deferred_stats["cold_redos"] += 1
        exact = self.forward_logits(token_ids)
        mx.eval(exact)
        return exact

    def _decode_one_token_deferred(self, context_ids: list[int]) -> int:
        """One deferred decode step over a FRESH cache (eval-count harness helper).

        Prefills ``context_ids`` through the deferred path and returns the argmax
        next token, paying exactly the deferred path's per-token sync (the single
        cold-flag+argmax eval on the all-resident case). Used by the eval-count
        proof to measure deferred-vs-synced syncs on one comparable forward; not on
        the generation hot path (which threads ONE persistent cache).
        """
        import mlx.core as mx

        model = self.session.model
        norm_f = model.backbone.norm_f
        lm_head = model.lm_head
        self._install_deferred_hotset()
        try:
            cache = model.make_cache()
            h = self._stream_forward_tokens_deferred(
                [context_ids], cache=cache, events=[], pass_kind="decode"
            )
            logits = lm_head(norm_f(h))
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            cold = self._deferred_cold_flag()
            mx.eval(tok, cold)  # the single deferred per-token sync
            return int(tok[0])
        finally:
            self._uninstall_deferred_hotset()

    def _decode_one_token_synced(self, context_ids: list[int]) -> int:
        """One synced fixed-path decode step over a FRESH cache (eval-count baseline).

        Mirror of :meth:`_decode_one_token_deferred` but on the synced fixed path,
        so the eval-count proof compares like with like (same forward, same fresh
        cache). The synced path's per-MoE-layer ``mx.eval(inds)`` is exactly the
        per-token sync the deferred path removes.
        """
        import mlx.core as mx

        model = self.session.model
        norm_f = model.backbone.norm_f
        lm_head = model.lm_head
        h = self._stream_forward_tokens(
            [context_ids], cache=model.make_cache(), events=[], pass_kind="decode"
        )
        logits = lm_head(norm_f(h))
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        return int(tok[0])

    def generate_greedy(
        self, prompt_ids: list[int], max_tokens: int
    ) -> dict[str, Any]:
        """Autoregressive greedy decode over ONE persistent cache, with timing.

        Drives the streaming forward path through prefill + ``max_tokens`` decode
        steps against a SINGLE ``make_cache()`` object, so the hybrid cache
        (Mamba-2 SSM state on M/E layers, KV on '*' layers) advances exactly once
        per token — the same regime stock mlx_lm uses. Greedy (argmax) selection
        with no sampling/temperature makes the output a deterministic function of
        the logits, so it can be checked token-for-token against stock generation.

        Works for both ``page_experts`` modes (selective vs. resident) and for
        quantized / unquantized models, because it only touches the shared
        ``_stream_forward_tokens`` + ``norm_f`` + ``lm_head`` path.

        Returns a dict with the generated ``tokens`` (length ``max_tokens``),
        ``prefill_s`` / ``decode_s`` wall times, ``decode_tok_s`` throughput, the
        raw streaming ``events``, and a ``summary`` from :meth:`run_summary`.
        """
        import mlx.core as mx

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")

        self._install_fused_ssm_norm()
        model = self.session.model
        norm_f = model.backbone.norm_f
        lm_head = model.lm_head
        ev: list[dict[str, Any]] = []

        # ONE cache for the whole generation: prefill seeds it, every decode step
        # advances the SAME object (Mamba state + KV), mirroring stock decode.
        cache = model.make_cache()

        def _next_token(hidden: Any) -> int:
            # PRE-norm hidden -> final norm -> lm_head -> argmax over last position.
            logits = lm_head(norm_f(hidden))
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            return int(tok[0])

        # --- Prefill: process the full prompt in one pass, take the first token.
        prefill_start = time.perf_counter()
        hidden = self._stream_forward_tokens(
            [list(prompt_ids)], cache=cache, events=ev, pass_kind="prefill"
        )
        first = _next_token(hidden)  # _next_token's mx.eval forces prefill compute
        prefill_s = time.perf_counter() - prefill_start

        tokens: list[int] = [first]

        # --- Decode: feed back the previous token, one at a time, same cache.
        decode_start = time.perf_counter()
        nxt = first
        for step in range(max_tokens - 1):
            hidden = self._stream_forward_tokens(
                [[nxt]],
                cache=cache,
                events=ev,
                pass_kind="decode",
                token_step=step,
            )
            nxt = _next_token(hidden)
            tokens.append(nxt)
        decode_s = time.perf_counter() - decode_start

        decode_tok_s = max_tokens / decode_s if decode_s > 0 else float("inf")
        summary = self.run_summary(ev, tokens=max_tokens, elapsed_s=decode_s)
        return {
            "tokens": tokens,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "decode_tok_s": decode_tok_s,
            "events": ev,
            "summary": summary,
        }

    def generate_greedy_deferred(
        self, prompt_ids: list[int], max_tokens: int
    ) -> dict[str, Any]:
        """DEFERRED greedy decode — token-EXACT vs :meth:`generate_greedy`, fewer syncs.

        Same one-persistent-cache greedy loop as :meth:`generate_greedy`, but each
        step runs OPTIMISTICALLY through the deferred MoE path (on-GPU global->slot
        remap, ZERO per-layer eval) so the whole 108-layer forward is one lazy
        graph. The per-step sync collapses to ONE ``mx.eval`` of the next-token
        argmax PLUS the on-GPU cold flag (vs the synced fixed path's one eval per
        MoE layer). Requires a built fixed hot-set.

        Cold-redo (the exactness guarantee). A routed expert outside the resident
        set can't be Python-checked without a sync, so it's caught AFTER the fact:
        if the cold flag is set, the optimistic forward gathered a clamped wrong
        row and is INVALID. We snapshot the hybrid cache BEFORE every optimistic
        step; on cold we RESTORE that snapshot (undoing the bogus cache mutation,
        including non-trimmable Mamba SSM state — reusing the speculation rollback)
        and REDO the token on the exact synced fixed path (which falls back to the
        exact page path per cold layer). So every emitted token is bit-identical to
        plain greedy: all-resident tokens via the fast deferred forward, cold tokens
        via the proven-exact redo. Cold is rare at high K, so most tokens are fast.
        """
        import mlx.core as mx

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if not self._fixed_hotset:
            raise ValueError(
                "generate_greedy_deferred requires a fixed hot-set "
                "(call build_fixed_hotset first)"
            )

        self._install_fused_ssm_norm()
        model = self.session.model
        norm_f = model.backbone.norm_f
        lm_head = model.lm_head
        ev: list[dict[str, Any]] = []
        cache = model.make_cache()  # ONE cache for the whole generation

        def _argmax_last(hidden: Any) -> Any:
            # Lazy argmax over the last position (NOT evaluated here — the caller
            # evals it together with the cold flag as the single per-token sync).
            logits = lm_head(norm_f(hidden))
            return mx.argmax(logits[:, -1, :], axis=-1)

        def _exact_step(token_rows: list[list[int]], step: int) -> int:
            # Run ONE token on the exact synced fixed path. The deferred install
            # is removed first so the synced path sees the stock switch_mlp + gate,
            # then reinstalled so the next optimistic step is fast again.
            self._uninstall_deferred_hotset()
            try:
                hidden = self._stream_forward_tokens(
                    token_rows, cache=cache, events=ev, pass_kind="redo",
                    token_step=step,
                )
                tok = mx.argmax(lm_head(norm_f(hidden))[:, -1, :], axis=-1)
                mx.eval(tok)
                return int(tok[0])
            finally:
                self._install_deferred_hotset()

        def _optimistic_step(
            token_rows: list[list[int]], step: int, kind: str, hot: bool = False
        ) -> int:
            # One optimistic deferred step against the SHARED cache. Snapshot first
            # so a cold detection can roll the cache back to exactly pre-step. The
            # cold flag + argmax are evaluated together — the SINGLE per-token sync.
            # ``hot`` takes the lean per-layer loop (resident decode steps only).
            snapshot = self._snapshot_cache(cache)
            hidden = self._stream_forward_tokens_deferred(
                token_rows, cache=cache, events=ev, pass_kind=kind, token_step=step,
                _hot=hot,
            )
            nxt = _argmax_last(hidden)
            cold = self._deferred_cold_flag()
            mx.eval(nxt, cold)
            if int(cold) == 0:
                self._deferred_stats["deferred_tokens"] += 1
                return int(nxt[0])
            # COLD: undo the bogus cache mutation, redo the token exactly.
            self._deferred_stats["cold_redos"] += 1
            self._restore_cache(cache, snapshot)
            return _exact_step(token_rows, step)

        self._install_deferred_hotset()
        try:
            # --- Prefill (optimistic): process the whole prompt, take token 0.
            prefill_start = time.perf_counter()
            first = _optimistic_step([list(prompt_ids)], 0, "prefill")
            prefill_s = time.perf_counter() - prefill_start
            tokens: list[int] = [first]

            # --- Decode: feed back the previous token, one at a time, same cache.
            # The prefill above already made every layer resident, so decode steps
            # take the HOT PATH (lean per-layer loop, no no-op loader / event churn).
            decode_start = time.perf_counter()
            nxt = first
            for step in range(max_tokens - 1):
                nxt = _optimistic_step([[nxt]], step, "decode", hot=True)
                tokens.append(nxt)
            decode_s = time.perf_counter() - decode_start
        finally:
            self._uninstall_deferred_hotset()

        decode_tok_s = max_tokens / decode_s if decode_s > 0 else float("inf")
        summary = self.run_summary(ev, tokens=max_tokens, elapsed_s=decode_s)
        return {
            "tokens": tokens,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "decode_tok_s": decode_tok_s,
            "cold_redos": self._deferred_stats["cold_redos"],
            "deferred_tokens": self._deferred_stats["deferred_tokens"],
            # Cold experts substituted by a resident buddy at build (0 when
            # ``cold_substitution`` is off). >0 + cold_redos==0 == the unlock:
            # the deferred fast path never went cold despite missing experts.
            "cold_substitutions": self._deferred_stats["cold_substitutions"],
            "events": ev,
            "summary": summary,
        }

    def _snapshot_cache(self, cache: list[Any]) -> list[dict[str, Any]]:
        """Capture the full hybrid cache state for an exact speculative rollback.

        Delegates to the proven module-level :func:`snapshot_cache_states`, which
        copies EVERY attribute of each cache item (not just ``.state``) and forces
        a fresh ``mx.array`` copy of each array. That matters because the two cache
        types this runner uses mutate differently:

        * ``KVCache`` index-assigns keys/values into a PREALLOCATED buffer
          in place (``self.keys[..., prev:offset, :] = ...``), so a reference
          snapshot would be silently overwritten by a later forward — the forced
          copy defeats that, and copying ``offset`` directly avoids relying on the
          truncating ``.state`` getter.
        * Mamba ('M') ``ArraysCache`` REPLACES ``cache[0]``/``cache[1]`` with new
          arrays each step (functional), so its list is captured safely too.

        The batched verify advances the cache by k tokens; on partial/zero
        acceptance the recurrent Mamba SSM state cannot be trimmed after the fact,
        so restoring this snapshot and re-forwarding only the committed tokens is
        the simplest provably-exact rollback.
        """
        return snapshot_cache_states(cache)

    def _restore_cache(
        self, cache: list[Any], snapshot: list[dict[str, Any]]
    ) -> None:
        """Restore ``cache`` in place to a :meth:`_snapshot_cache` capture."""
        restore_cache_states(cache, snapshot)

    def generate_greedy_speculative(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        *,
        drafter: Any,
        block_size: int = 4,
    ) -> dict[str, Any]:
        """Batched speculative greedy decode — token-EXACT vs :meth:`generate_greedy`.

        A drafter proposes a block of ``block_size`` candidate tokens; this runner
        VERIFIES the whole block in ONE batched ``_stream_forward_tokens`` pass
        (not k separate decode forwards — that is the entire speedup), accepts the
        longest prefix that matches the model's own greedy, emits the accepted
        tokens plus one bonus token, and continues. It NEVER changes the output,
        so the returned ``tokens`` are bit-for-bit identical to plain greedy decode.

        Performance honesty: this is NOT a FLOP-saving compute-skip. A block costs
        one k-position verify forward PLUS a (j+1)-position commit re-forward
        (rollback below), i.e. MORE matmul than plain greedy's j+1 single-token
        forwards. The win is purely AMORTIZATION — verifying k tokens in one
        forward spreads the per-forward fixed cost and, decisively for this MoE,
        the per-token expert-LOAD/disk cost over the block's expert UNION (which
        widens sublinearly in k). So speculation helps only when per-token
        disk/expert-load dominates (cold/streaming); in a fully-warm,
        compute-bound regime (resident weight-page cache) it can be net SLOWER.
        Future work to cut the overhead: truncate the KV cache + capture
        per-position Mamba SSM state during verify, avoiding the commit re-forward.

        Verify off-by-one (the subtle part). At the top of each round the cache
        reflects the FULL processed context ``C = prompt + tokens`` and the model's
        greedy next token ``gp = greedy(C)`` is already known (carried over from
        the previous round's re-forward, or from prefill). The drafter sees the
        SAME context ``C``, so its proposal ``[d0, d1, ..., d_{k-1}]`` guesses the
        continuation of ``C`` — d0 is a guess for ``gp``. We forward that block in
        one pass; the argmax at position ``i`` is ``greedy([C, d0..d_i])`` — i.e.
        the prediction AFTER ``d_i``. So ``preds[i] == greedy([C, d0..d_i])`` and
        the token ``d_{i+1}`` must match ``preds[i]``. Acceptance:

        * ``d0`` is accepted iff ``gp == d0`` (``gp`` predates this forward);
        * ``d_i`` (i>=1) is accepted iff ``preds[i-1] == d_i``.

        Take the longest matching prefix length ``j`` (0 <= j <= k). The bonus
        token is the model's greedy right after the last accepted token:
        ``preds[j-1]`` for ``j>=1``, or ``gp`` itself for ``j==0``. The bonus is
        always exact, so it is always emitted. The committed block is
        ``[d0..d_{j-1}, bonus]``.

        Rollback rationale. The batched verify advanced the cache by k. We
        committed only ``j+1`` tokens, so we RESTORE the pre-verify snapshot and
        re-forward exactly ``[d0..d_{j-1}, bonus]`` in one pass. That single
        re-forward (a) advances the hybrid cache by precisely the committed
        tokens — leaving it identical to what plain greedy would hold after
        emitting them, the only provably-exact way to rewind the non-trimmable
        Mamba SSM state — and (b) its last-position argmax is
        ``greedy([C, d0..d_{j-1}, bonus])``, which is the next round's ``gp``.

        Returns the generated ``tokens`` (length ``max_tokens``) plus acceptance
        telemetry: ``blocks`` (verify rounds), ``accepted_tokens_total`` (draft
        tokens that matched, excluding bonus tokens), ``accepted_length_mean``
        (mean accepted draft tokens per block), and ``bonus_tokens`` (always one
        per committed block).
        """
        import mlx.core as mx

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if block_size < 1:
            raise ValueError("block_size must be positive")

        model = self.session.model
        norm_f = model.backbone.norm_f
        lm_head = model.lm_head
        ev: list[dict[str, Any]] = []

        # ONE persistent cache for the whole generation, exactly like plain
        # greedy: prefill seeds it and every committed token advances it once.
        cache = model.make_cache()

        def _argmax_all(hidden: Any) -> Any:
            # PRE-norm hidden -> norm_f -> lm_head -> argmax over the vocab at
            # EVERY position (batch row 0). Returns an int array of length T.
            logits = lm_head(norm_f(hidden))
            toks = mx.argmax(logits[0], axis=-1)
            mx.eval(toks)
            return toks

        # --- Prefill: process the whole prompt; ``gp`` is the model's pending
        # greedy continuation of the CURRENT cache context (initially the prompt).
        prefill_start = time.perf_counter()
        hidden = self._stream_forward_tokens(
            [list(prompt_ids)], cache=cache, events=ev, pass_kind="prefill"
        )
        # gp = greedy(prompt): argmax at the LAST prompt position. The cache now
        # reflects exactly ``prompt`` and gp is its greedy next token.
        gp = int(_argmax_all(hidden)[-1])
        prefill_s = time.perf_counter() - prefill_start

        tokens: list[int] = []

        blocks = 0
        accepted_tokens_total = 0
        bonus_tokens = 0
        full_accept_skips = 0
        rollback_blocks = 0

        # ``pending`` carries a bonus token that was EMITTED (already in
        # ``tokens``) but whose own cache-advance was deferred: the FULL-ACCEPT
        # SKIP keeps the verify's cache (which stops one short of the bonus) and
        # lets the bonus ride into the cache as the LEAD position of the NEXT
        # round's verify. So the invariant is generalized:
        #   cache reflects (prompt + tokens) MINUS ``pending``;
        #   ``gp`` is valid (== greedy of the cache context) ONLY when pending==[].
        # On the rollback path the lead is re-forwarded in the commit, leaving the
        # cache holding the full (prompt + tokens) and pending==[] again.
        pending: list[int] = []

        decode_start = time.perf_counter()
        # The drafter always sees context = prompt + tokens (ALL emitted tokens,
        # including any pending bonus), so its first proposal token guesses the
        # greedy continuation of that full context.
        while len(tokens) < max_tokens:
            context = list(prompt_ids) + tokens
            proposal = [int(t) for t in drafter.propose(context, block_size)]
            proposal = proposal[:block_size]

            if not proposal:
                # No candidate. First settle any pending bonus into the cache (its
                # deferred advance), refreshing ``gp``; then take one plain greedy
                # step. Both stay exact.
                if pending:
                    hidden = self._stream_forward_tokens(
                        [pending], cache=cache, events=ev, pass_kind="decode",
                        token_step=blocks,
                    )
                    gp = int(_argmax_all(hidden)[-1])
                    pending = []
                tokens.append(gp)
                if len(tokens) >= max_tokens:
                    break
                hidden = self._stream_forward_tokens(
                    [[gp]], cache=cache, events=ev, pass_kind="decode",
                    token_step=blocks,
                )
                gp = int(_argmax_all(hidden)[-1])
                continue

            blocks += 1

            # --- Batched verify: forward the WHOLE block (LEAD + proposal) in one
            # pass. ``lead`` is the carried pending bonus (or empty); it is NOT in
            # the cache, so forwarding it here both settles it AND gives the
            # proposal's offset-0 prediction (greedy after the bonus) without a
            # precomputed ``gp``. (Speedup: k+lead candidates verified by ONE
            # forward, not k separate decode forwards.)
            lead = list(pending)
            block = lead + proposal
            snapshot = self._snapshot_cache(cache)
            verify_hidden = self._stream_forward_tokens(
                [block], cache=cache, events=ev, pass_kind="verify",
                token_step=blocks - 1,
            )
            # preds[i] = greedy(C' + block[:i+1]) = the prediction AFTER block[i],
            # where C' is the cache context (prompt+tokens minus the lead).
            preds = [int(t) for t in _argmax_all(verify_hidden).tolist()]

            # Acceptance of ``proposal``. proposal[i] sits at block index
            # ``len(lead)+i``; its predictor is block index ``len(lead)+i-1``:
            #   * with a lead -> preds[len(lead)+i-1] (== preds[i] for a 1-tok lead);
            #   * without a lead -> ``gp`` for i==0 (gp predates this forward),
            #     else preds[i-1].
            def _expected(i: int) -> int:
                idx = len(lead) + i - 1
                if idx < 0:
                    return gp
                return preds[idx]

            j = 0
            while j < len(proposal):
                if proposal[j] != _expected(j):
                    break
                j += 1

            # Bonus = greedy right after the last accepted token in ``block``. The
            # last accepted block token is at index ``len(lead)+j-1`` (the j-th
            # proposal token, or the lead itself when j==0 and a lead exists), so
            # its prediction is preds[len(lead)+j-1]; with no lead and j==0 it is
            # ``gp`` (greedy of C'). The bonus is always exact, so always emitted.
            bonus_idx = len(lead) + j - 1
            bonus = gp if bonus_idx < 0 else preds[bonus_idx]
            committed = proposal[:j] + [bonus]  # NEW tokens this round (lead was already emitted)

            accepted_tokens_total += j
            bonus_tokens += 1

            # Truncate to the token budget. If truncation lands inside the block we
            # finish after committing (the refreshed gp / pending is irrelevant).
            if len(tokens) + len(committed) > max_tokens:
                tokens.extend(committed[: max_tokens - len(tokens)])
                break

            tokens.extend(committed)

            if j == len(proposal):
                # --- FULL ACCEPT: SKIP the rollback + commit re-forward. The verify
                # forwarded exactly [lead + d0..d_{k-1}] from the pre-verify cache,
                # so the cache now reflects (prompt + tokens) MINUS the new bonus —
                # IDENTICAL to what plain greedy holds after emitting every accepted
                # token (all of which equal the model's greedy, since they matched).
                # The bonus = greedy after the last accepted token = the next
                # pending token. So keeping the verify cache and carrying the bonus
                # as ``pending`` yields the exact same cache state AND emitted tokens
                # the rollback+commit path would, with NO re-forward.
                full_accept_skips += 1
                pending = [bonus]
                if len(tokens) >= max_tokens:
                    break
                continue

            # --- PARTIAL / ZERO accept: rollback + commit re-forward. The verify
            # advanced the cache by len(block); we committed only the accepted
            # prefix + bonus. Restore the pre-verify snapshot (cache -> C') and
            # re-forward EXACTLY [lead + proposal[:j] + bonus] — the lead is
            # included because C' does NOT contain it. That leaves the hybrid cache
            # holding precisely (prompt + tokens) (the only exact way to rewind the
            # non-trimmable Mamba SSM state), its last-position argmax is the next
            # round's gp, and the pending bonus is now resolved into the cache.
            rollback_blocks += 1
            self._restore_cache(cache, snapshot)
            if len(tokens) >= max_tokens:
                pending = []
                break
            hidden = self._stream_forward_tokens(
                [lead + committed], cache=cache, events=ev, pass_kind="commit",
                token_step=blocks - 1,
            )
            gp = int(_argmax_all(hidden)[-1])
            pending = []

        decode_s = time.perf_counter() - decode_start

        tokens = tokens[:max_tokens]
        accepted_length_mean = (
            accepted_tokens_total / blocks if blocks else 0.0
        )
        decode_tok_s = (
            len(tokens) / decode_s if decode_s > 0 else float("inf")
        )
        summary = self.run_summary(ev, tokens=len(tokens), elapsed_s=decode_s)
        return {
            "tokens": tokens,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "decode_tok_s": decode_tok_s,
            "blocks": blocks,
            "accepted_tokens_total": accepted_tokens_total,
            "accepted_length_mean": accepted_length_mean,
            "bonus_tokens": bonus_tokens,
            # Blocks that took the full-accept SKIP (verify-only, no re-forward) vs
            # the rollback+commit path. A truncating final block may end before
            # being classified, so these need not sum to ``blocks``.
            "full_accept_skips": full_accept_skips,
            "rollback_blocks": rollback_blocks,
            "events": ev,
            "summary": summary,
        }

    def generate_greedy_speculative_deferred(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        *,
        drafter: Any,
        block_size: int = 4,
    ) -> dict[str, Any]:
        """Speculative greedy decode whose VERIFY runs on the DEFERRED fast path.

        Composes the runner's two token-exact mechanisms:

        * :meth:`generate_greedy_speculative` — the drafter proposes a block of
          ``block_size`` candidates, the runner verifies the WHOLE block in ONE
          batched forward, accepts the longest greedy-matching prefix + a bonus,
          rolls the cache back and re-forwards exactly the committed tokens. The
          acceptance / bonus / rollback logic here is identical to that method —
          only the VERIFY forward is swapped.
        * :meth:`generate_greedy_deferred` — the deferred fixed-hot-set forward
          (``_stream_forward_tokens_deferred``): on-GPU global->slot remap, ZERO
          per-layer eval, the whole 108-layer pass composes into ONE lazy graph
          evaluated once at ``lm_head``. A residency miss can't be Python-checked
          without a sync, so it is caught AFTER the fact via the lazy cold flag.

        THE COMPOSE POINT. The batched k-token verify forward goes through
        ``_stream_forward_tokens_deferred`` (the deferred MoE dispatch) instead of
        the synced ``_stream_forward_tokens``. On a bandwidth-bound forward this is
        the win: verifying k tokens in one lazy pass amortises the per-token
        active-weight read over the block's expert UNION (which widens sublinearly
        in k), and pays ONE sync for the whole block (the verify argmaxes + the
        cold flag), not one eval per MoE layer per position.

        COLD-IN-VERIFY-BLOCK handling (the exactness subtlety unique to this
        compose). The deferred verify's cold flag is the OR over ALL k verify
        positions x ALL MoE layers (the remap gates append a per-layer flag and
        :meth:`_deferred_cold_flag` OR-reduces them; the verify does NOT reset the
        accumulator mid-block). So a single non-resident routed expert at ANY
        position makes ``verify_cold == 1``. When that fires, the optimistic
        per-position predictions are gathered from clamped WRONG rows and are
        INVALID, so we DISCARD them, restore the pre-verify snapshot, and REDO the
        verify of the SAME block on the proven-exact synced path
        (:meth:`_stream_forward_tokens` -> :meth:`_nemotron_moe_layer_forward_fixed`,
        which itself falls back per-layer to the exact page path on a cold expert).
        The synced re-verify produces the exact per-position predictions, and
        acceptance proceeds from those. (The verify always advances the cache by k
        and is ALWAYS rolled back afterward — see below — so the cache mutation
        during either the optimistic or the synced verify is irrelevant; only the
        PREDICTIONS matter for acceptance, and those are made exact.)

        Every CACHE-ADVANCING forward (prefill, the no-proposal single step, and
        the commit re-forward) is itself run through the deferred-with-cold-redo
        helper :func:`_advance` below — snapshot -> optimistic deferred -> if cold
        restore + exact synced redo — exactly as :meth:`generate_greedy_deferred`
        does. So the persistent cache only ever advances by EXACT tokens, and the
        re-forward's last-position argmax (the next round's ``gp``) is exact.

        EXACTNESS. Speculation only ever emits tokens matching the model's own
        greedy (the bonus is the exact greedy next token). The deferred verify is
        exact on all-resident blocks (the on-GPU ``mx.take`` remap reproduces the
        numpy remap bit-for-bit — proven == stock == page) and falls back to the
        exact synced verify on cold blocks. The cache-advancing forwards are exact
        by the same cold-redo. So the emitted ``tokens`` are bit-identical to plain
        :meth:`generate_greedy` (== stock). Requires a built fixed hot-set.

        Returns the same shape as :meth:`generate_greedy_speculative` plus
        ``verify_cold_blocks`` (verify blocks that hit the synced-verify fallback)
        and ``deferred_verify_blocks`` (verify blocks served purely by the deferred
        forward).
        """
        import mlx.core as mx

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if not self._fixed_hotset:
            raise ValueError(
                "generate_greedy_speculative_deferred requires a fixed hot-set "
                "(call build_fixed_hotset first)"
            )

        model = self.session.model
        norm_f = model.backbone.norm_f
        lm_head = model.lm_head
        ev: list[dict[str, Any]] = []

        # ONE persistent cache for the whole generation, exactly like plain greedy
        # and the synced speculative path: prefill seeds it; every COMMITTED token
        # advances it once.
        cache = model.make_cache()

        verify_cold_blocks = 0
        deferred_verify_blocks = 0

        def _argmax_all_lazy(hidden: Any) -> Any:
            # PRE-norm hidden -> norm_f -> lm_head -> argmax over the vocab at EVERY
            # position (batch row 0). LAZY: not evaluated here so the verify can
            # eval it together with the cold flag as the single per-block sync.
            return mx.argmax(lm_head(norm_f(hidden))[0], axis=-1)

        def _synced_verify_preds(token_rows: list[list[int]], step: int) -> list[int]:
            # EXACT synced re-verify of a cold block. The deferred install is
            # removed first so the synced fixed path sees the stock switch_mlp +
            # gate (its ``_nemotron_moe_layer_forward_fixed`` gates-and-evals to
            # read GLOBAL ids — it must NOT see the on-GPU remap gate), then
            # reinstalled so the next optimistic forward is fast again. The cache
            # mutation here is discarded by the caller's rollback; only the
            # per-position predictions are used.
            self._uninstall_deferred_hotset()
            try:
                hidden = self._stream_forward_tokens(
                    token_rows, cache=cache, events=ev, pass_kind="verify-redo",
                    token_step=step,
                )
                preds = mx.argmax(lm_head(norm_f(hidden))[0], axis=-1)
                mx.eval(preds)
                return [int(t) for t in preds.tolist()]
            finally:
                self._install_deferred_hotset()

        def _verify_preds(
            token_rows: list[list[int]], step: int, snapshot: list[dict[str, Any]]
        ) -> list[int]:
            # Batched verify of the proposed block THROUGH THE DEFERRED FORWARD.
            # One optimistic deferred pass over the k-token block; the verify
            # argmaxes (all positions) and the OR-over-block cold flag are evaluated
            # TOGETHER — the single per-block sync. All-resident: the deferred preds
            # are exact (on-GPU remap == numpy remap) and returned directly. Cold:
            # the preds are invalid, so restore the pre-verify ``snapshot`` (taken by
            # the caller before this forward) and redo the verify on the exact
            # synced path. Either way the caller rolls the cache back to ``snapshot``
            # after acceptance, so this forward's cache mutation is discarded.
            #
            # HOT PATH: the verify only ever runs AFTER the prefill ``_advance`` has
            # made every layer resident, so its per-layer loader call moves zero
            # bytes — take the lean loop (overhead strip only, width-independent).
            nonlocal verify_cold_blocks, deferred_verify_blocks
            verify_hidden = self._stream_forward_tokens_deferred(
                token_rows, cache=cache, events=ev, pass_kind="verify",
                token_step=step, _hot=True,
            )
            preds_lazy = _argmax_all_lazy(verify_hidden)
            cold = self._deferred_cold_flag()
            mx.eval(preds_lazy, cold)  # SINGLE per-block sync (preds + cold flag)
            if int(cold) == 0:
                deferred_verify_blocks += 1
                return [int(t) for t in preds_lazy.tolist()]
            # COLD verify block: discard the optimistic preds, rewind to the
            # pre-verify snapshot, re-verify exactly so the synced re-verify starts
            # from the same context the deferred verify saw.
            verify_cold_blocks += 1
            self._restore_cache(cache, snapshot)
            return _synced_verify_preds(token_rows, step)

        def _advance(
            token_rows: list[list[int]], step: int, kind: str, hot: bool = False
        ) -> int:
            # Advance the PERSISTENT cache by ``token_rows`` and return the
            # last-position greedy token, EXACTLY (deferred-with-cold-redo, the same
            # contract as generate_greedy_deferred). Snapshot first so a cold
            # detection can roll the cache back to exactly pre-step; the cold flag +
            # argmax are evaluated together — the single sync. Used for prefill, the
            # no-proposal single step, and the commit re-forward — every forward
            # that MUTATES the persistent cache, so the cache only ever advances by
            # exact tokens. ``hot`` takes the lean loop (post-prefill steps only).
            snapshot = self._snapshot_cache(cache)
            hidden = self._stream_forward_tokens_deferred(
                token_rows, cache=cache, events=ev, pass_kind=kind, token_step=step,
                _hot=hot,
            )
            tok = mx.argmax(lm_head(norm_f(hidden))[:, -1, :], axis=-1)
            cold = self._deferred_cold_flag()
            mx.eval(tok, cold)
            if int(cold) == 0:
                self._deferred_stats["deferred_tokens"] += 1
                return int(tok[0])
            # COLD: undo the bogus cache mutation, redo exactly on the synced path.
            self._deferred_stats["cold_redos"] += 1
            self._restore_cache(cache, snapshot)
            self._uninstall_deferred_hotset()
            try:
                hidden = self._stream_forward_tokens(
                    token_rows, cache=cache, events=ev, pass_kind="redo",
                    token_step=step,
                )
                tok = mx.argmax(lm_head(norm_f(hidden))[:, -1, :], axis=-1)
                mx.eval(tok)
                return int(tok[0])
            finally:
                self._install_deferred_hotset()

        self._install_deferred_hotset()
        try:
            # --- Prefill: process the whole prompt; ``gp`` is the model's pending
            # greedy continuation of the CURRENT cache context (initially prompt).
            prefill_start = time.perf_counter()
            gp = _advance([list(prompt_ids)], 0, "prefill")
            prefill_s = time.perf_counter() - prefill_start

            tokens: list[int] = []
            blocks = 0
            accepted_tokens_total = 0
            bonus_tokens = 0
            full_accept_skips = 0
            rollback_blocks = 0

            # ``pending`` carries a bonus token that was EMITTED (already in
            # ``tokens``) but whose own cache-advance was deferred by the FULL-
            # ACCEPT SKIP below: the verify cache stops one short of the bonus, and
            # the bonus rides into the cache as the LEAD position of the NEXT
            # round's verify. Generalized invariant:
            #   cache reflects (prompt + tokens) MINUS ``pending``;
            #   ``gp`` is valid (== greedy of the cache context) ONLY when pending==[].
            pending: list[int] = []

            decode_start = time.perf_counter()
            # The drafter always sees context = prompt + tokens (ALL emitted
            # tokens, including any pending bonus), so its first proposal token
            # guesses the greedy continuation of that full context.
            while len(tokens) < max_tokens:
                context = list(prompt_ids) + tokens
                proposal = [int(t) for t in drafter.propose(context, block_size)]
                proposal = proposal[:block_size]

                if not proposal:
                    # No candidate. First settle any pending bonus into the cache
                    # (its deferred advance) to refresh ``gp``, then emit gp + step.
                    if pending:
                        gp = _advance([pending], blocks, "decode", hot=True)
                        pending = []
                    tokens.append(gp)
                    if len(tokens) >= max_tokens:
                        break
                    gp = _advance([[gp]], blocks, "decode", hot=True)
                    continue

                blocks += 1

                # Snapshot the cache BEFORE the verify advances it. This one
                # snapshot serves BOTH the cold re-verify inside ``_verify_preds``
                # (restore -> exact synced verify) AND the partial-accept commit
                # rollback below.
                snapshot = self._snapshot_cache(cache)

                # --- Batched verify through the DEFERRED forward (the compose
                # point), over LEAD + proposal. ``lead`` is the carried pending
                # bonus (or empty); it is NOT in the cache, so forwarding it here
                # both settles it AND gives the proposal's offset-0 prediction
                # (greedy after the bonus) without a precomputed ``gp``. After
                # ``_verify_preds`` returns, the cache holds C' + (lead+proposal)
                # in BOTH the all-resident and cold-re-verify cases (the cold path
                # restored to ``snapshot`` then synced-forwarded the same block),
                # so the post-block cache state is exact regardless of branch.
                lead = list(pending)
                block = lead + proposal
                preds = _verify_preds([block], blocks - 1, snapshot)

                # Acceptance of ``proposal``. proposal[i] is at block index
                # ``len(lead)+i``; its predictor is block index ``len(lead)+i-1``:
                #   * with a lead -> preds[len(lead)+i-1] (== preds[i] for 1-tok lead);
                #   * without a lead -> ``gp`` for i==0, else preds[i-1].
                def _expected(i: int) -> int:
                    idx = len(lead) + i - 1
                    if idx < 0:
                        return gp
                    return preds[idx]

                j = 0
                while j < len(proposal):
                    if proposal[j] != _expected(j):
                        break
                    j += 1

                # Bonus = greedy right after the last accepted block token (index
                # len(lead)+j-1), or ``gp`` when no lead and j==0. Always exact.
                bonus_idx = len(lead) + j - 1
                bonus = gp if bonus_idx < 0 else preds[bonus_idx]
                committed = proposal[:j] + [bonus]  # NEW tokens (lead already emitted)

                accepted_tokens_total += j
                bonus_tokens += 1

                # Truncate to the token budget; if truncation lands inside the
                # block we finish after committing (refreshed gp/pending irrelevant).
                if len(tokens) + len(committed) > max_tokens:
                    tokens.extend(committed[: max_tokens - len(tokens)])
                    break

                tokens.extend(committed)

                if j == len(proposal) and len(block) >= 2:
                    # --- FULL ACCEPT: SKIP the rollback + commit re-forward. The
                    # verify forwarded exactly [lead + d0..d_{k-1}] from the pre-
                    # verify cache, so the cache now reflects (prompt + tokens)
                    # MINUS the new bonus — IDENTICAL to what plain greedy holds
                    # after emitting every accepted token (all matched the model's
                    # greedy). This holds on the all-resident deferred verify (its
                    # cache is bit-exact to the synced/commit path, the same
                    # guarantee deferred decode relies on) AND on a cold block (the
                    # synced re-verify advanced the cache to the same point). The
                    # bonus = greedy after the last accepted token = the next
                    # pending token. So keeping the verify cache + carrying the
                    # bonus as ``pending`` reproduces the rollback+commit path's
                    # cache state AND emitted tokens with NO re-forward.
                    #
                    # WHY ``len(block) >= 2`` (the deferred-only guard). The kept
                    # cache here is a LAZY deferred forward; the NEXT round forwards
                    # LEAD+proposal on top of it (another lazy pass). Chaining lazy
                    # deferred forwards on a kept cache is bit-exact EXCEPT for one
                    # MLX hazard: keeping a SINGLE-token deferred forward and then
                    # extending it with a >=2-token deferred forward yields
                    # non-deterministic reads (the 1->>=2 in-place KV growth on a
                    # not-yet-consumed buffer). So we never KEEP a 1-token verify:
                    # if the whole block is 1 token (no lead + 1-token proposal) we
                    # fall through to the proven rollback+commit instead. After ANY
                    # skip ``pending`` is set, so the next block carries a lead and
                    # is >=2 tokens — the chain stays in the safe regime. (A 1-token
                    # block only occurs right after a rollback / at the start, where
                    # rollback is cheap and exact anyway. The synced speculative
                    # path has no such hazard and skips unconditionally.)
                    full_accept_skips += 1
                    pending = [bonus]
                    if len(tokens) >= max_tokens:
                        break
                    continue

                # --- PARTIAL / ZERO accept (or a 1-token full-accept block, kept
                # off the skip path by the guard above): rollback + commit. Restore
                # the pre-verify snapshot (cache -> C') and re-forward EXACTLY
                # [lead + proposal[:j] + bonus] via ``_advance`` (so the advance is
                # itself exact even if a committed token is cold); the lead is
                # included because C' does NOT contain it. That leaves the hybrid
                # cache holding precisely (prompt + tokens), its last-position
                # argmax is the next round's gp, and the pending bonus is resolved.
                # (``_verify_preds`` on a cold block already advanced the cache by
                # len(block); restoring ``snapshot`` here rewinds either case.)
                rollback_blocks += 1
                self._restore_cache(cache, snapshot)
                if len(tokens) >= max_tokens:
                    pending = []
                    break
                gp = _advance([lead + committed], blocks - 1, "commit", hot=True)
                pending = []
            decode_s = time.perf_counter() - decode_start
        finally:
            # The model must be left in stock state for any later exact path.
            self._uninstall_deferred_hotset()

        tokens = tokens[:max_tokens]
        accepted_length_mean = (
            accepted_tokens_total / blocks if blocks else 0.0
        )
        decode_tok_s = (
            len(tokens) / decode_s if decode_s > 0 else float("inf")
        )
        summary = self.run_summary(ev, tokens=len(tokens), elapsed_s=decode_s)
        return {
            "tokens": tokens,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "decode_tok_s": decode_tok_s,
            "blocks": blocks,
            "accepted_tokens_total": accepted_tokens_total,
            "accepted_length_mean": accepted_length_mean,
            "bonus_tokens": bonus_tokens,
            # Blocks that took the full-accept SKIP (verify-only, no re-forward) vs
            # the rollback+commit path. A truncating final block may end before
            # being classified, so these need not sum to ``blocks``.
            "full_accept_skips": full_accept_skips,
            "rollback_blocks": rollback_blocks,
            "verify_cold_blocks": verify_cold_blocks,
            "deferred_verify_blocks": deferred_verify_blocks,
            # Cold experts substituted by a resident buddy at build (0 when
            # ``cold_substitution`` is off). With it on + verify_cold_blocks==0 the
            # wide verify blocks never went cold -> no exact-verify fallback.
            "cold_substitutions": self._deferred_stats["cold_substitutions"],
            "events": ev,
            "summary": summary,
        }

    def run_summary(
        self,
        events: list[dict[str, Any]],
        *,
        tokens: int | None = None,
        elapsed_s: float | None = None,
    ) -> dict[str, Any]:
        """Aggregate streaming telemetry for a run into a flat report dict.

        Pure aggregation over ``events`` (the side-channel list filled by
        ``_stream_forward_tokens``) plus current session / OS state — no global
        mutation. Byte/second fields read the ``compact_event`` shape:
        ``nbytes_loaded`` for loaded bytes, ``seconds`` for load wall time.

        ``peak_rss_bytes`` is ``ru_maxrss`` from ``getrusage``; the UNIT is
        platform-specific (BYTES on macOS/Darwin, KiB on Linux) — on this Darwin
        host it is bytes.
        """
        import resource

        load_events = [e for e in events if e.get("kind") == "load"]
        loaded_bytes = sum(int(e.get("nbytes_loaded", 0) or 0) for e in load_events)
        load_seconds = sum(float(e.get("seconds", 0.0) or 0.0) for e in load_events)
        tok_per_s = (
            tokens / elapsed_s
            if tokens is not None and elapsed_s
            else None
        )
        return {
            "loaded_bytes": loaded_bytes,
            "load_seconds": load_seconds,
            "load_events": len(load_events),
            "resident_bytes": self.session.resident_bytes,
            # ru_maxrss: BYTES on Darwin (this host), KiB on Linux.
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "tokens": tokens,
            "tok_per_s": tok_per_s,
            # Resident expert-cache signal (additive). The loader getters return
            # 0 / None when no weight-page cache is attached, so these degrade
            # cleanly with the cache off. NOTE: ``loaded_bytes`` above is computed
            # from the full selected expert set per load and does NOT drop on a
            # cache hit — the real hit/miss signal lives here.
            "weight_page_resident_bytes": self.session.loader.weight_page_resident_bytes,
            "weight_page_summary": self.session.loader.weight_page_summary(),
            # Fixed-hot-set signal (additive). ``native_hits`` is MoE passes
            # served by the resident set (native on-GPU gather, no per-token
            # assembly); ``cold_fallbacks`` is passes with a non-resident routed
            # expert that took the exact page path. Both are 0 when the feature is
            # off (no ``build_fixed_hotset``), so this degrades cleanly.
            "hotset_native_hits": self._hotset_stats["native_hits"],
            "hotset_cold_fallbacks": self._hotset_stats["cold_fallbacks"],
            "hotset_layers": len(self._fixed_hotset),
        }


class Qwen35MoeStreamingForwardRunner(StreamingChatRunner):
    """Generate with Qwen3.5-MoE through SmartTensor layer streaming."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        evaluate: bool = True,
        retain_layers: set[int] | None = None,
        resident_budget_bytes: int | None = None,
        backend: str = "native",
        clear_on_evict: bool = True,
        pin_policy: str = "all",
        warm_embeddings: bool = False,
        expert_prefetch: str = "off",
        expert_prefetch_cap: int = 32,
        expert_reuse: bool = False,
        expert_reuse_cap: int = 16,
        native_layers: int = 0,
    ) -> None:
        if pin_policy not in {"all", "phase"}:
            raise ValueError("pin_policy must be 'all' or 'phase'")
        if native_layers < 0:
            raise ValueError("native_layers must be non-negative")
        if warm_embeddings and pin_policy != "phase":
            raise ValueError("warm_embeddings only applies to pin_policy='phase'")
        if expert_prefetch not in {"off", "previous"}:
            raise ValueError("expert_prefetch must be 'off' or 'previous'")
        if expert_prefetch_cap < 1:
            raise ValueError("expert_prefetch_cap must be positive")
        if expert_reuse_cap < 1:
            raise ValueError("expert_reuse_cap must be positive")
        self.session = MlxModelSession(
            model_dir,
            evaluate=evaluate,
            retain_layers=retain_layers,
            resident_budget_bytes=None,
            backend=backend,
            clear_on_evict=clear_on_evict,
            pin_policy=pin_policy,
            warm_embeddings=warm_embeddings,
        )
        self.pin_policy = pin_policy
        self.warm_embeddings = warm_embeddings
        self.warm_output = pin_policy == "phase"
        self.expert_prefetch = expert_prefetch
        self.expert_prefetch_cap = expert_prefetch_cap
        self.expert_reuse = expert_reuse
        self.expert_reuse_cap = expert_reuse_cap
        self._expert_history: dict[int, list[int]] = {}
        self._pending_expert_prefetch: dict[str, Any] | None = None
        self._prefetch_stats = ExpertPrefetchStats()
        self._expert_table_cache: dict[int, dict[str, Any]] = {}
        self._expert_cache_bytes = 0
        self.loader_executor = ThreadPoolExecutor(max_workers=1)
        self._embed_prefix = "language_model.model.embed_tokens"
        if self.session.config.get("model_type") != "qwen3_5_moe":
            raise ValueError("Qwen35MoeStreamingForwardRunner currently supports model_type=qwen3_5_moe only")
        if retain_layers is not None:
            self.base_retain_layers = set(retain_layers)
        elif resident_budget_bytes is not None:
            self.base_retain_layers = select_qwen_base_layers_for_budget(
                self.session.loader.manifest,
                resident_budget_bytes,
                top_k=self.session.model.language_model.args.num_experts_per_tok,
            )
        else:
            self.base_retain_layers = set()

        total_layers = len(self.session.loader.manifest.layers)
        if native_layers > total_layers:
            raise ValueError(
                f"native_layers {native_layers} exceeds layer count {total_layers}"
            )
        # Native-resident layers run MLX's fused DecoderLayer forward directly:
        # the full layer (base + all experts) is loaded once and kept, so the
        # per-layer selective-expert path, clear, and evict are all skipped.
        # The remaining layers degrade gracefully to the streaming path.
        self.native_layers: set[int] = set(range(native_layers))
        self._native_loaded: set[int] = set()

        from mlx_lm.utils import load_tokenizer

        self.tokenizer = load_tokenizer(self.session.model_dir)

    def close(self) -> None:
        self.loader_executor.shutdown(wait=True)
        self._expert_table_cache.clear()
        self._expert_cache_bytes = 0
        self.session.close()

    def _reset_stream_state(self) -> None:
        self._expert_history.clear()
        self._pending_expert_prefetch = None
        self._prefetch_stats = ExpertPrefetchStats()
        self._expert_table_cache.clear()
        self._expert_cache_bytes = 0

    def _expert_telemetry(self) -> dict[str, Any] | None:
        if self.expert_prefetch != "off" or self.expert_reuse:
            return self._prefetch_stats.to_dict()
        return None

    def _embed_module(self) -> Any:
        return self.session.model.language_model.model.embed_tokens

    def _stream_forward_tokens(
        self,
        token_rows: list[list[int]],
        *,
        cache: list[Any],
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None = None,
        manage_embedding: bool = True,
    ) -> Any:
        import mlx.core as mx
        from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

        core = self.session.model.language_model.model
        if self.pin_policy == "phase" and manage_embedding:
            load_embedding, local_token_rows = self._load_embedding_slices(token_rows)
            events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_embedding)})
            inputs = mx.array(local_token_rows)
            try:
                x = core.embed_tokens(inputs)
                mx.eval(x)
            finally:
                self._clear_embedding_slices()
                evict_embedding = MlxStreamEvent(
                    action="evict-embedding-slices",
                    seconds=0.0,
                    resident_bytes=self.session.resident_bytes,
                    requested=load_embedding.requested,
                    evicted=load_embedding.loaded,
                )
                self.session.events.append(evict_embedding)
                events.append({"kind": "evict", "pass": pass_kind, "token_step": token_step, **compact_event(evict_embedding)})
        else:
            inputs = mx.array(token_rows)
            x = core.embed_tokens(inputs)
            mx.eval(x)

        fa_mask = create_attention_mask(x, cache[core.fa_idx])
        ssm_mask = create_ssm_mask(x, cache[core.ssm_idx])

        for layer_index, (layer, layer_cache) in enumerate(zip(core.layers, cache)):
            self._maybe_submit_expert_prefetch(layer_index)

            # Native-resident layer: full layer loaded once, run MLX's fused
            # DecoderLayer forward; no selective-expert path, clear, or evict.
            # x stays lazy across native layers (the resident pipeline) and is
            # evaluated once after the loop / by the caller.
            if layer_index in self.native_layers:
                if layer_index not in self._native_loaded:
                    self._load_native_qwen_layer(layer_index)
                mask = ssm_mask if layer.is_linear else fa_mask
                x = layer(x, mask, layer_cache)
                continue

            load_event = self._load_qwen_layer_base(layer_index)
            events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_event)})

            compute_started = time.perf_counter()
            mask = ssm_mask if layer.is_linear else fa_mask
            try:
                x = self._qwen_layer_forward_selective_experts(
                    layer_index,
                    layer,
                    x,
                    mask=mask,
                    cache=layer_cache,
                    events=events,
                    pass_kind=pass_kind,
                    token_step=token_step,
                )
            finally:
                self._clear_qwen_selected_experts(layer)
            events.append(
                {
                    "kind": "compute",
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "seconds": time.perf_counter() - compute_started,
                    "hidden_shape": list(x.shape),
                    "is_linear": bool(layer.is_linear),
                }
            )

            if layer_index in self.base_retain_layers:
                evict_event = MlxStreamEvent(
                    action="retain-layer-base",
                    layer=layer_index,
                    seconds=0.0,
                    resident_bytes=self.session.resident_bytes,
                    requested=self.session.loader.manifest.layers[layer_index].tensor_names,
                )
                self.session.events.append(evict_event)
            else:
                evict_event = self.session.evict_layer(layer_index)
            events.append({"kind": "evict", "pass": pass_kind, "token_step": token_step, **compact_event(evict_event)})

        return x

    def _load_qwen_layer_base(self, layer_index: int) -> MlxStreamEvent:
        layer = self.session.loader.manifest.layers[layer_index]
        names = tuple(name for name in layer.tensor_names if ".mlp.switch_mlp." not in name)
        return self.session._load_into_model(names, action="load-layer-base", layer=layer_index)

    def _load_native_qwen_layer(self, layer_index: int) -> MlxStreamEvent:
        # Load the FULL layer (base + every expert) once and keep it resident;
        # native layers are never evicted, so MLX's fused forward can run.
        layer = self.session.loader.manifest.layers[layer_index]
        event = self.session._load_into_model(
            layer.tensor_names, action="load-native-layer", layer=layer_index
        )
        self._native_loaded.add(layer_index)
        return event

    def _qwen_layer_forward_selective_experts(
        self,
        layer_index: int,
        layer: Any,
        x: Any,
        *,
        mask: Any,
        cache: Any,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        if layer.is_linear:
            residual = layer.linear_attn(layer.input_layernorm(x), mask, cache)
        else:
            residual = layer.self_attn(layer.input_layernorm(x), mask, cache)
        hidden = x + residual
        moe_input = layer.post_attention_layernorm(hidden)
        return hidden + self._qwen_moe_forward_selective_experts(
            layer_index,
            layer.mlp,
            moe_input,
            events=events,
            pass_kind=pass_kind,
            token_step=token_step,
        )

    def _qwen_moe_forward_selective_experts(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        *,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx

        gates = mlp.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        top_k = mlp.top_k
        expert_indices = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
        scores = mx.take_along_axis(gates, expert_indices, axis=-1)
        if mlp.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        mx.eval(expert_indices)
        selected_experts = selected_expert_ids(expert_indices)
        self._expert_history[layer_index] = selected_experts

        pending = self._pending_expert_prefetch
        self._pending_expert_prefetch = None
        use_prefetch = pending is not None and pending["layer"] == layer_index
        if pending is not None and not use_prefetch:
            pending["future"].result()

        expert_load = None
        reuse_plan = None
        if self.expert_reuse:
            reuse_plan = self._plan_expert_reuse(layer_index, selected_experts)
            if reuse_plan["load_ids"]:
                expert_load = self.loader_executor.submit(
                    self._load_expert_slice_batch,
                    layer_index,
                    reuse_plan["load_ids"],
                )
        elif not use_prefetch:
            expert_load = self.loader_executor.submit(
                self._load_qwen_selected_experts,
                layer_index,
                mlp,
                selected_experts,
            )

        shared_y = mlp.shared_expert(x)
        shared_y = mx.sigmoid(mlp.shared_expert_gate(x)) * shared_y
        mx.async_eval(shared_y)

        if reuse_plan is not None:
            table_order = self._consume_expert_reuse(
                reuse_plan,
                expert_load,
                mlp,
                layer_index,
                selected_experts,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        elif use_prefetch:
            table_order = self._consume_expert_prefetch(
                pending,
                mlp,
                selected_experts,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        else:
            load_event = expert_load.result()
            events.append(
                {
                    "kind": "load",
                    "action": "load-selected-experts",
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    **compact_event(load_event),
                }
            )
            table_order = selected_experts

        local_indices = remap_expert_indices(expert_indices, table_order)
        expert_y = mlp.switch_mlp(x, local_indices)
        expert_y = (expert_y * scores[..., None]).sum(axis=-2)
        return expert_y + shared_y

    def _plan_expert_reuse(self, layer_index: int, selected_experts: list[int]) -> dict[str, Any]:
        cached = self._expert_table_cache.get(layer_index)
        if cached is None:
            return {"mode": "rebuild", "load_ids": list(selected_experts), "base": None}

        ordering, hits, missing = expert_merge_plan(cached["order"], selected_experts)
        if not missing:
            return {"mode": "hit", "load_ids": [], "base": cached, "order": list(cached["order"]), "hits": hits}
        if len(ordering) > self.expert_reuse_cap:
            return {
                "mode": "rebuild",
                "load_ids": list(selected_experts),
                "base": None,
                "evicted_rows": len(cached["order"]),
            }
        return {"mode": "extend", "load_ids": missing, "base": cached, "order": ordering, "hits": hits}

    def _consume_expert_reuse(
        self,
        plan: dict[str, Any],
        expert_load: Any,
        mlp: Any,
        layer_index: int,
        selected_experts: list[int],
        *,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> list[int]:
        import mlx.core as mx

        stats = self._prefetch_stats
        stats.true_rows += len(selected_experts)
        mode = plan["mode"]
        join_wait = 0.0
        assemble_seconds = 0.0
        loaded_bytes = 0

        if mode == "hit":
            stats.attempted_layers += 1
            stats.hit_rows += len(plan["hits"])
            stats.full_hits += 1
            arrays = plan["base"]["arrays"]
            order = plan["order"]
            nbytes = plan["base"]["nbytes"]
        elif mode == "extend":
            stats.attempted_layers += 1
            stats.hit_rows += len(plan["hits"])
            stats.missing_rows += len(plan["load_ids"])
            join_started = time.perf_counter()
            batch = expert_load.result()
            join_wait = time.perf_counter() - join_started
            loaded_bytes = batch.nbytes
            assemble_started = time.perf_counter()
            base_arrays = plan["base"]["arrays"]
            arrays = {
                name: mx.concatenate([base_arrays[name], batch.arrays[name]], axis=0)
                for name in base_arrays
            }
            mx.eval(list(arrays.values()))
            assemble_seconds = time.perf_counter() - assemble_started
            order = plan["order"]
            nbytes = plan["base"]["nbytes"] + batch.nbytes
        else:
            stats.skipped_no_history += 1
            stats.missing_rows += len(plan["load_ids"])
            stats.wasted_rows += plan.get("evicted_rows", 0)
            join_started = time.perf_counter()
            batch = expert_load.result()
            join_wait = time.perf_counter() - join_started
            loaded_bytes = batch.nbytes
            arrays = batch.arrays
            order = list(plan["load_ids"])
            nbytes = batch.nbytes

        stats.fallback_bytes += loaded_bytes
        stats.fallback_load_seconds += join_wait
        stats.assemble_seconds += assemble_seconds

        previous = self._expert_table_cache.pop(layer_index, None)
        if previous is not None:
            self._expert_cache_bytes -= previous["nbytes"]
        # Oversized tables (prompt-pass unions) are used but never cached: they
        # would pin several GB across layers and will not recur next token.
        if len(order) <= self.expert_reuse_cap:
            self._expert_table_cache[layer_index] = {"order": order, "arrays": arrays, "nbytes": nbytes}
            self._expert_cache_bytes += nbytes

        self._assign_expert_tables(mlp, layer_index, arrays)
        self.session.set_external_resident_bytes(self._expert_cache_bytes)

        event = MlxStreamEvent(
            action=f"reuse-experts-{mode}",
            layer=layer_index,
            seconds=join_wait + assemble_seconds,
            resident_bytes=self.session.resident_bytes,
            requested=self._expert_slice_names(layer_index),
            loaded=self._expert_slice_names(layer_index) if loaded_bytes else (),
            nbytes_loaded=loaded_bytes,
        )
        self.session.events.append(event)
        events.append(
            {
                "kind": "load",
                "action": f"reuse-experts-{mode}",
                "pass": pass_kind,
                "token_step": token_step,
                "layer": layer_index,
                "expert_count": len(selected_experts),
                "table_rows": len(order),
                "loaded_rows": len(plan["load_ids"]),
                "cache_bytes": self._expert_cache_bytes,
                **compact_event(event),
            }
        )
        return order

    def _maybe_submit_expert_prefetch(self, layer_index: int) -> None:
        if self.expert_reuse:
            return
        if self.expert_prefetch != "previous":
            return
        if self._pending_expert_prefetch is not None:
            return
        history = self._expert_history.get(layer_index)
        if not history:
            self._prefetch_stats.skipped_no_history += 1
            return
        if len(history) > self.expert_prefetch_cap:
            self._prefetch_stats.skipped_over_cap += 1
            return
        predicted = list(history)
        future = self.loader_executor.submit(
            self._load_expert_slice_batch, layer_index, predicted
        )
        self._pending_expert_prefetch = {
            "layer": layer_index,
            "experts": predicted,
            "future": future,
        }

    def _consume_expert_prefetch(
        self,
        pending: dict[str, Any],
        mlp: Any,
        selected_experts: list[int],
        *,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> list[int]:
        import mlx.core as mx

        layer_index = pending["layer"]
        predicted = pending["experts"]

        join_started = time.perf_counter()
        batch = pending["future"].result()
        join_wait = time.perf_counter() - join_started

        ordering, hits, missing = expert_merge_plan(predicted, selected_experts)
        stats = self._prefetch_stats
        stats.attempted_layers += 1
        stats.predicted_rows += len(predicted)
        stats.true_rows += len(selected_experts)
        stats.hit_rows += len(hits)
        stats.missing_rows += len(missing)
        wasted = len(predicted) - len(hits)
        stats.wasted_rows += wasted
        stats.prefetched_bytes += batch.nbytes
        row_bytes = batch.nbytes // len(predicted) if predicted else 0
        stats.wasted_bytes += row_bytes * wasted
        stats.prefetch_load_seconds += batch.seconds
        stats.join_wait_seconds += join_wait
        if not missing:
            stats.full_hits += 1

        arrays = dict(batch.arrays)
        fallback_bytes = 0
        fallback_seconds = 0.0
        if missing:
            fallback_started = time.perf_counter()
            missing_batch = self._load_expert_slice_batch(layer_index, missing)
            fallback_seconds = time.perf_counter() - fallback_started
            fallback_bytes = missing_batch.nbytes
            stats.fallback_bytes += fallback_bytes
            stats.fallback_load_seconds += fallback_seconds

            assemble_started = time.perf_counter()
            arrays = {
                name: mx.concatenate([arrays[name], missing_batch.arrays[name]], axis=0)
                for name in arrays
            }
            mx.eval(list(arrays.values()))
            stats.assemble_seconds += time.perf_counter() - assemble_started

        self._assign_expert_tables(mlp, layer_index, arrays)
        self.session.set_external_resident_bytes(batch.nbytes + fallback_bytes)

        event = MlxStreamEvent(
            action="prefetch-selected-experts",
            layer=layer_index,
            seconds=batch.seconds + fallback_seconds,
            resident_bytes=self.session.resident_bytes,
            requested=self._expert_slice_names(layer_index),
            loaded=self._expert_slice_names(layer_index),
            nbytes_loaded=batch.nbytes + fallback_bytes,
        )
        self.session.events.append(event)
        events.append(
            {
                "kind": "load",
                "action": "prefetch-selected-experts",
                "pass": pass_kind,
                "token_step": token_step,
                "layer": layer_index,
                "expert_count": len(selected_experts),
                "experts": selected_experts,
                "predicted_experts": predicted,
                "hit_count": len(hits),
                "missing_count": len(missing),
                "join_wait_seconds": join_wait,
                "fallback_seconds": fallback_seconds,
                **compact_event(event),
            }
        )
        return ordering

    def _expert_slice_names(self, layer_index: int) -> tuple[str, ...]:
        prefix = f"language_model.model.layers.{layer_index}.mlp.switch_mlp"
        return tuple(
            f"{prefix}.{projection}.{field}"
            for projection in ("gate_proj", "up_proj", "down_proj")
            for field in ("weight", "scales", "biases")
        )

    def _load_expert_slice_batch(
        self,
        layer_index: int,
        expert_ids: list[int],
    ) -> MlxTensorBatch:
        return self.session.loader.load_first_dim_slices(
            self._expert_slice_names(layer_index),
            expert_ids,
            evaluate=self.session.evaluate,
        )

    def _assign_expert_tables(self, mlp: Any, layer_index: int, arrays: dict[str, Any]) -> None:
        prefix = f"language_model.model.layers.{layer_index}.mlp.switch_mlp"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(mlp.switch_mlp, projection)
            module.weight = arrays[f"{prefix}.{projection}.weight"]
            module.scales = arrays[f"{prefix}.{projection}.scales"]
            module.biases = arrays[f"{prefix}.{projection}.biases"]

    def _load_qwen_selected_experts(
        self,
        layer_index: int,
        mlp: Any,
        selected_experts: list[int],
    ) -> MlxStreamEvent:
        started = time.perf_counter()
        batch = self._load_expert_slice_batch(layer_index, selected_experts)
        self._assign_expert_tables(mlp, layer_index, batch.arrays)
        self.session.set_external_resident_bytes(batch.nbytes)
        names = self._expert_slice_names(layer_index)
        event = MlxStreamEvent(
            action="load-selected-experts",
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names,
            nbytes_loaded=batch.nbytes,
        )
        self.session.events.append(event)
        return event

    def _clear_qwen_selected_experts(self, layer: Any) -> None:
        import mlx.core as mx

        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp.switch_mlp, projection)
            module.weight = mx.zeros((0, 0, 0), dtype=mx.uint32)
            module.scales = mx.zeros((0, 0, 0), dtype=mx.bfloat16)
            module.biases = mx.zeros((0, 0, 0), dtype=mx.bfloat16)
        if self._expert_cache_bytes:
            self.session.set_external_resident_bytes(self._expert_cache_bytes)
        else:
            self.session.clear_external_resident_bytes()

    def _logits_from_hidden(self, hidden: Any, events: list[dict[str, Any]]) -> Any:
        import mlx.core as mx

        language_model = self.session.model.language_model
        core = language_model.model
        tied_embeddings = language_model.args.tie_word_embeddings

        if self.pin_policy == "phase" and not self.warm_output:
            load_event = self.session.load_embedding() if tied_embeddings else self.session.load_output()
            events.append({"kind": "load", **compact_event(load_event)})

        hidden = core.norm(hidden)
        if tied_embeddings:
            logits = core.embed_tokens.as_linear(hidden)
        else:
            logits = language_model.lm_head(hidden)
        mx.eval(logits)

        if self.pin_policy == "phase" and not self.warm_output:
            evict_event = self.session.evict_embedding() if tied_embeddings else self.session.evict_output()
            events.append({"kind": "evict", **compact_event(evict_event)})
        return logits


class DeepSeekV3StreamingForwardRunner(StreamingChatRunner):
    """Generate with DeepSeek V3/R1 through SmartTensor selective MoE streaming."""

    EXPERT_MARKER = ".mlp.switch_mlp."
    SUPPORTED_MODEL_TYPES = {"deepseek_v3", "glm_moe_dsa"}

    def __init__(
        self,
        model_dir: str | Path,
        *,
        evaluate: bool = True,
        retain_layers: set[int] | None = None,
        resident_budget_bytes: int | None = None,
        backend: str = "native",
        clear_on_evict: bool = False,
        pin_policy: str = "all",
        warm_embeddings: bool = False,
        weight_page_budget_bytes: int | None = None,
        weight_page_policy: str = "auto",
        weight_page_rows: int = 1,
        expert_prefetch: str = "off",
        expert_prefetch_cap: int = 32,
        expert_compute_mode: str = "table",
        expert_slot_capacity: int | None = None,
        pack_dir: str | Path | None = None,
        pack_read_workers: int = 1,
        drop_mmap_cache_after_read: bool | None = None,
        trace: bool = False,
    ) -> None:
        if pin_policy not in {"all", "phase"}:
            raise ValueError("pin_policy must be 'all' or 'phase'")
        if warm_embeddings and pin_policy != "phase":
            raise ValueError("warm_embeddings only applies to pin_policy='phase'")
        if expert_prefetch not in {"off", "previous", "previous_table"}:
            raise ValueError("expert_prefetch must be 'off', 'previous', or 'previous_table'")
        if expert_compute_mode not in {
            "table",
            "direct_qmm",
            "per_expert",
            "table_overlap_shared",
            "split_overlap_down",
            "split_overlap_shared_down",
            "slot_arena_direct_qmm",
            "slot_arena_guarded_direct_qmm",
            "slot_arena_mixed_direct_qmm",
            "slot_arena_compact_direct_qmm",
            "slot_arena_compact_defer_eval",
            "slot_arena_static_direct_defer",
            "slot_arena_hotcold_qmv",
        }:
            raise ValueError(
                "expert_compute_mode must be 'table', 'direct_qmm', 'per_expert', "
                "'table_overlap_shared', 'split_overlap_down', or "
                "'split_overlap_shared_down', 'slot_arena_direct_qmm', or "
                "'slot_arena_guarded_direct_qmm', 'slot_arena_mixed_direct_qmm', "
                "'slot_arena_compact_direct_qmm', 'slot_arena_compact_defer_eval', "
                "'slot_arena_static_direct_defer', or 'slot_arena_hotcold_qmv'"
            )
        if expert_prefetch_cap < 1:
            raise ValueError("expert_prefetch_cap must be positive")
        if expert_slot_capacity is not None and expert_slot_capacity < 1:
            raise ValueError("expert_slot_capacity must be positive")
        if weight_page_rows < 1:
            raise ValueError("weight_page_rows must be positive")
        if weight_page_rows != 1 and weight_page_budget_bytes is None:
            raise ValueError("weight_page_rows requires weight_page_budget_bytes")
        if expert_prefetch != "off" and weight_page_budget_bytes is None:
            raise ValueError("DeepSeek expert prefetch requires --weight-page-budget")
        if pack_read_workers < 1:
            raise ValueError("pack_read_workers must be positive")
        self.session = MlxModelSession(
            model_dir,
            evaluate=evaluate,
            retain_layers=retain_layers,
            resident_budget_bytes=None,
            backend=backend,
            clear_on_evict=clear_on_evict,
            pin_policy=pin_policy,
            warm_embeddings=warm_embeddings,
            trace=trace,
        )
        self._trace = trace
        self._pass_trace = None
        self.pin_policy = pin_policy
        self.warm_embeddings = warm_embeddings
        # DeepSeek's output head is large enough that holding it resident
        # throughout the streamed layer pass can exceed a 2 GiB budget before
        # the first routed expert table is even loaded. Load it only for the
        # logits projection in low-memory phase mode.
        self.warm_output = False
        self._embed_prefix = "model.embed_tokens"
        self._expert_cache_bytes = 0
        self.model_type = str(self.session.config.get("model_type"))
        if drop_mmap_cache_after_read is None:
            drop_mmap_cache_after_read = self.model_type == "glm_moe_dsa"
        self.session.loader.drop_mmap_cache_after_read = bool(drop_mmap_cache_after_read)
        weight_page_policy = resolve_weight_page_policy(
            self.model_type,
            weight_page_policy,
            has_weight_page_budget=weight_page_budget_bytes is not None,
        )
        self.resident_budget_bytes = resident_budget_bytes
        self.weight_page_budget_bytes = weight_page_budget_bytes
        self.weight_page_policy = weight_page_policy
        self.weight_page_rows = weight_page_rows
        self.expert_prefetch = expert_prefetch
        self.expert_prefetch_cap = expert_prefetch_cap
        self.expert_compute_mode = expert_compute_mode
        self.expert_slot_update_missing = (
            expert_compute_mode == "slot_arena_direct_qmm"
        )
        self.expert_slot_capacity = expert_slot_capacity
        self.pack_dir = pack_dir
        self.pack_read_workers = pack_read_workers
        self._expert_history: dict[int, list[int]] = {}
        self._last_deepseek_slot_arena_full_hit: dict[int, bool] = {}
        self._expert_slot_arenas: dict[int, DeepSeekExpertSlotArena] = {}
        self._expert_slot_arena_bytes = 0
        self._expert_slot_clock = 0
        self._expert_slot_stats = DeepSeekExpertSlotArenaStats()
        self.preserve_slot_arenas_on_reset = False
        self._pending_expert_prefetch: dict[str, Any] | None = None
        self._pending_expert_prefetch_bytes = 0
        self._pending_layer_base_prefetch: dict[str, Any] | None = None
        self._pending_layer_base_prefetch_bytes = 0
        self._prefetch_stats = ExpertPrefetchStats()
        self.loader_executor = ThreadPoolExecutor(max_workers=1)
        if pack_dir is not None:
            self.session.loader.attach_pack_dir(
                pack_dir,
                pack_read_workers=pack_read_workers,
            )
        if weight_page_budget_bytes is not None:
            self.session.loader.attach_weight_page_cache(
                weight_page_budget_bytes,
                eviction_policy=weight_page_policy,
                rows_per_page=weight_page_rows,
            )
        if self.model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                "DeepSeekV3StreamingForwardRunner currently supports "
                f"model_type in {sorted(self.SUPPORTED_MODEL_TYPES)}, got {self.model_type!r}"
            )

        top_k = int(self.session.config.get("num_experts_per_tok", 1))
        if self.expert_slot_capacity is None:
            self.expert_slot_capacity = top_k
        if retain_layers is not None:
            self.base_retain_layers = set(retain_layers)
        elif self.model_type == "glm_moe_dsa":
            # GLM 5.2 must stream expert rows, but on Studio-class budgets the
            # non-expert layer tensors are small enough to retain and save a
            # repeated ~18 GiB prompt-pass load. Tight budgets stay
            # conservative and stream base layers.
            base_budget_bytes = reserve_weight_page_budget_for_base_retention(
                resident_budget_bytes,
                weight_page_budget_bytes,
            )
            self.base_retain_layers = (
                select_qwen_base_layers_for_budget(
                    self.session.loader.manifest,
                    base_budget_bytes,
                    top_k=top_k,
                    expert_marker=self.EXPERT_MARKER,
                )
                if resident_budget_bytes is not None
                else set()
            )
        elif resident_budget_bytes is not None:
            base_budget_bytes = reserve_weight_page_budget_for_base_retention(
                resident_budget_bytes,
                weight_page_budget_bytes,
            )
            self.base_retain_layers = select_qwen_base_layers_for_budget(
                self.session.loader.manifest,
                base_budget_bytes,
                top_k=top_k,
                expert_marker=self.EXPERT_MARKER,
            )
        else:
            self.base_retain_layers = set()

        from mlx_lm.utils import load_tokenizer

        self.tokenizer = load_tokenizer(self.session.model_dir)

    def close(self) -> None:
        error: BaseException | None = None
        try:
            self._drain_deepseek_expert_prefetch()
            self._drain_deepseek_layer_base_prefetch()
        except BaseException as exc:
            error = exc
        try:
            self.loader_executor.shutdown(wait=True)
        finally:
            self.session.close()
        if error is not None:
            raise error

    def _reset_stream_state(self) -> None:
        try:
            self._drain_deepseek_expert_prefetch()
            self._drain_deepseek_layer_base_prefetch()
        finally:
            self._expert_cache_bytes = 0
            self._expert_history.clear()
            if not getattr(self, "preserve_slot_arenas_on_reset", False):
                self._clear_deepseek_slot_arenas()
            self._prefetch_stats = ExpertPrefetchStats()
            self._expert_slot_stats = DeepSeekExpertSlotArenaStats()
            self._set_deepseek_external_resident_bytes()

    def _drain_deepseek_expert_prefetch(self) -> None:
        pending = self._pending_expert_prefetch
        self._pending_expert_prefetch = None
        self._pending_expert_prefetch_bytes = 0
        if pending is not None:
            pending["future"].result()

    def _drain_deepseek_layer_base_prefetch(self) -> None:
        pending = getattr(self, "_pending_layer_base_prefetch", None)
        self._pending_layer_base_prefetch = None
        self._pending_layer_base_prefetch_bytes = 0
        if pending is not None:
            pending["future"].result()

    def _resident_sidecar_bytes(self) -> int:
        return (
            self._expert_cache_bytes
            + int(getattr(self, "_expert_slot_arena_bytes", 0) or 0)
            + self._pending_expert_prefetch_bytes
            + int(getattr(self, "_pending_layer_base_prefetch_bytes", 0) or 0)
            + self.session.loader.weight_page_resident_bytes
        )

    def _set_deepseek_external_resident_bytes(self, temporary_bytes: int = 0) -> None:
        total = self._resident_sidecar_bytes() + temporary_bytes
        if total:
            self.session.set_external_resident_bytes(total)
        else:
            self.session.clear_external_resident_bytes()

    def _deepseek_budget_base_bytes(self) -> int:
        current = int(getattr(self.session, "resident_bytes", 0) or 0)
        external = int(getattr(self.session, "external_resident_bytes", 0) or 0)
        return max(current - external, 0)

    def _deepseek_budget_allows(
        self,
        *,
        additional_sidecar_bytes: int = 0,
        temporary_bytes: int = 0,
    ) -> bool:
        budget = getattr(self, "resident_budget_bytes", None)
        if budget is None:
            return True
        proposed_external = (
            self._resident_sidecar_bytes()
            + additional_sidecar_bytes
            + temporary_bytes
        )
        return self._deepseek_budget_base_bytes() + proposed_external <= budget

    def _deepseek_total_resident_allows(self, additional_bytes: int = 0) -> bool:
        budget = getattr(self, "resident_budget_bytes", None)
        if budget is None:
            return True
        return self.session.resident_bytes + additional_bytes <= budget

    def _require_deepseek_budget(
        self,
        *,
        action: str,
        additional_sidecar_bytes: int = 0,
        temporary_bytes: int = 0,
    ) -> None:
        if self._deepseek_budget_allows(
            additional_sidecar_bytes=additional_sidecar_bytes,
            temporary_bytes=temporary_bytes,
        ):
            return
        proposed_external = (
            self._resident_sidecar_bytes()
            + additional_sidecar_bytes
            + temporary_bytes
        )
        proposed_peak = self._deepseek_budget_base_bytes() + proposed_external
        budget = getattr(self, "resident_budget_bytes", None)
        raise MemoryError(
            f"{action} would exceed resident budget "
            f"({proposed_peak} > {budget})"
        )

    def _embed_module(self) -> Any:
        return self.session.model.model.embed_tokens

    def _stream_forward_tokens(
        self,
        token_rows: list[list[int]],
        *,
        cache: list[Any],
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None = None,
        manage_embedding: bool = True,
    ) -> Any:
        import mlx.core as mx

        trace = self._trace
        core = self.session.model.model
        if self.pin_policy == "phase" and manage_embedding:
            load_embedding, local_token_rows = self._load_embedding_slices(token_rows)
            if trace and load_embedding is not None:
                events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_embedding)})
            inputs = mx.array(local_token_rows)
            try:
                x = core.embed_tokens(inputs)
                mx.eval(x)
            finally:
                self._clear_embedding_slices()
                if trace and load_embedding is not None:
                    evict_embedding = MlxStreamEvent(
                        action="evict-embedding-slices",
                        seconds=0.0,
                        resident_bytes=self.session.resident_bytes,
                        requested=load_embedding.requested,
                        evicted=load_embedding.loaded,
                    )
                    self.session.events.append(evict_embedding)
                    events.append({"kind": "evict", "pass": pass_kind, "token_step": token_step, **compact_event(evict_embedding)})
        else:
            inputs = mx.array(token_rows)
            x = core.embed_tokens(inputs)
            mx.eval(x)

        mask = self._deepseek_create_attention_mask(x, cache)

        for layer_index, (layer, layer_cache) in enumerate(zip(core.layers, cache)):
            load_event = self._load_deepseek_layer_base(layer_index)
            if trace:
                events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_event)})
            self._maybe_submit_deepseek_expert_prefetch(
                layer_index,
                layer,
                events=events if trace else None,
                pass_kind=pass_kind,
                token_step=token_step,
            )

            compute_started = time.perf_counter() if trace else 0.0
            try:
                x = self._deepseek_layer_forward_selective_experts(
                    layer_index,
                    layer,
                    x,
                    mask=mask,
                    cache=layer_cache,
                    events=events if trace else None,
                    pass_kind=pass_kind,
                    token_step=token_step,
                )
                self._maybe_submit_deepseek_layer_base_prefetch(
                    layer_index + 1,
                    len(core.layers),
                )
                if self._deepseek_should_defer_layer_eval(layer_index):
                    self._expert_slot_stats.deferred_layer_evals += 1
                else:
                    if self.expert_compute_mode == "slot_arena_compact_defer_eval":
                        self._expert_slot_stats.forced_layer_evals += 1
                    mx.eval(x)
            finally:
                if (
                    hasattr(layer.mlp, "switch_mlp")
                    and self._deepseek_compute_binds_switch_mlp()
                ):
                    self._clear_deepseek_selected_experts(layer)
            if trace:
                events.append(
                    {
                        "kind": "compute",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "seconds": time.perf_counter() - compute_started,
                        "hidden_shape": list(x.shape),
                        "is_moe": hasattr(layer.mlp, "switch_mlp"),
                    }
                )

            if layer_index in self.base_retain_layers:
                evict_event = MlxStreamEvent(
                    action="retain-layer-base",
                    layer=layer_index,
                    seconds=0.0,
                    resident_bytes=self.session.resident_bytes,
                    requested=self.session.loader.manifest.layers[layer_index].tensor_names,
                )
                if trace:
                    self.session.events.append(evict_event)
            else:
                self._clear_deepseek_mla_projection_tables(layer)
                evict_event = self.session.evict_layer(layer_index)
            if trace:
                events.append({"kind": "evict", "pass": pass_kind, "token_step": token_step, **compact_event(evict_event)})

        return x

    def _deepseek_create_attention_mask(self, hidden: Any, cache: list[Any]) -> Any:
        if self.model_type == "glm_moe_dsa":
            from mlx_lm.models.deepseek_v32 import create_attention_mask

            first_cache = cache[0][0] if cache and cache[0] is not None else None
            return create_attention_mask(hidden, first_cache, return_array=True)

        from mlx_lm.models.base import create_attention_mask

        first_cache = cache[0] if cache else None
        return create_attention_mask(hidden, first_cache, return_array=True)

    def _load_deepseek_layer_base(self, layer_index: int) -> MlxStreamEvent:
        pending = self._pending_layer_base_prefetch
        self._pending_layer_base_prefetch = None
        if pending is not None and pending["layer"] == layer_index:
            started = time.perf_counter()
            names = pending["names"]
            batch: MlxTensorBatch | None = None
            try:
                batch = pending["future"].result()
                missing = tuple(name for name in names if name not in self.session.resident)
                skipped = tuple(name for name in names if name in self.session.resident)
                if missing:
                    self.session._apply_tensor_batch(missing, batch)
                resident_bytes = self.session.resident_bytes
                self.session.peak_resident_bytes = max(
                    self.session.peak_resident_bytes,
                    resident_bytes,
                )
                event = MlxStreamEvent(
                    action="load-layer-base-prefetched",
                    layer=layer_index,
                    seconds=time.perf_counter() - started,
                    resident_bytes=resident_bytes,
                    requested=names,
                    loaded=missing,
                    skipped=skipped,
                    nbytes_loaded=batch.nbytes if missing else 0,
                )
                if self._trace:
                    self.session.events.append(event)
                return event
            finally:
                self._pending_layer_base_prefetch_bytes = 0
                self._set_deepseek_external_resident_bytes()
        if pending is not None:
            try:
                pending["future"].result()
            finally:
                self._pending_layer_base_prefetch_bytes = 0
                self._set_deepseek_external_resident_bytes()

        names = self._deepseek_layer_base_names(layer_index)
        return self.session._load_into_model(names, action="load-layer-base", layer=layer_index)

    def _deepseek_layer_base_names(self, layer_index: int) -> tuple[str, ...]:
        layer = self.session.loader.manifest.layers[layer_index]
        return tuple(name for name in layer.tensor_names if self.EXPERT_MARKER not in name)

    def _maybe_submit_deepseek_layer_base_prefetch(
        self,
        layer_index: int,
        total_layers: int,
    ) -> None:
        if self._pending_layer_base_prefetch is not None:
            return
        if layer_index >= total_layers:
            return
        if layer_index in self.base_retain_layers:
            return
        if self.expert_prefetch != "off":
            return
        if self.session.clear_on_evict:
            return
        names = tuple(
            name
            for name in self._deepseek_layer_base_names(layer_index)
            if name not in self.session.resident
        )
        if not names:
            return
        nbytes = sum(self.session.loader.manifest.tensors[name].nbytes for name in names)
        if not self._deepseek_budget_allows(additional_sidecar_bytes=nbytes):
            return
        future = self.loader_executor.submit(
            self.session.loader.load_tensors,
            names,
            evaluate=self.session.evaluate,
        )
        self._pending_layer_base_prefetch_bytes = nbytes
        self._set_deepseek_external_resident_bytes()
        self._pending_layer_base_prefetch = {
            "layer": layer_index,
            "names": names,
            "future": future,
        }

    def _deepseek_layer_forward_selective_experts(
        self,
        layer_index: int,
        layer: Any,
        x: Any,
        *,
        mask: Any,
        cache: Any,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        residual = layer.self_attn(layer.input_layernorm(x), mask, cache)
        hidden = x + residual
        mlp_input = layer.post_attention_layernorm(hidden)
        if not hasattr(layer.mlp, "switch_mlp"):
            return hidden + layer.mlp(mlp_input)
        return hidden + self._deepseek_moe_forward_selective_experts(
            layer_index,
            layer.mlp,
            mlp_input,
            events=events,
            pass_kind=pass_kind,
            token_step=token_step,
        )

    def _prefill_cache_tokens(
        self,
        token_rows: list[list[int]],
        *,
        cache: list[Any],
        events: list[dict[str, Any]],
        manage_embedding: bool,
    ) -> None:
        if not token_rows or not token_rows[0]:
            return
        if self.model_type == "glm_moe_dsa":
            for position in range(len(token_rows[0])):
                self._stream_forward_tokens(
                    [[row[position]] for row in token_rows],
                    cache=cache,
                    events=events,
                    pass_kind="prefill-cache",
                    manage_embedding=manage_embedding,
                )
            return
        self._stream_forward_tokens(
            token_rows,
            cache=cache,
            events=events,
            pass_kind="prefill-cache",
            manage_embedding=manage_embedding,
        )

    def _maybe_detach_glm_array(self, array: Any) -> Any:
        if self.model_type != "glm_moe_dsa":
            return array
        import mlx.core as mx

        if array.dtype == mx.bfloat16:
            raw = np.array(array.view(mx.uint16))
            detached = mx.array(raw).view(mx.bfloat16)
        else:
            detached = mx.array(np.array(array))
        mx.eval(detached)
        return detached

    def _before_phase_output_load(self, events: list[dict[str, Any]]) -> None:
        if not getattr(self, "_expert_slot_arenas", None):
            return
        output_bytes = role_budget_bytes(self.session.loader.manifest, "output")
        # Loading lm_head also materializes logits/norm work. The raw output
        # tensor size alone underestimates the phase peak on tight tiers.
        output_required_bytes = output_bytes + 1536 * 1024 * 1024
        if self._deepseek_total_resident_allows(output_required_bytes):
            return
        before_count = len(self._expert_slot_arenas)
        before_bytes = self._expert_slot_arena_bytes
        self._evict_deepseek_slot_arenas_until(additional_total_bytes=output_required_bytes)
        if not self._deepseek_total_resident_allows(output_required_bytes):
            self._clear_deepseek_slot_arenas()
            self._set_deepseek_external_resident_bytes()
        if self._trace:
            events.append(
                {
                    "kind": "evict",
                    "action": "evict-slot-arenas-for-output",
                    "evicted_count": before_count - len(self._expert_slot_arenas),
                    "evicted_bytes": before_bytes - self._expert_slot_arena_bytes,
                    "resident_bytes": self.session.resident_bytes,
                }
            )

    def _deepseek_moe_forward_selective_experts(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx

        expert_indices, scores = mlp.gate(x)
        mx.eval(expert_indices)
        selected_experts = selected_expert_ids(expert_indices)
        if (
            self.expert_compute_mode
            in {
                "slot_arena_direct_qmm",
                "slot_arena_guarded_direct_qmm",
                "slot_arena_mixed_direct_qmm",
                "slot_arena_compact_direct_qmm",
                "slot_arena_compact_defer_eval",
                "slot_arena_static_direct_defer",
                "slot_arena_hotcold_qmv",
            }
            and self.expert_prefetch == "off"
        ):
            self._expert_history[layer_index] = selected_experts
            if self.expert_compute_mode == "slot_arena_mixed_direct_qmm":
                return self._deepseek_moe_forward_slot_arena_mixed_direct_qmm(
                    layer_index,
                    mlp,
                    x,
                    expert_indices,
                    selected_experts,
                    scores,
                    events=events,
                    pass_kind=pass_kind,
                    token_step=token_step,
                )
            if self.expert_compute_mode in {
                "slot_arena_compact_direct_qmm",
                "slot_arena_compact_defer_eval",
            }:
                return self._deepseek_moe_forward_slot_arena_compact_direct_qmm(
                    layer_index,
                    mlp,
                    x,
                    expert_indices,
                    selected_experts,
                    scores,
                    events=events,
                    pass_kind=pass_kind,
                    token_step=token_step,
                )
            if self.expert_compute_mode == "slot_arena_hotcold_qmv":
                return self._deepseek_moe_forward_slot_arena_hotcold_qmv(
                    layer_index,
                    mlp,
                    x,
                    expert_indices,
                    selected_experts,
                    scores,
                    events=events,
                    pass_kind=pass_kind,
                    token_step=token_step,
                )
            if self.expert_compute_mode == "slot_arena_static_direct_defer":
                return self._deepseek_moe_forward_slot_arena_static_direct_defer(
                    layer_index,
                    mlp,
                    x,
                    expert_indices,
                    selected_experts,
                    scores,
                    events=events,
                    pass_kind=pass_kind,
                    token_step=token_step,
                )
            return self._deepseek_moe_forward_slot_arena_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        if (
            self.expert_compute_mode == "direct_qmm"
            and self.expert_prefetch == "off"
        ):
            self._expert_history[layer_index] = selected_experts
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        if (
            self.expert_compute_mode in {"split_overlap_down", "split_overlap_shared_down"}
            and self.expert_prefetch == "off"
        ):
            self._expert_history[layer_index] = selected_experts
            return self._deepseek_moe_forward_split_overlap_down(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
                overlap_shared=self.expert_compute_mode == "split_overlap_shared_down",
            )
        if (
            self.expert_compute_mode == "table_overlap_shared"
            and self.expert_prefetch == "off"
            and getattr(mlp, "shared_experts", None) is not None
        ):
            self._expert_history[layer_index] = selected_experts
            return self._deepseek_moe_forward_table_overlap_shared(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        if (
            self.expert_compute_mode == "per_expert"
            and self.expert_prefetch == "off"
            and self._can_stream_deepseek_experts_per_expert(expert_indices)
        ):
            topk_experts = [int(value) for value in np.asarray(expert_indices).reshape(-1)]
            self._expert_history[layer_index] = selected_experts
            return self._deepseek_moe_forward_per_expert(
                layer_index,
                mlp,
                x,
                topk_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        table_order = self._consume_deepseek_expert_prefetch(
            layer_index,
            mlp,
            selected_experts,
            events=events,
            pass_kind=pass_kind,
            token_step=token_step,
        )
        self._expert_history[layer_index] = selected_experts

        if table_order is None:
            load_event = self._load_deepseek_selected_experts(layer_index, mlp, selected_experts)
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": "load-selected-experts",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                        **compact_event(load_event),
                    }
                )
            table_order = selected_experts

        local_indices = remap_expert_indices(expert_indices, table_order)
        expert_y = mlp.switch_mlp(x, local_indices)
        expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
        shared_experts = getattr(mlp, "shared_experts", None)
        if shared_experts is not None:
            expert_y = expert_y + shared_experts(x)
        return expert_y

    def _deepseek_moe_forward_table_overlap_shared(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx

        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        use_weight_page_cache = True
        if getattr(self, "resident_budget_bytes", None) is not None:
            estimated_table_bytes = self._deepseek_slice_nbytes(layer_index, selected_experts)
            estimated_page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=True,
            )
            estimated_transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=False,
            )
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=estimated_table_bytes + estimated_page_growth_bytes,
                temporary_bytes=estimated_transient_page_bytes,
            ):
                if estimated_transient_page_bytes and self._deepseek_budget_allows(
                    additional_sidecar_bytes=estimated_table_bytes,
                ):
                    use_weight_page_cache = False
                    estimated_transient_page_bytes = 0
                else:
                    self._require_deepseek_budget(
                        action="load-selected-experts-overlap-shared",
                        additional_sidecar_bytes=estimated_table_bytes,
                        temporary_bytes=estimated_transient_page_bytes,
                    )
            self._set_deepseek_external_resident_bytes(
                estimated_table_bytes + estimated_transient_page_bytes
            )

        future = self.loader_executor.submit(
            self._load_deepseek_slice_batch,
            layer_index,
            selected_experts,
            use_weight_page_cache=use_weight_page_cache,
            evaluate=self.session.evaluate,
        )
        shared_y = mlp.shared_experts(x)
        mx.async_eval(shared_y)
        try:
            batch = future.result()
            batch_transient_page_bytes = int(
                getattr(batch, "transient_page_bytes", 0) or 0
            )
            if batch_transient_page_bytes:
                self._set_deepseek_external_resident_bytes(
                    batch.nbytes + batch_transient_page_bytes
                )
            self._assign_deepseek_expert_tables(mlp, layer_index, batch.arrays)
            self._set_deepseek_external_resident_bytes(batch.nbytes)
            local_indices = remap_expert_indices(expert_indices, selected_experts)
            expert_y = mlp.switch_mlp(x, local_indices)
            expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
            event = MlxStreamEvent(
                action=(
                    "load-selected-experts-overlap-shared"
                    if use_weight_page_cache
                    else "load-selected-experts-overlap-shared-direct-budget"
                ),
                layer=layer_index,
                seconds=time.perf_counter() - started,
                resident_bytes=self.session.resident_bytes,
                requested=names,
                loaded=names,
                nbytes_loaded=batch.nbytes,
                transient_page_bytes=batch_transient_page_bytes,
            )
            if self._trace:
                self.session.events.append(event)
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": event.action,
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                        **compact_event(event),
                    }
                )
            return expert_y + shared_y
        finally:
            self._set_deepseek_external_resident_bytes()

    def _deepseek_moe_forward_direct_qmm(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
        overlap_shared: bool = False,
    ) -> Any:
        import mlx.core as mx

        started = time.perf_counter()
        use_weight_page_cache = True
        estimated_table_bytes = self._deepseek_slice_nbytes(layer_index, selected_experts)
        estimated_transient_page_bytes = 0
        if getattr(self, "resident_budget_bytes", None) is not None:
            estimated_page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=True,
            )
            estimated_transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=False,
            )
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=estimated_table_bytes + estimated_page_growth_bytes,
                temporary_bytes=estimated_transient_page_bytes,
            ):
                if estimated_transient_page_bytes and self._deepseek_budget_allows(
                    additional_sidecar_bytes=estimated_table_bytes,
                    temporary_bytes=0,
                ):
                    use_weight_page_cache = False
                    estimated_transient_page_bytes = 0
                else:
                    self._require_deepseek_budget(
                        action="load-selected-experts-direct-qmm",
                        additional_sidecar_bytes=estimated_table_bytes,
                        temporary_bytes=estimated_transient_page_bytes,
                    )

        batch = self._load_deepseek_slice_batch(
            layer_index,
            selected_experts,
            use_weight_page_cache=use_weight_page_cache,
            evaluate=False,
        )
        batch_transient_page_bytes = int(
            getattr(batch, "transient_page_bytes", 0) or 0
        )
        self._set_deepseek_external_resident_bytes(
            batch.nbytes + batch_transient_page_bytes
        )
        try:
            local_indices = remap_expert_indices(expert_indices, selected_experts)
            x_expanded = mx.expand_dims(x, (-2, -3))
            up = self._deepseek_projection_qmm(
                mlp,
                layer_index,
                "up_proj",
                x_expanded,
                local_indices,
                batch.arrays,
            )
            gate = self._deepseek_projection_qmm(
                mlp,
                layer_index,
                "gate_proj",
                x_expanded,
                local_indices,
                batch.arrays,
            )
            activated = mlp.switch_mlp.activation(up, gate)
            expert_y = self._deepseek_projection_qmm(
                mlp,
                layer_index,
                "down_proj",
                activated,
                local_indices,
                batch.arrays,
            )
            expert_y = expert_y.squeeze(-2)
            expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
            shared_experts = getattr(mlp, "shared_experts", None)
            if shared_experts is not None:
                expert_y = expert_y + shared_experts(x)
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": (
                            "load-selected-experts-direct-qmm"
                            if use_weight_page_cache
                            else "load-selected-experts-direct-qmm-direct-budget"
                        ),
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                        "seconds": time.perf_counter() - started,
                        "resident_bytes": self.session.resident_bytes,
                        "nbytes_loaded": batch.nbytes,
                        "transient_page_bytes": batch_transient_page_bytes,
                    }
                )
            return expert_y
        finally:
            self._set_deepseek_external_resident_bytes()

    def _deepseek_moe_forward_slot_arena_direct_qmm(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        started = time.perf_counter()
        self._expert_slot_stats.attempts += 1
        self._expert_slot_stats.selected_rows += len(selected_experts)
        arena_result = self._ensure_deepseek_slot_arena(
            layer_index,
            selected_experts,
        )
        if arena_result is None:
            self._expert_slot_stats.fallback_direct += 1
            direct_bytes = self._deepseek_slice_nbytes(layer_index, selected_experts)
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=direct_bytes
            ) or not self._deepseek_total_resident_allows(direct_bytes):
                self._evict_deepseek_slot_arenas_until(
                    additional_sidecar_bytes=direct_bytes,
                    additional_total_bytes=direct_bytes,
                )
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=direct_bytes
            ) or not self._deepseek_total_resident_allows(direct_bytes):
                self._clear_deepseek_slot_arenas()
                self._set_deepseek_external_resident_bytes()
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": "slot-arena-fallback-direct-qmm",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                    }
                )
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        arena, arena_info = arena_result
        local_indices = remap_expert_indices_to_slots(expert_indices, arena.expert_to_slot)
        expert_y = self._deepseek_routed_qmm_from_arrays(
            mlp,
            layer_index,
            x,
            local_indices,
            scores,
            arena.arrays,
        )
        if events is not None:
            events.append(
                {
                    "kind": "load",
                    "action": arena_info["action"],
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    "seconds": time.perf_counter() - started,
                    "resident_bytes": self.session.resident_bytes,
                    **arena_info,
                }
            )
        return expert_y

    def _deepseek_routed_qmm_from_arrays(
        self,
        mlp: Any,
        layer_index: int,
        x: Any,
        local_indices: Any,
        scores: Any,
        arrays: dict[str, Any],
        *,
        add_shared: bool = True,
    ) -> Any:
        expert_y = self._deepseek_routed_qmm_outputs_from_arrays(
            mlp,
            layer_index,
            x,
            local_indices,
            arrays,
        )
        expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
        shared_experts = getattr(mlp, "shared_experts", None)
        if add_shared and shared_experts is not None:
            expert_y = expert_y + shared_experts(x)
        return expert_y

    def _deepseek_routed_qmm_outputs_from_arrays(
        self,
        mlp: Any,
        layer_index: int,
        x: Any,
        local_indices: Any,
        arrays: dict[str, Any],
    ) -> Any:
        import mlx.core as mx

        x_expanded = mx.expand_dims(x, (-2, -3))
        up = self._deepseek_projection_qmm(
            mlp,
            layer_index,
            "up_proj",
            x_expanded,
            local_indices,
            arrays,
        )
        gate = self._deepseek_projection_qmm(
            mlp,
            layer_index,
            "gate_proj",
            x_expanded,
            local_indices,
            arrays,
        )
        activated = mlp.switch_mlp.activation(up, gate)
        expert_y = self._deepseek_projection_qmm(
            mlp,
            layer_index,
            "down_proj",
            activated,
            local_indices,
            arrays,
        )
        return expert_y.squeeze(-2)

    def _deepseek_moe_forward_slot_arena_mixed_direct_qmm(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        started = time.perf_counter()
        self._expert_slot_stats.attempts += 1
        self._expert_slot_stats.selected_rows += len(selected_experts)
        arena = self._expert_slot_arenas.get(layer_index)
        if arena is None:
            self._set_deepseek_slot_arena_full_hit(layer_index, False)
            self._expert_slot_stats.arena_misses += 1
            self._expert_slot_stats.missing_rows += len(selected_experts)
            self._expert_slot_stats.fallback_direct += 1
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        self._expert_slot_clock += 1
        arena.last_used = self._expert_slot_clock
        partition_started = time.perf_counter()
        hits = [expert for expert in selected_experts if expert in arena.expert_to_slot]
        missing = [expert for expert in selected_experts if expert not in arena.expert_to_slot]
        partition_seconds = time.perf_counter() - partition_started
        self._expert_slot_stats.hit_rows += len(hits)
        self._expert_slot_stats.missing_rows += len(missing)
        if not hits:
            self._set_deepseek_slot_arena_full_hit(layer_index, False)
            self._expert_slot_stats.arena_misses += 1
            self._expert_slot_stats.fallback_direct += 1
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        if missing:
            self._expert_slot_stats.arena_misses += 1
        else:
            self._expert_slot_stats.arena_hits += 1
        self._set_deepseek_slot_arena_full_hit(layer_index, not missing)

        import mlx.core as mx

        hit_indices, hit_mask = remap_expert_indices_to_slots_with_mask(
            expert_indices,
            arena.expert_to_slot,
        )
        hit_outputs = self._deepseek_routed_qmm_outputs_from_arrays(
            mlp,
            layer_index,
            x,
            hit_indices,
            arena.arrays,
        )

        nbytes_loaded = 0
        load_seconds = 0.0
        transient_page_bytes = 0
        if missing:
            missing_to_slot = {expert: slot for slot, expert in enumerate(missing)}
            missing_indices, missing_mask = remap_expert_indices_to_slots_with_mask(
                expert_indices,
                missing_to_slot,
            )
            load_started = time.perf_counter()
            batch = self._load_deepseek_slice_batch(
                layer_index,
                missing,
                use_weight_page_cache=True,
                evaluate=False,
            )
            load_seconds = time.perf_counter() - load_started
            nbytes_loaded = batch.nbytes
            transient_page_bytes = int(getattr(batch, "transient_page_bytes", 0) or 0)
            self._expert_slot_stats.loaded_bytes += batch.nbytes
            self._expert_slot_stats.load_seconds += load_seconds
            self._set_deepseek_external_resident_bytes(
                arena.nbytes + batch.nbytes + transient_page_bytes
            )
            missing_outputs = self._deepseek_routed_qmm_outputs_from_arrays(
                mlp,
                layer_index,
                x,
                missing_indices,
                batch.arrays,
            )
            hit_mask_expanded = hit_mask.astype(hit_outputs.dtype)[..., None]
            missing_mask_expanded = missing_mask.astype(missing_outputs.dtype)[..., None]
            outputs = (hit_outputs * hit_mask_expanded) + (
                missing_outputs * missing_mask_expanded
            )
            expert_y = (outputs * scores[..., None]).sum(axis=-2).astype(outputs.dtype)
        else:
            expert_y = (hit_outputs * scores[..., None]).sum(axis=-2).astype(
                hit_outputs.dtype
            )

        shared_experts = getattr(mlp, "shared_experts", None)
        if shared_experts is not None:
            expert_y = expert_y + shared_experts(x)
        self._set_deepseek_external_resident_bytes()
        if events is not None:
            events.append(
                {
                    "kind": "load",
                    "action": (
                        "slot-arena-mixed-hit"
                        if not missing
                        else "slot-arena-mixed-hit-missing-direct-qmm"
                    ),
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    "hit_count": len(hits),
                    "missing_count": len(missing),
                    "seconds": time.perf_counter() - started,
                    "resident_bytes": self.session.resident_bytes,
                    "nbytes_loaded": nbytes_loaded,
                    "transient_page_bytes": transient_page_bytes,
                    "load_seconds": load_seconds,
                    "arena_resident_bytes": self._expert_slot_arena_bytes,
                    "arena_count": len(self._expert_slot_arenas),
                }
            )
        return expert_y

    def _deepseek_moe_forward_slot_arena_compact_direct_qmm(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        started = time.perf_counter()
        self._expert_slot_stats.attempts += 1
        self._expert_slot_stats.selected_rows += len(selected_experts)
        arena = self._expert_slot_arenas.get(layer_index)
        if arena is None:
            self._set_deepseek_slot_arena_full_hit(layer_index, False)
            self._expert_slot_stats.arena_misses += 1
            self._expert_slot_stats.missing_rows += len(selected_experts)
            self._expert_slot_stats.fallback_direct += 1
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        self._expert_slot_clock += 1
        arena.last_used = self._expert_slot_clock
        partition_started = time.perf_counter()
        hits = [expert for expert in selected_experts if expert in arena.expert_to_slot]
        missing = [expert for expert in selected_experts if expert not in arena.expert_to_slot]
        partition_seconds = time.perf_counter() - partition_started
        self._expert_slot_stats.hit_rows += len(hits)
        self._expert_slot_stats.missing_rows += len(missing)
        if not hits:
            self._set_deepseek_slot_arena_full_hit(layer_index, False)
            self._expert_slot_stats.arena_misses += 1
            self._expert_slot_stats.fallback_direct += 1
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        if missing:
            self._expert_slot_stats.arena_misses += 1
        else:
            self._expert_slot_stats.arena_hits += 1
        self._set_deepseek_slot_arena_full_hit(layer_index, not missing)

        nbytes_loaded = 0
        load_seconds = 0.0
        transient_page_bytes = 0
        missing_arrays: dict[str, Any] | None = None
        if missing:
            load_started = time.perf_counter()
            batch = self._load_deepseek_slice_batch(
                layer_index,
                missing,
                use_weight_page_cache=True,
                evaluate=False,
            )
            load_seconds = time.perf_counter() - load_started
            nbytes_loaded = batch.nbytes
            transient_page_bytes = int(getattr(batch, "transient_page_bytes", 0) or 0)
            missing_arrays = batch.arrays
            self._expert_slot_stats.loaded_bytes += batch.nbytes
            self._expert_slot_stats.load_seconds += load_seconds
            self._set_deepseek_external_resident_bytes(
                arena.nbytes + batch.nbytes + transient_page_bytes
            )

        assemble_started = time.perf_counter()
        compact_arrays = self._compact_slot_arena_arrays(
            arena,
            selected_experts,
            missing,
            missing_arrays,
        )
        assemble_seconds = time.perf_counter() - assemble_started
        remap_started = time.perf_counter()
        local_indices = remap_expert_indices(expert_indices, selected_experts)
        remap_seconds = time.perf_counter() - remap_started
        qmm_started = time.perf_counter()
        expert_y = self._deepseek_routed_qmm_from_arrays(
            mlp,
            layer_index,
            x,
            local_indices,
            scores,
            compact_arrays,
        )
        qmm_graph_seconds = time.perf_counter() - qmm_started
        total_seconds = time.perf_counter() - started
        self._expert_slot_stats.compact_calls += 1
        self._expert_slot_stats.compact_partition_seconds += partition_seconds
        self._expert_slot_stats.compact_assemble_seconds += assemble_seconds
        self._expert_slot_stats.compact_remap_seconds += remap_seconds
        self._expert_slot_stats.compact_qmm_graph_seconds += qmm_graph_seconds
        self._expert_slot_stats.compact_total_seconds += total_seconds
        self._set_deepseek_external_resident_bytes()
        if events is not None:
            events.append(
                {
                    "kind": "load",
                    "action": (
                        "slot-arena-compact-hit"
                        if not missing
                        else "slot-arena-compact-hit-missing-direct-qmm"
                    ),
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    "hit_count": len(hits),
                    "missing_count": len(missing),
                    "seconds": time.perf_counter() - started,
                    "resident_bytes": self.session.resident_bytes,
                    "nbytes_loaded": nbytes_loaded,
                    "transient_page_bytes": transient_page_bytes,
                    "load_seconds": load_seconds,
                    "partition_seconds": partition_seconds,
                    "compact_assemble_seconds": assemble_seconds,
                    "remap_seconds": remap_seconds,
                    "qmm_graph_seconds": qmm_graph_seconds,
                    "arena_resident_bytes": self._expert_slot_arena_bytes,
                    "arena_count": len(self._expert_slot_arenas),
                }
            )
        return expert_y

    def _deepseek_moe_forward_slot_arena_static_direct_defer(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        started = time.perf_counter()
        self._expert_slot_stats.attempts += 1
        self._expert_slot_stats.selected_rows += len(selected_experts)
        arena = self._expert_slot_arenas.get(layer_index)
        if arena is None:
            self._set_deepseek_slot_arena_full_hit(layer_index, False)
            self._expert_slot_stats.arena_misses += 1
            self._expert_slot_stats.missing_rows += len(selected_experts)
            self._expert_slot_stats.fallback_direct += 1
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        missing = [expert for expert in selected_experts if expert not in arena.expert_to_slot]
        if missing:
            self._set_deepseek_slot_arena_full_hit(layer_index, False)
            return self._deepseek_moe_forward_slot_arena_compact_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        self._expert_slot_clock += 1
        arena.last_used = self._expert_slot_clock
        self._expert_slot_stats.hit_rows += len(selected_experts)
        self._expert_slot_stats.arena_hits += 1
        self._set_deepseek_slot_arena_full_hit(layer_index, True)
        local_indices = remap_expert_indices_to_slots(expert_indices, arena.expert_to_slot)
        qmm_started = time.perf_counter()
        expert_y = self._deepseek_routed_qmm_from_arrays(
            mlp,
            layer_index,
            x,
            local_indices,
            scores,
            arena.arrays,
        )
        qmm_graph_seconds = time.perf_counter() - qmm_started
        total_seconds = time.perf_counter() - started
        self._expert_slot_stats.compact_calls += 1
        self._expert_slot_stats.compact_qmm_graph_seconds += qmm_graph_seconds
        self._expert_slot_stats.compact_total_seconds += total_seconds
        if events is not None:
            events.append(
                {
                    "kind": "load",
                    "action": "slot-arena-static-direct-full-hit",
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    "hit_count": len(selected_experts),
                    "missing_count": 0,
                    "seconds": time.perf_counter() - started,
                    "resident_bytes": self.session.resident_bytes,
                    "nbytes_loaded": 0,
                    "transient_page_bytes": 0,
                    "load_seconds": 0.0,
                    "qmm_graph_seconds": qmm_graph_seconds,
                    "arena_resident_bytes": self._expert_slot_arena_bytes,
                    "arena_count": len(self._expert_slot_arenas),
                }
            )
        return expert_y

    def _compact_slot_arena_take_plan(
        self,
        arena: DeepSeekExpertSlotArena,
        selected_experts: list[int],
        missing: list[int],
    ) -> dict[str, list[int] | list[bool]]:
        missing_to_row = {expert: row for row, expert in enumerate(missing)}
        hit_indices: list[int] = []
        missing_indices: list[int] = []
        hit_mask: list[bool] = []
        for expert in selected_experts:
            slot = arena.expert_to_slot.get(expert)
            if slot is not None:
                hit_indices.append(slot)
                missing_indices.append(0)
                hit_mask.append(True)
                continue
            if expert not in missing_to_row:
                raise KeyError(f"missing expert {expert} is not present in loaded rows")
            hit_indices.append(0)
            missing_indices.append(missing_to_row[expert])
            hit_mask.append(False)
        return {
            "hit_indices": hit_indices,
            "missing_indices": missing_indices,
            "hit_mask": hit_mask,
        }

    def _compact_slot_arena_arrays(
        self,
        arena: DeepSeekExpertSlotArena,
        selected_experts: list[int],
        missing: list[int],
        missing_arrays: dict[str, Any] | None,
    ) -> dict[str, Any]:
        import mlx.core as mx

        missing_to_row = {expert: row for row, expert in enumerate(missing)}
        compact: dict[str, Any] = {}
        for name, arena_array in arena.arrays.items():
            rows = []
            missing_array = missing_arrays.get(name) if missing_arrays is not None else None
            for expert in selected_experts:
                slot = arena.expert_to_slot.get(expert)
                if slot is not None:
                    rows.append(arena_array[slot : slot + 1])
                    continue
                if missing_array is None:
                    raise KeyError(f"missing expert {expert} has no loaded rows for {name}")
                row = missing_to_row[expert]
                rows.append(missing_array[row : row + 1])
            compact[name] = mx.concatenate(rows, axis=0)
        return compact

    def _deepseek_hotcold_route_metadata(
        self,
        expert_ids: list[int],
        arena: DeepSeekExpertSlotArena,
        missing: list[int],
    ) -> tuple[list[int], list[int]]:
        missing_to_row = {expert: row for row, expert in enumerate(missing)}
        route_source: list[int] = []
        local_indices: list[int] = []
        for expert in expert_ids:
            slot = arena.expert_to_slot.get(expert)
            if slot is not None:
                route_source.append(0)
                local_indices.append(slot)
                continue
            if expert not in missing_to_row:
                raise KeyError(f"missing expert {expert} is not present in loaded rows")
            route_source.append(1)
            local_indices.append(missing_to_row[expert])
        return route_source, local_indices

    def _deepseek_should_defer_layer_eval(self, layer_index: int) -> bool:
        if self.expert_compute_mode not in {
            "slot_arena_compact_defer_eval",
            "slot_arena_static_direct_defer",
        }:
            return False
        if layer_index not in getattr(self, "base_retain_layers", set()):
            return False
        return bool(
            getattr(self, "_last_deepseek_slot_arena_full_hit", {}).get(layer_index, False)
        )

    def _set_deepseek_slot_arena_full_hit(self, layer_index: int, full_hit: bool) -> None:
        last_hits = getattr(self, "_last_deepseek_slot_arena_full_hit", None)
        if last_hits is None:
            last_hits = {}
            self._last_deepseek_slot_arena_full_hit = last_hits
        last_hits[layer_index] = full_hit

    def _deepseek_moe_forward_slot_arena_hotcold_qmv(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx
        from smarttensor.hotcold_qmm import (
            hotcold_gather_qmv_affine8_norepeat,
            hotcold_gather_qmv_affine8_weighted_sum,
        )

        started = time.perf_counter()
        self._expert_slot_stats.attempts += 1
        self._expert_slot_stats.selected_rows += len(selected_experts)
        arena = self._expert_slot_arenas.get(layer_index)
        if arena is None:
            self._expert_slot_stats.arena_misses += 1
            self._expert_slot_stats.missing_rows += len(selected_experts)
            self._expert_slot_stats.fallback_direct += 1
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        self._expert_slot_clock += 1
        arena.last_used = self._expert_slot_clock
        partition_started = time.perf_counter()
        hits = [expert for expert in selected_experts if expert in arena.expert_to_slot]
        missing = [expert for expert in selected_experts if expert not in arena.expert_to_slot]
        partition_seconds = time.perf_counter() - partition_started
        self._expert_slot_stats.hit_rows += len(hits)
        self._expert_slot_stats.missing_rows += len(missing)
        if not hits:
            self._expert_slot_stats.arena_misses += 1
            self._expert_slot_stats.fallback_direct += 1
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        if missing:
            self._expert_slot_stats.arena_misses += 1
        else:
            self._expert_slot_stats.arena_hits += 1

        nbytes_loaded = 0
        load_seconds = 0.0
        transient_page_bytes = 0
        cold_arrays = arena.arrays
        if missing:
            load_started = time.perf_counter()
            batch = self._load_deepseek_slice_batch(
                layer_index,
                missing,
                use_weight_page_cache=True,
                evaluate=False,
            )
            load_seconds = time.perf_counter() - load_started
            nbytes_loaded = batch.nbytes
            transient_page_bytes = int(getattr(batch, "transient_page_bytes", 0) or 0)
            cold_arrays = batch.arrays
            self._expert_slot_stats.loaded_bytes += batch.nbytes
            self._expert_slot_stats.load_seconds += load_seconds
            self._set_deepseek_external_resident_bytes(
                arena.nbytes + batch.nbytes + transient_page_bytes
            )

        metadata_started = time.perf_counter()
        flat_experts = [int(v) for v in np.asarray(expert_indices).reshape(-1)]
        route_source_host, local_indices_host = self._deepseek_hotcold_route_metadata(
            flat_experts,
            arena,
            missing,
        )
        route_source = mx.array(route_source_host, dtype=mx.int32)
        local_indices = mx.array(local_indices_host, dtype=mx.int32)
        top_k = int(expert_indices.shape[-1])
        input_shape = tuple(int(dim) for dim in x.shape)
        hidden_dim = input_shape[-1]
        input_x = x.reshape((-1, hidden_dim))
        flat_scores = scores.reshape((-1,))
        metadata_seconds = time.perf_counter() - metadata_started

        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"

        def projection_arrays(source_arrays: dict[str, Any], projection: str) -> tuple[Any, Any, Any]:
            return (
                source_arrays[f"{prefix}.{projection}.weight"],
                source_arrays[f"{prefix}.{projection}.scales"],
                source_arrays[f"{prefix}.{projection}.biases"],
            )

        qmv_started = time.perf_counter()
        up = hotcold_gather_qmv_affine8_norepeat(
            input_x,
            *projection_arrays(arena.arrays, "up_proj"),
            *projection_arrays(cold_arrays, "up_proj"),
            route_source,
            local_indices,
            top_k=top_k,
        )
        gate = hotcold_gather_qmv_affine8_norepeat(
            input_x,
            *projection_arrays(arena.arrays, "gate_proj"),
            *projection_arrays(cold_arrays, "gate_proj"),
            route_source,
            local_indices,
            top_k=top_k,
        )
        activated = mlp.switch_mlp.activation(up, gate)
        down = hotcold_gather_qmv_affine8_weighted_sum(
            activated,
            *projection_arrays(arena.arrays, "down_proj"),
            *projection_arrays(cold_arrays, "down_proj"),
            route_source,
            local_indices,
            flat_scores,
            top_k=top_k,
        )
        expert_y = down.reshape(input_shape)
        shared_experts = getattr(mlp, "shared_experts", None)
        if shared_experts is not None:
            expert_y = expert_y + shared_experts(x)
        qmv_graph_seconds = time.perf_counter() - qmv_started
        total_seconds = time.perf_counter() - started
        self._expert_slot_stats.compact_calls += 1
        self._expert_slot_stats.compact_partition_seconds += partition_seconds
        self._expert_slot_stats.compact_remap_seconds += metadata_seconds
        self._expert_slot_stats.compact_qmm_graph_seconds += qmv_graph_seconds
        self._expert_slot_stats.compact_total_seconds += total_seconds
        self._set_deepseek_external_resident_bytes()
        if events is not None:
            events.append(
                {
                    "kind": "load",
                    "action": (
                        "slot-arena-hotcold-qmv-hit"
                        if not missing
                        else "slot-arena-hotcold-qmv-hit-missing-direct"
                    ),
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    "hit_count": len(hits),
                    "missing_count": len(missing),
                    "seconds": time.perf_counter() - started,
                    "resident_bytes": self.session.resident_bytes,
                    "nbytes_loaded": nbytes_loaded,
                    "transient_page_bytes": transient_page_bytes,
                    "load_seconds": load_seconds,
                    "partition_seconds": partition_seconds,
                    "metadata_seconds": metadata_seconds,
                    "qmv_graph_seconds": qmv_graph_seconds,
                    "arena_resident_bytes": self._expert_slot_arena_bytes,
                    "arena_count": len(self._expert_slot_arenas),
                }
            )
        return expert_y

    def _deepseek_moe_forward_split_overlap_down(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
        overlap_shared: bool = False,
    ) -> Any:
        import mlx.core as mx

        started = time.perf_counter()
        use_weight_page_cache = True
        estimated_table_bytes = self._deepseek_slice_nbytes(layer_index, selected_experts)
        estimated_page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            selected_experts,
            cap_to_headroom=True,
        )
        estimated_transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            selected_experts,
            cap_to_headroom=False,
        )
        if getattr(self, "resident_budget_bytes", None) is not None:
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=estimated_table_bytes + estimated_page_growth_bytes,
                temporary_bytes=estimated_transient_page_bytes,
            ):
                if estimated_transient_page_bytes and self._deepseek_budget_allows(
                    additional_sidecar_bytes=estimated_table_bytes,
                ):
                    use_weight_page_cache = False
                    estimated_transient_page_bytes = 0
                else:
                    self._require_deepseek_budget(
                        action="load-selected-experts-split-overlap-down",
                        additional_sidecar_bytes=estimated_table_bytes,
                        temporary_bytes=estimated_transient_page_bytes,
                    )
        gate_up_names = self._expert_projection_slice_names(layer_index, ("gate_proj", "up_proj"))
        down_names = self._expert_projection_slice_names(layer_index, ("down_proj",))
        gate_up_bytes = self._deepseek_named_slice_nbytes(gate_up_names, selected_experts)
        down_bytes = self._deepseek_named_slice_nbytes(down_names, selected_experts)
        gate_up_page_transient = (
            self.session.loader.estimate_first_dim_slice_page_miss_bytes(
                gate_up_names,
                selected_experts,
                cap_to_headroom=False,
            )
            if use_weight_page_cache
            else 0
        )
        self._set_deepseek_external_resident_bytes(
            gate_up_bytes + gate_up_page_transient
        )
        shared_experts = getattr(mlp, "shared_experts", None)
        shared_y = None
        gate_up_future = None
        if overlap_shared and shared_experts is not None:
            gate_up_future = self.loader_executor.submit(
                self.session.loader.load_first_dim_slices,
                gate_up_names,
                selected_experts,
                evaluate=False,
                use_weight_page_cache=use_weight_page_cache,
            )
            shared_y = shared_experts(x)
            mx.async_eval(shared_y)
            gate_up = gate_up_future.result()
        else:
            gate_up = self.session.loader.load_first_dim_slices(
                gate_up_names,
                selected_experts,
                evaluate=False,
                use_weight_page_cache=use_weight_page_cache,
            )
        local_indices = remap_expert_indices(expert_indices, selected_experts)
        x_expanded = mx.expand_dims(x, (-2, -3))
        up = self._deepseek_projection_qmm(
            mlp,
            layer_index,
            "up_proj",
            x_expanded,
            local_indices,
            gate_up.arrays,
        )
        gate = self._deepseek_projection_qmm(
            mlp,
            layer_index,
            "gate_proj",
            x_expanded,
            local_indices,
            gate_up.arrays,
        )
        activated = mlp.switch_mlp.activation(up, gate)
        mx.async_eval(activated)
        self._set_deepseek_external_resident_bytes(
            gate_up.nbytes + down_bytes + int(gate_up.transient_page_bytes or 0)
        )
        future = self.loader_executor.submit(
            self.session.loader.load_first_dim_slices,
            down_names,
            selected_experts,
            evaluate=False,
            use_weight_page_cache=use_weight_page_cache,
        )
        try:
            down = future.result()
            down_transient = int(getattr(down, "transient_page_bytes", 0) or 0)
            self._set_deepseek_external_resident_bytes(
                gate_up.nbytes + down.nbytes + down_transient
            )
            expert_y = self._deepseek_projection_qmm(
                mlp,
                layer_index,
                "down_proj",
                activated,
                local_indices,
                down.arrays,
            )
            expert_y = expert_y.squeeze(-2)
            expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
            if shared_y is not None:
                expert_y = expert_y + shared_y
            elif shared_experts is not None:
                expert_y = expert_y + shared_experts(x)
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": (
                            "load-selected-experts-split-overlap-shared-down"
                            if overlap_shared and shared_experts is not None
                            else "load-selected-experts-split-overlap-down"
                        ),
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                        "seconds": time.perf_counter() - started,
                        "resident_bytes": self.session.resident_bytes,
                        "nbytes_loaded": gate_up.nbytes + down.nbytes,
                        "transient_page_bytes": int(gate_up.transient_page_bytes or 0) + down_transient,
                    }
                )
            return expert_y
        finally:
            self._set_deepseek_external_resident_bytes()

    def _can_stream_deepseek_experts_per_expert(self, expert_indices: Any) -> bool:
        shape = tuple(int(dim) for dim in getattr(expert_indices, "shape", ()))
        return len(shape) == 3 and shape[0] == 1 and shape[1] == 1

    def _deepseek_moe_forward_per_expert(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        topk_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx

        started = time.perf_counter()
        x_expanded = mx.expand_dims(x, (-2, -3))
        local_index = mx.array([[[0]]], dtype=mx.int32)
        outputs: list[Any] = []
        nbytes_loaded = 0
        transient_page_bytes = 0

        try:
            for expert_id in topk_experts:
                row_bytes = self._deepseek_slice_nbytes(layer_index, [expert_id])
                page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
                    layer_index,
                    [expert_id],
                    cap_to_headroom=True,
                )
                page_transient_bytes = self._deepseek_slice_page_miss_nbytes(
                    layer_index,
                    [expert_id],
                    cap_to_headroom=False,
                )
                use_weight_page_cache = True
                if not self._deepseek_budget_allows(
                    additional_sidecar_bytes=row_bytes + page_growth_bytes,
                    temporary_bytes=page_transient_bytes,
                ):
                    if page_transient_bytes and self._deepseek_budget_allows(
                        additional_sidecar_bytes=row_bytes,
                    ):
                        use_weight_page_cache = False
                        page_transient_bytes = 0
                    else:
                        self._require_deepseek_budget(
                            action="stream-selected-expert",
                            additional_sidecar_bytes=row_bytes,
                            temporary_bytes=page_transient_bytes,
                        )

                self._set_deepseek_external_resident_bytes(
                    row_bytes + page_transient_bytes
                )
                batch = self._load_deepseek_slice_batch(
                    layer_index,
                    [expert_id],
                    use_weight_page_cache=use_weight_page_cache,
                )
                batch_transient_page_bytes = int(
                    getattr(batch, "transient_page_bytes", 0) or 0
                )
                nbytes_loaded += batch.nbytes
                transient_page_bytes += batch_transient_page_bytes
                self._set_deepseek_external_resident_bytes(
                    batch.nbytes + batch_transient_page_bytes
                )
                out = self._deepseek_single_expert_glu(
                    mlp,
                    layer_index,
                    expert_id,
                    x_expanded,
                    local_index,
                    batch.arrays,
                )
                mx.eval(out)
                out = self._maybe_detach_glm_array(out)
                del batch
                outputs.append(out)
                self._set_deepseek_external_resident_bytes()

            expert_y = mx.concatenate(outputs, axis=-2).squeeze(-3)
            expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
            shared_experts = getattr(mlp, "shared_experts", None)
            if shared_experts is not None:
                expert_y = expert_y + shared_experts(x)
            return expert_y
        finally:
            self._set_deepseek_external_resident_bytes()
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": "stream-selected-experts-per-expert",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(topk_experts),
                        "experts": topk_experts,
                        "seconds": time.perf_counter() - started,
                        "resident_bytes": self.session.resident_bytes,
                        "nbytes_loaded": nbytes_loaded,
                        "transient_page_bytes": transient_page_bytes,
                    }
                )

    def _deepseek_single_expert_glu(
        self,
        mlp: Any,
        layer_index: int,
        expert_id: int,
        x_expanded: Any,
        local_index: Any,
        arrays: dict[str, Any],
    ) -> Any:
        import mlx.core as mx

        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"

        def project(name: str, inputs: Any) -> Any:
            module = getattr(mlp.switch_mlp, name)
            return mx.gather_qmm(
                inputs,
                arrays[f"{prefix}.{name}.weight"],
                arrays[f"{prefix}.{name}.scales"],
                arrays[f"{prefix}.{name}.biases"],
                rhs_indices=local_index,
                transpose=True,
                group_size=module.group_size,
                bits=module.bits,
                mode=module.mode,
                sorted_indices=False,
            )

        up = project("up_proj", x_expanded)
        gate = project("gate_proj", x_expanded)
        return project("down_proj", mlp.switch_mlp.activation(up, gate))

    def _ensure_deepseek_slot_arena(
        self,
        layer_index: int,
        selected_experts: list[int],
    ) -> tuple[DeepSeekExpertSlotArena, dict[str, Any]] | None:
        capacity = int(self.expert_slot_capacity or len(selected_experts))
        if not selected_experts or len(selected_experts) > capacity:
            return None
        if self.expert_prefetch != "off":
            return None
        if getattr(self, "resident_budget_bytes", None) is None:
            return None
        # A retained slot arena adds persistent sidecar memory on top of the
        # already-tight DeepSeek phase path. Below ~8 GiB the direct lazy-QMM
        # path has a lower measured peak, so keep the hard low-memory tier
        # honest and use the arena only when there is real headroom.
        if self.resident_budget_bytes < 8 * 1024 * 1024 * 1024:
            return None

        arena = self._expert_slot_arenas.get(layer_index)
        if arena is not None and arena.capacity < capacity:
            self._evict_deepseek_slot_arena(layer_index)
            arena = None

        if arena is None:
            return self._create_deepseek_slot_arena(layer_index, selected_experts, capacity)

        self._expert_slot_clock += 1
        arena.last_used = self._expert_slot_clock
        hits = [expert for expert in selected_experts if expert in arena.expert_to_slot]
        missing = [expert for expert in selected_experts if expert not in arena.expert_to_slot]
        self._expert_slot_stats.hit_rows += len(hits)
        self._expert_slot_stats.missing_rows += len(missing)
        if not missing:
            self._expert_slot_stats.arena_hits += 1
            return arena, {
                "action": "slot-arena-hit",
                "hit_count": len(hits),
                "missing_count": 0,
                "nbytes_loaded": 0,
                "arena_resident_bytes": self._expert_slot_arena_bytes,
                "arena_count": len(self._expert_slot_arenas),
            }
        self._expert_slot_stats.arena_misses += 1
        if not getattr(self, "expert_slot_update_missing", True):
            return None
        updated = self._update_deepseek_slot_arena(layer_index, arena, selected_experts, missing)
        if updated is None:
            return None
        return updated

    def _create_deepseek_slot_arena(
        self,
        layer_index: int,
        selected_experts: list[int],
        capacity: int,
    ) -> tuple[DeepSeekExpertSlotArena, dict[str, Any]] | None:
        import mlx.core as mx

        names = self._expert_slice_names(layer_index)
        arena_bytes = self._deepseek_slot_arena_nbytes(layer_index, capacity)
        load_bytes = self._deepseek_named_slice_nbytes(names, selected_experts)
        page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            selected_experts,
            cap_to_headroom=True,
        )
        transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            selected_experts,
            cap_to_headroom=False,
        )
        use_weight_page_cache = True
        temporary_bytes = transient_page_bytes + (load_bytes if capacity > len(selected_experts) else 0)
        total_required_bytes = arena_bytes + temporary_bytes + 512 * 1024 * 1024
        if not self._deepseek_budget_allows(
            additional_sidecar_bytes=arena_bytes + page_growth_bytes,
            temporary_bytes=temporary_bytes,
        ) or not self._deepseek_total_resident_allows(total_required_bytes):
            self._evict_deepseek_slot_arenas_until(
                additional_sidecar_bytes=arena_bytes + page_growth_bytes,
                temporary_bytes=temporary_bytes,
                additional_total_bytes=total_required_bytes,
                protected_layer=layer_index,
                current_layer=layer_index,
            )
        if not self._deepseek_budget_allows(
            additional_sidecar_bytes=arena_bytes + page_growth_bytes,
            temporary_bytes=temporary_bytes,
        ) or not self._deepseek_total_resident_allows(total_required_bytes):
            direct_total_required = (
                arena_bytes
                + (load_bytes if capacity > len(selected_experts) else 0)
                + 512 * 1024 * 1024
            )
            if transient_page_bytes and self._deepseek_budget_allows(
                additional_sidecar_bytes=arena_bytes,
                temporary_bytes=(load_bytes if capacity > len(selected_experts) else 0),
            ) and self._deepseek_total_resident_allows(direct_total_required):
                use_weight_page_cache = False
                transient_page_bytes = 0
                temporary_bytes = load_bytes if capacity > len(selected_experts) else 0
            else:
                return None

        started = time.perf_counter()
        batch = self._load_deepseek_slice_batch(
            layer_index,
            selected_experts,
            use_weight_page_cache=use_weight_page_cache,
            evaluate=False,
        )
        load_seconds = time.perf_counter() - started
        batch_transient_page_bytes = int(getattr(batch, "transient_page_bytes", 0) or 0)
        arrays = batch.arrays
        update_seconds = 0.0
        if capacity > len(selected_experts):
            update_started = time.perf_counter()
            padded: dict[str, Any] = {}
            self._set_deepseek_external_resident_bytes(
                arena_bytes + batch.nbytes + batch_transient_page_bytes
            )
            for name, array in batch.arrays.items():
                slot_array = mx.zeros((capacity, *array.shape[1:]), dtype=array.dtype)
                for row_index in range(len(selected_experts)):
                    slot_array = mx.slice_update(
                        slot_array,
                        array[row_index : row_index + 1],
                        start_indices=mx.array(row_index),
                        axes=(0,),
                    )
                padded[name] = slot_array
            mx.eval(list(padded.values()))
            update_seconds = time.perf_counter() - update_started
            arrays = padded
        else:
            self._set_deepseek_external_resident_bytes(arena_bytes + batch_transient_page_bytes)

        slot_to_expert: list[int | None] = [None] * capacity
        expert_to_slot: dict[int, int] = {}
        for slot, expert in enumerate(selected_experts):
            slot_to_expert[slot] = expert
            expert_to_slot[expert] = slot
        self._expert_slot_clock += 1
        arena = DeepSeekExpertSlotArena(
            layer_index=layer_index,
            capacity=capacity,
            arrays=arrays,
            slot_to_expert=slot_to_expert,
            expert_to_slot=expert_to_slot,
            nbytes=arena_bytes,
            last_used=self._expert_slot_clock,
        )
        self._expert_slot_arenas[layer_index] = arena
        self._expert_slot_arena_bytes += arena_bytes
        self._expert_slot_stats.arena_creates += 1
        self._expert_slot_stats.arena_misses += 1
        self._expert_slot_stats.missing_rows += len(selected_experts)
        self._expert_slot_stats.loaded_bytes += batch.nbytes
        self._expert_slot_stats.load_seconds += load_seconds
        self._expert_slot_stats.update_seconds += update_seconds
        self._set_deepseek_external_resident_bytes()
        return arena, {
            "action": (
                "slot-arena-create"
                if use_weight_page_cache
                else "slot-arena-create-direct-budget"
            ),
            "hit_count": 0,
            "missing_count": len(selected_experts),
            "nbytes_loaded": batch.nbytes,
            "transient_page_bytes": batch_transient_page_bytes,
            "load_seconds": load_seconds,
            "update_seconds": update_seconds,
            "arena_resident_bytes": self._expert_slot_arena_bytes,
            "arena_count": len(self._expert_slot_arenas),
        }

    def _update_deepseek_slot_arena(
        self,
        layer_index: int,
        arena: DeepSeekExpertSlotArena,
        selected_experts: list[int],
        missing: list[int],
    ) -> tuple[DeepSeekExpertSlotArena, dict[str, Any]] | None:
        import mlx.core as mx

        selected_set = set(selected_experts)
        free_slots = [
            slot for slot, expert in enumerate(arena.slot_to_expert) if expert is None
        ]
        evictable_slots = [
            slot
            for slot, expert in enumerate(arena.slot_to_expert)
            if expert is not None and expert not in selected_set
        ]
        target_slots = (free_slots + evictable_slots)[: len(missing)]
        if len(target_slots) < len(missing):
            return None
        slot_assignments = list(zip(sorted(target_slots), missing, strict=True))
        ordered_slots = [slot for slot, _expert in slot_assignments]
        ordered_missing = [expert for _slot, expert in slot_assignments]
        update_runs: list[tuple[int, int, int]] = []
        if ordered_slots:
            run_start_row = 0
            run_start_slot = ordered_slots[0]
            previous_slot = run_start_slot
            for row_index, slot in enumerate(ordered_slots[1:], start=1):
                if slot != previous_slot + 1:
                    update_runs.append(
                        (run_start_slot, run_start_row, row_index - run_start_row)
                    )
                    run_start_row = row_index
                    run_start_slot = slot
                previous_slot = slot
            update_runs.append(
                (run_start_slot, run_start_row, len(ordered_slots) - run_start_row)
            )

        names = self._expert_slice_names(layer_index)
        load_bytes = self._deepseek_named_slice_nbytes(names, missing)
        page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            missing,
            cap_to_headroom=True,
        )
        transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            missing,
            cap_to_headroom=False,
        )
        use_weight_page_cache = True
        temporary_bytes = arena.nbytes + load_bytes + transient_page_bytes
        total_required_bytes = temporary_bytes + 512 * 1024 * 1024
        if not self._deepseek_budget_allows(
            additional_sidecar_bytes=page_growth_bytes,
            temporary_bytes=temporary_bytes,
        ) or not self._deepseek_total_resident_allows(total_required_bytes):
            self._evict_deepseek_slot_arenas_until(
                additional_sidecar_bytes=page_growth_bytes,
                temporary_bytes=temporary_bytes,
                additional_total_bytes=total_required_bytes,
                protected_layer=layer_index,
                current_layer=layer_index,
            )
        if not self._deepseek_budget_allows(
            additional_sidecar_bytes=page_growth_bytes,
            temporary_bytes=temporary_bytes,
        ) or not self._deepseek_total_resident_allows(total_required_bytes):
            if transient_page_bytes and self._deepseek_budget_allows(
                additional_sidecar_bytes=0,
                temporary_bytes=arena.nbytes + load_bytes,
            ) and self._deepseek_total_resident_allows(
                arena.nbytes + load_bytes + 512 * 1024 * 1024
            ):
                use_weight_page_cache = False
                transient_page_bytes = 0
                temporary_bytes = arena.nbytes + load_bytes
            else:
                return None

        load_started = time.perf_counter()
        batch = self._load_deepseek_slice_batch(
            layer_index,
            ordered_missing,
            use_weight_page_cache=use_weight_page_cache,
            evaluate=False,
        )
        load_seconds = time.perf_counter() - load_started
        batch_transient_page_bytes = int(getattr(batch, "transient_page_bytes", 0) or 0)
        update_started = time.perf_counter()
        self._set_deepseek_external_resident_bytes(
            arena.nbytes + batch.nbytes + batch_transient_page_bytes
        )
        updated_arrays: dict[str, Any] = {}
        for name, array in arena.arrays.items():
            updated = array
            loaded = batch.arrays[name]
            for slot, row_index, count in update_runs:
                updated = mx.slice_update(
                    updated,
                    loaded[row_index : row_index + count],
                    start_indices=mx.array(slot),
                    axes=(0,),
                )
            updated_arrays[name] = updated
        mx.eval(list(updated_arrays.values()))
        update_seconds = time.perf_counter() - update_started

        for slot, expert in slot_assignments:
            old_expert = arena.slot_to_expert[slot]
            if old_expert is not None:
                arena.expert_to_slot.pop(old_expert, None)
            arena.slot_to_expert[slot] = expert
            arena.expert_to_slot[expert] = slot
        arena.arrays = updated_arrays
        self._expert_slot_clock += 1
        arena.last_used = self._expert_slot_clock
        self._expert_slot_stats.arena_updates += 1
        self._expert_slot_stats.loaded_bytes += batch.nbytes
        self._expert_slot_stats.load_seconds += load_seconds
        self._expert_slot_stats.update_seconds += update_seconds
        self._expert_slot_stats.slot_update_ops += len(update_runs) * len(updated_arrays)
        self._set_deepseek_external_resident_bytes()
        return arena, {
            "action": (
                "slot-arena-update"
                if use_weight_page_cache
                else "slot-arena-update-direct-budget"
            ),
            "hit_count": len(selected_experts) - len(missing),
            "missing_count": len(missing),
            "nbytes_loaded": batch.nbytes,
            "transient_page_bytes": batch_transient_page_bytes,
            "load_seconds": load_seconds,
            "update_seconds": update_seconds,
            "slot_update_ops": len(update_runs) * len(updated_arrays),
            "arena_resident_bytes": self._expert_slot_arena_bytes,
            "arena_count": len(self._expert_slot_arenas),
        }

    def _deepseek_slot_arena_nbytes(self, layer_index: int, capacity: int) -> int:
        if capacity <= 0:
            return 0
        total = 0
        for name in self._expert_slice_names(layer_index):
            record = self.session.loader.manifest.tensors[name]
            total += record.nbytes * capacity // record.shape[0]
        return total

    def _evict_deepseek_slot_arena(self, layer_index: int) -> int:
        arena = self._expert_slot_arenas.pop(layer_index, None)
        if arena is None:
            return 0
        self._expert_slot_arena_bytes -= arena.nbytes
        self._expert_slot_stats.evictions += 1
        self._set_deepseek_external_resident_bytes()
        return arena.nbytes

    def _evict_deepseek_slot_arenas_until(
        self,
        *,
        additional_sidecar_bytes: int = 0,
        temporary_bytes: int = 0,
        additional_total_bytes: int = 0,
        protected_layer: int | None = None,
        current_layer: int | None = None,
    ) -> None:
        while (
            self._expert_slot_arenas
            and (
                not self._deepseek_budget_allows(
                    additional_sidecar_bytes=additional_sidecar_bytes,
                    temporary_bytes=temporary_bytes,
                )
                or not self._deepseek_total_resident_allows(
                    additional_total_bytes
                )
            )
        ):
            candidates = [
                (arena.last_used, layer)
                for layer, arena in self._expert_slot_arenas.items()
                if layer != protected_layer
            ]
            if current_layer is not None:
                past_candidates = [
                    (last_used, layer)
                    for last_used, layer in candidates
                    if layer < current_layer
                ]
                if past_candidates:
                    candidates = past_candidates
                else:
                    return
            if not candidates:
                return
            _, layer_to_evict = min(candidates)
            self._evict_deepseek_slot_arena(layer_to_evict)

    def _clear_deepseek_slot_arenas(self) -> None:
        arenas = getattr(self, "_expert_slot_arenas", None)
        if not arenas:
            self._expert_slot_arena_bytes = 0
            return
        arenas.clear()
        self._expert_slot_arena_bytes = 0

    def _deepseek_projection_qmm(
        self,
        mlp: Any,
        layer_index: int,
        projection: str,
        inputs: Any,
        local_indices: Any,
        arrays: dict[str, Any],
    ) -> Any:
        import mlx.core as mx

        prefix = f"model.layers.{layer_index}.mlp.switch_mlp.{projection}"
        module = getattr(mlp.switch_mlp, projection)
        return mx.gather_qmm(
            inputs,
            arrays[f"{prefix}.weight"],
            arrays[f"{prefix}.scales"],
            arrays[f"{prefix}.biases"],
            rhs_indices=local_indices,
            transpose=True,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
            sorted_indices=False,
        )

    def _expert_slice_names(self, layer_index: int) -> tuple[str, ...]:
        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"
        return tuple(
            f"{prefix}.{projection}.{field}"
            for projection in ("gate_proj", "up_proj", "down_proj")
            for field in ("weight", "scales", "biases")
        )

    def _expert_projection_slice_names(
        self,
        layer_index: int,
        projections: tuple[str, ...],
    ) -> tuple[str, ...]:
        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"
        return tuple(
            f"{prefix}.{projection}.{field}"
            for projection in projections
            for field in ("weight", "scales", "biases")
        )

    def _deepseek_named_slice_nbytes(
        self,
        names: tuple[str, ...],
        expert_ids: list[int],
    ) -> int:
        if not expert_ids:
            return 0
        total = 0
        for name in names:
            record = self.session.loader.manifest.tensors[name]
            total += record.nbytes * len(expert_ids) // record.shape[0]
        return total

    def _load_deepseek_selected_experts(
        self,
        layer_index: int,
        mlp: Any,
        selected_experts: list[int],
    ) -> MlxStreamEvent:
        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        use_weight_page_cache = True
        if getattr(self, "resident_budget_bytes", None) is not None:
            estimated_table_bytes = self._deepseek_slice_nbytes(layer_index, selected_experts)
            estimated_page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=True,
            )
            estimated_transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=False,
            )
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=estimated_table_bytes + estimated_page_growth_bytes,
                temporary_bytes=estimated_transient_page_bytes,
            ):
                if estimated_transient_page_bytes and self._deepseek_budget_allows(
                    additional_sidecar_bytes=estimated_table_bytes,
                    temporary_bytes=0,
                ):
                    use_weight_page_cache = False
                else:
                    self._require_deepseek_budget(
                        action="load-selected-experts",
                        additional_sidecar_bytes=estimated_table_bytes,
                        temporary_bytes=estimated_transient_page_bytes,
                    )
        batch = self._load_deepseek_slice_batch(
            layer_index,
            selected_experts,
            use_weight_page_cache=use_weight_page_cache,
            evaluate=False,
        )
        batch_transient_page_bytes = int(
            getattr(batch, "transient_page_bytes", 0) or 0
        )
        if batch_transient_page_bytes:
            self._set_deepseek_external_resident_bytes(
                batch.nbytes + batch_transient_page_bytes
            )
        self._assign_deepseek_expert_tables(mlp, layer_index, batch.arrays)
        self._set_deepseek_external_resident_bytes(batch.nbytes)
        event = MlxStreamEvent(
            action=(
                "load-selected-experts"
                if use_weight_page_cache
                else "load-selected-experts-direct-budget"
            ),
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names,
            nbytes_loaded=batch.nbytes,
            transient_page_bytes=batch_transient_page_bytes,
        )
        if self._trace:
            self.session.events.append(event)
        return event

    def _load_deepseek_slice_batch(
        self,
        layer_index: int,
        expert_ids: list[int],
        *,
        use_weight_page_cache: bool = True,
        evaluate: bool | None = None,
    ) -> MlxTensorBatch:
        if evaluate is None:
            evaluate = self.session.evaluate
        return self.session.loader.load_first_dim_slices(
            self._expert_slice_names(layer_index),
            expert_ids,
            evaluate=evaluate,
            use_weight_page_cache=use_weight_page_cache,
        )

    def _deepseek_slice_nbytes(self, layer_index: int, expert_ids: list[int]) -> int:
        if not expert_ids:
            return 0
        total = 0
        for name in self._expert_slice_names(layer_index):
            record = self.session.loader.manifest.tensors[name]
            total += record.nbytes * len(expert_ids) // record.shape[0]
        return total

    def _deepseek_slice_page_miss_nbytes(
        self,
        layer_index: int,
        expert_ids: list[int],
        *,
        cap_to_headroom: bool = False,
    ) -> int:
        return self.session.loader.estimate_first_dim_slice_page_miss_bytes(
            self._expert_slice_names(layer_index),
            expert_ids,
            cap_to_headroom=cap_to_headroom,
        )

    def _assign_deepseek_expert_tables(
        self,
        mlp: Any,
        layer_index: int,
        arrays: dict[str, Any],
    ) -> None:
        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(mlp.switch_mlp, projection)
            module.weight = arrays[f"{prefix}.{projection}.weight"]
            module.scales = arrays[f"{prefix}.{projection}.scales"]
            module.biases = arrays[f"{prefix}.{projection}.biases"]

    def _maybe_submit_deepseek_expert_prefetch(
        self,
        layer_index: int,
        layer: Any,
        *,
        events: list[dict[str, Any]] | None = None,
        pass_kind: str | None = None,
        token_step: int | None = None,
    ) -> None:
        if self.expert_prefetch not in {"previous", "previous_table"}:
            return
        if not hasattr(layer.mlp, "switch_mlp"):
            return
        if self._pending_expert_prefetch is not None:
            return
        history = self._expert_history.get(layer_index)
        if not history:
            self._prefetch_stats.skipped_no_history += 1
            return
        if len(history) > self.expert_prefetch_cap:
            self._prefetch_stats.skipped_over_cap += 1
            return
        predicted = list(history)
        mode = self.expert_prefetch
        pending_page_bytes = self._deepseek_slice_page_miss_nbytes(layer_index, predicted)
        if mode == "previous_table":
            predicted_table_bytes = self._deepseek_slice_nbytes(layer_index, predicted)
            predicted_pending_bytes = predicted_table_bytes + pending_page_bytes
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=predicted_pending_bytes,
            ):
                if self._deepseek_budget_allows(additional_sidecar_bytes=pending_page_bytes):
                    mode = "previous"
                    self._prefetch_stats.table_prefetch_downgrades += 1
                    if events is not None:
                        events.append(
                            {
                                "kind": "load",
                                "action": "prefetch-table-downgraded-budget",
                                "pass": pass_kind,
                                "token_step": token_step,
                                "layer": layer_index,
                                "predicted_experts": predicted,
                                "predicted_table_bytes": predicted_table_bytes,
                                "pending_page_bytes": pending_page_bytes,
                                "resident_budget_bytes": self.resident_budget_bytes,
                            }
                        )
                else:
                    self._prefetch_stats.skipped_over_budget += 1
                    if events is not None:
                        events.append(
                            {
                                "kind": "load",
                                "action": "prefetch-skipped-over-budget",
                                "pass": pass_kind,
                                "token_step": token_step,
                                "layer": layer_index,
                                "predicted_experts": predicted,
                                "predicted_table_bytes": predicted_table_bytes,
                                "pending_page_bytes": pending_page_bytes,
                                "resident_budget_bytes": self.resident_budget_bytes,
                            }
                        )
                    return
        elif not self._deepseek_budget_allows(additional_sidecar_bytes=pending_page_bytes):
            self._prefetch_stats.skipped_over_budget += 1
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": "prefetch-skipped-over-budget",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "predicted_experts": predicted,
                        "pending_page_bytes": pending_page_bytes,
                        "resident_budget_bytes": self.resident_budget_bytes,
                    }
                )
            return

        if mode == "previous_table":
            future = self.loader_executor.submit(
                self._load_deepseek_slice_batch,
                layer_index,
                predicted,
            )
            self._pending_expert_prefetch_bytes = (
                self._deepseek_slice_nbytes(layer_index, predicted) + pending_page_bytes
            )
        else:
            future = self.loader_executor.submit(
                self.session.loader.warm_first_dim_slices,
                self._expert_slice_names(layer_index),
                predicted,
            )
            self._pending_expert_prefetch_bytes = pending_page_bytes
        self._set_deepseek_external_resident_bytes()
        self._pending_expert_prefetch = {
            "layer": layer_index,
            "experts": predicted,
            "future": future,
            "mode": mode,
        }

    def _consume_deepseek_expert_prefetch(
        self,
        layer_index: int,
        mlp: Any,
        selected_experts: list[int],
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> list[int] | None:
        pending = self._pending_expert_prefetch
        self._pending_expert_prefetch = None
        self._pending_expert_prefetch_bytes = 0
        if pending is None:
            return None

        join_started = time.perf_counter()
        batch = pending["future"].result()
        join_wait = time.perf_counter() - join_started
        if pending["layer"] != layer_index:
            self._prefetch_stats.skipped_no_history += 1
            return None

        predicted = pending["experts"]
        ordering, hits, missing = expert_merge_plan(predicted, selected_experts)
        wasted = sorted(set(predicted) - set(selected_experts))

        stats = self._prefetch_stats
        stats.attempted_layers += 1
        stats.predicted_rows += len(predicted)
        stats.true_rows += len(selected_experts)
        stats.hit_rows += len(hits)
        stats.missing_rows += len(missing)
        stats.wasted_rows += len(wasted)
        stats.prefetched_bytes += batch.nbytes
        row_bytes = batch.nbytes // len(predicted) if predicted else 0
        stats.wasted_bytes += row_bytes * len(wasted)
        stats.prefetch_load_seconds += batch.seconds
        stats.join_wait_seconds += join_wait
        if not missing:
            stats.full_hits += 1

        fallback_seconds = 0.0
        fallback_bytes = 0
        fallback_transient_page_bytes = 0
        assemble_seconds = 0.0
        assemble_temporary_bytes = 0
        if pending.get("mode") == "previous_table":
            arrays = dict(batch.arrays)
            batch_transient_page_bytes = int(
                getattr(batch, "transient_page_bytes", 0) or 0
            )
            if not missing and not self._deepseek_budget_allows(
                additional_sidecar_bytes=batch.nbytes,
            ):
                stats.skipped_over_budget += 1
                if events is not None:
                    events.append(
                        {
                            "kind": "load",
                            "action": "prefetch-selected-expert-table-over-budget",
                            "pass": pass_kind,
                            "token_step": token_step,
                            "layer": layer_index,
                            "predicted_experts": predicted,
                            "experts": selected_experts,
                            "estimated_table_bytes": batch.nbytes,
                        }
                    )
                self._set_deepseek_external_resident_bytes()
                return None
            if missing:
                if getattr(self, "resident_budget_bytes", None) is not None:
                    estimated_fallback_bytes = self._deepseek_slice_nbytes(layer_index, missing)
                    estimated_fallback_transient = self._deepseek_slice_page_miss_nbytes(
                        layer_index,
                        missing,
                    )
                    estimated_assembly_peak = (
                        batch_transient_page_bytes
                        + estimated_fallback_transient
                        + 2 * (batch.nbytes + estimated_fallback_bytes)
                    )
                    if not self._deepseek_budget_allows(
                        temporary_bytes=estimated_assembly_peak,
                    ):
                        stats.skipped_over_budget += 1
                        if events is not None:
                            events.append(
                                {
                                    "kind": "load",
                                    "action": "prefetch-selected-expert-table-over-budget",
                                    "pass": pass_kind,
                                    "token_step": token_step,
                                    "layer": layer_index,
                                    "predicted_experts": predicted,
                                    "experts": selected_experts,
                                    "estimated_assembly_peak": estimated_assembly_peak,
                                }
                            )
                        self._set_deepseek_external_resident_bytes()
                        return None

                fallback_started = time.perf_counter()
                missing_batch = self._load_deepseek_slice_batch(layer_index, missing)
                fallback_seconds = time.perf_counter() - fallback_started
                fallback_bytes = missing_batch.nbytes
                fallback_transient_page_bytes = int(
                    getattr(missing_batch, "transient_page_bytes", 0) or 0
                )
                stats.fallback_bytes += fallback_bytes
                stats.fallback_load_seconds += fallback_seconds

                import mlx.core as mx

                assemble_started = time.perf_counter()
                assemble_temporary_bytes = (
                    batch_transient_page_bytes
                    + fallback_transient_page_bytes
                    + 2 * (batch.nbytes + missing_batch.nbytes)
                )
                stats.max_assemble_temporary_bytes = max(
                    stats.max_assemble_temporary_bytes,
                    assemble_temporary_bytes,
                )
                self._set_deepseek_external_resident_bytes(assemble_temporary_bytes)
                arrays = {
                    name: mx.concatenate([arrays[name], missing_batch.arrays[name]], axis=0)
                    for name in arrays
                }
                mx.eval(list(arrays.values()))
                assemble_seconds = time.perf_counter() - assemble_started
                stats.assemble_seconds += assemble_seconds

            self._assign_deepseek_expert_tables(mlp, layer_index, arrays)
            self._set_deepseek_external_resident_bytes(batch.nbytes + fallback_bytes)

            event = MlxStreamEvent(
                action="prefetch-selected-expert-table",
                layer=layer_index,
                seconds=join_wait + fallback_seconds + assemble_seconds,
                resident_bytes=self.session.resident_bytes,
                requested=self._expert_slice_names(layer_index),
                loaded=self._expert_slice_names(layer_index),
                nbytes_loaded=batch.nbytes + fallback_bytes,
                transient_page_bytes=batch_transient_page_bytes + fallback_transient_page_bytes,
            )
            if self._trace:
                self.session.events.append(event)
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": "prefetch-selected-expert-table",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "predicted_experts": predicted,
                        "experts": selected_experts,
                        "table_order": ordering,
                        "hit_count": len(hits),
                        "missing_count": len(missing),
                        "wasted_count": len(wasted),
                        "join_wait_seconds": join_wait,
                        "fallback_seconds": fallback_seconds,
                        "assemble_seconds": assemble_seconds,
                        "assemble_temporary_bytes": assemble_temporary_bytes,
                        "fallback_transient_page_bytes": fallback_transient_page_bytes,
                        "weight_page_cache_bytes": batch.weight_page_cache_bytes,
                        **compact_event(event),
                    }
                )
            return ordering

        if events is not None:
            events.append(
                {
                    "kind": "load",
                    "action": "warm-selected-expert-pages",
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "predicted_experts": predicted,
                    "experts": selected_experts,
                    "hit_count": len(hits),
                    "missing_count": len(missing),
                    "wasted_count": len(wasted),
                    "join_wait_seconds": join_wait,
                    "nbytes_loaded": batch.nbytes,
                    "weight_page_cache_bytes": batch.weight_page_cache_bytes,
                }
            )
        return None

    def _clear_deepseek_selected_experts(self, layer: Any) -> None:
        import mlx.core as mx

        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp.switch_mlp, projection)
            module.weight = mx.zeros((0, 0, 0), dtype=mx.uint32)
            module.scales = mx.zeros((0, 0, 0), dtype=mx.bfloat16)
            module.biases = mx.zeros((0, 0, 0), dtype=mx.bfloat16)
        self._set_deepseek_external_resident_bytes()

    def _deepseek_compute_binds_switch_mlp(self) -> bool:
        if self.expert_prefetch != "off":
            return True
        return self.expert_compute_mode in {"table", "table_overlap_shared"}

    def _expert_telemetry(self) -> dict[str, Any] | None:
        weight_pages = self.session.loader.weight_page_summary()
        slot_arena_active = self.expert_compute_mode in {
            "slot_arena_direct_qmm",
            "slot_arena_guarded_direct_qmm",
            "slot_arena_mixed_direct_qmm",
            "slot_arena_compact_direct_qmm",
            "slot_arena_compact_defer_eval",
            "slot_arena_static_direct_defer",
            "slot_arena_hotcold_qmv",
        }
        if weight_pages is None and self.expert_prefetch == "off" and not slot_arena_active:
            return None
        return {
            "mode": "deepseek-weight-pages",
            "weight_page_budget_bytes": self.weight_page_budget_bytes,
            "weight_page_policy": self.weight_page_policy,
            "weight_page_rows": self.weight_page_rows,
            "weight_pages": weight_pages,
            "expert_prefetch_mode": self.expert_prefetch,
            "expert_prefetch": (
                self._prefetch_stats.to_dict()
                if self.expert_prefetch != "off"
                else None
            ),
            "expert_slot_arena": (
                self._expert_slot_stats.to_dict(
                    resident_bytes=self._expert_slot_arena_bytes,
                    arena_count=len(self._expert_slot_arenas),
                    capacity=int(self.expert_slot_capacity or 0),
                )
                if slot_arena_active
                else None
            ),
        }

    def _clear_deepseek_mla_projection_tables(self, layer: Any) -> None:
        import mlx.core as mx

        for module in (layer.self_attn.embed_q, layer.self_attn.unembed_out):
            module.weight = mx.zeros((0, 0, 0), dtype=mx.uint32)
            if hasattr(module, "scales"):
                module.scales = mx.zeros((0, 0, 0), dtype=mx.bfloat16)
            if hasattr(module, "biases"):
                module.biases = mx.zeros((0, 0, 0), dtype=mx.bfloat16)

    def _deepseek_missing_role_bytes(self, role: str) -> int:
        return sum(
            record.nbytes
            for record in self.session.loader.manifest.tensors.values()
            if record.role == role and record.name not in self.session.resident
        )

    def _clear_weight_pages_for_deepseek_output_budget(
        self,
        events: list[dict[str, Any]] | None,
    ) -> None:
        budget = getattr(self, "resident_budget_bytes", None)
        page_bytes = self.session.loader.weight_page_resident_bytes
        if budget is None or not page_bytes:
            return
        output_bytes = self._deepseek_missing_role_bytes("output")
        if self.session.resident_bytes + output_bytes <= budget:
            return

        evicted_bytes = self.session.loader.clear_weight_page_cache()
        self._set_deepseek_external_resident_bytes()
        if events is not None:
            events.append(
                {
                    "kind": "evict",
                    "action": "clear-weight-pages-for-output-budget",
                    "seconds": 0.0,
                    "resident_bytes": self.session.resident_bytes,
                    "nbytes_evicted": evicted_bytes,
                    "output_bytes": output_bytes,
                    "budget_bytes": budget,
                }
            )

    def _logits_from_hidden(self, hidden: Any, events: list[dict[str, Any]]) -> Any:
        import mlx.core as mx

        trace = self._trace
        if self.pin_policy == "phase" and not self.warm_output:
            self._before_phase_output_load(events if trace else [])
            self._clear_weight_pages_for_deepseek_output_budget(events if trace else None)
            load_output = self.session.load_output()
            if trace:
                events.append({"kind": "load", **compact_event(load_output)})

        logits_started = time.perf_counter() if trace else 0.0
        hidden = self.session.model.model.norm(hidden)
        logits = self.session.model.lm_head(hidden)
        mx.eval(logits)
        if trace:
            events.append(
                {
                    "kind": "compute",
                    "action": "logits",
                    "seconds": time.perf_counter() - logits_started,
                    "logits_shape": list(logits.shape),
                }
            )
        if self.pin_policy == "phase" and not self.warm_output:
            evict_output = self.session.evict_output()
            if trace:
                events.append({"kind": "evict", **compact_event(evict_output)})
        return logits



def load_mlx_config(model_dir: Path) -> dict[str, Any]:
    from mlx_lm.utils import load_config

    config = load_config(model_dir)
    if "quantization_config" not in config:
        text_config = config.get("text_config", {})
        if "quantization_config" in text_config:
            config["quantization_config"] = text_config["quantization_config"]
    return config


def load_native_mlx_array(safe_file: SafeTensorFile, record: TensorRecord) -> Any:
    import mlx.core as mx

    tensor = safe_file.tensor(record.name)
    try:
        dtype = numpy_dtype(record.dtype)
        np_array = np.frombuffer(tensor.view, dtype=dtype).reshape(record.shape)
        mlx_array = mx.array(np_array)
        if record.dtype == "BF16":
            mlx_array = mlx_array.view(mx.bfloat16)
        return mlx_array
    finally:
        tensor.release()


def contiguous_runs(indices: tuple[int, ...]) -> list[tuple[int, int]]:
    """Collapse a strictly-ascending index tuple into [start, stop) runs.

    Adjacent ascending ids merge into one run, so the concatenation of
    ``arr[start:stop]`` over the runs reproduces ``arr`` restricted to those
    ids in the given order. A request for adjacent experts (or a single
    expert) yields one run -> a pure basic slice with no gather. Callers must
    pass an ascending, duplicate-free tuple (see ``gather_first_dim_rows``).
    """

    runs: list[tuple[int, int]] = []
    for index in indices:
        if runs and index == runs[-1][1]:
            start, _ = runs[-1]
            runs[-1] = (start, index + 1)
        else:
            runs.append((index, index + 1))
    return runs


def gather_first_dim_rows(np_array: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Extract first-dim rows, preferring contiguous basic slices over a gather.

    numpy fancy indexing (``np_array[[e0, e1, ...]]``) forces a strided copy
    that measured ~1.4-2.0x slower than a contiguous basic slice of the same
    bytes (Sprint 5 forensics; re-measured this lane). When ``indices`` is
    already strictly ascending -- which every runtime caller is, since expert
    ids arrive via ``selected_expert_ids`` (sorted unique) -- we collapse it
    into contiguous runs and extract each run as ``np_array[start:stop]``. A
    single run (the common top-k case) needs no concatenation at all.

    The returned rows are in exactly ``indices`` order, so this is a bitwise
    drop-in for ``np_array[list(indices)]``. If ``indices`` is not strictly
    ascending we fall back to the fancy-index gather to preserve that order.
    """

    is_ascending = all(
        indices[i] < indices[i + 1] for i in range(len(indices) - 1)
    )
    if not is_ascending:
        return np_array[list(indices)]
    if not indices:
        return np.ascontiguousarray(np_array[:0])

    runs = contiguous_runs(indices)
    if len(runs) == 1:
        start, stop = runs[0]
        # Basic slice: contiguous source rows, no fancy-index gather.
        return np.ascontiguousarray(np_array[start:stop])
    rows = np.empty((len(indices), *np_array.shape[1:]), dtype=np_array.dtype)
    position = 0
    for start, stop in runs:
        count = stop - start
        rows[position : position + count] = np_array[start:stop]
        position += count
    return rows


def load_native_mlx_array_first_dim_indices(
    safe_file: SafeTensorFile,
    record: TensorRecord,
    indices: tuple[int, ...],
) -> Any:
    import mlx.core as mx

    tensor = safe_file.tensor(record.name)
    try:
        dtype = numpy_dtype(record.dtype)
        np_array = np.frombuffer(tensor.view, dtype=dtype).reshape(record.shape)
        selected = gather_first_dim_rows(np_array, indices)
        mlx_array = mx.array(selected)
        if record.dtype == "BF16":
            mlx_array = mlx_array.view(mx.bfloat16)
        return mlx_array
    finally:
        tensor.release()


def load_native_mlx_array_first_dim_indices_pread(
    safe_file: SafeTensorFile,
    record: TensorRecord,
    indices: tuple[int, ...],
) -> Any:
    """``pread`` twin of ``load_native_mlx_array_first_dim_indices``.

    Returns a result BYTE-IDENTICAL to the mmap reader for the same args, but
    reads ONLY the selected rows via ``safe_file.pread_range`` -- so it leaves
    no persistent mmap page-cache residency (the macOS jetsam fix). The runtime
    always passes strictly ascending unique row ids (expert ids arrive sorted
    unique), the same precondition ``gather_first_dim_rows`` relies on. For any
    non-ascending order (defensive; the runtime never hits it) we delegate to
    the mmap reader so the requested-row order is preserved exactly.
    """
    import mlx.core as mx

    is_ascending = all(
        indices[i] < indices[i + 1] for i in range(len(indices) - 1)
    )
    if not is_ascending:
        # Order semantics for unsorted/duplicate indices live in the mmap
        # reader (fancy-index gather); keep one source of truth for them.
        return load_native_mlx_array_first_dim_indices(safe_file, record, indices)

    dtype = numpy_dtype(record.dtype)
    if not indices:
        np_rows = np.empty((0, *record.shape[1:]), dtype=dtype)
        mlx_array = mx.array(np_rows)
        if record.dtype == "BF16":
            mlx_array = mlx_array.view(mx.bfloat16)
        return mlx_array

    meta = safe_file.tensors[record.name]
    abs_start = meta.absolute_offsets[0]
    row_bytes = record.nbytes // record.shape[0]

    runs = contiguous_runs(indices)
    if len(runs) == 1:
        start, stop = runs[0]
        buf: bytes | bytearray = safe_file.pread_range(
            abs_start + start * row_bytes, abs_start + stop * row_bytes
        )
    else:
        assembled = bytearray()
        for start, stop in runs:
            assembled += safe_file.pread_range(
                abs_start + start * row_bytes, abs_start + stop * row_bytes
            )
        buf = assembled

    np_rows = np.frombuffer(buf, dtype=dtype).reshape(len(indices), *record.shape[1:])
    mlx_array = mx.array(np_rows)
    if record.dtype == "BF16":
        mlx_array = mlx_array.view(mx.bfloat16)
    return mlx_array


def numpy_dtype(dtype: str) -> np.dtype:
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
        "BF16": np.dtype("<u2"),
        "F32": np.dtype("<f4"),
        "F64": np.dtype("<f8"),
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported native MLX dtype: {dtype}") from exc


def compact_event(event: MlxStreamEvent) -> dict[str, Any]:
    return {
        "action": event.action,
        "layer": event.layer,
        "seconds": event.seconds,
        "resident_bytes": event.resident_bytes,
        "requested_count": len(event.requested),
        "loaded_count": len(event.loaded),
        "skipped_count": len(event.skipped),
        "evicted_count": len(event.evicted),
        "nbytes_loaded": event.nbytes_loaded,
        "transient_page_bytes": event.transient_page_bytes,
    }


def kv_cache_nbytes(cache: list[Any]) -> int:
    total = 0
    for item in cache:
        total += int(getattr(item, "nbytes", 0) or 0)
    return total


def select_retained_layers_for_budget(
    manifest: SmartTensorManifest,
    resident_budget_bytes: int,
    *,
    pin_policy: str = "all",
    warm_embeddings: bool = False,
) -> set[int]:
    """Select a prefix of layers that fits within the runtime overlap budget."""

    if resident_budget_bytes < 0:
        raise ValueError("resident_budget_bytes must be non-negative")
    if pin_policy not in {"all", "phase"}:
        raise ValueError("pin_policy must be 'all' or 'phase'")
    if warm_embeddings and pin_policy != "phase":
        raise ValueError("warm_embeddings only applies to pin_policy='phase'")

    base_bytes = pinned_budget_bytes(
        manifest,
        pin_policy=pin_policy,
        warm_embeddings=warm_embeddings,
    )
    phase_role_bytes = (
        phase_role_budget_bytes(manifest, warm_embeddings=warm_embeddings)
        if pin_policy == "phase"
        else 0
    )
    retained: set[int] = set()
    retained_bytes = 0
    layers = tuple(sorted(manifest.layers.items()))
    if resident_budget_bytes <= base_bytes:
        return retained

    for layer_index, layer in layers:
        candidate = retained | {layer_index}
        candidate_retained_bytes = retained_bytes + layer.nbytes
        streamed_layer_bytes = max(
            (candidate_layer.nbytes for candidate_index, candidate_layer in layers if candidate_index not in candidate),
            default=0,
        )
        runtime_overlap_bytes = max(streamed_layer_bytes, phase_role_bytes)
        if base_bytes + candidate_retained_bytes + runtime_overlap_bytes > resident_budget_bytes:
            break
        retained.add(layer_index)
        retained_bytes = candidate_retained_bytes
    return retained


def pinned_budget_bytes(
    manifest: SmartTensorManifest,
    *,
    pin_policy: str,
    warm_embeddings: bool = False,
) -> int:
    if pin_policy == "phase":
        total = sum(
            record.nbytes
            for record in pinned_tensors(manifest)
            if record.residency_hint == "pin-small"
        )
        if warm_embeddings:
            total += role_budget_bytes(manifest, "embedding")
        return total
    return sum(record.nbytes for record in pinned_tensors(manifest))


def phase_role_budget_bytes(
    manifest: SmartTensorManifest,
    *,
    warm_embeddings: bool = False,
) -> int:
    roles = ("output",) if warm_embeddings else ("embedding", "output")
    return max(
        (
            role_budget_bytes(manifest, role)
            for role in roles
        ),
        default=0,
    )


def role_budget_bytes(manifest: SmartTensorManifest, role: str) -> int:
    return sum(record.nbytes for record in manifest.tensors.values() if record.role == role)


def selected_expert_ids(indices: Any) -> list[int]:
    values = np.asarray(indices).reshape(-1)
    return sorted({int(value) for value in values})


def selected_expert_counts(indices: Any) -> dict[int, int]:
    values = np.asarray(indices).reshape(-1)
    counts: dict[int, int] = {}
    for value in values:
        expert = int(value)
        counts[expert] = counts.get(expert, 0) + 1
    return counts


def should_refresh_hot_set(
    current_order: list[int],
    missing: list[int],
    counts: dict[int, int],
    hysteresis: int,
) -> bool:
    """Gate hot-set membership rebuilds behind a count margin.

    A refresh copies the whole resident table (16 rows x 12 tensors), so a
    cold candidate must beat the weakest resident's frequency count by
    ``hysteresis`` before a rebuild is even considered. Without the margin,
    early noisy counts churned 142 rebuilds in 65 passes for no measured
    row-hit-rate gain.
    """

    if not current_order:
        return True
    if not missing:
        return False
    weakest = min(counts.get(expert, 0) for expert in current_order)
    strongest = max(counts.get(expert, 0) for expert in missing)
    return strongest >= weakest + hysteresis


def choose_frequency_hot_order(
    current_order: list[int],
    missing: list[int],
    counts: dict[int, int],
    cap: int,
) -> list[int]:
    if cap <= 0:
        return []

    current_set = set(current_order)
    missing_unique = [expert for expert in dict.fromkeys(missing) if expert not in current_set]
    candidates = list(current_order) + missing_unique
    current_position = {expert: index for index, expert in enumerate(current_order)}
    missing_position = {expert: index for index, expert in enumerate(missing_unique)}

    def key(expert: int) -> tuple[int, int, int]:
        return (
            -counts.get(expert, 0),
            0 if expert in current_set else 1,
            current_position.get(expert, len(current_order) + missing_position.get(expert, 0)),
        )

    return sorted(candidates, key=key)[:cap]


def remap_expert_indices(indices: Any, selected_experts: list[int]) -> Any:
    """Map global expert ids to row positions in ``selected_experts``.

    Vectorized position-lookup (callers guarantee every value appears in the
    table). Accepts an MLX array or an already-fetched numpy array, so hot
    loops can reuse one host copy of the indices.
    """

    import mlx.core as mx

    values = np.asarray(indices)
    table = np.asarray(selected_experts, dtype=np.int64)
    positions = np.zeros(int(table.max()) + 1 if table.size else 1, dtype=np.int32)
    positions[table] = np.arange(table.size, dtype=np.int32)
    return mx.array(positions[values])


def remap_expert_indices_to_slots(indices: Any, expert_to_slot: dict[int, int]) -> Any:
    """Map global expert ids to persistent arena slots."""

    import mlx.core as mx

    values = np.asarray(indices)
    if not expert_to_slot:
        return mx.array(np.zeros_like(values, dtype=np.int32))
    max_expert = max(expert_to_slot)
    positions = np.zeros(max_expert + 1, dtype=np.int32)
    for expert, slot in expert_to_slot.items():
        positions[expert] = slot
    return mx.array(positions[values])


def remap_expert_indices_to_slots_with_mask(
    indices: Any,
    expert_to_slot: dict[int, int],
) -> tuple[Any, Any]:
    """Map resident expert ids to slots and mark non-resident routes with zero.

    Non-resident positions map to slot 0 so gather_qmm receives valid dense
    indices; the paired mask lets callers zero their scores exactly.
    """

    import mlx.core as mx

    values = np.asarray(indices)
    max_value = int(values.max()) if values.size else 0
    max_expert = max(expert_to_slot, default=0)
    size = max(max_value, max_expert) + 1
    positions = np.zeros(size, dtype=np.int32)
    mask = np.zeros(size, dtype=np.float32)
    for expert, slot in expert_to_slot.items():
        positions[expert] = int(slot)
        mask[expert] = 1.0
    return mx.array(positions[values]), mx.array(mask[values])


def select_qwen_base_layers_for_budget(
    manifest: SmartTensorManifest,
    resident_budget_bytes: int,
    *,
    top_k: int,
    expert_marker: str = ".mlp.switch_mlp.",
) -> set[int]:
    if resident_budget_bytes < 0:
        raise ValueError("resident_budget_bytes must be non-negative")

    phase_role_bytes = phase_role_budget_bytes(manifest)
    pin_small_bytes = pinned_budget_bytes(manifest, pin_policy="phase")
    selected_expert_bytes = qwen_selected_expert_bytes(
        manifest, top_k=top_k, expert_marker=expert_marker
    )
    retained: set[int] = set()
    layers = tuple(sorted(manifest.layers.items()))
    if resident_budget_bytes <= phase_role_bytes + pin_small_bytes + selected_expert_bytes:
        return retained

    retained_bytes = 0
    for layer_index, layer in layers:
        base_bytes = qwen_layer_base_bytes(manifest, layer_index, expert_marker=expert_marker)
        candidate = retained | {layer_index}
        streamed_base_bytes = max(
            (
                qwen_layer_base_bytes(manifest, candidate_index, expert_marker=expert_marker)
                for candidate_index, _ in layers
                if candidate_index not in candidate
            ),
            default=0,
        )
        peak_bytes = (
            pin_small_bytes
            + phase_role_bytes
            + selected_expert_bytes
            + retained_bytes
            + base_bytes
            + streamed_base_bytes
        )
        if peak_bytes > resident_budget_bytes:
            break
        retained.add(layer_index)
        retained_bytes += base_bytes
    return retained


def reserve_weight_page_budget_for_base_retention(
    resident_budget_bytes: int,
    weight_page_budget_bytes: int | None,
) -> int:
    if resident_budget_bytes < 0:
        raise ValueError("resident_budget_bytes must be non-negative")
    if weight_page_budget_bytes is None:
        return resident_budget_bytes
    if weight_page_budget_bytes < 0:
        raise ValueError("weight_page_budget_bytes must be non-negative")
    return max(resident_budget_bytes - weight_page_budget_bytes, 0)


def qwen_layer_base_bytes(
    manifest: SmartTensorManifest,
    layer_index: int,
    *,
    expert_marker: str = ".mlp.switch_mlp.",
) -> int:
    layer = manifest.layers[layer_index]
    return sum(
        manifest.tensors[name].nbytes
        for name in layer.tensor_names
        if expert_marker not in name
    )


def qwen_selected_expert_bytes(
    manifest: SmartTensorManifest,
    *,
    top_k: int,
    expert_marker: str = ".mlp.switch_mlp.",
) -> int:
    first_layer = next(
        (
            layer
            for _, layer in sorted(manifest.layers.items())
            if any(expert_marker in name for name in layer.tensor_names)
        ),
        None,
    )
    if first_layer is None:
        return 0
    total = 0
    for name in first_layer.tensor_names:
        if expert_marker not in name:
            continue
        record = manifest.tensors[name]
        if not record.shape:
            continue
        total += record.nbytes * top_k // record.shape[0]
    return total


def build_mlx_model_shell(config: dict[str, Any], manifest: SmartTensorManifest) -> Any:
    from mlx import nn
    from mlx_lm.utils import _get_classes

    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    quantization = config.get("quantization")
    if quantization is None and (quantization_config := config.get("quantization_config")):
        if all(key in quantization_config for key in ("group_size", "bits")):
            quantization = quantization_config

    if quantization is not None:
        weight_names = set(manifest.tensors)
        group_size = quantization.get("group_size")
        bits = quantization.get("bits")
        mode = quantization.get("mode", "affine")

        def class_predicate(path: str, module: Any) -> bool | dict[str, Any]:
            custom = quantization.get(path)
            if isinstance(custom, dict):
                return custom
            if not hasattr(module, "to_quantized"):
                return False
            if config.get("model_type") == "deepseek_v3" and (
                path.endswith(".self_attn.embed_q")
                or path.endswith(".self_attn.unembed_out")
            ):
                return any(
                    candidate in weight_names
                    for candidate in (
                        path.replace(".embed_q", ".kv_b_proj") + ".scales",
                        path.replace(".unembed_out", ".kv_b_proj") + ".scales",
                    )
                )
            return f"{path}.scales" in weight_names

        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            mode=mode,
            class_predicate=class_predicate,
        )

    model.eval()
    return model


def sanitize_weights(model: Any, arrays: dict[str, Any]) -> dict[str, Any]:
    if hasattr(model, "sanitize"):
        return model.sanitize(arrays)
    return arrays


def placeholder_for_record(record: Any) -> Any:
    import mlx.core as mx

    return mx.zeros(record.shape, dtype=mlx_dtype(record.dtype))


def mlx_dtype(dtype: str) -> Any:
    import mlx.core as mx

    mapping = {
        "BOOL": mx.bool_,
        "U8": mx.uint8,
        "I8": mx.int8,
        "U16": mx.uint16,
        "I16": mx.int16,
        "U32": mx.uint32,
        "I32": mx.int32,
        "U64": mx.uint64,
        "I64": mx.int64,
        "F16": mx.float16,
        "BF16": mx.bfloat16,
        "F32": mx.float32,
        "F64": mx.float64,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported MLX dtype for placeholder: {dtype}") from exc


def clear_mlx_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()
    except Exception:
        pass
