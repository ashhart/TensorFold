"""Resident page cache for streamed model weights.

The cache is deliberately independent of MLX and safetensors. It owns only the
policy for admitting, protecting, touching, and evicting weight pages; callers
decide what a page payload is. In the MLX adapter a page is usually one or more
rows from a leading expert axis.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class WeightPageKey:
    """Stable identity for a row range from one tensor."""

    tensor_name: str
    row_start: int
    row_stop: int

    def __post_init__(self) -> None:
        if self.row_start < 0:
            raise ValueError("row_start must be non-negative")
        if self.row_stop <= self.row_start:
            raise ValueError("row_stop must be greater than row_start")


@dataclass(frozen=True)
class WeightPageSpec:
    """A requested page plus the requested rows inside it."""

    key: WeightPageKey
    nbytes: int
    row_indices: tuple[int, ...]


@dataclass
class WeightPageStats:
    """Runtime counters for page-cache decisions."""

    hits: int = 0
    misses: int = 0
    insertions: int = 0
    evictions: int = 0
    admission_rejections: int = 0
    protected_eviction_skips: int = 0
    bytes_read: int = 0
    bytes_loaded: int = 0
    bytes_evicted: int = 0
    resident_bytes: int = 0
    peak_resident_bytes: int = 0

    def to_dict(self) -> dict[str, int | float]:
        requests = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / requests if requests else 0.0,
            "insertions": self.insertions,
            "evictions": self.evictions,
            "admission_rejections": self.admission_rejections,
            "protected_eviction_skips": self.protected_eviction_skips,
            "bytes_read": self.bytes_read,
            "bytes_loaded": self.bytes_loaded,
            "bytes_evicted": self.bytes_evicted,
            "resident_bytes": self.resident_bytes,
            "peak_resident_bytes": self.peak_resident_bytes,
        }


@dataclass
class _ResidentPage:
    key: WeightPageKey
    nbytes: int
    value: Any
    pin_count: int = 0
    active_count: int = 0
    frequency: int = 1

    @property
    def protected(self) -> bool:
        return self.pin_count > 0 or self.active_count > 0


class PageLease:
    """Context manager that keeps one page eviction-protected while in use."""

    def __init__(self, cache: "PagedWeightCache", key: WeightPageKey, value: Any) -> None:
        self._cache = cache
        self.key = key
        self.value = value
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._cache.release(self.key)
            self._released = True

    def __enter__(self) -> Any:
        return self.value

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


class PagedWeightCache:
    """Budgeted cache for resident weight pages."""

    _LRU = "lru"
    _FREQUENCY = "frequency"
    _TWO_QUEUE = "two_queue"
    _POLICY_ALIASES = {
        _LRU: _LRU,
        _FREQUENCY: _FREQUENCY,
        _TWO_QUEUE: _TWO_QUEUE,
        "protected_window": _TWO_QUEUE,
        "probationary": _TWO_QUEUE,
        "scan_resistant": _FREQUENCY,
    }

    def __init__(self, budget_bytes: int, *, eviction_policy: str = _LRU) -> None:
        if budget_bytes < 0:
            raise ValueError("budget_bytes must be non-negative")
        policy = self._POLICY_ALIASES.get(eviction_policy)
        if policy is None:
            valid = ", ".join(sorted(self._POLICY_ALIASES))
            raise ValueError(f"eviction_policy must be one of: {valid}")
        self.budget_bytes = int(budget_bytes)
        self.eviction_policy = policy
        self._pages: OrderedDict[WeightPageKey, _ResidentPage] = OrderedDict()
        self._frequencies: dict[WeightPageKey, int] = {}
        self.stats = WeightPageStats()

    @property
    def resident_bytes(self) -> int:
        return self.stats.resident_bytes

    @property
    def resident_count(self) -> int:
        return len(self._pages)

    @property
    def _tracks_frequency(self) -> bool:
        return self.eviction_policy in {self._FREQUENCY, self._TWO_QUEUE}

    @property
    def _can_reject_admission(self) -> bool:
        return self.eviction_policy in {self._FREQUENCY, self._TWO_QUEUE}

    def contains(self, key: WeightPageKey) -> bool:
        return key in self._pages

    def get_or_load(
        self,
        key: WeightPageKey,
        *,
        nbytes: int,
        loader: Callable[[], Any],
        pin: bool = False,
    ) -> Any:
        """Return a page value, loading and admitting it on a miss."""

        return self._get_or_load(
            key,
            nbytes=nbytes,
            loader=loader,
            pin=pin,
            force_admit=pin,
        )

    def _get_or_load(
        self,
        key: WeightPageKey,
        *,
        nbytes: int,
        loader: Callable[[], Any],
        pin: bool = False,
        force_admit: bool = False,
    ) -> Any:
        page = self._pages.get(key)
        if page is not None:
            self.stats.hits += 1
            self._record_frequency(key, page)
            if pin:
                page.pin_count += 1
            self._pages.move_to_end(key)
            return page.value

        self.stats.misses += 1
        self._record_frequency(key)
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if nbytes > self.budget_bytes:
            raise ValueError(
                f"page {key} is {nbytes} bytes, larger than cache budget "
                f"{self.budget_bytes} bytes"
            )
        value = loader()
        self.stats.bytes_read += int(nbytes)
        self._put(
            key,
            value,
            nbytes=nbytes,
            pin=pin,
            allow_admission_reject=not force_admit,
        )
        return value

    def lease(
        self,
        key: WeightPageKey,
        *,
        nbytes: int,
        loader: Callable[[], Any],
        pin: bool = False,
    ) -> PageLease:
        """Return a context-managed, temporarily protected page value."""

        value = self._get_or_load(
            key,
            nbytes=nbytes,
            loader=loader,
            pin=pin,
            force_admit=True,
        )
        page = self._pages[key]
        page.active_count += 1
        return PageLease(self, key, value)

    def put(
        self,
        key: WeightPageKey,
        value: Any,
        *,
        nbytes: int,
        pin: bool = False,
    ) -> None:
        self._put(
            key,
            value,
            nbytes=nbytes,
            pin=pin,
            allow_admission_reject=False,
        )

    def _put(
        self,
        key: WeightPageKey,
        value: Any,
        *,
        nbytes: int,
        pin: bool = False,
        allow_admission_reject: bool = False,
    ) -> bool:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if nbytes > self.budget_bytes:
            raise ValueError(
                f"page {key} is {nbytes} bytes, larger than cache budget "
                f"{self.budget_bytes} bytes"
            )

        existing = self._pages.get(key)
        if existing is not None:
            if existing.protected:
                self.stats.protected_eviction_skips += 1
                raise MemoryError(f"cannot replace protected weight page {key}")

        if (
            allow_admission_reject
            and existing is None
            and self._should_reject_without_eviction_plan(key, nbytes)
        ):
            self.stats.admission_rejections += 1
            return False

        eviction_plan = self._eviction_plan_for(
            nbytes,
            replacing=key if existing is not None else None,
        )
        if eviction_plan is None:
            if allow_admission_reject and self._can_reject_admission:
                self.stats.admission_rejections += 1
                return False
            raise MemoryError("weight page cache budget is exhausted by protected pages")

        if (
            allow_admission_reject
            and self._can_reject_admission
            and self._should_reject_admission(key, eviction_plan)
        ):
            self.stats.admission_rejections += 1
            return False

        for victim_key in eviction_plan:
            self.evict(victim_key)
        if existing is not None:
            self._pages.pop(key)
            self._drop_page(existing)

        page = _ResidentPage(
            key=key,
            nbytes=int(nbytes),
            value=value,
            pin_count=1 if pin else 0,
            frequency=self._frequency_for(key),
        )
        self._pages[key] = page
        self.stats.insertions += 1
        self.stats.bytes_loaded += int(nbytes)
        self.stats.resident_bytes += int(nbytes)
        self.stats.peak_resident_bytes = max(
            self.stats.peak_resident_bytes,
            self.stats.resident_bytes,
        )
        return True

    def pin(self, key: WeightPageKey) -> None:
        page = self._pages[key]
        page.pin_count += 1
        self._pages.move_to_end(key)

    def unpin(self, key: WeightPageKey) -> None:
        page = self._pages[key]
        if page.pin_count <= 0:
            raise ValueError(f"page {key} is not pinned")
        page.pin_count -= 1

    def release(self, key: WeightPageKey) -> None:
        page = self._pages.get(key)
        if page is None:
            return
        if page.active_count <= 0:
            raise ValueError(f"page {key} has no active lease")
        page.active_count -= 1

    def evict(self, key: WeightPageKey, *, force: bool = False) -> bool:
        page = self._pages.get(key)
        if page is None:
            return False
        if page.protected and not force:
            self.stats.protected_eviction_skips += 1
            return False
        self._pages.pop(key)
        self._drop_page(page)
        return True

    def clear(self, *, force: bool = False) -> None:
        keys = list(self._pages)
        for key in keys:
            self.evict(key, force=force)

    def keys(self) -> tuple[WeightPageKey, ...]:
        return tuple(self._pages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.budget_bytes,
            "eviction_policy": self.eviction_policy,
            "resident_count": self.resident_count,
            "pages": [
                {
                    "tensor_name": page.key.tensor_name,
                    "row_start": page.key.row_start,
                    "row_stop": page.key.row_stop,
                    "nbytes": page.nbytes,
                    "pin_count": page.pin_count,
                    "active_count": page.active_count,
                    "frequency": page.frequency,
                }
                for page in self._pages.values()
            ],
            **self.stats.to_dict(),
        }

    def _evict_for(self, incoming_bytes: int) -> None:
        eviction_plan = self._eviction_plan_for(incoming_bytes)
        if eviction_plan is None:
            raise MemoryError("weight page cache budget is exhausted by protected pages")
        for victim_key in eviction_plan:
            self.evict(victim_key)

    def _first_evictable_key(self) -> WeightPageKey | None:
        victim_key: WeightPageKey | None = None
        victim_frequency: int | None = None
        for key, page in self._pages.items():
            if page.protected:
                self.stats.protected_eviction_skips += 1
                continue
            if self.eviction_policy == self._LRU:
                return key
            if victim_frequency is None or page.frequency < victim_frequency:
                victim_key = key
                victim_frequency = page.frequency
        return victim_key

    def _eviction_plan_for(
        self,
        incoming_bytes: int,
        *,
        replacing: WeightPageKey | None = None,
    ) -> tuple[WeightPageKey, ...] | None:
        replaced_bytes = 0
        if replacing is not None:
            existing = self._pages.get(replacing)
            replaced_bytes = existing.nbytes if existing is not None else 0
        effective_resident_bytes = self.stats.resident_bytes - replaced_bytes
        if effective_resident_bytes + incoming_bytes <= self.budget_bytes:
            return ()

        needed_bytes = effective_resident_bytes + incoming_bytes - self.budget_bytes
        freed_bytes = 0
        victims: list[WeightPageKey] = []
        candidates = []
        for position, (key, page) in enumerate(self._pages.items()):
            if key == replacing:
                continue
            if page.protected:
                self.stats.protected_eviction_skips += 1
                continue
            candidates.append((position, key, page))

        if self.eviction_policy == self._FREQUENCY:
            candidates.sort(key=lambda candidate: (candidate[2].frequency, candidate[0]))
        elif self.eviction_policy == self._TWO_QUEUE:
            candidates.sort(
                key=lambda candidate: (
                    0 if candidate[2].frequency <= 1 else 1,
                    candidate[2].frequency,
                    candidate[0],
                )
            )

        for _, key, page in candidates:
            victims.append(key)
            freed_bytes += page.nbytes
            if freed_bytes >= needed_bytes:
                return tuple(victims)
        return None

    def _should_reject_admission(
        self,
        key: WeightPageKey,
        eviction_plan: tuple[WeightPageKey, ...],
    ) -> bool:
        if not eviction_plan:
            return False
        incoming_frequency = self._frequency_for(key)
        if self.eviction_policy == self._TWO_QUEUE:
            victim_frequencies = [
                self._pages[victim_key].frequency for victim_key in eviction_plan
            ]
            if victim_frequencies and max(victim_frequencies) <= 1:
                return False
            return incoming_frequency <= max(victim_frequencies)
        return incoming_frequency <= max(
            self._pages[victim_key].frequency for victim_key in eviction_plan
        )

    def _should_reject_without_eviction_plan(
        self,
        key: WeightPageKey,
        incoming_bytes: int,
    ) -> bool:
        if self.eviction_policy != self._FREQUENCY:
            return False
        if self.stats.resident_bytes + incoming_bytes <= self.budget_bytes:
            return False

        incoming_frequency = self._frequency_for(key)
        weakest_frequency: int | None = None
        for page in self._pages.values():
            if page.protected:
                continue
            if weakest_frequency is None or page.frequency < weakest_frequency:
                weakest_frequency = page.frequency
        return weakest_frequency is not None and incoming_frequency <= weakest_frequency

    def _record_frequency(
        self,
        key: WeightPageKey,
        page: _ResidentPage | None = None,
    ) -> None:
        if not self._tracks_frequency:
            return
        frequency = self._frequencies.get(key, 0) + 1
        self._frequencies[key] = frequency
        if page is not None:
            page.frequency = frequency

    def _frequency_for(self, key: WeightPageKey) -> int:
        if not self._tracks_frequency:
            return 1
        return max(1, self._frequencies.get(key, 0))

    def _drop_page(self, page: _ResidentPage) -> None:
        self.stats.evictions += 1
        self.stats.bytes_evicted += page.nbytes
        self.stats.resident_bytes -= page.nbytes


def row_page_specs(
    tensor_name: str,
    *,
    shape: tuple[int, ...],
    tensor_nbytes: int,
    indices: tuple[int, ...],
    rows_per_page: int = 1,
) -> tuple[WeightPageSpec, ...]:
    """Plan first-axis row pages needed to serve ``indices``."""

    if not shape:
        raise ValueError("cannot page a scalar tensor by row")
    row_count = int(shape[0])
    if row_count <= 0:
        raise ValueError("tensor first dimension must be positive")
    if tensor_nbytes % row_count != 0:
        raise ValueError("tensor bytes must divide evenly by first dimension")
    if rows_per_page <= 0:
        raise ValueError("rows_per_page must be positive")

    row_nbytes = tensor_nbytes // row_count
    pages: dict[WeightPageKey, list[int]] = {}
    for raw_index in indices:
        index = int(raw_index)
        if index < 0 or index >= row_count:
            raise IndexError(
                f"row {index} out of range for {tensor_name} with {row_count} rows"
            )
        start = (index // rows_per_page) * rows_per_page
        stop = min(start + rows_per_page, row_count)
        key = WeightPageKey(tensor_name, start, stop)
        pages.setdefault(key, []).append(index)

    specs = []
    for key, requested in pages.items():
        specs.append(
            WeightPageSpec(
                key=key,
                nbytes=(key.row_stop - key.row_start) * row_nbytes,
                row_indices=tuple(requested),
            )
        )
    return tuple(sorted(specs, key=lambda spec: spec.key))
