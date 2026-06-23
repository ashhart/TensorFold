"""Persistent expert-slot arena contracts.

This module is intentionally model-free. It defines the slot/miss/remap behavior
needed by a future MLX arena that feeds resident expert slots to `gather_qmm`
and asks the loader only for cold misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable


@dataclass
class ArenaResult:
    hits: list[int] = field(default_factory=list)
    misses: list[int] = field(default_factory=list)
    evicted: list[int] = field(default_factory=list)


@runtime_checkable
class ExpertArena(Protocol):
    """Residency contract for planner-driven expert arenas."""

    def pin(self, layer_index: int, expert_ids: Sequence[int]) -> None: ...
    def ensure_resident(self, layer_index: int, expert_ids: Sequence[int]) -> ArenaResult: ...
    def slot_indices(self, layer_index: int, expert_ids: Sequence[int]) -> list[int]: ...
    def resident(self, layer_index: int) -> frozenset[int]: ...
    def resident_bytes(self) -> int: ...


class DictArena:
    """CPU reference arena with per-layer slots, hotset pins, and LRU/LFU eviction."""

    def __init__(self, capacity_per_layer: int, bytes_per_expert: int, policy: str = "lru") -> None:
        if capacity_per_layer < 1:
            raise ValueError("capacity_per_layer must be positive")
        if bytes_per_expert < 0:
            raise ValueError("bytes_per_expert must be non-negative")
        if policy not in {"lru", "lfu"}:
            raise ValueError("policy must be 'lru' or 'lfu'")
        self.capacity = int(capacity_per_layer)
        self.bytes_per_expert = int(bytes_per_expert)
        self.policy = policy
        self._layers: dict[int, dict[int, list[int]]] = {}
        self._pinned: dict[int, set[int]] = {}
        self._tick = 0

    def pin(self, layer_index: int, expert_ids: Sequence[int]) -> None:
        self._pinned.setdefault(int(layer_index), set()).update(int(expert) for expert in expert_ids)

    def ensure_resident(self, layer_index: int, expert_ids: Sequence[int]) -> ArenaResult:
        layer_index = int(layer_index)
        resident = self._layers.setdefault(layer_index, {})
        pinned = self._pinned.get(layer_index, set())
        result = ArenaResult()
        for expert in (int(expert) for expert in expert_ids):
            self._tick += 1
            if expert in resident:
                result.hits.append(expert)
                resident[expert][1] += 1
                resident[expert][2] = self._tick
                continue

            result.misses.append(expert)
            if len(resident) >= self.capacity:
                evictable = [candidate for candidate in resident if candidate not in pinned]
                if evictable:
                    if self.policy == "lru":
                        victim = min(evictable, key=lambda candidate: resident[candidate][2])
                    else:
                        victim = min(
                            evictable,
                            key=lambda candidate: (resident[candidate][1], resident[candidate][2]),
                        )
                    slot = resident[victim][0]
                    del resident[victim]
                    result.evicted.append(victim)
                    resident[expert] = [slot, 1, self._tick]
                    continue
            resident[expert] = [self._free_slot(resident), 1, self._tick]
        return result

    def slot_indices(self, layer_index: int, expert_ids: Sequence[int]) -> list[int]:
        resident = self._layers.get(int(layer_index), {})
        return [resident[int(expert)][0] for expert in expert_ids]

    def resident(self, layer_index: int) -> frozenset[int]:
        return frozenset(self._layers.get(int(layer_index), {}))

    def resident_bytes(self) -> int:
        return sum(len(resident) for resident in self._layers.values()) * self.bytes_per_expert

    @staticmethod
    def _free_slot(resident: dict[int, list[int]]) -> int:
        used = {meta[0] for meta in resident.values()}
        slot = 0
        while slot in used:
            slot += 1
        return slot


class ArenaDriver:
    """Connect a planner hotset to an arena and report realized online hits."""

    def __init__(self, arena: ExpertArena, plan: dict[str, Any]) -> None:
        self.arena = arena
        self.plan = plan
        self._hits = 0
        self._misses = 0

    def prewarm(self) -> None:
        for layer, experts in self.plan.get("resident_per_layer", {}).items():
            self.arena.pin(int(layer), experts)
            self.arena.ensure_resident(int(layer), experts)

    def route(self, layer_index: int, selected_experts: Sequence[int]) -> ArenaResult:
        result = self.arena.ensure_resident(layer_index, selected_experts)
        self._hits += len(result.hits)
        self._misses += len(result.misses)
        return result

    def totals(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }


@dataclass(frozen=True)
class ExpertArenaAccess:
    layer_index: int
    selected_experts: tuple[int, ...]
    selected_slots: tuple[int, ...]
    missing_experts: tuple[int, ...]
    slot_updates: tuple[tuple[int, int], ...]
    evicted_experts: tuple[int, ...]
    hit_count: int
    needs_fallback: bool = False

    @property
    def all_hot(self) -> bool:
        return not self.needs_fallback and not self.missing_experts


@dataclass
class ExpertArenaStats:
    accesses: int = 0
    fallback_accesses: int = 0
    selected_experts: int = 0
    hit_experts: int = 0
    missing_experts: int = 0
    slot_updates: int = 0
    evictions: int = 0

    def record(self, access: ExpertArenaAccess, *, arena_count: int) -> None:
        self.accesses += 1
        self.selected_experts += len(access.selected_experts)
        self.hit_experts += access.hit_count
        self.missing_experts += len(access.missing_experts)
        self.slot_updates += len(access.slot_updates)
        self.evictions += len(access.evicted_experts)
        if access.needs_fallback:
            self.fallback_accesses += 1
        self._arena_count = arena_count

    def to_dict(self) -> dict[str, int | float]:
        hit_rate = self.hit_experts / self.selected_experts if self.selected_experts else 0.0
        return {
            "accesses": self.accesses,
            "fallback_accesses": self.fallback_accesses,
            "selected_experts": self.selected_experts,
            "hit_experts": self.hit_experts,
            "missing_experts": self.missing_experts,
            "hit_rate": hit_rate,
            "slot_updates": self.slot_updates,
            "evictions": self.evictions,
            "arena_count": int(getattr(self, "_arena_count", 0)),
        }


@dataclass
class ExpertSlotArena:
    layer_index: int
    capacity: int
    bytes_per_slot: int
    slot_to_expert: list[int | None] = field(init=False)
    expert_to_slot: dict[int, int] = field(default_factory=dict)
    pinned_experts: set[int] = field(default_factory=set)
    _last_used: dict[int, int] = field(default_factory=dict)
    _clock: int = 0

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if self.bytes_per_slot < 0:
            raise ValueError("bytes_per_slot must be non-negative")
        self.slot_to_expert = [None] * self.capacity

    @property
    def resident_bytes(self) -> int:
        return self.capacity * self.bytes_per_slot

    def pin(self, experts: Iterable[int]) -> ExpertArenaAccess:
        selected = tuple(sorted(set(int(expert) for expert in experts)))
        access = self.plan_access(selected)
        if not access.needs_fallback:
            self.pinned_experts.update(selected)
        return access

    def plan_access(self, selected_experts: Iterable[int]) -> ExpertArenaAccess:
        selected = tuple(int(expert) for expert in selected_experts)
        unique_selected = tuple(sorted(set(selected)))
        if len(unique_selected) > self.capacity:
            return ExpertArenaAccess(
                layer_index=self.layer_index,
                selected_experts=selected,
                selected_slots=(),
                missing_experts=unique_selected,
                slot_updates=(),
                evicted_experts=(),
                hit_count=0,
                needs_fallback=True,
            )

        self._clock += 1
        hit_unique = [expert for expert in unique_selected if expert in self.expert_to_slot]
        missing = [expert for expert in unique_selected if expert not in self.expert_to_slot]
        for expert in hit_unique:
            self._last_used[expert] = self._clock

        if len(missing) > self._available_slot_count(protected=set(unique_selected)):
            return ExpertArenaAccess(
                layer_index=self.layer_index,
                selected_experts=selected,
                selected_slots=(),
                missing_experts=tuple(missing),
                slot_updates=(),
                evicted_experts=(),
                hit_count=len(selected) - sum(1 for expert in selected if expert in missing),
                needs_fallback=True,
            )

        slot_updates: list[tuple[int, int]] = []
        evicted: list[int] = []
        for expert in missing:
            slot, old_expert = self._claim_slot(protected=set(unique_selected))
            if old_expert is not None:
                self.expert_to_slot.pop(old_expert, None)
                self._last_used.pop(old_expert, None)
                evicted.append(old_expert)
            self.slot_to_expert[slot] = expert
            self.expert_to_slot[expert] = slot
            self._last_used[expert] = self._clock
            slot_updates.append((slot, expert))

        selected_slots = tuple(self.expert_to_slot[expert] for expert in selected)
        return ExpertArenaAccess(
            layer_index=self.layer_index,
            selected_experts=selected,
            selected_slots=selected_slots,
            missing_experts=tuple(missing),
            slot_updates=tuple(slot_updates),
            evicted_experts=tuple(evicted),
            hit_count=len(selected) - sum(1 for expert in selected if expert in missing),
        )

    def _available_slot_count(self, *, protected: set[int]) -> int:
        count = 0
        for expert in self.slot_to_expert:
            if expert is None:
                count += 1
            elif expert not in protected and expert not in self.pinned_experts:
                count += 1
        return count

    def _claim_slot(self, *, protected: set[int]) -> tuple[int, int | None]:
        for slot, expert in enumerate(self.slot_to_expert):
            if expert is None:
                return slot, None

        candidates = [
            (slot, expert)
            for slot, expert in enumerate(self.slot_to_expert)
            if expert is not None and expert not in protected and expert not in self.pinned_experts
        ]
        if not candidates:
            raise RuntimeError("no evictable slot; selection should have fallen back")
        slot, expert = min(candidates, key=lambda item: (self._last_used.get(item[1], -1), item[1]))
        return slot, expert


@dataclass
class ExpertArenaBank:
    capacity_per_layer: int
    bytes_per_slot: int
    arenas: dict[int, ExpertSlotArena] = field(default_factory=dict)
    stats: ExpertArenaStats = field(default_factory=ExpertArenaStats)

    @property
    def resident_bytes(self) -> int:
        return sum(arena.resident_bytes for arena in self.arenas.values())

    def access(self, *, layer_index: int, selected_experts: Iterable[int]) -> ExpertArenaAccess:
        arena = self.arenas.get(layer_index)
        if arena is None:
            arena = ExpertSlotArena(
                layer_index=layer_index,
                capacity=self.capacity_per_layer,
                bytes_per_slot=self.bytes_per_slot,
            )
            self.arenas[layer_index] = arena
        access = arena.plan_access(selected_experts)
        self.stats.record(access, arena_count=len(self.arenas))
        return access

    def pin(self, *, layer_index: int, experts: Iterable[int]) -> ExpertArenaAccess:
        arena = self.arenas.get(layer_index)
        if arena is None:
            arena = ExpertSlotArena(
                layer_index=layer_index,
                capacity=self.capacity_per_layer,
                bytes_per_slot=self.bytes_per_slot,
            )
            self.arenas[layer_index] = arena
        access = arena.pin(experts)
        self.stats.record(access, arena_count=len(self.arenas))
        return access
