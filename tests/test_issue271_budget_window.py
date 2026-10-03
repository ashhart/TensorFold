"""Issue #271: on a 128 GB M5 Max the 89.6 GiB budget fitted a 48,128-token window, the 107.5 GiB one only 10,240."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tensorfold.engine.prefill_step import CONTEXT_FLOOR, choose
from tensorfold.server.memory_budget import PROCESS_BYTES, CacheMemory
from tensorfold.server.prompt_memory import PromptMemory

GIB = 1024**3
MODEL_WINDOW = 262_144         # Flash Next's own window
RESIDENT = int(77.2 * GIB)     # the report's weights kept resident, at both budgets
CACHE_BPT = 48 * 1024          # prompt-cache bytes per token (6.0 GiB a copy at 128K)
# a full chunk's workspace: 2,048 and 8,192 as the report measured them, 4,096 between (never measured)
WORK = {2048: int(2.6 * GIB), 4096: 9 * GIB, 8192: int(25.9 * GIB)}
STEPS = (8192, 4096, 2048)     # the chunk sizes qwen4_exp's engine_settings() offers with tensor units
PROBE = list(range(4096))      # stand-in token ids for the probe prompt (the server probes on source text)
BUDGETS = [80.0 + 2.5 * k for k in range(13)]  # GiB, 80 to 110


class FakeMemory:
    """The report's memory readings, answering for choose()'s probes and PromptMemory's runtime alike."""

    def __init__(self, held: int) -> None:
        self.held, self.chunk = int(held), 0
        self.probes: list[tuple[int, int, int]] = []    # (chunk, held, workspace) of every probe run

    def arm(self, chunk_tokens: int) -> None:
        """One full chunk runs until the next peak reset; it is recorded with what it holds and needs."""
        self.chunk = chunk_tokens
        self.probes.append((chunk_tokens, self.held, WORK[chunk_tokens]))

    def synchronize(self) -> None:
        pass

    def clear_cache(self) -> None:
        pass

    def eval(self, *arrays) -> None:
        pass

    def get_cache_memory(self) -> int:
        return 0

    def get_active_memory(self) -> int:
        return self.held

    def reset_peak_memory(self) -> None:
        self.chunk = 0

    def get_peak_memory(self) -> int:
        return self.held + (WORK[self.chunk] if self.chunk else 0)


class FakeCache(list):
    """What one probe left behind: no attention state left in the graph, cache bytes as measured."""

    def __init__(self) -> None:
        super().__init__([SimpleNamespace(state=None, memory_growth=lambda: (0, CACHE_BPT))])


def _factory(memory: FakeMemory):
    """Engine factory for choose(): each step's engine probes one full chunk of that step plus a 64-token tail."""

    def probe_prefix(probe, cache=None, cached_tokens=0):
        memory.arm(len(probe) - 64)
        return FakeCache()

    return lambda _step: SimpleNamespace(model=SimpleNamespace(), prefill_prefix=probe_prefix)


def _choose(budget_gib: float, monkeypatch, window: int = MODEL_WINDOW) -> tuple[int, list[tuple[int, int, int]]]:
    """choose() on the report's readings at one budget: the chunk it takes and every probe it ran."""

    memory = FakeMemory(RESIDENT)
    mlx = ModuleType("mlx")
    mlx.core = memory
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", memory)
    return choose(_factory(memory), STEPS, int(budget_gib * GIB) - PROCESS_BYTES, PROBE, window), memory.probes


def _startup(budget_gib: float, monkeypatch) -> tuple[int, int]:
    """Relive one report startup: choose() the chunk, then fit the resumable window beside it."""

    budget = int(budget_gib * GIB)
    step, _ = _choose(budget_gib, monkeypatch)
    # ChatApp's own fit: two cache copies and the window's growth beside the weights and this chunk's workspace
    prompt_memory = PromptMemory(budget, SimpleNamespace(), runtime=FakeMemory(RESIDENT),
                                 store=SimpleNamespace(nbytes=0), window_tokens=MODEL_WINDOW, chunk_rows=step)
    prompt_memory.profile = CacheMemory(0, CACHE_BPT)
    prompt_memory.observed_work = WORK[step]
    try:
        window, fitted = prompt_memory.fit_window(MODEL_WINDOW, True)
    except ValueError:                 # no room for any window: the server refuses to start
        return step, 0
    assert fitted, f"at {budget_gib} GiB memory does not lower the window: the case cannot reverse"
    return step, window


def test_a_larger_budget_never_fits_a_smaller_window(monkeypatch):
    """The report's pair: 89.6 GiB gave 48,128 tokens with 2,048-token chunks, 107.5 GiB only 10,240 with 8,192."""

    small_step, small = _startup(89.6, monkeypatch)
    large_step, large = _startup(107.5, monkeypatch)
    assert small > 0, "89.6 GiB must fit some window for the case to be the report's"
    assert large >= small, (
        f"a larger budget fits a smaller window: 89.6 GiB gives {small:,} tokens with {small_step:,}-token "
        f"chunks, 107.5 GiB only {large:,} with {large_step:,}-token chunks")


def test_the_window_never_shrinks_below_the_floor_as_the_budget_grows(monkeypatch):
    """From 80 to 110 GiB a larger budget fits no smaller window, but a larger chunk's that still keeps the floor."""

    floor = min(MODEL_WINDOW, CONTEXT_FLOOR)
    seen = [(budget, *_startup(budget, monkeypatch)) for budget in BUDGETS]
    assert all(large >= small or (large_step > small_step and large >= floor)
               for (_, small_step, small), (_, large_step, large) in zip(seen, seen[1:])), (
        "the fitted window shrank while the budget grew: "
        + ", ".join(f"{budget:g} GiB -> {step:,}-token chunks, {window:,}" for budget, step, window in seen))


# --context 32768 lowers the floor: 4,096 leaves it from 95 GiB, where an 8,192 probe would pass the budget
@pytest.mark.parametrize("window", [MODEL_WINDOW, 32_768])
def test_no_probe_passes_the_budget(monkeypatch, window):
    """From 80 to 110 GiB no probe passes the budget, and none past the smallest chunk passes MLX's share of it."""

    for budget_gib in BUDGETS:
        budget = int(budget_gib * GIB)
        for step, held, work in _choose(budget_gib, monkeypatch, window)[1]:
            limit = budget if step == min(STEPS) else budget - PROCESS_BYTES
            assert held + work <= limit, (f"at {budget_gib:g} GiB the {step:,}-token probe peaks at "
                                          f"{(held + work) / GIB:.1f} GiB, past {limit / GIB:.1f} GiB")
