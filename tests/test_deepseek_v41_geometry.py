"""DeepSeek-V4.1-Flash's admission: the geometry equals what the constructors allocate, and the pack fits its window."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from test_deepseek_v41_split import FIXTURE, RESIDENT, inventory

from tensorfold.cuda import capacity
from tensorfold.cuda import experts as grouped
from tensorfold.cuda.exl3 import experts as exl3_experts
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import MAX_ROWS, PREFILL_ROWS, buffers, geometry, split

CFG = Config.read(FIXTURE.parent)
GIB = capacity.GIB
NATIVE = 1 << 20
BUDGET = int(103.8 * GIB)       # an idle GB10's MemAvailable less the admission reserve


class Allocation:
    def __init__(self, shape, dtype, device):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device

    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


@pytest.fixture
def allocated(monkeypatch):
    """Runs ``build`` with every tensor its constructors ask for recorded instead of allocated (pinned staging too)."""

    def run(build) -> list[Allocation]:
        recorded = []

        def allocate(shape, **kw):
            recorded.append(Allocation(shape, kw.get("dtype", torch.float32), kw.get("device")))
            return recorded[-1]

        fake = SimpleNamespace(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32,
                               int32=torch.int32, uint8=torch.uint8,
                               device=lambda d: SimpleNamespace(type=str(d)), zeros=allocate, empty=allocate,
                               full=lambda shape, fill, **kw: allocate(shape, **kw),
                               cuda=SimpleNamespace(Event=object))
        with monkeypatch.context() as m:
            for module in (buffers, grouped, exl3_experts):
                m.setattr(module, "torch", fake)
            build()
        return recorded

    return run


def _on_device(recorded) -> int:
    return sum(t.nbytes() for t in recorded if getattr(t.device, "type", None) == "cuda")


@pytest.mark.parametrize("slots", [4096, 65538 + MAX_ROWS, NATIVE + MAX_ROWS])
def test_estimate_equals_the_allocations(allocated, slots):
    def build():
        buffers.State(CFG, slots, "cuda")
        buffers.Buffers(CFG, 2, MAX_ROWS, slots, device="cuda")
        buffers.Buffers(CFG, 2, PREFILL_ROWS, slots, prefill=True, device="cuda")

    recorded = allocated(build)
    assert any(t.shape == (slots, CFG.head_dim) for t in recorded)          # the ratio-1 cache: the constructors ran
    assert any(t.device is None for t in recorded)                          # pinned staging, outside the estimate
    estimate = geometry.dsv41_geometry(CFG, 2).bytes_at(slots)
    assert _on_device(recorded) + 512 * slots == estimate                   # RoPE: fp32 cos and sin, two kinds x 32


def test_bytes_grow_by_kv_rope_and_score_rows():
    """Per slot: 3,200 B of caches, 512 B of RoPE tables, a decode window's and a prompt block's fp32 scores."""

    g = geometry.dsv41_geometry(CFG, 2)
    fixed = None
    for slots in (4096, 8192, 131072, 131088, 262144, 1 << 19, NATIVE, NATIVE + 16, 3 << 20):
        rows = buffers.score_rows(slots)
        assert rows * slots * 4 <= GIB
        rest = g.bytes_at(slots) - slots * (3200 + 512 + 4 * MAX_ROWS + 4 * rows)
        fixed = rest if fixed is None else fixed
        assert rest == fixed, slots
    assert buffers.score_rows(NATIVE) == 256 and buffers.score_rows(65544) == 2048
    assert 1.3 * GIB < fixed < 1.6 * GIB         # a prompt chunk's, a decode window's and DSpark's scratch, rings


def test_minimum_slots_and_reserve():
    g = geometry.dsv41_geometry(CFG, 2)
    assert (g.reserve, g.minimum_slots) == (MAX_ROWS, 4096)
    assert g.needed(1) == g.needed(4090) == g.bytes_at(4096)
    assert g.needed(65538) == g.bytes_at(65544)


@pytest.fixture
def weights(monkeypatch) -> capacity.Weights:
    """split.weights_estimate over every header the pack holds, read from a synthetic header dict."""

    names = inventory(json.loads(FIXTURE.read_text()))
    monkeypatch.setattr(capacity, "headers", lambda model_dir, **kw: names)
    found = capacity.estimate_weights(Path("unused"), split.weights_estimate)
    assert found.resident == RESIDENT and found.mapped == 0
    return found


def test_default_context_is_the_native_window(weights):
    g = geometry.dsv41_geometry(CFG, 2)
    plan = capacity.make_plan(NATIVE, None, False, BUDGET, weights, g)
    assert plan.fitting == plan.largest == NATIVE
    assert capacity.choose(plan) == NATIVE
    receipt = plan.receipt(NATIVE)
    assert receipt["cache_slots"] == NATIVE + MAX_ROWS
    kept = geometry.kept_bytes(receipt, geometry.cache_wanted({}))
    assert kept == 3 * GIB                                                  # the window leaves the whole arena
    assert receipt["total_bytes_estimate"] + kept < 83 * GIB


def test_explicit_context_past_the_fit_is_refused_with_the_fitting_size(weights):
    g = geometry.dsv41_geometry(CFG, 2)
    budget = 79 * GIB
    plan = capacity.make_plan(NATIVE, NATIVE, True, budget, weights, g)
    assert 0 < plan.fitting < NATIVE
    assert weights.resident + g.needed(plan.fitting) <= budget
    with pytest.raises(ValueError, match=f"largest fitting prompt-plus-reply window: {plan.fitting} tokens"):
        capacity.choose(plan)
    fits = capacity.make_plan(NATIVE, plan.fitting, True, budget, weights, g)
    assert capacity.choose(fits) == plan.fitting
    assert geometry.kept_bytes(fits.receipt(plan.fitting), 3 * GIB) < GIB     # what the tight budget leaves


@pytest.mark.parametrize("value, want", [("", 3 * GIB), ("0", 0), ("1.5", 3 * GIB // 2), ("12", 12 * GIB)])
def test_cache_wanted(value, want):
    assert geometry.cache_wanted({geometry.CACHE_ENV: value}) == want


@pytest.mark.parametrize("value", ["-1", "lots", "nan", "inf"])
def test_cache_wanted_refuses_bad_values(value):
    with pytest.raises(ValueError, match=geometry.CACHE_ENV):
        geometry.cache_wanted({geometry.CACHE_ENV: value})
