"""Discrete CUDA startup needs host loading buffers, while unified GPUs keep their shared budget."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import capacity

GIB = capacity.GIB


@pytest.fixture
def startup(monkeypatch):
    from tensorfold.cuda import build

    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    memory = {"MemTotal": 16 * GIB, "MemAvailable": 8 * GIB}
    monkeypatch.setattr(capacity, "_meminfo", lambda: memory)
    monkeypatch.setattr(capacity, "unified", lambda torch: False)
    monkeypatch.setattr(build, "refuse_old_gpu", lambda *a: None)
    monkeypatch.setattr(capacity, "floor", lambda *a: (9, 0))
    monkeypatch.setattr(capacity, "config", lambda *a: {"max_position_embeddings": 4096})
    weights = {"target": capacity.Weights(40 * GIB, 3 * GIB)}
    monkeypatch.setattr(capacity, "estimate_weights", lambda path, *a, **kw: weights[str(path)])
    torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (80 * GIB, 80 * GIB)))

    def admit(**kwargs):
        return capacity.admit("target", None, False, torch, capacity.Geometry(lambda slots: slots * 32, 8),
                              lambda *a: None, **kwargs)

    return memory, weights, torch, admit


def test_discrete_weights_do_not_have_to_fit_host(startup):
    _, _, _, admit = startup
    receipt = admit()
    assert receipt["context_window"] == 4096
    assert receipt["budget_bytes"] == 72 * GIB
    assert receipt["weight_bytes_estimate"] == 40 * GIB


def test_insufficient_host_staging_refuses_even_when_gpu_fits(startup):
    memory, _, _, admit = startup
    memory["MemAvailable"] = 3 * GIB
    with pytest.raises(ValueError, match="host staging needs.*3.00 GiB.*1.00 GiB"):
        admit()


@pytest.mark.parametrize("override,room", [(None, 6), ("2", 6), ("6", 2)])
def test_host_staging_reserve(startup, monkeypatch, override, room):
    if override is not None:
        monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", override)
    assert capacity.host_stream_bytes() == room * GIB


def test_device_startup_copies_do_not_require_host_copies(startup):
    memory, weights, _, admit = startup
    memory["MemAvailable"] = 6 * GIB
    weights["target"] = capacity.Weights(12 * GIB, GIB)
    receipt = admit(startup_copies=1)
    assert receipt["loading_bytes_estimate"] == 13 * GIB
    assert receipt["context_window"] == 4096


def test_drafter_does_not_hide_targets_host_loading_peak(startup):
    memory, weights, _, admit = startup
    memory["MemAvailable"] = 5 * GIB
    weights["target"] = capacity.Weights(10 * GIB, 4 * GIB)
    weights["draft"] = capacity.Weights(4 * GIB, GIB)
    with pytest.raises(ValueError, match="host staging needs.*4.00 GiB.*3.00 GiB"):
        admit(draft_dir=Path("draft"))


def test_custom_drafter_loading_peak_is_checked(startup):
    _, _, _, admit = startup
    with pytest.raises(ValueError, match="host staging needs.*7.00 GiB.*6.00 GiB"):
        admit(draft_dir=Path("draft"), draft_weights=lambda path: capacity.Weights(GIB, 7 * GIB))


def test_extra_file_loading_peak_is_checked(startup, monkeypatch):
    _, weights, _, admit = startup
    monkeypatch.setattr(capacity, "estimate_weights", lambda path, *a, **kw:
                        capacity.Weights(GIB, 7 * GIB) if kw.get("files") else weights["target"])
    with pytest.raises(ValueError, match="host staging needs.*7.00 GiB.*6.00 GiB"):
        admit(extra_files=(Path("head.safetensors"),))


@pytest.mark.parametrize("override", [None, "2", "6"])
@pytest.mark.parametrize("host_free", [5 * GIB, 110 * GIB + 3])
def test_unified_admission_keeps_release_budget_and_receipt(startup, monkeypatch, override, host_free):
    memory, weights, torch, admit = startup
    memory.update(MemTotal=121 * GIB + 7, MemAvailable=host_free)
    weights["target"] = capacity.Weights(GIB, GIB, 2 * GIB)
    monkeypatch.setattr(capacity, "unified", lambda torch: True)
    if override is not None:
        monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", override)
    reserve = int(override) * GIB if override else max(4 * GIB, memory["MemTotal"] // 10)
    budget = max(0, host_free - reserve)
    assert capacity.available_bytes(torch) == budget
    plan = capacity.make_plan(4096, None, False, budget, weights["target"],
                              capacity.Geometry(lambda slots: slots * 32, 8), room=host_free)
    if plan.fitting:
        expected = {**plan.receipt(capacity.choose(plan)), "largest_window": plan.largest}
        assert admit() == expected
    else:
        with pytest.raises(ValueError, match="0 tokens"):
            admit()


def test_missing_meminfo_keeps_gpu_only_fallback(startup, monkeypatch):
    _, _, _, admit = startup
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    assert capacity.host_stream_bytes() is None
    assert admit()["budget_bytes"] == 72 * GIB


def test_host_staging_failure_is_agreed_by_both_ranks(startup):
    memory, _, _, admit = startup
    memory["MemAvailable"] = 3 * GIB
    status = []

    def gather(row):
        status.append(row)
        return [row, [0, 4096, -1, 0, 4096, 4096]]

    with pytest.raises(ValueError, match="every rank.*host staging"):
        admit(world=2, gather=gather)
    memory["MemAvailable"] = 8 * GIB
    with pytest.raises(ValueError, match="every rank.*another rank"):
        admit(world=2, gather=lambda row: [status[0], row])
