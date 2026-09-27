"""Prompt memory limits without MLX buffers or model weights."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tensorfold import cli, families, hub
from tensorfold.server.memory_budget import (
    CacheMemory,
    cache_nbytes,
    configure_mlx,
    fits,
    largest_context,
    memory_limit_bytes,
    needed_bytes,
)

GIB = 1024**3


@pytest.mark.parametrize("metal", [False, True])
def test_serve_limits_memory_before_loading_without_changing_residency(monkeypatch, tmp_path, metal):
    calls = []
    core = ModuleType("mlx.core")
    core.set_cache_limit = lambda value: calls.append(("cache", value))
    core.set_memory_limit = lambda value: calls.append(("memory", value))
    core.device_info = lambda: {"max_recommended_working_set_size": 64 * GIB, "memory_size": 128 * GIB}
    core.metal = SimpleNamespace(is_available=lambda: metal)
    core.set_wired_limit = lambda value: calls.append(("wired", value)) or 7 * GIB
    core.synchronize = lambda: calls.append(("sync", None))
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setenv("TENSORFOLD_MEMORY_LIMIT_GB", "12")
    monkeypatch.setattr("faulthandler.register", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_config_dir", lambda model: tmp_path)
    monkeypatch.setattr(cli, "_model_context", lambda path: 262144)
    monkeypatch.setattr(cli, "_backend", lambda *args: "mlx")
    monkeypatch.setattr(cli, "_note_untested", lambda *args: None)
    monkeypatch.setattr(cli, "_drafter", lambda *args: "")
    monkeypatch.setattr(families, "require_readable", lambda *args: None)
    monkeypatch.setattr(families, "read_config", lambda *args: {})
    monkeypatch.setattr(hub, "is_repo_id", lambda model: False)
    monkeypatch.setattr(hub, "resolve", lambda *args, **kwargs: tmp_path)

    class LoadingReached(Exception):
        pass

    def load(*args, **kwargs):
        calls.append(("load", None))
        raise LoadingReached

    family = SimpleNamespace(title="fixture", model_type="fixture", package=SimpleNamespace(load=load))
    monkeypatch.setattr(families, "detect", lambda path: family)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-update-check"])
    with pytest.raises(LoadingReached):
        cli.cmd_serve(args)
    assert ("memory", 12 * GIB) in calls
    assert calls.index(("memory", 12 * GIB)) < calls.index(("load", None))
    assert not any(kind in ("wired", "sync") for kind, _ in calls)


def test_default_budget_respects_physical_and_recommended_memory():
    mx = SimpleNamespace(device_info=lambda: {"max_recommended_working_set_size": 80 * GIB})
    assert memory_limit_bytes(mx, environ={}, physical_bytes=48 * GIB) == int(0.70 * 48 * GIB)
    mx.device_info = lambda: {"max_recommended_working_set_size": 24 * GIB}
    assert memory_limit_bytes(mx, environ={}, physical_bytes=48 * GIB) == 24 * GIB
    mx = SimpleNamespace(metal=SimpleNamespace(device_info=mx.device_info))
    assert memory_limit_bytes(mx, environ={}, physical_bytes=48 * GIB) == 24 * GIB


def test_emulated_class_and_freed_cache_are_capped_by_the_budget():
    calls = []
    mx = SimpleNamespace(set_memory_limit=lambda n: calls.append(("memory", n)),
                         set_cache_limit=lambda n: calls.append(("cache", n)))
    budget = configure_mlx(mx, 8 * GIB, environ={"TENSORFOLD_MEMORY_LIMIT_GB": "2.5"},
                           physical_bytes=128 * GIB)
    assert budget == int(2.5 * GIB)
    assert calls == [("memory", budget), ("cache", budget)]
    budget = memory_limit_bytes(mx, environ={"TENSORFOLD_MEMORY_LIMIT_GB": "1000"}, physical_bytes=48 * GIB)
    assert budget == int(0.70 * 48 * GIB)


@pytest.mark.parametrize("value", ["", "0", "-1", "nan", "inf", "12GB"])
def test_invalid_env_limit_is_refused(value):
    with pytest.raises(ValueError, match="TENSORFOLD_MEMORY_LIMIT_GB"):
        memory_limit_bytes(SimpleNamespace(), environ={"TENSORFOLD_MEMORY_LIMIT_GB": value},
                           physical_bytes=48 * GIB)


class Array:
    def __init__(self, shape, element_bytes=2):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        size = element_bytes
        for axis in shape:
            size *= axis
        self.nbytes = size


def test_cache_profile_prices_kv_and_fixed_recurrent_state_from_the_arrays():
    keys = Array((1, 4, 512, 256))
    values = Array(keys.shape)
    recurrent = Array((1, 48, 128, 128), element_bytes=4)
    cache = [SimpleNamespace(keys=keys, values=values, state=(keys, values)),
             SimpleNamespace(state=[recurrent])]
    profile = CacheMemory.from_cache(cache)
    assert profile.bytes_per_token == 4 * 256 * 2 * 2
    assert profile.fixed_bytes == recurrent.nbytes
    assert profile.cache_bytes(513) == recurrent.nbytes + 768 * profile.bytes_per_token


def test_bounded_cache_and_alternating_spare_are_reserved_without_existing_spare_arrays():
    keys = Array((1, 2, 256, 128))
    values = Array(keys.shape)
    cache = [SimpleNamespace(keys=keys, values=values, nbytes=keys.nbytes + values.nbytes,
                             spare_keys=None, spare_values=None, grow=2048),
             SimpleNamespace(keys=keys, values=values, state=(keys, values), max_size=2048)]
    profile = CacheMemory.from_cache(cache)
    each = 2 * 128 * 2 * 2
    assert profile == CacheMemory(2048 * each, 2 * each, 2048)
    assert profile.cache_bytes(2049) == 2048 * each + 4096 * 2 * each


def test_an_unpopulated_kv_cannot_claim_zero_memory():
    with pytest.raises(ValueError, match="populated probe"):
        CacheMemory.from_cache([SimpleNamespace(keys=None, values=None, state=[])])


def test_attention_index_arrays_grow_with_context_and_state_views_do_not_hide_capacity():
    keys = Array((1, 2, 2048, 128))
    values = Array(keys.shape)
    index_keys = Array((1, 2048, 64))
    pooled = Array((1, 512, 64))
    class IndexedCache:
        def __init__(self):
            self.keys, self.values = keys, values
            self.index_keys, self.pooled = index_keys, pooled
            self.offset = 2048

        @property
        def state(self):
            return Array((1, 2, 64, 128)), Array((1, 2, 64, 128))

    cache = [IndexedCache()]
    total = keys.nbytes + values.nbytes + index_keys.nbytes + pooled.nbytes
    assert cache_nbytes(cache) == total
    profile = CacheMemory.from_cache(cache)
    assert profile.cache_bytes(4096) == 2 * total
    assert cache_nbytes(cache) > sum(array.nbytes for array in cache[0].state)


def test_admission_reserves_reply_work_and_checkpoint_copies():
    profile = CacheMemory(100, 2, 16)
    projected = needed_bytes(profile, 65, resident_bytes=1000, working_bytes=100,
                             cache_copies=2, reserve_tokens=32)
    assert projected == 1000 + 100 + 2 * (100 + 112 * 2)
    assert fits(profile, 65, budget_bytes=projected, resident_bytes=1000, working_bytes=100,
                cache_copies=2, reserve_tokens=32)
    assert not fits(profile, 65, budget_bytes=projected - 1, resident_bytes=1000, working_bytes=100,
                    cache_copies=2, reserve_tokens=32)


def test_refusal_names_a_context_that_fits_and_leaves_reply_room():
    profile = CacheMemory(100, 2, 16)
    options = dict(budget_bytes=1500, resident_bytes=1000, working_bytes=100, reserve_tokens=32)
    largest = largest_context(profile, 256, **options)
    assert largest == 112
    assert fits(profile, largest, **options)
    assert not fits(profile, largest + 1, **options)
    assert largest_context(profile, 96, **options) == 64
    assert largest_context(profile, 256, budget_bytes=999, resident_bytes=1000) == 0


def test_a_non_kv_entry_such_as_a_draft_slot_counts_as_fixed_memory():
    keys = Array((1, 2, 256, 128))
    values = Array(keys.shape)
    kv = SimpleNamespace(keys=keys, values=values, state=(keys, values), offset=256)
    slot = SimpleNamespace(keys=None, state=[], context=Array((1, 16, 64)))    # a drafter slot, no KV
    profile = CacheMemory.from_cache([kv, slot])
    assert profile.bytes_per_token == 2 * 128 * 2 * 2
    assert profile.fixed_bytes == Array((1, 16, 64)).nbytes
    with pytest.raises(ValueError):
        CacheMemory.from_cache([SimpleNamespace(keys=None, values=None, offset=0)])
