"""Prompt memory limits without MLX buffers or model weights."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tensorfold import cli, families, hub
from tensorfold.server.memory_budget import (
    PROCESS_BYTES,
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
    assert ("memory", 12 * GIB - PROCESS_BYTES) in calls
    assert calls.index(("memory", 12 * GIB - PROCESS_BYTES)) < calls.index(("load", None))
    assert not any(kind in ("wired", "sync") for kind, _ in calls)


def test_default_budget_respects_physical_and_recommended_memory():
    mx = SimpleNamespace(device_info=lambda: {"max_recommended_working_set_size": 80 * GIB})
    assert memory_limit_bytes(mx, environ={}, physical_bytes=48 * GIB) == int(0.70 * 48 * GIB)
    mx.device_info = lambda: {"max_recommended_working_set_size": 24 * GIB}
    assert memory_limit_bytes(mx, environ={}, physical_bytes=48 * GIB) == 24 * GIB
    mx = SimpleNamespace(metal=SimpleNamespace(device_info=mx.device_info))
    assert memory_limit_bytes(mx, environ={}, physical_bytes=48 * GIB) == 24 * GIB


def test_mlx_buffers_are_capped_at_the_budget_less_the_rest_of_the_process():
    calls = []
    mx = SimpleNamespace(set_memory_limit=lambda n: calls.append(("memory", n)),
                         set_cache_limit=lambda n: calls.append(("cache", n)))
    budget = configure_mlx(mx, 8 * GIB, reserve_bytes=GIB, environ={"TENSORFOLD_MEMORY_LIMIT_GB": "2.5"},
                           physical_bytes=128 * GIB)
    assert budget == int(2.5 * GIB)
    assert calls == [("memory", budget - GIB), ("cache", budget - GIB)]
    calls.clear()
    assert configure_mlx(mx, GIB // 2, environ={}, physical_bytes=48 * GIB) == int(0.70 * 48 * GIB)
    assert calls == [("memory", int(0.70 * 48 * GIB) - PROCESS_BYTES), ("cache", GIB // 2)]
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
    assert profile == CacheMemory(2048 * each, 2 * each, 2048, each)
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


def _serve_to_app(monkeypatch, tmp_path, argv, capsys):
    """Run `tensorfold serve` up to its HTTP loop with a stub app; returns the app's keyword arguments and output."""

    core = ModuleType("mlx.core")
    core.set_cache_limit = lambda value: None
    core.set_memory_limit = lambda value: None
    core.device_info = lambda: {"max_recommended_working_set_size": 64 * GIB, "memory_size": 128 * GIB}
    core.__version__ = "0.0"
    core.synchronize = core.clear_cache = lambda: None               # the weights' wiring after load
    core.get_active_memory = lambda: 0
    core.set_wired_limit = lambda value: 0
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr("faulthandler.register", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_config_dir", lambda model: tmp_path)
    monkeypatch.setattr(cli, "_model_context", lambda path: 262144)
    monkeypatch.setattr(cli, "_backend", lambda *args: "mlx")
    monkeypatch.setattr(cli, "_note_untested", lambda *args: None)
    monkeypatch.setattr(cli, "_drafter", lambda *args: "")
    monkeypatch.setattr(families, "require_readable", lambda *args: None)
    monkeypatch.setattr(families, "read_config", lambda *args: {})
    monkeypatch.setattr(families, "kernel_version", lambda *args: "k")
    monkeypatch.setattr(hub, "is_repo_id", lambda model: False)
    monkeypatch.setattr(hub, "resolve", lambda *args, **kwargs: tmp_path)
    family = SimpleNamespace(title="fixture", model_type="fixture",
                             package=SimpleNamespace(load=lambda *args, **kwargs: (object(), object())))
    monkeypatch.setattr(families, "detect", lambda path: family)
    made = {}

    class App:
        def __init__(self, model, tokenizer, **kwargs):
            made.update(kwargs)
            self.context_window = 61440 if kwargs["fit_context"] else kwargs["context_window"]
            self.context_fitted = kwargs["fit_context"]

        def close(self):
            pass

    class Server:
        def __init__(self, *args):
            pass

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    import tensorfold.server.app as app_module
    import tensorfold.server.http as http_module

    monkeypatch.setattr(app_module, "ChatApp", App)
    monkeypatch.setattr(http_module, "Server", Server)
    monkeypatch.setattr(http_module, "make_handler", lambda app: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-update-check", *argv])
    assert cli.cmd_serve(args) == 0
    return made, capsys.readouterr().out


def test_an_omitted_context_is_fitted_to_memory_and_the_banner_shows_the_window(monkeypatch, tmp_path, capsys):
    made, out = _serve_to_app(monkeypatch, tmp_path, [], capsys)
    assert made["fit_context"] is True and made["context_window"] == 262144
    assert "context: 61440" in out
    assert "context window 61,440 tokens: the most one request can use in the 64.0 GiB memory budget" in out
    assert "model's window is 262,144" in out
    made, out = _serve_to_app(monkeypatch, tmp_path, ["--context", "32768"], capsys)
    assert made["fit_context"] is False and made["context_window"] == 32768
    assert "context: 32768" in out


def test_weights_past_the_budget_are_refused_before_loading(monkeypatch, tmp_path):
    calls = []
    core = ModuleType("mlx.core")
    core.set_cache_limit = lambda value: calls.append(("cache", value))
    core.set_memory_limit = lambda value: calls.append(("memory", value))
    core.device_info = lambda: {"max_recommended_working_set_size": 64 * GIB, "memory_size": 128 * GIB}
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setenv("TENSORFOLD_MEMORY_LIMIT_GB", "1")
    monkeypatch.setattr("faulthandler.register", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_config_dir", lambda model: tmp_path)
    monkeypatch.setattr(cli, "_model_context", lambda path: 262144)
    monkeypatch.setattr(cli, "_backend", lambda *args: "mlx")
    monkeypatch.setattr(cli, "_note_untested", lambda *args: None)
    monkeypatch.setattr(families, "require_readable", lambda *args: None)
    monkeypatch.setattr(families, "read_config", lambda *args: {})
    monkeypatch.setattr(hub, "is_repo_id", lambda model: False)
    monkeypatch.setattr(hub, "resolve", lambda *args, **kwargs: tmp_path)
    with open(tmp_path / "model-00001-of-00001.safetensors", "wb") as handle:
        handle.truncate(GIB + 1)              # sparse: 1 GiB of weights on disk, no blocks written
    family = SimpleNamespace(title="fixture", model_type="fixture",
                             package=SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("weights loaded")))
    monkeypatch.setattr(families, "detect", lambda path: family)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-update-check"])
    with pytest.raises(ValueError, match=r"weights \(1\.0 GiB\) do not fit.*1\.0 GiB memory budget"):
        cli.cmd_serve(args)



def test_the_27b_draft_slot_counts_its_drafter_context_as_fixed_memory_and_a_copy_as_none():
    import copy

    from tensorfold.families.qwen3_5.dflash_head import DraftSlot

    keys = Array((1, 4, 512, 256))
    kv = SimpleNamespace(keys=keys, values=Array(keys.shape), state=(keys, keys), offset=512)
    slot = DraftSlot(drafter=None)
    assert CacheMemory.from_cache([kv, slot]).fixed_bytes == 0
    taps = Array((1, 2047, 25600))
    drafter_keys = Array((1, 8, 2048, 128))
    slot.proposer = SimpleNamespace(context=taps, cache=[SimpleNamespace(keys=drafter_keys, values=drafter_keys,
                                                                         offset=2047)] * 5)
    profile = CacheMemory.from_cache([kv, slot])
    assert profile.fixed_bytes == taps.nbytes + 5 * 2 * drafter_keys.nbytes
    assert profile.bytes_per_token == 4 * 256 * 2 * 2
    assert copy.copy(slot).nbytes == 0


def test_growth_is_reserved_for_entries_in_flight_not_a_second_copy_of_every_layer():
    layers = [SimpleNamespace(keys=Array((1, 4, 512, 256)), values=Array((1, 4, 512, 256)), offset=512)
              for _ in range(16)]
    recurrent = SimpleNamespace(state=[Array((1, 48, 128, 128), element_bytes=4)])
    profile = CacheMemory.from_cache([*layers, recurrent])
    entry = 4 * 256 * 2 * 2
    assert profile.bytes_per_token == 16 * entry and profile.entry_bytes_per_token == entry
    fixed = recurrent.state[0].nbytes
    assert profile.growth_bytes(1000) == fixed + 1024 * 2 * entry        # two entries' old buffers in flight
    single = CacheMemory.from_cache([layers[0], recurrent])
    assert single.growth_bytes(1000) == fixed + 1024 * entry            # never more than the whole cache
    assert CacheMemory(10, 8, 256).growth_bytes(256) == 10 + 256 * 8    # a size reported without entries: all of it


def test_a_family_allowance_sets_the_budget():
    from tensorfold.server.memory_budget import model_fraction

    mx = SimpleNamespace(device_info=lambda: {"max_recommended_working_set_size": 240 * GIB})
    ram = 256 * GIB
    assert memory_limit_bytes(mx, environ={}, physical_bytes=ram) == int(0.70 * ram)
    assert memory_limit_bytes(mx, fraction=0.85, environ={}, physical_bytes=ram) == int(0.85 * ram)
    assert model_fraction(SimpleNamespace(), ram) == 0.70
    assert model_fraction(SimpleNamespace(memory_fraction=lambda ram: 0.85 if ram <= 256 * GIB else None), ram) == 0.85
    assert model_fraction(SimpleNamespace(memory_fraction=lambda ram: 0.85 if ram <= 256 * GIB else None),
                          512 * GIB) == 0.70


def test_serve_applies_the_family_allowance_before_loading(monkeypatch, tmp_path):
    from tensorfold.server import memory_budget

    calls = []
    core = ModuleType("mlx.core")
    core.set_cache_limit = lambda value: calls.append(("cache", value))
    core.set_memory_limit = lambda value: calls.append(("memory", value))
    core.device_info = lambda: {"max_recommended_working_set_size": 240 * GIB}
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.delenv("TENSORFOLD_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.setattr("faulthandler.register", lambda *args, **kwargs: None)
    monkeypatch.setattr(memory_budget, "physical_memory_bytes", lambda: 256 * GIB)
    for name, value in (("_config_dir", lambda model: tmp_path), ("_model_context", lambda path: 262144),
                        ("_backend", lambda *args: "mlx"), ("_note_untested", lambda *args: None),
                        ("_drafter", lambda *args: "")):
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(families, "require_readable", lambda *args: None)
    monkeypatch.setattr(families, "read_config", lambda *args: {})
    monkeypatch.setattr(hub, "is_repo_id", lambda model: False)
    monkeypatch.setattr(hub, "resolve", lambda *args, **kwargs: tmp_path)

    class LoadingReached(Exception):
        pass

    def load(*args, **kwargs):
        raise LoadingReached

    package = SimpleNamespace(load=load, memory_fraction=lambda ram: 0.85)
    monkeypatch.setattr(families, "detect", lambda path: SimpleNamespace(title="f", model_type="f", package=package))
    with pytest.raises(LoadingReached):
        cli.cmd_serve(cli.build_parser().parse_args(["serve", str(tmp_path), "--no-update-check"]))
    assert ("memory", int(0.85 * 256 * GIB) - memory_budget.PROCESS_BYTES) in calls      # MLX gets the rest
