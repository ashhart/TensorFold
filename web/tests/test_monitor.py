import pytest

from tensorfold_web import engine as engine_mod
from tensorfold_web.engine import Engine, kv_fill, parse_health, parse_metrics
from tensorfold_web.settings import Settings

# the Mac's Python line behind a key, 10 Oct 2026: the KV fill is only on labelled series
METRICS = """# HELP tensorfold:kv_cache_usage_ratio share of the pool held
tensorfold:requests_total{key="env-1",status="200"} 238
tensorfold:num_requests_running 1
tensorfold:kv_cache_usage_ratio{pool="0"} 0.25
tensorfold:kv_cache_usage_ratio{pool="1"} 0.5
tensorfold:kv_cache_usage_perc{stream="0"} 0.5
tensorfold:preemptions_total 3
tensorfold:process_footprint_bytes 20010510968
"""


def test_the_kv_fill_is_read_from_labelled_series():
    m = parse_metrics(METRICS)
    assert m["kv_cache_usage_ratio"] == 0.5 and "requests_total" not in m and m["preemptions_total"] == 3
    assert kv_fill({}, m) == 0.5
    assert kv_fill({}, {"gpu_cache_usage_perc": 42.0}) == pytest.approx(0.42)   # a percentage read as one
    assert kv_fill({}, {}) is None


def test_the_cuda_servers_pool_and_streams_from_health():
    h = parse_health({"status": "ok", "model": "m", "pool_tokens": 200000, "pool_free_tokens": 150000,
                      "kept_prompts": 4, "streams": {"decoding": 2, "filling": 1, "paused": 0}})
    assert h.kv_fill == 0.25 and h.kept_prompts == 4
    assert h.streams == {"decoding": 2, "prefilling": 1, "paused": 0}
    bare = parse_health({"status": "ok"})
    assert bare.kv_fill is None and bare.streams is None


def test_metrics_behind_a_key_give_the_fill_and_the_preemptions():
    h = engine_mod.Health(ok=True, status="running")
    Engine._read_metrics(h, parse_metrics(METRICS))
    assert h.kv_fill == 0.5 and h.preemptions == 3 and h.memory == {"footprint": 20010510968}


def test_memory_by_holder_and_headroom_in_gb(monkeypatch):
    gib = 1024**3

    class VM:
        total, available = 32 * gib, 3 * gib

    monkeypatch.setattr(engine_mod.psutil, "virtual_memory", lambda: VM)
    e = Engine(Settings())
    m = e.machine_memory({"footprint": 20 * gib})
    assert m == {"total": 32 * gib, "available": 3 * gib, "engine": 20 * gib, "headroom": "tight"}
    VM.available = 1 * gib
    assert e.machine_memory(None)["headroom"] == "low" and e.machine_memory(None)["engine"] is None
    VM.total, VM.available = 128 * gib, 10 * gib            # 10 GB free of 128 is under a tenth: tight there
    assert e.machine_memory(None)["headroom"] == "tight"
