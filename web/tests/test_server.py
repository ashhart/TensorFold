import time
import types

from server_fixtures import _client, _mock

from tensorfold_web import engine as engine_mod
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine, parse_health, rates_from_counters
from tensorfold_web.server import create_app


def test_page_never_serves_files_outside_its_folder(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        client, _ = _client(tmp_path, monkeypatch, port)
        for p in ("/../server.py", "/..%2Fserver.py", "/assets/../../server.py", "/%2e%2e/pyproject.toml"):
            r = client.get(p)
            assert "fastapi" not in r.text and "[project]" not in r.text, p
        assert client.get("/api/nothing").status_code == 404
        assert client.get("/engine-ish").status_code in (200, 404)
    finally:
        server.shutdown()

def test_cuda_counters_become_rates_and_unknown_stays_unknown():
    h = parse_health({"ok": True, "status": "ok", "requests_running": 1, "completion_tokens_total": 100,
                      "decode_seconds_total": 2.0, "prompt_tokens_total": 1000, "prefill_seconds_total": 0.5})
    assert h.ok and h.counters["completion_tokens_total"] == 100 and h.decode_tps is None
    later = {"completion_tokens_total": 190.0, "decode_seconds_total": 4.0, "prompt_tokens_total": 1000.0,
             "prefill_seconds_total": 0.5}
    assert rates_from_counters(None, later) == (None, None)
    assert rates_from_counters(h.counters, later) == (45.0, 0.0)
    assert rates_from_counters(h.counters, {"completion_tokens_total": 190.0}) == (None, None)
    minimal = parse_health({"status": "ok"})                      # the MLX server once an API key is on
    assert minimal.ok and minimal.model is None and minimal.decode_tps is None and minimal.counters is None

def test_memory_limit_variable_follows_the_backend(monkeypatch):
    import tensorfold_web.catalog as catalog_mod
    from tensorfold_web import fit
    from tensorfold_web import machine as machine_mod

    machine = types.SimpleNamespace(backend="cuda", ram_bytes=64 * fit.GIB, unified_memory=False, gpu_ceiling_bytes=None,
                                    gpus=[types.SimpleNamespace(memory_bytes=24 * fit.GIB)])
    monkeypatch.setattr(machine_mod, "detect", lambda: machine)
    monkeypatch.setattr(catalog_mod, "capacity_of", lambda m: fit.cuda_capacity(24 * fit.GIB, unified=False))
    assert engine_mod.memory_limit_env(0.5) == {"TENSORFOLD_CUDA_MEMORY_LIMIT_GB": "12.0"}

def test_a_request_cancelled_before_its_worker_runs_takes_nothing(tmp_path, monkeypatch):
    import asyncio
    import concurrent.futures

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=False)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    eng = Engine(s)
    eng.stop = lambda: None
    app = create_app(settings=s, engine=eng, catalog=cat)
    stop = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/api/engine/stop")

    async def go():
        loop = asyncio.get_running_loop()
        busy = concurrent.futures.ThreadPoolExecutor(max_workers=1)    # the only worker is taken: the stop queues
        loop.set_default_executor(busy)
        block = asyncio.Event()
        hold = loop.run_in_executor(busy, lambda: asyncio.run_coroutine_threadsafe(block.wait(), loop).result())
        task = asyncio.create_task(stop())
        await asyncio.sleep(0.1)
        task.cancel()                                                  # gone before its worker ever ran
        await asyncio.sleep(0.05)
        block.set()
        await hold
        await asyncio.sleep(0.1)
        assert app.state.gate.holder is None
    asyncio.run(go())

def test_the_served_model_is_named_as_the_catalog_names_it(tmp_path, monkeypatch):
    """The page says "Qwen3.8-27B", not the file the engine was given."""

    from tensorfold_web import mock_engine

    monkeypatch.setattr(mock_engine, "MODEL", "TensorFold/Qwen3.8-27B-MLX-4bit")
    server, port = _mock()
    try:
        client, _ = _client(tmp_path, monkeypatch, port)
        end = time.time() + 10
        while client.get("/api/engine").json()["status"] != "running" and time.time() < end:
            time.sleep(0.1)
        info = client.get("/api/engine").json()
        assert info["model"] == "TensorFold/Qwen3.8-27B-MLX-4bit" and info["model_name"] == "Qwen3.8-27B"
    finally:
        server.shutdown()

def test_one_model_is_recommended_when_one_runs_here(tmp_path, monkeypatch):
    from tensorfold_web.catalog import Catalog

    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    from tensorfold_web.machine import Machine

    machine = Machine(os="macOS", os_version="27.0", arch="arm64", chip="Apple M6", ram_bytes=32 * 1024**3,
                      ram_available_bytes=20 * 1024**3, unified_memory=True, backend="mlx")
    monkeypatch.setattr(cat, "native_here", lambda _e: False)
    rows = cat.view(machine, backend="mlx")
    picked = [m for m in rows if m.get("recommended")]
    assert len(picked) <= 1
    if picked:
        assert picked[0]["runs_here"] and picked[0]["verdict"]["level"] in ("fits", "tight")

def test_chat_preset_routes(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, 9)
    p = {"name": "Mine", "system": "", "temperature": 0.5, "top_p": 0.9}
    assert client.post("/api/chat-presets", json={"preset": p}).status_code == 200
    assert client.post("/api/chat-presets", json={"preset": p}).status_code == 409
    assert client.post("/api/chat-presets", json={"preset": {**p, "top_p": 2}, "replace": True}).status_code == 400
    assert client.put("/api/chat-presets/Mine", json={"name": "Yours"}).json()["top_p"] == 0.9
    assert [x["name"] for x in client.get("/api/chat-presets").json()["presets"]] == ["Yours"]
    assert client.delete("/api/chat-presets/Yours").status_code == 200
    assert client.delete("/api/chat-presets/Yours").status_code == 404
