import pytest
from fastapi.testclient import TestClient

from tensorfold_web import catalog, native, server_system, settings
from tensorfold_web.engine import Engine
from tensorfold_web.engines import PythonLauncher, choose
from tensorfold_web.machine import GIB, Gpu, Machine
from tensorfold_web.server import create_app

MODEL = "TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP"
DENSE = "TensorFold/Qwen3.8-27B-MLX-4bit"


@pytest.mark.parametrize(("backend", "has_python"), [
    ("mlx", True),
    ("mlx", False),
    ("cuda", False),
])
def test_qwen35b_is_cuda_only_in_the_catalog_and_start_route(tmp_path, monkeypatch, backend, has_python):
    machine = Machine("macOS" if backend == "mlx" else "Linux", "test", "arm64", "test",
                      128 * GIB, 120 * GIB, True, gpus=[Gpu("test", 128 * GIB)], backend=backend)
    monkeypatch.setattr(settings, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(server_system, "detect", lambda: machine)
    monkeypatch.setattr("tensorfold_web.machine.detect", lambda: machine)
    monkeypatch.setattr(catalog, "installed", lambda *_: False)
    monkeypatch.setattr(native, "find_binary", lambda: "/x/tensorfold")
    monkeypatch.setattr(native, "refused", lambda *_: None)
    monkeypatch.setattr(native, "capabilities", lambda _: {
        "backends": ["metal" if backend == "mlx" else "cuda"],
        "families": {"qwen3_5": ["mlx-q4g64"], "qwen3_5_moe": ["mlx-q4g64"]},
    })
    monkeypatch.setattr(PythonLauncher, "available", lambda _: has_python)
    cat = catalog.Catalog()
    cat.refresh_sizes_async = lambda: None
    monkeypatch.setattr(cat, "size_of", lambda repo: 20 * GIB if repo == MODEL else None)
    s = settings.Settings(engine_url="http://127.0.0.1:9", remote=False)
    engine = Engine(s)
    starts = []

    def start(model, *, native_model, drafter):
        starts.append((model, native_model, drafter))
        choose(native_model)  # check the same launcher availability as Engine.start
        if native_model:
            engine.exit_code = 0  # end the launch observation without a child process
        return {"status": "starting"}

    monkeypatch.setattr(engine, "start", start)
    monkeypatch.setattr(engine, "info", dict)
    client = TestClient(create_app(settings=s, engine=engine, catalog=cat))
    rows = {row["id"]: row for row in client.get("/api/catalog").json()["models"]}
    response = client.post("/api/engine/start", json={"model": MODEL})
    row = rows[MODEL]
    assert row["backends"] == ["cuda"]
    assert row["runs_here"] is (backend == "cuda")
    assert rows[DENSE]["native"] is True  # another Qwen checkpoint still uses the native engine
    if backend == "mlx":
        assert not row.get("recommended")
        assert "CUDA" in row["runs_note"] and "0.6.6" not in row["runs_note"]
        assert response.status_code == 409
        assert "CUDA" in response.json()["detail"]
        assert starts == []
    else:
        assert row["native"] is True
        assert response.status_code == 200
        assert starts == [(MODEL, True, None)]
