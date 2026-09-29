"""The managed benchmark server must stay local and clean up only its owned child."""

from __future__ import annotations

import json
import signal
import subprocess

import pytest

from tensorfold import hub
from tensorfold.benchmark import process


class Child:
    def __init__(self, *, exited=False, stubborn=False):
        self.exited, self.stubborn = exited, stubborn
        self.terminated = self.killed = self.waited = 0

    def poll(self):
        return 1 if self.exited else None

    def terminate(self):
        self.terminated += 1
        if not self.stubborn:
            self.exited = True

    def kill(self):
        self.killed += 1
        self.exited = True

    def wait(self, timeout):
        self.waited += 1
        if not self.exited:
            raise subprocess.TimeoutExpired("tensorfold", timeout)
        return 0


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.setattr(process.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(process.ManagedServer, "_prepare", lambda self: (tmp_path / "checkpoint", "none"))
    monkeypatch.setattr(process, "_free_port", lambda: 32123)
    monkeypatch.setattr(process.ManagedServer, "_ready", lambda self: True)
    child, calls = Child(), []

    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return child

    monkeypatch.setattr(process.subprocess, "Popen", popen)
    return child, calls


def checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64}}))
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"test-weights")
    return path


def test_command_uses_loopback_no_cache_no_request_logs_and_one_rank(launch, monkeypatch):
    child, calls = launch
    monkeypatch.setenv("TENSORFOLD_REQUEST_LOG", "/tmp/private-request-log")
    monkeypatch.setenv("TENSORFOLD_BENCHMARK_TOKEN", "test-token-that-child-does-not-need")
    with process.ManagedServer("example/model", "cuda", context=2048, output_tokens=256) as server:
        assert server.base_url == "http://127.0.0.1:32123" and server.served_name == ".tensorfold-benchmark"
        args, options = calls[0]
        assert args[:4] == [process.sys.executable, "-m", "tensorfold", "serve"]
        for flag, value in (("--host", "127.0.0.1"), ("--port", "32123"), ("--tp", "1"), ("--rank", "0"),
                            ("--snapshot-dir", "none"), ("--checkpoint-slots", "0"), ("--prompt-cache-gib", "0")):
            assert args[args.index(flag) + 1] == value
        assert "--no-update-check" in args and "--master" not in args
        assert options["start_new_session"] is True
        assert "TENSORFOLD_REQUEST_LOG" not in options["env"]
        assert "TENSORFOLD_BENCHMARK_TOKEN" not in options["env"]
        assert options["env"]["HF_HUB_OFFLINE"] == "1"
        assert options["stdout"] is options["stderr"]
    assert child.terminated == 1 and child.killed == 0 and child.waited == 1


def test_serial_benchmark_passes_no_drafts(launch):
    _, calls = launch
    with process.ManagedServer("example/model", "mlx", serial=True):
        assert "--no-drafts" in calls[0][0]


def test_exception_in_benchmark_stops_owned_child_and_unlocks(launch):
    child, _ = launch
    with pytest.raises(RuntimeError, match="benchmark failed"):
        with process.ManagedServer("example/model", "mlx"):
            raise RuntimeError("benchmark failed")
    assert child.terminated == 1
    server = process.ManagedServer("example/model", "mlx")
    server._lock_benchmark()
    server.close()


def test_keyboard_interrupt_during_startup_stops_child(launch, monkeypatch):
    child, _ = launch
    monkeypatch.setattr(process.ManagedServer, "_ready", lambda self: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        process.ManagedServer("example/model", "mlx").__enter__()
    assert child.terminated == 1


def test_sigterm_cleans_child_and_restores_original_handler(launch):
    child, _ = launch
    original = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as raised:
        with process.ManagedServer("example/model", "mlx") as server:
            server._terminate_signal(signal.SIGTERM, None)
    assert raised.value.code == 128 + signal.SIGTERM
    assert child.terminated == 1 and signal.getsignal(signal.SIGTERM) == original


def test_exited_child_startup_error_contains_no_log_or_path(launch):
    child, _ = launch
    child.exited = True
    with pytest.raises(ValueError, match="exited during startup") as raised:
        process.ManagedServer("example/model", "mlx").__enter__()
    assert "/" not in str(raised.value) and child.terminated == 0 and child.waited == 1


def test_timeout_stops_child_without_leaking_private_output(launch, monkeypatch):
    child, _ = launch
    times = iter((0, 0, 2))
    monkeypatch.setattr(process.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(process.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(process.ManagedServer, "_ready", lambda self: False)
    with pytest.raises(ValueError, match="timed out"):
        process.ManagedServer("example/model", "mlx", startup_timeout=1).__enter__()
    assert child.terminated == 1


def test_stubborn_child_gets_killed_and_reaped(launch):
    child, _ = launch
    child.stubborn = True
    with process.ManagedServer("example/model", "mlx"):
        pass
    assert child.terminated == 1 and child.killed == 1 and child.waited == 2


def test_exclusive_lock_rejects_second_benchmark_without_spawning(launch):
    _, calls = launch
    first = process.ManagedServer("example/model", "mlx")
    first._lock_benchmark()
    try:
        with pytest.raises(ValueError, match="Another TensorFold benchmark"):
            process.ManagedServer("example/another-model", "mlx").__enter__()
        assert calls == []
    finally:
        first.close()


def test_missing_cached_checkpoint_never_downloads_or_spawns(tmp_path, monkeypatch):
    monkeypatch.setattr(process.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(hub, "cached", lambda model: None)
    monkeypatch.setattr(hub, "resolve", lambda *args, **kwargs: pytest.fail("must not download"))
    monkeypatch.setattr(process.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not spawn"))
    with pytest.raises(ValueError, match="not cached"):
        process.ManagedServer("example/missing", "mlx").__enter__()


def test_config_only_checkpoint_rejected_before_spawn(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    (path / "model.safetensors").unlink()
    monkeypatch.setattr(hub, "cached", lambda model: path)
    monkeypatch.setattr(hub, "resolve", lambda *args, **kwargs: pytest.fail("must not download"))
    with pytest.raises(ValueError, match="incomplete"):
        process.ManagedServer("example/model", "mlx")._prepare()


def test_cached_target_and_drafter_are_pinned_to_local_directories(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    monkeypatch.setattr(hub, "cached", lambda model: path)
    target, draft = process.ManagedServer("example/model", "mlx")._prepare()
    assert target == path.resolve() and draft == str(path.resolve())
    target, draft = process.ManagedServer("example/model", "mlx", serial=True)._prepare()
    assert draft == "none"


def test_explicit_download_is_only_model_network_path(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    seen = []
    monkeypatch.setattr(hub, "resolve", lambda model, **options: seen.append((model, options)) or path)
    target, _ = process.ManagedServer("example/model", "mlx", download=True, drafter="none")._prepare()
    assert target == path.resolve() and seen == [("example/model", {"download": True})]


def test_missing_required_mtp_head_is_refused(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    (path / "config.json").write_text('{"model_type":"nemotron_h","quantization":{"bits":4,"group_size":64}}')
    monkeypatch.setattr(hub, "cached", lambda model: path)
    with pytest.raises(ValueError, match="incomplete"):
        process.ManagedServer("Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "mlx")._prepare()


def test_two_host_cuda_family_requires_existing_server(tmp_path):
    path = checkpoint(tmp_path)
    (path / "config.json").write_text('{"model_type":"glm5_next","quantization":{"bits":4,"group_size":64}}')
    with pytest.raises(ValueError, match="use --server"):
        process.ManagedServer(str(path), "cuda")._prepare()


def test_readiness_requires_matching_model_and_finished_warmup(monkeypatch):
    server = process.ManagedServer("example/model", "mlx")
    values = {"/health": {"status": "ok", "warming": True}, "/v1/models": {"data": [{"id": server.served_name}]}}
    monkeypatch.setattr(process, "_json", lambda base, route: values[route])
    assert server._ready() is False
    values["/health"]["warming"] = False
    assert server._ready() is True
    values["/v1/models"]["data"][0]["id"] = "another-server"
    assert server._ready() is False
    values["/health"] = {"ok": True}
    values["/v1/models"]["data"][0]["id"] = server.served_name
    assert server._ready() is True


def test_spawn_error_is_sanitized_and_lock_released(launch, monkeypatch):
    monkeypatch.setattr(process.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("private-path")))
    server = process.ManagedServer("example/model", "mlx")
    with pytest.raises(ValueError, match="Python installation") as raised:
        server.__enter__()
    assert "private-path" not in str(raised.value) and server._lock is None and server._log is None


def test_malformed_nested_checkpoint_is_refused_cleanly(tmp_path):
    path = checkpoint(tmp_path)
    (path / "config.json").write_text("[" * 2000 + "]" * 2000)
    with pytest.raises(ValueError, match="supported TensorFold family"):
        process.ManagedServer(str(path), "mlx")._prepare()
    (path / "config.json").write_text("{}")
    (path / "model.safetensors.index.json").write_text("[" * 2000 + "]" * 2000)
    assert process._complete(path) is False
