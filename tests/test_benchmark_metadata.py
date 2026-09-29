"""Benchmark evidence must stay offline and exclude personal identifiers and local paths."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from tensorfold import hub
from tensorfold.benchmark import hardware, metadata

REVISION = "0123456789abcdef0123456789abcdef01234567"
_PACKAGE_PROBE = metadata._package_version
_CUDA_PROBE = metadata._cuda_version


@pytest.fixture(autouse=True)
def no_runtime_probes(monkeypatch):
    monkeypatch.setattr(metadata, "_package_version", lambda name: None)
    monkeypatch.setattr(metadata, "_cuda_version", lambda: None)
    monkeypatch.setattr(hardware, "driver_version", lambda: None)


def checkpoint(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64},
                                               "_name_or_path": "/home/example/private-checkpoint"}))
    (path / "tokenizer.json").write_text('{"version":"1.0"}')
    return path


def test_local_checkpoint_never_exposes_name_or_path(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / "private-checkpoint")
    monkeypatch.setattr(hub, "cached", lambda repo: None)
    result = metadata.collect(str(path), "mlx")
    model = result["model"]
    assert model["repo_id"] is None and model["revision"] is None and model["local"] is True
    assert model["family"] == "qwen3_5"
    assert model["config_sha256"] == hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
    assert model["tokenizer_sha256"] is not None
    assert model["quantization"] == "mlx-4bit-g64"
    encoded = json.dumps(result)
    assert str(tmp_path) not in encoded and "private-checkpoint" not in encoded and "_name_or_path" not in encoded
    assert set(result["runtime"]["dependencies"]) == {"mlx", "mlx_lm", "torch", "triton", "cuda", "driver"}
    assert result["runtime"]["platform"] in ("macos", "linux")


def test_repo_revision_comes_from_cached_snapshot_not_config(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / "snapshots" / REVISION)
    monkeypatch.setattr(hub, "cached", lambda repo: path if repo == "example/model" else None)
    monkeypatch.setattr(hub, "pull", lambda *args, **kwargs: pytest.fail("must not download"))
    result = metadata.collect("example/model", "cuda")
    assert result["model"]["repo_id"] == "example/model"
    assert result["model"]["revision"] == REVISION and result["model"]["local"] is False


def test_missing_evidence_is_null_and_stays_offline(monkeypatch):
    monkeypatch.setattr(hub, "cached", lambda repo: None)
    monkeypatch.setattr(hub, "pull", lambda *args, **kwargs: pytest.fail("must not download"))
    model = metadata.collect("example/missing", "mlx")["model"]
    assert model == {"repo_id": "example/missing", "revision": None, "family": None,
                     "config_sha256": None, "tokenizer_sha256": None, "quantization": None,
                     "drafter_repo_id": None, "local": False}


def test_family_and_quantization_do_not_copy_arbitrary_config_strings(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / "checkpoint")
    (path / "config.json").write_text(json.dumps({"model_type": "private-family", "quantization_config": {
        "quant_method": "private-format", "quant_algo": "/home/example/private", "bits": "secret"}}))
    model = metadata.collect(str(path), "mlx")["model"]
    assert model["family"] is None and model["quantization"] is None
    assert "private" not in json.dumps(model)


def test_auto_drafter_only_reports_complete_cached_repo(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / "target")
    draft = checkpoint(tmp_path / "draft")
    monkeypatch.setattr(hub, "cached", lambda repo: draft)
    assert metadata.collect(str(path), "mlx")["model"]["drafter_repo_id"] is None
    (draft / "model.safetensors").write_bytes(b"test-weights")
    assert metadata.collect(str(path), "mlx")["model"]["drafter_repo_id"] == "z-lab/Qwen3.8-27B-DFlash2"
    assert metadata.collect(str(path), "mlx", drafter="none")["model"]["drafter_repo_id"] is None
    assert metadata.collect(str(path), "mlx", drafter=str(draft))["model"]["drafter_repo_id"] is None


def test_tokenizer_hash_covers_template_changes_without_weight_hashing(tmp_path):
    path = checkpoint(tmp_path / "target")
    before = metadata._tokenizer_hash(path)
    (path / "model.safetensors").write_bytes(b"huge weights would not be read")
    assert metadata._tokenizer_hash(path) == before
    (path / "chat_template.jinja").write_text("{{ messages }}")
    assert metadata._tokenizer_hash(path) != before
    assert metadata._file_hash(path / "missing.json") is None


def test_metadata_file_limits_are_nullable(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / "target")
    monkeypatch.setattr(metadata, "_TOKENIZER_LIMIT", 1)
    assert metadata._tokenizer_hash(path) is None
    (path / "config.json").write_bytes(b" " * (4 * 1024**2 + 1))
    assert metadata._config(path) == {} and metadata._file_hash(path / "config.json") is None


def test_runtime_versions_read_files_without_importing_gpu_modules(tmp_path, monkeypatch):
    version_file = tmp_path / "version.py"
    version_file.write_text("cuda: str = '12.8'\n__version__ = '2.8.0'\n")
    monkeypatch.setattr(metadata.packages, "distribution", lambda name: SimpleNamespace(locate_file=lambda file: version_file))
    monkeypatch.setattr(metadata.packages, "version", lambda name: "2.8.0+cu128")
    assert _CUDA_PROBE() == "12.8"
    assert _PACKAGE_PROBE("torch") == "2.8.0+cu128"
    monkeypatch.setattr(metadata.packages, "version", lambda name: "/home/example/private-version")
    assert _PACKAGE_PROBE("torch") is None


def test_mac_hardware_selects_only_chip_memory_and_cores(monkeypatch):
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    calls = []

    def run(args):
        calls.append(args)
        if args == ["sysctl", "-n", "machdep.cpu.brand_string"]:
            return "Apple M5 Max\n"
        if args == ["sysctl", "-n", "hw.memsize"]:
            return str(128 * 1024**3)
        return json.dumps({"SPDisplaysDataType": [{"sppci_model": "Apple M5 Max", "sppci_cores": "40",
                                                 "serial_number": "private-device-value",
                                                 "_name": "private-computer", "sppci_device_id": "private-id"}]})

    monkeypatch.setattr(hardware, "_run", run)
    result = hardware.detect("mlx")
    assert result == {"cpu_model": "Apple M5 Max", "system_memory_bytes": 128 * 1024**3,
                      "memory_type": "unified", "gpus": [{"name": "Apple M5 Max", "memory_bytes": 128 * 1024**3,
                                                           "cores": 40}], "gpu_count": 1, "source": "detected"}
    assert "private" not in json.dumps(result)
    assert calls[-1] == ["system_profiler", "SPDisplaysDataType", "-json"]


def test_cuda_hardware_queries_no_identifiers(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(hardware, "_cpu_model", lambda: "AMD EPYC 9004")
    monkeypatch.setattr(hardware, "_system_memory", lambda: 256 * 1024**3)
    calls = []
    monkeypatch.setattr(hardware, "_run", lambda args: calls.append(args) or "NVIDIA RTX 4090, 24564, 580.65.06\nNVIDIA GB10, N/A, 580.65.06\n")
    result = hardware.detect("cuda")
    assert result["gpu_count"] == 2 and result["memory_type"] == "dedicated"
    assert result["gpus"][0]["memory_bytes"] == 24564 * 1024**2
    assert result["gpus"][1]["memory_bytes"] is None and result["source"] == "partial"
    assert calls == [["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]]


def test_visible_cuda_devices_and_unified_memory(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(hardware, "_cpu_model", lambda: None)
    monkeypatch.setattr(hardware, "_system_memory", lambda: 128 * 1024**3)
    monkeypatch.setattr(hardware, "_run", lambda args: "NVIDIA RTX 4090, 24564, 580.65.06\nNVIDIA GB10, N/A, 580.65.06\n")
    assert hardware.detect("cuda")["memory_type"] == "unified"
    assert hardware.detect("cuda")["gpu_count"] == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-private-selector")
    assert hardware.detect("cuda")["gpus"] == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert hardware.detect("cuda")["gpus"] == []


def test_unavailable_hardware_and_unsafe_names_stay_null(monkeypatch):
    monkeypatch.setattr(hardware, "_cpu_model", lambda: None)
    monkeypatch.setattr(hardware, "_system_memory", lambda: None)
    monkeypatch.setattr(hardware, "_run", lambda args: "/home/example/private-device, 1234, 123\n")
    assert hardware.detect("cuda") == {"cpu_model": None, "system_memory_bytes": None, "memory_type": "unknown",
                                       "gpus": [], "gpu_count": None, "source": "unavailable"}
    for value in ("private-machine", "/home/example/private", "NVIDIA /home/example/private", "NVIDIA " + "a" * 121):
        assert hardware._model_name(value) is None
    assert hardware._model_name("Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz") == "Intel(R) Xeon(R) CPU E5-2680 v4"


def test_malformed_profiler_and_smi_output_are_nullable(monkeypatch):
    monkeypatch.setattr(hardware, "_run", lambda args: "{bad json")
    assert hardware._mac_gpus() == [] and hardware._nvidia_rows() == []
    monkeypatch.setattr(hardware, "_run", lambda args: json.dumps({"SPDisplaysDataType": "not-a-list"}))
    assert hardware._mac_gpus() == []


def test_deeply_nested_config_does_not_raise_a_traceback(tmp_path):
    path = checkpoint(tmp_path / "target")
    (path / "config.json").write_text("[" * 2000 + "]" * 2000)
    assert metadata._config(path) == {}
    assert metadata.collect(str(path), "mlx")["model"]["family"] is None
