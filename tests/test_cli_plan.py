"""Plan checks local weight evidence and effective budgets without model loads or network calls."""

import argparse
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tensorfold import cli, families, hub
from tensorfold.server import memory_budget

GIB = 1024**3


def args(model, *, memory_gb=None, ram=()):
    return argparse.Namespace(model=str(model), memory_gb=memory_gb, ram=list(ram))


def model_dir(tmp_path, *, size=8 * GIB):
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 262144}))
    if size is not None:
        with (tmp_path / "model.safetensors").open("wb") as stream:
            stream.truncate(size)
    return tmp_path


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv(memory_budget.LIMIT_ENV, raising=False)
    mx = ModuleType("mlx.core")
    mx.device_info = lambda: {"max_recommended_working_set_size": 120 * GIB}
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setattr(memory_budget, "physical_memory_bytes", lambda: 48 * GIB)
    package = SimpleNamespace()
    family = SimpleNamespace(title="Fixture family", package=package)
    monkeypatch.setattr(families, "detect", lambda path: family)
    monkeypatch.setattr(families, "require_readable", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_note_untested", lambda *a: None)
    monkeypatch.setattr(hub, "pull", lambda *a, **kw: pytest.fail("plan downloaded weights"))
    hf = ModuleType("huggingface_hub")
    hf.hf_hub_download = lambda *a, **kw: pytest.fail("plan fetched config")
    hf.snapshot_download = lambda *a, **kw: pytest.fail("plan fetched a snapshot")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    package.load = lambda *a, **kw: pytest.fail("plan loaded a model")
    return package, mx


def test_sparse_fixture_preserves_size_without_allocating_the_checkpoint(tmp_path):
    model = model_dir(tmp_path)
    path = model / "model.safetensors"
    assert path.stat().st_size == 8 * GIB
    assert path.stat().st_blocks * 512 < 1024 * 1024


def test_local_file_bytes_are_provenance_and_scope_is_weights_only(tmp_path, host, capsys):
    assert cli.cmd_plan(args(model_dir(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "weights 8.0 GiB (local safetensors file sizes" in out
    assert "this Mac's current budget: 33.6 GiB budget; weights within" in out
    assert "scope: checkpoint weights plus 3 GiB process reserve" in out
    assert "prompt workspace, caches and streams" in out
    assert "one prompt chunk" not in out and "fits;" not in out


def test_family_resident_estimate_uses_local_files_without_a_model(tmp_path, host, capsys):
    package, _ = host
    seen = []
    def estimate(directory, ple_on_ssd=False):
        seen.append((directory, ple_on_ssd))
        return 6 * GIB
    package.weight_bytes = estimate
    model = model_dir(tmp_path)
    assert cli.cmd_plan(args(model)) == 0
    assert seen == [(model, False)]
    assert "weights 6.0 GiB (family resident-weight estimate from local files" in capsys.readouterr().out


def test_ram_class_calls_its_own_family_allowance(tmp_path, host, monkeypatch, capsys):
    package, _ = host
    seen = []
    def fraction(ram):
        seen.append(ram)
        return 0.85 if ram <= 64 * GIB else 0.70
    package.memory_fraction = fraction
    monkeypatch.setattr(memory_budget, "physical_memory_bytes", lambda: 512 * GIB)
    assert cli.cmd_plan(args(model_dir(tmp_path), ram=[32])) == 0
    assert 32 * GIB in seen and 512 * GIB in seen
    assert "--ram 32 GiB class: 27.2 GiB budget" in capsys.readouterr().out


def test_environment_override_caps_each_ram_class_and_current_ram(tmp_path, host, monkeypatch, capsys):
    monkeypatch.setenv(memory_budget.LIMIT_ENV, "1000")
    assert cli.cmd_plan(args(model_dir(tmp_path), ram=[16])) == 0
    out = capsys.readouterr().out
    assert "this Mac's current budget: 48.0 GiB budget" in out
    assert "--ram 16 GiB class: 16.0 GiB budget" in out
    assert f"current budgets honor {memory_budget.LIMIT_ENV}=1000" in out


def test_current_environment_and_explicit_budget_are_distinct_checks(tmp_path, host, monkeypatch, capsys):
    monkeypatch.setenv(memory_budget.LIMIT_ENV, "12")
    assert cli.cmd_plan(args(model_dir(tmp_path), memory_gb=30)) == 0
    out = capsys.readouterr().out
    assert "this Mac's current budget: 12.0 GiB budget" in out
    assert "--memory-gb: 30.0 GiB budget" in out
    assert "=12" in out and memory_budget.LIMIT_ENV in out


def test_device_ceiling_caps_current_explicit_and_class_budgets(tmp_path, host, capsys):
    _, mx = host
    mx.device_info = lambda: {"max_recommended_working_set_size": 24 * GIB}
    assert cli.cmd_plan(args(model_dir(tmp_path), memory_gb=1000, ram=[64])) == 0
    out = capsys.readouterr().out
    assert "this Mac's current budget: 24.0 GiB budget" in out
    assert "--memory-gb: 24.0 GiB budget" in out
    assert "--ram 64 GiB class: 24.0 GiB budget" in out


def test_a_weights_only_refusal_names_reserve_without_inventing_workspace(tmp_path, host, capsys):
    package, _ = host
    package.weight_bytes = lambda directory: 21 * GIB
    assert cli.cmd_plan(args(model_dir(tmp_path), ram=[32])) == 1
    out = capsys.readouterr().out
    assert "--ram 32 GiB class: 22.4 GiB budget; weights exceed" in out
    assert "need more than 24.0 GiB for weights and process reserve" in out
    assert "one prompt chunk" not in out and "would fit" not in out


@pytest.mark.parametrize("size", [None, 0])
def test_missing_or_empty_weights_never_produce_a_fit(tmp_path, host, capsys, size):
    with pytest.raises(FileNotFoundError, match="weights.*missing or empty"):
        cli.cmd_plan(args(model_dir(tmp_path, size=size)))
    assert "weights within" not in capsys.readouterr().out


def test_index_only_and_metadata_size_do_not_count_as_local_weights(tmp_path, host, capsys):
    model_dir(tmp_path, size=None)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 8 * GIB}, "weight_map": {"x": "model-00001-of-00001.safetensors"}}))
    with pytest.raises(FileNotFoundError, match="weight shard is missing"):
        cli.cmd_plan(args(tmp_path))
    assert "weights within" not in capsys.readouterr().out


@pytest.mark.parametrize("estimate", [0, None, -1, float("nan"), float("inf")])
def test_unknown_family_estimate_never_falls_back_to_a_false_fit(tmp_path, host, estimate):
    package, _ = host
    package.weight_bytes = lambda directory: estimate
    with pytest.raises(ValueError, match="family weight estimate"):
        cli.cmd_plan(args(model_dir(tmp_path)))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_bad_explicit_memory_names_the_flag(tmp_path, host, value):
    with pytest.raises(ValueError, match="--memory-gb"):
        cli.cmd_plan(args(model_dir(tmp_path), memory_gb=value))


@pytest.mark.parametrize("value", [0, -1, 1.5, float("nan")])
def test_bad_ram_class_names_the_flag(tmp_path, host, value):
    with pytest.raises(ValueError, match="--ram"):
        cli.cmd_plan(args(model_dir(tmp_path), ram=[value]))


def test_bad_environment_uses_the_shared_validation(tmp_path, host, monkeypatch):
    monkeypatch.setenv(memory_budget.LIMIT_ENV, "nan")
    with pytest.raises(ValueError, match=memory_budget.LIMIT_ENV):
        cli.cmd_plan(args(model_dir(tmp_path)))


def test_missing_cached_repository_never_fetches_config(tmp_path, host, monkeypatch):
    monkeypatch.setattr(hub, "cached", lambda repo, **kw: None)
    with pytest.raises(FileNotFoundError, match="not in the Hugging Face cache"):
        cli.cmd_plan(args("fixture/not-cached"))


def test_a_complete_cached_repository_needs_no_network(tmp_path, host, monkeypatch, capsys):
    model = model_dir(tmp_path)
    monkeypatch.setattr(hub, "cached", lambda repo, **kw: model)
    assert cli.cmd_plan(args("fixture/already-cached")) == 0
    assert "weights 8.0 GiB (local safetensors file sizes" in capsys.readouterr().out


def test_nonpositive_physical_ram_is_rejected(tmp_path, host, monkeypatch):
    monkeypatch.setattr(memory_budget, "physical_memory_bytes", lambda: 0)
    with pytest.raises(ValueError, match="physical memory must be positive"):
        cli.cmd_plan(args(model_dir(tmp_path)))


def test_huge_finite_explicit_budget_is_clamped_before_byte_conversion(tmp_path, host, capsys):
    assert cli.cmd_plan(args(model_dir(tmp_path), memory_gb=1e308)) == 0
    assert "--memory-gb: 48.0 GiB budget" in capsys.readouterr().out


def test_platform_rejection_precedes_resolving_config_or_importing_mlx(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(hub, "resolve", lambda *a, **kw: pytest.fail("platform check resolved a model"))
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    with pytest.raises(ValueError, match="Mac MLX"):
        cli.cmd_plan(args("fixture/not-cached"))
