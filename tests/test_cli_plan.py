"""The plan command's budget arithmetic: fits or refuses without loading weights."""

import argparse
import json
from types import SimpleNamespace

import pytest

from tensorfold import cli
from tensorfold.server import memory_budget


def _family(estimate=None, fraction=None):
    package = SimpleNamespace(weight_bytes=estimate)
    if fraction is not None:
        package.memory_fraction = fraction
    family = SimpleNamespace(title="Test family", model_type="test", module="tensorfold.families.test",
                             package=package)
    return family


def _model_dir(tmp_path, layers=64):
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 262144,
                                                      "num_hidden_layers": layers}))
    (tmp_path / "model.safetensors").write_bytes(b"0" * (8 * 1024**3))      # 8 GiB on disk
    return tmp_path


def _args(model, *, memory_gb=None, ram=()):
    return argparse.Namespace(model=str(model), memory_gb=memory_gb, ram=list(ram))


def test_a_small_checkpoint_fits_the_default_budget(tmp_path, monkeypatch, capsys):
    model = _model_dir(tmp_path)
    family = _family(estimate=lambda *a, **k: 6 * 1024**3)
    monkeypatch.setattr("tensorfold.families.detect", lambda path: family)
    monkeypatch.setattr("tensorfold.families.require_readable", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_note_untested", lambda *a: None)
    monkeypatch.setattr(memory_budget, "physical_memory_bytes", lambda: 48 * 1024**3)
    monkeypatch.setattr(memory_budget, "budget_ceiling", lambda mx, physical_bytes=None: 37.4 * 1024**3)
    assert cli.cmd_plan(_args(model)) == 0
    out = capsys.readouterr().out
    assert "fits" in out and "does not fit" not in out
    assert "weights 6.0 GiB" in out


def test_a_big_checkpoint_refuses_and_names_the_need(tmp_path, monkeypatch, capsys):
    model = _model_dir(tmp_path)
    family = _family(estimate=lambda *a, **k: 21 * 1024**3)      # needs 24 with the reserve
    monkeypatch.setattr("tensorfold.families.detect", lambda path: family)
    monkeypatch.setattr("tensorfold.families.require_readable", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_note_untested", lambda *a: None)
    monkeypatch.setattr(memory_budget, "physical_memory_bytes", lambda: 48 * 1024**3)
    monkeypatch.setattr(memory_budget, "budget_ceiling", lambda mx, physical_bytes=None: 37.4 * 1024**3)
    assert cli.cmd_plan(_args(model, memory_gb=30.0, ram=[32])) == 1
    out = capsys.readouterr().out
    assert "this Mac's default: 33.6 GiB budget — fits; about 9.6 GiB" in out
    assert "--ram 32 GB class: 22.4 GiB budget — does not fit; the model and one prompt chunk need about 24.0 GiB" in out
    assert "--memory-gb: 30.0 GiB budget — fits; about 6.0 GiB" in out   # an explicit budget that clears the class


def test_the_hint_names_the_smallest_budget_that_fits(tmp_path, monkeypatch, capsys):
    model = _model_dir(tmp_path)
    family = _family(estimate=lambda *a, **k: 21 * 1024**3)
    monkeypatch.setattr("tensorfold.families.detect", lambda path: family)
    monkeypatch.setattr("tensorfold.families.require_readable", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_note_untested", lambda *a: None)
    monkeypatch.setattr(memory_budget, "physical_memory_bytes", lambda: 32 * 1024**3)
    monkeypatch.setattr(memory_budget, "budget_ceiling", lambda mx, physical_bytes=None: 37.4 * 1024**3)
    assert cli.cmd_plan(_args(model)) == 1
    out = capsys.readouterr().out
    assert "TENSORFOLD_MEMORY_LIMIT_GB=27" in out               # weights 21 + two reserves 6


def test_plan_is_mac_only(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "linux")
    model = _model_dir(tmp_path)
    family = _family()
    monkeypatch.setattr("tensorfold.families.detect", lambda path: family)
    monkeypatch.setattr("tensorfold.families.require_readable", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_note_untested", lambda *a: None)
    with pytest.raises(ValueError, match="Mac"):
        cli.cmd_plan(_args(model))
