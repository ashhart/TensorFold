"""Cortheon (TensorFold 1.0.9): web research in the engine, a setting of its own that says what leaves the machine."""

import pytest

from tensorfold_web import native
from tensorfold_web import settings as settings_mod
from tensorfold_web.settings import Settings


def _offers(monkeypatch, flags):
    monkeypatch.setattr(native, "capabilities", lambda _b: {"serve": {"flags": {f: {} for f in flags}}})
    monkeypatch.setattr(native, "serve_flags", lambda _b: {f: {} for f in flags})


def test_web_research_is_off_unless_turned_on(monkeypatch):
    _offers(monkeypatch, ["--context", "--cortheon"])
    cmd, _, _ = native.build_command("tf", "m", Settings())
    assert "--cortheon" not in cmd
    cmd, _, notes = native.build_command("tf", "m", Settings(cortheon=True))
    assert "--cortheon" in cmd and not notes


def test_an_engine_without_it_starts_without_it_and_says_so(monkeypatch):
    _offers(monkeypatch, ["--context"])
    cmd, _, notes = native.build_command("tf", "m", Settings(cortheon=True))
    assert "--cortheon" not in cmd and any("--cortheon" in n for n in notes)


def test_it_is_the_settings_switch_not_an_extra_flag():
    with pytest.raises(ValueError, match="Web research in Settings"):
        settings_mod.validate({"extra_args": ["--cortheon"]})
    assert settings_mod.validate({"cortheon": True}) == {"cortheon": True}
    with pytest.raises(ValueError, match="true or false"):
        settings_mod.validate({"cortheon": "yes"})
