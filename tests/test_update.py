"""Version tracking and `tensorfold update`, with GitHub and pip replaced by fakes (no network, no installs)."""

import io
import json
import sys
from types import SimpleNamespace

import pytest

from tensorfold import __version__, cli, update


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "CACHE", tmp_path / "update-check.json")


def _no_network(*args, **kwargs):
    raise AssertionError("the network was used")


def test_versions_compare_numerically():
    assert update.parse_version("v0.3.10") == (0, 3, 10)
    assert update.newer("v0.3.10", "0.3.9") and not update.newer("v0.3.1", "0.3.1")
    assert not update.newer("nightly", "0.3.1")


def test_latest_release_reads_github_once_a_day(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(json.dumps({"tag_name": "v9.9.9"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert update.latest_release() == "v9.9.9"
    monkeypatch.setattr("urllib.request.urlopen", _no_network)
    assert update.latest_release() == "v9.9.9"             # from the day's cache
    assert calls == [update.RELEASES_API]


def test_offline_is_silent(monkeypatch):
    def urlopen(request, timeout):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert update.latest_release() is None
    assert update.notice(None) is None


def test_notice_only_for_a_newer_release():
    assert "run `tensorfold update`" in update.notice("v99.0.0")
    assert update.notice(f"v{__version__}") is None


def test_background_check_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(update, "latest_release", _no_network)
    assert update.check_in_background() is None


def test_background_check_prints_the_notice(monkeypatch, capsys):
    monkeypatch.delenv("TENSORFOLD_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(update, "latest_release", lambda: "v99.0.0")
    update.check_in_background().join(5)
    assert "TensorFold 99.0.0 is available" in capsys.readouterr().out


def test_update_installs_the_latest_tag_with_this_python(monkeypatch):
    commands = []
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: "v99.0.0")
    monkeypatch.setattr(update, "_editable_clone", lambda: None)
    monkeypatch.setattr(update.subprocess, "call", lambda command, **kw: commands.append(command) or 0)
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="99.0.0\n", returncode=0))
    assert update.update() == 0
    assert commands == [[sys.executable, "-m", "pip", "install", "--upgrade",
                         "git+https://github.com/ashhart/TensorFold.git@v99.0.0"]]


def test_check_only_and_current_install_nothing(monkeypatch, capsys):
    monkeypatch.setattr(update.subprocess, "call", _no_network)
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: "v99.0.0")
    assert update.update(check_only=True) == 0
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: f"v{__version__}")
    assert update.update() == 0
    assert "is the latest release" in capsys.readouterr().out


def test_editable_clone_fast_forwards_only_when_clean(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: "v99.0.0")
    monkeypatch.setattr(update, "_editable_clone", lambda: tmp_path)
    monkeypatch.setattr(update.subprocess, "call", lambda command, **kw: commands.append(command) or 0)
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=0))
    assert update.update() == 0
    assert commands == [["git", "-C", str(tmp_path), "fetch", "--tags", "origin"],
                        ["git", "-C", str(tmp_path), "merge", "--ff-only", "v99.0.0"]]
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=" M file.py\n", returncode=0))
    assert update.update() == 1                            # local changes: left alone


def test_the_cli_has_update_and_the_serve_switch(monkeypatch):
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: f"v{__version__}")
    assert cli.main(["update", "--check"]) == 0
    assert cli.build_parser().parse_args(["serve", "owner/model", "--no-update-check"]).no_update_check
