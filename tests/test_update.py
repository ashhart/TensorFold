"""Version tracking and `tensorfold update`, with GitHub and pip replaced by fakes (no network, no installs)."""

import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold import __version__, cli, update


SAMPLE = """# What's new in TensorFold

## 99.0.0 (1 Jan 2099)

- Faster everything.

## 0.3.5.1 (28 Sep 2026)

- The M1 and M2 fix.
"""


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "CACHE", tmp_path / "update-check.json")
    monkeypatch.setattr(update, "SEEN", tmp_path / "version-seen")
    monkeypatch.setattr("urllib.request.urlopen", _offline)


def _offline(*args, **kwargs):
    raise OSError("offline")


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


def test_whats_new_keeps_the_versions_after_this_one():
    notes = update.whats_new(SAMPLE, "0.3.5.1", "v99.0.0")
    assert notes.startswith("## 99.0.0") and "Faster everything" in notes and "M1 and M2" not in notes
    assert update.whats_new(SAMPLE, "0.3.5", "v0.3.5.1").startswith("## 0.3.5.1")
    assert update.whats_new(SAMPLE, "99.0.0", "v99.0.0") == ""


def test_update_shows_whats_new_from_the_new_tag(monkeypatch, capsys):
    fetched = []

    def urlopen(request, timeout):
        fetched.append(request.full_url)
        return io.BytesIO(SAMPLE.encode())

    monkeypatch.setattr(update, "latest_release", lambda **kwargs: "v99.0.0")
    monkeypatch.setattr(update, "_editable_clone", lambda: None)
    monkeypatch.setattr(update.subprocess, "call", lambda command, **kw: 0)
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="99.0.0\n", returncode=0))
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert update.update() == 0
    out = capsys.readouterr().out
    assert f"What's new since {__version__}" in out and "Faster everything" in out
    assert fetched == ["https://raw.githubusercontent.com/ashhart/TensorFold/v99.0.0/CHANGELOG.md"]
    assert update.SEEN.read_text() == "99.0.0"             # the first run after it stays quiet


def test_an_editable_clone_reads_its_own_changelog(monkeypatch, tmp_path, capsys):
    (tmp_path / "CHANGELOG.md").write_text(SAMPLE)
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: "v99.0.0")
    monkeypatch.setattr(update, "_editable_clone", lambda: tmp_path)
    monkeypatch.setattr(update.subprocess, "call", lambda command, **kw: 0)
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=0))
    monkeypatch.setattr("urllib.request.urlopen", _no_network)
    assert update.update() == 0
    assert "Faster everything" in capsys.readouterr().out


def test_offline_the_notes_link_still_prints(capsys):
    update.show_whats_new("0.3.5.1", "v99.0.0")
    out = capsys.readouterr().out
    assert "What's new" not in out and "blob/v99.0.0/CHANGELOG.md" in out


def test_the_first_run_of_a_version_points_at_its_notes_once(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_NO_UPDATE_CHECK", raising=False)
    assert f"TensorFold {__version__}" in update.first_run_notice()
    assert update.first_run_notice() is None
    update.SEEN.write_text("0.0.1")
    assert "what's new" in update.first_run_notice()       # updated some other way
    update.SEEN.write_text("99.0.0")
    assert update.first_run_notice() is None               # a downgrade
    monkeypatch.setenv("TENSORFOLD_NO_UPDATE_CHECK", "1")
    update.SEEN.write_text("0.0.1")
    assert update.first_run_notice() is None


def test_the_changelog_has_this_version():
    text = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    assert re.search(rf"^## {re.escape(__version__)} ", text, re.M), f"CHANGELOG.md needs a ## {__version__} section"
