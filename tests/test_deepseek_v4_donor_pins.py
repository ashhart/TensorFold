"""Pinned native inputs are verified and reused without unnecessary fetches."""

import hashlib
import subprocess
from unittest.mock import patch

import pytest

from tensorfold.families.deepseek_v4.cuda import sources as s


@pytest.fixture
def local_donor(tmp_path, monkeypatch):
    repository = tmp_path / "donor"
    repository.mkdir()
    (repository / "ds4.c").write_text("pinned native input\n")
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "ds4.c"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "pin",
        ],
        check=True,
    )
    pin = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "repository": str(repository),
        "revision": pin,
        "sha256": {"ds4.c": hashlib.sha256((repository / "ds4.c").read_bytes()).hexdigest()},
    }
    monkeypatch.setattr(s, "PIN", pin)
    monkeypatch.setattr(s, "source_manifest", lambda: manifest)
    monkeypatch.delenv("TENSORFOLD_DS4_SOURCE", raising=False)
    return repository


def test_packaged_pin_and_license():
    manifest = s.source_manifest()
    assert manifest["revision"] == s.PIN
    assert len(manifest["sha256"]) == 42
    assert (
        manifest["sha256"]["LICENSE"]
        == hashlib.sha256((s.PACKAGE.parents[4] / "LICENSES/ds4.txt").read_bytes()).hexdigest()
    )


def test_local_and_cached_sources_never_fetch(local_donor, tmp_path):
    cache = tmp_path / "cache"
    with patch.object(s.subprocess, "run", side_effect=AssertionError("Git reached")):
        assert s.resolve_sources(local_donor, cache_dir=cache, offline=True) == cache
        assert s.resolve_sources(cache_dir=cache) == cache
    (cache / "ds4.c").write_text("damaged cache")
    with pytest.raises(ValueError, match="missing or changed"):
        s.resolve_sources(cache_dir=cache)


def test_modified_local_checkout_uses_pinned_object_without_fetch(local_donor, tmp_path):
    (local_donor / "ds4.c").write_text("local edits must remain intact")
    original_run = subprocess.run

    def no_fetch(command, **kwargs):
        assert "fetch" not in command
        return original_run(command, **kwargs)

    with patch.object(s.subprocess, "run", side_effect=no_fetch):
        cache = s.resolve_sources(local_donor, cache_dir=tmp_path / "cache", offline=True)
    assert (cache / "ds4.c").read_text() == "pinned native input\n"
    assert (local_donor / "ds4.c").read_text() == "local edits must remain intact"


def test_missing_source_fetches_once_and_offline_refuses(local_donor, tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="cache missing"):
        s.resolve_sources(cache_dir=cache, offline=True)
    with patch.object(s.subprocess, "run", wraps=subprocess.run) as run:
        s.resolve_sources(cache_dir=cache)
        s.resolve_sources(cache_dir=cache)
    assert sum("fetch" in call.args[0] for call in run.call_args_list) == 1
    s.verify_sources(cache)
