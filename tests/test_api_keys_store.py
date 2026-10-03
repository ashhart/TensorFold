"""Key rotation never turns a configured gate into an open server."""

import hashlib
import os
import signal
from types import SimpleNamespace

import pytest

from tensorfold.server.authentication import KeyStore, configure


def file_at(tmp_path, text="alpha: fixture-first\n"):
    path = tmp_path / "keys"
    path.write_text(text)
    path.chmod(0o600)
    return path


def bearer(key):
    return {"Authorization": "Bearer " + key}


def test_sources_are_additive_and_only_digests_survive(tmp_path):
    store = KeyStore(["fixture-command"], key_file=file_at(tmp_path, "# comment\n\nalpha: fixture-first\nfixture-second\n"),
                     environment="fixture-environment,fixture-extra")
    assert store.match(bearer("fixture-command")) == "cli-1"
    assert store.match(bearer("fixture-environment")) == "env-1"
    assert store.match({"x-api-key": "fixture-first"}) == "alpha"
    assert store.match(bearer("fixture-second")) == "file-2"
    assert "fixture-command" not in repr(store.__dict__)
    assert "fixture-first" not in repr(store.__dict__)
    assert all(isinstance(digest, bytes) and len(digest) == 32 for digest, _ in store._static + store._file)


def test_every_digest_is_compared_for_first_last_and_wrong_keys(monkeypatch):
    import tensorfold.server.authentication as auth

    original = auth.hmac.compare_digest
    calls = []
    monkeypatch.setattr(auth.hmac, "compare_digest", lambda a, b: calls.append((a, b)) or original(a, b))
    store = KeyStore(["fixture-first", "fixture-second", "fixture-third"])
    for key, expected in (("fixture-first", "cli-1"), ("fixture-third", "cli-3"), ("wrong", None)):
        calls.clear()
        assert store.match(bearer(key)) == expected
        assert len(calls) == 6
        assert {a for a, _ in calls} == {hashlib.sha256(k.encode()).digest()
                                      for k in ("fixture-first", "fixture-second", "fixture-third")}


def test_mtime_rotation_is_polled_at_most_once_a_second(tmp_path):
    now = [0.0]
    path = file_at(tmp_path)
    store = KeyStore(key_file=path, clock=lambda: now[0])
    assert store.match(bearer("fixture-first")) == "alpha"
    path.write_text("beta: fixture-next\n")
    now[0] = 0.5
    assert store.match(bearer("fixture-first")) == "alpha"
    assert store.match(bearer("fixture-next")) is None
    now[0] = 1.0
    assert store.match(bearer("fixture-first")) is None
    assert store.match(bearer("fixture-next")) == "beta"


def test_sighup_rotates_immediately_and_restores_the_signal_handler(tmp_path):
    path = file_at(tmp_path)
    store = KeyStore(key_file=path, clock=lambda: 0.0)
    before = signal.getsignal(signal.SIGHUP)
    with store.signals():
        assert store.match(bearer("fixture-first")) == "alpha"
        path.write_text("beta: fixture-next\n")
        os.kill(os.getpid(), signal.SIGHUP)
        assert store.match(bearer("fixture-next")) == "beta"
        assert store.match(bearer("fixture-first")) is None
    assert signal.getsignal(signal.SIGHUP) == before


@pytest.mark.parametrize("mode", [0o604, 0o640, 0o644])
def test_readable_by_other_users_refuses_startup(tmp_path, mode):
    path = file_at(tmp_path)
    path.chmod(mode)
    with pytest.raises(ValueError, match="unreadable by other users"):
        KeyStore(key_file=path)


@pytest.mark.parametrize("damage", ["blank", "permissions", "missing", "bad-label"])
def test_invalid_rotation_fails_closed_and_recovers(tmp_path, damage, capsys):
    path = file_at(tmp_path)
    store = KeyStore(["fixture-static"], key_file=path)
    if damage == "blank":
        path.write_text("# empty\n")
    elif damage == "permissions":
        path.chmod(0o644)
    elif damage == "missing":
        path.unlink()
    else:
        path.write_text("bad label: fixture-first\n")
    store.request_reload()
    assert store.enabled and store.match(bearer("fixture-first")) is None
    assert store.match(bearer("fixture-static")) is None
    path.write_text("restored: fixture-next\n")
    path.chmod(0o600)
    store.request_reload()
    assert store.match(bearer("fixture-next")) == "restored"
    assert store.match(bearer("fixture-static")) == "cli-1"
    assert "fixture" not in capsys.readouterr().out


def test_configuration_clears_plaintext_sources_and_warns_once(monkeypatch, capsys):
    args = SimpleNamespace(api_key=["fixture-command"], host="0.0.0.0")
    monkeypatch.setenv("TENSORFOLD_API_KEY", "fixture-environment")
    store = configure(args)
    assert args.api_key == [] and "TENSORFOLD_API_KEY" not in os.environ
    assert store.match(bearer("fixture-environment")) == "env-1"
    assert not capsys.readouterr().out
    configure(SimpleNamespace(host="0.0.0.0"))
    assert capsys.readouterr().out.count("warning:") == 1
    configure(SimpleNamespace(host="::1"))
    assert not capsys.readouterr().out


@pytest.mark.parametrize("value", ["", "white space", "new\nline"])
def test_invalid_key_error_does_not_echo_key(value):
    with pytest.raises(ValueError, match="nonempty text without whitespace") as exc:
        KeyStore([value])
    if value:
        assert value not in str(exc.value)
