from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import pytest

from tensorfold.control.config import Paths, Profile, Store, read_environment, validate_env
from tensorfold.control.safety import ControlError, atomic_write, clean, private_read, redact


@pytest.mark.parametrize("name", ["", "../bad", "UPPER", "a/b", "-x", "a.b", "a" * 49, "a\n"])
def test_bad_name(name):
    with pytest.raises(ControlError):
        Profile(name, "Org/Model")


@pytest.mark.parametrize("port", [0, -1, 80, 65536, True, 8080.0, "8080"])
def test_bad_port(port):
    with pytest.raises(ControlError):
        Profile("default", "Org/Model", port=port)


@pytest.mark.parametrize("host,allow,endpoint", [("127.0.0.1", False, "http://127.0.0.1:8080"),
    ("::1", False, "http://[::1]:8080"), ("0.0.0.0", True, "http://127.0.0.1:8080"),
    ("::", True, "http://[::1]:8080"), ("192.0.2.1", True, "http://192.0.2.1:8080")])
def test_endpoint(host, allow, endpoint):
    assert Profile("default", "Org/Model", host=host, allow_network=allow).endpoint == endpoint


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.1", "localhost", "$(uname)"])
def test_exposure_requires_ack(host):
    with pytest.raises(ControlError):
        Profile("default", "Org/Model", host=host)


@pytest.mark.parametrize("arg", ["--host", "--host=0.0.0.0", "--ho=0.0.0.0", "--port", "--name=x",
    "--backend=cuda", "--", "--api-key=bad", "--password=bad", "bad\x1b[2J"])
def test_managed_flags_cannot_be_overridden(arg):
    with pytest.raises(ControlError):
        Profile("default", "Org/Model", args=(arg,))


def test_literal_arguments_no_shell(profile):
    p = replace(profile, model="a model;$(touch /tmp/never)", args=("--context", "32768", "--max-tokens=2048"))
    command = p.command()
    assert command[5] == p.model
    assert "--context" in command
    assert Profile.decode(p.encode()) == p


def test_interpreter_symlink_is_preserved(tmp_path):
    link = tmp_path / "venv/bin/python"
    link.parent.mkdir(parents=True)
    link.symlink_to(sys.executable)
    p = Profile("default", "Org/Model", python=str(link))
    assert p.python == str(link)  # MUST NOT resolve() into the base interpreter


@pytest.mark.parametrize("env", [{"HF_TOKEN": "x"}, {"DYLD_INSERT_LIBRARIES": "x"}, {"PATH": "x"},
    {"PYTHONPATH": "x"}, {"tf_bad": "x"}, {"TF_X": "bad\nvalue"}, {"TF_X": True}])
def test_bad_environment(env):
    with pytest.raises(ControlError):
        validate_env(env, secrets=False)


def test_environment_file_permissions_and_offline(tmp_path, profile):
    path = tmp_path / "private.json"
    atomic_write(path, json.dumps({"HF_TOKEN": "secret", "HF_HUB_OFFLINE": "0"}).encode())
    p = replace(profile, environment_file=str(path))
    env = read_environment(p)
    assert env["HF_TOKEN"] == "secret" and env["HF_HUB_OFFLINE"] == "1"
    assert "secret" not in p.encode().decode()
    if os.name != "nt":
        path.chmod(0o644)
        with pytest.raises(ControlError):
            read_environment(p)


def test_store_corruption_is_visible(tmp_path, profile):
    store = Store(Paths(tmp_path))
    store.put(profile)
    atomic_write(store.paths.profile("broken"), b'{"unexpected":true}')
    profiles, errors = store.list()
    assert profiles == [profile] and len(errors) == 1


def test_private_files_reject_symlinks(tmp_path):
    original = tmp_path / "real"
    atomic_write(original, b"x")
    target = tmp_path / "link"
    target.symlink_to(original)
    for operation in (lambda: private_read(target), lambda: atomic_write(target, b"y")):
        with pytest.raises(ControlError):
            operation()
    assert original.read_bytes() == b"x"


def test_controls_and_secrets_do_not_render():
    text = 'before\x1b]52;c;c2VjcmV0\x07 after\x1b[2J\x00\u202eevil'
    assert clean(text) == "before afterevil"
    assert "[red]" in clean("[red]literal markup")
    for value in ["Authorization: Bearer sensitive", "HF_TOKEN=hf_abcdefghijklmno", 'api_key="abc xyz"']:
        assert "[REDACTED]" in redact(value)
        assert "sensitive" not in redact(value)


@pytest.mark.parametrize("values", [{"schema": 2}, {"schema": True}, {"log_bytes": 1}, {"log_backups": 0},
    {"allow_download": "yes"}, {"python": "python"}, {"environment_file": "relative"}, {"backend": "cuda"}])
def test_strict_profile_schema(profile, values):
    with pytest.raises(ControlError):
        replace(profile, **values)
