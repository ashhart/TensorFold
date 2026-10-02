import os
from pathlib import Path
import plistlib
import subprocess
import sys

import pytest

from tensorfold.control.cli import _profile, main, parser
from tensorfold.control.config import Profile


def test_smoke_name_loads_and_install_refuses_it(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    profile = Profile("control-smoke", "Org/Model")
    assert profile.label == "dev.tensorfold.control-smoke"
    assert Profile.decode(profile.encode()) == profile
    assert main(["service", "install", "Org/Model", "--name", "control-smoke", "--dry-run"]) == 1
    assert "reserved" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_dry_run_is_side_effect_free(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert main(["service", "install", "Org/Model", "--dry-run", "--context", "32768"]) == 0
    xml = capsys.readouterr().out
    data = plistlib.loads(xml.encode())
    assert data["Label"] == "dev.tensorfold.default"
    assert not list(tmp_path.iterdir())


def test_root_parser_registration():
    args = parser().parse_args(["tui", "--demo", "--interval", "1"])
    assert args.command == "tui" and args.demo and args.interval == 1


def test_no_accidental_uninstall(capsys):
    assert main(["service", "uninstall", "default"]) == 1
    assert "--yes" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", ["svg", "html", "txt"])
def test_real_snapshot_cli(tmp_path, suffix):
    path = tmp_path / f"snapshot.{suffix}"
    assert main(["tui", "--demo", "--snapshot", str(path)]) == 0
    data = path.read_text()
    assert "TENSORFOLD" in data and "DEMO" in data
    assert "cdnjs.cloudflare" not in data


def test_service_import_does_not_import_tui_or_gpu():
    code = "before=set(__import__('sys').modules); import tensorfold.control.cli; " \
           "loaded=set(__import__('sys').modules)-before; " \
           "assert not any(n.split('.')[0] in {'rich','prompt_toolkit','torch','mlx','tokenizers'} for n in loaded)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_missing_token_does_not_launch(capsys, monkeypatch):
    monkeypatch.delenv("TF_TEST_NO_TOKEN", raising=False)
    assert main(["tui", "--token-env", "TF_TEST_NO_TOKEN"]) == 1
    assert "unset or empty" in capsys.readouterr().err


def test_service_definition_carries_limit_port_and_parallel(manager, monkeypatch):
    from tensorfold.control import runner
    from tensorfold.control.config import read_environment

    service, _transport = manager
    args = parser().parse_args([
        "service", "install", "Org/Nemotron", "--name", "nemotron",
        "--port", "8081", "--parallel", "1", "--env", "TENSORFOLD_MEMORY_LIMIT_GB=48",
    ])
    service.install(_profile(args))
    stored = service.store.get("nemotron")
    assert stored.port == 8081
    assert stored.environment == {"TENSORFOLD_MEMORY_LIMIT_GB": "48"}
    command = stored.command()
    assert command[command.index("--port") + 1] == "8081"
    assert command[command.index("--parallel") + 1] == "1"
    assert read_environment(stored)["TENSORFOLD_MEMORY_LIMIT_GB"] == "48"
    text = service.paths.profile("nemotron").read_text()
    assert '"TENSORFOLD_MEMORY_LIMIT_GB": "48"' in text
    captured = {}

    def fake_supervise(argv, environment, log, stop, *, grace=15):
        captured["argv"] = list(argv)
        captured["environment"] = dict(environment)
        return 0

    monkeypatch.setattr(runner, "supervise", fake_supervise)
    assert runner.main(["--profile", str(service.paths.profile("nemotron")),
                        "--log", str(service.paths.log("nemotron"))]) == 0
    argv = captured["argv"]
    assert argv[argv.index("--port") + 1] == "8081"
    assert argv[argv.index("--parallel") + 1] == "1"
    assert "serve" in argv
    assert captured["environment"]["TENSORFOLD_MEMORY_LIMIT_GB"] == "48"
