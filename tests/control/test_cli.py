import os
from pathlib import Path
import plistlib
import subprocess
import sys

import pytest

from tensorfold.control.cli import main, parser


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
