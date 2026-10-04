"""`tensorfold serve --engine`: the Python engine unchanged without a gated cell, the native engine exec'd with one."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

import tensorfold
from tensorfold import cli, native
from tensorfold.control import cli as control_cli
from tensorfold.control.runner import logger_for, supervise
from tensorfold.native import contract, switch
from tests.test_native_contract import CAPS as BASE_CAPS, fake_binary

FLAGS = {"--port": {}, "--context": {}, "--alias": {}, "--no-thinking": {}, "--name": {},
         "--backend": {"values": ["auto", "mlx"]}, "--kv-dtype": {"values": ["bf16"]}}
CAPS = {**BASE_CAPS, "serve": {"flags": FLAGS}}
NEMOTRON = {"model_type": "nemotron_h", "max_position_embeddings": 4096, "quantization": {"group_size": 64, "bits": 4}}
LINE = "[tensorfold] engine: zig 0.6.6 (release gate rel066a: nemotron_h mlx-q4g64 on apple-m5)"
FAKE = """
import os, signal, time
if sys.argv[1:] == ["capabilities", "--json"]:
    print(json.dumps(CAPS))
    sys.exit(0)
signal.signal(signal.SIGTERM, lambda *_: (print("term", flush=True), sys.exit(0)))
print("native", os.getpid(), json.dumps(sys.argv[1:]), os.environ.get("TENSORFOLD_ENGINE", "-"), flush=True)
if "--port" in sys.argv:
    time.sleep(60)
sys.exit(7)
"""
RUN = ("import sys; from pathlib import Path; import tensorfold.native as native; native.ROOT = Path(sys.argv[1]); "
       "from tensorfold.cli import main; raise SystemExit(main(sys.argv[2:]))")


class Execd(Exception):
    """os.execv was reached; a real one never returns."""


@pytest.fixture(autouse=True)
def quiet_environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith(switch.KNOBS):
            monkeypatch.delenv(key)
    monkeypatch.setenv("TENSORFOLD_NO_UPDATE_CHECK", "1")
    before = {number: signal.getsignal(number) for number in (signal.SIGPIPE, signal.SIGXFSZ)}
    yield
    for number, handler in before.items():
        signal.signal(number, handler)


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(NEMOTRON))
    return path


@pytest.fixture
def install(tmp_path, monkeypatch):
    """A fake native bundle and gate manifest at tensorfold.native.ROOT; entries default to this bundle's cell."""

    root = tmp_path / "native"

    def make(caps=CAPS, entry=None, manifest=None):
        fake_binary(root / native.BINARY, f"CAPS = {caps!r}\n{FAKE}")
        (root / "kernels").mkdir(exist_ok=True)
        (root / "kernels" / "lane.metallib").write_bytes(b"kernels")
        monkeypatch.setattr(native, "ROOT", root)
        cell = {"family": "nemotron_h", "format": "mlx-q4g64", "backend": "metal", "chip": "apple-m5",
                "run": "rel066a", "bundle": native.bundle_digest(), **(entry or {})}
        (root / native.MANIFEST).write_text(manifest if manifest is not None else
                                            json.dumps({"schema": 1, "entries": [cell]}))
        return root

    return make


@pytest.fixture
def served(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "cmd_serve", lambda args: calls.append(args) or 0)
    return calls


@pytest.fixture
def execd(monkeypatch):
    calls = []

    def execv(path, argv):
        calls.append((Path(path), list(argv), signal.getsignal(signal.SIGPIPE)))
        raise Execd

    monkeypatch.setattr(switch.os, "execv", execv)
    return calls


def _environment():
    keep = {key: value for key, value in os.environ.items() if not key.startswith(switch.KNOBS)}
    return {**keep, "PYTHONPATH": str(Path(tensorfold.__file__).resolve().parents[1]),
            "PYTHONDONTWRITEBYTECODE": "1", "TENSORFOLD_NO_UPDATE_CHECK": "1"}


def test_without_gate_entries_auto_looks_at_nothing_else(tmp_path, monkeypatch, model, served, capsys):
    looked = []
    monkeypatch.setattr(native, "ROOT", tmp_path)
    for target, name in ((native, "binary"), (native, "bundle_digest"), (contract, "capabilities"),
                         (cli, "_config_dir"), (switch, "scan")):
        monkeypatch.setattr(target, name, lambda *args, name=name: looked.append(name))
    assert cli.main(["serve", str(model), "--context", "5"]) == 0              # no manifest at all: a pure wheel
    (tmp_path / native.MANIFEST).write_bytes((Path(__file__).parents[1] / "zig" / "gate.json").read_bytes())
    assert cli.main([str(model), "--context", "5"]) == 0                       # the shipped empty manifest
    assert looked == [] and capsys.readouterr() == ("", "")
    expected = vars(cli.build_parser().parse_args(["serve", str(model), "--context", "5"]))
    assert [vars(args) for args in served] == [expected, expected] and expected["engine"] is None


@pytest.mark.parametrize("argv", [["serve", "--help"], ["service", "install", "--help"], ["--help"]])
def test_help_never_shows_the_engine_flag(argv, capsys):
    with pytest.raises(SystemExit) as done:
        cli.main(argv)
    assert done.value.code == 0 and "--engine" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["serve"])                                          # the usage line of an error
    assert "--engine" not in capsys.readouterr().err


@pytest.mark.parametrize("flag, variable, said", [(["--engine", "python"], None, "--engine python"),
                                                  ([], "python", "TENSORFOLD_ENGINE=python"),
                                                  ([], " Python ", "TENSORFOLD_ENGINE=python"),
                                                  (["--engine", "python"], "zig", "--engine python")])
def test_python_asked_for_is_said_and_served_by_python(flag, variable, said, model, install, served, execd,
                                                       monkeypatch, capsys):
    install()
    if variable:
        monkeypatch.setenv("TENSORFOLD_ENGINE", variable)
    assert cli.main(["serve", str(model), "--backend", "mlx", *flag]) == 0
    assert capsys.readouterr().out == f"[tensorfold] engine: python ({said})\n"
    assert len(served) == 1 and not execd


def test_an_engine_variable_outside_the_three_is_refused(model, served, monkeypatch, capsys):
    monkeypatch.setenv("TENSORFOLD_ENGINE", "rust")
    assert cli.main(["serve", str(model)]) == 1
    assert capsys.readouterr().err == "tensorfold: TENSORFOLD_ENGINE is auto, zig or python, not 'rust'\n"
    assert not served


@pytest.mark.parametrize("extra, variables, config, expected", [
    (["--vision"], {}, NEMOTRON, "the native engine 0.6.6 lacks --vision"),
    (["--kv-dtype", "int8", "--backend", "cuda"], {}, NEMOTRON,
     "lacks --kv-dtype int8, --backend cuda, the cuda backend"),
    ([], {"TENSORFOLD_SEED_SALT": "3", "TF_LANE_TILE": "8"}, NEMOTRON, "lacks TENSORFOLD_SEED_SALT, TF_LANE_TILE"),
    ([], {}, {**NEMOTRON, "model_type": "qwen3_5"}, "lacks qwen3_5 checkpoints"),
    ([], {}, {**NEMOTRON, "quantization": {"group_size": 64, "bits": 8}}, "lacks mlx-q8g64 weights for nemotron_h"),
    (["--temp", "0.5"], {}, NEMOTRON, "takes full flag names and no '--' (--temp)"),
])
def test_engine_zig_names_what_is_missing(extra, variables, config, expected, model, install, served, execd,
                                          monkeypatch, capsys):
    install()
    (model / "config.json").write_text(json.dumps(config))
    for key, value in variables.items():
        monkeypatch.setenv(key, value)
    assert cli.main(["serve", str(model), "--engine", "zig", *extra]) == 1
    error = capsys.readouterr().err
    assert error.startswith("tensorfold: --engine zig: ") and expected in error
    assert not served and not execd


def test_engine_zig_without_a_native_engine_says_so(tmp_path, model, served, monkeypatch, capsys):
    monkeypatch.setattr(native, "ROOT", tmp_path)
    monkeypatch.setenv("TENSORFOLD_ENGINE", "zig")
    assert cli.main(["serve", str(model)]) == 1
    assert capsys.readouterr().err == (f"tensorfold: TENSORFOLD_ENGINE=zig: this install has no native engine "
                                       f"({tmp_path / native.BINARY} is missing)\n")
    assert not served


@pytest.mark.parametrize("argv, expected, line", [
    (["serve", "MODEL", "--backend", "mlx", "--engine", "auto", "--port=9", "--alias", "a", "--no-thinking"],
     ["serve", "MODEL", "--backend", "mlx", "--port=9", "--alias", "a", "--no-thinking"], LINE),
    (["MODEL", "--port", "9", "--engine=zig", "--name", "-1"], ["serve", "MODEL", "--port", "9", "--name", "-1"],
     "[tensorfold] engine: zig 0.6.6 (--engine zig)"),
    (["serve", "--eng", "zig", "MODEL"], ["serve", "MODEL"], "[tensorfold] engine: zig 0.6.6 (--engine zig)"),
])
def test_the_native_engine_gets_the_users_argv_without_the_engine_flag(argv, expected, line, model, install, served,
                                                                       execd, capsys):
    root = install()
    with pytest.raises(Execd):
        cli.main([str(model) if token == "MODEL" else token for token in argv])
    expected = [str(model) if token == "MODEL" else token for token in expected]
    assert execd == [(root / native.BINARY, ["tensorfold", *expected], signal.SIG_DFL)]
    assert capsys.readouterr().out == line + "\n" and not served


@pytest.mark.parametrize("change", ["chip", "bundle", "format", "family", "backend", "flag", "value", "variable",
                                    "abbreviation", "separator", "manifest", "probe", "mode"])
def test_auto_keeps_python_wherever_the_cell_or_the_command_differs(change, model, install, served, execd,
                                                                    monkeypatch, capsys):
    entry = {"chip": "apple-m3"} if change == "chip" else {"bundle": "0" * 64} if change == "bundle" else {}
    caps = {**CAPS, "chip": None} if change == "probe" else CAPS
    root = install(caps=caps, entry=entry, manifest="{" if change == "manifest" else None)
    config = {**NEMOTRON, "quantization": {"group_size": 32, "bits": 4}} if change == "format" else \
        {**NEMOTRON, "model_type": "qwen3_5"} if change == "family" else NEMOTRON
    (model / "config.json").write_text(json.dumps(config))
    argv = {"backend": ["--backend", "cuda"], "flag": ["--vision"], "value": ["--kv-dtype", "int8"],
            "abbreviation": ["--temp", "0.5"]}.get(change, [])
    if change == "variable":
        monkeypatch.setenv("TF_DRAFT_LEAN", "1")
    if change == "mode":
        (root / native.BINARY).chmod(0o644)
    if change == "probe":
        (root / native.BINARY).write_text((root / native.BINARY).read_text().replace("sys.exit(0)", "sys.exit(3)", 1))
    head = ["serve", "--", str(model)] if change == "separator" else ["serve", str(model)]
    assert cli.main([*head, *argv]) == 0
    assert capsys.readouterr() == ("", "") and len(served) == 1 and not execd


def test_auto_follows_the_platform_backend_like_the_python_engine():
    assert [switch.backend(choice, "darwin") for choice in ("auto", "mlx", "cuda")] == ["metal", "metal", "cuda"]
    assert [switch.backend(choice, "linux") for choice in ("auto", "mlx", "cuda")] == ["cuda", "metal", "cuda"]


@pytest.mark.skipif(os.name != "posix", reason="exec and signals are POSIX")
def test_the_native_engine_takes_over_the_process_its_pid_output_signals_and_status(model, install):
    root = install()
    command = [sys.executable, "-B", "-c", RUN, str(root), "serve", str(model), "--backend", "mlx"]
    done = subprocess.run([*command, "--context", "100"], capture_output=True, text=True, timeout=60,
                          env={**_environment(), "TENSORFOLD_ENGINE": "auto"})
    first, second = done.stdout.splitlines()
    assert done.returncode == 7 and first == LINE and done.stderr == ""
    given = json.dumps(["serve", str(model), "--backend", "mlx", "--context", "100"])
    assert second.split(" ", 2)[2] == f"{given} auto"
    process = subprocess.Popen([*command, "--port", "9"], stdout=subprocess.PIPE, text=True, env=_environment())
    try:
        assert process.stdout.readline().rstrip() == LINE
        assert process.stdout.readline().split()[:2] == ["native", str(process.pid)]     # the same process
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20) == 0 and process.stdout.read() == "term\n"
    finally:
        process.kill()
        process.wait()


def test_service_install_writes_the_engine_on_the_serve_command():
    def command(*flags):
        args = control_cli.parser().parse_args(["service", "install", "Org/Model", *flags])
        return control_cli._profile(args).command()

    assert command()[2:5] == ["-m", "tensorfold", "serve"]            # a service runs the switch like any serve
    assert "--engine" not in command() and "--engine" not in command("--engine", "auto")
    assert command("--engine", "python")[-4:] == ["--parallel", "auto", "--engine", "python"]


@pytest.mark.skipif(os.name != "posix", reason="exec and signals are POSIX")
def test_a_control_room_service_supervises_the_native_engine_as_its_server(tmp_path, model, install):
    root = install()
    log = logger_for(tmp_path / "server.log", 65536, 2)
    stop = threading.Event()

    class Started(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("native "):
                stop.set()

    log.addHandler(Started())
    watchdog = threading.Timer(30, stop.set)
    watchdog.start()
    try:
        code = supervise([sys.executable, "-B", "-u", "-c", RUN, str(root), "serve", str(model), "--backend", "mlx",
                          "--port", "9"], _environment(), log, stop, grace=10)
    finally:
        watchdog.cancel()
        for handler in log.handlers:
            handler.close()
    lines = [line.split(" ", 1)[1] for line in (tmp_path / "server.log").read_text().splitlines()]
    pid = lines[0].rsplit("=", 1)[1]
    assert code == 0 and lines[0].startswith("[control] server process started pid=")
    assert lines[1] == LINE and lines[2].startswith(f"native {pid} ")
    assert lines[3:] == ["[control] forwarding SIGTERM; allowing 10.0s to exit", "term",
                         "[control] server exited status=0"]


def test_scan_reads_flags_as_argparse_does():
    parser = cli.build_parser()
    argv = ["serve", "m", "--temperature=0", "--alias", "a", "--alias", "b", "--name", "--x y", "--no-thinking",
            "--engine", "zig", "--top-k", "-1"]
    parser.parse_args(argv)
    flags, kept, odd = switch.scan(parser, argv)
    assert flags == [("--temperature", "0"), ("--alias", "a"), ("--alias", "b"), ("--name", "--x y"),
                     ("--no-thinking", None), ("--top-k", "-1")]
    assert kept == [token for token in argv if token not in ("--engine", "zig")] and odd == []
