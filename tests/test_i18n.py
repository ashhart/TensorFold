"""The Simplified Chinese catalogs: locale detection, translation mechanics and call-site coverage."""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tensorfold.i18n import LANG_ENV, _catalog, locale, pad, t
from tensorfold.i18n.parser import usage_error

ROOT = Path(__file__).resolve().parents[1] / "src" / "tensorfold"
# The engine, family, kernel and server layers raise diagnostics that stay English on purpose; the command
# line and the control room are the translated surface. See i18n/GLOSSARY_zh.md section 1.7.
FRONT_END = ("cli.py", "cli_args.py", "cli_plan.py", "serve_options.py", "update.py", "hub.py",
             "server/memory_budget.py", "server/thinking_notes.py")


def modules():
    """Every module that translates text, plus the front end that must keep doing so."""

    for path in sorted(ROOT.rglob("*.py")):
        if "vendor" not in path.parts:
            yield path


def imports_t(path: Path) -> bool:
    """Whether ``path`` binds ``t`` from the i18n package (a local ``t`` is unrelated)."""

    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tensorfold.i18n")
                and any(alias.name == "t" for alias in node.names)):
            return True
    return False


def call_site_keys(path: Path) -> list[tuple[int, str]]:
    keys = []
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t"
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            keys.append((node.lineno, node.args[0].value))
    return keys


def parser_help_strings(node: argparse.ArgumentParser) -> list[str]:
    """Every help string the parser tree can print, after argparse has expanded its own defaults."""

    found = []
    for action in node._actions:
        if action.help and action.help is not argparse.SUPPRESS:
            found.append(action.help)
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                found.extend(parser_help_strings(sub))
    return found


def test_locale_resolution(monkeypatch):
    monkeypatch.setenv("LC_ALL", "zh_CN.UTF-8")
    monkeypatch.delenv(LANG_ENV, raising=False)
    monkeypatch.setattr("tensorfold.i18n._under_test", lambda: False)
    assert locale() == "zh"
    for value in ("zh", "zh-Hans", "zh_SG.UTF-8", "ZH_CN", "zh-CN"):
        monkeypatch.setenv(LANG_ENV, value)
        assert locale() == "zh", value
    for value in ("en_US.UTF-8", "ja_JP.UTF-8", "zh_TW.UTF-8", "zh-HK", "zh-Hant", "C", "POSIX", "C.UTF-8", ""):
        monkeypatch.setenv(LANG_ENV, value)
        if value:
            assert locale() == "en", value
    # A Traditional Chinese environment selects English; a Simplified one wins over LC_ALL.
    monkeypatch.delenv(LANG_ENV, raising=False)
    monkeypatch.setenv("LC_ALL", "zh_TW.UTF-8")
    assert locale() == "en"
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("LC_MESSAGES", "")
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    assert locale() == "zh"


def test_tests_run_in_english(monkeypatch):
    monkeypatch.delenv(LANG_ENV, raising=False)
    monkeypatch.setenv("LC_ALL", "zh_CN.UTF-8")
    assert locale() == "en"  # pytest is detected, so assertions stay stable


def test_english_is_byte_identical(monkeypatch):
    monkeypatch.setenv(LANG_ENV, "en")
    for key in _catalog():
        assert t(key) == key
    assert t("{title} ({kind})", title="Qwen", kind="mlx") == "Qwen (mlx)"


def test_every_entry_is_translated_and_keeps_its_fields(monkeypatch):
    monkeypatch.setenv(LANG_ENV, "zh")
    fields = re.compile(r"\{(\w+)(?:![rsa])?(?::[^}]*)?\}")
    for key, value in _catalog().items():
        assert value.strip(), key
        assert value != key, f"untranslated entry: {key!r}"
        assert set(fields.findall(value)) == set(fields.findall(key)), key
    assert t(", rank {rank} of 2", rank=1) == "，rank 1/2"


def test_no_entry_leaks_a_placeholder_format(monkeypatch):
    monkeypatch.setenv(LANG_ENV, "zh")
    for key in _catalog():
        t(key)  # a value whose placeholders do not match its key raises here when fields are passed
    with pytest.raises(KeyError):
        t("{missing}", other=1)


def test_literal_call_sites_are_translated():
    assert imports_t(ROOT / "cli.py")
    missing = [(path, line, key) for path in modules() if imports_t(path)
               for line, key in call_site_keys(path) if key not in _catalog()]
    assert not missing, "call sites with no catalog entry: " + repr(missing)


def test_front_end_modules_translate_their_user_facing_text():
    for name in FRONT_END:
        assert imports_t(ROOT / name), name


def test_values_translated_at_display_time_are_covered(monkeypatch):
    """States, palette actions and session labels reach t() through a variable, so check them by value."""

    monkeypatch.setenv(LANG_ENV, "zh")
    from tensorfold.control.view import _ACTION_TITLES, _HELP_ROWS, actions

    for state in ("ready", "warming", "unhealthy", "unauthorized", "running", "stopped", "monitor-only",
                  "installed", "uninstalled", "loaded", "unknown", "live counter", "completed requests",
                  "not sampled", "monitor only"):
        assert t(state) != state, state
    for name, label in actions():
        assert label != _ACTION_TITLES[name], name
    for _, description in _HELP_ROWS:
        assert t(description) != description, description
    for verb in ("start", "stop", "restart", "install"):
        assert t(verb) != verb, verb
    for mode in ("DEMO / SIMULATED", "PAUSED", "LIVE / READ-ONLY TELEMETRY"):
        assert t(mode) != mode, mode


def test_argparse_help_is_fully_translated():
    from tensorfold import cli
    from tensorfold.cli_args import build_parser
    from tensorfold.control.cli import parser as control_parser

    catalog = _catalog()
    handlers = {"serve": cli.cmd_serve, "pull": cli.cmd_pull, "models": cli.cmd_models, "update": cli.cmd_update,
                "info": cli.cmd_info, "plan": cli.cmd_plan}
    missing = sorted({text for text in (*parser_help_strings(build_parser(handlers)),
                                        *parser_help_strings(control_parser()))
                      if text not in catalog})
    assert not missing, f"help strings with no catalog entry: {missing}"


def test_usage_errors_are_translated(monkeypatch):
    monkeypatch.setenv(LANG_ENV, "zh")
    assert usage_error("the following arguments are required: model") == "缺少必需参数：model"
    assert usage_error("unrecognized arguments: --nope") == "无法识别的参数：--nope"
    assert usage_error("argument --backend: invalid choice: 'x' (choose from 'auto', 'mlx')") == \
        "参数 --backend：选择无效：'x'（可选：'auto', 'mlx'）"
    assert usage_error("argument --port: invalid int value: 'x'") == "参数 --port：int 值无效：'x'"
    assert usage_error("argument --tp: expected one argument") == "参数 --tp：需要一个值"
    monkeypatch.setenv(LANG_ENV, "en")
    assert usage_error("the following arguments are required: model") == "the following arguments are required: model"


def test_pad_counts_terminal_cells():
    assert pad("abcd", 6) == "abcd  "
    assert pad("模型", 6) == "模型  "      # two characters, four cells
    assert pad("模型", 4) == "模型"
    assert pad("abcdef", 3) == "abcdef"
    assert pad("", 2) == "  "


def test_parser_labels_and_usage_follow_the_locale(monkeypatch):
    from tensorfold.i18n.parser import Parser
    monkeypatch.setenv(LANG_ENV, "zh")
    parser = Parser(prog="demo", description="演示")
    parser.add_argument("model", help="一个模型")
    parser.add_argument("--count", type=int, default=1, help="数量")
    text = parser.format_help()
    assert "用法：" in text and "位置参数：" in text and "选项：" in text
    assert "usage:" not in text and "options:" not in text
    with pytest.raises(SystemExit):
        parser.parse_args([])
    monkeypatch.setenv(LANG_ENV, "en")
    plain = Parser(prog="demo")
    plain.add_argument("model")
    assert plain.format_help().startswith("usage: demo")


@pytest.mark.skipif(sys.version_info < (3, 11), reason="argparse wording")
def test_parser_reports_a_translated_error(monkeypatch, capsys):
    from tensorfold.i18n.parser import Parser
    monkeypatch.setenv(LANG_ENV, "zh")
    parser = Parser(prog="demo")
    parser.add_argument("--port", type=int)
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--port", "x"])
    assert exit_info.value.code == 2
    assert "参数 --port：int 值无效：'x'" in capsys.readouterr().err


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def run_tensorfold(*args: str, language: str) -> subprocess.CompletedProcess:
    """One real command line, so the catalogs are exercised through argparse and the terminal output."""

    environment = {**os.environ, LANG_ENV: language}
    return subprocess.run([sys.executable, "-m", "tensorfold", *args], capture_output=True, text=True,
                          env=environment, timeout=120, check=False)


def test_command_line_help_is_chinese_only_in_a_chinese_environment():
    english = run_tensorfold("--help", language="en")
    assert english.returncode == 0 and "usage: tensorfold" in english.stdout
    assert not has_cjk(english.stdout)
    chinese = run_tensorfold("--help", language="zh")
    assert chinese.returncode == 0 and "用法：tensorfold" in chinese.stdout
    assert "usage:" not in chinese.stdout and "options:" not in chinese.stdout
    serve = run_tensorfold("serve", "--help", language="zh")
    assert "位置参数：" in serve.stdout and "监听的地址" in serve.stdout


def test_command_line_usage_errors_are_chinese():
    failed = run_tensorfold("models", "--not-a-command", language="zh")
    assert failed.returncode == 2 and "无法识别的参数" in failed.stderr
    assert failed.stderr.startswith("tensorfold：错误：")
    failed = run_tensorfold("serve", "--backend", "nonsense", language="zh")
    assert failed.returncode == 2 and "选择无效" in failed.stderr


def test_tui_snapshot_is_chinese_in_a_chinese_environment(tmp_path):
    from tensorfold.control.cli import main
    plain = tmp_path / "en.txt"
    assert main(["tui", "--demo", "--snapshot", str(plain)]) == 0
    assert not has_cjk(plain.read_text())
    chinese = tmp_path / "zh.txt"
    os.environ[LANG_ENV] = "zh"
    try:
        assert main(["tui", "--demo", "--snapshot", str(chinese)]) == 0
    finally:
        os.environ.pop(LANG_ENV, None)
    text = chinese.read_text()
    assert "控制台" in text and "演示 / 模拟" in text and "输出历史" in text
    assert not any(label in text for label in ("CONTROL ROOM", "OUTPUT HISTORY", "SESSION", "DECODING", "LIVE LOG"))


def test_catalog_punctuation_and_spacing(monkeypatch):
    """The typography rules of GLOSSARY_zh.md section 3: full-width punctuation, spaced Latin, no '...'."""

    monkeypatch.setenv(LANG_ENV, "zh")
    ideograph = "\u4e00-\u9fff"
    literal = re.compile(r"“[^”]*[,:][^”]*”")           # half-width punctuation inside a parsed literal is kept
    checks = (("Chinese then Latin needs a space", re.compile(rf"[{ideograph}][A-Za-z0-9]")),
              ("Latin then Chinese needs a space", re.compile(rf"[A-Za-z0-9][{ideograph}]")),
              ("space after full-width punctuation", re.compile(r"[，。：；？！、）】》] ")),
              ("space before full-width punctuation", re.compile(r" [，。：；？！、（【《]")),
              ("ASCII ellipsis", re.compile(r"\.\.\.")))
    problems = []
    for key, value in _catalog().items():
        masked = literal.sub("「」", value)
        for label, pattern in checks:
            if found := pattern.search(masked):
                problems.append(f"{label}: {found.group(0)!r} in {key!r} -> {value!r}")
    assert not problems, "\n".join(problems)
