"""Local checkpoint routing and safe options for the CUDA sleep qualification tool."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.families import detect


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location(
        "qualify_sleep", Path(__file__).parents[1] / "tools/qualify_sleep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_family(tmp_path, kind):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": kind}))
    return detect(tmp_path)


def options():
    return SimpleNamespace(draft=None, no_drafts=False, parallel=1, context=1024, preserve_cache=False)


@pytest.mark.parametrize("kind,draft,no_drafts,expected", [
    ("qwen3_5", None, False, True),
    ("qwen3_5", Path("draft"), False, False),
    ("qwen3_5", None, True, True),
    ("nemotron_h", None, False, False),
    ("nemotron_h", None, True, True),
])
def test_family_draft_defaults(tool, tmp_path, kind, draft, no_drafts, expected):
    args = options()
    args.draft, args.no_drafts = draft, no_drafts
    got = tool.engine_options(args, local_family(tmp_path, kind))
    assert got["no_drafts"] is expected
    assert got["drafter"] == (str(draft.resolve()) if draft else "")


def test_parallel_qwen_reserves_each_preserved_prefix(tool, tmp_path):
    args = options()
    args.parallel, args.preserve_cache = 4, True
    got = tool.engine_options(args, local_family(tmp_path, "qwen3_5"))
    assert got["parallel"] == got["checkpoint_slots"] == 4


def test_parallel_nemotron_is_refused_before_cuda_import(tool, tmp_path, monkeypatch):
    local_family(tmp_path, "nemotron_h")
    args = options()
    args.parallel = 2
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="parallel.*dense Qwen"):
        tool.qualify(args, tmp_path)


def test_external_drafter_is_refused_for_integrated_mtp(tool, tmp_path):
    args = options()
    args.draft = Path("draft")
    with pytest.raises(ValueError, match="integrated MTP"):
        tool.engine_options(args, local_family(tmp_path, "nemotron_h"))


def test_unsupported_sleep_family_is_refused(tool, tmp_path):
    with pytest.raises(ValueError, match="sleep qualification"):
        tool.engine_options(options(), local_family(tmp_path, "qwen4_exp"))


def test_no_drafts_flag_reaches_qualification(tool, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tool, "qualify", lambda args, model, cache: calls.append((args, model, cache)))
    monkeypatch.setattr(sys, "argv", ["qualify_sleep", "--model", str(tmp_path), "--no-drafts",
                                     "--preserve-cache", "--output", str(tmp_path / "result.json")])
    tool.main()
    args, model, cache = calls[0]
    assert args.no_drafts and model == tmp_path and cache.name == "cache"


@pytest.mark.parametrize("flags", [["--draft", "draft"], ["--synthetic-draft"]])
def test_no_drafts_refuses_conflicting_drafter_flags(tool, tmp_path, monkeypatch, capsys, flags):
    monkeypatch.setattr(tool, "qualify", lambda *args: pytest.fail("loaded conflicting options"))
    monkeypatch.setattr(sys, "argv", ["qualify_sleep", "--synthetic", "--no-drafts", *flags,
                                     "--output", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit) as refused:
        tool.main()
    assert refused.value.code == 2
    error = capsys.readouterr().err
    assert "--no-drafts" in error and flags[0] in error
    assert "unrecognized arguments" not in error
