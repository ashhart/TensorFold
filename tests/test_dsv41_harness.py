"""The soak and stress clients (tools/dsv41_soak.py, tools/dsv41_stress.py) against the CUDA server over a fake engine
with V4.1's DSML tokens: the key comes from TF_API_KEY, greedy repeats share a token_sha, tool calls arrive with no
markup, the server drains; completions take a prompt as token ids (what the stress client sends)."""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

import pytest

from tests.test_cuda_admission import http_server, post
from tests.test_dsv41_dsml import CALLS, EOS, THINK, Tokens, dsml_app, render_calls

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
KEY = "sk-harness-test"


def tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Engine:
    """Deterministic replies: 391 for 17*23, a DSML call when tools are offered (the prompt names them), reasoning
    first when the prompt opened a think block, else ``max_tokens`` letters (as ignore_eos decodes)."""

    eos = (EOS,)

    def __init__(self):
        self.lock = threading.Lock()

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        text = Tokens().decode(prompt)
        if "17*23" in text:
            reply = "391"
        elif "tools:" in text:
            reply = "Checking." + render_calls(CALLS[:1])
        elif prompt[-1] == THINK:
            reply = "count them</think>\n\n21 weekdays."
        elif "assistant:" in text:
            reply = "A short answer about " + text[-30:].replace(";", "")
        else:
            reply = "".join(chr(97 + (i % 26)) for i in range(max_tokens))
        ids = (Tokens().encode(reply).ids + ([EOS] if "assistant:" in text else []))[:max_tokens]
        for at in range(0, len(ids), 4):
            if on_tokens(ids[at:at + 4]):
                break
        return {"rounds": 1 + len(ids) // 4}


def app_with_key(tmp_path):
    app = dsml_app(tmp_path, "")
    app.engine = Engine()
    app.api_key = KEY
    app.served = "DeepSeek-v4.1-Flash-EXL3"
    app.sampling = {"temperature": 0.0, "top_k": 20, "top_p": 0.95}
    app.native_context_window = app.context_window = 0
    return app


def test_soak_passes_against_a_healthy_server(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_API_KEY", KEY)
    soak = tool("dsv41_soak")
    out = tmp_path / "soak.json"
    with http_server(app_with_key(tmp_path)) as port:
        rc = soak.main(["--base", f"http://127.0.0.1:{port}", "--minutes", "0.05", "--drain-s", "10",
                        "--kinds", "code,prose,tool,thinking,nonstream", "--cancel-p", "0", "--out", str(out)])
    report = json.loads(out.read_text())
    assert rc == 0, {k: report[k] for k in ("errors", "check_fails", "error_samples", "check_samples", "sha_splits")}
    assert report["requests"] > 0 and report["errors"] == 0 and report["check_fails"] == 0
    assert report["drained"] and "391" in report["answer_17x23"] and report["greedy_bodies"] > 0
    assert report["by_kind"]["tool"]["n"] == 0 or report["by_kind"]["tool"]["check_fails"] == 0


def test_soak_fails_without_the_key_and_flags_split_replies(tmp_path, monkeypatch):
    soak = tool("dsv41_soak")
    monkeypatch.delenv("TF_API_KEY", raising=False)
    with http_server(app_with_key(tmp_path)) as port:
        rc = soak.main(["--base", f"http://127.0.0.1:{port}", "--minutes", "0.02", "--drain-s", "5",
                        "--kinds", "nonstream", "--out", str(tmp_path / "a.json")])
    report = json.loads((tmp_path / "a.json").read_text())
    assert rc == 1 and report["errors"] > 0 and "HTTP 401" in report["error_samples"][0]["error"]
    s = soak.Soak(soak.argparse.Namespace(kinds="", seed=1))
    s.shas = {"body": {"aaa", "bbb"}}
    s.rec, s.health, s.a.minutes, s.a.base, s.a.model = [], [], 0, "http://127.0.0.1:9", "m"
    assert s.report(True, 0)["sha_splits"] == 1 and s.report(True, 0)["pass"] is False


def test_soak_check_reads_streamed_calls_and_markup():
    check = tool("dsv41_soak").Soak.check
    assert check("tool", True, "", "", 1, {0: '{"city":"Oslo"}'}) is None
    assert "not JSON" in check("tool", True, "", "", 1, {0: '{"city":'})
    assert "markup" in check("prose", True, "x", "<｜DSML｜ calls>", 0, {})
    assert check("tool", False, "", "", 0, {}) == "no tool call"


def test_stress_sends_token_ids_and_decodes_to_max_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_API_KEY", KEY)
    stress = tool("dsv41_stress")
    out = tmp_path / "stress.json"
    with http_server(app_with_key(tmp_path)) as port:
        rc = stress.main(["stress", "--base", f"http://127.0.0.1:{port}", "--long", "300", "--dctx", "100",
                          "--decode", "20", "--stagger", "0", "--vocab-lo", "97", "--vocab-hi", "123",
                          "--require-full", "--out", str(out)])
        assert stress.main(["admit", "--base", f"http://127.0.0.1:{port}", "--tokens", "50", "--vocab-lo", "97",
                            "--vocab-hi", "123"]) == 0
    report = json.loads(out.read_text())
    assert rc == 0 and report["decode_full"] and [s["prompt"] for s in report["streams"]] == [300, 100, 100, 100]
    assert all(s["tokens"] == 20 for s in report["streams"][1:])


def test_completions_take_a_prompt_of_token_ids(tmp_path):
    app = app_with_key(tmp_path)
    app.api_key = ""
    with http_server(app) as port:
        status, body = post(port, {"prompt": [104, 105], "max_tokens": 3, "temperature": 0}, False)
        assert status == 200 and json.loads(body)["usage"]["prompt_tokens"] == 2
        for bad in ([], [1, -2], [1.5], ["a"], [True]):
            status, body = post(port, {"prompt": bad, "max_tokens": 3}, False)
            assert status == 400 and "token ids" in body, bad
    app.tok.get_vocab_size = lambda with_added_tokens=True: 200
    with pytest.raises(Exception, match="below 200"):
        app._token_ids([1, 200])
    assert app._token_ids([0, 199]) == [0, 199]


def test_structured_bodies_ask_for_token_ids_and_the_key_stays_off_the_command_line(monkeypatch):
    structured = tool("dsv41_structured")
    a = structured.argparse.Namespace(model="m")
    body = structured.body_for(a, "hi", schema=structured.SCHEMAS[0], thinking=True)
    assert body["return_token_ids"] is True and body["response_format"]["type"] == "json_schema"
    monkeypatch.setenv("TF_API_KEY", KEY)
    assert structured.headers()["Authorization"] == f"Bearer {KEY}"
    for name in ("dsv41_soak", "dsv41_stress", "dsv41_structured"):
        assert "--api-key" not in (TOOLS_DIR / f"{name}.py").read_text()
