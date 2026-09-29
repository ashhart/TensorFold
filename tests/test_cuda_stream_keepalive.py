"""A streamed reply that is writing a tool call sends empty deltas, so a client's read timeout never fires mid-call."""

from __future__ import annotations

import json
import re
import threading

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tensorfold.cuda import server
from tests.test_cuda_admission import http_server, post

SPECIAL = {"<tool_call>": 1000, "</tool_call>": 1001}
TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {
    "type": "object", "properties": {"content": {"type": "string"}}}}}]
BODY = "x" * 400
REPLY = f"Writing it.\n<tool_call>\n<function=write>\n<parameter=content>\n{BODY}\n</parameter>\n</function>\n</tool_call>"


class Tokens:
    def encode(self, text, **kwargs):
        parts = re.split("(<tool_call>|</tool_call>)", text)
        return type("Encoding", (), {"ids": [i for p in parts for i in ([SPECIAL[p]] if p in SPECIAL
                                                                         else [ord(c) for c in p])]})()

    def decode(self, ids, **kwargs):
        names = {v: k for k, v in SPECIAL.items()}
        return "".join(names.get(i, "" if i == 0 else chr(i)) for i in ids)


class Engine:
    eos = (0,)

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        reply = Tokens().encode(REPLY).ids + [0]
        for at in range(0, len(reply), 4):
            on_tokens(reply[at:at + 4])
        return {"rounds": 1}


def app_for(tmp_path):
    template = "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = Engine(), "fake-cuda", Tokens()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking, app.sampling, app.max_tokens = False, {"temperature": 0.0}, 2048
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def streamed(tmp_path):
    with http_server(app_for(tmp_path)) as port:
        status, body = post(port, {"messages": [{"role": "user", "content": "Write it"}], "tools": TOOLS,
                                   "stream": True}, True)
    assert status == 200 and "<tool_call>" not in body
    chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    calls = [d["tool_calls"][0]["function"] for d in deltas if "tool_calls" in d]
    assert "".join(d.get("content", "") for d in deltas).strip() == "Writing it."
    assert [(c["name"], json.loads(c["arguments"])) for c in calls] == [("write", {"content": BODY})]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    return [c["choices"][0]["delta"] for c in chunks if c.get("choices") and not c["choices"][0]["finish_reason"]]


def test_a_hidden_tool_call_sends_empty_deltas_while_it_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "KEEPALIVE_SECONDS", 0.0)
    assert sum(d == {} for d in streamed(tmp_path)) > 10


def test_a_reply_faster_than_the_keepalive_sends_none(tmp_path):
    assert {} not in streamed(tmp_path)
