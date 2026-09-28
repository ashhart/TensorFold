"""A tool call written inside a think block that never closes reaches tool_calls (#60), on the Mac and CUDA servers,
streamed and not; a call only mentioned before the block closes stays reasoning."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from http_fakes import post
from tensorfold.server.text import CHANNEL_MARKERS, split_thinking

CALL = "<tool_call>\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
REASONING = "The user wants the files listed. I will run ls.\n"
TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}}}}]


def test_a_call_in_an_unclosed_block_is_the_answer_and_never_streams_as_reasoning():
    reply = REASONING + CALL
    sent = ""
    for n in range(1, len(reply) + 1):
        reasoning, answer = split_thinking(reply[:n], finished=False)
        assert reasoning.startswith(sent) and "<tool" not in reasoning and answer == ""
        sent = reasoning
    assert split_thinking(reply, finished=True) == (REASONING, CALL)
    assert sent == REASONING


def test_a_mention_before_the_block_closes_stays_reasoning():
    reply = REASONING + CALL + "\nNo, first check the path.\n</think>\n\nDone."
    assert split_thinking(reply, finished=True) == (REASONING + CALL + "\nNo, first check the path.\n", "Done.")
    assert split_thinking(REASONING + CALL[:-5], finished=True) == (REASONING + CALL[:-5], "")    # an unfinished call


def test_gemma_calls_inside_its_open_thought_channel():
    call = '<|tool_call>call:terminal{command:<|"|>ls<|"|>}<tool_call|>'
    reply = CHANNEL_MARKERS[0] + "\nList files.\n" + call
    assert split_thinking(reply, finished=True, markers=CHANNEL_MARKERS) == ("List files.\n", call)


# -- the Mac server: scripted tokens through ChatApp and the HTTP handler --------------------------------------------

PIECES = ["<p>", "<q>", "<a>", "<|im_end|>", REASONING, "<tool_call>", "\n<function=terminal>\n",
          "<parameter=command>\n", "ls\n</parameter>\n", "</function>\n", "</tool_call>"]
EOS = 3
SCRIPT = [4, 5, 6, 7, 8, 9, 10, EOS]


class ScriptTokenizer:
    eos_token_ids = {EOS}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        return [0, 1, 2]

    def decode(self, ids: list[int], **_: Any) -> str:
        return "".join(PIECES[int(t)] for t in ids)

    def encode(self, text: str, **_: Any) -> list[int]:
        return [PIECES.index(text)] if text in PIECES else []

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return PIECES.index(token) if token in PIECES else None


def script_app():
    pytest.importorskip("mlx.core")
    from tensorfold.server.app import ChatApp
    from tests.lane_fakes import FakeEngine, FakeFamily

    class ScriptFamily(FakeFamily):
        """Writes SCRIPT after the three-token prompt, whatever it is fed."""

        def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> Any:
            import mlx.core as mx
            import numpy as np

            history, out = cache[0].rows[0], []
            for token in np.array(inputs).reshape(-1).tolist():
                history.append(int(token))
                out.append(SCRIPT[min(len(history) - 3, len(SCRIPT) - 1)])
            return mx.array(out, dtype=mx.float32).reshape(1, -1, 1)

    family = ScriptFamily()
    return ChatApp(None, ScriptTokenizer(), served_name="fake", lanes=1, max_rows=16, max_draft=4,
                   default_max_tokens=32, checkpoint_slots=0, use_proposer=True, enable_thinking=True,
                   engine_factory=lambda model, **kw: FakeEngine(family, **kw))


def served_post(app, body):
    """A real socket: the Mac app watches it for a client that leaves."""

    import http.client
    from http.server import ThreadingHTTPServer

    from tensorfold.server.http import make_handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=30)
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        httpd.shutdown()
        httpd.server_close()


def events(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


@pytest.mark.parametrize("draft", [True, False])
def test_the_mac_server_returns_the_call(draft):
    app = script_app()
    try:
        status, body = served_post(app, {"messages": [{"role": "user", "content": "List files"}], "tools": TOOLS,
                                         "draft": draft})
        choice = json.loads(body)["choices"][0]
        assert status == 200 and choice["finish_reason"] == "tool_calls"
        assert choice["message"]["reasoning_content"] == REASONING.strip()
        assert [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in choice["message"]["tool_calls"]] == [("terminal", {"command": "ls"})]
        status, text = served_post(app, {"messages": [{"role": "user", "content": "List files"}], "tools": TOOLS,
                                         "draft": draft, "stream": True})
        chunks = events(text)
        deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        names = [d["tool_calls"][0]["function"]["name"] for d in deltas if d.get("tool_calls") and
                 d["tool_calls"][0].get("function", {}).get("name")]
        assert reasoning == REASONING and names == ["terminal"] and "<tool_call>" not in text
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert json.loads(body)["tensorfold"]["token_sha"] == chunks[-1]["tensorfold"]["token_sha"]
    finally:
        app.close()


# -- the CUDA server: an engine that writes the same reply ------------------------------------------------------------

def test_the_cuda_server_returns_the_call(tmp_path):
    import re

    from tensorfold.cuda import server
    from tests.test_cuda_admission import http_server, post as cuda_post

    special = {"<tool_call>": 1000, "</tool_call>": 1001}

    class Tokens:
        def encode(self, text, **kwargs):
            parts = re.split("(<tool_call>|</tool_call>)", text)
            return type("Encoding", (), {"ids": [i for p in parts for i in ([special[p]] if p in special
                                                                             else [ord(c) for c in p])]})()

        def decode(self, ids, **kwargs):
            names = {v: k for k, v in special.items()}
            return "".join(names.get(i, "" if i == 0 else chr(i)) for i in ids)

    class Engine:
        eos = (0,)

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            reply = Tokens().encode(REASONING + CALL).ids + [0]
            for at in range(0, len(reply), 4):
                on_tokens(reply[at:at + 4])
            return {"rounds": 1}

    def app_for(tmp_path, engine):
        template = "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:<think>"
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
        app = server.App.__new__(server.App)
        app.engine, app.served, app.tok = engine, "fake-cuda", Tokens()
        app.template = server.ChatTemplate(tmp_path)
        app.default_thinking, app.sampling, app.max_tokens = True, {"temperature": 0.0}, 256
        app.native_context_window = app.context_window = 0
        app.lock = threading.Lock()
        return app

    app = app_for(tmp_path, Engine())
    for stream in (False, True):
        with http_server(app) as port:
            status, body = cuda_post(port, {"messages": [{"role": "user", "content": "List files"}], "tools": TOOLS,
                                            "chat_template_kwargs": {"enable_thinking": True}, "stream": stream}, True)
        assert status == 200
        if stream:
            chunks = events(body)
            deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
            assert "".join(d.get("reasoning_content", "") for d in deltas) == REASONING
            assert [d["tool_calls"][0]["function"]["name"] for d in deltas if "tool_calls" in d] == ["terminal"]
            assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls" and "<tool_call>" not in body
        else:
            choice = json.loads(body)["choices"][0]
            assert choice["finish_reason"] == "tool_calls" and choice["message"]["reasoning_content"] == REASONING
            assert choice["message"]["tool_calls"][0]["function"]["name"] == "terminal"
    assert server.split_thinking is split_thinking


# -- chat_template_kwargs.reasoning_effort, as vLLM's clients send it ------------------------------------------------

class _App:
    served_name = "fake"
    model_ids = ["fake"]
    max_batch_size = 1
    exact_mode = {"mode": "exact"}
    accepts_sampling = True

    def chat(self, messages, **kwargs):
        self.sampling = kwargs.get("sampling")
        return {"content": "Hi", "finish_reason": "stop", "prompt_tokens": 3, "cached_tokens": 0,
                "completion_tokens": 1}


@pytest.mark.parametrize("body, effort", [({"chat_template_kwargs": {"reasoning_effort": "low"}}, "low"),
                                          ({"chat_template_kwargs": {"reasoning_effort": "high"}}, "xhigh"),
                                          ({"reasoning_effort": "medium",
                                            "chat_template_kwargs": {"reasoning_effort": "low"}}, "medium")])
def test_the_template_kwargs_effort_is_read(body, effort):
    app = _App()
    status, _ = post(app, {"messages": [{"role": "user", "content": "Hi"}], **body})
    assert status == 200 and app.sampling["reasoning_effort"] == effort and app.sampling["enable_thinking"]


def test_a_stream_ends_with_what_the_final_reply_adds():
    from tensorfold.server.stopping import StopPolicy

    sent = []
    policy = StopPolicy({}, None, threading.Lock(), frozenset())
    policy.flush(sent.append, "Done <too", "thinking </thi", "Done ", "thinking ", tools=False)
    assert sent == [{"reasoning_content": "</thi"}, "<too"]
