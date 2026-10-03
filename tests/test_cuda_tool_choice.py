"""The CUDA server enforces tool_choice "required" (#52) the way the Mac engine does: the answer's first word becomes
the tool-call opener, here by stopping the engine at the cut and decoding on from the reply and the opener."""

import json
import re
import threading

import pytest

from tensorfold.cuda import server
from tests.test_cuda_admission import http_server, post

EOS, OPEN, CLOSE, THINK, END = 0, 1000, 1001, 1002, 1003
SPECIAL = {"<tool_call>": OPEN, "</tool_call>": CLOSE, "<think>": THINK, "</think>": END}
CALL = "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n"
TOOLS = [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
         for name in ("get_weather", "search")]


class Tokens:
    """Characters as their code points, the markup as single tokens."""

    def encode(self, text, **kwargs):
        ids, parts = [], re.split("(" + "|".join(re.escape(s) for s in SPECIAL) + ")", text)
        for part in parts:
            ids += [SPECIAL[part]] if part in SPECIAL else [ord(c) for c in part]
        return type("Encoding", (), {"ids": ids})()

    def decode(self, ids, **kwargs):
        names = {v: k for k, v in SPECIAL.items()}
        return "".join(names.get(i, "" if i == EOS else chr(i)) for i in ids)

    def token_to_id(self, text):
        return SPECIAL.get(text)


class Engine:
    """Writes a call after the opener (to ``tool``), else its reasoning (when the prompt opened a block) then prose,
    three tokens a round; ``stops`` says whether it honors on_tokens asking it to stop, as Qwen's does and GLM's not."""

    eos = (EOS,)

    def __init__(self, stops=True, tool="get_weather"):
        self.stops, self.calls, self.tool = stops, [], tool

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.calls.append((list(prompt), max_tokens, draft))
        written = Tokens().decode(prompt)
        if written.endswith("<function=get_weather>"):
            reply = Tokens().encode("\n<parameter=city>\nOslo\n</parameter>\n</function>\n").ids + [CLOSE, EOS]
        elif written.endswith("<function="):
            reply = Tokens().encode(CALL.replace("get_weather", self.tool)[len("\n<function="):]).ids + [CLOSE, EOS]
        elif prompt[-1] == OPEN:
            reply = Tokens().encode(CALL.replace("get_weather", self.tool)).ids + [CLOSE, EOS]
        elif prompt[-1] == THINK:
            reply = [ord(c) for c in "the user wants hi"] + [END] + [ord("\n")] * 2 + [ord(c) for c in "Hi!"] + [EOS]
        else:
            reply = [ord(c) for c in "Hello! How can I help?"] + [EOS]
        reply = reply[:max_tokens]
        for at in range(0, len(reply), 3):
            if on_tokens(reply[at:at + 3]) and self.stops:
                break
        return {"rounds": 1 + len(reply) // 3, "decode_s": 0.5}


def app_for(tmp_path, engine):
    template = ("{% for m in messages %}{{ m.role }}:{{ m.content }}{% for c in m.tool_calls or [] %}<tool_call>\n"
                "<function={{ c.function.name }}>\n</function>\n</tool_call>{% endfor %};{% endfor %}"
                "{% if tools %}tools:{{ tools | map(attribute='function.name') | join(',') }};{% endif %}"
                "assistant:{% if enable_thinking %}<think>{% endif %}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = engine, "fake-cuda", Tokens()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking = False
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 256
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def ask(app, stream=False, **body):
    body = {"messages": [{"role": "user", "content": "Say hello."}], "tools": TOOLS, "tool_choice": "required",
            "stream": stream, **body}
    with http_server(app) as port:
        return post(port, body, True)


def events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


@pytest.mark.parametrize("stops", [True, False])
@pytest.mark.parametrize("thinking", [False, True])
def test_a_required_call_is_written_and_drafted_equals_serial(tmp_path, stops, thinking):
    replies = []
    for draft in (True, False):
        engine = Engine(stops)
        status, body = ask(app_for(tmp_path, engine), draft=draft,
                           chat_template_kwargs={"enable_thinking": thinking})
        assert status == 200
        payload = json.loads(body)
        choice = payload["choices"][0]
        assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] is None
        assert [c["function"]["name"] for c in choice["message"]["tool_calls"]] == ["get_weather"]
        first, then = engine.calls
        kept = then[0][len(first[0]):]                 # what the second run starts from: the reply to the cut
        lead = Tokens().encode("\n<function=").ids
        assert kept[-len(lead) - 1:] == [OPEN, *lead] and first[2] == then[2] == draft
        assert kept == ([ord(c) for c in "the user wants hi"] + [END, 10, 10] if thinking else []) + [OPEN, *lead]
        replies.append((payload["tensorfold"]["token_sha"], payload["usage"]["completion_tokens"]))
    assert replies[0] == replies[1]


def test_a_streamed_required_call_arrives_as_tool_call_deltas(tmp_path):
    status, body = ask(app_for(tmp_path, Engine()), stream=True)
    chunks = events(body)
    deltas = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
              if c.get("choices") and "tool_calls" in c["choices"][0]["delta"]]
    assert status == 200 and chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    # the arguments stream as they are written, after one header that names the call
    assert [d["function"]["name"] for d in deltas if "name" in d["function"]] == ["get_weather"]
    assert len(deltas) > 2 and {d["index"] for d in deltas} == {0}
    assert json.loads("".join(d["function"]["arguments"] for d in deltas)) == {"city": "Oslo"}
    assert "Hello" not in body and "<tool_call>" not in body


def test_a_streamed_single_call_request_sends_its_call_whole(tmp_path):
    status, body = ask(app_for(tmp_path, Engine()), stream=True, parallel_tool_calls=False)
    deltas = [c["choices"][0]["delta"]["tool_calls"][0] for c in events(body)
              if c.get("choices") and "tool_calls" in c["choices"][0]["delta"]]
    assert status == 200 and [d["function"]["name"] for d in deltas] == ["get_weather"]
    assert json.loads(deltas[0]["function"]["arguments"]) == {"city": "Oslo"}


def test_auto_leaves_the_reply_alone(tmp_path):
    engine = Engine()
    status, body = ask(app_for(tmp_path, engine), tool_choice="auto")
    assert status == 200 and len(engine.calls) == 1
    assert json.loads(body)["choices"][0]["message"]["content"] == "Hello! How can I help?"


def test_tool_choice_shapes_the_tools_offered(tmp_path):
    app = app_for(tmp_path, Engine())
    named = {"type": "function", "function": {"name": "search"}}

    def render(choice):
        body = {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS, "tool_choice": choice}
        return app.tok.decode(app._prepare(body, True).prompt)

    assert "tools:get_weather,search;" in render("required")
    assert "tools:search;" in render(named) and "tools:" not in render("none")
    status, body = ask(app, tools=[])
    assert status == 400 and "offers no tools" in json.loads(body)["error"]["message"]


def test_a_tool_the_request_did_not_offer_becomes_the_offered_one(tmp_path):
    engine = Engine(tool="update_plan")                # the model reaches for a tool it was not given
    status, body = ask(app_for(tmp_path, engine), tools=TOOLS[:1])
    choice = json.loads(body)["choices"][0]
    assert status == 200 and choice["finish_reason"] == "tool_calls"
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in choice["message"]["tool_calls"]] \
        == [("get_weather", {"city": "Oslo"})]
    assert len(engine.calls) == 3                      # cut at the prose, then at the name "u..."
