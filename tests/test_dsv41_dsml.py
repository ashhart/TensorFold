# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
# Adapted for tensorfold dsv41-cuda: a local renderer instead of the family's encoding module, more lenient cases,
# max_calls, unknown tools, random chunkings.
"""DSML replies: render -> parse round trips; streaming in any chunking gives the same reasoning, content and call
arguments as the whole parse; the lenient cases (calls in an unclosed think block, a call cut by max_tokens, V4-style
tags, typed values, unknown tools, NaN, repeated parameters, max_calls)."""

from __future__ import annotations

import json
import random
import re
import threading

import pytest

from tensorfold.families.deepseek_v41.cuda import dsml

D = dsml.DSML
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}}}}},
         {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
             "q": {"type": "string"}, "filters": {"type": "object"}, "top": {"type": "array"},
             "n": {"type": "integer"}}}}}]
CALLS = [{"function": {"name": "get_weather", "arguments": {"city": "Paris, \"FR\"", "days": 3}}},
         {"function": {"name": "search", "arguments": {"q": "naïve <b>", "filters": {"lang": "fr", "n": None},
                                                       "top": [1, 2]}}}]


def render_call(name, arguments):
    params = [f'<{D} parameter name="{k}" string="{"true" if isinstance(v, str) else "false"}">'
              f'{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</{D} parameter>'
              for k, v in arguments.items()]
    return f'<{D} invoke name="{name}">\n' + "\n".join(params) + f"\n</{D} invoke>"


def render_calls(calls):
    body = "\n".join(render_call(c["function"]["name"], c["function"]["arguments"]) for c in calls)
    return f"\n\n<{D} calls>\n{body}\n</{D} calls>"


def reply_text(reasoning, content, calls, thinking=True):
    head = (reasoning + dsml.THINK_END) if thinking else ""
    return head + content + (render_calls(calls) if calls else "")


def collect(stream: dsml.Stream, pieces) -> dict:
    out = {"reasoning": "", "content": "", "calls": {}, "events": []}
    text = ""
    events = []
    for p in pieces:
        text += p
        events += stream.feed(text)
    events += stream.finish(text)
    out["events"] = events
    for e in events:
        if "reasoning" in e:
            out["reasoning"] += e["reasoning"]
        if "content" in e:
            out["content"] += e["content"]
        for tc in e.get("tool_calls", []):
            c = out["calls"].setdefault(tc["index"], {"name": None, "args": "", "ids": set()})
            if "id" in tc:
                c["ids"].add(tc["id"])
                c["name"] = tc["function"]["name"]
            c["args"] += tc["function"]["arguments"]
    return out


def same(got: dict, whole: dsml.Reply) -> None:
    assert got["reasoning"] == whole.reasoning and got["content"] == whole.content
    assert [got["calls"][i]["name"] for i in sorted(got["calls"])] == [c.name for c in whole.calls]
    for i, c in enumerate(whole.calls):
        assert got["calls"][i]["args"] == c.arguments()                     # byte for byte, valid JSON
        json.loads(got["calls"][i]["args"])
        assert len(got["calls"][i]["ids"]) == 1
    if whole.calls:                                                          # the markup never reaches a client
        assert all(D not in e.get("content", "") for e in got["events"])


@pytest.mark.parametrize("thinking", [True, False])
def test_round_trip(thinking):
    text = reply_text("why", "", CALLS, thinking)
    r = dsml.parse(text, thinking=thinking, tools=TOOLS)
    assert [json.loads(c.arguments()) for c in r.calls] == [c["function"]["arguments"] for c in CALLS]
    assert r.content == "" and all(c.closed for c in r.calls)
    assert text.endswith(render_calls([c.openai() | {"function": {"name": c.name,
                                                                  "arguments": json.loads(c.arguments())}}
                                       for c in r.calls]))


@pytest.mark.parametrize("step", [1, 2, 3, 7])
def test_stream_chunks_equal_parse(step):
    text = reply_text("thinking about <｜DSML maybe", "Answer:\n\nhere", CALLS)
    whole = dsml.parse(text, thinking=True, tools=TOOLS)
    assert whole.reasoning == "thinking about <｜DSML maybe" and whole.content == "Answer:\n\nhere"
    same(collect(dsml.Stream(thinking=True, tools=TOOLS), [text[i:i + step] for i in range(0, len(text), step)]),
         whole)


def test_lenient_cases():
    # calls inside an unclosed think block
    t = "I should call.\n\n" + render_calls(CALLS[:1])[2:]
    r = dsml.parse(t, thinking=True, tools=TOOLS)
    assert r.reasoning == "I should call." and r.calls[0].name == "get_weather" and r.content == ""
    # cut by max_tokens inside the second parameter: the first one is kept, the call is not closed
    full = reply_text("", "", CALLS[:1])
    cut = full[:full.index('name="days"') + 15]
    r = dsml.parse(cut, thinking=True, tools=TOOLS)
    assert json.loads(r.calls[0].arguments()) == {"city": "Paris, \"FR\""} and not r.calls[0].closed
    # V4-style tags (no space, tool_calls), function_calls, and no blank line before the block
    for tag, sp in (("tool_calls", ""), ("function_calls", ""), ("calls", ""), (" calls", " ")):
        v4 = (f'</think>Sure.<｜DSML｜{tag}>\n<｜DSML｜{sp}invoke name="search">\n<｜DSML｜{sp}parameter name="q" '
              f'string="true">x</｜DSML｜{sp}parameter>\n</｜DSML｜{sp}invoke>\n</｜DSML｜{tag}>')
        r = dsml.parse(v4, thinking=True, tools=TOOLS)
        assert r.content == "Sure." and r.calls[0].name == "search" and r.calls[0].arguments() == '{"q":"x"}'
    # no tools: nothing is parsed
    assert not dsml.parse(reply_text("", "x", CALLS), thinking=True, tools=None).calls


def one(param: str, tool: str = "search") -> dsml.Call:
    text = f'</think><{D} calls>\n<{D} invoke name="{tool}">\n{param}\n</{D} invoke>\n</{D} calls>'
    calls = dsml.parse(text, thinking=True, tools=TOOLS).calls
    assert len(calls) == 1
    return calls[0]


def p(name, string, raw):
    return f'<{D} parameter name="{name}" string="{string}">{raw}</{D} parameter>'


def test_values():
    assert json.loads(one(p("days", "true", "4"), "get_weather").arguments()) == {"days": 4}       # typed by schema
    assert json.loads(one(p("city", "true", "4"), "get_weather").arguments()) == {"city": "4"}     # string stays
    assert json.loads(one(p("q", "true", "[1]")).arguments()) == {"q": "[1]"}
    assert json.loads(one(p("top", "false", "[1, 2")).arguments()) == {"top": [1, 2]}             # closed (#87)
    assert json.loads(one(p("filters", "false", '{"a":[1,2')).arguments()) == {"filters": {"a": [1, 2]}}
    assert json.loads(one(p("n", "false", "three")).arguments()) == {"n": "three"}                # not JSON: text
    assert json.loads(one(p("filters", "false", "{lang: fr}")).arguments()) == {"filters": "{lang: fr}"}
    assert json.loads(one(p("n", "false", "NaN")).arguments()) == {"n": "NaN"}                    # never NaN
    assert json.loads(one(p("top", "false", "[Infinity")).arguments()) == {"top": "[Infinity"}
    assert json.loads(one(p("zzz", "false", "true")).arguments()) == {"zzz": True}                # not in the schema
    # a repeated parameter: the first wins
    assert json.loads(one(p("q", "true", "a") + "\n" + p("q", "true", "b")).arguments()) == {"q": "a"}


def test_unknown_tools():
    unk = reply_text("", "x", [{"function": {"name": "rm_rf", "arguments": {}}}])
    r = dsml.parse(unk, thinking=True, tools=TOOLS)
    assert not r.calls and r.content == "x" + render_calls([{"function": {"name": "rm_rf", "arguments": {}}}])
    got = collect(dsml.Stream(thinking=True, tools=TOOLS), list(unk))
    same(got, r)
    mixed = reply_text("", "x", [{"function": {"name": "rm_rf", "arguments": {"a": 1}}}, CALLS[1]])
    r = dsml.parse(mixed, thinking=True, tools=TOOLS)
    assert [c.name for c in r.calls] == ["search"] and r.content == "x"
    same(collect(dsml.Stream(thinking=True, tools=TOOLS), list(mixed)), r)
    # a name in another case or with a namespace is the offered one
    assert dsml.parse(reply_text("", "", [{"function": {"name": "functions::Search", "arguments": {"q": "a"}}}]),
                      thinking=True, tools=TOOLS).calls[0].name == "search"


def test_max_calls():
    text = reply_text("r", "", [{"function": {"name": "rm_rf", "arguments": {}}}, *CALLS])
    r = dsml.parse(text, thinking=True, tools=TOOLS, max_calls=1)
    assert [c.name for c in r.calls] == ["get_weather"]
    got = collect(dsml.Stream(thinking=True, tools=TOOLS, max_calls=1), list(text))
    same(got, r)
    assert len(got["calls"]) == 1


def test_think_block_cases():
    # a complete block before </think> was ever written, streamed or not
    t = "plan\n\n" + render_calls(CALLS)[2:]
    r = dsml.parse(t, thinking=True, tools=TOOLS)
    assert r.reasoning == "plan" and len(r.calls) == 2
    same(collect(dsml.Stream(thinking=True, tools=TOOLS), list(t)), r)
    # cut: an invoke left open inside an unclosed think block
    cut = t[:t.index("naïve") + 3]
    r = dsml.parse(cut, thinking=True, tools=TOOLS)
    assert r.reasoning == "plan" and [c.closed for c in r.calls] == [True, False]
    assert json.loads(r.calls[1].arguments()) == {}
    same(collect(dsml.Stream(thinking=True, tools=TOOLS), list(cut)), r)
    # blank lines after </think> are dropped, as split_thinking drops them
    t = "why</think>\n\nHello"
    r = dsml.parse(t, thinking=True, tools=TOOLS)
    assert (r.reasoning, r.content) == ("why", "Hello")
    same(collect(dsml.Stream(thinking=True, tools=TOOLS), list(t)), r)


def texts():
    calls = [{"function": {"name": "rm_rf", "arguments": {"x": "y"}}}, *CALLS]
    yield reply_text("think <b> \n</thin", "Answer:\n\n here\n", CALLS)
    yield reply_text("", "Answer", CALLS, thinking=False)
    yield reply_text("r", "", calls[:1])
    yield reply_text("r\n", "\n", calls)
    yield "a\n\n\n" + render_calls(CALLS)[2:]
    yield "a\n" + render_calls(CALLS)[2:]
    full = reply_text("r", "c", CALLS)
    for k in range(len(full) - 60, len(full), 7):
        yield full[:k]


@pytest.mark.parametrize("seed", range(40))
def test_random_chunkings_equal_parse(seed):
    rng = random.Random(seed)
    for text in texts():
        for thinking in (True, False):
            for max_calls in (None, 1):
                whole = dsml.parse(text, thinking=thinking, tools=TOOLS, max_calls=max_calls)
                pieces, at = [], 0
                while at < len(text):
                    n = rng.randint(1, 9)
                    pieces.append(text[at:at + n])
                    at += n
                same(collect(dsml.Stream(thinking=thinking, tools=TOOLS, max_calls=max_calls), pieces), whole)


# -- through the CUDA server (App.run behind the HTTP handler) -------------------------------------------------------

EOS, THINK, END, MARK = 0, 1002, 1003, 1004
SPECIAL = {"<think>": THINK, "</think>": END, D: MARK}


class Tokens:
    """Characters as their code points; <think>, </think> and ｜DSML｜ as single tokens (V4.1's tokenizer has them)."""

    def encode(self, text, **kwargs):
        ids = []
        for part in re.split("(" + "|".join(re.escape(s) for s in SPECIAL) + ")", text):
            ids += [SPECIAL[part]] if part in SPECIAL else [ord(c) for c in part]
        return type("Encoding", (), {"ids": ids})()

    def decode(self, ids, **kwargs):
        names = {v: k for k, v in SPECIAL.items()}
        return "".join(names.get(i, "" if i == EOS else chr(i)) for i in ids)

    def token_to_id(self, text):
        return SPECIAL.get(text)


class Engine:
    """Writes ``reply`` (after its reasoning when the prompt opened a think block), three tokens a round."""

    eos = (EOS,)

    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        text = ("let me look it up</think>\n\n" if prompt[-1] == THINK else "") + self.reply
        ids = (Tokens().encode(text).ids + [EOS])[:max_tokens]
        for at in range(0, len(ids), 3):
            if on_tokens(ids[at:at + 3]):
                break
        return {"rounds": 1 + len(ids) // 3, "decode_s": 0.5}


def dsml_app(tmp_path, reply):
    from tensorfold.cuda import server

    template = ("{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}"
                "{% if tools %}tools:{{ tools | map(attribute='function.name') | join(',') }};{% endif %}"
                "assistant:{% if enable_thinking %}<think>{% endif %}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = Engine(reply), "fake-dsv41", Tokens()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking = False
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 4096
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def served(tmp_path, reply, stream, **body):
    """(finish_reason, reasoning, content, [(name, arguments text)]) as a client reads them, and the raw body."""

    from tests.test_cuda_admission import http_server, post

    body = {"messages": [{"role": "user", "content": "Weather?"}], "tools": TOOLS, "stream": stream, **body}
    with http_server(dsml_app(tmp_path, reply)) as port:
        status, raw = post(port, body, True)
    assert status == 200, raw
    if not stream:
        choice = json.loads(raw)["choices"][0]
        message = choice["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []]
        return (choice["finish_reason"], message.get("reasoning_content") or "", message["content"] or "", calls), raw
    chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    reasoning = content = ""
    names, args = {}, {}
    for chunk in chunks:
        delta = chunk["choices"][0]["delta"]
        reasoning += delta.get("reasoning_content", "")
        content += delta.get("content") or ""
        for d in delta.get("tool_calls", []):
            names[d["index"]] = names.get(d["index"], "") + d["function"].get("name", "")
            args[d["index"]] = args.get(d["index"], "") + d["function"].get("arguments", "")
    calls = [(names[i], args[i]) for i in sorted(names)]
    return (chunks[-1]["choices"][0]["finish_reason"], reasoning, content, calls), raw


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("parallel", [True, False])
def test_server_streamed_equals_whole(tmp_path, thinking, parallel):
    reply = "Checking." + render_calls(CALLS)
    body = {"chat_template_kwargs": {"enable_thinking": thinking}, "parallel_tool_calls": parallel}
    whole, _ = served(tmp_path, reply, False, **body)
    streamed, raw = served(tmp_path, reply, True, **body)
    assert whole == streamed
    finish, reasoning, content, calls = whole
    assert finish == "tool_calls" and content == "Checking."
    assert reasoning == ("let me look it up" if thinking else "")
    wanted = [(c["function"]["name"], c["function"]["arguments"]) for c in CALLS][:None if parallel else 1]
    assert [(n, json.loads(a)) for n, a in calls] == wanted
    assert D not in raw                                          # no markup leaks, the second call included


def test_server_cut_call_finishes_with_length(tmp_path):
    reply = render_calls(CALLS)
    cut = len(Tokens().encode(reply[:reply.index("naïve") + 2]).ids)
    whole, _ = served(tmp_path, reply, False, max_tokens=cut)
    streamed, _ = served(tmp_path, reply, True, max_tokens=cut)
    assert whole == streamed
    finish, _, content, calls = whole
    assert finish == "length" and content == "" and [n for n, _ in calls] == ["get_weather", "search"]
    assert json.loads(calls[1][1]) == {}                         # the cut call's complete parameters only


def test_server_unknown_tool_stays_content(tmp_path):
    reply = "Sure." + render_calls([{"function": {"name": "rm_rf", "arguments": {"path": "/"}}}])
    whole, _ = served(tmp_path, reply, False)
    streamed, _ = served(tmp_path, reply, True)
    assert whole == streamed == ("stop", "", reply, [])


def test_server_without_dsml_token_keeps_the_shared_parser(tmp_path):
    from tests.test_cuda_tool_choice import Tokens as Plain

    app = dsml_app(tmp_path, "hi")
    app.tok = Plain()
    assert app._dsml is False and dsml_app(tmp_path, "hi")._dsml is True


# -- TF_DSV41_TOOL_GRAMMAR: required / named / strict tools as a grammar (xgrammar itself: test_grammar_tools.py) ----

class GrammarEngine(Engine):
    def __init__(self, reply):
        super().__init__(reply)
        self.constraints = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
        self.constraints.append(constraint)
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft)


class FakeGrammars:
    """Records what is compiled; a constraint is (spec, think_end)."""

    def __init__(self):
        self.specs = []

    def compile(self, spec):
        self.specs.append(spec)
        return spec

    def constraint(self, compiled, *, think_end=None, spec=None):
        return (spec, think_end)


def grammar_app(tmp_path, reply):
    app = dsml_app(tmp_path, reply)
    app.engine = GrammarEngine(reply)
    app.grammars = FakeGrammars()
    app._call_gate = lambda *a: pytest.fail("the grammar writes the call: no call gate")
    return app


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "search"}}])
def test_a_required_call_is_a_tools_grammar(tmp_path, monkeypatch, thinking, choice):
    from tests.test_cuda_admission import http_server, post

    monkeypatch.setenv("TF_DSV41_TOOL_GRAMMAR", "required")
    app = grammar_app(tmp_path, render_calls(CALLS[1:]))
    body = {"messages": [{"role": "user", "content": "Find it."}], "tools": TOOLS, "tool_choice": choice,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    with http_server(app) as port:
        status, raw = post(port, body, True)
    assert status == 200, raw
    reply = json.loads(raw)["choices"][0]
    assert reply["finish_reason"] == "tool_calls" and reply["message"]["tool_calls"][0]["function"]["name"] == "search"
    (spec, think_end), = app.engine.constraints
    assert spec.kind == "tools" and think_end == (END if thinking else None)        # after </think> when thinking
    wanted = ["search"] if isinstance(choice, dict) else ["get_weather", "search"]
    assert [t["function"]["name"] for t in json.loads(spec.text)["tools"]] == wanted


def test_tool_grammars_are_off_by_default(tmp_path, monkeypatch):
    from tests.test_cuda_admission import http_server, post

    monkeypatch.delenv("TF_DSV41_TOOL_GRAMMAR", raising=False)
    app = grammar_app(tmp_path, render_calls(CALLS[1:]))
    app._call_gate = lambda *a: (_ for _ in ()).throw(server_request_error())
    strict = [dict(TOOLS[0], function=dict(TOOLS[0]["function"], strict=True))]
    with http_server(app) as port:
        status, _ = post(port, {"messages": [{"role": "user", "content": "x"}], "tools": strict}, True)
        assert status == 200 and app.engine.constraints == [None] and app.grammars.specs == []
        status, raw = post(port, {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS,
                                  "tool_choice": "required"}, True)
    assert status == 400 and "gate refused" in raw                               # today's answer, unchanged


def test_all_holds_every_tools_request_and_engines_without_grammars_refuse(tmp_path, monkeypatch):
    from tests.test_cuda_admission import http_server, post

    monkeypatch.setenv("TF_DSV41_TOOL_GRAMMAR", "all")
    app = grammar_app(tmp_path, "Hi.")
    with http_server(app) as port:
        status, _ = post(port, {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS}, True)
    assert status == 200 and app.engine.constraints[0][0].kind == "tools"
    assert all(t["function"]["strict"] for t in json.loads(app.engine.constraints[0][0].text)["tools"])
    plain = dsml_app(tmp_path, "Hi.")                                           # generate() takes no constraint
    with http_server(plain) as port:
        status, raw = post(port, {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS}, True)
    assert status == 400 and "tool-call grammars" in raw


def server_request_error():
    from tensorfold.server.errors import RequestError

    return RequestError("gate refused")
