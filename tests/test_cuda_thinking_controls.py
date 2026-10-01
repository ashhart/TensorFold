"""reasoning_effort, enable_thinking and thinking_budget on the CUDA server, read as the Mac server reads them: the
effort reaches the template (as the template names it), "none" turns thinking off, and the thinking budget's close
lands where the lane engine puts it, whatever the engine's round widths."""

import json
import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda import server
from tests.test_cuda_admission import http_server, post
from tests.test_cuda_tool_choice import CLOSE, END, EOS, THINK, TOOLS, Tokens, events
from tests.test_cuda_tool_choice import app_for as tool_app

QWEN = "{# 'xhigh' 'medium' 'low' #}"          # the efforts a template names (Qwen3.8's; GLM-5.3's are low and high)
GLM = "{# 'low' 'high' #}"
TEMPLATE = ("{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}"
            "{% if reasoning_effort is defined %}effort={{ reasoning_effort }};{% endif %}"
            "{% if tools %}tools;{% endif %}assistant:{% if enable_thinking %}<think>{% endif %}")
LETTERS = "abcdefghij"


def after(token):
    """The chain the fake engine writes: each token the next letter of ten, from any token."""

    return ord(LETTERS[(int(token) * 7 + 3) % 10])


class ChainEngine:
    """Writes ``after`` of its last token, ``width`` tokens a round (1: the serial reference), a think end at reply
    index ``end_at``; records every prompt."""

    eos = (EOS,)

    def __init__(self, width=3, end_at=None):
        self.width, self.end_at, self.prompts = width, end_at, []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        last, reply = prompt[-1], []
        while len(reply) < max_tokens:
            last = END if len(reply) == self.end_at else after(last)
            reply.append(last)
        for at in range(0, len(reply), self.width if draft else 1):
            if on_tokens(reply[at:at + (self.width if draft else 1)]):
                break
        return {"rounds": 1}


def app_for(tmp_path, engine, names=QWEN, effort=None, budget=0, thinking=True):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": names + TEMPLATE}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = engine, "fake-cuda", Tokens()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking, app.reasoning_effort, app.thinking_budget = thinking, effort, budget
    app.sampling, app.max_tokens = {"temperature": 0.0}, 24
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def ask(app, stream=False, **fields):
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": stream, "return_token_ids": True, **fields}
    with http_server(app) as port:
        status, text = post(port, body, True)
    if status != 200 or not stream:
        return status, json.loads(text)
    chunks = [c for c in events(text) if c.get("choices")]
    deltas = [c["choices"][0]["delta"] for c in chunks]
    return status, {"reasoning": "".join(d.get("reasoning_content", "") for d in deltas),
                    "content": "".join(d.get("content", "") for d in deltas), "tensorfold": chunks[-1]["tensorfold"]}


def rendered(engine):
    text = Tokens().decode(engine.prompts[0])
    return text[text.index(";") + 1:]


@pytest.mark.parametrize("fields, names, want", [
    ({"reasoning_effort": "high"}, QWEN, "effort=xhigh;assistant:<think>"),
    ({"reasoning_effort": "minimal"}, QWEN, "effort=low;assistant:<think>"),
    ({"reasoning_effort": "medium"}, QWEN, "effort=medium;assistant:<think>"),
    ({"chat_template_kwargs": {"reasoning_effort": "high"}}, QWEN, "effort=xhigh;assistant:<think>"),
    ({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}}, QWEN,
     "effort=low;assistant:<think>"),
    ({"reasoning_effort": "high"}, GLM, "effort=high;assistant:<think>"),     # a template's own "high" is kept
    ({"reasoning_effort": "minimal"}, GLM, "effort=low;assistant:<think>"),
    ({"reasoning_effort": "low"}, GLM, "effort=low;assistant:<think>"),
    ({"reasoning_effort": "medium"}, GLM, "effort=high;assistant:<think>"),   # not Max: medium is not a GLM name
    ({"reasoning_effort": "xhigh"}, GLM, "effort=xhigh;assistant:<think>"),  # GLM's template renders this as Max
    ({"reasoning_effort": "none"}, GLM, "assistant:"),
    ({}, GLM, "effort=high;assistant:<think>"),                                # server default medium, heard as high
    ({"reasoning_effort": "none"}, QWEN, "assistant:"),
    ({"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}}, QWEN, "assistant:"),
    ({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": True}}, QWEN,
     "effort=medium;assistant:<think>"),                                      # the server's default effort
    ({}, QWEN, "effort=medium;assistant:<think>"),
    ({"reasoning_effort": None}, QWEN, "effort=medium;assistant:<think>"),
    ({"chat_template_kwargs": {"enable_thinking": False}}, QWEN, "assistant:"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_the_effort_reaches_the_template_as_on_the_mac(tmp_path, fields, names, want, stream):
    engine = ChainEngine()
    status, _ = ask(app_for(tmp_path, engine, names, effort="medium"), stream, max_tokens=2, **fields)
    assert status == 200 and rendered(engine) == want


def test_no_server_default_leaves_the_template_its_own(tmp_path):
    engine = ChainEngine()
    assert ask(app_for(tmp_path, engine), max_tokens=2)[0] == 200
    assert rendered(engine) == "assistant:<think>"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fields", [{"reasoning_effort": "extreme"}, {"reasoning_effort": 3},
                                    {"chat_template_kwargs": {"reasoning_effort": "max"}},
                                    {"thinking_budget": "lots"}, {"thinking_budget": 2.5}])
def test_a_bad_effort_or_budget_is_refused_before_the_stream(tmp_path, stream, fields):
    engine = ChainEngine()
    status, body = ask(app_for(tmp_path, engine), stream, **fields)
    words = "thinking_budget must be an integer" if "thinking_budget" in fields else "reasoning_effort must be none"
    assert status == 400 and body["error"]["message"].startswith(words) and not engine.prompts


def meant(prompt_last, budget, count, close=(10, END, 10, 10)):
    """The lane engine's rule: the budget-th reply token is the close, and the reply goes on after it."""

    out, last = [], prompt_last
    while len(out) < count:
        if budget and len(out) + 1 == budget:
            out += close
            last = close[-1]
            continue
        last = after(last)
        out.append(last)
    return out[:count]


@pytest.mark.parametrize("budget", [1, 2, 5, 21, 23, 24, 30])
@pytest.mark.parametrize("stream", [False, True])
def test_the_budget_closes_the_think_block_where_the_lane_engine_does(tmp_path, budget, stream):
    replies = []
    for width in (1, 3, 4, 7):
        status, body = ask(app_for(tmp_path, ChainEngine(width)), stream, thinking_budget=budget)
        assert status == 200
        replies.append(body["tensorfold"]["token_ids"])
    want = meant(THINK, budget, 24)
    assert all(reply == want for reply in replies)
    text = Tokens().decode(want)
    if "</think>" in text:                            # the reasoning ends in the close's newline
        reasoning, content = (body["reasoning"], body["content"]) if stream else (
            body["choices"][0]["message"].get("reasoning_content"), body["choices"][0]["message"]["content"])
        assert reasoning == text.split("</think>")[0] and (content or "") == text.split("</think>")[1].lstrip("\n")


@pytest.mark.parametrize("request_budget, server_budget, cut", [
    (None, 4, 4), (0, 4, 4), (6, 4, 6), (-1, 4, None), (None, 0, None), (3, 0, 3)])
def test_the_request_budget_else_the_server_default(tmp_path, request_budget, server_budget, cut):
    engine = ChainEngine()
    fields = {} if request_budget is None else {"thinking_budget": request_budget}
    status, body = ask(app_for(tmp_path, engine, budget=server_budget), **fields)
    assert status == 200 and body["tensorfold"]["token_ids"] == meant(THINK, cut or 0, 24)
    assert len(engine.prompts) == (1 if cut is None else 2)


def test_no_budget_without_thinking_or_after_the_model_closes(tmp_path):
    engine = ChainEngine()
    status, body = ask(app_for(tmp_path, engine), thinking_budget=3, chat_template_kwargs={"enable_thinking": False})
    assert status == 200 and len(engine.prompts) == 1 and END not in body["tensorfold"]["token_ids"]
    engine = ChainEngine(end_at=1)
    status, body = ask(app_for(tmp_path, engine), thinking_budget=8)
    ids = body["tensorfold"]["token_ids"]
    assert status == 200 and len(engine.prompts) == 1 and ids.index(END) == 1   # the model closed first: no cut


def test_the_budget_then_a_required_call(tmp_path):
    """The budget closes the think block, then the answer's first word becomes the call's opener."""

    class Engine(ChainEngine):
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            self.prompts.append(list(prompt))
            written = Tokens().decode(prompt)
            if written.endswith("<function="):
                reply = Tokens().encode("get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n").ids
                reply += [CLOSE, EOS]
            elif prompt[-1] == THINK:
                reply = Tokens().encode("the user wants the weather").ids + [END, 10, 10]
            else:
                reply = Tokens().encode("Sure, here it is.").ids + [EOS]
            for at in range(0, len(reply), 3):
                if on_tokens(reply[at:at + 3]):
                    break
            return {}

    engine = Engine()
    status, body = ask(tool_app(tmp_path, engine), tools=TOOLS, tool_choice="required", thinking_budget=5,
                       chat_template_kwargs={"enable_thinking": True})
    choice = body["choices"][0]
    assert status == 200 and choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning_content"] == "the \n"
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in choice["message"]["tool_calls"]] \
        == [("get_weather", {"city": "Oslo"})]
    assert Tokens().decode(engine.prompts[1]).endswith("<think>the \n</think>\n\n")
    assert Tokens().decode(engine.prompts[2]).endswith("<think>the \n</think>\n\n<tool_call>\n<function=")


def test_the_budget_under_a_grammar_closes_at_think_end_and_the_grammar_starts_at_once(tmp_path):
    xgr = pytest.importorskip("xgrammar")
    torch = pytest.importorskip("torch")
    from tensorfold.engine import grammar

    V, STOP, THINK_END = 128, 0, 127
    info = xgr.TokenizerInfo([""] + [chr(t) for t in range(1, V)], xgr.VocabType.RAW, vocab_size=V,
                             stop_token_ids=[STOP])

    class Text:
        def encode(self, text, **kwargs):
            return SimpleNamespace(ids=[THINK_END if c == "\x7f" else ord(c) for c in text.replace("</think>", "\x7f")])

        def decode(self, ids, **kwargs):
            return "".join("</think>" if t == THINK_END else chr(t) for t in ids if t != STOP)

        def token_to_id(self, text):
            return THINK_END if text == "</think>" else None

    class Engine:
        eos = (STOP,)

        def __init__(self):
            self.calls = []

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
            self.calls.append((list(prompt), constraint.think_end, constraint.active))
            thinking = not constraint.active
            for t in ([ord(c) for c in "abcdefghijklmnop"] if thinking else [ord(c) for c in '{"k":42}'] + [STOP]):
                if not thinking:
                    logits = torch.zeros(1, V)
                    logits[0, t] = 1.0
                    t = int(constraint.mask(logits).argmax())
                constraint.advance([t])
                if on_tokens([t]) or t == STOP:
                    break
            return {}

    engine = Engine()
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": TEMPLATE}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok, app.model_dir = engine, "fake-cuda", Text(), tmp_path
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking, app.reasoning_effort, app.thinking_budget = True, None, 0
    app.sampling, app.max_tokens = {"temperature": 0.0}, 64
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    app.grammars = grammar.Grammars(info)
    schema = {"type": "object", "properties": {"k": {"type": "integer"}}, "required": ["k"]}
    status, body = ask(app, response_format={"type": "json_schema", "json_schema": {"name": "v", "schema": schema}},
                       thinking_budget=4)
    message = body["choices"][0]["message"]
    assert status == 200 and message["reasoning_content"] == "abc\n" and json.loads(message["content"]) == {"k": 42}
    (first, end0, active0), (second, end1, active1) = engine.calls
    assert (end0, active0, end1, active1) == (THINK_END, False, None, True)
    assert second[len(first):] == [ord("a"), ord("b"), ord("c"), 10, THINK_END]     # no blank line under a grammar


def test_the_budget_matches_the_lane_engine(tmp_path):
    """The same chain through the Mac's lane engine (drafted and serial) and through the CUDA server's cut."""

    pytest.importorskip("mlx.core")
    from test_family_streams import END as LANE_END, NL, NLNL, StreamsModel, after as lane_after

    from tensorfold.engine.lane_engine import LaneEngine, LaneStream

    prompt, count = [8, 2], 12

    def lane(budget, drafts):
        engine = LaneEngine(StreamsModel())
        stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=count, think_budget=budget,
                            think_close=(NL, LANE_END, NLNL), think_end=LANE_END, think_open=budget > 0,
                            drafts=drafts)
        engine.add_stream(stream)
        while engine.active_count:
            engine.step()
        return stream.emitted

    class Words:
        def encode(self, text, **kwargs):
            return SimpleNamespace(ids={"\n": [NL], "\n\n": [NLNL]}[text])

        def decode(self, ids, **kwargs):
            return "".join({NL: "\n", NLNL: "\n\n", LANE_END: "</think>"}.get(t, chr(0x4e00 + t)) for t in ids)

        def token_to_id(self, text):
            return LANE_END if text == "</think>" else None

    class Engine:
        eos = (1000,)

        def __init__(self, width):
            self.width = width

        def generate(self, ids, max_tokens, sampling, on_tokens, draft=True):
            last, reply = ids[-1], []
            for _ in range(max_tokens):
                last = lane_after(last)
                reply.append(last)
            for at in range(0, len(reply), self.width):
                if on_tokens(reply[at:at + self.width]):
                    break
            return {}

    for budget in (1, 2, 3, 7, 11, 12):
        want = lane(budget, True)
        assert want == lane(budget, False)
        for width in (1, 2, 5):
            app = server.App.__new__(server.App)
            app.engine, app.tok, app.lock = Engine(width), Words(), threading.Lock()
            prepared = server.PreparedRequest(list(prompt), count, [], True, None, think_budget=budget)
            result = app.run({"return_token_ids": True}, True, lambda delta: True, prepared=prepared)
            assert result["stats"]["token_ids"] == want, (budget, width)
