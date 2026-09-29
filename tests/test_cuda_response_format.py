"""Structured output: request parsing, grammars compiled or refused, the grammar's windows and masks (engine.grammar),
and the CUDA server's requests (no GPU)."""

import json
import random
import sys
import threading

import pytest

from tensorfold.cuda import server
from tensorfold.engine import grammar
from tensorfold.server.errors import RequestError
from tests.test_cuda_admission import http_server, post
from tests.test_cuda_request_policy import TextTokenizer

SCHEMA = {"type": "object", "properties": {"k": {"type": "integer"}, "tag": {"type": "string", "enum": ["a", "bb"]}},
          "required": ["k"], "additionalProperties": False}


# -- parsing (no xgrammar) -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body, want", [
    ({}, None),
    ({"response_format": None}, None),
    ({"response_format": {"type": "text"}}, None),
    ({"response_format": {"type": "json_object"}}, grammar.Spec("json")),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "v", "schema": SCHEMA, "strict": True}}},
     grammar.Spec("json_schema", json.dumps(SCHEMA))),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": json.dumps(SCHEMA)}}},
     grammar.Spec("json_schema", json.dumps(SCHEMA))),
    ({"guided_json": SCHEMA}, grammar.Spec("json_schema", json.dumps(SCHEMA), "guided_json")),
    ({"structured_outputs": {"json": SCHEMA}}, grammar.Spec("json_schema", json.dumps(SCHEMA), "structured_outputs")),
    ({"structured_outputs": {"json_object": True}}, grammar.Spec("json", field="structured_outputs")),
    ({"structured_outputs": {"json": None, "regex": None}}, None),
    ({"guided_regex": "[a-z]+"}, grammar.Spec("regex", "[a-z]+", "guided_regex")),
    ({"guided_choice": ["a", "b c"]}, grammar.Spec("choice", '["a", "b c"]', "guided_choice")),
    ({"guided_grammar": 'root ::= "a"'}, grammar.Spec("grammar", 'root ::= "a"', "guided_grammar")),
    ({"structured_outputs": {"regex": "[a-z]+"}}, grammar.Spec("regex", "[a-z]+", "structured_outputs")),
    ({"structured_outputs": {"choice": ["x"]}}, grammar.Spec("choice", '["x"]', "structured_outputs")),
])
def test_request_spec_reads_openai_and_vllm_fields(body, want):
    assert grammar.request_spec(body) == want


@pytest.mark.parametrize("body, words", [
    ({"response_format": "json"}, "must be an object"),
    ({"response_format": {"type": "yaml"}}, "text, json_object or json_schema"),
    ({"response_format": {"type": "json_schema"}}, "needs json_schema.schema"),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "v"}}}, "needs json_schema.schema"),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": "{nope"}}}, "not valid JSON"),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": [1, 2]}}}, "JSON schema object"),
    ({"guided_regex": ""}, "guided_regex must be a non-empty string"),
    ({"guided_choice": "a"}, "guided_choice must be a non-empty list"),
    ({"guided_grammar": 3}, "guided_grammar must be a non-empty string"),
    ({"structured_outputs": {"xml": "<a/>"}}, "structured_outputs xml is not supported"),
    ({"structured_outputs": "json"}, "must be an object"),
])
def test_request_spec_refuses_malformed_or_unsupported_requests(body, words):
    with pytest.raises(RequestError, match=words):
        grammar.request_spec(body)


# -- compiling and the grammar's windows (xgrammar, CPU) --------------------------------------------------------------
V = 128
STOP = 0                    # the toy vocabulary's stop token
THINK_END = 127             # and its </think>


@pytest.fixture(scope="module")
def grammars():
    """A toy vocabulary: token t is chr(t) (as TextTokenizer encodes), token 0 the stop token."""

    xgr = pytest.importorskip("xgrammar")
    pytest.importorskip("torch")
    vocab = [""] + [chr(t) for t in range(1, V)]
    info = xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[STOP])
    return grammar.Grammars(info)


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def _allowed(logits, row: int) -> set[int]:
    import torch

    return set(torch.nonzero(logits[row] > float("-inf")).flatten().tolist())


def _expected(grammars, prefix: list[int], spec=None) -> set[int]:
    """The tokens a fresh matcher allows after ``prefix``."""

    compiled = grammars.compile(spec or grammar.Spec("json_schema", json.dumps(SCHEMA)))
    m = grammars.xgr.GrammarMatcher(compiled)
    for t in prefix:
        assert m.accept_token(t)
    bits = grammars.xgr.allocate_token_bitmask(1, V)
    m.fill_next_token_bitmask(bits, 0)
    return {t for t in range(V) if (int(bits[0, t // 32]) >> (t % 32)) & 1}


def _chain(n: int) -> list[int]:
    return list(range(-1, n - 1))


def _schema(grammars, **kwargs):
    return grammars.constraint(grammars.compile(grammar.Spec("json_schema", json.dumps(SCHEMA))), **kwargs)


def test_schemas_compile_or_are_refused_with_the_reason(grammars):
    assert grammars.compile(grammar.Spec("json_schema", json.dumps(SCHEMA))) is not None
    obj = grammars.constraint(grammars.compile(grammar.Spec("json")))            # json_object: any object only
    text = ' {"any": [1, "x"]}'
    assert obj.window(_ids(text), _chain(len(text))).tokens == _ids(text)
    assert obj.window(_ids(" [1]"), _chain(4)).tokens == _ids(" ")
    for bad, words in (({"type": "nonsense"}, 'Unsupported type "nonsense"'),
                       ({"$ref": "#/definitions/missing"}, "definitions/missing"),
                       ({"type": "string", "pattern": "("}, "parenthesis")):
        with pytest.raises(RequestError, match="the grammar cannot be enforced") as err:
            grammars.compile(grammar.Spec("json_schema", json.dumps(bad), "guided_json"))
        message = str(err.value)
        assert message.startswith("guided_json: ") and words in message and ".cc:" not in message, message


def test_json_allows_pretty_printing_but_not_endless_blanks(grammars):
    c = _schema(grammars)
    c.advance(_ids('{"k":1') + [ord(" ")] * grammar.BLANKS)
    with pytest.raises(grammar.GrammarError, match="rejected chosen token"):
        c.advance([ord(" ")])
    c = _schema(grammars)
    c.advance(_ids('{\n  "k": 1,\n  "tag": "a"\n}'))                 # newlines and indents: pretty-printed JSON
    assert not c.finished
    c.advance([STOP])
    assert c.finished


def test_a_window_keeps_the_rows_a_path_can_hold_and_masks_each_by_its_path(grammars):
    import torch

    c = _schema(grammars)
    c.advance(_ids('{"k":'))
    #          0    1    2    3    4    5     6
    tokens = _ids(":1x2}y") + [STOP]
    parents = [-1, 0, 0, 1, 1, 2, 4]    # x: rejected; y under it; the stop token after the complete value
    window = c.window(tokens, parents)
    assert window.tokens == _ids(":12}") and window.parents == [-1, 0, 1, 1]
    assert window.rows == [0, 1, 2, 3]
    paths = [_ids('{"k":'), _ids('{"k":1'), _ids('{"k":12'), _ids('{"k":1}')]
    logits = torch.randn(4, V)
    masked = c.mask(logits.clone(), window)
    for r, path in enumerate(paths):
        want = _expected(grammars, path)
        assert _allowed(masked, r) == want, r
        assert torch.equal(masked[r][sorted(want)], logits[r][sorted(want)])       # allowed logits keep their bits
    assert _allowed(masked, 3) == {STOP}                                            # the grammar ends the reply
    # the matcher is back at the chosen tokens: the same window again, then a partial keep
    again = c.window(tokens, parents)
    assert again.tokens == window.tokens and (again.bits == window.bits).all()
    c.advance(_ids("1"))
    later = c.window(_ids("1}") + [STOP], [-1, 0, 1])
    assert later.tokens == _ids("1}") and (later.bits == window.bits[[1, 3]]).all()


def test_a_chain_stops_at_the_first_draft_the_grammar_rejects(grammars):
    c = _schema(grammars)                         # row 0 is the pending token, which the grammar has followed
    assert c.window(_ids('x{"k":1'), _chain(7)).tokens == _ids('x{"k":1')
    assert c.window(_ids('x{"x":1'), _chain(7)).tokens == _ids('x{"')
    assert c.window(_ids('x{"k":1}') + [STOP, 5], _chain(10)).tokens == _ids('x{"k":1}')


def test_the_grammar_follows_chosen_tokens_and_ends_with_the_stop_token(grammars):
    c = _schema(grammars)
    c.advance(_ids('{"k":1}'))
    assert not c.finished and c.window([ord("}"), STOP], [-1, 0]).tokens == [ord("}")]
    c.advance([STOP])
    assert c.finished
    c.advance([5])                                                                 # nothing follows the stop token
    assert c.finished and c.window([STOP, 5], [-1, 0]).rows == []                  # and nothing is masked
    fresh = grammars.constraint(grammars.compile(grammar.Spec("json")))
    with pytest.raises(grammar.GrammarError, match="rejected chosen token"):
        fresh.advance(_ids("x"))


def test_with_thinking_the_grammar_starts_after_think_end(grammars):
    import torch

    c = _schema(grammars, think_end=THINK_END)
    logits = torch.randn(4, V)
    window = c.window(_ids("abc"), [-1, 0, 1])
    assert window.tokens == _ids("abc") and window.rows == [] and torch.equal(c.mask(logits.clone(), window), logits)
    # a draft </think>: its row and the rows under it follow the grammar, the rows above it do not
    window = c.window([ord("a"), THINK_END, ord("x"), ord("{")], [-1, 0, 1, 1])
    assert window.tokens == [ord("a"), THINK_END, ord("{")] and window.rows == [1, 2]
    masked = c.mask(logits[:3].clone(), window)
    assert torch.equal(masked[0], logits[0]) and _allowed(masked, 1) == _expected(grammars, [])
    assert _allowed(masked, 2) == _expected(grammars, _ids("{"))
    c.advance(_ids("hmm") + [THINK_END])
    assert c.active and not c.finished and _allowed(c.mask(logits[:1].clone()), 0) == _expected(grammars, [])


def _logits(path: tuple[int, ...]):
    """A made-up model: each position's logits depend on its path only (as the CUDA forward's rows do)."""

    import numpy as np

    rng = np.random.default_rng(abs(hash(path)) % (1 << 32))
    return rng.standard_normal(V).astype(np.float32) * 3


def _choose(values, position: int, sampling) -> int:
    import numpy as np

    from tensorfold.engine.exact_sampling import choose_rows

    if sampling is None:
        return int(np.argmax(values))
    return choose_rows(values[None, :], np.arange(V, dtype=np.int64)[None, :], [position], sampling)[0]


def _masked(c, rows, window):
    import numpy as np
    import torch

    return c.mask(torch.tensor(np.stack(rows)), window).numpy()


def _serial(c, prompt: tuple[int, ...], count: int, sampling) -> list[int]:
    out: list[int] = []
    while len(out) < count and (not out or out[-1] != STOP):
        path = prompt + tuple(out)
        choice = _choose(_masked(c, [_logits(path)], None)[0], len(path), sampling)
        c.advance([choice])
        out.append(choice)
    return out


def _drafted(c, prompt: tuple[int, ...], count: int, sampling, rng: random.Random,
             oracle: list[int]) -> tuple[list[int], int, int]:
    """Draft trees verified as the 27B's draft_decode does: a chain of the reply's next tokens (a good drafter),
    then random nodes anywhere (tokens the grammar rejects, stop tokens)."""

    first = _choose(_masked(c, [_logits(prompt)], None)[0], len(prompt), sampling)
    c.advance([first])
    out, dropped, accepted = [first], 0, 0
    while len(out) < count and out[-1] != STOP:
        tokens, parents = [out[-1]], [-1]
        for t in oracle[len(out):len(out) + rng.randint(0, 6)]:
            parents.append(len(tokens) - 1)
            tokens.append(t)
        for _ in range(rng.randint(0, 8)):
            parents.append(rng.randrange(len(tokens)))
            tokens.append(rng.choice(_ids('{}":,0123456789k tagb') + [STOP, ord("x")]))
        window = c.window(tokens, parents)
        dropped += len(tokens) - len(window.tokens)
        tokens, parents = window.tokens, window.parents
        paths = []
        for r in range(len(tokens)):
            p, rows = r, []
            while p > 0:
                rows.append(tokens[p])
                p = parents[p]
            paths.append(prompt + tuple(out) + tuple(reversed(rows)))
        masked = _masked(c, [_logits(p) for p in paths], window)
        sampled = [_choose(masked[r], len(paths[r]), sampling) for r in range(len(tokens))]
        children = {}
        for r in range(1, len(tokens)):
            children.setdefault((parents[r], tokens[r]), r)
        path, terminal = [0], sampled[0]
        while len(out) + len(path) < count and terminal != STOP and (path[-1], terminal) in children:
            path.append(children[(path[-1], terminal)])
            terminal = sampled[path[-1]]
        new = [tokens[r] for r in path[1:]] + [terminal]
        c.advance(new)
        out += new
        accepted += len(path) - 1
    return out, dropped, accepted


@pytest.mark.parametrize("seed", [None, 3, 11], ids=["greedy", "seed3", "seed11"])
def test_drafted_constrained_replies_equal_serial_ones(grammars, seed):
    """Windows, masks and the accepted paths of random trees give the serial reply, token for token."""

    from tensorfold.engine.exact_sampling import Sampling

    sampling = None if seed is None else Sampling(seed, 1.0, 20, 0.95)
    rng = random.Random(seed or 0)
    dropped = accepted = 0
    for n in range(8):
        prompt = tuple(_ids(f"prompt {n}"))
        serial = _serial(_schema(grammars), prompt, 40, sampling)
        drafted, lost, kept = _drafted(_schema(grammars), prompt, 40, sampling, rng, serial)
        dropped, accepted = dropped + lost, accepted + kept
        assert drafted == serial, n
        m = grammars.xgr.GrammarMatcher(grammars.compile(grammar.Spec("json_schema", json.dumps(SCHEMA))))
        assert all(m.accept_token(t) for t in serial)                            # every token the grammar's
    assert dropped > 0 and accepted > 0                  # windows dropped drafts, and rounds kept several tokens


# -- the server --------------------------------------------------------------------------------------------------------
class GrammarEngine:
    """Replies with ``content`` through the constraint's mask (any disallowed token would fail the reply)."""

    eos = (STOP,)

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
        import torch

        self.calls.append({"draft": draft, "constraint": constraint})
        for t in _ids(self.content) + [STOP]:
            if constraint is not None:
                logits = torch.zeros(1, V)
                logits[0, t] = 1.0
                t = int(constraint.mask(logits).argmax())
                constraint.advance([t])
            if on_tokens([t]) or t == STOP:
                break
        return {}


class PlainEngine:
    """An engine without a ``constraint`` argument, as every engine was before structured output."""

    eos = (STOP,)

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.calls.append({"draft": draft})
        on_tokens(_ids(self.content))
        return {}


class FailingEngine(GrammarEngine):
    """The grammar fails on the first request only; the second is served."""

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
        if not self.calls:
            self.calls.append({"failed": True})
            raise grammar.GrammarError("the reply's grammar rejected chosen token 120")
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft, constraint)


def _app(tmp_path, engine, grammars=None):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(
        {"chat_template": "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:"}))
    app = server.App.__new__(server.App)
    app.engine = engine
    app.served = "fake-cuda"
    app.model_dir = tmp_path
    app.tok = TextTokenizer()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking = False
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 64
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    if grammars is not None:
        app.grammars = grammars
    return app


def _body(stream, **extra):
    return {"messages": [{"role": "user", "content": "Hi"}], "stream": stream, **extra}


def _content(stream: bool, text: str) -> str:
    if not stream:
        return json.loads(text)["choices"][0]["message"]["content"]
    parts = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            parts += [c["delta"].get("content") or "" for c in chunk.get("choices", [])]
    return "".join(parts)


@pytest.mark.parametrize("stream", [False, True])
def test_a_schema_reaches_the_engine_as_a_fresh_constraint(tmp_path, grammars, stream):
    engine = GrammarEngine('{"k":42}')
    app = _app(tmp_path, engine, grammars)
    rf = {"type": "json_schema", "json_schema": {"name": "v", "schema": SCHEMA}}
    with http_server(app) as port:
        for _ in range(2):
            status, text = post(port, _body(stream, response_format=rf), True)
            assert status == 200 and json.loads(_content(stream, text)) == {"k": 42}
        status, text = post(port, _body(stream, response_format={"type": "json_object"}, draft=False), True)
        assert status == 200
    first, second, third = (call["constraint"] for call in engine.calls)
    assert isinstance(first, grammar.Constraint) and first is not second and first.finished and second.finished
    assert third.finished and engine.calls[2]["draft"] is False and first.active


@pytest.mark.parametrize("stream", [False, True])
def test_absent_or_text_response_format_calls_the_engine_as_before(tmp_path, stream):
    engine = PlainEngine("Hello")
    app = _app(tmp_path, engine)                   # no grammar compiler: none is built
    with http_server(app) as port:
        for extra in ({}, {"response_format": {"type": "text"}}, {"response_format": None}):
            status, text = post(port, _body(stream, **extra), True)
            assert status == 200 and _content(stream, text) == "Hello"
    assert engine.calls == [{"draft": True}] * 3 and getattr(app, "grammars", None) is None


@pytest.mark.parametrize("stream", [False, True])
def test_bad_schemas_and_engines_without_grammars_are_refused_before_generating(tmp_path, grammars, stream):
    engine = GrammarEngine("{}")
    app = _app(tmp_path, engine, grammars)
    plain = PlainEngine("{}")
    plain_app = _app(tmp_path, plain, grammars)
    bad = {"type": "json_schema", "json_schema": {"name": "v", "schema": {"type": "nonsense"}}}
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    with http_server(app) as port:
        for body, words in ((_body(stream, response_format=bad), 'Unsupported type "nonsense"'),
                            (_body(stream, response_format={"type": "xml"}), "text, json_object or json_schema"),
                            (_body(stream, guided_regex="("), "the grammar cannot be enforced"),
                            (_body(stream, response_format={"type": "json_object"}, tools=[tool],
                                   tool_choice="required"), 'cannot be combined with tool_choice "required"')):
            status, text = post(port, body, True)
            assert status == 400 and words in json.loads(text)["error"]["message"], text
    with http_server(plain_app) as port:
        status, text = post(port, _body(stream, response_format={"type": "json_object"}), True)
        assert status == 400 and "does not enforce structured output" in json.loads(text)["error"]["message"]
    assert engine.calls == [] and plain.calls == []


def test_without_xgrammar_a_structured_request_is_refused_with_the_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "xgrammar", None)          # import xgrammar raises ImportError
    monkeypatch.setattr(grammar, "_MODELS", {})
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": V}))
    engine = GrammarEngine("{}")
    app = _app(tmp_path, engine)
    with http_server(app) as port:
        status, text = post(port, _body(False, response_format={"type": "json_object"}), True)
        assert status == 400 and "pip install 'tensorfold[grammar]'" in json.loads(text)["error"]["message"]
        status, text = post(port, _body(False), True)            # plain requests are served as before
        assert status == 200
    assert [call["constraint"] for call in engine.calls] == [None]


@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_grammar_ends_its_request_only(tmp_path, grammars, stream):
    engine = FailingEngine('{"k":7}')
    app = _app(tmp_path, engine, grammars)
    with http_server(app) as port:
        status, text = post(port, _body(stream, response_format={"type": "json_object"}), True)
        if stream:
            events = [json.loads(line[6:]) for line in text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            assert status == 200 and events[-1]["error"]["type"] == "server_error"
            assert "rejected chosen token" in events[-1]["error"]["message"]
        else:
            error = json.loads(text)["error"]                 # the server's 500 body, as for any failed reply
            assert status == 500 and "rejected chosen token" in error["message"]
        status, text = post(port, _body(stream, response_format={"type": "json_object"}), True)
        assert status == 200 and json.loads(_content(stream, text)) == {"k": 7}


def test_the_first_structured_request_builds_the_compiler_from_the_checkpoint(tmp_path, monkeypatch):
    built = []

    def fake(model_dir, vocab, stop_ids):
        built.append((str(model_dir), vocab, stop_ids))
        return "compiler"

    monkeypatch.setattr(grammar, "for_model", fake)
    app = _app(tmp_path, GrammarEngine(""))
    with pytest.raises(RequestError, match="config.json vocab_size"):
        app._grammars()
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 128}, "vocab_size": 7}))
    assert app._grammars() == "compiler" and app._grammars() == "compiler"
    assert built == [(str(tmp_path), 128, (STOP,))]
