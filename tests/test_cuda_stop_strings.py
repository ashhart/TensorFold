"""CUDA server stop strings and ignore_eos: drafted and serial replies cut at the same token, bad fields refused early."""

import json
import threading
import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from tensorfold.cuda import server
from tensorfold.families.glm5_next.cuda.app import GlmApp, ThinkingOffTemplate
from tests.test_cuda_admission import http_server, post
from tests.test_request_policy import CALL, TOOLS

WIDTHS = (1, 3, 5, 2, 7, 4)            # a drafted reply's rounds; "draft": false delivers one token a round
END = "<|end|>"


def byte_tokenizer() -> Tokenizer:
    """One token a byte (no merges), so a character can span several tokens, and one special end token."""

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={ch: i for i, ch in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([END])
    return tok


TOK = byte_tokenizer()
END_ID = TOK.token_to_id(END)


def ids(text: str) -> list[int]:
    return TOK.encode(text, add_special_tokens=False).ids


class Engine:
    """A fixed reply in rounds; it ends after the round whose ``on_tokens`` returns True, or at an end token."""

    eos = (END_ID,)

    def __init__(self, reply: list[int], honors_stop: bool = True):
        self.reply, self.honors_stop = reply, honors_stop
        self.calls: list[dict] = []
        self.stop_eos = True

    def _stop_eos(self) -> bool:
        return self.stop_eos

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        rounds: list[list[int]] = []
        stop_eos, i = self._stop_eos(), 0
        while i < min(max_tokens, len(self.reply)):
            width = WIDTHS[len(rounds) % len(WIDTHS)] if draft else 1
            chunk = self.reply[i:min(i + width, max_tokens)]
            if stop_eos and any(t in self.eos for t in chunk):     # nothing past an end token
                chunk = chunk[:next(k for k, t in enumerate(chunk) if t in self.eos) + 1]
            rounds.append(chunk)
            i += len(chunk)
            stop = on_tokens(chunk)
            if (stop and self.honors_stop) or (stop_eos and chunk[-1] in self.eos):
                break
        self.calls.append({"draft": draft, "rounds": rounds, "stop_eos": stop_eos})
        return {"generated": i}


class GlmEngine(Engine):
    """Reads its request's stop-at-EOS as the GLM engine does (set by ``GlmApp.run``); ignores the stop return."""

    def __init__(self, reply):
        super().__init__(reply, honors_stop=False)
        self.request = threading.local()

    def _stop_eos(self) -> bool:
        return bool(getattr(self.request, "stop_eos", True))


def write_template(tmp_path):
    template = ("{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:"
                "{% if enable_thinking %}<think>{% endif %}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))


def make_app(tmp_path, engine, cls=server.App):
    write_template(tmp_path)
    app = cls.__new__(cls)
    app.engine = engine
    app.served = "fake-cuda"
    app.tok = TOK
    app.template = server.ChatTemplate(tmp_path)
    if cls is GlmApp:
        app.template = ThinkingOffTemplate(app.template)
    app.default_thinking = False
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 4096
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def body(chat, stream, **fields):
    prompt = {"messages": [{"role": "user", "content": "x"}]} if chat else {"prompt": "x"}
    return {**prompt, "stream": stream, "stream_options": {"include_usage": True}, "temperature": 0, **fields}


def events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


def reply(port, chat, stream, **fields):
    """(content, reasoning, finish, completion_tokens, token_sha, tool calls) of one request, streamed or not."""

    status, text = post(port, body(chat, stream, **fields), chat)
    assert status == 200, text
    if not stream:
        payload = json.loads(text)
        choice = payload["choices"][0]
        message = choice.get("message", {})
        content = message.get("content") or "" if chat else choice["text"]
        calls = [c["function"]["arguments"] for c in message.get("tool_calls") or []]
        return (content, message.get("reasoning_content") or "", choice["finish_reason"],
                payload["usage"]["completion_tokens"], payload["tensorfold"]["token_sha"], calls)
    chunks = events(text)
    assert all("error" not in c for c in chunks) and text.count("data: [DONE]") == 1
    shown, reasoning, calls = "", "", []
    for c in chunks[:-1]:
        piece = c["choices"][0].get("delta", {}).get("content") if chat else c["choices"][0].get("text")
        shown += piece or ""
        reasoning += c["choices"][0].get("delta", {}).get("reasoning_content") or "" if chat else ""
        calls += [t["function"]["arguments"] for t in c["choices"][0].get("delta", {}).get("tool_calls", [])]
    end = chunks[-1]
    return (shown, reasoning, end["choices"][0]["finish_reason"], end["usage"]["completion_tokens"],
            end["tensorfold"]["token_sha"], calls)


def delivered_through(engine_call) -> int:
    return sum(len(r) for r in engine_call["rounds"])


REPLY = "Hello, world! Matrix products 数学 run on the GPU; STOP here, then more text follows and ends."
# stops at several offsets: in the first round, across round boundaries, multi-byte, repeated, at the very start
STOPS = ["H", "Hello", "o, w", "world", "!", " Matrix", "数学", "学 run", "GPU;", "STOP", "e, then", "ends."]


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stop", STOPS)
def test_stop_cuts_drafted_and_serial_replies_at_the_same_token(tmp_path, chat, stream, stop):
    reply_ids = ids(REPLY)
    engine = Engine(reply_ids)
    app = make_app(tmp_path, engine)
    at = REPLY.find(stop)
    through = len(ids(REPLY[:at + len(stop)]))           # tokens through the one that completes the match
    expected = (REPLY[:at], "", "stop", through, server.token_sha(reply_ids[:through]), [])
    with http_server(app) as port:
        got = [reply(port, chat, stream, stop=stop, draft=draft) for draft in (True, False)]
    assert got == [expected, expected]
    drafted, serial = engine.calls
    # each engine ended in the round that holds the match: no further round ran
    assert delivered_through(drafted) - len(drafted["rounds"][-1]) < through <= delivered_through(drafted)
    assert delivered_through(serial) == through and all(len(r) == 1 for r in serial["rounds"])


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_stop_list_takes_the_earliest_match_and_an_unmatched_prefix_is_flushed(tmp_path, chat, stream):
    reply_ids = ids(REPLY)
    app = make_app(tmp_path, Engine(reply_ids))
    with http_server(app) as port:
        first = [reply(port, chat, stream, stop=["absent", "STOP", "world"], draft=d) for d in (True, False)]
        # "ends.!" begins at the reply's last characters and never completes: they are held back, then sent
        whole = [reply(port, chat, stream, stop=["ends.!"], draft=d) for d in (True, False)]
    cut = len(ids("Hello, world"))
    assert first == [("Hello, ", "", "stop", cut, server.token_sha(reply_ids[:cut]), [])] * 2
    assert whole == [(REPLY, "", "length", len(reply_ids), server.token_sha(reply_ids), [])] * 2


@pytest.mark.parametrize("stream", [False, True])
def test_a_stop_longer_in_bytes_than_the_mac_tail_is_still_found(tmp_path, stream):
    # five three-byte characters, one byte a token: 15 tokens, more than len(stop) + 8 = 13
    text = "ab" + "数学数学数" + "cd"
    reply_ids = ids(text)
    app = make_app(tmp_path, Engine(reply_ids))
    with http_server(app) as port:
        got = [reply(port, False, stream, stop="数学数学数", draft=d) for d in (True, False)]
    assert got == [("ab", "", "stop", 17, server.token_sha(reply_ids[:17]), [])] * 2


@pytest.mark.parametrize("stream", [False, True])
def test_stop_strings_match_reasoning_and_answer_before_the_split(tmp_path, stream):
    text = "think about it</think>The answer is 42. STOP and more"
    reply_ids = ids(text)
    app = make_app(tmp_path, Engine(reply_ids))
    thinking = {"chat_template_kwargs": {"enable_thinking": True}}
    with http_server(app) as port:
        in_answer = [reply(port, True, stream, stop="STOP", draft=d, **thinking) for d in (True, False)]
        in_reasoning = [reply(port, True, stream, stop="about", draft=d, **thinking) for d in (True, False)]
    through = len(ids(text[:text.find("STOP") + 4]))
    assert in_answer == [("The answer is 42. ", "think about it", "stop", through,
                          server.token_sha(reply_ids[:through]), [])] * 2
    through = len(ids("think about"))
    assert in_reasoning == [("", "think ", "stop", through, server.token_sha(reply_ids[:through]), [])] * 2


@pytest.mark.parametrize("stream", [False, True])
def test_tool_calls_are_parsed_from_the_truncated_text(tmp_path, stream):
    text = "Sure. " + CALL.format(1) + " STOP " + CALL.format(2)
    reply_ids = ids(text)
    app = make_app(tmp_path, Engine(reply_ids))
    with http_server(app) as port:
        got = [reply(port, True, stream, stop="STOP", tools=TOOLS, draft=d) for d in (True, False)]
    through = len(ids(text[:text.find("STOP") + 4]))
    # the content is the cut text less its call (a streamed reply keeps the space the non-streamed one strips)
    assert [(g[0].strip(), g[2], g[3], g[4], [json.loads(a) for a in g[5]]) for g in got] == \
        [("Sure.", "tool_calls", through, server.token_sha(reply_ids[:through]), [{"value": 1}])] * 2


@pytest.mark.parametrize("stream", [False, True])
def test_an_engine_that_decodes_on_after_the_stop_changes_nothing(tmp_path, stream):
    reply_ids = ids(REPLY)
    honest, deaf = Engine(reply_ids), Engine(reply_ids, honors_stop=False)
    with http_server(make_app(tmp_path, honest)) as port:
        want = [reply(port, True, stream, stop="STOP", draft=d) for d in (True, False)]
    with http_server(make_app(tmp_path, deaf)) as port:
        got = [reply(port, True, stream, stop="STOP", draft=d) for d in (True, False)]
    assert got == want and delivered_through(deaf.calls[0]) == len(reply_ids)


@pytest.mark.parametrize("stream", [False, True])
def test_return_token_ids_ends_at_the_token_that_completes_the_stop(tmp_path, stream):
    reply_ids = ids(REPLY)
    app = make_app(tmp_path, Engine(reply_ids, honors_stop=False))
    through = len(ids(REPLY[:REPLY.find("STOP") + 4]))
    got = []
    with http_server(app) as port:
        for draft in (True, False):
            status, text = post(port, body(True, stream, stop="STOP", draft=draft, return_token_ids=True), True)
            assert status == 200, text
            block = events(text)[-1]["tensorfold"] if stream else json.loads(text)["tensorfold"]
            got.append((block["token_ids"], block["token_sha"]))
    assert got == [(reply_ids[:through], server.token_sha(reply_ids[:through]))] * 2


BAD = [("ignore_eos", None), ("ignore_eos", 0), ("ignore_eos", 1), ("ignore_eos", "true"), ("ignore_eos", []),
       ("stop", ""), ("stop", [""]), ("stop", 5), ("stop", [1]), ("stop", {"a": "b"}), ("stop", ["ok", None])]


@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("field,value", BAD)
def test_bad_stop_or_ignore_eos_is_refused_before_any_stream_bytes(tmp_path, glm, chat, stream, field, value):
    engine = GlmEngine(ids(REPLY)) if glm else Engine(ids(REPLY))
    app = make_app(tmp_path, engine, GlmApp if glm else server.App)
    with http_server(app) as port:
        status, text = post(port, body(chat, stream, **{field: value}), chat)
    assert status == 400 and not text.startswith("data:")
    assert field in json.loads(text)["error"]["message"] and engine.calls == []
    assert app.check(body(chat, stream, **{field: value})) is not None


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ignore", [None, False, True])
def test_ignore_eos_on_an_engine_that_reads_it_keeps_end_tokens_as_text(tmp_path, chat, stream, ignore):
    text = "one" + END + "two" + END + "three"
    reply_ids = ids(text)
    engine = GlmEngine(reply_ids)
    app = make_app(tmp_path, engine, GlmApp)
    fields = {} if ignore is None else {"ignore_eos": ignore}
    n = len(ids("one")) + 1                            # through the first end token
    with http_server(app) as port:
        got = [reply(port, chat, stream, draft=d, max_tokens=len(reply_ids), **fields) for d in (True, False)]
        stopped = [reply(port, chat, stream, draft=d, stop="wo", **fields) for d in (True, False)]
        # the reply limit falls on an end token
        at_limit = [reply(port, chat, stream, draft=d, max_tokens=n, **fields) for d in (True, False)]
    if ignore:
        assert got == [(text, "", "length", len(reply_ids), server.token_sha(reply_ids), [])] * 2
        through = len(ids("one" + END + "two"))
        assert stopped == [("one" + END + "t", "", "stop", through, server.token_sha(reply_ids[:through]), [])] * 2
        assert at_limit == [("one" + END, "", "length", n, server.token_sha(reply_ids[:n]), [])] * 2
        assert {c["stop_eos"] for c in engine.calls} == {False}
    else:
        assert got == [("one", "", "stop", n, server.token_sha(reply_ids[:n]), [])] * 2
        assert stopped == got                          # the end token comes before "wo"
        assert at_limit == got
        assert {c["stop_eos"] for c in engine.calls} == {True}


@pytest.mark.parametrize("stream", [False, True])
def test_ignore_eos_tool_calls_are_parsed_from_text_that_keeps_end_tokens(tmp_path, stream):
    text = "Sure." + END + " " + CALL.format(1)
    reply_ids = ids(text)
    app = make_app(tmp_path, GlmEngine(reply_ids), GlmApp)
    with http_server(app) as port:
        got = [reply(port, True, stream, tools=TOOLS, ignore_eos=True, max_tokens=len(reply_ids), draft=d)
               for d in (True, False)]
    assert [(g[0].strip(), g[2], g[3], g[4], [json.loads(a) for a in g[5]]) for g in got] == \
        [("Sure." + END, "tool_calls", len(reply_ids), server.token_sha(reply_ids), [{"value": 1}])] * 2


@pytest.mark.parametrize("stream", [False, True])
def test_ignore_eos_on_an_engine_that_does_not_read_it_leaves_replies_as_before(tmp_path, stream):
    text = "one" + END + "two"
    reply_ids = ids(text)
    app = make_app(tmp_path, Engine(reply_ids))
    with http_server(app) as port:
        got = [reply(port, True, stream, draft=d, ignore_eos=True) for d in (True, False)]
    n = len(ids("one")) + 1
    assert got == [("one", "", "stop", n, server.token_sha(reply_ids[:n]), [])] * 2


def test_prepared_request_carries_the_checked_fields(tmp_path):
    app = make_app(tmp_path, Engine(ids(REPLY)))
    prepared = app.prepare(body(True, False, stop=["a", "b"], ignore_eos=True), True)
    assert prepared.stop == ("a", "b") and prepared.ignore_eos is True
    prepared = app.prepare(body(False, False), False)
    assert prepared.stop == () and prepared.ignore_eos is False


def test_stop_scan_reads_only_the_tail_that_can_hold_a_new_match():
    seen = []

    class Recording:
        def decode(self, got, **kwargs):
            seen.append(list(got))
            return TOK.decode(got, **kwargs)

    stops = server.StopStrings(("数学",), Recording(), (END_ID,))
    tokens = ids("x" * 40 + "数学") + [END_ID]
    assert stops.tail == 6 + 8
    assert not stops.hit(tokens[:-3]) and stops.hit(tokens[:-1]) and stops.hit(tokens)
    assert all(len(s) <= 14 for s in seen) and END_ID not in seen[-1]
    assert stops.visible("ab数", partial=True) == "ab" and stops.visible("ab数学c") == "ab"
