"""A GLM tool call the model's end token left open (written before ``</tool_call>``) is closed and sent as a call when it
then parses whole; otherwise its markup stays the reply's text. A call whose ``<arg_key>`` the model wrote as
whitespace gets it back in both servers' parsers. A call the token limit cut, and every other reply, is unchanged."""

from __future__ import annotations

import json
import threading

import pytest

from tensorfold.cuda.reply_text import close_glm_call, parse_tool_calls
from tensorfold.server.tools import parse_tool_calls_from_content, repair_glm_keys

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "run", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}},
]
PARSERS = pytest.mark.parametrize("parser", [parse_tool_calls, parse_tool_calls_from_content],
                                  ids=["cuda-server", "http-server"])


def args(text, tools=TOOLS, parser=parse_tool_calls, **kw):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in parser(text, tools, **kw)[1] or []]


# -- the missing <arg_key> --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("block, repaired", [
    ("read_file path</arg_key><arg_value>/x</arg_value>",
     "read_file <arg_key>path</arg_key><arg_value>/x</arg_value>"),
    ("read_file\npath</arg_key><arg_value>/x</arg_value>",
     "read_file\n<arg_key>path</arg_key><arg_value>/x</arg_value>"),
    ("read_file<arg_key>path</arg_key><arg_value>/x</arg_value>\nlimit</arg_key><arg_value>3</arg_value>",
     "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>\n<arg_key>limit</arg_key><arg_value>3</arg_value>"),
    ("read_file path</arg_key><arg_value>/x</arg_value> limit</arg_key><arg_value>3</arg_value>",
     "read_file <arg_key>path</arg_key><arg_value>/x</arg_value> <arg_key>limit</arg_key><arg_value>3</arg_value>"),
])
def test_a_missing_arg_key_is_put_back(block, repaired):
    assert repair_glm_keys(block) == repaired
    assert repair_glm_keys(repaired) == repaired


@pytest.mark.parametrize("block", [
    "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>",      # nothing missing
    "read_file",
    "read_filepath</arg_key><arg_value>/x</arg_value>",               # no space: no name left before the key
    " path</arg_key><arg_value>/x</arg_value>",                       # no name at all
    '{"name": "run", "arguments": {"command": "ls"}}',
])
def test_blocks_without_a_missing_key_are_unchanged(block):
    assert repair_glm_keys(block) == block


@PARSERS
def test_both_parsers_read_a_repaired_call(parser):
    # before: the first came back as text, the second lost its second argument (or was refused as one call)
    text = "<tool_call>read_file filePath</arg_key><arg_value>/home/a/main.go</arg_value></tool_call>"
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}}}}}]
    assert args(text, tools, parser) == [("read_file", {"filePath": "/home/a/main.go"})]
    assert args(text, tools, parser, max_calls=1) == [("read_file", {"filePath": "/home/a/main.go"})]
    later = ("<tool_call>read_file<arg_key>path</arg_key><arg_value>a</arg_value>\nlimit</arg_key><arg_value>3"
             "</arg_value></tool_call>")
    assert args(later, parser=parser) == [("read_file", {"path": "a", "limit": 3})]
    assert args(later, parser=parser, max_calls=1) == [("read_file", {"path": "a", "limit": 3})]


# -- closing a call the end token left open ------------------------------------------------------------------------

@pytest.mark.parametrize("text, suffix", [
    ("<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>", "</tool_call>"),
    ("<tool_call>run<arg_key>command</arg_key><arg_value>ls -la", "</arg_value></tool_call>"),
    ("Let me look:<tool_call>now", "</tool_call>"),
    ("<tool_call>now</tool_call>\n<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>\n",
     "</tool_call>"),
    ("<tool_call>read_file path</arg_key><arg_value>x", "</arg_value></tool_call>"),       # and repaired
])
def test_a_call_that_parses_once_closed_is_closed(text, suffix):
    assert close_glm_call(text, TOOLS) == suffix
    assert args(text + suffix)[-1][0] in ("read_file", "run", "now")


@pytest.mark.parametrize("text", [
    "no call at all",
    "<tool_call>now</tool_call>",                                       # closed already
    "<tool_call>launch<arg_key>x</arg_key><arg_value>1",                # a tool the client did not offer
    "<tool_call>run<arg_key>comm",                                      # a key with no value
    "<tool_call>run<arg_key>command</arg_key>",                         # ... or no value yet
    "<tool_call>run command=ls",                                        # not GLM's markup
    "<tool_call>",                                                      # nothing written
    "<tool_call>now</think>The answer.",                                # the think block went on past it
    "<tool_call>run<arg_key>command</arg_key><arg_value>ls</arg_value> and then",       # text after an argument
])
def test_a_call_that_would_not_parse_is_left_open(text):
    assert close_glm_call(text, TOOLS) == ""


# -- through the CUDA server's GlmApp ------------------------------------------------------------------------------

tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402

from tensorfold.cuda import server  # noqa: E402
from tensorfold.families.glm5_next.cuda.app import GlmApp, ThinkingOffTemplate  # noqa: E402
from tests.test_cuda_admission import http_server, post  # noqa: E402
from tests.test_cuda_stop_strings import GlmEngine, events, write_template  # noqa: E402

END = "<|observation|>"
MARKS = ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<think>",
         "</think>")


def glm_tokenizer() -> Tokenizer:
    """One token a byte, plus GLM's call and think marks and an end token as single tokens (as GLM-5.3's are)."""

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={ch: i for i, ch in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([END, *MARKS])
    return tok


TOK = glm_tokenizer()
END_ID = TOK.token_to_id(END)


class Engine(GlmEngine):
    eos = (END_ID,)


def make_app(tmp_path, reply_ids, thinking):
    write_template(tmp_path)
    app = GlmApp.__new__(GlmApp)
    app.engine = Engine(reply_ids)
    app.served = "fake-glm"
    app.tok = TOK
    app.template = ThinkingOffTemplate(server.ChatTemplate(tmp_path))
    app.default_thinking = thinking
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 4096
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def _reply(tmp_path, written, streamed, *, ended=True, thinking=False, **fields):
    """(content, reasoning, calls, finish) of a reply that writes ``written`` (then the end token when ``ended``)."""

    reply_ids = TOK.encode(written, add_special_tokens=False).ids + ([END_ID] if ended else [])
    app = make_app(tmp_path, reply_ids, thinking)
    body = {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS, "temperature": 0, "stream": streamed,
            **fields}
    with http_server(app) as port:
        status, text = post(port, body, True)
    assert status == 200, text
    if not streamed:
        choice = json.loads(text)["choices"][0]
        message = choice["message"]
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in message.get("tool_calls") or []]
        return message.get("content") or "", message.get("reasoning_content") or "", calls, choice["finish_reason"]
    content, reasoning, calls, finish = "", "", {}, None
    for chunk in events(text):
        if not chunk.get("choices"):
            continue
        delta = chunk["choices"][0].get("delta", {})
        content += delta.get("content") or ""
        reasoning += delta.get("reasoning_content") or ""
        for t in delta.get("tool_calls", []):
            entry = calls.setdefault(t["index"], ["", ""])
            entry[0] += t["function"].get("name") or ""
            entry[1] += t["function"].get("arguments") or ""
        finish = chunk["choices"][0].get("finish_reason") or finish
    return content, reasoning, [(n, json.loads(a)) for n, a in (calls[i] for i in sorted(calls))], finish


OPEN = [   # (the reply, ended by the end token; the calls; the content)
    ("Reading it.<tool_call>read_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value>",
     [("read_file", {"path": "notes.txt"})], "Reading it."),
    ("<tool_call>run<arg_key>command</arg_key><arg_value>ls -la", [("run", {"command": "ls -la"})], ""),
    ("The time:<tool_call>now", [("now", {})], "The time:"),
    ("<tool_call>now</tool_call><tool_call>run<arg_key>command</arg_key><arg_value>date</arg_value>",
     [("now", {}), ("run", {"command": "date"})], ""),
    ("<tool_call>read_file path</arg_key><arg_value>notes.txt</arg_value>", [("read_file", {"path": "notes.txt"})], ""),
]


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("written, calls, content", OPEN, ids=range(len(OPEN)))
def test_a_call_the_end_token_left_open_is_a_call(tmp_path, written, calls, content, streamed):
    got_content, _, got_calls, finish = _reply(tmp_path, written, streamed)
    assert (got_content.strip(), got_calls, finish) == (content, calls, "tool_calls")


@pytest.mark.parametrize("streamed", [False, True])
def test_a_call_at_the_end_of_an_unclosed_think_block_is_a_call(tmp_path, streamed):
    got = _reply(tmp_path, "I need the file.<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>",
                 streamed, thinking=True)
    assert got == ("", "I need the file.", [("read_file", {"path": "x"})], "tool_calls")


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("written", [
    "<tool_call>launch_rocket<arg_key>when</arg_key><arg_value>now</arg_value>",   # a tool not offered
    "<tool_call>run<arg_key>comm",                                                 # a key with no value
    "<tool_call>",
])
def test_a_call_left_open_that_would_not_parse_stays_text(tmp_path, written, streamed):
    content, _, calls, finish = _reply(tmp_path, "Trying." + written, streamed)
    assert (content, calls, finish) == ("Trying." + written, [], "stop")


@pytest.mark.parametrize("streamed", [False, True])
def test_a_call_the_token_limit_cut_is_not_closed(tmp_path, streamed):
    written = "Removing it. <tool_call>run<arg_key>command</arg_key><arg_value>rm -rf build"
    content, _, calls, finish = _reply(tmp_path, written, streamed, ended=False,
                                       max_tokens=len(TOK.encode(written, add_special_tokens=False).ids))
    assert (content, calls, finish) == (written, [], "length")


@pytest.mark.parametrize("streamed", [False, True])
def test_a_call_a_stop_string_cut_is_not_closed(tmp_path, streamed):
    written = "Removing it. <tool_call>run<arg_key>command</arg_key><arg_value>rm -rf build STOP and more"
    content, _, calls, finish = _reply(tmp_path, written, streamed, stop=" STOP")
    assert (content, calls, finish) == (written[:written.find(" STOP")], [], "stop")


@pytest.mark.parametrize("streamed", [False, True])
def test_a_closed_call_with_a_missing_key_is_a_call(tmp_path, streamed):
    got = _reply(tmp_path, "<tool_call>read_file path</arg_key><arg_value>notes.txt</arg_value></tool_call>", streamed)
    assert got == ("", "", [("read_file", {"path": "notes.txt"})], "tool_calls")
