"""DeepSeek-V4.1 tool calls held to their schemas (TF_DSV41_TOOL_GRAMMAR): xgrammar's ``deepseek_v4_1`` structural tag,
compiled over a byte-level tokenizer that keeps the ｜DSML｜ token, takes the calls V4.1's template writes, refuses an
unknown tool and a mistyped value, forces a named tool, starts after </think>, and crosses to rank 1 by ``pack``."""

from __future__ import annotations

import json

import pytest

xgr = pytest.importorskip("xgrammar")
pytest.importorskip("transformers")

from tensorfold.engine import grammar  # noqa: E402
from tensorfold.server.errors import RequestError  # noqa: E402

D = "｜DSML｜"
SPECIALS = ["<｜end▁of▁sentence｜>", "<think>", "</think>", D]
VOCAB = 300
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "search", "parameters": {
             "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]


def _bytes_to_unicode() -> dict[int, str]:
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, n = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + n)
            n += 1
    return dict(zip(keep, map(chr, chars)))


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A byte-level BPE tokenizer (every byte one token, no merges) with V4.1's added tokens, as tokenizer.json."""

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    path = tmp_path_factory.mktemp("dsv41")
    vocab = {c: i for i, c in enumerate(_bytes_to_unicode().values())}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    tok.save(str(path / "tokenizer.json"))
    (path / "config.json").write_text(json.dumps({"vocab_size": VOCAB}))
    return path


@pytest.fixture(scope="module")
def tok(model_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(model_dir / "tokenizer.json"))


@pytest.fixture(scope="module")
def grammars(model_dir, tok):
    return grammar.for_model(model_dir, VOCAB, [tok.token_to_id(SPECIALS[0])])


def calls(*invokes: tuple[str, dict]) -> str:
    """The calls block as V4.1's template renders it (after the reply's text, or right after </think>)."""

    def invoke(name, args):
        params = "".join(f'<{D} parameter name="{k}" string="{"true" if isinstance(v, str) else "false"}">'
                         f'{v if isinstance(v, str) else json.dumps(v)}</{D} parameter>\n' for k, v in args.items())
        return f'<{D} invoke name="{name}">\n{params}</{D} invoke>'
    return f"\n\n<{D} calls>\n" + "\n".join(invoke(n, a) for n, a in invokes) + f"\n</{D} calls>"


def walk(grammars, tok, spec, text, *, think: str | None = None):
    end = tok.token_to_id("</think>")
    c = grammars.constraint(grammars.compile(spec), think_end=end if think is not None else None, spec=spec)
    ids = (tok.encode(think + "</think>", add_special_tokens=False).ids if think is not None else []) + \
        tok.encode(text, add_special_tokens=False).ids + [tok.token_to_id(SPECIALS[0])]
    c.advance(ids)
    return c


def test_the_spec_follows_tool_choice():
    assert grammar.tool_spec({}, TOOLS) is None                              # auto: free text, unless strict
    assert grammar.tool_spec({"tool_choice": "none"}, []) is None
    req = json.loads(grammar.tool_spec({"tool_choice": "required", "parallel_tool_calls": False}, TOOLS).text)
    assert req["tool_choice"] == "required" and req["parallel_tool_calls"] is False
    named = json.loads(grammar.tool_spec({"tool_choice": {"type": "function", "function": {"name": "search"}}},
                                         TOOLS[1:]).text)
    assert named["tool_choice"] == {"type": "function", "function": {"name": "search"}}
    strict = [dict(TOOLS[0], function=dict(TOOLS[0]["function"], strict=True))]
    assert json.loads(grammar.tool_spec({}, strict).text)["tool_choice"] == "auto"
    every = json.loads(grammar.tool_spec({"tool_choice": "auto"}, TOOLS, auto=True).text)
    assert every["tool_choice"] == "auto" and all(t["function"]["strict"] for t in every["tools"])
    with pytest.raises(RequestError):
        grammar.tool_spec({"tool_choice": "required"}, [{"type": "function", "function": {"name": "x",
                                                                                        "parameters": "no"}}])


def test_mode(monkeypatch):
    for value, mode in ((None, "off"), ("0", "off"), ("1", "required"), ("required", "required"), ("ALL", "all")):
        if value is None:
            monkeypatch.delenv("TF_DSV41_TOOL_GRAMMAR", raising=False)
        else:
            monkeypatch.setenv("TF_DSV41_TOOL_GRAMMAR", value)
        assert grammar.tool_grammar_mode() == mode
    monkeypatch.setenv("TF_DSV41_TOOL_GRAMMAR", "yes")
    with pytest.raises(ValueError):
        grammar.tool_grammar_mode()


def test_a_required_call_as_the_template_writes_it_is_taken(grammars, tok):
    spec = grammar.tool_spec({"tool_choice": "required"}, TOOLS)
    text = calls(("get_weather", {"city": "Paris", "days": 3}), ("search", {"q": "x"}))
    assert walk(grammars, tok, spec, text).finished
    assert walk(grammars, tok, spec, text, think="the user wants weather").finished      # after </think>


def test_the_grammar_refuses_what_the_tools_do_not_allow(grammars, tok):
    spec = grammar.tool_spec({"tool_choice": "required"}, TOOLS)
    for text in (calls(("rm_rf", {"path": "/"})), "Sure, here you go."):
        with pytest.raises(grammar.GrammarError):
            walk(grammars, tok, spec, text)
    # as OpenAI's: only a strict function's arguments are held to its schema (``all`` makes every function strict)
    mistyped, missing = calls(("get_weather", {"city": "Paris", "days": "three"})), calls(("get_weather", {"days": 3}))
    assert walk(grammars, tok, spec, mistyped).finished and walk(grammars, tok, spec, missing).finished
    strict = grammar.tool_spec({"tool_choice": "required"}, TOOLS, auto=True)
    for text in (mistyped, missing):
        with pytest.raises(grammar.GrammarError):
            walk(grammars, tok, strict, text)
    one = grammar.tool_spec({"tool_choice": "required", "parallel_tool_calls": False}, TOOLS)
    with pytest.raises(grammar.GrammarError):
        walk(grammars, tok, one, calls(("search", {"q": "a"}), ("search", {"q": "b"})))
    named = grammar.tool_spec({"tool_choice": {"type": "function", "function": {"name": "search"}}}, TOOLS[1:])
    assert walk(grammars, tok, named, calls(("search", {"q": "a"}))).finished
    with pytest.raises(grammar.GrammarError):
        walk(grammars, tok, named, calls(("get_weather", {"city": "Paris"})))


def test_rank_one_compiles_the_same_tool_grammar(grammars, tok):
    spec = grammar.tool_spec({"tool_choice": "required"}, TOOLS)
    end = tok.token_to_id("</think>")
    packed = grammar.pack(grammars.constraint(grammars.compile(spec), think_end=end, spec=spec))
    assert packed[0] == grammar.KINDS.index("tools") == 5
    again = grammars.follow(packed)
    assert again.spec == grammar.Spec("tools", spec.text) and again.think_end == end
