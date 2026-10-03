"""A CUDA family's App brings its own prompt renderer, final call parser and streamed-content hider; the default App
keeps the Jinja template, the ``<tool_call>`` parser and hider, and ``prefill_cold build`` renders with the family's."""

import importlib.util
import json
import re
import sys
import threading
import types
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, models, pre_tokenizers

from tensorfold import families
from tensorfold.cuda import server
from tensorfold.cuda.reply_text import hide_tool_calls, parse_tool_calls
from tensorfold.server.tool_policy import ToolCallPolicy
from tests.test_cuda_admission import http_server, post

EOS = 0
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}]
REPLY = 'Checking. [[get_weather|{"city":"Oslo"}]]'
JINJA = ("{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
         "{% if tools %}tools: {{ tools | map(attribute='function.name') | join(',') }}\n{% endif %}assistant:")


class Tokens:
    """Characters as their code points."""

    def encode(self, text, **kwargs):
        return types.SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids, **kwargs):
        return "".join("" if i == EOS else chr(i) for i in ids)

    def token_to_id(self, text):
        return None


class Engine:
    """Writes ``REPLY`` three tokens a round."""

    eos = (EOS,)

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        reply = [ord(c) for c in REPLY][:max_tokens] + [EOS]
        for at in range(0, len(reply), 3):
            if on_tokens(reply[at:at + 3]):
                break
        return {}


class Template:
    """A renderer with no Jinja source."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.rendered = 0

    def render(self, messages, *, tools, enable_thinking, extra=None, allow_images=False):
        self.rendered += 1
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def bracket_calls(text, tools, *, max_calls=None):
    """``[[name|json]]`` calls."""

    calls = [{"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}
             for i, (name, args) in enumerate(re.findall(r"\[\[(\w+)\|(.*?)\]\]", text))]
    return re.sub(r"\[\[.*?\]\]", "", text).strip(), (calls[:max_calls] if max_calls else calls) or None


class BracketApp(server.App):
    template_class = Template
    parse_calls = staticmethod(bracket_calls)

    def _visible_answer(self, answer, policy, finished):
        return answer.split("[[")[0] if finished else answer.split("[")[0]


def jinja_dir(path, model_type=None):
    words = ["[UNK]"] + [f"w{i}" for i in range(40)]
    tok = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": JINJA}))
    if model_type is not None:
        (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return path


def served(app_class, tmp_path):
    app = app_class.__new__(app_class)
    app.engine, app.served, app.tok = Engine(), "fake-cuda", Tokens()
    app.template = server.ChatTemplate(jinja_dir(tmp_path))
    app.default_thinking = False
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 256
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def ask(app, stream):
    body = {"messages": [{"role": "user", "content": "Weather?"}], "tools": TOOLS, "stream": stream}
    with http_server(app) as port:
        return post(port, body, True)


def streamed(text):
    chunks = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    return "".join(d.get("content") or "" for d in deltas), [c for d in deltas for c in d.get("tool_calls") or []]


def test_init_builds_the_template_from_template_class(tmp_path):
    app = BracketApp(Engine(), jinja_dir(tmp_path), "fake-cuda")
    assert type(app.template) is Template and app.template.model_dir == tmp_path


def test_the_default_app_keeps_the_jinja_template_parser_and_hider(tmp_path):
    assert server.App.template_class is server.ChatTemplate and server.App.parse_calls is parse_tool_calls
    app = server.App(Engine(), jinja_dir(tmp_path), "fake-cuda")
    assert type(app.template) is server.ChatTemplate
    text = 'Sure. <tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call> after <tool_'
    for parallel in (True, False):
        policy = ToolCallPolicy({"parallel_tool_calls": parallel})
        for finished in (False, True):
            old = (policy.content(text, finished=finished) if policy.single
                   else hide_tool_calls(text, finished=finished))
            assert app._visible_answer(text, policy, finished) == old


def test_the_final_parse_uses_parse_calls(tmp_path):
    status, body = ask(served(BracketApp, tmp_path), stream=False)
    choice = json.loads(body)["choices"][0]
    assert status == 200 and choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Checking."
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in choice["message"]["tool_calls"]] \
        == [("get_weather", {"city": "Oslo"})]


def test_streamed_content_goes_through_visible_answer(tmp_path):
    status, body = ask(served(BracketApp, tmp_path), stream=True)
    content, calls = streamed(body)
    assert status == 200 and "[" not in content and content.strip() == "Checking."
    assert [c["function"]["name"] for c in calls] == ["get_weather"]


def test_the_default_app_streams_unknown_markup_as_content(tmp_path):
    status, body = ask(served(server.App, tmp_path), stream=True)
    content, calls = streamed(body)
    assert status == 200 and content == REPLY and calls == []


# -- prefill_cold build ----------------------------------------------------------------------


@pytest.fixture
def prefill_cold(monkeypatch):
    spec = importlib.util.spec_from_file_location("prefill_cold", Path(__file__).parents[1] / "tools/prefill_cold.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    text = "".join(f"w{i % 40} line {i}\n" for i in range(6000))
    monkeypatch.setattr(tool, "corpus", lambda: text)
    monkeypatch.setattr(tool, "LENGTHS", (64,))
    monkeypatch.setattr(tool, "REPS", 2)
    return tool


def built(tool, model_dir):
    out = model_dir / "prompts.json"
    tool.build(str(model_dir), str(out))
    return out.read_bytes()


def test_prefill_cold_renders_with_the_family_apps_template(tmp_path, monkeypatch, prefill_cold):
    made = []

    class Recorded(Template):
        def __init__(self, model_dir):
            super().__init__(model_dir)
            made.append(self)

    package = types.ModuleType("fake_family")
    package.CUDA_APP = type("FakeApp", (BracketApp,), {"template_class": Recorded})
    monkeypatch.setitem(sys.modules, "fake_family", package)
    monkeypatch.setattr(families, "families",
                        lambda: {"fake_type": families.Family("fake_type", "Fake", "fake_family", False)})
    model_dir = jinja_dir(tmp_path, "fake_type")
    items = json.loads(built(prefill_cold, model_dir))["items"]
    assert len(made) == 1 and made[0].model_dir == model_dir and made[0].rendered > len(items) == 3


def test_prefill_cold_falls_back_to_the_jinja_template_for_an_unknown_model_type(tmp_path, monkeypatch,
                                                                                  prefill_cold):
    made, init = [], server.ChatTemplate.__init__
    monkeypatch.setattr(server.ChatTemplate, "__init__", lambda self, model_dir: made.append(model_dir) or init(
        self, model_dir))
    model_dir = jinja_dir(tmp_path, "no_such_family")
    items = json.loads(built(prefill_cold, model_dir))["items"]
    assert made == [model_dir] and [it["length"] for it in items] == [1024, 64, 64]


@pytest.mark.parametrize("model_type", ["glm5_next", "qwen3_5"])
def test_prefill_cold_prompts_are_byte_identical_to_the_plain_jinja_build(tmp_path, monkeypatch, prefill_cold,
                                                                          model_type):
    family_dir, plain_dir = tmp_path / "family", tmp_path / "plain"
    family_dir.mkdir()
    plain_dir.mkdir()
    got = built(prefill_cold, jinja_dir(family_dir, model_type))

    def no_family(model_dir):
        raise ValueError("no family")

    monkeypatch.setattr(families, "detect", no_family)          # the build before families could bring a template
    assert got == built(prefill_cold, jinja_dir(plain_dir, model_type))
