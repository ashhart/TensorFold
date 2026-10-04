"""DeepSeek-V4.1's CUDA App: the encoder's prompts, DSML calls parsed at the end and never streamed as content, integer
efforts, thinking budgets refused, and DeepSeek's no-top-k default."""

import json
import re
import threading

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, models, pre_tokenizers

from tensorfold.families import deepseek_v41
from tensorfold.families.deepseek_v41.cuda.app import DeepSeekV41App
from tensorfold.families.deepseek_v41.prompts import Template
from tests.test_cuda_admission import http_server, post

BOS, EOS, SYSTEM, USER, ASSISTANT, THINK, END, DSML = 0, 1, 128799, 128803, 128804, 128821, 128822, 128825
SPECIAL = {"<｜begin▁of▁sentence｜>": BOS, "<｜end▁of▁sentence｜>": EOS, "<｜System｜>": SYSTEM, "<｜User｜>": USER,
           "<｜Assistant｜>": ASSISTANT, "<think>": THINK, "</think>": END, "｜DSML｜": DSML}
SMOKE = [{"role": "user", "content": "What is 17*19? Return only the integer."}]
SMOKE_CHAT = "<｜begin▁of▁sentence｜><｜User｜>What is 17*19? Return only the integer.<｜Assistant｜></think>"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "location": {"type": "string"}, "days": {"type": "array"}}}}},
         {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
             "q": {"type": "string"}}}}}]
BLOCK = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
         '<｜DSML｜ parameter name="location" string="true">Oslo</｜DSML｜ parameter>\n'
         '<｜DSML｜ parameter name="days" string="false">[1, 2]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n'
         '<｜DSML｜ invoke name="search">\n<｜DSML｜ parameter name="q" string="true">rain</｜DSML｜ parameter>\n'
         '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
REASONING, ANSWER = "the user wants the weather", "Checking.\n\n" + BLOCK
CALLS = [("get_weather", {"location": "Oslo", "days": [1, 2]}), ("search", {"q": "rain"})]


class Tokens:
    """Characters as their code points, the V4.1 markup as its single token ids."""

    def encode(self, text, **kwargs):
        ids = []
        for part in re.split("(" + "|".join(re.escape(s) for s in SPECIAL) + ")", text):
            ids += [SPECIAL[part]] if part in SPECIAL else [ord(c) for c in part]
        return type("Encoding", (), {"ids": ids})()

    def decode(self, ids, **kwargs):
        names = {v: k for k, v in SPECIAL.items()}
        return "".join(names.get(i, chr(i)) for i in ids)

    def token_to_id(self, text):
        return SPECIAL.get(text)


def ids(text):
    return Tokens().encode(text).ids


class Engine:
    """Reasons up to </think> when the prompt opened a think block, then answers with a DSML block, three tokens a
    round; the family engine's ``generate`` signature, and like it decodes on whatever ``on_tokens`` returns."""

    eos = (EOS,)

    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, stop_eos=True):
        self.calls.append((list(prompt), sampling))
        reply = (ids(REASONING) + [END] if prompt[-1] == THINK else []) + ids(ANSWER) + [EOS]
        reply = reply[:max_tokens]
        for at in range(0, len(reply), 3):
            on_tokens(reply[at:at + 3])
        return {"rounds": 1 + len(reply) // 3}


def app_for(engine, thinking=False):
    app = DeepSeekV41App.__new__(DeepSeekV41App)
    app.engine, app.served, app.tok = engine, "fake-cuda", Tokens()
    app.template = Template()
    app.default_thinking = thinking
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 1024
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def ask(app, stream=False, **body):
    body = {"messages": [{"role": "user", "content": "Weather in Oslo?"}], "tools": TOOLS, "stream": stream, **body}
    with http_server(app) as port:
        status, reply = post(port, body, True)
    assert status == 200, reply
    return reply


def deltas(reply):
    chunks = [json.loads(line[6:]) for line in reply.splitlines() if line.startswith("data: {")]
    return [(c["choices"][0]["delta"], c["choices"][0].get("finish_reason")) for c in chunks if c.get("choices")]


def calls_of(calls):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]


def test_the_family_exports_the_app():
    assert deepseek_v41.CUDA_APP is DeepSeekV41App and DeepSeekV41App.template_class is Template


@pytest.mark.parametrize("thinking", [False, True])
def test_a_dsml_reply_becomes_tool_calls(thinking):
    choice = json.loads(ask(app_for(Engine(), thinking)))["choices"][0]
    message = choice["message"]
    assert choice["finish_reason"] == "tool_calls" and calls_of(message["tool_calls"]) == CALLS
    assert message["content"] == "Checking."
    assert message.get("reasoning_content") == (REASONING if thinking else None)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("parallel", [True, False])
def test_streamed_content_never_holds_dsml_and_calls_come_at_the_end(thinking, parallel):
    sent = []
    app = app_for(Engine(), thinking)
    body = {"messages": [{"role": "user", "content": "Weather?"}], "tools": TOOLS, "stream": True,
            "parallel_tool_calls": parallel}
    result = app.run(body, True, lambda delta: sent.append(delta) or True)
    assert sent and all("tool_calls" not in d and "DSML" not in d.get("content", "") for d in sent)
    assert result["calls_streamed"] == 0 and calls_of(result["calls"]) == (CALLS if parallel else CALLS[:1])
    assert "".join(d.get("content", "") for d in sent).strip() == "Checking."

    got = deltas(ask(app_for(Engine(), thinking), stream=True, parallel_tool_calls=parallel))
    assert all("DSML" not in (d.get("content") or "") for d, _ in got)
    first = next(i for i, (d, _) in enumerate(got) if d.get("tool_calls"))
    assert all(d.get("tool_calls") for d, _ in got[first:-1]) and got[-1][1] == "tool_calls"
    assert calls_of(c for d, _ in got for c in d.get("tool_calls") or []) == (CALLS if parallel else CALLS[:1])


def test_one_call_when_parallel_tool_calls_is_false():
    message = json.loads(ask(app_for(Engine()), parallel_tool_calls=False))["choices"][0]["message"]
    assert calls_of(message["tool_calls"]) == CALLS[:1] and message["content"] == "Checking."


def test_an_integer_effort_renders_as_the_reasoning_effort_line():
    engine = Engine()
    ask(app_for(engine), reasoning_effort=42)
    prompt = Tokens().decode(engine.calls[0][0])
    assert "<｜System｜>Reasoning Effort: 42 (range 1-100" in prompt and prompt.endswith("<｜Assistant｜><think>")


def test_the_smoke_body_renders_the_chat_prompt():
    engine = Engine()
    app = app_for(engine, thinking=True)
    ask(app, messages=SMOKE, tools=None, chat_template_kwargs={"thinking": False})
    assert engine.calls[0][0] == ids(SMOKE_CHAT)
    assert app.tokenize({"messages": SMOKE, "chat_template_kwargs": {"thinking": False}})["tokens"] == ids(SMOKE_CHAT)


def test_tokenize_without_the_generation_prompt_drops_the_header():
    reply = app_for(Engine()).tokenize({"messages": SMOKE, "add_generation_prompt": False})
    assert reply["tokens"] == ids(SMOKE_CHAT)[:-2] and reply["tokens"][-1] != ASSISTANT


@pytest.mark.parametrize("server_budget, request_budget", [(0, 4), (8, None)])
def test_a_thinking_budget_is_refused_before_decoding(server_budget, request_budget):
    engine = Engine()
    app = app_for(engine, thinking=True)
    app.thinking_budget = server_budget
    body = {"messages": SMOKE, **({"thinking_budget": request_budget} if request_budget else {})}
    with http_server(app) as port:
        status, reply = post(port, body, True)
    assert status == 400 and "thinking_budget" in reply and engine.calls == []
    ask(app, tools=None, chat_template_kwargs={"thinking": False})        # thinking off: no budget applies
    assert len(engine.calls) == 1


def test_an_anthropic_image_block_is_refused_not_dropped():
    engine = Engine()
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "describe"}, image]}]}
    with http_server(app_for(engine)) as port:
        status, reply = post(port, body, True)
    assert status == 400 and "vision" in reply and engine.calls == []


def test_default_sampling_has_no_top_k(tmp_path):
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "w": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(tmp_path / "tokenizer.json"))
    app = DeepSeekV41App(Engine(), tmp_path, "fake-cuda")
    assert app.sampling == {"temperature": 1.0, "top_k": 0, "top_p": 0.95} and type(app.template) is Template
    assert DeepSeekV41App(Engine(), tmp_path, "fake-cuda", sampling={"top_k": 20}).sampling["top_k"] == 20
    sampling = app.sampling_for({}, [1, 2, 3])
    assert (sampling.top_k, sampling.top_p) == (0, 0.95)
