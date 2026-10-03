"""Qwen XML tool calls through the CUDA server keep their schema types: an integer, a number, a boolean, arrays and
nested objects arrive as JSON values and a string stays a string, streamed or not, thinking or not, "auto" or
"required", one call or two."""

import json

import pytest

from tests.test_cuda_admission import http_server, post
from tests.test_cuda_tool_choice import CLOSE, END, EOS, OPEN, THINK, Tokens, app_for, events

PROPS = {"days": {"type": "integer"}, "rate": {"type": "number"}, "refundable": {"type": "boolean"},
         "cities": {"type": "array", "items": {"type": "string"}},
         "budget": {"type": "object", "properties": {"currency": {"type": "string"},
                                                     "limits": {"type": "object"}}},
         "stops": {"type": "array", "items": {"type": "object"}}, "note": {"type": "string"}}
TOOLS = [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": PROPS}}}
         for name in ("plan_trip", "book")]
ARGS = {"days": 3, "rate": 42.5, "refundable": False, "cities": ["Oslo", "Bergen"],
        "budget": {"currency": "NOK", "limits": {"hotel": 1500, "food": [200, 350.5], "extras": None}},
        "stops": [{"city": "Oslo", "nights": 2, "tags": []}, {"city": "Bergen", "nights": 1, "tags": ["fjord"]}],
        "note": "42"}
SECOND = {"days": 1, "cities": [], "budget": {}, "note": "[1, 2]"}


def call(name, args):
    params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                     for k, v in args.items())
    return f"\n<function={name}>\n{params}</function>\n"


class Engine:
    """Writes the calls (after its reasoning when the prompt opened a think block), three characters a round."""

    eos = (EOS,)

    def __init__(self, calls):
        self.calls = calls

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        enc = lambda text: Tokens().encode(text).ids                               # noqa: E731
        reply = [t for name, args in self.calls for t in [OPEN, *enc(call(name, args)), CLOSE]] + [EOS]
        if prompt[-1] == THINK:
            reply = enc("plan the trip") + [END] + enc("\n\n") + reply
        reply = reply[:max_tokens]
        for at in range(0, len(reply), 3):
            if on_tokens(reply[at:at + 3]):
                break
        return {"rounds": 1 + len(reply) // 3, "decode_s": 0.5}


def calls_of(body, stream):
    if not stream:
        message = json.loads(body)["choices"][0]["message"]
        assert message["content"] is None
        return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in message["tool_calls"]]
    chunks = [c for c in events(body) if c.get("choices")]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    names, args = {}, {}
    for chunk in chunks:
        assert "<tool_call>" not in chunk["choices"][0]["delta"].get("content", "")
        for d in chunk["choices"][0]["delta"].get("tool_calls", []):
            names[d["index"]] = names.get(d["index"], "") + d["function"].get("name", "")
            args[d["index"]] = args.get(d["index"], "") + d["function"].get("arguments", "")
    return [(names[i], json.loads(args[i])) for i in sorted(names)]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required"])
def test_typed_arguments_arrive_as_their_schema_types(tmp_path, stream, thinking, choice):
    wanted = [("plan_trip", ARGS), ("book", SECOND)]
    body = {"messages": [{"role": "user", "content": "Plan it."}], "tools": TOOLS, "tool_choice": choice,
            "stream": stream, "chat_template_kwargs": {"enable_thinking": thinking}, "max_tokens": 2048}
    with http_server(app_for(tmp_path, Engine(wanted))) as port:
        status, reply = post(port, body, True)
    assert status == 200, reply
    got = calls_of(reply, stream)
    assert got == wanted
    days, rate, refundable = (got[0][1][k] for k in ("days", "rate", "refundable"))
    assert type(days) is int and type(rate) is float and refundable is False
    assert got[0][1]["note"] == "42" and got[1][1]["note"] == "[1, 2]"            # strings stay strings


@pytest.mark.parametrize("stream", [False, True])
def test_one_call_when_parallel_calls_are_off_keeps_its_types(tmp_path, stream):
    body = {"messages": [{"role": "user", "content": "Plan it."}], "tools": TOOLS, "stream": stream,
            "parallel_tool_calls": False, "max_tokens": 2048}
    with http_server(app_for(tmp_path, Engine([("plan_trip", ARGS), ("book", SECOND)]))) as port:
        status, reply = post(port, body, True)
    assert status == 200, reply
    assert calls_of(reply, stream) == [("plan_trip", ARGS)]
